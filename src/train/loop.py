"""One stage. Flow matching in observable space, on training combinations only.

    source   p0 = Phi(x_ctrl + sum_a w_a)      the additive move is already done
    target   p1 = Phi(x_S)                     real cells of the condition
    couple   minibatch UOT(p0, p1)             in observable space
    field    v(p) = B_S p                      linear AND autonomous
    endpoint exp(B_S) p0                       exact, one matrix_exp, whole batch

WHAT IS TRAINED ON: TRAINING COMBINATIONS ONLY. A single perturbation's prediction is
the additive component, which is closed form and has nothing to learn, so single
conditions enter only through the ridge fit. That is not a limitation to work around -
it is why the single block cannot be damaged, and it is the block where both published
models lose to a closed-form fit (over 5 folds of Norman's combination holdout, ridge
1.416 against scDFM's 1.619 and scPKFM's 1.754).

THE FIELD IS AUTONOMOUS. Flow matching is unchanged - same interpolant, same coupling -
but one operator has to work at every point on the path, which is what makes this an ODE
rather than a field fitted to an interpolant. The endpoint is then closed form, so it can
be a statement about the WHOLE BATCH: scPKFM needed RK4 and could only afford a single
mean point, and its own plan recorded the batch version as the one untried item with a
measured bound of 0.20-0.26.

FOUR TERMS, and what each is for:

  fm            the field along the path. Shapes the operator's direction.
  endpoint      exp(B_S) p0 against p1, in OBSERVABLE space. The flow map, exactly.
  gene_endpoint the decoded prediction's mean against the condition's mean, in GENE
                space. THIS TERM IS THE REPORTED METRIC, so it is the one loss that
                cannot be mismatched with what is scored - scPKFM's endpoint lived in a
                latent space and the mismatch was measured, with the conditions it
                supervised most directly coming out worst.
  head          the likelihood of the real cells under the predicted distribution.
                Trains the gate and the dispersion; it cannot move the mean, because the
                head is mean-preserving by construction.

READING `residual_cos` AS AN EXPECTED L2. If the predicted residual has cosine rho with
the true one and norm ratio k, the remaining error is

    L2 = ||r|| * sqrt(1 - 2 k rho + k^2)

which is minimised at k = rho and gives ||r|| * sqrt(1 - rho^2). Since ||r|| IS the
additive baseline's L2 - its error is exactly the residual it leaves - that turns a
diagnostic into a prediction, and it sets the requirement exactly: passing scDFM on
Table 3 needs sqrt(1 - rho^2) <= 1.6567 / 1.8577, i.e. rho >= 0.4525.

A 12-epoch cpu smoke run reached rho = 0.5484 with k = 0.154, so L2 1.8577 -> 1.7761
where optimal scaling of the SAME residual would have given 1.5536. The direction is
therefore already better than required and the SCALE is what is missing - and the reason
is visible in the loss trace: fm sat at 1.67 and endpoint at 1.51 without moving, because
both ask a single shared linear operator to carry one cell to one other cell, which it
cannot do. They saturate and dominate the gradient while the term that owns the mean
displacement is outweighed.

THE WEIGHTS ARE THEREFORE NOT SETTLED, and they must not be settled on the numbers above:
those are TEST conditions, measured to check that the machinery runs. Selecting weights
against them is exactly what scPKFM did for three weeks. They belong to
scripts/dev_queue.py and the pre-registered rule in scripts/dev_score.py.

GENE-SPACE LOSSES USE EVERY MODELLED GENE, never scdfm_eval_genes. That function reads
the TEST cells to choose its 1,000 - that is scDFM's protocol and reproducing it is right
for scoring - so training on those genes would be a leak.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..eval.scdfm_metrics import median_sigmas, mmd2_unbiased_multi_sigma
from .coupling import sample_pairs


def trainable_conditions(data, stats, train_conditions: list[str],
                         additive_weights: np.ndarray, pert_index: dict) -> list[str]:
    """Training COMBINATIONS with cells and a usable additive source.

    A combination whose perturbations the ridge could not cover has an all-zero w_a, so
    its additive source is the control cell unshifted and the operator would be asked to
    produce the whole displacement for that one condition alone. Excluded rather than
    trained on, and counted so the log says how many.
    """
    naming = data.naming
    out = []
    for condition in train_conditions:
        if not naming.is_double(condition) or condition not in data.rows:
            continue
        if not stats.has(condition):
            continue
        indices = [pert_index[g] for g in naming.genes(condition)]
        if not all(np.abs(additive_weights[i]).sum() > 0 for i in indices):
            continue
        out.append(condition)
    return out


@torch.no_grad()
def condition_means(data, stats, conditions: list[str], device: str) -> dict:
    """m_S per condition, on the device. Constants: cached once, never refitted."""
    return {c: torch.as_tensor(stats.mean[c], dtype=torch.float32, device=device)
            for c in conditions}


def train(model, data, stats, train_conditions: list[str], config: dict,
          device: str, rng: np.random.Generator, log, run_dir: str | None = None) -> dict:
    """The whole of training. Returns the last epoch's loss parts."""
    train_cfg = config["train"]
    weights = model.additive_weights.detach().cpu().numpy()
    conditions = trainable_conditions(data, stats, train_conditions, weights,
                                      data.pert_index)
    if not conditions:
        raise SystemExit(
            "no trainable combination: every training double has a perturbation the "
            "ridge could not cover. Check split.method and the fold - with nothing to "
            "train on the model would score exactly ridge_additive and say nothing.")
    skipped = sum(1 for c in train_conditions
                  if data.naming.is_double(c) and c not in conditions)
    log(f"  training on {len(conditions)} combinations"
        + (f" ({skipped} skipped: a perturbation the ridge does not cover)"
           if skipped else ""))

    means = condition_means(data, stats, conditions, device)
    control = data.cells(data.control_condition)
    batch = int(train_cfg["batch_size"])
    fm_weight = float(train_cfg["fm_weight"])
    endpoint_weight = float(train_cfg["endpoint_weight"])
    gene_weight = float(train_cfg["gene_endpoint_weight"])
    mmd_weight = float(train_cfg["mmd_weight"])
    if mmd_weight > 0:
        log("  [warn] mmd_weight > 0: scPKFM measured its MMD estimate at NOISE LEVEL "
            "at weight 10. It has to earn this on the validation folds.")

    optimiser = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    scheduler = None
    if train_cfg["lr_cosine"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=int(train_cfg["epochs"]),
            eta_min=float(train_cfg["lr_min"]))
        log(f"  lr schedule: cosine {train_cfg['lr']:g} -> {train_cfg['lr_min']:g} "
            f"over {train_cfg['epochs']} epochs")

    coupling_stats: dict = {}
    parts: dict = {}
    started = time.time()
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        order = rng.permutation(len(conditions))
        if train_cfg["max_steps_per_epoch"]:
            order = order[:int(train_cfg["max_steps_per_epoch"])]
        totals = {k: [] for k in ("loss", "fm", "endpoint", "gene", "head", "mmd")}

        for index in order:
            condition = conditions[index]
            perturbations = [data.pert_index[g]
                             for g in data.naming.genes(condition)]
            target_cells = data.cells(condition)
            source = control[rng.choice(control.shape[0],
                                        size=min(batch, control.shape[0]),
                                        replace=False)]
            target = target_cells[rng.choice(target_cells.shape[0],
                                             size=min(batch, target_cells.shape[0]),
                                             replace=False)]
            x_source = torch.as_tensor(source, device=device)
            x_target = torch.as_tensor(target, device=device)

            p0 = model.source_observables(x_source, perturbations)
            p1 = model.observables(x_target)
            # Indices, not rows: the pair is needed in observable space for the field and
            # in gene space for the head, and re-deriving one from the other is how the
            # two drift apart.
            rows, columns = sample_pairs(p0, p1, train_cfg["coupling"],
                                         float(train_cfg["uot_reg"]),
                                         float(train_cfg["uot_reg_marginal"]),
                                         rng, stats=coupling_stats,
                                         stop_thr=float(train_cfg["uot_stop_thr"]))
            p0c, p1c = p0[rows], p1[columns]
            x_target_c = x_target[columns]

            loss = torch.zeros((), device=device)

            if fm_weight > 0:
                # One t PER SAMPLE. A single scalar for the batch gives the time axis one
                # sample per step, so [0, 1] is covered sparsely; scPKFM measured the
                # consequence as a predicted displacement 72 % of the true one.
                t = torch.rand(p0c.shape[0], 1, device=device)
                p_t = (1.0 - t) * p0c + t * p1c
                fm = torch.nn.functional.mse_loss(
                    model.operators.velocity(p_t, perturbations), p1c - p0c)
                loss = loss + fm_weight * fm
                totals["fm"].append(float(fm))

            if endpoint_weight > 0:
                end = torch.nn.functional.mse_loss(
                    model.operators.flow(p0c, perturbations), p1c)
                loss = loss + endpoint_weight * end
                totals["endpoint"].append(float(end))

            # The prediction, and everything gene-space hangs off it.
            predicted_mean = x_source + model.displacement(x_source, perturbations)

            if gene_weight > 0:
                # SQUARED NORM, not a per-gene mean. The reported metric is
                # ||mu_hat - mu||_2, so its loss is the squared norm; dividing by the
                # 5,000 genes made this term 5e-4 against the observable-space terms'
                # 1.6, i.e. three orders down, and the one loss that IS the metric had
                # no influence on the gradient at all. Measured that way in the first
                # smoke run: residual_cos reached +0.31 while L2 moved +0.0018, because
                # nothing was asking the residual to have the right SCALE.
                #
                # Every modelled gene, never scdfm_eval_genes - that function reads the
                # test cells to pick its 1,000.
                gene = (predicted_mean.mean(dim=0) - means[condition]).square().sum()
                loss = loss + gene_weight * gene
                totals["gene"].append(float(gene))

            head_loss, head_parts = model.head.loss(model.head(predicted_mean),
                                                    x_target_c)
            loss = loss + head_loss
            totals["head"].append(float(head_loss))

            if mmd_weight > 0:
                realised = model.head.point_estimate(model.head(predicted_mean))
                sigmas = median_sigmas(x_target_c, scales=(0.5, 1.0, 2.0, 4.0))
                mmd = mmd2_unbiased_multi_sigma(realised, x_target_c, sigmas)
                loss = loss + mmd_weight * mmd
                totals["mmd"].append(float(mmd))

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(train_cfg["grad_clip"]))
            optimiser.step()
            totals["loss"].append(float(loss))

        if scheduler is not None:
            scheduler.step()

        # The fallback is counted, not silent. scPKFM's used to be, and a reg too small
        # for the cost scale made it fire on every batch (600/600 without cost
        # normalisation at reg 0.1) while the run looked healthy.
        plans = max(coupling_stats.get("plans", 0), 1)
        share = coupling_stats.get("fallbacks", 0) / plans
        parts = {k: float(np.mean(v)) for k, v in totals.items() if v}
        if (epoch + 1) % max(1, int(train_cfg["epochs"]) // 40) == 0 or epoch == 0:
            extra = "".join(f"  {k} {parts[k]:.5f}" for k in
                            ("fm", "endpoint", "gene", "head", "mmd") if k in parts)
            log(f"  epoch {epoch + 1:4d}/{train_cfg['epochs']}  "
                f"loss {parts.get('loss', float('nan')):.5f}{extra}  "
                f"| operator |B| {float(model.operators.compose(perturbations).abs().max()):.4f}"
                f"  readout |W| {_readout_scale(model):.4f}"
                f"  fallback {share:.1%}")
        if share > float(train_cfg["coupling_fallback_max"]):
            raise SystemExit(
                f"coupling fell back to random pairing on {share:.1%} of batches, over "
                f"train.coupling_fallback_max={train_cfg['coupling_fallback_max']}. The "
                f"plan is degenerate - usually uot_reg too small for the cost scale - and "
                f"training on random pairs is the one thing this method must not do.")

        if run_dir and train_cfg["save_every"] and (epoch + 1) % int(train_cfg["save_every"]) == 0:
            import os
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch + 1},
                       os.path.join(run_dir, "checkpoint_partial.pt"))

    log(f"  trained in {time.time() - started:.1f}s")
    return parts


def _readout_scale(model) -> float:
    readout = model.readout
    tensor = readout.weight if readout.kind == "dense" else readout.values
    return float(tensor.abs().max())
