"""Turn a trained model into the quantities the baselines are scored on.

Prediction transports control cells: they are shifted in gene space by the additive
component, corrected by the pathway-Koopman residual, and realised through the
mean-preserving head. Nothing is sampled from a prior, so the predicted population
inherits the control population's own heterogeneity - and nothing passes through an
autoencoder, so it pays no reconstruction cost. That cost was 0.97 of scPKFM's 2.138 on
Table 3, 45 % of its error, spent before transport began.

THE MEAN IS EXACTLY THE ADDITIVE BASELINE'S UNTIL THE RESIDUAL LEARNS SOMETHING. Under
the soft gate the head returns the predicted mean unchanged, and the residual is zero at
initialisation, so an untrained run scores exactly what scripts/baseline_l2.py reports
for ridge_additive: measured 1.857668 against 1.857669 on Table 3. That identity is the
premise of the design and tests/test_structure.py holds it.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import condition_genes
from . import metrics


@torch.no_grad()
def condition_residual(model, cells: np.ndarray, perturbations: list[int], device: str,
                       chunk: int = 512) -> tuple[np.ndarray, float]:
    """The condition's mean RAW residual and the scale the calibration gives it.

    One definition used by both scoring paths. predict_cells needs the scale before it can
    predict and measure_transport needs the mean residual to report anyway, and when the two
    computed it separately they would drift - which is exactly how scPKFM ended up with an
    alpha correction that was a no-op in one path and live in the other.

    Returns the UNSCALED mean so a caller can report ||r|| and s separately; the model forces
    residual_scale to 1 whenever the rule is active, so this really is the raw residual.
    """
    total, seen = None, 0
    for start in range(0, cells.shape[0], max(chunk, 1)):
        x = torch.as_tensor(cells[start:start + chunk], device=device)
        part = model.residual(x, perturbations).sum(dim=0)
        total = part if total is None else total + part
        seen += x.shape[0]
    mean = (total / max(seen, 1)) if total is not None else torch.zeros(
        model.n_genes, device=device)
    return mean.cpu().numpy(), model.condition_scale(mean)


@torch.no_grad()
def predict_cells(model, control_cells: np.ndarray, condition: str,
                  pert_index: dict[str, int], device: str, naming,
                  chunk: int = 512) -> np.ndarray:
    """THE choke point: every scoring path in this repository ends up here.

    `naming` is positional and has no default. It used to fall back to a module-level
    'ctrl' in scPKFM, which is right for Norman and wrong for combosciplex - where the
    control is spelled control+control - and the fallback turned 'control+Panobinostat'
    into two perturbations and raised KeyError: 'control' a thousand lines from the
    cause.

    `chunk` bounds peak memory. Phi is one [K, G] matmul so it is cheap, but the readout
    materialises [B, edges] on the sparse path and the head works in [B, G]; scPKFM's
    equivalent was left unchunked and died with a 10.79 GiB single allocation the first
    time a finished run was scored beside a training one.
    """
    model.eval()
    perturbations = [pert_index[g] for g in condition_genes(condition, naming)]
    # The per-condition calibration needs the whole population's mean residual, so it costs
    # a first pass. Only when it is switched on: without it `scale` is the constant the
    # model already carries and the single pass is unchanged.
    scale = (condition_residual(model, control_cells, perturbations, device, chunk)[1]
             if model.residual_coefficient is not None else model.residual_scale)
    pieces = []
    for start in range(0, control_cells.shape[0], max(chunk, 1)):
        x = torch.as_tensor(control_cells[start:start + chunk], device=device)
        pieces.append(model.predict_scaled(x, perturbations, scale).cpu().numpy())
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


@torch.no_grad()
def additive_cells(model, control_cells: np.ndarray, condition: str,
                   pert_index: dict[str, int], device: str, naming) -> np.ndarray:
    """The same transport with the residual switched off: the model's own baseline.

    This replaces scPKFM's `autoencode`, which measured what its bottleneck cost. There
    is no bottleneck here, so the useful reference is the other one: what the model
    scores before the learned term contributes. Every reported number should be read
    against it, because a model that does not beat it has learned nothing that matters -
    and unlike an autoencoder floor this reference is a real, strong predictor
    (Table 3 1.858, Table 1 1.669, at or above scDFM on two of four tables).
    """
    model.eval()
    perturbations = [pert_index[g] for g in condition_genes(condition, naming)]
    x = torch.as_tensor(control_cells, device=device)
    mean = x + model.additive(perturbations)
    return model.head.point_estimate(model.head(mean)).cpu().numpy()


def evaluate_model(model, data, stats, folds, method, config,
                   rng: np.random.Generator) -> dict:
    """Same protocol as scripts/run_baselines.py, so the numbers are comparable."""
    device = config["train"]["device"]
    n_gen = config["eval"]["n_gen_cells"]
    control_cells = data.cells(data.control_condition)

    numerator, denominator = 0.0, 0.0
    edists, de20s, per_double, additive_edists = [], [], [], []

    for fold in folds:
        for double in [c for c in fold["test"] if data.naming.is_double(c)]:
            a, b = data.naming.genes(double)
            single_a, single_b = stats.single_of(a), stats.single_of(b)
            if not all(stats.has(c) for c in (double, single_a, single_b)):
                continue

            pick = rng.choice(control_cells.shape[0],
                              size=min(n_gen, control_cells.shape[0]), replace=False)
            control_sample = control_cells[pick]
            predicted = predict_cells(model, control_sample, double, data.pert_index,
                                      device, data.naming)

            m_hat = predicted.mean(axis=0)
            m_ab, m_a = stats.mean[double], stats.mean[single_a]
            m_b, m_ctrl = stats.mean[single_b], stats.control
            e_noise = sum(float((stats.var[c] / max(stats.n[c], 1)).sum())
                          for c in (double, single_a, single_b, stats.control_condition))

            r = metrics.residual(m_ab, m_a, m_b, m_ctrl)
            r_hat = metrics.residual(m_hat, m_a, m_b, m_ctrl)
            numerator += float((r_hat - r) @ (r_hat - r)) - e_noise
            denominator += float(r @ r) - e_noise
            per_double.append(metrics.residual_r2(m_hat, m_ab, m_a, m_b, m_ctrl, e_noise))
            de20s.append(metrics.de20_pearson(m_hat - m_ctrl, m_ab - m_ctrl))
            edists.append(metrics.edist_rel(
                predicted, data.cells(double), control_sample,
                power=config["eval"]["edist_power"], device=config["eval"]["device"]))
            # The same comparison with the learned term off. Not a floor: a strong
            # predictor the model has to beat.
            additive_edists.append(metrics.edist_rel(
                additive_cells(model, control_sample, double, data.pert_index,
                               device, data.naming),
                data.cells(double), control_sample,
                power=config["eval"]["edist_power"], device=config["eval"]["device"]))

    scored = int(np.sum(~np.isnan(per_double))) if per_double else 0
    return {
        "n_evaluated": len(edists),
        # THE headline. Pooled, so no per-condition denominator can blow up. Zero means
        # tying the additive arithmetic exactly, which is also where an untrained run of
        # this model sits by construction.
        "resid_R2_pooled": 1.0 - numerator / denominator if edists else float("nan"),
        # Per-condition mean over the doubles whose residual clears their own noise
        # floor. Quoted separately because it is a different statement: scPKFM once had
        # pooled -1.12 and per-condition -2.89 on the same run.
        "resid_R2_mean": float(np.nanmean(per_double)) if scored else float("nan"),
        "resid_R2_mean_n": scored,
        "edist_rel": float(np.nanmean(edists)) if edists else float("nan"),
        "r_de20": float(np.nanmean(de20s)) if edists else float("nan"),
        # What the additive component alone scores on the same cells. A model whose
        # edist_rel is not below this has learned nothing that reaches the population.
        "edist_rel_additive": (float(np.nanmean(additive_edists))
                               if additive_edists else float("nan")),
    }
