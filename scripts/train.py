"""Train one configuration end to end.

    python scripts/train.py --tag ftA --set model.composition=anticommutator
    python scripts/train.py --tag smoke --set train.epochs=5 train.device=cpu

Every value lives in src/config.py and any key can be overridden with dot notation. The
resolved config is written next to the results, so a run is reproducible from its own
directory.

THE PREMISE IS CHECKED BEFORE TRAINING, ON THIS RUN'S OWN DATA. An untrained model must
predict exactly the additive baseline: W starts at zero and the head is mean-preserving,
so the residual is identically zero and the prediction is m_ctrl + sum_a w_a. That is
worth a unit test (tests/test_structure.py has it) and also worth checking here, every
run, because the test uses a synthetic observable space and this uses the real one. The
check reads TRAINING conditions only.

It is not a formality. The first version of the head clamped the mean at zero per cell,
which broke the premise on real data - 2.7318 where ridge_additive scores 1.8577 - while
every structural test still passed, because those are about the displacement and that bug
was about the mean of a nonlinearity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import splits
from src.eval import baselines
from src.eval.diagnostics import _dataset, build_model, training_rows
from src.eval.predict import evaluate_model
from src.train.loop import train, trainable_conditions


def build_logger(path: str):
    handle = open(path, "a", encoding="utf-8")

    def log(message: str = "") -> None:
        print(message)
        handle.write(message + "\n")
        handle.flush()

    return log, handle


@torch.no_grad()
def verify_premise(model, data, stats, conditions: list[str], log,
                   tolerance: float = 1e-4) -> None:
    """An untrained model is the additive baseline. Checked on TRAINING conditions.

    Two assertions, both exact by construction:
      - the residual is identically zero, because W is zero
      - the prediction's mean is the control mean plus sum_a w_a, because the head is
        mean-preserving

    The second is the one that catches a head which reparameterises the mean. Run under
    the SOFT gate, since sample is only unbiased and would need thousands of draws to
    show a violation this small.
    """
    gate = model.head.gate_mode
    model.head.gate_mode = "soft"
    try:
        control = torch.as_tensor(data.cells(data.control_condition),
                                  device=model.additive_weights.device)
        control_mean = control.mean(dim=0).cpu().numpy()
        worst_residual, worst_mean = 0.0, 0.0
        for condition in conditions[:8]:
            perturbations = [data.pert_index[g] for g in data.naming.genes(condition)]
            residual = model.residual(control, perturbations)
            worst_residual = max(worst_residual, float(residual.abs().max()))
            predicted = model.predict(control, perturbations).mean(dim=0).cpu().numpy()
            expected = control_mean + model.additive(perturbations).cpu().numpy()
            worst_mean = max(worst_mean, float(np.abs(predicted - expected).max()))
    finally:
        model.head.gate_mode = gate

    log(f"  premise: residual is zero to {worst_residual:.2e}, "
        f"prediction equals the additive baseline to {worst_mean:.2e}")
    if worst_residual > 0.0 or worst_mean > tolerance:
        raise SystemExit(
            f"THE PREMISE IS BROKEN. An untrained model must predict exactly the "
            f"additive baseline: residual {worst_residual:.3e} (must be 0) and mean "
            f"deviation {worst_mean:.3e} (must be under {tolerance:g}).\n"
            f"Training from here would not start at a measured baseline, which is the "
            f"one thing this design buys. Do not train around it - the last time this "
            f"fired the cause was the head clamping its mean per cell, which cost "
            f"0.87 of L2 on Table 3 while every structural test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--tag", default=None, help="run directory name")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing checkpoint.pt. Off by default: "
                             "scPKFM opened run directories with exist_ok and "
                             "overwrote checkpoints without a word, so a reused tag "
                             "destroyed the run it was being compared against.")
    args = parser.parse_args()

    config = config_module.load(args.overrides)
    device = config["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable, falling back to cpu")
        device = config["train"]["device"] = config["eval"]["device"] = "cpu"

    tag = args.tag or f"scpkr_{time.strftime('%m%d-%H%M')}"
    run_dir = os.path.join(config["train"]["out_dir"], tag)
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path) and not args.force:
        raise SystemExit(f"{checkpoint_path} exists. Use a new --tag, or --force to "
                         f"overwrite it deliberately.")
    os.makedirs(run_dir, exist_ok=True)
    log, handle = build_logger(os.path.join(run_dir, "train.log"))

    torch.manual_seed(config["train"]["seed"])
    rng = np.random.default_rng(config["train"]["seed"])

    log(f"run {tag}")
    log(f"config {json.dumps(config['model'])}")
    log("")

    data, stats = _dataset(config)
    method = config["split"]["method"]
    fold = splits.folds(config, method)[config["split"]["fold"]]
    log(f"data: {data.x.shape[0]:,} cells x {data.n_genes:,} genes, "
        f"{data.n_perturbations} perturbations, {len(data.conditions)} conditions")

    log("\n=== the additive component (closed form, not learned) ===")
    model, train_conditions, rows = build_model(config, data, stats, fold, method, device)
    # Leak accounting, in the log so it is in the artifact rather than in someone's head.
    held_out = data.x.shape[0] - len(rows)
    log(f"  cells visible to training: {len(rows):,} / {data.x.shape[0]:,}  "
        f"({held_out:,} held-out cells excluded from the ridge fit, from Phi's "
        f"standardisation and from its anchor ranking)")
    log(f"  training conditions: {len(train_conditions)}")
    covered = int((np.abs(model.additive_weights.cpu().numpy()).sum(axis=1) > 0).sum())
    log(f"  perturbations the ridge covers: {covered} / {data.n_perturbations}"
        + ("" if covered == data.n_perturbations else
           "  <- the rest predict the control unchanged"))

    log("\n=== the observables (fixed) ===")
    for key, value in model.observables.summary().items():
        log(f"  {key:20s} {value}")

    log("\n=== the model ===")
    log(f"  {model.extra_repr()}")
    for key, value in model.learned_parameters().items():
        log(f"  {key:32s} {value:,}")

    log("\n=== premise check (training conditions only) ===")
    combinations = trainable_conditions(
        data, stats, train_conditions,
        model.additive_weights.detach().cpu().numpy(), data.pert_index)
    verify_premise(model, data, stats, combinations or train_conditions, log)

    log("\n=== training ===")
    parts = train(model, data, stats, train_conditions, config, device, rng, log,
                  run_dir=run_dir)

    log("\n=== evaluation (same protocol as the baselines) ===")
    results = evaluate_model(model, data, stats, [fold], method, config, rng)
    for key, value in results.items():
        log(f"  {key:24s} {value}")
    log("  resid_R2 is measured against the ADDITIVE arithmetic, so 0.0 means tying it. "
        "An untrained run of this model sits there by construction, and the additive "
        "component is a stronger predictor than that: see edist_rel_additive.")

    torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as out:
        json.dump({"config": config, "results": results, "train_loss": parts}, out,
                  indent=2)
    log(f"-> {run_dir}")
    handle.close()


if __name__ == "__main__":
    main()
