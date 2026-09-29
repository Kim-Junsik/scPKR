"""Score our runs with scDFM's own metric code, unmodified.

    python scripts/eval_scdfm_style.py results/runs/<tag>
    python scripts/eval_scdfm_style.py --filter s2_pcab
    python scripts/eval_scdfm_style.py --filter s2_ --csv results/scdfm_style.csv

src/eval/scdfm_metrics.py holds their FlowMatchingMetrics verbatim. This file only
feeds it: for every held-out double it transports a control sample, decodes, and
hands the predicted and real matrices over.

TWO THINGS THIS DOES NOT DO, both worth knowing before quoting a number.

It does not reproduce the reported table. Of those eight columns only MSE and MAE
exist in their metric file; L2, DE-Spearman, Pearson delta, DS and the two
delta-hats are not implemented there. The paper's text says its table follows
cell-eval, which scripts/run_celleval.py already drives - use that for the table
and this for the extra distributional metrics.

It cannot pair cells. Their reconstruction, correlation and embedding metrics
index row i of the prediction against row i of the truth. Our prediction
transports control cells, so no such correspondence exists; counts are matched per
condition, and the pairing within a condition is arbitrary. Order-dependent
metrics are therefore marked [pair] in the output. They are not meaningless - an
elementwise MAE over an arbitrary pairing still estimates E|X-Y| between two
independent draws - but that is a distributional quantity, and it only compares to
scDFM if scDFM pairs the same way. Everything unmarked is order-invariant and
comparable without that assumption.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import condition_groups, load_run
from src.eval.predict import _head_aux
from src.models.flow import integrate
from src.eval.scdfm_metrics import (evaluate_flow_generation, median_sigmas,
                                    mmd2_unbiased_multi_sigma, print_metrics)

# Metrics whose value depends on which predicted cell is lined up against which
# real cell. Marked rather than dropped - see the module docstring.
NEEDS_PAIRING = {
    "recon_mae", "recon_mse", "recon_rmse", "recon_pos_mae", "recon_zero_mae",
    "recon_r2_score", "recon_relative_error", "dist_mean_correlation",
    "emb_mean_cosine_similarity", "emb_mean_euclidean_distance",
    "emb_neighbor_preservation",
}


@torch.no_grad()
def score_run(run_dir: str, device: str, max_cells: int) -> dict:
    """Their comprehensive_evaluation, averaged over the fold's test doubles."""
    config, data, stats, fold, vae, field = load_run(run_dir, device)
    rng = np.random.default_rng(config["eval"]["seed"])
    n_steps = config["train"]["n_integration_steps"]
    control = data.cells(data.control_condition)
    doubles = condition_groups(data, stats, fold, config["split"]["method"])["test doubles"]

    totals: dict[str, list[float]] = {}
    for condition in doubles:
        real = data.cells(condition)
        if max_cells and real.shape[0] > max_cells:
            real = real[rng.choice(real.shape[0], size=max_cells, replace=False)]
        n = real.shape[0]

        # Counts matched to the real condition, as celleval.build_pair does: an
        # unmatched count shows up as a distribution difference that has nothing
        # to do with the model.
        pick = rng.choice(control.shape[0], size=n, replace=control.shape[0] < n)
        x0 = torch.as_tensor(control[pick], device=device)
        z0, _ = vae.encode_z(x0)
        perturbations = [data.pert_index[g] for g in data.naming.genes(condition)]
        z1 = integrate(field, z0, perturbations, n_steps)
        generated = vae.reconstruction(vae.decode_z(z1), **_head_aux(vae, x0))

        target = torch.as_tensor(real, device=device)
        z_target, _ = vae.encode_z(target)

        scores = evaluate_flow_generation(generated, target, z1, z_target, device=device)
        # MMD is in their file but outside comprehensive_evaluation, so it is added
        # here rather than left unused. Latent space: at 3,074 genes the gene-space
        # kernel matrices are the same size but the median heuristic is dominated
        # by the ambient dimension.
        # Named with their flow_ prefix on purpose: print_metrics buckets by
        # prefix and silently drops anything that matches none of them.
        sigmas = median_sigmas(z_target)
        scores["flow_mmd2_latent"] = float(mmd2_unbiased_multi_sigma(z1, z_target, sigmas))

        for key, value in scores.items():
            if np.isfinite(value):
                totals.setdefault(key, []).append(float(value))

    return {"n_evaluated": len(doubles),
            **{key: float(np.mean(values)) for key, values in totals.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cells", type=int, default=256,
                        help="cap cells per condition; their per-gene correlation "
                             "loop is O(genes) scipy calls per condition")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    # A --filter widens the search to every run, then narrows by name. Filtering
    # the s2_* default instead made --filter useless for any other tag: the
    # candidate list never contained the run being asked for, and the script said
    # "no finished run matched" for a run sitting right there on disk.
    if args.runs:
        runs = args.runs
    elif args.filter:
        runs = [r for r in sorted(glob.glob("results/runs/*"))
                if args.filter in os.path.basename(r)]
    else:
        runs = sorted(glob.glob("results/runs/s2_*"))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "checkpoint.pt"))]
    if not runs:
        print("no finished run matched.")
        return

    table = []
    for run_dir in runs:
        name = os.path.basename(run_dir.rstrip("/\\"))
        scores = score_run(run_dir, args.device, args.max_cells)
        print_metrics({k: v for k, v in scores.items() if k != "n_evaluated"},
                      f"{name}  (n={scores['n_evaluated']} test doubles)")
        table.append({"run": name, **scores})

    if len(table) > 1:
        keys = [k for k in table[0] if k not in ("run", "n_evaluated")]
        header = f"{'run':32s} " + " ".join(f"{k[:13]:>14s}" for k in keys)
        print(header)
        print("-" * len(header))
        for row in table:
            print(f"{row['run'][:31]:32s} " +
                  " ".join(f"{row.get(k, float('nan')):14.5f}" for k in keys))

    marked = sorted(k for k in table[0] if k in NEEDS_PAIRING)
    print("\n[pair] these depend on which predicted cell faces which real cell, and"
          "\n       transported control cells have no such counterpart:"
          "\n       " + ", ".join(marked))
    print("everything else above is order-invariant.")
    print("\nnot in this metric file at all: L2, DE-Spearman, Pearson delta, DS,"
          "\nPearson delta-hat, delta-hat-20. For those:"
          "\n  python scripts/paper_table.py --filter s2_")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
