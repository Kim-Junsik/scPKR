"""The reported metric table, joined from the two places its numbers live.

    python scripts/paper_table.py                    every s2_* run
    python scripts/paper_table.py --filter s2_pcab   one sweep
    python scripts/paper_table.py --no-l2            skip the checkpoint pass

Eight metrics are reported in the literature this is compared against. They do not
all come from one place, and two of them cannot be produced here at all:

  MSE, MAE, DE-Spearman, Pearson delta, DS   cell-eval, from celleval/agg_results.csv
  L2                                          not a cell-eval metric; computed here
  Pearson delta-hat, delta-hat-20             NOT COMPUTABLE under this protocol

The last two are printed as n/a rather than dropped, because a missing column in a
paper table reads as an oversight. Both are defined as a per-cell correlation of
residuals taken against x_train^(p), the centroid of perturbation p IN THE TRAINING
SET. This split holds out whole double perturbations, so no evaluated p appears in
training and that centroid does not exist; and prediction transports control cells,
so no cell-to-cell correspondence exists to correlate over either. Reporting them
would take a second evaluation under a within-perturbation cell split - a different
experiment, not a missing function.

L2 is eq. (15): mean over test perturbations of ||mu_hat_p - mu_p||_2, on the mean
vectors themselves rather than on the delta from control. It needs the model, so it
costs one transport pass per run; --no-l2 skips it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import (condition_groups, load_run, measure_transport,
                                  scdfm_eval_genes)
# The export folder's name comes from the script that WRITES it, never rebuilt here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_celleval import celleval_dir

# (column in agg_results.csv, header, higher-is-better). ASCII headers on purpose:
# a Korean Windows console is cp949 and mangles the arrows and greek the paper uses.
CELLEVAL = [
    ("mse", "MSE", False),
    ("mae", "MAE", False),
    ("de_spearman_lfc_sig", "DE_rho", True),
    ("pearson_delta", "Pearson_d", True),
    ("discrimination_score_l1", "DS", True),
]
NOT_COMPUTABLE = ["Pears_dhat", "Pears_dhat20"]


def celleval_means(run_dir: str, gate: str | None = None,
                   group: str = "double",
                   alpha: str | None = None) -> dict[str, str] | None:
    """Reads the scoring produced under the SAME gate the L2 pass will use.

    run_celleval.py writes to celleval_<gate>[_a<alpha>]/, and the name is built by
    ITS OWN celleval_dir, imported here rather than rebuilt - so mixing a soft L2
    with a sample cell-eval, or a corrected L2 with uncorrected columns (five of the
    eight silently coming from different predictions), is not expressible.

    Averaged from the per-condition results.csv rather than read off
    agg_results.csv, because the aggregate cannot be split. Under the
    combination holdout celleval.build_pair scores everything in fold["test"],
    and that is 15 doubles AND the ~23 singles of the held-out genes; the
    literature reports those as two blocks. Taking the aggregate would have
    compared our mixed number against their Double column, with the easier
    singles making up the majority of our rows.

    The two agree exactly when nothing is filtered: checked on an additive run,
    all five metrics identical to 1e-16.
    """
    path = os.path.join(run_dir, celleval_dir(gate, alpha), "results.csv")
    if not os.path.exists(path):
        return None
    # celleval labels a double 'A+B' and a single 'A' (celleval.to_celleval_label),
    # so the separator is the arity.
    keep = {"double": lambda p: "+" in p,
            "single": lambda p: "+" not in p,
            "all": lambda p: True}[group]
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if keep(r.get("perturbation", ""))]
    if not rows:
        return None
    means: dict[str, str] = {}
    for key, _, _ in CELLEVAL:
        values = []
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            means[key] = str(float(np.mean(values)))
    means["n"] = str(len(rows))
    return means


def compute_l2(run_dir: str, device: str, n_cells: int, gate: str | None = None,
               infer_top_gene: int | None = None, group: str = "double",
               alpha: str | None = None) -> float:
    """Eq. (15) over the fold's test doubles - the same conditions resid_R2 uses.

    `infer_top_gene` restricts the gene space to the subset scDFM scores on, which
    is the only way the L2 columns compare: theirs is 1,000 scanpy-HVG genes of
    the test subset, ours is every gene in the cache.
    """
    config, data, stats, fold, vae, field = load_run(run_dir, device, gate,
                                                     alpha=alpha)
    rng = np.random.default_rng(config["eval"]["seed"])
    conditions = condition_groups(data, stats, fold, config["split"]["method"])
    genes = scdfm_eval_genes(data, fold, infer_top_gene) if infer_top_gene else None
    wanted = {"double": conditions["test doubles"],
              "single": conditions["test singles"],
              "all": conditions["test doubles"] + conditions["test singles"]}[group]
    rows = measure_transport(vae, field, data, stats, wanted,
                             config, rng, device, n_cells, genes=genes)
    return float(np.mean([r["l2"] for r in rows])) if rows else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--no-l2", action="store_true", help="skip the model pass")
    parser.add_argument("--alpha", default=None, choices=("none", "mean", "cell"),
                        help="post-hoc magnitude correction (eval.magnitude_alpha), "
                             "fitted on this run's TRAINING conditions. Must be "
                             "passed here for any checkpoint written before the "
                             "option existed, which carries no such key.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=256)
    parser.add_argument("--infer-top-gene", type=int, default=None,
                        help="score L2 on scanpy-HVG genes of the test subset, as "
                             "scDFM does (their run.sh uses 1000)")
    parser.add_argument("--gate", default=None,
                        choices=["soft", "hard", "sample"],
                        help="score under this hurdle gate instead of the one in "
                             "the checkpoint. L2 is recomputed with it and the "
                             "cell-eval columns are read from celleval_<gate>/, "
                             "so all six columns come from one point estimate. "
                             "Produce that folder first: "
                             "run_celleval.py <run> --gate <gate> --profile full")
    parser.add_argument("--mean", action="store_true",
                        help="add a row averaging every run listed, with the "
                             "per-column standard deviation underneath. This is "
                             "the number to report: the cited baselines are means "
                             "over the same five folds.")
    parser.add_argument("--group", default="double",
                        choices=["double", "single", "all"],
                        help="which held-out conditions to score. The additive "
                             "split holds out only doubles, so this changes "
                             "nothing there. The combination holdout also holds "
                             "out the singles of every held-out gene, and the "
                             "literature reports Single and Double as separate "
                             "blocks - run this twice to fill both.")
    parser.add_argument("--csv", default=None, help="also write the table here")
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

    print(f"scoring the {args.group} held-out conditions")
    headers = ["L2"] + [h for _, h, _ in CELLEVAL] + NOT_COMPUTABLE
    line = f"{'run':32s} " + " ".join(f"{h:>12s}" for h in headers)
    print(line)
    print("-" * len(line))

    table = []
    for run_dir in runs:
        name = os.path.basename(run_dir.rstrip("/\\"))
        values = celleval_means(run_dir, args.gate, args.group, args.alpha)
        l2 = (float("nan") if args.no_l2 else
              compute_l2(run_dir, args.device, args.n_cells, args.gate,
                         args.infer_top_gene, args.group, args.alpha))

        cells = [f"{l2:12.4f}" if np.isfinite(l2) else f"{'-':>12s}"]
        record = {"run": name, "L2": l2}
        for key, header, _ in CELLEVAL:
            raw = (values or {}).get(key)
            try:
                number = float(raw)
                cells.append(f"{number:12.4f}")
                record[header] = number
            except (TypeError, ValueError):
                # No agg_results.csv yet, or the metric was not in the profile that
                # was run. Distinguished from a real zero by the dash.
                cells.append(f"{'-':>12s}")
                record[header] = None
        cells += [f"{'n/a':>12s}" for _ in NOT_COMPUTABLE]
        print(f"{name[:31]:32s} " + " ".join(cells))
        table.append(record)

    print("\nlower is better: L2, MSE, MAE.  higher is better: DE_rho, Pearson_d, DS.")
    print("'-' means the run has no cell-eval score yet:"
          "\n  python scripts/run_celleval.py <run> --profile full")
    print("'n/a' means the metric is not defined under a combination-holdout split;"
          "\n  see this script's docstring for why.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run", "L2"] +
                                    [h for _, h, _ in CELLEVAL])
            writer.writeheader()
            writer.writerows(table)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
