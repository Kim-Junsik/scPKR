"""The baselines in the metric the paper reports: L2, eq. (15).

    python scripts/baseline_l2.py --from results/runs/fin_combosciplex_scdfm7_affine_learned_f0
    python scripts/baseline_l2.py --from results/runs/fin_affine_learned_f1 --group double
    python scripts/baseline_l2.py --from results/runs/fin_combinations_affine_learned_f1 \
        --group single --csv results/baseline_l2_norman_comb_single.csv

WHY THIS EXISTS. scripts/run_baselines.py scores the baselines on resid_R2 and
edist_rel, and only over test DOUBLES. The reported tables are L2, and Table 3's L2
is a mean over five combinations AND two singles. So the target line every model is
measured against has never been computed in the metric the model is reported in, and
the baseline row of Table 3 is empty - which is the first thing a reviewer asks
about, because ridge_additive reaches resid_R2 0.5329 with no interaction term at
all and 88 % of this task's signal is additive.

COMPARABLE TO paper_table.py BY CONSTRUCTION. Same conditions (condition_groups),
same gene subset (scdfm_eval_genes), same reference (stats.mean), same reduction
(mean of ||mu_hat - mu||_2 over conditions). --from reads the config out of a
finished run's checkpoint, so the cache, split, fold and gene space are that run's
and not a guess.

ONE DIFFERENCE, AND IT FAVOURS NOTHING. A model's mu_hat is the mean of a
transported cell POPULATION; a baseline's is a mean vector directly. L2 is a
statement about means either way, so the comparison is fair - but it is also why
this script reports no distributional metric. edist_rel and DS need a population
and the baselines do not produce one.

BUILT-IN CHECK. The `control` row predicts the control mean and nothing else, so its
L2 is the distance from control to each condition - the scale of the whole table.
On combosciplex Table 3 it should come out near 5.36 (the brief records 5.3606
against the reference pipeline's 5.3716). If it does not, the protocol here does not
match the one the tables were made with, and no other row should be quoted.

SINGLES. A single condition is only computable for the baselines that never read it:
`control`, and `ridge_additive` whose per-perturbation effect vector is fitted over
the TRAINING conditions the drug appears in. `additive`, `per_gene_scaled` and
`pairwise_ridge` all need that drug's own single, which is the held-out condition -
so they are reported as skipped rather than filled in from it. That asymmetry is the
point: on combosciplex 13 of 17 drugs never appear alone in training, and both test
singles are drugs of that kind.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import splits
from src.data.conventions import ConditionNaming
from src.data.dataset import PerturbationData
from src.eval import baselines
from src.eval.conditions import condition_groups, scdfm_eval_genes

BASELINES = ["control", "additive", "per_gene_scaled", "ridge_additive", "pairwise_ridge"]


def predict_any(name: str, condition: str, stats: baselines.ConditionMeans,
                available: set[str], scale, ridge, pairwise) -> np.ndarray | None:
    """baselines.predict, extended to single conditions.

    baselines.predict unpacks exactly two perturbations, so it cannot be called on a
    single. The rules below are the same ones it applies: a baseline may read only
    `available`, and a baseline that would have to read the condition it is
    predicting returns None instead of leaking it.
    """
    arity = stats.naming.arity(condition)
    if arity == 2:
        return baselines.predict(name, condition, stats, available, scale=scale,
                                 ridge=ridge, pairwise=pairwise)
    if arity != 1:
        return None

    if name == "control":
        return stats.control.copy()
    gene = stats.naming.genes(condition)[0]
    if name == "ridge_additive":
        if gene not in ridge["covered"]:
            return None
        return stats.control + ridge["w"][ridge["index"][gene]]
    # additive, per_gene_scaled and pairwise_ridge are all functions of this drug's
    # own single, which IS this condition. Not computable without leaking it.
    return None


def load_config(args) -> tuple[dict, str]:
    if args.run_dir:
        path = os.path.join(args.run_dir, "checkpoint.pt")
        if not os.path.exists(path):
            raise SystemExit(f"no checkpoint at {path}")
        config = torch.load(path, map_location="cpu", weights_only=False)["config"]
        config = config_module.apply_overrides(config, args.overrides)
        return config, os.path.basename(args.run_dir.rstrip("/\\"))
    return config_module.load(args.overrides), "config"


def build(config: dict):
    naming = ConditionNaming.from_config(config)
    data = PerturbationData(config["data"]["cache_h5ad"], naming=naming)
    labels = np.empty(data.x.shape[0], dtype=object)
    for condition, rows in data.rows.items():
        labels[rows] = condition
    stats = baselines.ConditionMeans(data.x, labels, naming)
    method = config["split"]["method"]
    fold = splits.folds(config, method)[config["split"]["fold"]]
    return data, stats, method, fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="run_dir", default=None,
                        help="a finished run whose config defines the cache, split, "
                             "fold and gene space. Strongly preferred: it is what "
                             "makes these numbers comparable to that run's L2.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="config overrides, applied on top of --from")
    parser.add_argument("--group", default="all", choices=("single", "double", "all"),
                        help="which held-out block. Table 3 is 'all' (5 combinations "
                             "+ 2 singles); Table 1 is 'double'; Table 2 reports "
                             "'single' and 'double' separately.")
    parser.add_argument("--infer-top-gene", type=int, default=1000,
                        help="score on the scanpy-HVG genes of the test subset, as "
                             "the reported tables do. 0 uses every modelled gene, "
                             "which is NOT the reported protocol.")
    parser.add_argument("--fallback", default="skip", choices=("skip", "control"),
                        help="what to do with a condition the baseline cannot "
                             "predict. skip drops it, so the row is a mean over "
                             "FEWER conditions than the models are scored on and "
                             "the two numbers do not compare. control predicts the "
                             "control mean instead, which is what a model does for "
                             "the same conditions anyway: a perturbation with no "
                             "training condition keeps its zero-initialised "
                             "operator and transports nothing (10-15 % of Table 2's "
                             "Single block). Use control to put a baseline in the "
                             "same table as a model.")
    parser.add_argument("--per-condition", action="store_true",
                        help="also print each condition's L2")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    config, label = load_config(args)
    data, stats, method, fold = build(config)
    groups = condition_groups(data, stats, fold, method)
    wanted = {"double": groups["test doubles"],
              "single": groups["test singles"],
              "all": groups["test doubles"] + groups["test singles"]}[args.group]
    genes = (scdfm_eval_genes(data, fold, args.infer_top_gene)
             if args.infer_top_gene else None)

    train_conditions = baselines.training_conditions(stats, fold, method)
    available = set(train_conditions)
    train_doubles = [c for c in fold["train"] if stats.naming.is_double(c)]
    perturbations = sorted({g for c in stats.mean for g in stats.naming.genes(c)})
    scale = baselines.fit_per_gene_scale(stats, train_doubles, available)
    ridge = baselines.fit_ridge_additive(
        stats, train_conditions, perturbations,
        alpha=config["eval"]["ridge_alpha"],
        weight_by_cells=config["eval"]["ridge_weight_by_cells"])
    pairwise = baselines.fit_pairwise_ridge(
        stats, train_doubles, available, alpha=config["eval"]["ridge_alpha"])

    print(f"config from      : {label}")
    print(f"cache            : {config['data']['cache_h5ad']}")
    print(f"split            : {method} fold {config['split']['fold']}")
    print(f"group            : {args.group}  ({len(wanted)} conditions)")
    print(f"genes            : {len(genes) if genes is not None else data.n_genes}"
          f"{' (scanpy HVG of the test subset)' if genes is not None else ' (all modelled)'}")
    print(f"training conditions fitted on: {len(train_conditions)}")
    print()

    rows, per_condition = [], {}
    for name in BASELINES:
        values, skipped = [], []
        for condition in wanted:
            if not stats.has(condition):
                continue
            m_hat = predict_any(name, condition, stats, available, scale, ridge, pairwise)
            if m_hat is None:
                skipped.append(condition)
                if args.fallback != "control":
                    continue
                m_hat = stats.control
            truth, predicted = stats.mean[condition], m_hat
            if genes is not None:
                truth, predicted = truth[genes], predicted[genes]
            values.append(float(np.linalg.norm(predicted - truth)))
            per_condition.setdefault(condition, {})[name] = values[-1]
        rows.append({"baseline": name,
                     "L2": float(np.mean(values)) if values else float("nan"),
                     "n": len(values), "skipped": len(skipped),
                     "skipped_conditions": skipped})

    width = max(len(r["baseline"]) for r in rows)
    print(f"{'baseline':{width}s} {'L2':>9} {'n':>4} {'skipped':>8}")
    print("-" * (width + 24))
    for r in rows:
        value = f"{r['L2']:9.4f}" if np.isfinite(r["L2"]) else f"{'-':>9}"
        print(f"{r['baseline']:{width}s} {value} {r['n']:4d} {r['skipped']:8d}")
    filled = ("predicted as the CONTROL mean, so n matches the models"
              if args.fallback == "control" else
              "NOT filled in, so n is smaller than the models' and the rows do not "
              "compare;\n            re-run with --fallback control for that")
    print(f"\nlower is better. 'skipped' = conditions the baseline cannot predict "
          f"without\nreading the held-out condition itself. They are {filled}.")

    control = next(r for r in rows if r["baseline"] == "control")
    print(f"\nscale check: control L2 = {control['L2']:.4f}. This is the distance from "
          f"the\ncontrol mean to each condition - the size of the whole problem. "
          f"Compare it\nagainst the value recorded for this table before quoting any "
          f"other row.")

    for r in rows:
        if r["skipped_conditions"]:
            shown = ", ".join(r["skipped_conditions"][:4])
            more = ", ..." if len(r["skipped_conditions"]) > 4 else ""
            print(f"\n{r['baseline']}: skipped {r['skipped']} ({shown}{more})")

    if args.per_condition:
        print()
        names = [r["baseline"] for r in rows]
        print(f"{'condition':34s} " + " ".join(f"{n[:11]:>11s}" for n in names))
        for condition in wanted:
            cells = [f"{per_condition.get(condition, {}).get(n, float('nan')):11.4f}"
                     if n in per_condition.get(condition, {}) else f"{'-':>11s}"
                     for n in names]
            print(f"{condition[:33]:34s} " + " ".join(cells))

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["baseline", "L2", "n", "skipped"])
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in ("baseline", "L2", "n", "skipped")})
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
