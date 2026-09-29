"""Step 0: compute the target line every model has to beat.

Metrics are evaluated in the MODELLED gene space, not the full 19,264. The handoff
table was computed on all genes, which the model never predicts, so those numbers
are a cross-check rather than the acceptance target - see docs/PIPELINE-v1.pdf.

    python scripts/run_baselines.py
    python scripts/run_baselines.py --set split.method=combinations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import anndata as ad
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import splits
from src.data.conventions import ConditionNaming
from src.eval import baselines, metrics

BASELINES = ["control", "additive", "per_gene_scaled", "ridge_additive", "pairwise_ridge"]


def scored(per_double: list[float]) -> int:
    """How many conditions actually produced a per-double R2.

    metrics.residual_r2 returns nan for a double whose residual does not stand
    clear of its own sampling noise, so this is the n behind resid_R2_mean.
    """
    return int(np.sum(~np.isnan(per_double))) if per_double else 0


def noise_floor_for(stats: baselines.ConditionMeans, double: str,
                    single_a: str, single_b: str) -> float:
    """Var(r) summed over genes, from each condition's own sampling variance.

    A control-cell null at matched sizes reproduces this exactly (measured
    lambda = 1.000 across n_eff 16-375), so no empirical correction is applied.
    """
    total = 0.0
    for condition in (double, single_a, single_b, stats.control_condition):
        total += float((stats.var[condition] / max(stats.n[condition], 1)).sum())
    return total


def evaluate(config: dict, x: np.ndarray, conditions: np.ndarray,
             stats: baselines.ConditionMeans, folds: list[dict], method: str,
             rng: np.random.Generator) -> dict:
    control_cells = x[conditions == stats.control_condition]
    n_gen = config["eval"]["n_gen_cells"]
    power = config["eval"]["edist_power"]
    device = config["eval"]["device"]
    perturbations = sorted({g for c in stats.mean for g in stats.naming.genes(c)})

    accumulators = {
        name: {"num": 0.0, "den": 0.0, "edist": [], "de20": [], "per_double": [], "skipped": 0}
        for name in BASELINES
    }

    for fold_index, fold in enumerate(folds):
        train_conditions = baselines.training_conditions(stats, fold, method)
        available = set(train_conditions)
        train_doubles = [c for c in fold["train"] if stats.naming.is_double(c)]
        scale = baselines.fit_per_gene_scale(stats, train_doubles, available)
        ridge = baselines.fit_ridge_additive(
            stats, train_conditions, perturbations,
            alpha=config["eval"]["ridge_alpha"],
            weight_by_cells=config["eval"]["ridge_weight_by_cells"])
        pairwise = baselines.fit_pairwise_ridge(
            stats, train_doubles, available, alpha=config["eval"]["ridge_alpha"])

        test_doubles = [c for c in fold["test"] if stats.naming.is_double(c)]
        for double in test_doubles:
            a, b = stats.naming.genes(double)
            single_a, single_b = stats.single_of(a), stats.single_of(b)
            if not all(stats.has(c) for c in (double, single_a, single_b)):
                continue

            m_ab, m_a, m_b = stats.mean[double], stats.mean[single_a], stats.mean[single_b]
            m_ctrl = stats.control
            real_cells = x[conditions == double]
            e_noise = noise_floor_for(stats, double, single_a, single_b)
            r = metrics.residual(m_ab, m_a, m_b, m_ctrl)
            delta_true = m_ab - m_ctrl

            pick = rng.choice(control_cells.shape[0], size=min(n_gen, control_cells.shape[0]),
                              replace=False)
            control_sample = control_cells[pick]

            for name in BASELINES:
                m_hat = baselines.predict(name, double, stats, available, scale=scale,
                                            ridge=ridge, pairwise=pairwise)
                if m_hat is None:
                    accumulators[name]["skipped"] += 1
                    continue
                acc = accumulators[name]
                r_hat = metrics.residual(m_hat, m_a, m_b, m_ctrl)
                acc["num"] += float((r_hat - r) @ (r_hat - r)) - e_noise
                acc["den"] += float(r @ r) - e_noise
                acc["per_double"].append(
                    metrics.residual_r2(m_hat, m_ab, m_a, m_b, m_ctrl, e_noise))
                acc["de20"].append(metrics.de20_pearson(m_hat - m_ctrl, delta_true))
                pred_cells = control_sample + (m_hat - m_ctrl)
                acc["edist"].append(
                    metrics.edist_rel(pred_cells, real_cells, control_sample,
                                      power=power, device=device))
        print(f"  fold {fold_index} done", end="\r")
    print()

    results = {}
    for name, acc in accumulators.items():
        computed = len(acc["edist"])
        results[name] = {
            "n_evaluated": computed,
            "n_skipped": acc["skipped"],
            "resid_R2_pooled": 1.0 - acc["num"] / acc["den"] if computed else float("nan"),
            # Doubles whose residual does not clear its own noise floor score nan
            # rather than a blown-up ratio (metrics.residual_r2), so the mean runs
            # over scored conditions only and n says how many those were.
            "resid_R2_mean": (float(np.nanmean(acc["per_double"]))
                              if scored(acc["per_double"]) else float("nan")),
            "resid_R2_mean_n": scored(acc["per_double"]),
            "edist_rel": float(np.nanmean(acc["edist"])) if computed else float("nan"),
            "r_de20": float(np.nanmean(acc["de20"])) if computed else float("nan"),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()

    config = config_module.load(args.overrides)
    method = config["split"]["method"]

    print(f"loading {config['data']['cache_h5ad']} ...")
    adata = ad.read_h5ad(config["data"]["cache_h5ad"])
    x = np.asarray(adata.X.todense(), dtype=np.float32)
    conditions = adata.obs["condition"].astype(str).values
    print(f"  {x.shape[0]:,} cells x {x.shape[1]:,} genes (modelled space)")

    started = time.time()
    naming = ConditionNaming.from_config(config)
    stats = baselines.ConditionMeans(x, conditions, naming)
    folds = splits.folds(config, method)
    rng = np.random.default_rng(config["eval"]["seed"])

    print(f"\n=== baselines on the '{method}' split, {len(folds)} folds ===")
    results = evaluate(config, x, conditions, stats, folds, method, rng)

    header = f"{'baseline':18s} {'n':>5s} {'skip':>5s} {'r_de20':>9s} {'edist_rel':>11s} {'resid_R2':>11s} {'(per-dbl)':>11s}"
    print("\n" + header)
    print("-" * len(header))
    for name in BASELINES:
        row = results[name]
        print(f"{name:18s} {row['n_evaluated']:5d} {row['n_skipped']:5d} "
              f"{row['r_de20']:9.4f} {row['edist_rel']:11.4f} "
              f"{row['resid_R2_pooled']:11.4f} "
              f"{row['resid_R2_mean']:11.4f} ({row['resid_R2_mean_n']:d})")
    print(f"\nelapsed {time.time() - started:.1f}s")

    os.makedirs("results", exist_ok=True)
    # The dataset goes in the filename. Without it a combosciplex run overwrites
    # the Norman baselines under the same name, and the summary then shows the
    # wrong target line - a bar that looks passed when it was not.
    dataset = os.path.splitext(os.path.basename(config["data"]["cache_h5ad"]))[0]
    out = f"results/baselines_{dataset}_{method}.json"
    with open(out, "w") as handle:
        json.dump({"config": config, "results": results}, handle, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
