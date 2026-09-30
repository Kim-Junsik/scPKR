"""Sweep the residual multiplier s over ALREADY-TRAINED runs. No training, no GPU hours.

    python scripts/dev_shrink.py results/dev/w6/manifest.json --arms wd1e5 \
        --scales 0 0.2 0.4 0.6 0.8 1.0 --device cuda --out results/dev/w6/shrink.csv

WHY THIS IS EXACT RATHER THAN AN APPROXIMATION. The reported metric is the L2 between the
predicted population MEAN and the true one, the scoring gate is `soft`, and the head is
mean-preserving - so the predicted mean is

    mean_i(x_i) + sum_a w_a + s * mean_i(residual(x_i))

which is AFFINE in s. One forward pass per condition gives the two vectors and every s
follows in closed form, including values no run was ever trained at. The script does not
take that on trust: --check re-scores one run through scripts/dev_score.py's own path at
s = 1 and prints both numbers.

WHY s EXISTS AT ALL. Measured over 27 validation conditions: the residual's cosine against
the true residual runs from -0.51 to +0.82 while its norm ratio does not track it. The
error is ||r|| sqrt(1 - 2 k rho + k^2), so a condition with rho < 0 is made WORSE by any
residual whatsoever - four conditions regressed, one by +0.555 - and the optimum k = rho is
a PER-CONDITION quantity that the model currently answers with a constant. A global s
cannot repair that, but it can buy the average of it, and this measures how much.

WHY NOT WEIGHT DECAY, WHICH WAS TRIED FIRST. Experiment w6 raised it from 1e-5 to 1e-2, a
thousandfold, and L2 moved by 0.003 on both families. The gene-space endpoint loss returns
the residual to whatever scale fits the TRAINING conditions and a penalty on ||W|| cannot
outbid it. s acts after training, where no loss can push back.

s IS SELECTED ON THESE VALIDATION FOLDS, which makes it one more validation-chosen scalar
alongside the loss weights and the epoch count, and it is reported as one.
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

from dev_score import TABLE3_WEIGHTS, family, find_run

DEFAULT_SCALES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@torch.no_grad()
def affine_pieces(run_dir: str, device: str, n_cells: int,
                  chunk: int = 512) -> list[dict]:
    """Per condition, the two vectors L2(s) is built from, on the reported gene set.

    Reproduces scripts/dev_score.py's protocol exactly: the same gate, the same eval
    genes, the same rng seed, and the same draw ORDER - doubles before singles, one
    rng.choice per condition, skipping the same conditions - because measure_transport
    consumes its draws that way and a different order would sample different control
    cells and give a different number at the same s.
    """
    from src.eval.diagnostics import load_run
    from src.eval.conditions import condition_groups, scdfm_eval_genes

    config, data, stats, fold, model = load_run(run_dir, device, "soft")
    rng = np.random.default_rng(config["eval"]["seed"])
    groups = condition_groups(data, stats, fold, config["split"]["method"])
    doubles, singles = groups["test doubles"], groups["test singles"]
    genes = scdfm_eval_genes(data, fold, 1000)
    control = data.cells(data.control_condition)

    rows = []
    for condition in doubles + singles:
        if condition == data.control_condition or not stats.has(condition):
            continue
        pick = rng.choice(control.shape[0], size=min(n_cells, control.shape[0]),
                          replace=False)
        sample = control[pick]
        perturbations = [data.pert_index[g] for g in data.naming.genes(condition)]
        additive = model.additive(perturbations).cpu().numpy()
        parts = []
        for start in range(0, sample.shape[0], max(chunk, 1)):
            x = torch.as_tensor(sample[start:start + chunk], device=device)
            parts.append(model.residual(x, perturbations).cpu().numpy())
        residual = np.concatenate(parts, axis=0).mean(axis=0)
        rows.append({
            "condition": condition,
            "block": "single" if condition in singles else "double",
            # base is the prediction at s = 0: the additive baseline on this exact draw.
            "base": (sample.mean(axis=0) + additive)[genes],
            "residual": residual[genes],
            "true": stats.mean[condition][genes],
        })
    return rows


def l2_curve(row: dict, scales) -> list[float]:
    """||base + s r - true|| at each s. Affine in s, so nothing is re-predicted."""
    error = row["base"] - row["true"]
    return [float(np.linalg.norm(error + s * row["residual"])) for s in scales]


def quadratic(row: dict) -> tuple[float, float, float]:
    """(||e||^2, e.r, ||r||^2), so L2(s) = sqrt(ee + 2 s er + s^2 rr) for ANY s.

    Emitted because it makes every downstream question free. A per-condition scaling rule
    is evaluated by putting its s into this expression - no run is reloaded, no cell is
    re-predicted, and the answer is exact rather than interpolated off the grid. A single
    perturbation has r = 0, so rr = 0 and the L2 is constant in s, which is the structural
    property rather than a special case to guard.
    """
    error, residual = row["base"] - row["true"], row["residual"]
    return (float(error @ error), float(error @ residual), float(residual @ residual))


def optimum(row: dict) -> tuple[float, float]:
    """The s minimising THIS condition's L2, in closed form, and the L2 there.

    d/ds ||e + s r||^2 = 0 at s = -(e . r) / ||r||^2. Reported UNCLIPPED, negative values
    included: a condition whose residual points the wrong way has its optimum below zero,
    and clipping would hide the exact failure mode being measured.
    """
    error, residual = row["base"] - row["true"], row["residual"]
    denominator = float(residual @ residual)
    if denominator <= 0.0:
        return float("nan"), float(np.linalg.norm(error))
    s = -float(error @ residual) / denominator
    return s, float(np.linalg.norm(error + s * residual))


def weighted(double: float, single: float | None) -> float:
    """dev_score.weighted_l2's rule: Table 3's weighting when a single was scored."""
    if single is None or not np.isfinite(single):
        return double
    return TABLE3_WEIGHTS["double"] * double + TABLE3_WEIGHTS["single"] * single


def collect(manifests: list[str], arms: list[str] | None) -> list[tuple[dict, str]]:
    jobs = []
    for path in manifests:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for job in manifest["jobs"]:
            if job.get("kind") != "arm" or (arms and job["arm"] not in arms):
                continue
            run_dir = find_run(job["tag"])
            if run_dir is not None:
                jobs.append((job, run_dir))
    return jobs


def report(records: list[dict], scales: list[float]) -> None:
    for key in sorted({r["family"] for r in records}):
        mine = [r for r in records if r["family"] == key]
        for arm in sorted({r["arm"] for r in mine}):
            rows = [r for r in mine if r["arm"] == arm]
            runs = len({(r["group"], r["seed"]) for r in rows})
            print(f"\n=== {key}  arm {arm}   {runs} runs, {len(rows)} condition-runs ===")
            print(f"{'s':>6} {'double':>9} {'single':>9} {'metric':>9} {'vs s=0':>9}  "
                  f"{'regressed':>9}")
            # Every regression is counted against the SAME condition-run at s = 0, which
            # is the additive baseline on the same draw - not against a pooled mean.
            reference = {(r["group"], r["seed"], r["condition"]): r["s0"] for r in rows}
            zero = float("nan")
            for s in scales:
                column = f"s{s:g}"
                doubles = [r[column] for r in rows if r["block"] == "double"]
                singles = [r[column] for r in rows if r["block"] == "single"]
                double = float(np.mean(doubles)) if doubles else float("nan")
                single = float(np.mean(singles)) if singles else None
                total = weighted(double, single)
                if s == 0.0:
                    zero = total
                worse = sum(1 for r in rows
                            if r[column] > reference[(r["group"], r["seed"],
                                                      r["condition"])] + 1e-9)
                print(f"{s:6.2f} {double:9.4f} "
                      f"{(float('nan') if single is None else single):9.4f} "
                      f"{total:9.4f} {total - zero:+9.4f}  {worse:4d}/{len(rows):<4d}")

            # The per-condition optimum is the CEILING any scaling rule could reach. The
            # gap between it and the best global s is exactly what a constant costs, and
            # the spread of its s says whether a constant was ever going to be enough.
            best_d = [r["best_l2"] for r in rows if r["block"] == "double"]
            best_s = [r["best_l2"] for r in rows if r["block"] == "single"]
            ceiling = weighted(float(np.mean(best_d)) if best_d else float("nan"),
                               float(np.mean(best_s)) if best_s else None)
            finite = [r["best_s"] for r in rows if np.isfinite(r["best_s"])]
            print(f"  per-condition optimum, a ceiling and not achievable: "
                  f"{ceiling:.4f}  ({ceiling - zero:+.4f})")
            if finite:
                print(f"  its s: median {np.median(finite):+.3f}, range "
                      f"{min(finite):+.3f} to {max(finite):+.3f}, "
                      f"{sum(1 for v in finite if v < 0)}/{len(finite)} negative")


def check(run_dir: str, device: str, n_cells: int, rows: list[dict]) -> None:
    """Re-score one run through dev_score's own path and print both numbers at s = 1.

    The affine identity rests on the head being mean-preserving under the soft gate. That
    is asserted by scripts/train.py on every run and by tests/test_structure.py, but it is
    the one assumption this whole script stands on, so it is measured here too.
    """
    from dev_score import score_run

    blocks = score_run(run_dir, device, n_cells)
    for block in ("double", "single"):
        mine = [r["s1"] for r in rows if r["block"] == block]
        if not mine or blocks.get(block) is None:
            continue
        theirs = blocks[block]
        print(f"  check {block:7s} sweep {np.mean(mine):.6f}  "
              f"dev_score {theirs:.6f}  diff {abs(np.mean(mine) - theirs):.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="arms to sweep; default every arm in the manifests")
    parser.add_argument("--scales", nargs="*", type=float, default=list(DEFAULT_SCALES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--out", default=None, help="per-condition CSV")
    parser.add_argument("--check", action="store_true",
                        help="re-score the first run through dev_score at s = 1")
    args = parser.parse_args()

    scales = sorted(set(args.scales) | {0.0})   # s = 0 is the additive baseline and the
                                                # reference every regression is counted
                                                # against, so it is never optional.
    jobs = collect(args.manifests, args.arms)
    if not jobs:
        raise SystemExit("no finished runs matched")
    print(f"{len(jobs)} runs, s in {[f'{s:g}' for s in scales]}\n")

    records, first = [], None
    for job, run_dir in jobs:
        rows = affine_pieces(run_dir, args.device, args.n_cells)
        for row in rows:
            curve = l2_curve(row, scales)
            best_s, best_l2 = optimum(row)
            ee, er, rr = quadratic(row)
            records.append({
                "family": family(job["dataset"], job["group"]), "group": job["group"],
                "arm": job["arm"], "seed": job["seed"], "condition": row["condition"],
                "block": row["block"], "best_s": best_s, "best_l2": best_l2,
                "ee": ee, "er": er, "rr": rr,
                **{f"s{s:g}": value for s, value in zip(scales, curve)},
            })
        print(f"  {job['tag']:38s} {len(rows)} conditions")
        if first is None:
            first = (run_dir, [dict(r, s1=l2_curve(r, [1.0])[0]) for r in rows])

    if args.check and first is not None:
        print("\n=== the affine identity, against dev_score's own path ===")
        check(first[0], args.device, args.n_cells, first[1])

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\n-> {args.out}")

    report(records, scales)
    print("\ns is selected on these validation folds only. A negative per-condition "
          "optimum means that condition's residual points the wrong way, so no positive "
          "global s can help it - only a per-condition rule could.")


if __name__ == "__main__":
    main()
