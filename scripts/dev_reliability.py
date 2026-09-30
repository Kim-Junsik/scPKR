"""Can a test-time quantity predict whether this condition's residual is trustworthy?

    python scripts/dev_reliability.py results/dev/w6/manifest.json \
        --shrink results/dev/w6/shrink.csv --arms wd1e5 --device cuda

THE MEASUREMENT THIS ANSWERS. scripts/dev_shrink.py found that a global residual scale s
buys -0.085 on combosciplex:sv and -0.056 on norman:nval, while the PER-CONDITION optimum
buys -0.250 and -0.114. A constant therefore captures a third to a half of what scaling can
reach, and the rest needs a scale that varies by condition. Passing scDFM on Table 3 needs
-10.8 % against ridge; the constant gives -5.1 % and the per-condition ceiling -14.9 %, so
this gap is exactly the difference between failing and passing.

It also explains why shrinking does not remove regressions: the regression count floors at
the number of conditions whose optimum is NEGATIVE (6 of 36 on combosciplex, 12 of 45 on
Norman). Those are sign failures, not scale failures, and since the optimum is
k = rho = cos(predicted residual, true residual), sign(best_s) IS sign(rho). No positive
constant can help them and only a rule that knows which conditions they are can.

WHAT MAY AND MAY NOT BE USED. The target, best_s, is computed from the held-out truth and
is therefore an ORACLE - it exists here to be predicted, never to be applied. Every
predictor below is computable at test time from the model and the training conditions
alone:

  overlap_obs     cos(|M w_a|, |M w_b|), the two perturbations' footprints in OBSERVABLE
                  coordinates. This is the model's own claim: B_S is an anticommutator, so
                  A_a A_b + A_b A_a has something to say only where a's and b's action
                  overlaps. Defined through w_a rather than through KEGG target membership
                  because combosciplex's perturbations are drugs with no target gene in the
                  data, so a membership rule would not exist on the family with the largest
                  gap. w_a is a ridge fit over TRAINING conditions and already a buffer in
                  the model, so this adds no leak and no parameters.
  overlap_gene    the same cosine in GENE coordinates. The control for the one above: if
                  the pathway pooling contributes nothing, these two will predict equally
                  well and the observable space is not earning its place.
  support_min     the smaller of the two perturbations' training-condition counts - how
                  well determined the weaker of w_a, w_b is.
  obs_norm        ||(exp(B_S) - I) Phi||, the size of the model's own extrapolation.
  b_norm          ||B_S||_F, and abscissa its spectral abscissa: how far from the identity
                  the composed flow is asked to travel.
  additive_norm   ||sum_a w_a||, the size of the move the closed-form part already made.

REPORTED AS RANK CORRELATION AND AS AUC, not as R^2. The quantity to be predicted is an
optimal scale with a heavy tail - single conditions reach -34 and +6.8 - so a squared
error would be decided by two of them. Spearman asks the question that matters, whether the
ORDER is right, and the AUC asks the cheaper and more valuable one: can a predictor
separate the conditions the residual helps from the ones it harms? A predictor with an AUC
of 0.5 is worth nothing whatever its correlation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dev_score import family, find_run

PREDICTORS = ("overlap_obs", "overlap_gene", "support_min", "support_sum",
              "obs_norm", "b_norm", "abscissa", "additive_norm", "residual_norm")


def ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation."""
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values), dtype=float)
    out[order] = np.arange(len(values), dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, out)
        out = (sums / counts)[inverse]
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return float("nan")
    ra, rb = ranks(a[keep]), ranks(b[keep])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    scale = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb) / scale if scale else float("nan")


def auc(score: np.ndarray, positive: np.ndarray) -> tuple[float, int, int]:
    """P(score higher on a helpful condition than on a harmful one), from ranks.

    Mann-Whitney U over n_pos * n_neg. Reported with both counts because an AUC over four
    negatives says nothing however far from 0.5 it lands.
    """
    keep = np.isfinite(score)
    score, positive = score[keep], positive[keep]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    r = ranks(score)
    u = float(r[positive].sum()) - n_pos * (n_pos - 1) / 2.0
    return u / (n_pos * n_neg), n_pos, n_neg


@torch.no_grad()
def predictors(run_dir: str, device: str, n_cells: int, chunk: int = 512) -> dict:
    """Per condition, every test-time quantity, keyed by condition name."""
    from src.eval.diagnostics import load_run, training_rows
    from src.eval.baselines import training_conditions
    from src.eval.conditions import condition_groups

    config, data, stats, fold, model = load_run(run_dir, device, "soft")
    method = config["split"]["method"]
    groups = condition_groups(data, stats, fold, method)
    train_conditions = training_conditions(stats, fold, method)

    # How many TRAINING conditions each perturbation appears in. The ridge determines w_a
    # from exactly these, so it is the natural measure of how well determined it is.
    support = collections.Counter()
    for condition in train_conditions:
        for gene in data.naming.genes(condition):
            support[gene] += 1

    weights = model.additive_weights.detach()
    matrix = model.observables.matrix            # [K, G], fixed
    control = data.cells(data.control_condition)
    rng = np.random.default_rng(config["eval"]["seed"] + 1)  # +1: these are predictors,
    # not the scored metric, so they must not consume the scoring draw sequence.

    out = {}
    for condition in groups["test doubles"] + groups["test singles"]:
        if condition == data.control_condition or not stats.has(condition):
            continue
        names = list(data.naming.genes(condition))
        idx = [data.pert_index[g] for g in names]
        row = {"n_perturbations": len(idx),
               "support_min": min((support[g] for g in names), default=0),
               "support_sum": sum(support[g] for g in names)}

        # Footprint overlap. |.| because a shared pathway pushed up by one perturbation and
        # down by the other is still a shared pathway - the question is whether the two act
        # on the same coordinates, not whether they agree about the direction.
        if len(idx) == 2:
            wa, wb = weights[idx[0]], weights[idx[1]]
            row["overlap_gene"] = _cos(wa.abs(), wb.abs())
            row["overlap_obs"] = _cos((matrix @ wa).abs(), (matrix @ wb).abs())
        else:
            row["overlap_gene"] = row["overlap_obs"] = float("nan")

        composed = model.operators.compose(idx)
        row["b_norm"] = float(torch.linalg.norm(composed))
        if len(idx) >= 2:
            eigenvalues = torch.linalg.eigvals(composed.to(torch.float32))
            row["abscissa"] = float(eigenvalues.real.max())
        else:
            row["abscissa"] = float("nan")
        row["additive_norm"] = float(torch.linalg.norm(model.additive(idx)))

        pick = rng.choice(control.shape[0], size=min(n_cells, control.shape[0]),
                          replace=False)
        sample = control[pick]
        obs, res = [], []
        for start in range(0, sample.shape[0], max(chunk, 1)):
            x = torch.as_tensor(sample[start:start + chunk], device=device)
            obs.append(model.observable_displacement(x, idx).cpu().numpy())
            res.append(model.residual(x, idx).cpu().numpy())
        row["obs_norm"] = float(np.linalg.norm(np.concatenate(obs).mean(axis=0)))
        row["residual_norm"] = float(np.linalg.norm(np.concatenate(res).mean(axis=0)))
        out[condition] = row
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = float(torch.linalg.norm(a) * torch.linalg.norm(b))
    return float(a @ b) / scale if scale else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--shrink", required=True,
                        help="the CSV scripts/dev_shrink.py --out wrote")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--out", default=None, help="the joined per-condition CSV")
    args = parser.parse_args()

    with open(args.shrink, encoding="utf-8") as handle:
        oracle = [r for r in csv.DictReader(handle)
                  if not args.arms or r["arm"] in args.arms]
    if not oracle:
        raise SystemExit(f"no rows in {args.shrink} matched --arms")

    jobs = []
    for path in args.manifests:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for job in manifest["jobs"]:
            if job.get("kind") != "arm" or (args.arms and job["arm"] not in args.arms):
                continue
            run_dir = find_run(job["tag"])
            if run_dir is not None:
                jobs.append((job, run_dir))
    print(f"{len(jobs)} runs, {len(oracle)} oracle rows\n")

    table = {}
    for job, run_dir in jobs:
        for condition, row in predictors(run_dir, args.device, args.n_cells).items():
            table[(job["group"], job["arm"], job["seed"], condition)] = row
        print(f"  {job['tag']:38s} done")

    records = []
    for row in oracle:
        key = (row["group"], row["arm"], int(row["seed"]), row["condition"])
        if key not in table:
            continue
        best = float(row["best_s"])
        records.append({
            "family": row["family"], "group": row["group"], "arm": row["arm"],
            "seed": int(row["seed"]), "condition": row["condition"],
            "block": row["block"], "best_s": best,
            # Carried through from the sweep so this CSV alone determines L2 at any s:
            # L2(s) = sqrt(ee + 2 s er + s^2 rr).
            "ee": float(row["ee"]), "er": float(row["er"]), "rr": float(row["rr"]),
            # The oracle, split into the two questions. `helps` is sign(rho), which is the
            # only thing that decides whether ANY positive scale can improve a condition.
            "helps": best > 0.0,
            "gain": float(row["s0"]) - float(row["best_l2"]),
            **table[key],
        })
    if not records:
        raise SystemExit("the CSV and the manifests did not join - same --arms?")

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\n-> {args.out}")

    for key in sorted({r["family"] for r in records}):
        mine = [r for r in records if r["family"] == key and r["block"] == "double"]
        if len(mine) < 6:
            continue
        best = np.array([r["best_s"] for r in mine])
        helps = np.array([r["helps"] for r in mine])
        print(f"\n=== {key}   {len(mine)} double condition-runs, "
              f"{int(helps.sum())} helped / {int((~helps).sum())} harmed ===")
        print(f"{'predictor':>16} {'rho(best_s)':>12} {'AUC(helps)':>11} "
              f"{'rho(gain)':>10}")
        n_pos, n_neg = int(helps.sum()), int((~helps).sum())
        for name in PREDICTORS:
            values = np.array([float(r.get(name, float("nan"))) for r in mine])
            if not np.isfinite(values).any():
                continue
            area, n_pos, n_neg = auc(values, helps)
            print(f"{name:>16} {spearman(values, best):12.3f} {area:11.3f} "
                  f"{spearman(values, np.array([r['gain'] for r in mine])):10.3f}")
        print(f"  (AUC over {n_pos} helped and {n_neg} harmed conditions; 0.5 is "
              f"worthless and below 0.5 means the predictor is inverted, which is "
              f"equally usable.)")
    print("\nbest_s is an ORACLE computed from held-out truth. It is the target, never an "
          "input. Every predictor above uses only the model and the training conditions.")


if __name__ == "__main__":
    main()
