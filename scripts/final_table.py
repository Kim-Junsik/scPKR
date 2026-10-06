"""Score the TEST set once and print the four reported tables.

    python scripts/final_table.py results/final/manifest.json \
        --calibration norman:additive=1.2,0.7 norman:combinations=2.22,0.7 \
                      combosciplex:list=1.6,0.8 \
        --score-the-test --device cuda --csv results/final/table.csv

THIS SCORES THE REPORTED TEST SET. It refuses to run without --score-the-test, because
that is the one action this repository is organised to do exactly once: scPKFM scored its
test set thirteen times while selecting and afterwards none of its numbers could be read.
Every choice behind these runs - whitened observables, dense readout, the calibration's
exponent, epochs, loss weights - was made on validation folds, recorded in
docs/FINDINGS.md.

WHAT IS AVERAGED OVER WHAT. Norman's reported numbers are means over FIVE folds and
combosciplex has a single published split; within each fold the seeds are averaged first,
then the folds, so a fold with more seeds cannot outweigh another. Both are printed, since
the spread across folds is larger than the spread across seeds and the paper has to say so.

THE RIDGE ROW IS NOT COPIED FROM ANYWHERE. It is this same model at s = 0, which by
construction is exactly m_control + sum_a w_a. Printing it beside the published 1.669 /
1.416 / 2.251 / 1.858 is therefore a self-check on the whole pipeline - the gene set, the
split, the transport, the gate - and a disagreement means something is wrong here, not
that ridge changed.

THE CALIBRATION COMES FROM VALIDATION AND IS PASSED IN. `--calibration` takes c,p per
dataset:method for s = clip(c ||r||^-p, 0, 4), fitted by `dev_rule.py --fit-all` on the
validation runs. The one-exponent family is used rather than power2 because it is the one
src/models/model.py implements and the two agreed to within 0.002 on every family.

A caveat that belongs in the paper, not only here: those coefficients were fitted on
models trained on the VALIDATION folds, and ||r|| could sit at a different scale for a
model trained on the full training set. The s = 0 and s = 1 rows are printed beside the
calibrated one so a reader can see the whole interval the calibration moves within rather
than taking its transfer on trust.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dev_score import find_run

# docs/DESIGN.md section 0.1, measured with scripts/baseline_l2.py on the reported
# protocol. Printed for comparison only; nothing here is derived from them.
PUBLISHED = {
    "Table 1 Norman additive": {"ridge": 1.669, "scDFM": 1.704, "scPKFM": 1.957},
    "Table 2 Single": {"ridge": 1.416, "scDFM": 1.619, "scPKFM": 1.754},
    "Table 2 Double": {"ridge": 2.251, "scDFM": 2.031, "scPKFM": 2.375},
    "Table 3 ComboSciPlex": {"ridge": 1.858, "scDFM": 1.657, "scPKFM": 2.138},
}
# (dataset, method) -> the reported blocks it produces, as (block name, which half)
BLOCKS = {
    ("norman", "additive"): [("Table 1 Norman additive", "double")],
    ("norman", "combinations"): [("Table 2 Double", "double"),
                                 ("Table 2 Single", "single")],
    ("combosciplex", "list"): [("Table 3 ComboSciPlex", "table3")],
}


def score_run(run_dir: str, device: str, n_cells: int,
              calibration: tuple[float, float] | None,
              gate: str = "soft") -> dict[str, float]:
    """L2 on the TEST conditions, on the reported protocol, split into blocks.

    The same protocol scripts/dev_score.py uses on validation: the scanpy HVG of the test
    subset plus control, the soft gate, 1,024 transported cells, one rng in condition
    order. Only the conditions differ, and that is the whole point of this file.
    """
    from src.eval.diagnostics import load_run, measure_transport
    from src.eval.conditions import condition_groups, scdfm_eval_genes

    overrides = None
    if calibration is not None:
        overrides = {"residual_coefficient": calibration[0],
                     "residual_power": calibration[1]}
    config, data, stats, fold, model = load_run(run_dir, device, gate,
                                                eval_overrides=overrides)
    rng = np.random.default_rng(config["eval"]["seed"])
    groups = condition_groups(data, stats, fold, config["split"]["method"])
    doubles, singles = groups["test doubles"], groups["test singles"]
    genes = scdfm_eval_genes(data, fold, 1000)
    rows = measure_transport(model, data, stats, doubles + singles, config, rng,
                             device, n_cells, genes=genes)
    l2 = {r["condition"]: r["l2"] for r in rows}
    out = {"double": float(np.mean([l2[c] for c in doubles])) if doubles else float("nan"),
           "single": float(np.mean([l2[c] for c in singles])) if singles else float("nan"),
           "n_double": len(doubles), "n_single": len(singles),
           "scale": float(np.mean([r["residual_scale"] for r in rows]))}
    # Table 3's published L2 is the mean over all seven test conditions together, five
    # combinations and two singles, not a weighted average of two block means.
    every = doubles + singles
    out["table3"] = float(np.mean([l2[c] for c in every])) if every else float("nan")
    return out


def parse_calibration(items: list[str]) -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        dataset, _, method = key.partition(":")
        c, _, p = value.partition(",")
        out[(dataset, method)] = (float(c), float(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--calibration", nargs="*", default=[],
                        help="dataset:method=c,p from dev_rule.py --fit-all")
    parser.add_argument("--score-the-test", action="store_true")
    parser.add_argument("--gate", default="soft", choices=["soft", "hard", "sample"],
                        help="how the hurdle head realises the binary event. `soft` "
                             "returns the population MEAN exactly, which is what an L2 "
                             "between means wants. `sample` draws a realised cell and "
                             "clamps it at zero, which is what a DISTRIBUTION metric "
                             "wants - cell-eval 0.8.1 rejects a soft export outright, "
                             "`Invalid scale: min value -2.63 is negative`, because a "
                             "mean vector is not a cell. The clamp makes `sample` "
                             "slightly biased upward, so the two gates are compared here "
                             "rather than assumed interchangeable.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if not args.score_the_test:
        raise SystemExit(
            "This scores the REPORTED TEST SET. Pass --score-the-test to mean it.\n"
            "Everything selected so far was chosen on validation folds; scoring the test "
            "is meant to happen once, and a second time cannot be undone by deleting the "
            "output. scPKFM scored its test set thirteen times while selecting and "
            "afterwards none of its numbers could be read.")

    calibration = parse_calibration(args.calibration)
    with open(args.manifest, encoding="utf-8") as handle:
        jobs = json.load(handle)["jobs"]

    missing = [j["tag"] for j in jobs if find_run(j["tag"]) is None]
    if missing:
        raise SystemExit(f"{len(missing)} runs have not finished: "
                         f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")

    records = []
    for job in jobs:
        key = (job["dataset"], job["method"])
        run_dir = find_run(job["tag"])
        for label, setting in (("s=0", (0.0, 0.0)),
                               ("s=1", None),
                               ("calibrated", calibration.get(key))):
            if label == "calibrated" and setting is None:
                continue
            scored = score_run(run_dir, args.device, args.n_cells, setting, args.gate)
            records.append({"tag": job["tag"], "dataset": job["dataset"],
                            "method": job["method"], "fold": job["fold"],
                            "seed": job["seed"], "setting": label,
                            "gate": args.gate, **scored})
        print(f"  {job['tag']:36s} scored")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\n-> {args.csv}")

    # Seeds first, then folds: a fold with more seeds must not outweigh another.
    print(f"\n{'block':26s} {'setting':11s} {'ours':>8s} {'fold SD':>8s} "
          f"{'ridge':>7s} {'scDFM':>7s} {'scPKFM':>7s}   folds")
    print("-" * 94)
    for (dataset, method), blocks in BLOCKS.items():
        mine = [r for r in records if r["dataset"] == dataset and r["method"] == method]
        if not mine:
            continue
        for block, half in blocks:
            reference = PUBLISHED[block]
            for setting in ("s=0", "s=1", "calibrated"):
                chosen = [r for r in mine if r["setting"] == setting]
                if not chosen:
                    continue
                by_fold = collections.defaultdict(list)
                for row in chosen:
                    if np.isfinite(row[half]):
                        by_fold[row["fold"]].append(row[half])
                if not by_fold:
                    continue
                per_fold = np.array([np.mean(v) for v in by_fold.values()])
                sd = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else float("nan")
                note = "  <- ridge self-check" if setting == "s=0" else ""
                print(f"{block:26s} {setting:11s} {per_fold.mean():8.4f} {sd:8.4f} "
                      f"{reference['ridge']:7.3f} {reference['scDFM']:7.3f} "
                      f"{reference['scPKFM']:7.3f}   {len(per_fold)}{note}")
            print()

    print("s=0 is the model with its residual switched off, which by construction is "
          "exactly\nridge_additive. It should reproduce the published ridge column; if it "
          "does not, the\npipeline disagrees with the baseline measurement and no other "
          "row here can be read.")


if __name__ == "__main__":
    main()
