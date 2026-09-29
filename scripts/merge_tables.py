"""Merge paper-table CSVs from several machines into one five-fold table.

    python scripts/merge_tables.py results/fin_combinations_double_A.csv \
        results/fin_combinations_double_B.csv \
        --csv results/fin_combinations_double_5fold.csv

The folds of one experiment often finish on different hosts. paper_table.py needs
the checkpoint and the data cache to compute L2, so it has to run where each run
lives and write a CSV there. This joins those CSVs and needs neither - only the
standard library - so it runs on any machine the CSVs are copied to.

It refuses rather than averaging when the inputs cannot describe one experiment:
  - the same fold appears twice (a rerun copied next to the original)
  - more than one run prefix appears (e.g. fin_ next to fin_combinations_)
A missing fold is allowed but flagged, because a mean over three of five folds is
not the number the cited tables report.

This also stands in for paper_table.py --mean, which is declared but was never
implemented: that flag parses and does nothing.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys

COLUMNS = ["L2", "MSE", "MAE", "DE_rho", "Pearson_d", "DS"]
LOWER_IS_BETTER = {"L2", "MSE", "MAE"}
RUN_NAME = re.compile(r"^(?P<prefix>.+)_f(?P<fold>\d+)$")


def read(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                name = record["run"]
                match = RUN_NAME.match(name)
                if not match:
                    sys.exit(f"{path}: run name {name!r} does not end in _f<fold>")
                row = {"run": name, "prefix": match["prefix"],
                       "fold": int(match["fold"]), "source": path}
                for column in COLUMNS:
                    # paper_table.py writes a missing cell-eval score as an empty
                    # cell and a missing L2 as 'nan'; both become nan here.
                    try:
                        row[column] = float((record.get(column) or "").strip())
                    except ValueError:
                        row[column] = math.nan
                rows.append(row)
    return rows


def check(rows: list[dict], n_folds: int) -> list[int]:
    prefixes = sorted({r["prefix"] for r in rows})
    if len(prefixes) > 1:
        sys.exit("refusing to average different experiments together:\n  "
                 + "\n  ".join(prefixes))

    by_fold: dict[int, list[dict]] = {}
    for r in rows:
        by_fold.setdefault(r["fold"], []).append(r)
    duplicated = {f: rs for f, rs in by_fold.items() if len(rs) > 1}
    if duplicated:
        lines = [f"fold {f}: " + ", ".join(r["source"] for r in rs)
                 for f, rs in sorted(duplicated.items())]
        sys.exit("the same fold appears more than once - keep one copy:\n  "
                 + "\n  ".join(lines))

    return sorted(set(range(n_folds)) - set(by_fold))


def summarise(rows: list[dict]) -> dict[str, tuple[float, float, int]]:
    out = {}
    for column in COLUMNS:
        values = [r[column] for r in rows if math.isfinite(r[column])]
        mean = statistics.fmean(values) if values else math.nan
        # Sample standard deviation across folds; undefined for a single fold.
        std = statistics.stdev(values) if len(values) > 1 else math.nan
        out[column] = (mean, std, len(values))
    return out


def cell(value: float, width: int = 11) -> str:
    return f"{value:{width}.4f}" if math.isfinite(value) else f"{'-':>{width}s}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", help="CSVs written by paper_table.py --csv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--csv", default=None, help="also write the merged table here")
    args = parser.parse_args()

    rows = sorted(read(args.csvs), key=lambda r: r["fold"])
    if not rows:
        sys.exit("no rows found in the given CSVs")
    missing = check(rows, args.n_folds)
    stats = summarise(rows)

    header = f"{'run':38s} " + " ".join(f"{c:>11s}" for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run'][:37]:38s} " + " ".join(cell(r[c]) for c in COLUMNS))
    print("-" * len(header))
    print(f"{'mean':38s} " + " ".join(cell(stats[c][0]) for c in COLUMNS))
    print(f"{'std':38s} " + " ".join(cell(stats[c][1]) for c in COLUMNS))

    print("\nmean +- std (paste into the table):")
    print("  " + "  ".join(
        f"{c} {stats[c][0]:.4f} +- {stats[c][1]:.4f}" if math.isfinite(stats[c][1])
        else f"{c} {stats[c][0]:.4f}" for c in COLUMNS))
    print("lower is better: L2, MSE, MAE.  higher is better: DE_rho, Pearson_d, DS.")

    folds = [r["fold"] for r in rows]
    print(f"\nfolds present: {folds}")
    if missing:
        print(f"[warn] missing folds {missing}: this mean covers {len(rows)} of "
              f"{args.n_folds} and is not comparable to a {args.n_folds}-fold mean")
    incomplete = [c for c in COLUMNS if stats[c][2] < len(rows)]
    for c in incomplete:
        absent = [r["run"] for r in rows if not math.isfinite(r[c])]
        print(f"[warn] {c} is missing for {absent} - the mean above uses "
              f"{stats[c][2]} of {len(rows)} runs")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run", "fold"] + COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in ["run", "fold"] + COLUMNS})
            writer.writerow({"run": "mean", **{c: stats[c][0] for c in COLUMNS}})
            writer.writerow({"run": "std", **{c: stats[c][1] for c in COLUMNS}})
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
