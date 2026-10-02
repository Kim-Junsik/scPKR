"""Does a biological prior predict WHERE interactions happen? Answer before building.

    python scripts/prior_screen.py --h5ad data/norman/norman.h5ad \
        --string data/networks/string_links.txt.gz \
        --string-info data/networks/string_info.txt.gz \
        --collectri data/networks/collectri.tsv

THE LESSON THIS IMPLEMENTS. The previous model spent weeks on a KEGG observable space and
only then asked whether the biology was load-bearing; it was not - KEGG tied a random
dictionary of the same width. The order was backwards. A prior's signal can be measured
with NO MODEL AT ALL, in an hour, and if it is absent no architecture recovers it.

NO MODEL AND NO FITTING. The additive prediction comes from the MEASURED single
perturbations rather than from a ridge, so the residual is

    r_S = m_AB - m_A - m_B + m_ctrl

and nothing here has a parameter to tune or a split to leak across.

THE NORMALISATION IS THE WHOLE TEST. ||r_S|| on its own correlates with almost anything
that tracks effect size, because a pair of strong perturbations has a large residual for
trivial reasons - measured on Norman, STRING scored +0.273 and regulon overlap +0.302
against the raw residual. Dividing by ||w_A + w_B|| removes that, and the two collapsed to
-0.006 and -0.131. A screen that reports the raw correlation will approve any prior.

WHAT IT FOUND ON NORMAN, recorded so the next screen has a reference point: neither
physical interaction nor regulon overlap predicts relative non-additivity, while
log ||w_A + w_B|| reaches -0.499 and cos(residual, additive) is -0.428 with 99 of 125 pairs
negative. The dominant non-additivity there is SATURATION - the pair moves less than the
sum of its parts - which is a property of how far the additive component pushed, not of
which genes are related. On such a benchmark a relational prior has little left to explain.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip

import numpy as np


def perturbed(condition: str, control_tokens=("ctrl", "control")) -> list[str]:
    return [g for g in condition.split("+") if g not in control_tokens]


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 6:
        return float("nan"), int(keep.sum())
    rx = np.argsort(np.argsort(x[keep])).astype(float)
    ry = np.argsort(np.argsort(y[keep])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry))), int(keep.sum())


def condition_means(path: str, min_cells: int):
    import anndata as ad

    data = ad.read_h5ad(path)
    labels = data.obs["condition"].astype(str).to_numpy()
    x = data.X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    rows = collections.defaultdict(list)
    for i, label in enumerate(labels):
        rows[label].append(i)
    means = {c: x[np.asarray(idx)].mean(axis=0)
             for c, idx in rows.items() if len(idx) >= min_cells}
    control = next(c for c in means if not perturbed(c))
    return means, control, set(map(str, data.var_names))


def string_scores(links: str, info: str) -> dict:
    name = {}
    with gzip.open(info, "rt") as handle:
        next(handle)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            name[fields[0]] = fields[1]
    out = {}
    with gzip.open(links, "rt") as handle:
        next(handle)
        for line in handle:
            left, right, score = line.split()
            a, b = name.get(left), name.get(right)
            if a and b:
                out[(a, b)] = int(score)
    return out


def regulons(path: str, measured: set) -> dict:
    out = collections.defaultdict(set)
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source, target = row.get("source_genesymbol"), row.get("target_genesymbol")
            if source and target and target in measured:
                out[source].add(target)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--string", default=None)
    parser.add_argument("--string-info", default=None)
    parser.add_argument("--collectri", default=None)
    parser.add_argument("--min-cells", type=int, default=20)
    args = parser.parse_args()

    means, control, measured = condition_means(args.h5ad, args.min_cells)
    singles = {perturbed(c)[0]: c for c in means if len(perturbed(c)) == 1}
    string = (string_scores(args.string, args.string_info)
              if args.string and args.string_info else {})
    regulon = regulons(args.collectri, measured) if args.collectri else {}
    print(f"{len(means)} conditions with >= {args.min_cells} cells, "
          f"{len(singles)} measured singles")

    records = []
    for condition in means:
        pair = perturbed(condition)
        if len(pair) != 2 or not all(g in singles for g in pair):
            continue
        first = means[singles[pair[0]]] - means[control]
        second = means[singles[pair[1]]] - means[control]
        residual = means[condition] - means[control] - first - second
        additive = float(np.linalg.norm(first + second))
        if additive <= 0:
            continue
        union = regulon.get(pair[0], set()) | regulon.get(pair[1], set())
        shared = regulon.get(pair[0], set()) & regulon.get(pair[1], set())
        records.append({
            "resid": float(np.linalg.norm(residual)),
            "rel": float(np.linalg.norm(residual)) / additive,
            "string": float(max(string.get((pair[0], pair[1]), 0),
                                string.get((pair[1], pair[0]), 0)))
            if string else float("nan"),
            "jaccard": (len(shared) / len(union)) if union else float("nan"),
            "log_additive": float(np.log(additive)),
            "cos_singles": float(first @ second / (np.linalg.norm(first)
                                                   * np.linalg.norm(second))),
            "cos_resid_additive": float(residual @ (first + second)
                                        / (np.linalg.norm(residual) * additive)),
        })
    if not records:
        raise SystemExit("no double has both of its singles measured")

    relative = np.array([r["rel"] for r in records])
    raw = np.array([r["resid"] for r in records])
    print(f"\n{len(records)} doubles with both singles measured")
    print(f"relative non-additivity ||r|| / ||w_A + w_B||: "
          f"median {np.median(relative):.3f}, "
          f"range {relative.min():.3f}-{relative.max():.3f}")

    print(f"\n{'predictor':34s} {'vs ||r||':>12s} {'vs RELATIVE':>14s}   n")
    print("  " + "-" * 64)
    for key, label, is_prior in (("string", "STRING physical score", True),
                                 ("jaccard", "regulon overlap (jaccard)", True),
                                 ("log_additive", "log ||w_A + w_B||", False),
                                 ("cos_singles", "cos(w_A, w_B)", False)):
        values = np.array([r[key] for r in records], dtype=float)
        if not np.isfinite(values).any():
            continue
        a, _ = spearman(values, raw)
        b, n = spearman(values, relative)
        print(f"{label:34s} {a:+12.3f} {b:+14.3f}  {n:4d}"
              f"{'  <- prior' if is_prior else ''}")

    direction = np.array([r["cos_resid_additive"] for r in records])
    print(f"\ncos(residual, additive displacement): median {np.median(direction):+.3f}, "
          f"{int((direction < 0).sum())}/{len(direction)} negative")
    print("A median well below zero means the dominant non-additivity is SATURATION: the "
          "pair moves\nless than the sum of its parts. A scalar shrinkage captures that, "
          "and no relational prior\nis needed to explain it.")
    print("\nRead the RELATIVE column. The raw column tracks effect size and will approve "
          "any prior.")


if __name__ == "__main__":
    main()
