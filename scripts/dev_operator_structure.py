"""Did the learned operators recover pharmacology the model was never shown?

    python scripts/dev_operator_structure.py results/dev/w10/manifest.json \
        --arms keggwhite base --device cuda

THE TEST. ComboSciPlex annotates every drug with a mechanism class - HDAC inhibitor, EGFR
inhibitor, Sirtuin inhibitor, and three others, 17 drugs over 6 classes with the three
largest holding 4 each. THAT ANNOTATION NEVER ENTERS THE MODEL. Nothing in training,
in the ridge fit, in the observables or in the losses reads it. So if the learned
generators A_a cluster by mechanism class, the model recovered the pharmacology from
expression alone, and the clustering is a prediction that could have failed.

It is a quantitative test with a null, not a figure: the statistic is
(mean within-class similarity) - (mean between-class similarity), and its p-value comes
from permuting the class labels.

WHY THIS CANNOT BE ASKED OF PCA OBSERVABLES. A_a lives in the coordinates Phi defines. With
KEGG those coordinates ARE pathways, so A_a is a pathway-by-pathway operator and "which
pathway does this drug's effect flow out of" is a question with an answer. With PCA they are
variance directions of the training cells and the same question has none. This is the one
claim about the biological dictionary that does not reduce to a difference in L2.

TWO CORRECTIONS THAT THE TEST WOULD BE WORTHLESS WITHOUT.

Whitening is undone first. ZCA makes each coordinate a mixture of pathways, so the operator
is read back in the pathway basis as ZCA^-1 A ZCA, which is exact - the whitener is square
and invertible by construction.

The SHARED component is removed. A_a = U diag(c_a) V + P_a Q_a, and U, V are common to every
perturbation, so all operators are similar before any data is seen and a raw similarity
would report that shared structure as a discovery. The statistic is therefore computed on
A_a minus the mean operator across drugs. The raw version is printed beside it so the gap
between the two is visible rather than hidden.

Runs are not pooled. Two runs fit different bases and their operators are not comparable
entry by entry, so the statistic is computed per run and summarised across runs.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dev_score import find_run

PERMUTATIONS = 20000


def mechanism_classes(raw_h5ad: str) -> dict[str, str]:
    """drug -> mechanism class, from the dataset's own annotation."""
    import anndata as ad

    obs = ad.read_h5ad(raw_h5ad, backed="r").obs
    out: dict[str, set] = {}
    for columns in (("Drug1", "pathway1"), ("Drug2", "pathway2")):
        if columns[0] not in obs.columns:
            continue
        for drug, mechanism in zip(obs[columns[0]], obs[columns[1]]):
            drug = str(drug)
            if drug != "control":
                out.setdefault(drug, set()).add(str(mechanism))
    classes = {}
    for drug, found in out.items():
        if len(found) != 1:
            raise ValueError(f"{drug} has several mechanism classes: {sorted(found)}")
        classes[drug] = next(iter(found))
    return classes


@torch.no_grad()
def pathway_operators(run_dir: str, device: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """Each perturbation's generator in the PATHWAY basis, and the observable names."""
    from src.eval.diagnostics import load_run

    _, data, _, _, model = load_run(run_dir, device, "soft")
    whitener = model.observables.whitener.detach().cpu().numpy().astype(np.float64)
    # Exact, not an approximation: ZCA is a symmetric positive-definite square matrix, so
    # the inverse exists. Identity when the run was trained without whitening.
    inverse = np.linalg.inv(whitener)

    out = {}
    for name in data.perturbations:
        matrix = model.operators.matrix(data.pert_index[name])
        matrix = matrix.detach().cpu().numpy().astype(np.float64)
        out[name] = inverse @ matrix @ whitener
    return out, list(model.observables.names)


def similarity(operators: dict[str, np.ndarray], centre: bool) -> tuple[list[str], np.ndarray]:
    """Frobenius cosine between every pair of operators, optionally after centring."""
    names = sorted(operators)
    stack = np.stack([operators[n].ravel() for n in names])
    if centre:
        stack = stack - stack.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    unit = stack / norms
    return names, unit @ unit.T


def gap(matrix: np.ndarray, labels: np.ndarray) -> float:
    """mean(within-class) - mean(between-class), off-diagonal only."""
    n = len(labels)
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(n, dtype=bool)
    within, between = matrix[same & off], matrix[~same & off]
    if within.size == 0 or between.size == 0:
        return float("nan")
    return float(within.mean() - between.mean())


def permutation_p(matrix: np.ndarray, labels: np.ndarray, rng) -> tuple[float, float, float]:
    """The gap, its null mean, and a one-sided p-value from permuting the labels.

    The labels are permuted rather than the matrix, which keeps the class sizes - a class
    of 4 and a class of 1 contribute very differently and resampling that away would test
    a different null.
    """
    observed = gap(matrix, labels)
    null = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        null[i] = gap(matrix, rng.permutation(labels))
    return observed, float(null.mean()), float((null >= observed).mean())


def top_pathways(operators: dict[str, np.ndarray], drugs: list[str], names: list[str],
                 k: int = 6) -> list[tuple[str, float]]:
    """Observables this group's operators move the most, by row mass of |A|.

    A row of A is how strongly one coordinate is DRIVEN by the others, so a large row mass
    is an observable the perturbation pushes on.
    """
    total = np.zeros(len(names))
    for drug in drugs:
        total += np.abs(operators[drug]).sum(axis=1)
    total /= max(len(drugs), 1)
    order = np.argsort(-total)[:k]
    return [(names[i], float(total[i])) for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-h5ad", default="data/combosciplex/combosciplex.h5ad")
    parser.add_argument("--pathways", type=int, default=6,
                        help="how many top observables to print per mechanism class")
    args = parser.parse_args()

    classes = mechanism_classes(args.raw_h5ad)
    by_class = collections.defaultdict(list)
    for drug, mechanism in sorted(classes.items()):
        by_class[mechanism].append(drug)
    print(f"{len(classes)} drugs over {len(by_class)} mechanism classes "
          f"(sizes {sorted((len(v) for v in by_class.values()), reverse=True)})")
    print("THE MODEL NEVER SEES THESE LABELS.\n")

    jobs = []
    for path in args.manifests:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for job in manifest["jobs"]:
            if job.get("kind") != "arm" or (args.arms and job["arm"] not in args.arms):
                continue
            if job["dataset"] != "combosciplex":
                continue
            run_dir = find_run(job["tag"])
            if run_dir is not None:
                jobs.append((job, run_dir))
    if not jobs:
        raise SystemExit("no finished combosciplex runs matched")

    rng = np.random.default_rng(0)
    results = collections.defaultdict(list)
    pathway_report: dict[str, dict[str, float]] = {}
    for job, run_dir in jobs:
        operators, names = pathway_operators(run_dir, args.device)
        known = [d for d in operators if d in classes]
        subset = {d: operators[d] for d in known}
        labels = np.array([classes[d] for d in sorted(subset)])
        for centre, label in ((False, "raw"), (True, "shared removed")):
            order, matrix = similarity(subset, centre)
            observed, null, p = permutation_p(matrix, labels, rng)
            results[(job["arm"], label)].append((observed, null, p))
        if job["arm"] not in pathway_report:
            pathway_report[job["arm"]] = {
                mechanism: top_pathways(subset, [d for d in drugs if d in subset],
                                        names, args.pathways)
                for mechanism, drugs in by_class.items() if len(drugs) >= 2}
        print(f"  {job['tag']:34s} {len(known)} drugs")

    print(f"\n=== clustering by mechanism class, {PERMUTATIONS:,} label permutations ===")
    print(f"{'arm':12s} {'statistic':16s} {'gap':>8s} {'null':>8s} {'p':>8s}  runs")
    for (arm, label), values in sorted(results.items()):
        gaps = np.array([v[0] for v in values])
        nulls = np.array([v[1] for v in values])
        ps = np.array([v[2] for v in values])
        print(f"{arm:12s} {label:16s} {gaps.mean():+8.4f} {nulls.mean():+8.4f} "
              f"{np.median(ps):8.4f}  n={len(values)}, p<0.05 in "
              f"{int((ps < 0.05).sum())}/{len(ps)}")
    print("\n`shared removed` is the one to read: U and V are common to every perturbation,"
          "\nso the raw statistic is positive before any data is seen.")

    for arm, report in pathway_report.items():
        print(f"\n=== {arm}: observables each mechanism class drives hardest ===")
        for mechanism, rows in sorted(report.items()):
            print(f"  {mechanism}")
            for name, mass in rows:
                print(f"      {mass:8.4f}  {name}")


if __name__ == "__main__":
    main()
