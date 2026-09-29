"""Additive baselines that a model has to beat.

All of them predict a MEAN response; cell-level predictions for the energy
distance are formed by applying the predicted shift to sampled control cells,
which is also how the external cell-eval protocol builds its predictions.

Two families, and the difference decides what is computable on which split:

  delta-additive / per-gene-scaled
      read delta_A and delta_B off the SINGLE conditions 'A+ctrl', 'B+ctrl'.
      On the combinations split those singles are held out, so these are not
      computable and the harness must skip them rather than quietly leak.

  ridge-additive
      regresses delta of every training condition on one-hot perturbation
      indicators, so a per-perturbation effect w_A is identifiable from any
      training condition containing A - including doubles. It therefore still
      works when the single 'A+ctrl' itself is held out.
"""

from __future__ import annotations

import numpy as np

from ..data.conventions import DEFAULT, ConditionNaming

CONTROL = DEFAULT.control  # legacy alias; prefer stats.naming.control


def condition_genes(condition: str, naming: ConditionNaming = DEFAULT) -> list[str]:
    return naming.genes(condition)


class ConditionMeans:
    """Per-condition mean, variance and cell count over a fixed gene space."""

    def __init__(self, x: np.ndarray, conditions: np.ndarray,
                 naming: ConditionNaming = DEFAULT):
        self.naming = naming
        self.mean: dict[str, np.ndarray] = {}
        self.var: dict[str, np.ndarray] = {}
        self.n: dict[str, int] = {}
        for condition in np.unique(conditions):
            block = x[conditions == condition]
            self.mean[condition] = block.mean(axis=0)
            self.var[condition] = block.var(axis=0)
            self.n[condition] = int(block.shape[0])
        control = next((c for c in self.mean if naming.is_control(c)), None)
        if control is None:
            raise ValueError(f"no control condition found; expected {naming.control!r}")
        self.control_condition = control
        self.control = self.mean[control]

    def delta(self, condition: str) -> np.ndarray:
        return self.mean[condition] - self.control

    def has(self, condition: str) -> bool:
        return condition in self.mean

    def single_of(self, gene: str) -> str:
        """The single condition for `gene` as this dataset actually spells it."""
        for form in self.naming.single_forms(gene):
            if form in self.mean:
                return form
        return self.naming.single(gene)


def _single(gene: str, naming: ConditionNaming = DEFAULT) -> str:
    return naming.single(gene)


def fit_per_gene_scale(stats: ConditionMeans, train_doubles: list[str],
                       available: set[str]) -> np.ndarray:
    """One scalar per gene: how much of the additive sum actually materialises.

    s_g = argmin sum_d (delta_AB,g - s_g * (delta_A,g + delta_B,g))^2
    """
    numerator = None
    denominator = None
    for double in train_doubles:
        a, b = stats.naming.genes(double)
        sa, sb = stats.single_of(a), stats.single_of(b)
        if not all(c in available and stats.has(c) for c in (double, sa, sb)):
            continue
        additive = stats.delta(sa) + stats.delta(sb)
        target = stats.delta(double)
        numerator = additive * target if numerator is None else numerator + additive * target
        denominator = additive ** 2 if denominator is None else denominator + additive ** 2
    if numerator is None:
        return None
    return numerator / np.maximum(denominator, 1e-12)


def fit_ridge_additive(stats: ConditionMeans, train_conditions: list[str],
                       perturbations: list[str], alpha: float = 1.0,
                       weight_by_cells: bool = False) -> dict[str, np.ndarray]:
    """Per-perturbation effect vectors from a one-hot ridge over training conditions.

    Solves  min_W ||X W - Y||^2 + alpha ||W||^2  where X is n_conditions x n_perturbations
    (one-hot) and Y holds the per-condition deltas. The additive prediction for a
    double is then w_A + w_B.
    """
    index = {p: i for i, p in enumerate(perturbations)}
    rows, targets, weights = [], [], []
    for condition in train_conditions:
        if not stats.has(condition):
            continue
        genes = condition_genes(condition, stats.naming)
        if any(g not in index for g in genes):
            continue
        row = np.zeros(len(perturbations), dtype=np.float64)
        for g in genes:
            row[index[g]] = 1.0
        rows.append(row)
        targets.append(stats.delta(condition))
        weights.append(stats.n[condition])

    x = np.asarray(rows)
    y = np.asarray(targets, dtype=np.float64)
    if weight_by_cells:
        sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
        x, y = x * sqrt_w, y * sqrt_w

    gram = x.T @ x + alpha * np.eye(x.shape[1])
    coefficients = np.linalg.solve(gram, x.T @ y)  # n_perturbations x n_genes

    covered = set(np.asarray(perturbations)[np.asarray(rows).sum(axis=0) > 0].tolist())
    return {"w": coefficients, "index": index, "covered": covered}


def pseudo_single_shifts(stats: ConditionMeans, train_conditions: list[str],
                         alpha: float = 1.0) -> dict[str, np.ndarray]:
    """condition -> gene-space shift, for the singles that TRAINING DOES NOT HAVE.

    A drug whose only training conditions are combinations has an unidentified
    operator: the data fix u_A + u_B + rho and not how it splits, so u_a can be
    anything. On combosciplex that is 13 of 17 drugs, and it includes both drugs
    behind the two test singles - the block carrying 76 % of the gap to the Table 3
    target. Measured with scripts/rho_off_probe.py: held-out Dasatinib alone
    pointed OPPOSITE its true shift (cosine -0.26, -0.31).

    The ridge additive fit pools a drug's effect over every training condition it
    appears in, so w_a exists for every covered drug whether or not it was ever
    given alone. loop.latent_endpoint_targets encodes these shifts into latent
    targets for Phi_a(z_ctrl), which is the term those drugs otherwise never get.

    READS TRAINING CONDITIONS ONLY, the same rows anchor_deltas(kind="ridge") reads.
    Returns only drugs with NO training single: one that has a single is already
    supervised directly, and a weaker estimate must not compete with that.

    The canonical spelling is used for the key; ConditionSampler and the endpoint
    loss both look conditions up by name, and these names are not in the data.
    """
    naming = stats.naming
    perturbations = sorted({g for c in train_conditions if not naming.is_control(c)
                            for g in condition_genes(c, naming)})
    if not perturbations:
        return {}
    trained_alone = {condition_genes(c, naming)[0] for c in train_conditions
                     if naming.is_single(c) and stats.has(c)}
    fit = fit_ridge_additive(stats, train_conditions, perturbations, alpha=alpha)
    return {naming.single(gene): fit["w"][fit["index"][gene]]
            for gene in perturbations
            if gene in fit["covered"] and gene not in trained_alone}


def anchor_deltas(kind: str, stats: ConditionMeans, train_conditions: list[str],
                  conditions: list[str], alpha: float = 1.0) -> dict[str, np.ndarray]:
    """condition -> the GENE-SPACE shift applied to a control cell before transport.

    Only combinations get an entry. A single perturbation is what identifies u_a in
    the first place, so shifting its source would leave v(z,t,{a}) = u_a supervising
    nothing; `batch` and `predict_cells` both fall back to an unshifted control for
    any condition missing from this table.

    Everything here is fitted or read from `train_conditions`. `stats` holds the
    mean of EVERY condition including the evaluated doubles - it has to, the metrics
    are computed against them - so the discipline is which keys are read, and the
    only ones read below are the singles and the ridge fit's own training rows.

    Returns {} for kind == "none", which is what keeps the unanchored model exactly
    unchanged rather than shifted by zero.
    """
    if kind == "none":
        return {}
    if kind not in ("additive", "ridge"):
        raise ValueError(f"unknown anchor {kind!r}")

    naming = stats.naming
    trainable = set(train_conditions)
    if kind == "ridge":
        perturbations = sorted({g for c in train_conditions if not naming.is_control(c)
                                for g in condition_genes(c, naming)})
        fit = fit_ridge_additive(stats, train_conditions, perturbations, alpha=alpha)

    table: dict[str, np.ndarray] = {}
    for condition in conditions:
        if not naming.is_double(condition):
            continue
        a_gene, b_gene = naming.genes(condition)
        if kind == "additive":
            single_a, single_b = stats.single_of(a_gene), stats.single_of(b_gene)
            # A single that is not trainable would make the shift depend on a
            # held-out condition. In the additive split this never fires; under
            # combinations it fires for every evaluated double, which is the
            # mechanical reason anchoring is an additive-split-only option.
            if not all(c in trainable and stats.has(c) for c in (single_a, single_b)):
                continue
            table[condition] = stats.delta(single_a) + stats.delta(single_b)
        else:
            if a_gene not in fit["covered"] or b_gene not in fit["covered"]:
                continue
            table[condition] = (fit["w"][fit["index"][a_gene]]
                                + fit["w"][fit["index"][b_gene]])
    return table


def fit_pairwise_ridge(stats: ConditionMeans, train_doubles: list[str],
                       available: set[str], alpha: float = 1.0) -> np.ndarray:
    """Per-gene ridge on two SYMMETRIC features of the single deltas.

        delta_AB,g ~ a_g * (delta_A,g + delta_B,g) + b_g * (delta_A,g * delta_B,g)

    Both features are invariant under swapping A and B, which matters because
    which gene of a double is "A" is just alphabetical accident - fitting separate
    coefficients for delta_A and delta_B would not be identifiable.

    This is the strongest additive-family baseline: it contains per-gene-scaled as
    the b = 0 special case and additionally models multiplicative interaction, so
    it is the honest bar for any model claiming to capture non-additivity.
    """
    sums = np.zeros(5)  # placeholder shape, replaced on first double
    accumulated = None
    for double in train_doubles:
        a, b = stats.naming.genes(double)
        sa, sb = stats.single_of(a), stats.single_of(b)
        if not all(c in available and stats.has(c) for c in (double, sa, sb)):
            continue
        da, db = stats.delta(sa), stats.delta(sb)
        f1, f2 = da + db, da * db
        y = stats.delta(double)
        block = np.stack([f1 * f1, f1 * f2, f2 * f2, f1 * y, f2 * y])
        accumulated = block if accumulated is None else accumulated + block
    if accumulated is None:
        return None

    s11, s12, s22, t1, t2 = accumulated
    s11 = s11 + alpha
    s22 = s22 + alpha
    determinant = s11 * s22 - s12 ** 2
    safe = np.where(np.abs(determinant) < 1e-12, 1e-12, determinant)
    a_coefficient = (s22 * t1 - s12 * t2) / safe
    b_coefficient = (s11 * t2 - s12 * t1) / safe
    return np.stack([a_coefficient, b_coefficient])


def predict(name: str, double: str, stats: ConditionMeans,
            available: set[str],
            scale: np.ndarray | None = None,
            ridge: dict | None = None,
            pairwise: np.ndarray | None = None) -> np.ndarray | None:
    """Predicted MEAN for `double`, or None when the baseline is not computable.

    `available` is the set of conditions the PREDICTOR may read - the fold's
    training conditions. Checking that a condition merely exists in the data is
    not enough: on the combinations split the singles of every test double are
    held out, so reading them would leak exactly the information the split was
    built to withhold.
    """
    a, b = stats.naming.genes(double)
    single_a, single_b = stats.single_of(a), stats.single_of(b)
    readable = lambda c: c in available and stats.has(c)  # noqa: E731

    if name == "control":
        return stats.control.copy()

    if name in ("additive", "per_gene_scaled"):
        if not (readable(single_a) and readable(single_b)):
            return None  # singles held out -> not computable, do not fall back
        additive = stats.delta(single_a) + stats.delta(single_b)
        if name == "additive":
            return stats.control + additive
        return stats.control + scale * additive

    if name == "ridge_additive":
        if a not in ridge["covered"] or b not in ridge["covered"]:
            return None
        w = ridge["w"]
        return stats.control + w[ridge["index"][a]] + w[ridge["index"][b]]

    if name == "pairwise_ridge":
        if not (readable(single_a) and readable(single_b)):
            return None
        da, db = stats.delta(single_a), stats.delta(single_b)
        return stats.control + pairwise[0] * (da + db) + pairwise[1] * (da * db)

    raise ValueError(f"unknown baseline {name!r}")


def training_conditions(stats: ConditionMeans, fold: dict, method: str) -> list[str]:
    """Conditions a model is allowed to see for this fold.

    additive     : the fold's train doubles plus every single (singles are never
                   part of the additive split).
    combinations : the fold's own train list, which already excludes the
                   held-out singles.

    Whatever the method, nothing in fold["test"] is returned. Both rules above
    assume a test side with no singles, or with its singles recorded in
    held_out_singles; an explicit list (split.source=list) can hold out singles
    too, and "every single" would then hand them straight back to training. On
    Norman's additive and combinations folds the result is unchanged.
    """
    singles = [c for c in stats.mean if stats.naming.is_single(c)]
    held = (set(fold["test"]) | set(fold.get("held_out_singles", ()))
            | set(fold.get("excluded", ())))
    base = fold["train_doubles"] if method == "combinations" else fold["train"]
    # Order kept, duplicates dropped: a source whose train list already carries
    # singles (obs_column, list) would otherwise sample each of them twice.
    seen: set[str] = set()
    out: list[str] = []
    for condition in list(base) + singles:
        if condition not in held and condition not in seen:
            seen.add(condition)
            out.append(condition)
    return out
