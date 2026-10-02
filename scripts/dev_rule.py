"""What does a PREDICTED per-condition residual scale actually buy, cross-validated?

    python scripts/dev_rule.py results/dev/w6/reliability.csv

The three measurements that lead here. A global scale s buys -0.085 on combosciplex:sv and
-0.056 on norman:nval, while the per-condition oracle buys -0.250 and -0.114 - so a
constant captures a third to a half. Passing scDFM on Table 3 needs -10.8 % against ridge
where the constant gives -5.1 %, which is why the remainder matters. And the conditions the
residual HARMS are separable from the ones it helps: ||r|| reaches AUC 0.82 on combosciplex
and 0.94 on Norman, because a small residual's cosine is dominated by noise and its sign is
close to a coin toss.

WHAT THIS SCRIPT EXISTS TO DOUBT. High AUC does not imply a high payoff. The damage a
misaligned residual does is ||r|| sqrt(1 - 2 k rho + k^2), which for small k is barely worse
than not predicting at all - so the conditions a magnitude gate identifies are exactly the
conditions that were cheap to get wrong. A rule can therefore separate the failures almost
perfectly and still recover nothing. Only evaluating one settles it.

NOTHING IS RE-PREDICTED AND NOTHING IS RE-TRAINED. L2(s) = sqrt(ee + 2 s er + s^2 rr)
exactly, for any s, from three numbers per condition that scripts/dev_shrink.py computed
once. A rule is evaluated by putting its s into that expression.

THE CROSS-VALIDATION IS THE POINT, because every rule here has one or two free parameters
fitted against the same 36 or 45 rows whose answers are known, and a rule fitted and scored
on one set would report the oracle back. Held out BY CONDITION, never by row: the rows are
three seeds of the same condition, so splitting on rows would put a condition's own answer
in its training set. combosciplex is additionally held out by FOLD, which is the stricter
test - a fold's conditions are disjoint from the others' - and the gap between the two says
whether a rule generalises to new conditions or only to new seeds of known ones.
z is standardised from the training split only, for the same reason.

s = 0 IS THE ADDITIVE BASELINE EXACTLY, so every delta below is against a measured floor
rather than against another model's number.
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools

import numpy as np

TABLE3 = {"double": 5 / 7, "single": 2 / 7}
# The cap on s. It was set at 4 from combosciplex, where it bound only on the occasional
# near-zero residual, and that was a safety choice rather than a measured one. On
# norman:ncomb the fitted rule puts 45 of 60 conditions AT it, so the cap decides three
# quarters of that family and has become an unexamined hyperparameter doing real work.
# --scale-max exists to measure it rather than carry it into a one-time test scoring.
SCALE_MAX = 4.0
GRID = np.arange(0.0, SCALE_MAX + 0.001, 0.05)


def l2(rows: list[dict], scales: np.ndarray) -> np.ndarray:
    """sqrt(ee + 2 s er + s^2 rr) per row. Exact at any s, including untrained values."""
    ee = np.array([r["ee"] for r in rows])
    er = np.array([r["er"] for r in rows])
    rr = np.array([r["rr"] for r in rows])
    return np.sqrt(np.maximum(ee + 2.0 * scales * er + scales ** 2 * rr, 0.0))


def metric(doubles: list[dict], scales: np.ndarray, single: float | None) -> float:
    """The decision metric: Table 3's weighting when a single block was scored.

    The single block does not appear in `scales` because a single perturbation's residual
    is identically zero - B_S sums over pairs - so no scaling rule can move it. It enters
    as a constant, which is exactly the structural guarantee the design is built on.
    """
    double = float(np.mean(l2(doubles, scales)))
    if single is None or not np.isfinite(single):
        return double
    return TABLE3["double"] * double + TABLE3["single"] * single


# ------------------------------------------------------------------------ the rules
# Each fits on `train` and returns a callable giving s for any rows. Fitting minimises the
# mean L2 directly rather than the mean SQUARED L2, because the mean of norms is the
# reported metric and the two disagree: squares are decided by the largest-residual
# conditions, which are not the ones in question here.


def fit_const(train: list[dict], _name: str):
    best = min(GRID, key=lambda s: float(np.mean(l2(train, np.full(len(train), s)))))
    return lambda rows: np.full(len(rows), best), f"s={best:.2f}"


def fit_zero(_train: list[dict], _name: str):
    return lambda rows: np.zeros(len(rows)), "s=0"


def _standardise(train: list[dict], name: str):
    values = np.array([r[name] for r in train], dtype=float)
    centre, spread = float(np.mean(values)), float(np.std(values))
    spread = spread if spread > 1e-12 else 1.0
    return lambda rows: (np.array([r[name] for r in rows], dtype=float) - centre) / spread


def _fit_weighted(train: list[dict], name: str | None, slope: bool):
    """Weighted least squares of best_s, with or without a slope in z.

    The weights are ||r||^2 and they are not a choice: expanding ||e + s r||^2 gives
    rr (s - best_s)^2 up to a constant, so minimising the squared error in s weighted by rr
    IS minimising the squared L2. A condition whose residual is tiny therefore pulls on the
    fit only as much as it can affect the result, which is the behaviour a rule needs when
    the tiny-residual conditions are the unreliable ones.

    WHY THE SLOPELESS VERSION EXISTS AS ITS OWN RULE. `const` finds its scale by an argmin
    over a grid of the training mean L2, which on 69 rows is a noisier estimator than a
    weighted average - measured: a column of pure NOISE fitted this way beat `const` by
    0.0008 on all five test seeds, purely because its intercept was estimated more smoothly.
    Comparing linear(z) against `const` would therefore hand every predictor that same
    0.0008 whatever z contained. `wconst` is linear(z) with the slope forced to zero, so the
    two differ in the slope alone and the comparison isolates what z contributes.
    """
    target = np.array([r["best_s"] for r in train], dtype=float)
    weight = np.array([r["rr"] for r in train], dtype=float)
    keep = np.isfinite(target) & (weight > 0)
    z = _standardise(train, name) if slope else None
    if slope:
        zt = z(train)
        keep = keep & np.isfinite(zt)
    if keep.sum() < 3:
        return fit_const(train, name)
    columns = [np.ones(int(keep.sum()))] + ([zt[keep]] if slope else [])
    design = np.stack(columns, axis=1)
    sqrt_w = np.sqrt(weight[keep])[:, None]
    coefficients, *_ = np.linalg.lstsq(design * sqrt_w,
                                       target[keep] * sqrt_w[:, 0], rcond=None)
    a = float(coefficients[0])
    if not slope:
        return lambda rows: np.full(len(rows), min(max(a, 0.0), SCALE_MAX)), f"s={a:.2f}w"
    b = float(coefficients[1])
    return (lambda rows: np.clip(a + b * z(rows), 0.0, SCALE_MAX),
            f"s=clip({a:+.2f}{b:+.2f}z)")


def fit_wconst(train: list[dict], name: str | None = None):
    return _fit_weighted(train, None, slope=False)


def fit_linear(train: list[dict], name: str):
    return _fit_weighted(train, name, slope=True)


def fit_scaled(train: list[dict], name: str):
    """s = clip(c f, 0, 4) with ONE parameter, fitted the same weighted way as wconst.

    WHERE THE FORM COMES FROM AND WHY IT IS EXPECTED TO LOSE. The per-condition optimum is
    exactly rho ||e|| / ||r||, which falls as 1 / ||r||, so f = 1 / residual_norm and
    f = additive_norm / residual_norm are the algebra's own shape (||e|| is the additive
    baseline's error and is not available at test time; ||sum_a w_a|| is the closest
    test-time stand-in the same ridge fit produces).

    Reasoning from that form to a RULE is a mistake, and the synthetic case in
    tests/test_rule.py shows it costing +0.177 where a gate on the same column gains 0.028.
    The reason is that the quantity which varies is rho, which cannot be observed, and it
    carries the SIGN. Holding rho fixed per group, s* = rho ||e|| / ||r|| makes the
    unreliable small-||r|| conditions the ones with the LARGEST |s*|, so a rule proportional
    to 1 / ||r|| amplifies exactly where it should switch off. The 1/||r|| shape is right
    about the optimum and wrong about the decision.

    It is kept because it is cheap and because being refuted on the real data is worth
    more than leaving the family untested: a gate that beats it there confirms the failure
    mode is sign rather than scale.

    c = sum w f s* / sum w f^2 with w = ||r||^2, the no-intercept member of wconst's family,
    so the comparison between them isolates f.
    """
    feature = np.array([r[name] for r in train], dtype=float)
    target = np.array([r["best_s"] for r in train], dtype=float)
    weight = np.array([r["rr"] for r in train], dtype=float)
    keep = (np.isfinite(feature) & np.isfinite(target) & (weight > 0))
    denominator = float(np.sum(weight[keep] * feature[keep] ** 2))
    if keep.sum() < 3 or denominator <= 0.0:
        return fit_const(train, name)
    c = float(np.sum(weight[keep] * feature[keep] * target[keep])) / denominator

    def apply(rows: list[dict]) -> np.ndarray:
        f = np.array([r[name] for r in rows], dtype=float)
        return np.clip(np.where(np.isfinite(f), c * f, 0.0), 0.0, SCALE_MAX)

    return apply, f"s=clip({c:+.3g}x{name})"


POWERS = np.arange(0.0, 1.501, 0.1)


def fit_power(train: list[dict], name: str):
    """s = clip(c z^-p, 0, 4). ONE FAMILY that nests both rules the data picked.

    THIS EXISTS TO AVOID CHOOSING A FAMILY OFF A LEADERBOARD. Measured on w6, held out by
    fold on combosciplex and by condition on Norman, the winners were opposite: s = c/||r||
    recovered 59-62 % of the oracle on combosciplex where a constant recovered 19 %, and the
    same rule recovered 16 % on Norman where a constant recovered 51 %. Picking per dataset
    from seventeen rules scored on the validation folds is the selection problem that made
    scPKFM unreadable - thirteen structural additions, none of which replicated.

    p = 0 is exactly wconst and p = 1 is a CONSTANT CORRECTION MAGNITUDE, since ||s r||
    = c when s = c/||r||. So the exponent is a single interpretable quantity: how much of
    the model's predicted magnitude to keep. p = 0 keeps all of it, p = 1 keeps none and
    trusts only the direction. Both datasets are fitted in the same family and report their
    own p, which is a measurement rather than a choice.

    c is fitted by wconst's weighted least squares at each p, and p is then chosen by the
    training mean L2 - both on the training split alone.
    """
    feature_raw = np.array([r[name] for r in train], dtype=float)
    target = np.array([r["best_s"] for r in train], dtype=float)
    weight = np.array([r["rr"] for r in train], dtype=float)
    usable = np.isfinite(feature_raw) & (feature_raw > 1e-9) & np.isfinite(target)
    if (usable & (weight > 0)).sum() < 3:
        return fit_const(train, name)

    def coefficient(power: float) -> float:
        f = np.where(usable, feature_raw, 1.0) ** (-power)
        keep = usable & (weight > 0)
        denominator = float(np.sum(weight[keep] * f[keep] ** 2))
        if denominator <= 0.0:
            return 0.0
        return float(np.sum(weight[keep] * f[keep] * target[keep])) / denominator

    def scales(rows: list[dict], power: float, c: float) -> np.ndarray:
        f = np.array([r[name] for r in rows], dtype=float)
        good = np.isfinite(f) & (f > 1e-9)
        scaled = np.where(good, c * np.where(good, f, 1.0) ** (-power), 0.0)
        return np.clip(scaled, 0.0, SCALE_MAX)

    best, score = (0.0, 0.0), np.inf
    for power in POWERS:
        c = coefficient(power)
        value = float(np.mean(l2(train, scales(train, power, c))))
        if value < score:
            best, score = (power, c), value
    power, c = best
    return (lambda rows: scales(rows, power, c),
            f"s=clip({c:.3g}x{name}^-{power:.1f})")


def fit_power2(train: list[dict], _name: str | None = None):
    """s = clip(c ||sum_a w_a||^q ||r||^-p, 0, 4). TWO exponents, both measured.

    The per-condition optimum is exactly rho ||e|| / ||r||. ||e|| is the additive
    baseline's own error and is not available at test time, but ||sum_a w_a|| is - it comes
    from the same ridge fit - so this family asks how much of each the data wants:
    q is how far the additive displacement stands in for ||e||, p is how much of the
    model's own magnitude to discard.

    IT EXISTS TO AVOID A SECOND ROUND OF LEADERBOARD SHOPPING. Reading the full rule table
    for the whitened arms produced two different winners on two families -
    power(residual_norm) at 52 % on norman:ncomb and scaled(ratio_add_res) at 52 % on
    norman:nval, where power managed 33 % - and picking per family from twenty rules scored
    on the validation folds is exactly the selection problem this repository exists to
    avoid. This family nests both: q = 0 is power(residual_norm) and q = p = 1 is
    scaled(ratio_add_res), so the exponents are a measurement in the same sense p already
    was.

    c is fitted by wconst's weighted least squares at each (q, p) and the pair is chosen by
    the training mean L2, all on the training split.
    """
    target = np.array([r["best_s"] for r in train], dtype=float)
    weight = np.array([r["rr"] for r in train], dtype=float)
    residual = np.array([r.get("residual_norm", np.nan) for r in train], dtype=float)
    additive = np.array([r.get("additive_norm", np.nan) for r in train], dtype=float)
    usable = (np.isfinite(target) & (weight > 0) & np.isfinite(residual)
              & (residual > 1e-9) & np.isfinite(additive) & (additive > 1e-9))
    if usable.sum() < 4:
        return fit_const(train, None)

    def feature(a: np.ndarray, r: np.ndarray, q: float, p: float) -> np.ndarray:
        good = np.isfinite(a) & np.isfinite(r) & (a > 1e-9) & (r > 1e-9)
        out = np.zeros(len(a))
        out[good] = a[good] ** q * r[good] ** (-p)
        return out

    best, score = (0.0, 0.0, 0.0), np.inf
    for q in np.arange(0.0, 1.501, 0.25):
        for p in POWERS:
            f = feature(additive, residual, q, p)
            denominator = float(np.sum(weight[usable] * f[usable] ** 2))
            if denominator <= 0.0:
                continue
            c = float(np.sum(weight[usable] * f[usable] * target[usable])) / denominator
            value = float(np.mean(l2(train, np.clip(c * f, 0.0, SCALE_MAX))))
            if value < score:
                best, score = (q, p, c), value
    q, p, c = best

    def apply(rows: list[dict]) -> np.ndarray:
        a = np.array([r.get("additive_norm", np.nan) for r in rows], dtype=float)
        r = np.array([r.get("residual_norm", np.nan) for r in rows], dtype=float)
        return np.clip(c * feature(a, r, q, p), 0.0, SCALE_MAX)

    return apply, f"s=clip({c:.3g}xadd^{q:.2f}xres^-{p:.1f})"


def fit_gate(train: list[dict], name: str):
    """s = c where z is above a threshold, 0 below it. Two parameters, both on a grid.

    The blunt form of the same idea as fit_linear, and worth reporting beside it: if the
    gate matches the linear rule then all the rule is doing is switching the residual off
    on the unreliable conditions, and the ORDER within the reliable ones carries nothing.
    """
    values = np.array([r[name] for r in train], dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 4:
        return fit_const(train, name)
    thresholds = np.quantile(finite, np.arange(0.0, 0.81, 0.05))
    best, score = (thresholds[0], GRID[0]), np.inf
    for threshold, constant in itertools.product(thresholds, GRID):
        scales = np.where(values >= threshold, constant, 0.0)
        value = float(np.mean(l2(train, scales)))
        if value < score:
            best, score = (threshold, constant), value

    threshold, constant = best

    def apply(rows: list[dict]) -> np.ndarray:
        z = np.array([r[name] for r in rows], dtype=float)
        return np.where(np.isfinite(z) & (z >= threshold), constant, 0.0)

    return apply, f"s={constant:.2f} if {name}>={threshold:.3g}"


# The three parameter-free references every rule is measured against. `zero` is the
# additive baseline exactly; `const` and `wconst` are two estimators of the same single
# global scale, and a rule has to beat the BETTER of them to have earned its slope.
RULES = {"zero": (fit_zero, None), "const": (fit_const, None),
         "wconst": (fit_wconst, None), "power2": (fit_power2, None)}
REFERENCES = ("zero", "const", "wconst")


# ------------------------------------------------------------------- cross-validation

def splits_by(rows: list[dict], key: str) -> list[tuple[list[dict], list[dict]]]:
    """Leave-one-group-out, grouped by `key`: never by row, so seeds cannot leak."""
    groups = sorted({r[key] for r in rows})
    return [([r for r in rows if r[key] != g], [r for r in rows if r[key] == g])
            for g in groups]


def evaluate(doubles: list[dict], single: float | None, key: str,
             rules: dict) -> list[tuple]:
    """Each rule's CV metric: fitted on the training split, scored on the held-out one.

    Scored by POOLING the held-out rows across splits and taking one mean, not by averaging
    per-split means, because the splits have different sizes and the reported metric is a
    mean over conditions.
    """
    out = []
    for label, (fit, name) in rules.items():
        held, descriptions = [], []
        for train, test in splits_by(doubles, key):
            apply, description = fit(train, name)
            held.append(l2(test, apply(test)))
            descriptions.append(description)
        double = float(np.mean(np.concatenate(held)))
        total = (double if single is None or not np.isfinite(single)
                 else TABLE3["double"] * double + TABLE3["single"] * single)
        counts = collections.Counter(descriptions)
        out.append((label, total, counts.most_common(1)[0][0],
                    len(counts)))
    return out


def main() -> None:
    global SCALE_MAX, GRID       # --scale-max rewrites both; see the constant's comment
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="the CSV scripts/dev_reliability.py --out wrote")
    parser.add_argument("--predictors", nargs="*",
                        default=["residual_norm", "obs_norm", "abscissa", "b_norm",
                                 "additive_norm", "overlap_obs"])
    parser.add_argument("--scaled", nargs="*",
                        default=["inv_residual", "ratio_add_res", "inv_obs"],
                        help="features for the one-parameter s = c f rule, which is the "
                             "form the optimum rho ||e|| / ||r|| actually takes")
    parser.add_argument("--fit-all", action="store_true",
                        help="also fit power(residual_norm) on the WHOLE family and print "
                             "its coefficients. The cross-validated tables say what a rule "
                             "is WORTH; this says which numbers the FINAL run should carry, "
                             "because a run scored on the test set is calibrated from all "
                             "of the validation data rather than from one held-out split.")
    parser.add_argument("--power", nargs="*", default=["residual_norm", "obs_norm"],
                        help="bases for s = c z^-p, the two-parameter family that nests a "
                             "constant scale (p=0) and a constant correction magnitude "
                             "(p=1). residual_norm is the pre-registered base: ||s r|| is "
                             "what enters the prediction, so it is the quantity a "
                             "calibration should be expressed in.")
    parser.add_argument("--scale-max", type=float, default=SCALE_MAX,
                        help="the cap on s, default 4. It was never measured, and on "
                             "norman:ncomb the fitted rule puts 45 of 60 conditions at "
                             "it, so it decides that family rather than protecting it. "
                             "Sweep it before a final run.")
    args = parser.parse_args()

    SCALE_MAX = float(args.scale_max)
    GRID = np.arange(0.0, SCALE_MAX + 0.001, 0.05)

    with open(args.csv, encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    numeric = (set(raw[0]) - {"family", "group", "arm", "condition", "block", "helps"}
               ) | {"inv_residual", "inv_obs", "ratio_add_res"}
    rows = []
    for row in raw:
        out = {k: row[k] for k in ("family", "group", "arm", "condition", "block")}
        out["seed"] = int(row["seed"])
        for key in numeric:
            try:
                out[key] = float(row[key])
            except (ValueError, KeyError):
                out[key] = float("nan")
        # Derived features, all from columns already in the CSV. 1 / ||r|| is the shape the
        # optimum has; the ratio adds the only test-time stand-in for ||e||.
        residual = out.get("residual_norm", float("nan"))
        observable = out.get("obs_norm", float("nan"))
        out["inv_residual"] = 1.0 / residual if residual > 1e-9 else float("nan")
        out["inv_obs"] = 1.0 / observable if observable > 1e-9 else float("nan")
        out["ratio_add_res"] = (out.get("additive_norm", float("nan")) / residual
                                if residual > 1e-9 else float("nan"))
        rows.append(out)

    rules = dict(RULES)
    for name in args.predictors:
        if name not in numeric:
            continue
        rules[f"linear({name})"] = (fit_linear, name)
        rules[f"gate({name})"] = (fit_gate, name)
    for name in args.scaled:
        rules[f"scaled({name})"] = (fit_scaled, name)
    for name in args.power:
        rules[f"power({name})"] = (fit_power, name)

    # Split by ARM as well as family. Pooling arms would average models with different
    # residual directions into one set of quadratic coefficients, and the whole reason to run
    # this per arm is that an arm's rho sets its ceiling: w8 measured compsum at rho 0.533
    # against base's 0.464, a ceiling of -15.4 % against -11.4 %, while compsum LOST on L2 at
    # s = 1. Comparing arms before the calibration can reject the best one.
    for family, arm in sorted({(r["family"], r["arm"]) for r in rows}):
        mine = [r for r in rows if r["family"] == family and r["arm"] == arm]
        doubles = [r for r in mine if r["block"] == "double"]
        singles = [r for r in mine if r["block"] == "single"]
        single = float(np.mean(l2(singles, np.zeros(len(singles))))) if singles else None

        oracle_s = np.array([r["best_s"] if np.isfinite(r["best_s"]) else 0.0
                             for r in doubles])
        floor = metric(doubles, np.zeros(len(doubles)), single)
        ceiling = metric(doubles, oracle_s, single)
        conditions = len({r["condition"] for r in doubles})

        if args.fit_all:
            # Fitted on EVERY row of this family, with nothing held out, because this is
            # not an estimate of what the rule is worth - the tables below are - but the
            # coefficients a final run should carry. It uses the one-exponent family and
            # not power2 because that is the one src/models/model.py implements, and the
            # two were within 0.002 of each other on all three families.
            apply, description = fit_power(doubles, "residual_norm")
            scales = apply(doubles)
            print(f"\n=== {family}  arm {arm}  CALIBRATION FOR THE FINAL RUN ===")
            print(f"  {description}")
            print(f"  on all {len(doubles)} rows: metric "
                  f"{metric(doubles, scales, single):.4f} (in-sample, so optimistic - "
                  f"read the cross-validated tables for what it is worth)")
            print(f"  s applied: median {np.median(scales):.3f}, "
                  f"range {scales.min():.3f}-{scales.max():.3f}, "
                  f"{int((scales >= 3.999).sum())}/{len(scales)} at the clip")

        keys = [("condition", "held out by condition")]
        if len({r["group"] for r in doubles}) > 1:
            keys.append(("group", "held out by fold, the stricter test"))

        print(f"\n=== {family}  arm {arm}   {len(doubles)} double rows "
              f"over {conditions} conditions, "
              f"{len({r['group'] for r in doubles})} folds ===")
        print(f"  s=0, the additive baseline: {floor:.4f}")
        print(f"  per-condition oracle:       {ceiling:.4f}  ({ceiling - floor:+.4f})   "
              f"<- the ceiling, not a rule")
        for key, description in keys:
            print(f"\n  {description}")
            print(f"  {'rule':>26} {'metric':>8} {'vs s=0':>8} {'of oracle':>10}  "
                  f"{'fitted':>0}")
            scored = evaluate(doubles, single, key, rules)
            values = dict((label, value) for label, value, *_ in scored)
            # The bar is the better of the two constant estimators, never whichever one
            # happens to flatter the rule being tested.
            baseline = min(values["const"], values["wconst"])
            for label, value, description_of_fit, distinct in sorted(
                    scored, key=lambda item: item[1]):
                share = ((value - floor) / (ceiling - floor) * 100.0
                         if ceiling < floor else float("nan"))
                mark = "" if label in REFERENCES or value >= baseline else "  <-"
                print(f"  {label:>26} {value:8.4f} {value - floor:+8.4f} "
                      f"{share:9.0f}%  {description_of_fit}"
                      f"{'' if distinct == 1 else f' (+{distinct - 1} more)'}{mark}")
    print("\nEvery rule is fitted on the training split and scored on the held-out one. "
          "`of oracle` is the fraction of the per-condition ceiling recovered; `const` is "
          "what a single global scale already gives, and a rule that does not beat it is "
          "not worth its parameters.")


if __name__ == "__main__":
    main()
