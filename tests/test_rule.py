"""scripts/dev_rule.py must reward a real separator and punish a fake one.

These are tests of a STATISTICAL procedure, not of a model, and they exist because the
procedure's whole job is to answer "is this predictor worth its parameters?" - a question it
would answer yes to for anything at all if the cross-validation were wrong. Two of this
repository's earlier tests were themselves statistically wrong (one bounded 10,240 entries
at 4 sigma each and failed 65 % of the time by chance), so the check that a NOISE predictor
cannot win is the more important of the two below.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                "scripts")))

import dev_rule


def synthetic(n_conditions: int = 24, seeds: int = 3, seed: int = 0) -> list[dict]:
    """Half the conditions have a large well-aligned residual, half a tiny hostile one.

    Built in the coordinates the real measurement reports: ||e|| is the additive baseline's
    error, ||r|| the residual's norm and rho their cosine, from which
    ee = ||e||^2, er = -rho ||e|| ||r||, rr = ||r||^2 - the sign because e points from the
    truth to the prediction while r is added to it.

    `residual_norm` separates the two groups perfectly and every other column is noise, so
    the two tests below are the same data read twice.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_conditions):
        good = index % 2 == 0
        for s in range(seeds):
            e = 2.0
            r = (1.2 if good else 0.15) * float(np.exp(rng.normal(0.0, 0.05)))
            rho = (0.6 if good else -0.4) + rng.normal(0.0, 0.03)
            rows.append({"family": "synth:sv", "group": f"sv{index % 3}", "arm": "a",
                         "seed": s, "condition": f"c{index}", "block": "double",
                         "best_s": rho * e / r, "ee": e * e, "er": -rho * e * r,
                         "rr": r * r, "residual_norm": r, "inv_residual": 1.0 / r,
                         # power2 needs it; a constant would make q unidentifiable, so it
                         # varies and carries no signal about best_s.
                         "additive_norm": float(np.exp(rng.normal(0.0, 0.3))),
                         "noise": rng.normal(),
                         # A strictly positive noise column, since the power family needs
                         # a positive base and would otherwise silently fall back.
                         "noise_positive": float(np.exp(rng.normal()))})
    return rows


def scores(rows: list[dict], key: str = "condition") -> dict[str, float]:
    rules = {"zero": (dev_rule.fit_zero, None), "const": (dev_rule.fit_const, None),
             "wconst": (dev_rule.fit_wconst, None)}
    for name in ("residual_norm", "noise"):
        rules[f"linear({name})"] = (dev_rule.fit_linear, name)
        rules[f"gate({name})"] = (dev_rule.fit_gate, name)
        rules[f"scaled({name})"] = (dev_rule.fit_scaled, name)
    # 1 / ||r|| is the shape the optimum takes, so the one-parameter rule is given it
    # directly: this is the form the real measurement will be judged on.
    rules["scaled(inv_residual)"] = (dev_rule.fit_scaled, "inv_residual")
    # The two-parameter family that nests a constant scale at p = 0 and a constant
    # correction magnitude at p = 1. It must not beat the bar on a noise column either.
    rules["power(residual_norm)"] = (dev_rule.fit_power, "residual_norm")
    rules["power(noise_positive)"] = (dev_rule.fit_power, "noise_positive")
    # The two-exponent family, which nests power at q = 0 and the additive-ratio rule at
    # q = p = 1. More freedom means more room to fit noise, so it is held to the same bar.
    rules["power2"] = (dev_rule.fit_power2, None)
    return {label: value for label, value, *_ in
            dev_rule.evaluate(rows, None, key, rules)}


def test_zero_is_the_additive_baseline_exactly():
    """s = 0 must reproduce ||e||, since that is what the design's floor is."""
    rows = synthetic()
    assert scores(rows)["zero"] == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize("key", ["condition", "group"])
def test_a_real_separator_beats_a_single_global_scale(key):
    """A perfect separator must recover materially more of the oracle than a constant.

    Held out by condition AND by fold: a rule that only works when the held-out condition's
    siblings are in the training split would pass the first and fail the second.
    """
    rows = synthetic()
    result = scores(rows, key)
    bar = min(result["const"], result["wconst"])
    assert result["gate(residual_norm)"] < bar - 0.01
    assert result["linear(residual_norm)"] < bar - 0.005


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_noise_predictor_cannot_beat_a_single_global_scale(seed):
    """The load-bearing test. A column of noise must not look like a real predictor.

    Run over five seeds rather than one because the claim is about the PROCEDURE and a
    single seed on which noise happened to lose would not establish it.

    The bar is the better of the two constant estimators. That distinction is the reason
    this test exists: against `const` alone, noise WON by 0.0008 on all five seeds, because
    `const` takes an argmin over a grid of the training mean L2 while a fitted intercept is
    a weighted average and therefore smoother. The margin below is 0.002 - large enough to
    absorb that estimator gap, and small enough that the real separator's 0.01 in the test
    above is still comfortably out of reach of noise.
    """
    rows = synthetic(seed=seed)
    result = scores(rows)
    bar = min(result["const"], result["wconst"])
    assert result["gate(noise)"] >= bar - 0.002
    assert result["linear(noise)"] >= bar - 0.002
    assert result["scaled(noise)"] >= bar - 0.002
    assert result["power(noise_positive)"] >= bar - 0.002
    assert result["power2"] >= bar - 0.002


def test_the_oracle_is_a_ceiling_no_rule_passes():
    """Every cross-validated rule must sit above the per-condition oracle.

    A rule scoring below it would mean the held-out rows were used to fit it, which is the
    one failure this whole script is built to avoid.
    """
    rows = synthetic()
    oracle = float(np.mean(dev_rule.l2(
        rows, np.array([r["best_s"] for r in rows]))))
    for label, value in scores(rows).items():
        assert value >= oracle - 1e-9, f"{label} beat the oracle at {value:.4f}"
