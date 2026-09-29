"""Evaluation metrics for the internal harness.

The conventional reported metric (top-20 DE Pearson on the delta) is saturated at
~0.98 for every baseline and does not separate models, so it is reported but never
used to judge. Judging uses:

  edist_rel  energy distance between predicted and real cells, divided by the same
             quantity for control cells. 0 = perfect, 1 = no better than control.
  resid_R2   how much of the non-additive residual a model recovers. An additive
             prediction scores exactly 0 by construction, so this is the metric
             that actually asks whether interaction has been learned.

`power` selects the distance convention. power=1 is Szekely's energy distance,
a genuine divergence between distributions. power=2 collapses algebraically to
2*||mean_X - mean_Y||^2 and therefore sees only first moments - it is kept because
some of the literature (scPerturb) reports that variant under the same name.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _mean_pairwise(a: torch.Tensor, b: torch.Tensor, power: int, same: bool,
                   block: int = 2048) -> float:
    """Mean pairwise distance between rows of `a` and `b`.

    When `same` the diagonal is excluded, which is what an unbiased within-sample
    term needs; including it would shrink sigma and inflate the energy distance.
    """
    total, count = 0.0, 0
    for start in range(0, a.shape[0], block):
        chunk = a[start:start + block]
        d2 = torch.cdist(chunk, b, p=2) ** 2 if power == 2 else torch.cdist(chunk, b, p=2)
        if same:
            rows = torch.arange(start, min(start + block, a.shape[0]), device=a.device)
            d2[torch.arange(chunk.shape[0], device=a.device), rows] = 0.0
        total += float(d2.sum())
        count += chunk.shape[0] * b.shape[0]
    if same:
        count -= a.shape[0]
    return total / max(count, 1)


def energy_distance(x: np.ndarray, y: np.ndarray, power: int = 1,
                    device: str = "cpu") -> float:
    """E(X, Y) = 2*E||x-y|| - E||x-x'|| - E||y-y'||  (raised to `power`)."""
    xt, yt = _to_tensor(x, device), _to_tensor(y, device)
    cross = _mean_pairwise(xt, yt, power, same=False)
    within_x = _mean_pairwise(xt, xt, power, same=True)
    within_y = _mean_pairwise(yt, yt, power, same=True)
    return 2.0 * cross - within_x - within_y


def edist_rel(pred: np.ndarray, real: np.ndarray, control: np.ndarray,
              power: int = 1, device: str = "cpu") -> float:
    """Energy distance to the real cells, relative to what control already achieves."""
    denominator = energy_distance(control, real, power, device)
    if denominator <= 0:
        return float("nan")
    return energy_distance(pred, real, power, device) / denominator


def residual(m_ab: np.ndarray, m_a: np.ndarray, m_b: np.ndarray,
             m_ctrl: np.ndarray) -> np.ndarray:
    """The non-additive part of a double perturbation's mean response."""
    return m_ab - m_a - m_b + m_ctrl


def residual_r2(m_hat: np.ndarray, m_ab: np.ndarray, m_a: np.ndarray,
                m_b: np.ndarray, m_ctrl: np.ndarray, e_noise: float = 0.0,
                min_snr: float = 1.0) -> float:
    """Fraction of the recoverable residual signal that `m_hat` explains.

    Both numerator and denominator have the noise floor subtracted, so a model
    that only reproduces sampling noise scores 0 rather than something positive.
    An exactly additive prediction gives r_hat = 0 and therefore R2 = 0.

    PER-CONDITION USE IS THE UNSTABLE ONE. e_noise is a sum of var/n over the four
    conditions and they range from 46 to 1,005 cells, so on a small, weakly
    non-additive double the subtraction drives ||r||^2 - e_noise to nearly zero and
    the ratio explodes. The old guard tested abs(denominator), which also let a
    NEGATIVE denominator through - and that flips the sign of the score. Both are
    why the per-double mean and the pooled value diverged by 2.6x (-2.889 vs
    -1.117) on the same run: a handful of near-zero denominators carried the mean.

    `min_snr` therefore requires the residual to stand clear of its own noise floor
    by that factor - ||r||^2 >= (1 + min_snr) * e_noise - and returns nan otherwise.
    A skipped condition means "this double has no measurable non-additivity to
    score against", not "the model failed"; callers should report how many were
    skipped. min_snr=0 recovers the previous behaviour apart from the sign guard,
    and e_noise=0 makes the test vacuous, as it should be.

    The POOLED form - summing numerators and denominators before dividing, as
    evaluate_model does - needs none of this and must not be filtered: the
    per-condition subtractions are an unbiased estimator whose errors cancel in the
    sum, and dropping low-signal doubles would bias it toward the strong ones.
    """
    r = residual(m_ab, m_a, m_b, m_ctrl)
    r_hat = residual(m_hat, m_a, m_b, m_ctrl)
    signal = float(r @ r)
    denominator = signal - e_noise
    if denominator <= 0 or signal < (1.0 + min_snr) * e_noise:
        return float("nan")
    return 1.0 - (float((r_hat - r) @ (r_hat - r)) - e_noise) / denominator


def effective_n(sizes: tuple[int, ...]) -> float:
    """Harmonic-style effective sample size of the four-term residual.

    Var(r_g) ~ sigma_g^2 * sum(1/n_i), so this is the n that a single-sample
    estimate would need to carry the same variance.
    """
    return 1.0 / sum(1.0 / max(n, 1) for n in sizes)


def noise_floor(control_cells: np.ndarray, sizes: tuple[int, int, int, int],
                n_draws: int = 200, seed: int = 0) -> dict[str, float]:
    """Empirical E[||r||^2] under the null of no real perturbation effect.

    Four DISJOINT groups are drawn from control cells with the same sizes as the
    (AB, A, B, ctrl) conditions and combined by the residual formula. Because the
    groups are all control, any signal is sampling noise alone.

    Size matching matters: conditions range from 46 to 1,005 cells, a 20x spread,
    so a single global threshold would mark every small condition as a hit.
    """
    rng = np.random.default_rng(seed)
    n_control = control_cells.shape[0]
    wanted = sum(sizes)
    if wanted > n_control:
        # The real n_ctrl uses every control cell, which cannot be disjoint from
        # the other three. Shrink only the last group: its 1/n term is negligible.
        sizes = (*sizes[:3], max(n_control - sum(sizes[:3]), 1))

    draws = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        order = rng.permutation(n_control)
        cut, groups = 0, []
        for n in sizes:
            groups.append(control_cells[order[cut:cut + n]].mean(axis=0))
            cut += n
        r = residual(groups[0], groups[1], groups[2], groups[3])
        draws[i] = float(r @ r)

    variance = control_cells.var(axis=0).sum()
    n_eff = effective_n(sizes)
    analytic = variance / n_eff
    empirical = float(draws.mean())
    return {
        "e_noise": empirical,
        "e_noise_analytic": analytic,
        "lambda": empirical / analytic if analytic > 0 else float("nan"),
        "n_eff": n_eff,
        "std": float(draws.std()),
    }


def de20_pearson(delta_hat: np.ndarray, delta_true: np.ndarray, k: int = 20) -> float:
    """Pearson correlation on the top-k genes of the true delta. Saturated; report only."""
    top = np.argsort(-np.abs(delta_true))[:k]
    a, b = delta_hat[top], delta_true[top]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
