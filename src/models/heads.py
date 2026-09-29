"""Decoder output heads.

Measured on the modelled space: 41.2 % of entries are exactly zero, while an MSE
head produces exact zeros 0.000 % of the time and loses 24 % of the standard
deviation. Since the judging metric is an energy distance between POPULATIONS,
that lost spread is paid directly - which is why the head is a config axis rather
than a fixed choice.

    mse     Gaussian point estimate. Baseline and control arm.
    hurdle  gate * magnitude, the two-part (MAST-style) decomposition: a sigmoid
            decides zero vs non-zero, a softplus gives the magnitude. Distinct
            from zero-inflation - here zeros have exactly one source.
    zinb    scVI-style zero-inflated negative binomial on recovered counts.
            Available because raw counts turn out to be exactly recoverable
            (see src/data/counts.py), but it models a different space than the
            metric, so it carries a log1p round-trip.

All heads expose `point_estimate` in log1p space, because that is where every
metric and every baseline lives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HurdleHead(nn.Module):
    """x = B * M, with B ~ Bernoulli(sigma) and M drawn from p(M | z).

    A VAE decoder IS a distribution p(x|z) - that is what the ELBO's
    reconstruction term is a likelihood of. Training it with plain MSE silently
    fixes p(x|z) = N(f(z), sigma^2 I) with sigma^2 CONSTANT, so the network only
    ever learns the mean and inference returns E[x|z] rather than a sample. That
    is the same mechanism that makes VAE image reconstructions blurry; here it
    shows up as collapsed cell-to-cell variance (0.098 predicted vs 0.495 real).

    Both factors therefore have to be realised, not averaged:
      B  Bernoulli sampling  - restored edist_rel 6.64 -> 1.46
      M  a learned dispersion, so the magnitude is drawn rather than pinned to
         its conditional mean. Measured gap after fixing B alone: predicted std
         0.336 vs 0.495 real, i.e. std ~0.363 of magnitude spread still missing.
    """

    def __init__(self, width: int, n_genes: int, bce_weight: float = 1.0,
                 gate_mode: str = "sample", magnitude_mode: str = "gaussian"):
        super().__init__()
        self.gate = nn.Linear(width, n_genes)
        self.magnitude = nn.Linear(width, n_genes)
        self.bce_weight = bce_weight
        self.gate_mode = gate_mode
        self.magnitude_mode = magnitude_mode
        if magnitude_mode == "gaussian":
            self.log_scale = nn.Linear(width, n_genes)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        params = {"gate_logit": self.gate(h), "magnitude": F.softplus(self.magnitude(h))}
        if self.magnitude_mode == "gaussian":
            params["log_scale"] = self.log_scale(h).clamp(-6.0, 2.0)
        return params

    def loss(self, params: dict, x: torch.Tensor, **_) -> tuple[torch.Tensor, dict]:
        observed = (x > 0).float()
        gate_loss = F.binary_cross_entropy_with_logits(params["gate_logit"], observed)
        # Magnitude is only supervised where something was actually detected;
        # training it on zeros would drag every prediction toward zero and undo
        # the point of separating the two parts.
        denominator = observed.sum().clamp(min=1.0)
        residual = params["magnitude"] - x
        if self.magnitude_mode == "gaussian":
            # Gaussian NLL, which is what MSE already was - except sigma is now
            # learned instead of pinned, so the decoder keeps the spread that a
            # point estimate throws away.
            log_scale = params["log_scale"]
            nll = 0.5 * (residual / log_scale.exp()) ** 2 + log_scale
            magnitude_loss = (nll * observed).sum() / denominator
        else:
            magnitude_loss = ((residual ** 2) * observed).sum() / denominator
        total = magnitude_loss + self.bce_weight * gate_loss
        # A Gaussian NLL is a log-DENSITY, so it goes negative once sigma drops
        # below 1 - normal, but unreadable in a log and not a number to put in a
        # paper. rmse is the same fit expressed positively; sigma is reported so a
        # variance collapse (the known pathology of this loss, where shrinking
        # sigma improves the objective without improving predictions) is visible
        # rather than hidden inside the NLL.
        parts = {"recon": float(magnitude_loss), "gate_bce": float(gate_loss),
                 "rmse": float((((residual ** 2) * observed).sum()
                                / denominator).sqrt())}
        if self.magnitude_mode == "gaussian":
            scale = params["log_scale"].exp()
            # DEPRECATED. Kept only so the numbers in existing logs stay readable.
            # sigma averages EVERY entry while rmse is a root-mean-square over the
            # detected ones alone, so this ratio divides an arithmetic mean by a
            # quadratic one over two different supports. Under the heteroscedasticity
            # of expression data the Jensen gap drives it well below 1 on its own,
            # and the ~41 % of entries the NLL never supervises (`nll * observed`
            # masks them) drift freely inside the clamp yet still enter the mean. A
            # perfectly calibrated head scores low here, so a low value is not
            # evidence of under-dispersion. Use chi2 or calib_rms instead.
            parts["sigma"] = float(scale.mean())
            parts["calib"] = parts["sigma"] / max(parts["rmse"], 1e-8)
            # The statistics that actually test calibration. Both are restricted to
            # the entries the magnitude loss is trained on, and both compare like
            # with like.
            #   chi2       mean of (residual / sigma)^2, per entry. Exactly 1.0 when
            #              the predicted spread matches the observed error; above 1
            #              is over-confident (sigma too small), below 1 too wide.
            #   calib_rms  the same statement in the units of rmse - the quadratic
            #              mean of sigma over rmse, so the Jensen gap is gone and 1.0
            #              is again the calibrated value.
            # chi2 is the one to quote: it weights each entry equally rather than
            # letting the few high-variance genes set the scale.
            parts["chi2"] = float((((residual / scale.clamp(min=1e-6)) ** 2)
                                   * observed).sum() / denominator)
            parts["calib_rms"] = float((((scale ** 2) * observed).sum()
                                        / denominator).sqrt()) / max(parts["rmse"], 1e-8)
        return total, parts

    def point_estimate(self, params: dict, **_) -> torch.Tensor:
        """How the binary event is realised at inference. Three different answers:

        soft    sigma * magnitude. The conditional expectation, so it is optimal
                for anything mean-based - but it never emits an exact zero, while
                41.2 % of the real data is exactly zero.
        hard    threshold at 0.5. Emits zeros, but deterministically per gene: a
                gene with sigma = 0.4 becomes zero in EVERY cell instead of 40 %
                of them, which destroys cell-to-cell variability.
        sample  Bernoulli(sigma). Reproduces the marginal zero rate AND the
                per-gene spread, and stays unbiased for the mean. This is the one
                a population-level metric wants.
        """
        probability = torch.sigmoid(params["gate_logit"])
        if self.gate_mode == "soft":
            gate = probability
        elif self.gate_mode == "hard":
            gate = (probability > 0.5).to(probability.dtype)
        elif self.gate_mode == "sample":
            gate = torch.bernoulli(probability)
        else:
            raise ValueError(f"unknown hurdle gate_mode {self.gate_mode!r}")

        magnitude = params["magnitude"]
        # Draw the magnitude only when the gate is also being drawn: mixing a
        # sampled factor with an averaged one would give neither the right mean
        # nor the right spread. Clamped at zero because log1p values cannot be
        # negative; the resulting bias is small next to the variance recovered.
        if self.magnitude_mode == "gaussian" and self.gate_mode == "sample":
            noise = torch.randn_like(magnitude) * params["log_scale"].exp()
            magnitude = (magnitude + noise).clamp(min=0.0)
        return gate * magnitude


def build_head(config: dict, width: int, n_genes: int) -> nn.Module:
    """Only the hurdle head is built here; mse and zinb live in the original repo."""
    model_cfg = config["model"]
    kind = model_cfg["decoder_head"]
    if kind != "hurdle":
        raise ValueError(
            f"decoder_head {kind!r} is not in this build. 41 % of entries are "
            "exactly zero, so the detection/magnitude split is not optional here.")
    return HurdleHead(width, n_genes, model_cfg["hurdle_bce_weight"],
                      model_cfg["hurdle_gate"], model_cfg["hurdle_magnitude"])
