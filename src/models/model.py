"""scPKR: the additive component plus a pathway-Koopman residual.

    Delta x(S, x) = sum_{a in S} w_a                       closed form, gene space
                  + 1[|S| >= 2] * W (exp(B_S) - I) Phi(x)  learned, pathway scaffold

    xhat = x + Delta x(S, x),  realised through a mean-preserving hurdle head

TWO PROPERTIES HOLD BY CONSTRUCTION, and tests/test_structure.py asserts both as
numbers rather than trusting this docstring:

  1. AT INITIALISATION THE PREDICTION IS THE ADDITIVE BASELINE, EXACTLY. W starts at
     zero, so the residual term is identically zero whatever the operators do, and the
     head is mean-preserving, so it cannot move the result either. The model therefore
     starts at a measured 1.858 on Table 3 (1.669 / 1.416 / 2.251 on Tables 1 and 2)
     and training can only move down from there. scPKFM started at 2.138 with 0.97 of
     that being a decoder ceiling it could never remove, and after thirteen structural
     additions had no replicated improvement.

  2. A SINGLE PERTURBATION IS THE ADDITIVE BASELINE, EXACTLY, AT EVERY STEP. B_S is a
     sum over unordered PAIRS, so a set of size one has no term at all. This is not a
     tuned behaviour that training might erode: there is no parameter that could make
     a single move. It matters because the single block is where learned models lose -
     over 5 folds of Norman's combination holdout, ridge scores 1.416 against scDFM's
     1.619 and scPKFM's 1.754, so both published models are worse there than a
     closed-form fit, and this one cannot be.

WHAT IS LEARNED AND WHAT IS NOT. w_a comes from a ridge over the TRAINING conditions
and is a buffer, not a parameter: it is deterministic, so it contributes no seed noise,
and scPKFM's seed noise (0.28 L2 on combosciplex, measured from one replicate) exceeded
every effect it spent three weeks trying to detect. Phi is fixed from KEGG. What is
learned is A_a, W, and the head's gate - about 115 K parameters on combosciplex against
scPKFM's 5.2 M encoder alone.

WHY THE RESIDUAL IS THE WHOLE JOB. L2(ridge) = ||r_S|| identically, since ridge
predicts m_ctrl + sum_a w_a and its error IS the residual it leaves. Table 3's 1.8577
is therefore that residual's norm, and the floor here is only its unreachable part,
0.7734, against scDFM's 1.6567. So the requirement is exact: capture 24.8 % of the
REACHABLE residual energy and Table 3 passes scDFM. On Table 1 the requirement is
negative - ridge already passes - so there the job is not to make it worse, which
property 2 guarantees for singles and nothing guarantees for combinations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .heads import build_head
from .operator import KoopmanOperators
from .readout import Readout


class PathwayKoopmanResidual(nn.Module):
    """The model. `observables` is fixed; `additive` is a buffer, not a parameter."""

    def __init__(self, config: dict, observables, n_perturbations: int,
                 additive: np.ndarray | None = None,
                 detection: torch.Tensor | None = None,
                 dispersion: torch.Tensor | None = None):
        super().__init__()
        model_cfg = config["model"]
        self.observables = observables
        self.n_genes = len(observables.gene_names)
        self.n_perturbations = int(n_perturbations)
        self.additive_kind = model_cfg["additive"]

        # w_a, [P, G]. A BUFFER: it travels in the checkpoint so a scored run uses the
        # same fit it trained against, and it takes no gradient. additive=none is the
        # control arm - the model must then produce the whole displacement, which is
        # what scPKFM did, starting from zero with 88 % of the signal still to find.
        if self.additive_kind == "ridge":
            if additive is None:
                raise ValueError(
                    "model.additive=ridge needs the fitted w_a. Pass "
                    "baselines.fit_ridge_additive(...)['w'] reordered to "
                    "data.perturbations; it is the model's starting point, not an "
                    "evaluation baseline, and a run without it starts at zero.")
            weights = torch.as_tensor(np.asarray(additive), dtype=torch.float32)
            if tuple(weights.shape) != (self.n_perturbations, self.n_genes):
                raise ValueError(
                    f"additive is {tuple(weights.shape)}, expected "
                    f"({self.n_perturbations}, {self.n_genes})")
        elif self.additive_kind == "none":
            weights = torch.zeros(self.n_perturbations, self.n_genes)
        else:
            raise ValueError(f"unknown model.additive {self.additive_kind!r} "
                             f"(ridge | none)")
        self.register_buffer("additive_weights", weights, persistent=True)

        self.operators = KoopmanOperators(config, n_perturbations, observables.dim)
        self.readout = Readout(config, observables, self.n_genes)
        self.head = build_head(config, self.n_genes, detection=detection,
                               dispersion=dispersion)

    # ------------------------------------------------------------------ pieces

    def additive(self, perturbations: list[int]) -> torch.Tensor:
        """sum_a w_a, [G]. Zero for the control."""
        if not perturbations:
            return torch.zeros(self.n_genes, device=self.additive_weights.device,
                               dtype=self.additive_weights.dtype)
        return self.additive_weights[perturbations].sum(dim=0)

    def source_observables(self, x: torch.Tensor,
                           perturbations: list[int]) -> torch.Tensor:
        """Phi(x + sum_a w_a), [B, K]: the state AFTER the additive move.

        The operator acts here, not on the raw control cell, and the two are not
        interchangeable. With the raw cell the operator would be asked to produce the
        WHOLE observable displacement, additive part included, while W is asked to read
        out only the residual - two different quantities, so the flow-matching target
        and the readout's job would disagree.

        An earlier version of this file argued the opposite, that shifting the source
        would make the residual depend on the additive component's output and repeat
        scPKFM's coupling between its operators and rho. That argument does not apply:
        the additive component is a BUFFER, fitted in closed form and never updated, so
        there is nothing for the residual to compete with. scPKFM's failure needed two
        LEARNED terms able to explain the same displacement.

        It also shortens the transport the coupling has to estimate, which is the other
        half of why the OT plan should be better conditioned here.
        """
        return self.observables(x + self.additive(perturbations))

    def observable_displacement(self, x: torch.Tensor,
                                perturbations: list[int]) -> torch.Tensor:
        """(exp(B_S) - I) Phi(x + sum_a w_a), [B, K]. Zero unless |S| >= 2."""
        p0 = self.source_observables(x, perturbations)
        if len(perturbations) < 2:
            return torch.zeros_like(p0)
        return self.operators.flow(p0, perturbations) - p0

    def residual(self, x: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """W (exp(B_S) - I) Phi(x), [B, G]."""
        return self.readout(self.observable_displacement(x, perturbations))

    def displacement(self, x: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """The whole Delta x, [B, G]. At W = 0 this is exactly sum_a w_a."""
        return self.additive(perturbations) + self.residual(x, perturbations)

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor, perturbations: list[int]) -> dict[str, torch.Tensor]:
        """Head parameters for the predicted population. `mean` is the prediction."""
        return self.head(x + self.displacement(x, perturbations))

    def predict(self, x: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """Realised cells, [B, G]. soft returns the mean exactly; sample is unbiased."""
        return self.head.point_estimate(self(x, perturbations))

    # ------------------------------------------------------------- bookkeeping

    def learned_parameters(self) -> dict[str, int]:
        """What is actually fitted, split so the paper's budget table is not guessed.

        additive_weights is excluded because it is a buffer - the ridge fit is closed
        form and deterministic. Reporting it as model capacity would overstate what the
        data has to determine by a factor of several.
        """
        groups = {"operators": self.operators, "readout": self.readout, "head": self.head}
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in groups.items()}
        counts["total"] = sum(counts.values())
        counts["additive_buffer_not_learned"] = int(self.additive_weights.numel())
        return counts

    def extra_repr(self) -> str:
        return (f"additive={self.additive_kind}, n_genes={self.n_genes}, "
                f"n_perturbations={self.n_perturbations}, K={self.observables.dim}")
