"""The Koopman operators on the fixed observables, and how they compose.

    A_a = U diag(c_a) V + P_a Q_a            one per perturbation, on Phi's space
    B_S = sum_{a<b in S} (A_a A_b + A_b A_a) the composition
    p1  = exp(B_S) p0                        the flow map, in closed form

EVERY OPTION HERE IS LINEAR IN p, so the flow map is exactly a matrix exponential and
this model contains no integrator. scPKFM carried RK4 at 20 steps because its field
was nonlinear (a learned rho over the velocities), which made the only affordable
endpoint loss a single mean point - and its own plan recorded the batch version as the
one untried item with a measured bound of 0.20-0.26. Here it is one torch.matrix_exp
per condition and the whole batch comes along.

WHY THE ANTI-COMMUTATOR. Sum two generators and integrate, and the difference from an
additive flow map is

    exp(t(A+B)) - (exp(tA) + exp(tB) - I) = (t^2/2)(AB + BA) + O(t^3)

so the leading correction is the ANTI-commutator. It is symmetric under exchange,
which is what a SIMULTANEOUS double perturbation is. The commutator [A,B] is
antisymmetric and describes ORDER dependence - A then B against the reverse - for
which a simultaneous perturbation carries no signal at all. Verified numerically
against the real code path: at operator scales 0.05 / 0.02 / 0.01 the anticommutator
model's relative error is 3.5e-2 / 1.4e-2 / 7.0e-3, falling linearly in the scale as an
O(t^3) remainder must, while the cosine against the commutator stays at -0.30.

This is why scPKFM's Lie bracket term lost 5-0 across seven settings and two
backbones, and the prediction that `composition=commutator` does nothing is written
down here BEFORE the run rather than offered afterwards as an explanation.

INITIALISATION - AND THE ZERO IS NOT HERE. A_a starts small and RANDOM. It must:
B_S is second order in A, so dB_S/dA_a is proportional to A_b, and with every operator
at zero the gradient of the loss with respect to every operator is zero and nothing
ever moves. It is the vanishing-product trap one level up from zeroing both factors of
a low-rank product.

What makes the model's initial prediction exactly the additive baseline is the READOUT
being zero (models/readout.py). W is linear, so dL/dW is proportional to B_S Phi(x),
which is non-zero: W moves first and the operators follow once W is non-zero. That is
the zero-initialised-output-layer argument scPKFM used for rho, applied at the layer
where it holds.

WHY SHARED MODES. combosciplex has 17 drugs, 24 training conditions, and 13 drugs that
never appear alone. A full K x K operator at K = 413 is 170,000 parameters fitted from
two or three conditions. Most of each operator therefore comes from modes every
perturbation shares, and a perturbation chooses how much of each it uses; the private
part is small and rank-limited. m = 0 with p = r reduces to a plain per-perturbation
rank-r operator, which tests/test_structure.py asserts as the special case.
"""

from __future__ import annotations

import torch
import torch.nn as nn

COMPOSITIONS = ("anticommutator", "commutator", "bilinear", "sum")


class KoopmanOperators(nn.Module):
    """A_a for every perturbation, and B_S for a set of them."""

    def __init__(self, config: dict, n_perturbations: int, dim: int):
        super().__init__()
        model_cfg = config["model"]
        shared = int(model_cfg["shared_rank"])
        private = int(model_cfg["private_rank"])
        if shared < 0 or private < 0 or shared + private == 0:
            raise ValueError(
                f"need shared_rank + private_rank > 0, both >= 0 (got "
                f"shared_rank={shared}, private_rank={private})")
        kind = model_cfg["composition"]
        if kind not in COMPOSITIONS:
            raise ValueError(f"unknown model.composition {kind!r} "
                             f"({' | '.join(COMPOSITIONS)})")

        self.dim = int(dim)
        self.n_perturbations = int(n_perturbations)
        self.shared_rank, self.private_rank = shared, private
        self.composition = kind
        scale = float(model_cfg["operator_init_scale"])

        if shared:
            self.basis_u = nn.Parameter(torch.randn(dim, shared) * scale)
            self.basis_v = nn.Parameter(torch.randn(shared, dim) * scale)
            # ONE, not zero: every perturbation begins using every shared mode
            # equally, so the basis first learns the response the perturbations have
            # in common and the coefficients differentiate from there. U diag(c) V is
            # invariant to U -> kU, c -> c/k, so read A_a as a whole and never U or
            # c_a alone.
            self.coefficient = nn.Parameter(torch.ones(n_perturbations, shared))
        if private:
            self.private_u = nn.Parameter(
                torch.randn(n_perturbations, dim, private) * scale)
            self.private_v = nn.Parameter(
                torch.randn(n_perturbations, private, dim) * scale)

        if kind == "bilinear":
            # B_S = sum_{a<b} (A_a G A_b + A_b G A_a) with G = I + U_g V_g and U_g at
            # ZERO, so this arm STARTS as the anticommutator exactly. The ablation is
            # nested: it asks whether the anticommutator specifically is right, or
            # whether any symmetric second-order form does as well, from a shared
            # starting point rather than from a different initialisation.
            rank = int(model_cfg["composition_rank"])
            self.gain_u = nn.Parameter(torch.zeros(dim, rank))
            self.gain_v = nn.Parameter(torch.randn(rank, dim) * scale)

    # ---------------------------------------------------------------- operators

    def matrix(self, pert: int) -> torch.Tensor:
        """A_a as a dense [K, K].

        Formed rather than applied through the factors because every consumer needs
        it whole: B_S is a product of two operators, the flow map is its exponential,
        and the spectral figure is its eigendecomposition. At K around 450 this is
        200,000 entries and ~10 MFLOP, which is not where this model spends time.
        """
        out = torch.zeros(self.dim, self.dim, device=self.device, dtype=self.dtype)
        if self.shared_rank:
            out = out + (self.basis_u * self.coefficient[pert]) @ self.basis_v
        if self.private_rank:
            out = out + self.private_u[pert] @ self.private_v[pert]
        return out

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # -------------------------------------------------------------- composition

    def compose(self, perturbations: list[int]) -> torch.Tensor:
        """B_S as a dense [K, K]. Zero for a set with no pair, except `sum`.

        Structural properties, all asserted in tests/test_structure.py:

          - no parameter is indexed by a PAIR of perturbations, only by one. That is
            what makes an unseen combination expressible: there is no parameter that
            only a seen pair could have fitted. scPKFM had a [P, P] pair table once
            and it trained fine while being exactly wrong - a pair never seen kept its
            initial value, so on all 37 evaluation combinations the term was
            numerically zero.
          - the sum over unordered pairs and the symmetry of the anticommutator make
            the result independent of the order of `perturbations`.
          - for a single perturbation there is no pair, so B_S = 0 and the flow map is
            the identity. A single's prediction is therefore the additive component
            exactly, at every point in training. `sum` is the one arm where this does
            not hold, which is part of what that ablation measures.
        """
        if not perturbations:
            return torch.zeros(self.dim, self.dim, device=self.device, dtype=self.dtype)

        operators = [self.matrix(pert) for pert in perturbations]
        if self.composition == "sum":
            total = operators[0]
            for operator in operators[1:]:
                total = total + operator
            return total

        gain = None
        if self.composition == "bilinear":
            eye = torch.eye(self.dim, device=self.device, dtype=self.dtype)
            gain = eye + self.gain_u @ self.gain_v

        total = torch.zeros(self.dim, self.dim, device=self.device, dtype=self.dtype)
        for i in range(len(operators)):
            for j in range(i + 1, len(operators)):
                a, b = operators[i], operators[j]
                if gain is not None:
                    a_b, b_a = a @ gain @ b, b @ gain @ a
                else:
                    a_b, b_a = a @ b, b @ a
                # anticommutator: AB + BA, symmetric, the exchange symmetry a
                # simultaneous double perturbation has.
                # commutator:     AB - BA, antisymmetric, order dependence. Predicted
                #                 to carry no signal here.
                total = total + (a_b - b_a if self.composition == "commutator"
                                 else a_b + b_a)
        return total

    # --------------------------------------------------------------- the flow

    def velocity(self, p: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """v(p) = B_S p. AUTONOMOUS: no t.

        Flow matching is unchanged by that - same interpolant, same coupling - but the
        same operator has to work at every point on the path, which is what makes this
        an ODE rather than an arbitrary field. scPKFM's s(t) MLP was a scalar
        reparameterisation of a scale the operator already carried, so there is none
        here; the only thing feeding a time into the field would be a way to fit the
        interpolant instead of the dynamics.
        """
        return p @ self.compose(perturbations).T

    def flow(self, p: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """p(1) = exp(B_S) p(0). EXACT, and over the whole batch.

        One matrix exponential per condition, shared by every cell in the batch. That
        is the whole reason the endpoint loss can be a batch statement here: with a
        nonlinear field it cost n_steps * 4 field evaluations per cell, which is why
        scPKFM supervised a single mean point and measured the mismatch
        (Phi(mean) != mean(Phi), and the conditions it supervised most directly came
        out worst).
        """
        if not perturbations:
            return p
        return p @ torch.matrix_exp(self.compose(perturbations)).T

    # ------------------------------------------------------------- the figure

    @torch.no_grad()
    def spectrum(self, pert: int) -> torch.Tensor:
        """Eigenvalues of A_a. THE INTERPRETATION, and a stability check.

        A fixed observable dictionary is what makes this meaningful: A_a approximates
        the Koopman generator on a KEGG-indexed space, so its eigenvectors are pathway
        modes and its eigenvalues their rates. scPKFM could not make this statement -
        its latent was trained for reconstruction and frozen, so its operator acted on
        coordinates that meant nothing in particular, and it computed no spectrum at
        all in 500 lines of generator code.

        Also a guard: exp(B_S) grows like exp(Re lambda_max), so a large positive
        spectral abscissa turns a modest operator into an enormous displacement.
        """
        return torch.linalg.eigvals(self.matrix(pert))

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, n_perturbations={self.n_perturbations}, "
                f"shared_rank={self.shared_rank}, private_rank={self.private_rank}, "
                f"composition={self.composition}")
