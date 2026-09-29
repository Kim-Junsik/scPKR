"""The claims that must hold without training, asserted as numbers.

    python -m pytest tests/test_structure.py -v

These are not accuracy tests. Each one checks a property the construction is supposed
to guarantee, so a failure means the model is wrong rather than under-trained. They run
on a synthetic observable space and need no data, so there is no excuse for skipping
them.

THE FIRST GROUP IS A REGRESSION TEST FOR A BUG THAT ALREADY HAPPENED. The first version
of MeanPreservingHurdleHead clamped the predicted mean at zero inside forward. Every
structural test below still passed - they are about the displacement - while the
premise of the whole design silently broke: on combosciplex the untrained model scored
2.7318 where ridge_additive scores 1.8577. Non-negativity is a property of a REALISED
CELL, not of a conditional mean, and with 41 % of control entries at exactly zero,
E[max(0, x + w)] != max(0, E[x] + w) on about half the genes.

So the fixture below deliberately reproduces those two conditions - control cells with
many exact zeros, and additive weights with negative entries - and the test fails if
the clamp ever comes back.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.models.model import PathwayKoopmanResidual
from src.models.operator import KoopmanOperators

N_GENES = 40
N_PERTURBATIONS = 5          # not a square, so a [P, P] table cannot hide as [K, K]
N_PATHWAYS = 6
N_ANCHORS = 3
N_CELLS = 64
ZERO_FRACTION = 0.41         # the measured share of exact zeros in the real data


class FakeObservables:
    """Enough of models.observables.Observables to build the model, without KEGG.

    The matrix is genuinely sparse and leaves some genes untouched, because that is the
    real situation - KEGG reaches a quarter of combosciplex's genes - and a test on a
    dense scaffold would not notice a readout that ignored the support.
    """

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        matrix = np.zeros((N_PATHWAYS + N_ANCHORS, N_GENES), dtype=np.float32)
        for row in range(N_PATHWAYS):                      # pathway rows: pooled means
            members = rng.choice(N_GENES // 2, size=4, replace=False)
            matrix[row, members] = 1.0 / len(members)
        for j in range(N_ANCHORS):                         # anchor rows: single genes
            matrix[N_PATHWAYS + j, N_GENES // 2 + j] = 1.0
        self.matrix = torch.from_numpy(matrix)
        self.n_pathways, self.n_anchors = N_PATHWAYS, N_ANCHORS
        self.gene_names = np.array([f"g{i}" for i in range(N_GENES)])
        self.mean = torch.zeros(self.dim)
        self.std = torch.ones(self.dim)

    @property
    def dim(self) -> int:
        return self.n_pathways + self.n_anchors

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.matrix.T - self.mean) / self.std

    def edges(self):
        observable, gene = torch.nonzero(self.matrix, as_tuple=True)
        return gene, observable


def control_cells(seed: int = 1) -> torch.Tensor:
    """Cells with ZERO_FRACTION exact zeros, like the real data."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 0.6, size=(N_CELLS, N_GENES)).astype(np.float32)
    x[rng.random(x.shape) < ZERO_FRACTION] = 0.0
    return torch.from_numpy(x)


def additive_weights(seed: int = 2) -> np.ndarray:
    """w_a with NEGATIVE entries, like a real ridge fit. Half the genes go down."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.4, size=(N_PERTURBATIONS, N_GENES)).astype(np.float32)


def build(**overrides) -> PathwayKoopmanResidual:
    config = config_module.load([f"model.{k}={v}" for k, v in overrides.items()])
    torch.manual_seed(0)
    model = PathwayKoopmanResidual(config, FakeObservables(), N_PERTURBATIONS,
                                   additive_weights())
    return model.eval()


def fill_readout(model: PathwayKoopmanResidual, scale: float = 0.3) -> None:
    """W starts at zero, which would make most tests below pass for free. Break it."""
    with torch.no_grad():
        if model.readout.kind == "dense":
            model.readout.weight.normal_(0.0, scale)
        else:
            model.readout.values.normal_(0.0, scale)


# ==========================================================================
# THE PREMISE: an untrained model is the additive baseline, exactly
# ==========================================================================

def test_an_untrained_model_predicts_the_additive_baseline_exactly():
    """THE test. If this fails, nothing else about the design matters.

    W starts at zero, so the residual is identically zero, and the head is
    mean-preserving, so the prediction's mean must be the control mean plus sum_a w_a -
    which is what ridge_additive predicts. Verified against the real number too: on
    combosciplex the untrained model scores 1.857668 where scripts/baseline_l2.py
    reports 1.8577 for ridge_additive, matching to 6.4e-06 over every gene.
    """
    model = build(hurdle_gate="soft")
    x, w = control_cells(), additive_weights()
    for perturbations in ([0], [1, 3], [0, 2, 4]):
        with torch.no_grad():
            predicted = model.predict(x, perturbations).mean(dim=0).numpy()
        expected = x.mean(dim=0).numpy() + w[perturbations].sum(axis=0)
        np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-5)


def test_the_head_does_not_move_a_negative_mean():
    """The clamp regression, isolated to the head.

    A predicted mean can be negative - x is zero on 41 % of entries and w_a is negative
    on half the genes - and clamping it per cell is what broke the premise. Under the
    soft gate the head must return the mean it was handed, sign and all.
    """
    model = build(hurdle_gate="soft")
    mean = torch.linspace(-2.0, 2.0, N_GENES).expand(8, N_GENES).contiguous()
    with torch.no_grad():
        estimate = model.head.point_estimate(model.head(mean))
    torch.testing.assert_close(estimate, mean)
    assert (mean < 0).any(), "the fixture must include negative means to be a test"


def test_the_sampled_gate_is_unbiased_for_the_mean():
    """sample must reproduce the mean IN EXPECTATION, and emit real zeros doing it.

    Tested by POOLING rather than per entry. A draw is Bernoulli(q) * (mu / q), so its
    variance is mu^2 (1 - q) / q and a gene detected in 5 % of cells has a standard
    deviation of 4.4 mu - the price of emitting exact zeros, and the reason this is a
    statement about an expectation and not about any single draw. With 10,240 entries, a
    per-entry 4-sigma bound fails by chance about 65 % of the time, which is a property
    of the test and not of the head; the first version of this test did exactly that.

    So two statistically sound claims instead:
      - the POOLED deviation is within 4 standard errors of zero. This is the
        unbiasedness claim, and pooling makes it far sharper than any per-entry bound.
      - the deviations, divided by their own predicted standard errors, have RMS near 1.
        This checks the variance formula as well, so a head that hit the mean by
        accident while being wildly over- or under-dispersed would fail.
    """
    model = build(hurdle_gate="sample", hurdle_magnitude="point")
    mean = torch.rand(256, N_GENES) * 2.0
    draws_n = 600
    with torch.no_grad():
        q = model.head(mean)["q"]
        draws = torch.stack([model.head.point_estimate(model.head(mean))
                             for _ in range(draws_n)])
    standard_error = (mean.abs() * torch.sqrt((1.0 - q) / q)) / draws_n ** 0.5
    deviation = draws.mean(dim=0) - mean

    pooled = float(deviation.sum())
    pooled_error = float(standard_error.square().sum().sqrt())
    assert abs(pooled) <= 4.0 * pooled_error, (
        f"pooled bias {pooled:.4f} against 4 SE {4 * pooled_error:.4f}")

    normalised = float((deviation / standard_error.clamp(min=1e-9)).square().mean().sqrt())
    assert 0.6 < normalised < 1.6, f"deviation/SE has RMS {normalised:.3f}, expected ~1"
    assert (draws == 0).float().mean() > 0.05, "sample emitted no exact zeros"


# ==========================================================================
# The residual's structure
# ==========================================================================

def test_the_residual_is_identically_zero_at_initialisation():
    model = build()
    assert float(model.residual(control_cells(), [1, 3]).abs().max()) == 0.0


def test_a_single_perturbation_has_no_residual_term_even_once_trained():
    """Structural, not learned: B_S sums over unordered PAIRS, so |S| = 1 has no term.

    This is why the single block cannot be damaged. Over 5 folds of Norman's
    combination holdout ridge scores 1.416 against scDFM's 1.619 and scPKFM's 1.754, so
    both published models are worse there than a closed-form fit, and no parameter here
    could make a single move away from it.
    """
    model = build()
    fill_readout(model)
    x = control_cells()
    for pert in range(N_PERTURBATIONS):
        assert float(model.observable_displacement(x, [pert]).abs().max()) == 0.0
        assert float(model.residual(x, [pert]).abs().max()) == 0.0
        torch.testing.assert_close(model.displacement(x, [pert]),
                                   model.additive([pert]).expand(x.shape[0], N_GENES))


def test_the_control_has_no_displacement():
    model = build()
    fill_readout(model)
    x = control_cells()
    assert float(model.displacement(x, []).abs().max()) == 0.0


def test_order_does_not_change_the_prediction():
    model = build(hurdle_gate="soft")
    fill_readout(model)
    x = control_cells()
    with torch.no_grad():
        torch.testing.assert_close(model.predict(x, [1, 3]), model.predict(x, [3, 1]))
        torch.testing.assert_close(model.predict(x, [0, 2, 4]),
                                   model.predict(x, [4, 0, 2]))


def test_no_parameter_is_indexed_by_a_pair_of_perturbations():
    """What makes an unseen combination expressible at all.

    scPKFM stored a [P, P] table for its pair term once. It trained fine while being
    exactly wrong: a pair never seen kept its initial value, so on all 37 evaluation
    combinations the term was numerically zero and the model was identical to one
    without it.
    """
    for composition in ("anticommutator", "commutator", "bilinear", "sum"):
        model = build(composition=composition)
        for name, parameter in model.named_parameters():
            shape = tuple(parameter.shape)
            assert N_PERTURBATIONS * N_PERTURBATIONS not in shape, f"{name} {shape}"
            assert sum(1 for s in shape if s == N_PERTURBATIONS) <= 1, f"{name} {shape}"


def test_the_additive_weights_are_a_buffer_and_take_no_gradient():
    """The ridge fit is closed form. Making it a parameter would hand the optimiser the
    88 % of the signal it is there to remove from the problem."""
    model = build()
    names = {name for name, _ in model.named_parameters()}
    assert not any("additive" in name for name in names), names
    assert "additive_weights" in dict(model.named_buffers())
    assert not model.additive_weights.requires_grad


# ==========================================================================
# Initialisation: where the zero goes, and that gradients survive it
# ==========================================================================

def test_the_operators_do_not_start_at_zero():
    """They must not. B_S is second order in A, so dB_S/dA_a is proportional to A_b and
    an all-zero initialisation kills the gradient of every operator at once - the
    vanishing-product trap one level up from zeroing both factors of a low-rank
    product. config.py's comment says so; this asserts it."""
    model = build()
    for pert in range(N_PERTURBATIONS):
        assert float(model.operators.matrix(pert).abs().max()) > 0.0
    assert float(model.operators.compose([1, 3]).abs().max()) > 0.0


def test_the_gradient_reaches_the_readout_at_step_zero():
    """W is linear, so dL/dW is proportional to the observable displacement and is
    non-zero even though W itself is zero. W moves first."""
    model = build(hurdle_gate="soft")
    x = control_cells()
    model.predict(x, [1, 3]).square().mean().backward()
    assert float(model.readout.values.grad.abs().max()) > 0.0


def test_the_operators_receive_no_gradient_until_the_readout_is_non_zero():
    """The other half of the same argument, and the reason the premise costs nothing:
    at W = 0 the operators are frozen for exactly one step, then follow."""
    model = build(hurdle_gate="soft")
    x = control_cells()
    model.predict(x, [1, 3]).square().mean().backward()
    assert float(model.operators.basis_u.grad.abs().max()) == 0.0

    model.zero_grad(set_to_none=True)
    fill_readout(model)
    model.predict(x, [1, 3]).square().mean().backward()
    assert float(model.operators.basis_u.grad.abs().max()) > 0.0
    assert float(model.operators.private_u.grad.abs().max()) > 0.0


# ==========================================================================
# The operator and its flow
# ==========================================================================

def test_the_flow_map_solves_the_ode_it_claims_to():
    """exp(B_S) p against explicit Euler on dp/dt = B_S p.

    The point of the linear field: the flow map is closed form, so the endpoint loss is
    a batch statement at the cost of one matrix exponential. scPKFM needed RK4 and could
    only afford a single mean point, and measured the mismatch that followed.
    """
    config = config_module.load(["model.shared_rank=4", "model.private_rank=2"])
    torch.manual_seed(0)
    operators = KoopmanOperators(config, N_PERTURBATIONS, 12).double()
    torch.manual_seed(1)
    p = torch.randn(5, 12, dtype=torch.float64)
    matrix = operators.compose([1, 3])
    stepped, steps = p.clone(), 20000
    for _ in range(steps):
        stepped = stepped + (1.0 / steps) * (stepped @ matrix.T)
    torch.testing.assert_close(operators.flow(p, [1, 3]), stepped, rtol=1e-6, atol=1e-6)


def test_the_leading_non_additivity_is_the_anticommutator():
    """exp(A+B) - (exp(A) + exp(B) - I) = (1/2){A, B} + O(t^3).

    The claim, and the reason scPKFM's Lie bracket lost 5-0: the commutator is
    antisymmetric and describes ORDER dependence, which a SIMULTANEOUS double
    perturbation carries no signal for. The relative error must fall as the operator
    scale falls; the cosine against the commutator must not approach 1.
    """
    errors = []
    for scale in (0.05, 0.02, 0.01):
        config = config_module.load(["model.shared_rank=4", "model.private_rank=2",
                                     f"model.operator_init_scale={scale}"])
        torch.manual_seed(0)
        operators = KoopmanOperators(config, N_PERTURBATIONS, 12).double()
        a, b = operators.matrix(1), operators.matrix(3)
        eye = torch.eye(12, dtype=torch.float64)
        difference = (torch.matrix_exp(a + b)
                      - (torch.matrix_exp(a) + torch.matrix_exp(b) - eye))
        anticommutator = 0.5 * (a @ b + b @ a)
        commutator = 0.5 * (a @ b - b @ a)
        errors.append(float((difference - anticommutator).norm() / difference.norm()))
        cosine = torch.nn.functional.cosine_similarity(
            difference.flatten(), commutator.flatten(), dim=0)
        assert abs(float(cosine)) < 0.5, f"scale {scale}: cosine {float(cosine)}"
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 0.02, errors


def test_the_commutator_arm_is_antisymmetric():
    """Which is exactly why it is predicted to do nothing on a symmetric signal."""
    model = build(composition="commutator")
    torch.testing.assert_close(model.operators.compose([1, 3]),
                               -model.operators.compose([3, 1]))


def test_the_bilinear_arm_starts_as_the_anticommutator():
    """A nested ablation: it asks whether the anticommutator specifically is right, or
    whether any symmetric second-order form does as well, from the same start."""
    torch.testing.assert_close(build(composition="bilinear").operators.compose([1, 3]),
                               build().operators.compose([1, 3]))


def test_shared_modes_reduce_to_a_per_perturbation_operator():
    """m = 0 with p = r is a plain rank-r operator per perturbation.

    Asserted on the singular values with an explicit relative threshold rather than
    through matrix_rank's default. The product is formed in float32, so the singular
    values beyond the rank sit at ~1e-7 of the leading one - genuinely zero for a
    float32 product, but far above the float64 tolerance matrix_rank would apply if the
    matrix were cast first, which is how this test failed the first time it ran.
    """
    rank = 3
    model = build(shared_rank=0, private_rank=rank)
    assert model.operators.shared_rank == 0
    assert not hasattr(model.operators, "basis_u")
    values = torch.linalg.svdvals(model.operators.matrix(2).detach())
    assert float(values[rank] / values[0]) < 1e-4, values


def test_the_spectrum_is_computable_and_sized_right():
    """The interpretation figure, and a stability guard: exp(B_S) grows like
    exp(Re lambda_max), so a large positive abscissa turns a modest operator into an
    enormous displacement."""
    model = build()
    eigenvalues = model.operators.spectrum(1)
    assert eigenvalues.shape == (model.observables.dim,)
    assert torch.isfinite(eigenvalues.real).all()


# ==========================================================================
# The readout scaffold
# ==========================================================================

def test_the_residual_never_reaches_a_gene_off_the_scaffold():
    """THE pathway-mediation claim, as a support rather than a penalty - so it is not
    something the optimiser can trade away.

    Genes no observable touches receive the additive component alone. That is the honest
    consequence and it is what puts the floor under the model: 0.7734 on Table 3 against
    scDFM's 1.6567, so the scaffold is not the binding constraint.
    """
    model = build()
    fill_readout(model)
    gene, _ = model.observables.edges()
    reachable = torch.zeros(N_GENES, dtype=torch.bool)
    reachable[torch.unique(gene)] = True
    assert not reachable.all(), "the fixture must leave some genes unreachable"

    residual = model.residual(control_cells(), [1, 3])
    assert float(residual[:, ~reachable].abs().max()) == 0.0
    assert float(residual[:, reachable].abs().max()) > 0.0


def test_the_random_scaffold_keeps_the_edge_count():
    """The ablation has to differ in WHERE the edges are, not how many there are."""
    assert build(readout="random").readout.n_edges == build().readout.n_edges


def test_the_dense_readout_reaches_every_gene():
    """The other scaffold ablation. If dense is not better, the scaffold is
    load-bearing rather than a parameter saving."""
    model = build(readout="dense")
    fill_readout(model)
    residual = model.residual(control_cells(), [1, 3])
    assert int((residual.abs().max(dim=0).values > 0).sum()) == N_GENES


@pytest.mark.parametrize("readout", ("kegg", "random", "dense"))
def test_every_readout_starts_at_zero(readout):
    """Whatever the scaffold, the premise holds: the residual is zero at step 0."""
    model = build(readout=readout)
    assert float(model.residual(control_cells(), [1, 3]).abs().max()) == 0.0
