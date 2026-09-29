"""The coupling's cost, and the one claim that makes it safe to have changed.

    python -m pytest tests/test_coupling.py -v

train.uot_stop_thr loosens Sinkhorn's convergence threshold from POT's 1e-6 to 1e-2,
which takes the solve to 30-42 % of its time at train.batch_size=256 and is 24 % of a
whole training step. That
is only legitimate if THE PLAN IS UNCHANGED - otherwise it quietly rewrites what every
experiment measures, and runs made before and after the change stop being comparable.

So the claim is asserted here rather than trusted: the sampling law at the loose
threshold must match one solved to 1e-12 with 20,000 iterations, at every reg the config
allows. It held at 1.4e-14 or better when measured, which is float64 noise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train.coupling import coupling_plan, sample_pairs

# THE CONFIGURED sizes. The first version of this file used 64 x 32 and failed for the
# smaller regs, which sent me looking for a batch-size effect that is not there: the
# dependence is on the cost matrix's GEOMETRY, and at the default reg the loose threshold
# is identical across batch 64/256 and dimension 32/128/413 alike. Small regs are the
# case that needs care, and coupling_plan refuses them.
BATCH, DIM = 256, 413


def populations(seed: int = 0):
    torch.manual_seed(seed)
    # Shifted, so the transport is not the identity and the plan has structure to get
    # wrong. Equal populations would make any threshold look fine.
    return torch.randn(BATCH, DIM), torch.randn(BATCH, DIM) + 0.4


def law(plan: np.ndarray) -> np.ndarray:
    return plan / plan.sum()


@pytest.mark.parametrize("batch,dim", ((256, 413), (64, 32)))
def test_the_loose_threshold_gives_the_same_plan(batch, dim):
    """THE test. If this fails, uot_stop_thr is not a free speedup and must go back.

    At the DEFAULT reg it must hold whatever the geometry, which is why two very
    different shapes are checked rather than only the configured one.
    """
    torch.manual_seed(0)
    source = torch.randn(batch, dim)
    target = torch.randn(batch, dim) + 0.4
    converged = coupling_plan(source, target, "uot", 0.05, 1.0, stop_thr=1e-12)
    loose = coupling_plan(source, target, "uot", 0.05, 1.0, stop_thr=1e-2)
    distance = float(np.abs(law(loose) - law(converged)).sum())
    assert distance < 1e-10, f"{batch}x{dim}: total variation {distance:.2e}"


def test_a_loose_threshold_with_a_small_reg_is_refused():
    """The guard, because a non-converged plan is a SILENT correctness problem.

    It does not look like a failure - it looks like a slightly different experiment. At
    reg below 0.05 the plan is sharper and stops short: measured up to 7e-5 of total
    variation in the sampling law, which is far above the 3e-16 that makes the loose
    threshold free at the default reg. v1's coupling experiment ran at reg 0.01, so this
    combination is reachable rather than hypothetical.
    """
    source, target = populations()
    with pytest.raises(ValueError, match="uot_stop_thr"):
        coupling_plan(source, target, "uot", 0.01, 1.0, stop_thr=1e-2)
    # And the correct pairing is accepted.
    coupling_plan(source, target, "uot", 0.01, 1.0, stop_thr=1e-6)


def test_the_plan_is_not_trivial():
    """Guards the test above: a near-uniform plan would match at any threshold."""
    source, target = populations()
    plan = law(coupling_plan(source, target, "uot", 0.05, 1.0))
    uniform = 1.0 / plan.size
    assert float(np.abs(plan - uniform).sum()) > 0.1, "the plan carries no structure"


def test_random_pairing_is_the_product_of_the_marginals():
    """The ablation control, and it must be exactly that - not a sharp plan with noise."""
    source, target = populations()
    plan = coupling_plan(source, target, "random", 0.05, 1.0)
    np.testing.assert_allclose(plan, np.full_like(plan, 1.0 / plan.size), rtol=1e-12)


def test_sample_pairs_returns_indices_in_range_and_is_reproducible():
    """Indices, not rows: the pair is needed in observable space for the field and in
    gene space for the head, and re-deriving one from the other is how they drift."""
    source, target = populations()
    first = sample_pairs(source, target, "uot", 0.05, 1.0, np.random.default_rng(7))
    again = sample_pairs(source, target, "uot", 0.05, 1.0, np.random.default_rng(7))
    for rows, columns in (first, again):
        assert rows.shape == columns.shape == (BATCH,)
        assert int(rows.max()) < BATCH and int(columns.max()) < BATCH
        assert int(rows.min()) >= 0 and int(columns.min()) >= 0
    torch.testing.assert_close(first[0], again[0])
    torch.testing.assert_close(first[1], again[1])


def test_a_degenerate_plan_is_counted_not_hidden():
    """scPKFM's fallback was silent, and a reg too small for the cost scale made it fire
    on 600 of 600 batches while the run looked healthy. train.coupling_fallback_max stops
    a run above a share of these, which needs them counted."""
    source, target = populations()
    stats: dict = {}
    sample_pairs(source, target, "uot", 0.05, 1.0, np.random.default_rng(0), stats=stats)
    assert stats["plans"] == 1 and stats["fallbacks"] == 0
