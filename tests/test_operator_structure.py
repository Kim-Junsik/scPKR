"""The mechanism-clustering statistic must find planted structure and ignore noise.

This is a test of a STATISTICAL PROCEDURE, and it exists because the claim it supports -
that the learned generators recovered pharmacology the model was never shown - is exactly
the kind of claim a broken test would confirm for free. Two earlier statistical tests in
this repository were themselves wrong, so the null case below matters more than the
positive one.

The shared-component correction is the part most likely to be silently wrong: A_a shares
U and V across every perturbation, so operators are similar before any data is seen and a
raw similarity reports that as a discovery.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                "scripts")))

import dev_operator_structure as structure


def operators(planted: float, seed: int, dim: int = 40, per_class: int = 4,
              n_classes: int = 4):
    """Operators with a shared component, an optional per-class component, and noise."""
    rng = np.random.default_rng(seed)
    classes = {f"d{i}": f"c{i // per_class}" for i in range(per_class * n_classes)}
    shared = rng.standard_normal((dim, dim))
    centre = {c: rng.standard_normal((dim, dim)) for c in set(classes.values())}
    out = {d: shared + planted * centre[c] + rng.standard_normal((dim, dim))
           for d, c in classes.items()}
    return out, np.array([classes[d] for d in sorted(out)])


def statistic(ops, labels, centre: bool, seed: int = 1):
    _, matrix = structure.similarity(ops, centre)
    return structure.permutation_p(matrix, labels, np.random.default_rng(seed))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_class_structure_is_not_detected(seed):
    """The load-bearing one. Operators with no per-class component must not cluster.

    One-sided at 0.05 over three seeds; a procedure that reported structure here would
    have confirmed the paper's claim from a dataset that contains none of it.
    """
    ops, labels = operators(planted=0.0, seed=seed)
    for centre in (False, True):
        _, _, p = statistic(ops, labels, centre)
        assert p > 0.05, f"centre={centre} found structure in noise at p={p:.4f}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_planted_class_structure_is_detected(seed):
    ops, labels = operators(planted=0.6, seed=seed)
    for centre in (False, True):
        gap, null, p = statistic(ops, labels, centre)
        assert p < 0.01, f"centre={centre} missed planted structure at p={p:.4f}"
        assert gap > abs(null) + 0.02


def test_removing_the_shared_component_sharpens_a_real_signal():
    """Subtracting the mean operator must INCREASE the gap when the structure is real.

    A_a = U diag(c_a) V + P_a Q_a with U, V common to every perturbation, so the shared
    term is identical across drugs and dilutes any per-class similarity. If centring did
    not sharpen a planted signal the correction would be doing something other than what
    it claims, and the number reported for the real runs would not mean what it says.
    """
    ops, labels = operators(planted=0.6, seed=0)
    raw, _, _ = statistic(ops, labels, centre=False)
    centred, _, _ = statistic(ops, labels, centre=True)
    assert centred > raw * 1.5
