"""Condition-indexed access to the modelled-space cells.

Control and perturbed cells are UNPAIRED - no cell was measured in both states -
so training never assumes a correspondence. Batches deliver two independent draws
(control cells, perturbed cells of one condition) and the coupling step decides
which control cell transports to which perturbed cell.
"""

from __future__ import annotations

import numpy as np
import anndata as ad

from .conventions import DEFAULT, ConditionNaming

CONTROL = DEFAULT.control  # legacy alias; prefer data.naming.control


def condition_genes(condition: str, naming: ConditionNaming = DEFAULT) -> list[str]:
    return naming.genes(condition)


class PerturbationData:
    """Cells grouped by condition, plus the perturbation vocabulary."""

    def __init__(self, cache_path: str, dtype=np.float32,
                 naming: ConditionNaming = DEFAULT):
        self.naming = naming
        adata = ad.read_h5ad(cache_path)
        self.x = np.asarray(adata.X.todense(), dtype=dtype)
        self.gene_names = adata.var_names.to_numpy()
        conditions = adata.obs["condition"].astype(str).to_numpy()

        self.rows: dict[str, np.ndarray] = {
            condition: np.flatnonzero(conditions == condition)
            for condition in np.unique(conditions)
        }
        self.conditions = sorted(self.rows)
        control = next((c for c in self.conditions if naming.is_control(c)), None)
        if control is None:
            raise ValueError(
                f"no control condition found; expected {naming.control!r} - set "
                f"data.control_label if this dataset names it differently")
        self.control_condition = control
        self.control_rows = self.rows[control]

        self.perturbations = sorted(
            {g for c in self.conditions for g in naming.genes(c)})
        self.pert_index = {p: i for i, p in enumerate(self.perturbations)}

    @property
    def n_genes(self) -> int:
        return self.x.shape[1]

    @property
    def n_perturbations(self) -> int:
        return len(self.perturbations)

    def cells(self, condition: str) -> np.ndarray:
        return self.x[self.rows[condition]]

    def sample(self, condition: str, n: int, rng: np.random.Generator) -> np.ndarray:
        rows = self.rows[condition]
        # Always n rows. Conditions hold 46-1,005 cells, so a batch larger than a
        # small condition has to resample rather than silently shrink the batch.
        pick = rng.choice(rows, size=n, replace=len(rows) < n)
        return self.x[pick]

    def sample_control(self, n: int, rng: np.random.Generator) -> np.ndarray:
        pick = rng.choice(self.control_rows, size=n, replace=len(self.control_rows) < n)
        return self.x[pick]

    def encode_condition(self, condition: str) -> np.ndarray:
        """Multi-hot over the perturbation vocabulary. Control is the all-zero row,
        which is what makes v(z, t, empty) = 0 a structural rather than learned fact."""
        vector = np.zeros(self.n_perturbations, dtype=np.float32)
        for gene in self.naming.genes(condition):
            vector[self.pert_index[gene]] = 1.0
        return vector


class ConditionSampler:
    """Yields (control batch, perturbed batch, condition multi-hot) tuples.

    `singles_only` supports the warm-up phase: the generators u_a are identifiable
    from single perturbations alone, so letting them settle before combinations
    join keeps the interaction term from absorbing single-perturbation error.
    """

    def __init__(self, data: PerturbationData, train_conditions: list[str],
                 batch_size: int, rng: np.random.Generator,
                 anchor: dict[str, np.ndarray] | None = None):
        self.data = data
        self.batch_size = batch_size
        self.rng = rng
        # condition -> gene-space shift for the SOURCE of that condition's batch.
        # Empty for the unanchored model; see eval.baselines.anchor_deltas.
        self.anchor = anchor or {}
        naming = data.naming
        usable = [c for c in train_conditions
                  if c in data.rows and not naming.is_control(c)]
        self.singles = [c for c in usable if naming.is_single(c)]
        self.doubles = [c for c in usable if naming.is_double(c)]

    def epoch(self, singles_only: bool = False) -> list[str]:
        pool = self.singles if singles_only else self.singles + self.doubles
        order = self.rng.permutation(len(pool))
        return [pool[i] for i in order]

    def batch(self, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self.data.sample_control(self.batch_size, self.rng)
        shift = self.anchor.get(condition)
        if shift is not None:
            # The additive part of this combination, handed over rather than
            # learned. The coupling then matches an already-shifted population to
            # the target, so the OT problem the field has to solve is the residual
            # displacement and not the whole one.
            source = source + shift.astype(source.dtype, copy=False)
        target = self.data.sample(condition, self.batch_size, self.rng)
        return source, target, self.data.encode_condition(condition)
