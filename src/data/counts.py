"""Exact recovery of raw counts from the log1p-normalised matrix.

The source file keeps no count layer, but the normalisation is invertible. If
    x = log1p(count / library * S)
then expm1(x) is proportional to count, and the smallest non-zero entry of a cell
corresponds to a count of exactly 1. Dividing by it therefore recovers integers:

    counts = expm1(x) / min_nonzero(expm1(x))

Verified on this dataset: 100 % of entries land on integers, and the row sums
reproduce obs['total_count'] exactly (12550, 13474, 18097 on cells 0/1000/5000).

Only the ZINB head needs this. The default path stays in log1p space because that
is where every metric and baseline is defined, and a count round-trip introduces a
Jensen gap - E[log1p(count)] is not log1p(E[count]) - in exactly the quantity
being measured.
"""

from __future__ import annotations

import numpy as np
import torch


def recover_counts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (counts, library_size) for a batch of log1p rows."""
    expanded = torch.expm1(x)
    positive = torch.where(expanded > 0, expanded, torch.full_like(expanded, float("inf")))
    smallest = positive.min(dim=1, keepdim=True).values
    # A cell with no detected gene cannot be scaled; leave it at zero rather than
    # dividing by inf and producing NaNs downstream.
    smallest = torch.where(torch.isfinite(smallest), smallest, torch.ones_like(smallest))
    counts = torch.round(expanded / smallest)
    return counts, counts.sum(dim=1, keepdim=True).clamp(min=1.0)


def verify(x: np.ndarray, tolerance: float = 1e-2) -> dict[str, float]:
    """Fraction of entries that land on integers - a guard for new datasets.

    The recovery is only valid when the normalisation really was
    log1p(count / library * S); on a dataset preprocessed some other way this
    fraction drops and the ZINB head must not be used.
    """
    counts, library = recover_counts(torch.as_tensor(x, dtype=torch.float64))
    counts = counts.numpy()
    expanded = np.expm1(np.asarray(x, dtype=np.float64))
    positive = np.where(expanded > 0, expanded, np.inf)
    smallest = positive.min(axis=1, keepdims=True)
    raw = np.divide(expanded, smallest, out=np.zeros_like(expanded),
                    where=np.isfinite(smallest))
    return {
        "integer_fraction": float(np.mean(np.abs(raw - np.round(raw)) < tolerance)),
        "max_count": float(counts.max()),
        "median_library": float(np.median(library.numpy())),
    }
