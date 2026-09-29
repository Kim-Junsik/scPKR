"""Chunked reading of the source .h5ad.

norman.h5ad is 2.2 GB on disk, so both passes over it (variance, then subsetting)
stream row blocks straight out of the CSR arrays rather than materialising the
whole matrix. anndata's backed mode would also work but gives less control over
where the memory goes.
"""

from __future__ import annotations

from typing import Iterator

import h5py
import numpy as np
from scipy import sparse


def read_obs_column(path: str, key: str) -> np.ndarray:
    """Read one obs column as a string array, decoding categoricals."""
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        if key == "_index":
            key = _index_key(obs)
        node = obs[key]
        if isinstance(node, h5py.Group):  # categorical: codes + categories
            categories = np.array([c.decode() for c in node["categories"][:]])
            return categories[node["codes"][:]]
        values = node[:]
        # encoding-version 0.1.0 keeps categories in a sibling __categories group
        # and stores only integer codes in the column itself.
        if "__categories" in obs and key in obs["__categories"]:
            categories = np.array([
                c.decode() if isinstance(c, bytes) else str(c)
                for c in obs["__categories"][key][:]])
            return categories[values.astype(int)]
    if values.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])
    return values.astype(str)


def _index_key(group) -> str:
    """Which column holds the index.

    h5ad records it in the group's `_index` attribute, and it is not always
    literally "_index": Norman uses "_index", combosciplex (encoding-version
    0.1.0) uses "Cell" for obs and "gene_short_name" for var. Hardcoding the
    literal name works on one dataset and raises a bare KeyError on the other.
    """
    key = group.attrs.get("_index", "_index")
    return key.decode() if isinstance(key, bytes) else str(key)


def read_index(path: str, group_name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        raw = group[_index_key(group)][:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in raw])


def read_var_names(path: str) -> np.ndarray:
    return read_index(path, "var")


def shape(path: str) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        n_obs, n_var = handle["X"].attrs["shape"]
    return int(n_obs), int(n_var)


def _iter_converted(path: str, chunk_size: int):
    """Load a non-CSR matrix once, convert, then hand out row blocks."""
    import anndata as ad

    matrix = ad.read_h5ad(path).X.tocsr()
    for start in range(0, matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, matrix.shape[0])
        yield start, stop, matrix[start:stop]


def iter_row_chunks(path: str, chunk_size: int) -> Iterator[tuple[int, int, sparse.csr_matrix]]:
    """Yield (start, stop, csr_block) over the rows of X.

    Assumes X is stored CSR, which is what the h5ad encoding-type attribute
    reports for this file; anything else is a hard error rather than a silent
    wrong answer.
    """
    with h5py.File(path, "r") as handle:
        group = handle["X"]
        encoding = group.attrs.get("encoding-type", "")
        if encoding == "csc_matrix":
            # CSC is column-major, so a row block is not a contiguous slice and
            # streaming it row-wise would touch the whole file per chunk. Convert
            # once instead - combosciplex is stored this way.
            yield from _iter_converted(path, chunk_size)
            return
        if encoding != "csr_matrix":
            raise ValueError(
                f"X must be csr_matrix or csc_matrix, found {encoding!r}")
        n_obs, n_var = (int(v) for v in group.attrs["shape"])
        indptr = group["indptr"][:]
        data = group["data"]
        indices = group["indices"]

        for start in range(0, n_obs, chunk_size):
            stop = min(start + chunk_size, n_obs)
            lo, hi = int(indptr[start]), int(indptr[stop])
            block = sparse.csr_matrix(
                (data[lo:hi], indices[lo:hi], indptr[start:stop + 1] - lo),
                shape=(stop - start, n_var),
            )
            yield start, stop, block
