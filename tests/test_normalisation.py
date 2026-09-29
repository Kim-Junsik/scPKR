"""normalise_from_counts must reproduce scanpy's normalize_total + log1p exactly,
and must leave a dataset that does not set it (Norman) reading its shipped X.

    python -m pytest tests/test_normalisation.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import preprocess


def test_off_by_default_returns_the_shipped_file():
    config = config_module.load()
    assert config["data"]["normalise_from_counts"] is None
    assert preprocess.normalised_source(config) == config["data"]["raw_h5ad"]


def test_counts_layer_is_normalised_to_the_median_library(tmp_path):
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from scipy import sparse

    rng = np.random.default_rng(0)
    dense = rng.integers(0, 6, size=(40, 30)).astype(np.float32)
    dense[:, 0] += 1  # no empty cell, so the median is over every row
    counts = sparse.csr_matrix(dense)
    obs = pd.DataFrame({"condition": ["control+control"] * 20 + ["control+DrugA"] * 20},
                       index=[f"cell{i}" for i in range(40)])
    # The shipped X deliberately uses a DIFFERENT target, as combosciplex does.
    shipped = np.log1p(dense / dense.sum(1, keepdims=True) * 1e4)
    raw = ad.AnnData(X=sparse.csc_matrix(shipped), obs=obs,
                     var=pd.DataFrame(index=[f"g{j}" for j in range(30)]),
                     layers={"counts": counts})
    path = os.path.join(str(tmp_path), "toy.h5ad")
    raw.write_h5ad(path)

    config = config_module.load([f"data.raw_h5ad={path}",
                                 "data.normalise_from_counts=counts"])
    out = preprocess.normalised_source(config)
    assert out != path and os.path.exists(out)

    got = ad.read_h5ad(out)
    expected = ad.AnnData(X=counts.copy())
    sc.pp.normalize_total(expected)
    sc.pp.log1p(expected)
    np.testing.assert_allclose(got.X.toarray(), expected.X.toarray(), rtol=1e-5)
    np.testing.assert_allclose(np.expm1(got.X.toarray()).sum(1),
                               np.median(dense.sum(1)), rtol=1e-4)
    assert list(got.obs["condition"]) == list(obs["condition"])
    assert list(got.obs_names) == list(obs.index)
    # A second call reuses the copy instead of rebuilding it.
    assert preprocess.normalised_source(config) == out
