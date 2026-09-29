"""Which conditions to score, and on which genes. No model, no torch.

Both functions used to live in scPKFM's src/eval/diagnostics.py, which also held
the transport diagnostics and therefore imported the model. They depend on nothing
but the data, the condition means and the fold, so they are separated here: the
baselines and the reported gene space have to be computable before any model
exists, and on day one of this repository none does.
"""

from __future__ import annotations

import numpy as np

from .baselines import training_conditions


def condition_groups(data, stats, fold, method: str) -> dict[str, list[str]]:
    """The four populations worth reporting separately.

    Splitting train from test is the whole point: a model that reproduces the
    training conditions but not the held-out ones has a generalisation problem,
    while one that fails on both is underfitted, and the fix differs.

    "test singles" is empty under the additive split, where every held-out
    condition is a double. Under the combination holdout the singles of every
    held-out perturbation are held out too, and the literature reports them as
    their own block - scoring the two together averages a harder population with
    an easier one and the number belongs to neither. On combosciplex's scDFM split
    two of the seven test conditions are singles, and Table 3's L2 is the mean over
    all seven.
    """
    seen = training_conditions(stats, fold, method)
    naming = data.naming
    return {
        "train singles": [c for c in seen if naming.is_single(c) and stats.has(c)],
        "train doubles": [c for c in seen if naming.is_double(c) and stats.has(c)],
        "test doubles": [c for c in fold["test"]
                         if naming.is_double(c) and stats.has(c)],
        "test singles": [c for c in fold["test"]
                         if naming.is_single(c) and stats.has(c)],
    }


def scdfm_eval_genes(data, fold: dict, n_top: int) -> np.ndarray:
    """The gene subset scDFM reports on, reproduced from its src/data_process/data.py.

    Their pipeline runs scanpy's HVG TWICE: once over the whole dataset to pick the
    modelled space (`n_top_genes`, 5000 in their run.sh) and again over the TEST
    subset alone (`infer_top_gene`, 1000), and the reported table is scored on the
    second. Without this our numbers sit on raw-variance genes and theirs on
    dispersion-binned ones, which is most of the apparent gap: measured on our
    cells, the same conditions give Control L2 5.69 under raw variance and 3.51
    under dispersion, against their 3.99.

    scanpy is called rather than reimplemented. The seurat flavour bins genes by
    mean expression and z-scores dispersion WITHIN each bin, so a plain
    variance-to-mean ratio selects a different set.

    Their selection reads the test cells, so the evaluation gene set depends on the
    test distribution. That is their protocol; reproducing it is the point.
    """
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    wanted = set(fold["test"]) | {data.control_condition}
    rows = np.concatenate([data.rows[c] for c in sorted(wanted) if c in data.rows])
    subset = ad.AnnData(X=np.asarray(data.x[rows], dtype=np.float32),
                        var=pd.DataFrame(index=pd.Index(data.gene_names)))
    sc.pp.highly_variable_genes(subset, n_top_genes=min(n_top, subset.n_vars))
    return np.flatnonzero(subset.var["highly_variable"].to_numpy())
