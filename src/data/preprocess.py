"""Build the modelled-space cache from the raw Norman h5ad.

The only transformation is a gene subset. X in the source file is already log1p
normalised, and no raw-count layer survives, so re-running normalize_total/log1p
would double-normalise data that cannot be recovered.

Gene space = top `n_hvg` by the chosen criterion, unioned with every perturbation
target, kept in adata.var_names order. That order is the single alignment
convention for everything downstream (mask rows, decoder outputs, metrics); there
is no tokenizer and no alphabetical vocab.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from . import io


def perturbation_targets(conditions: np.ndarray, control_label: str = "ctrl") -> list[str]:
    """Genes named in any condition string, e.g. 'AHR+FEV' -> AHR, FEV."""
    targets = set()
    for condition in np.unique(conditions):
        if condition == control_label:
            continue
        for gene in condition.split("+"):
            if gene != control_label:
                targets.add(gene)
    return sorted(targets)


def normalised_source(config: dict) -> str:
    """The file every streaming pass reads: the shipped matrix, or a copy of it
    renormalised from a raw-count layer.

    `data.normalise_from_counts` null (Norman) returns data.raw_h5ad untouched -
    its X is already the log1p matrix the reference pipeline scores on. Set to a
    layer name (combosciplex: "counts"), the counts are normalised the way
    scDFM's combosciplex branch does it,

        adata.X = adata.layers["counts"]; sc.pp.normalize_total(adata); sc.pp.log1p(adata)

    over ALL cells, and the result is written next to the raw file once. This is
    not cosmetic: combosciplex ships an X normalised to 10,000 counts per cell,
    while scanpy's default target is the median library size (2,579 there).
    Measured over scDFM's seven test conditions, the shipped X gives Control L2
    8.2540 against their published 5.3716; the renormalised counts give 5.3260.

    scanpy is called rather than reimplemented, for the same reason as
    scanpy_hvg. The median is taken over every cell, test included, as scDFM
    does; it is a single scalar and carries no condition label.
    """
    import json
    import os

    data_cfg = config["data"]
    layer = data_cfg.get("normalise_from_counts")
    raw = data_cfg["raw_h5ad"]
    if not layer:
        return raw
    stem, _ = os.path.splitext(raw)
    target = f"{stem}_lognorm_{layer}_median.h5ad"
    if os.path.exists(target):
        print(f"  using renormalised copy {target}")
        return target

    import anndata as ad
    import pandas as pd
    import scanpy as sc

    print(f"  renormalising {raw} from layers[{layer!r}] "
          f"(normalize_total at the median library, then log1p) ...")
    source = ad.read_h5ad(raw)
    if layer not in source.layers:
        raise ValueError(f"{raw} has no layers[{layer!r}]; found {list(source.layers)}")
    counts = sparse.csr_matrix(source.layers[layer], dtype=np.float32)
    probe = counts.data[:100000]
    if probe.size and not np.allclose(probe, np.round(probe)):
        raise ValueError(f"layers[{layer!r}] is not integer-valued, so it does not hold counts")

    # Only what the cache build reads back: the condition, and any column a split
    # source may be keyed on. The index is kept, since the cache copies it.
    wanted = ["condition", config["split"].get("obs_key"), config["split"].get("group_key")]
    keep = list(dict.fromkeys(c for c in wanted if c and c in source.obs.columns))
    if "condition" not in keep:
        raise ValueError(f"{raw} has no obs['condition']")

    library = np.asarray(counts.sum(axis=1)).ravel()
    median = float(np.median(library[library > 0]))
    adata = ad.AnnData(X=counts, obs=source.obs[keep].copy(),
                       var=pd.DataFrame(index=source.var_names.copy()))
    del source
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    # log1p records {'base': None}, which some anndata versions refuse to write.
    adata.uns.pop("log1p", None)
    adata.uns["normalisation"] = json.dumps({
        "source": raw, "layer": layer, "target_sum": median,
        "recipe": "sc.pp.normalize_total(target_sum=None) + sc.pp.log1p"})
    # Written under a temporary name and renamed into place, so another process
    # never opens a half-written copy; two processes racing to build it both
    # finish a whole file and the rename keeps the last.
    partial = f"{stem}_lognorm_{layer}_median.tmp-{os.getpid()}.h5ad"
    adata.write_h5ad(partial)
    os.replace(partial, target)
    print(f"  median library size {median:.1f}  ->  wrote {target}")
    return target


def gene_statistics(path: str, chunk_size: int,
                    keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene mean and variance, in one streaming pass.

    `keep` restricts the pass to a set of row indices. Gene selection is part of
    the model: choosing the gene space from statistics that include the held-out
    conditions lets those conditions decide what the model is allowed to see.
    """
    n_obs, n_var = io.shape(path)
    total = np.zeros(n_var, dtype=np.float64)
    total_sq = np.zeros(n_var, dtype=np.float64)

    mask = None if keep is None else np.zeros(n_obs, dtype=bool)
    if mask is not None:
        mask[np.asarray(keep)] = True
        n_obs = int(mask.sum())

    for start, stop, block in io.iter_row_chunks(path, chunk_size):
        if mask is not None:
            block = block[mask[start:stop]]
            if block.shape[0] == 0:
                continue
        total += np.asarray(block.sum(axis=0)).ravel()
        total_sq += np.asarray(block.multiply(block).sum(axis=0)).ravel()
        print(f"  variance pass: {stop:,} / {n_obs:,} cells", end="\r")
    print()

    mean = total / n_obs
    variance = total_sq / n_obs - mean ** 2
    return mean, np.maximum(variance, 0.0)


def scanpy_hvg(path: str, n_top: int, keep: np.ndarray | None = None) -> np.ndarray:
    """scanpy's own HVG call, so the gene set matches scDFM's exactly.

    scDFM's Norman branch is one line:

        sc.pp.highly_variable_genes(adata, inplace=True, n_top_genes=n_top_genes)

    with no flavor argument, i.e. the 'seurat' default. That is NOT a
    variance-to-mean ratio: dispersion is taken on expm1(X), genes are binned into
    20 bins by mean, and dispersion is z-scored WITHIN each bin. Selecting by a
    plain ratio picks a different set - measured on our own cells, the two
    criteria move the Control-vs-double L2 by a factor of 1.6, which is most of
    why our absolute numbers did not line up with the published table.

    scanpy is CALLED rather than reimplemented. The binning, the ddof and the
    one-gene-per-bin handling are each a place where a reimplementation diverges
    silently, and matching theirs is the entire point of this criterion. The cost
    is holding the raw matrix in memory once; it is sparse, so that is affordable.
    """
    import anndata as ad
    import scanpy as sc

    adata = ad.read_h5ad(path)
    if keep is not None:
        # Subset before selecting: otherwise the held-out conditions vote on which
        # genes the model is built from.
        adata = adata[np.asarray(keep)].copy()
    sc.pp.highly_variable_genes(adata, inplace=True, n_top_genes=n_top)
    return np.flatnonzero(adata.var["highly_variable"].to_numpy())


def select_genes(config: dict, var_names: np.ndarray, conditions: np.ndarray,
                 mean: np.ndarray, variance: np.ndarray,
                 path: str | None = None,
                 keep: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Choose the modelled gene set and return its indices in var_names order."""
    data_cfg = config["data"]
    criterion = data_cfg["hvg_criterion"]
    n_hvg = data_cfg["n_hvg"]

    if criterion == "scanpy":
        if path is None:
            raise ValueError("hvg_criterion=scanpy needs the raw h5ad path")
        if n_hvg is None:
            hvg_indices = set(range(len(var_names)))
        else:
            hvg_indices = set(scanpy_hvg(path, n_hvg, keep).tolist())
        selected = set(hvg_indices)
        score = None
    else:
        if criterion == "raw_variance":
            score = variance
        elif criterion == "dispersion":
            # variance-to-mean ratio; guard the genes that are all-zero
            score = np.where(mean > 0, variance / np.maximum(mean, 1e-12), 0.0)
        else:
            raise ValueError(f"unknown hvg_criterion {criterion!r}")

        if n_hvg is None:
            selected = set(range(len(var_names)))
            hvg_indices: set[int] = set(selected)
        else:
            hvg_indices = set(np.argsort(-score)[:n_hvg].tolist())
            selected = set(hvg_indices)

    name_to_index = {name: i for i, name in enumerate(var_names)}
    targets = perturbation_targets(conditions, data_cfg["control_label"])
    missing = [t for t in targets if t not in name_to_index]
    forced: list[int] = []
    if data_cfg["force_include_targets"]:
        for target in targets:
            index = name_to_index.get(target)
            if index is not None and index not in selected:
                selected.add(index)
                forced.append(index)

    indices = np.array(sorted(selected), dtype=np.int64)
    stats = {
        "n_selected": len(indices),
        "n_hvg": len(hvg_indices),
        "n_targets": len(targets),
        "n_targets_forced_in": len(forced),
        "n_targets_missing_from_var": len(missing),
        "missing_targets": missing,
        "criterion": criterion,
    }
    return indices, stats


def build_matrix(path: str, gene_indices: np.ndarray, chunk_size: int) -> sparse.csr_matrix:
    """Second pass: stream the rows again, keeping only the selected columns."""
    n_obs, _ = io.shape(path)
    blocks = []
    for start, stop, block in io.iter_row_chunks(path, chunk_size):
        blocks.append(block[:, gene_indices])
        print(f"  subset pass:   {stop:,} / {n_obs:,} cells", end="\r")
    print()
    return sparse.vstack(blocks, format="csr")
