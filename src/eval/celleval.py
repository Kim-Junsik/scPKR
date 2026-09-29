"""Export predictions in the form cell-eval 0.5.42 consumes.

cell-eval lives in a separate Python 3.11 environment (.env-celleval) because it
pulls in a dependency set that conflicts with this project's. So the work is split
in two: this module runs inside the training environment and writes two .h5ad
files, and scripts/run_celleval.py then invokes the other interpreter on them.

Protocol note. The handoff spec's external comparison evaluates on all 19,264
genes, but this model predicts the 3,074-gene modelled space, so the export is
that space. Numbers are therefore comparable BETWEEN our runs, and comparable to
scDFM only after scDFM is re-scored on the same gene set - they are not directly
comparable to the published table.
"""

from __future__ import annotations

import numpy as np
import anndata as ad
import pandas as pd

from .predict import predict_cells

# cell-eval's own vocabulary: the control level is a named perturbation, and the
# column holding it has a fixed name.
PERT_COL = "target"
CONTROL_LABEL = "non-targeting"


def to_celleval_label(condition: str, naming) -> str:
    """'AHR+FEV' stays as is; 'AHR+ctrl' becomes 'AHR'; the control becomes
    cell-eval's own control level name."""
    genes = naming.genes(condition)
    if not genes:
        return CONTROL_LABEL
    return "+".join(genes)


def build_pair(vae, field, data, fold: dict, config: dict,
               rng: np.random.Generator, max_cells: int | None = None,
               genes: np.ndarray | None = None) -> tuple[ad.AnnData, ad.AnnData]:
    """Predicted and real AnnData over the fold's test conditions plus control.

    For every test condition the prediction transports a fresh sample of control
    cells, matched in count to the real condition so the two sides carry the same
    sampling noise - an unmatched count would show up as a distribution
    difference that has nothing to do with the model.
    """
    n_steps = config["train"]["n_integration_steps"]
    device = config["train"]["device"]
    control_cells = data.cells(data.control_condition)
    if max_cells:
        keep = rng.choice(control_cells.shape[0],
                          size=min(max_cells, control_cells.shape[0]), replace=False)
        control_cells = control_cells[keep]

    pred_blocks, pred_labels = [], []
    real_blocks, real_labels = [], []

    # Control appears on both sides: cell-eval needs it to form its deltas.
    pred_blocks.append(control_cells)
    pred_labels += [CONTROL_LABEL] * control_cells.shape[0]
    real_blocks.append(control_cells)
    real_labels += [CONTROL_LABEL] * control_cells.shape[0]

    for condition in fold["test"]:
        if condition not in data.rows:
            continue
        real = data.cells(condition)
        if max_cells and real.shape[0] > max_cells:
            real = real[rng.choice(real.shape[0], size=max_cells, replace=False)]
        n = real.shape[0]
        pick = rng.choice(control_cells.shape[0], size=n,
                          replace=control_cells.shape[0] < n)
        predicted = predict_cells(vae, field, control_cells[pick], condition,
                                  data.pert_index, n_steps, device, data.naming)
        label = to_celleval_label(condition, data.naming)
        pred_blocks.append(predicted)
        pred_labels += [label] * n
        real_blocks.append(real)
        real_labels += [label] * n

    def assemble(blocks, labels):
        matrix = np.concatenate(blocks, axis=0).astype(np.float32)
        names = data.gene_names
        if genes is not None:
            # Subset AFTER transport, not before: the model needs its full input
            # space to predict at all, and only the scoring is restricted.
            matrix, names = matrix[:, genes], names[genes]
        obs = pd.DataFrame({PERT_COL: pd.Categorical(labels)},
                           index=[f"cell{i}" for i in range(matrix.shape[0])])
        var = pd.DataFrame(index=pd.Index(names))
        return ad.AnnData(X=matrix, obs=obs, var=var)

    return assemble(pred_blocks, pred_labels), assemble(real_blocks, real_labels)


def export(vae, field, data, fold: dict, config: dict, out_dir: str,
           rng: np.random.Generator, max_cells: int | None = None,
           genes: np.ndarray | None = None) -> dict[str, str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    adata_pred, adata_real = build_pair(vae, field, data, fold, config, rng,
                                        max_cells, genes)
    paths = {"pred": os.path.join(out_dir, "pred.h5ad"),
             "real": os.path.join(out_dir, "real.h5ad")}
    adata_pred.write_h5ad(paths["pred"])
    adata_real.write_h5ad(paths["real"])
    paths["n_cells"] = str(adata_pred.n_obs)
    paths["n_conditions"] = str(adata_pred.obs[PERT_COL].nunique())
    paths["n_genes"] = str(adata_pred.n_vars)
    return paths
