"""Rebuild a finished run, and measure what its two terms actually did.

measure_transport here reports something scPKFM could not: the prediction splits into an
additive component and a learned residual, so the diagnostics can say whether the
learned term did anything and whether it pointed the right way. `residual_cos` is the
number to read - it is the per-condition version of what resid_R2 pools, and it answers
"is the residual right?" rather than "is the total error small?", which the additive
component already makes small on its own.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ..data.conventions import ConditionNaming
from ..data.dataset import PerturbationData
from ..data import splits
from ..models.model import PathwayKoopmanResidual
from ..models.observables import Observables
from . import baselines
from .baselines import ConditionMeans, training_conditions
from .conditions import condition_groups, scdfm_eval_genes  # re-exported for callers
from .predict import predict_cells

_DATASETS: dict = {}

__all__ = ["condition_groups", "scdfm_eval_genes", "load_run", "measure_transport",
           "ridge_weights", "training_rows"]


def _dataset(config: dict):
    """PerturbationData + ConditionMeans, shared across runs of one dataset.

    Keyed on what determines them rather than on the run, so a run pointing at a
    different cache or naming convention still builds its own instead of silently
    reusing the wrong cells.
    """
    data_cfg = config["data"]
    key = (data_cfg["cache_h5ad"], data_cfg.get("control_label", "ctrl"),
           data_cfg.get("condition_separator", "+"))
    if key not in _DATASETS:
        naming = ConditionNaming.from_config(config)
        data = PerturbationData(data_cfg["cache_h5ad"], naming=naming)
        labels = np.empty(data.x.shape[0], dtype=object)
        for condition, rows in data.rows.items():
            labels[rows] = condition
        _DATASETS[key] = (data, ConditionMeans(data.x, labels, naming))
    return _DATASETS[key]


def training_rows(data, train_conditions: list[str]) -> np.ndarray:
    """Every cell the model is allowed to see, control included.

    Phi's standardisation and its anchor ranking both read cells, so this is the leak
    surface and it has exactly one definition. A held-out condition's cells must never
    reach it, as they must never reach gene selection or the splits.
    """
    wanted = [data.rows[c] for c in train_conditions if c in data.rows]
    wanted.append(data.control_rows)
    return np.unique(np.concatenate(wanted))


def ridge_weights(config: dict, data, stats, train_conditions: list[str]) -> np.ndarray:
    """w_a as [P, G], ordered by data.perturbations.

    A perturbation the ridge cannot cover - one appearing in no training condition at
    all - gets zeros, so the model predicts the control for it unchanged. That is what
    scPKFM also did by accident, its operator staying at its zero initialisation, and it
    accounted for 10-15 % of Table 2's Single block. Here it is explicit.
    """
    naming = data.naming
    perturbations = sorted({g for c in stats.mean for g in naming.genes(c)})
    fit = baselines.fit_ridge_additive(
        stats, train_conditions, perturbations,
        alpha=config["model"]["additive_alpha"],
        weight_by_cells=config["eval"]["ridge_weight_by_cells"])
    weights = np.zeros((data.n_perturbations, data.n_genes), dtype=np.float32)
    for name in data.perturbations:
        if name in fit["covered"]:
            weights[data.pert_index[name]] = fit["w"][fit["index"][name]]
    return weights


def build_model(config: dict, data, stats, fold: dict, method: str, device: str):
    """The model for this config, and the pieces it needs from the data.

    Used both by training and by load_run, so a scored model is assembled exactly as a
    trained one was. The observables and the ridge fit are REBUILT from the data rather
    than unpickled - they are deterministic functions of the config and the training
    rows - and load_state_dict then overwrites both with the run's own saved tensors, so
    the rebuild only has to get the shapes right.
    """
    train_conditions = training_conditions(stats, fold, method)
    rows = training_rows(data, train_conditions)
    observables = Observables(config, data.gene_names, data.x, rows, data.perturbations)
    weights = ridge_weights(config, data, stats, train_conditions)
    cells = data.x[rows]
    detected = cells > 0
    counts = detected.sum(axis=0)
    detection = torch.from_numpy((counts / max(len(rows), 1)).astype(np.float32))
    # Per-gene spread of the NON-ZERO values, which is what the magnitude models: a
    # gene detected in 5 % of cells has magnitudes around 20 times its overall mean, so
    # a unit spread would be neither the right size nor comparable between two genes.
    total = np.where(detected, cells, 0.0).sum(axis=0)
    conditional_mean = total / np.maximum(counts, 1)
    variance = (np.where(detected, (cells - conditional_mean) ** 2, 0.0).sum(axis=0)
                / np.maximum(counts - 1, 1))
    dispersion = torch.from_numpy(np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32))
    model = PathwayKoopmanResidual(config, observables, data.n_perturbations,
                                  weights, detection, dispersion).to(device)
    return model, train_conditions, rows


def load_run(run_dir: str, device: str = "cpu", gate: str | None = None,
             checkpoint_name: str = "checkpoint.pt"):
    """Rebuild a finished run from its checkpoint.

    `gate` overrides model.hurdle_gate for THIS load only. The gate decides how the head
    realises the binary event at INFERENCE and appears in no training loss, so scoring a
    run under a different one re-reads the same weights rather than changing the model.

    Defaults to cpu: these are cheap and the usual reason to run them is that a sweep is
    occupying the gpu.

    The returned data and stats are SHARED between calls. Nothing here writes to them,
    but a caller that wants to mutate them must copy first.
    """
    checkpoint = torch.load(os.path.join(run_dir, checkpoint_name),
                            map_location=device, weights_only=False)
    config = checkpoint["config"]
    config["train"]["device"] = config["eval"]["device"] = device
    if gate:
        config["model"]["hurdle_gate"] = gate

    data, stats = _dataset(config)
    method = config["split"]["method"]
    fold = splits.folds(config, method)[config["split"]["fold"]]
    model, _, _ = build_model(config, data, stats, fold, method, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return config, data, stats, fold, model


def _ratio(predicted: np.ndarray, true: np.ndarray) -> float:
    norm = float(np.linalg.norm(true))
    return float(np.linalg.norm(predicted)) / norm if norm else float("nan")


def _cosine(predicted: np.ndarray, true: np.ndarray) -> float:
    scale = float(np.linalg.norm(predicted)) * float(np.linalg.norm(true))
    return float(predicted @ true) / scale if scale else float("nan")


@torch.no_grad()
def measure_transport(model, data, stats, conditions: list[str], config,
                      rng: np.random.Generator, device: str, n_cells: int,
                      genes: np.ndarray | None = None,
                      chunk: int = 512) -> list[dict]:
    """Per condition: where the prediction went, and which term took it there.

    Reported as ratios and cosines against the TRUE quantity rather than as error norms,
    because the two failures look identical in a norm: a term that reaches 40 % of the
    way and one that overshoots to 160 % both score badly and only the ratio says which.

    THE COLUMN TO READ IS `residual_cos`. The additive component already makes the total
    error small - it is ridge_additive, which beats both published models on two of four
    tables - so `l2` and `gene_cos` move slowly and say little about what was learned.
    `residual_cos` asks the only question the learned term is responsible for: does the
    predicted residual point along the true one? It is the per-condition form of what
    resid_R2 pools, and it is zero at initialisation by construction.
    """
    control = data.cells(data.control_condition)
    rows = []
    for condition in conditions:
        if condition == data.control_condition or not stats.has(condition):
            continue
        pick = rng.choice(control.shape[0], size=min(n_cells, control.shape[0]),
                          replace=False)
        sample = control[pick]
        perturbations = [data.pert_index[g] for g in data.naming.genes(condition)]

        additive = model.additive(perturbations).cpu().numpy()
        residual_parts, observable_parts = [], []
        for start in range(0, sample.shape[0], max(chunk, 1)):
            x = torch.as_tensor(sample[start:start + chunk], device=device)
            residual_parts.append(model.residual(x, perturbations).cpu().numpy())
            observable_parts.append(
                model.observable_displacement(x, perturbations).cpu().numpy())
        residual = np.concatenate(residual_parts, axis=0).mean(axis=0)
        observable = np.concatenate(observable_parts, axis=0).mean(axis=0)

        predicted = predict_cells(model, sample, condition, data.pert_index,
                                  device, data.naming, chunk=chunk)
        gene_hat = predicted.mean(axis=0)
        gene_true, control_mean = stats.mean[condition], stats.control
        # The residual the additive component leaves. L2(additive) = ||this||
        # identically, so it is not a diagnostic quantity but the error itself.
        residual_true = gene_true - control_mean - additive

        if genes is not None:
            gene_hat, gene_true = gene_hat[genes], gene_true[genes]
            control_mean = control_mean[genes]
            additive, residual, residual_true = (additive[genes], residual[genes],
                                                 residual_true[genes])

        rows.append({
            "condition": condition,
            "n_true": int(stats.n[condition]),
            # Gene space: what every reported metric is computed on.
            "gene_ratio": _ratio(gene_hat - control_mean, gene_true - control_mean),
            "gene_cos": _cosine(gene_hat - control_mean, gene_true - control_mean),
            "l2": float(np.linalg.norm(gene_hat - gene_true)),
            # The split. `residual_share` is the learned term's size against the
            # closed-form one, so a run with share ~0 has learned nothing regardless of
            # what its l2 says.
            "additive_norm": float(np.linalg.norm(additive)),
            "residual_norm": float(np.linalg.norm(residual)),
            "residual_share": _ratio(residual, additive),
            "residual_cos": _cosine(residual, residual_true),
            "residual_ratio": _ratio(residual, residual_true),
            # Observable space: the coordinates the operator actually acts in, so a gap
            # between this and residual_cos localises the loss to the readout.
            "observable_norm": float(np.linalg.norm(observable)),
        })
    return rows
