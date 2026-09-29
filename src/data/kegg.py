"""Assemble M_prior (K x G) from the KEGG REST download.

Measured on this gene space (docs/PIPELINE-v1.pdf 4.2):
  56/101 perturbation targets sit in some KEGG pathway, 48 survive the filters,
  K = 171 after dropping disease pathways and keeping sizes 10-300,
  density 0.87 %, pairwise Jaccard median 0.000.

53 of the 101 targets are in no usable pathway - HOX/FOX/DLX/LHX/POU3F2 and the
rest of the developmental TF block, which KEGG simply does not catalogue. Gene
selection cannot fix that, so `n_free_tokens` rows with an all-zero prior are
appended: for those the mask reduces to act(alpha * M_residual) and the model
builds its own tokens. That costs no extra data and leaks nothing.
"""

from __future__ import annotations

import collections
import os

import numpy as np

DISEASE_PREFIX = "hsa05"


def _read_gene_symbols(path: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    """KEGG entry -> primary symbol, plus alias -> entries for the fallback."""
    primary: dict[str, str] = {}
    aliases: dict[str, set[str]] = collections.defaultdict(set)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4 or ";" not in fields[3]:
                continue
            symbols = [s.strip() for s in fields[3].split(";")[0].split(",") if s.strip()]
            if not symbols:
                continue
            primary[fields[0]] = symbols[0]
            for symbol in symbols:
                aliases[symbol].add(fields[0])
    return primary, aliases


def load_pathways(kegg_dir: str) -> tuple[dict[str, set[str]], dict[str, str]]:
    """pathway id -> set of KEGG entries, and pathway id -> readable name."""
    members: dict[str, set[str]] = collections.defaultdict(set)
    with open(os.path.join(kegg_dir, "pathway_gene.tsv"), encoding="utf-8") as handle:
        for line in handle:
            pathway, entry = line.rstrip("\n").split("\t")
            members[pathway.replace("path:", "")].add(entry)

    names: dict[str, str] = {}
    with open(os.path.join(kegg_dir, "pathway_list.tsv"), encoding="utf-8") as handle:
        for line in handle:
            pathway, name = line.rstrip("\n").split("\t")
            names[pathway] = name.replace(" - Homo sapiens (human)", "")
    return members, names


def build_prior(config: dict, gene_names: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """M_prior in {0, 1}, rows aligned to the surviving pathways, columns to genes.

    Column order is `gene_names` as given - i.e. adata.var_names. That single
    convention replaces the old tokenizer/vocab alignment entirely, so there is no
    name-matching step that can silently misalign rows and columns.
    """
    model_cfg = config["model"]
    kegg_dir = config["data"]["kegg_dir"]
    gene_set = {name: i for i, name in enumerate(gene_names)}

    primary, aliases = _read_gene_symbols(os.path.join(kegg_dir, "gene_list.tsv"))
    members, names = load_pathways(kegg_dir)

    # Invert alias -> entries once. Scanning the whole alias table for every KEGG
    # entry is ~9,400 x 50,000 Python-level comparisons and dominated the cost of
    # building the model.
    entry_aliases: dict[str, list[str]] = collections.defaultdict(list)
    for alias, entries in aliases.items():
        for entry in entries:
            entry_aliases[entry].append(alias)

    def to_column(entry: str) -> int | None:
        symbol = primary.get(entry)
        if symbol in gene_set:
            return gene_set[symbol]
        if symbol is None:
            return None
        for alias in entry_aliases.get(entry, ()):  # alias fallback, +97 genes
            if alias in gene_set:
                return gene_set[alias]
        return None

    kept_rows, kept_names = [], []
    for pathway, entries in sorted(members.items()):
        if config["data"]["drop_disease_pathways"] and pathway.startswith(DISEASE_PREFIX):
            continue
        columns = {c for c in (to_column(e) for e in entries) if c is not None}
        if not (config["data"]["pathway_min_genes"] <= len(columns)
                <= config["data"]["pathway_max_genes"]):
            continue
        row = np.zeros(len(gene_names), dtype=np.float32)
        row[sorted(columns)] = 1.0
        kept_rows.append(row)
        kept_names.append(names.get(pathway, pathway))

    prior = np.stack(kept_rows) if kept_rows else np.zeros((0, len(gene_names)), np.float32)

    n_free = model_cfg["n_free_tokens"]
    if n_free:
        prior = np.concatenate([prior, np.zeros((n_free, len(gene_names)), np.float32)])
        kept_names += [f"<free {i}>" for i in range(n_free)]
    return prior, kept_names


def summarise(prior: np.ndarray, names: list[str], targets: list[str],
              gene_names: np.ndarray) -> dict:
    n_free = sum(1 for n in names if n.startswith("<free"))
    annotated = prior[:len(names) - n_free] if n_free else prior
    columns = {g: i for i, g in enumerate(gene_names)}
    covered = sum(1 for t in targets
                  if t in columns and annotated[:, columns[t]].sum() > 0)
    sizes = annotated.sum(axis=1)
    return {
        "K_total": int(prior.shape[0]),
        "K_annotated": int(annotated.shape[0]),
        "K_free": n_free,
        "density_percent": float(annotated.sum() / max(annotated.size, 1) * 100),
        "pathway_size_median": float(np.median(sizes)) if len(sizes) else 0.0,
        "targets_covered": covered,
        "targets_total": len(targets),
    }
