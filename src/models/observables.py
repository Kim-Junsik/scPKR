"""Phi: the FIXED map from a cell to its observables. No learned parameters.

    p = (x @ M.T - mean) / std          M is [K, G] and constant

THIS IS THE KOOPMAN CLAIM, and the whole design rests on `M` being fixed. eDMD's
central open problem is choosing the dictionary of observables; the usual answers
are polynomials, RBFs, or a learned encoder. KEGG is a biological answer: it comes
from outside the data so it cannot overfit it, and because it is fixed there is no
reconstruction loss and therefore no autoencoder ceiling. On scPKFM's Table 3 that
ceiling was 0.97 of a 2.138 total - 45 % of the error, paid before transport started,
and not removable while a latent autoencoder was in the design. Widening it made
things worse (a rank-16 readout took the single block from 2.57 to 3.94) and
tightening it collapsed the latent (KL 1e-2 gave Pearson delta 0.012), which is what
identifies it as a generalisation problem rather than a capacity one.

TWO BLOCKS.

  pathway   one coordinate per surviving KEGG pathway, the pooled expression of its
            member genes.
  anchor    one coordinate per anchor gene, read directly.

The anchor block is not a convenience. A PERTURBATION'S OWN TARGET MUST BE
OBSERVABLE: 53 of Norman's 101 targets sit in no usable KEGG pathway - the
HOX/FOX/DLX/LHX developmental TF block, which KEGG does not catalogue - and an
observable space that cannot see the perturbed gene cannot represent its effect. That
is a rank deficiency, not a tuning issue. It also covers combosciplex, whose
perturbations are drugs and have no target gene in the data at all: there the anchors
are the highest-variance genes, so the residual has somewhere to live.

scPKFM answered the same gap with `n_free_tokens` all-zero rows and let its attention
invent structure there. That does not transfer: this matrix IS the observable space,
and a row of zeros is an observable that reads nothing.

THE READOUT SCAFFOLD FALLS OUT OF THIS. W (observables -> genes) is allowed to be
non-zero exactly where M is, i.e. `W` has the support of `M.T`. So a non-additive
correction can only reach a gene through a pathway that gene belongs to, or through
its own anchor coordinate. That is the pathway-mediation claim, and `edges()` returns
the index pair it needs. Nothing else in the model has to define the sparsity.

STANDARDISATION IS PART OF Phi AND IS FITTED ONCE. Pathway means differ in scale by
orders of magnitude, and without standardisation the operator would spend its
capacity undoing that. The statistics come from TRAINING ROWS ONLY and are never
refitted - scPKFM measured refitting mid-training as strictly harmful (fm jumped
0.0087 -> 1.146 at every refit and each window came back worse), and its own config
settled on fit-once. They travel in the checkpoint, so a scored run uses the
coordinates it was trained in.

LEAK SURFACE, stated because it is the one here: the anchor selection and the
standardisation both read cells. Both take `train_rows` and nothing else. A held-out
condition's cells must never reach either, exactly as they must never reach the gene
selection (data.exclude_test_from_hvg) or the splits.
"""

from __future__ import annotations

import numpy as np
import torch

import torch.nn as nn

from ..data.kegg import build_prior

POOLS = ("mean", "sum", "l2")
KINDS = ("kegg", "pca", "random", "genes")
ANCHORS = ("targets", "variance", "targets+variance", "none")


def _pool_rows(membership: np.ndarray, pool: str) -> np.ndarray:
    """Scale each pathway row so its coordinate is a pooled expression.

    mean is the only pooling invariant to pathway size. sum makes a 300-gene pathway
    30x the scale of a 10-gene one; the filters keep sizes 10-300, so that is a real
    30x and the operator would have to absorb it. l2 sits between the two and is kept
    because it is the natural choice if the coordinate is read as a projection rather
    than as an average.
    """
    if pool not in POOLS:
        raise ValueError(f"unknown model.observable_pool {pool!r} ({' | '.join(POOLS)})")
    sizes = membership.sum(axis=1, keepdims=True)
    sizes = np.maximum(sizes, 1.0)
    if pool == "mean":
        return membership / sizes
    if pool == "l2":
        return membership / np.sqrt(sizes)
    return membership


def _select_anchors(kind: str, n_wanted: int, targets: list[str],
                    gene_names: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Gene columns that get their own observable coordinate.

    Targets come first and are never dropped for a variance ranking: the point of the
    block is that the perturbed quantity is observable, and a target whose expression
    barely varies in the control population is exactly the case where the model most
    needs to see it directly.
    """
    if kind not in ANCHORS:
        raise ValueError(f"unknown model.anchor_genes {kind!r} ({' | '.join(ANCHORS)})")
    if kind == "none" or n_wanted <= 0:
        return np.zeros(0, dtype=np.int64)

    column = {name: i for i, name in enumerate(gene_names)}
    chosen: list[int] = []
    seen: set[int] = set()
    if "targets" in kind:
        for target in targets:
            index = column.get(target)
            if index is not None and index not in seen:
                seen.add(index)
                chosen.append(index)
    if "variance" in kind:
        for index in np.argsort(-variance):
            if len(chosen) >= n_wanted:
                break
            if int(index) not in seen:
                seen.add(int(index))
                chosen.append(int(index))
    # Targets are kept even past n_anchor_genes. Truncating them would defeat the
    # block: on Norman there are 101 and all of them must be observable.
    return np.asarray(chosen, dtype=np.int64)


class Observables(nn.Module):
    """Phi as one [K, G] matrix plus a fixed standardisation. Nothing is LEARNED.

    An nn.Module with BUFFERS rather than a plain object, so the matrix and the
    standardisation travel in the checkpoint. A scored run then uses the coordinates it
    was trained in rather than ones rebuilt from the data and hoped to be identical -
    scPKFM's own comment on its anchor table says why: a stale table shipped alongside
    stale weights reads as a bad result instead of a bad load. Rebuilding from data
    gives the right shapes; load_state_dict then makes the saved tensors authoritative.

    `train_rows` are the cells the model is allowed to see. They set the variance
    ranking for the anchor block and the standardisation, and nothing else reads
    cells here.
    """

    def __init__(self, config: dict, gene_names: np.ndarray, x: np.ndarray,
                 train_rows: np.ndarray, targets: list[str]):
        super().__init__()
        model_cfg = config["model"]
        kind = model_cfg["observables"]
        if kind not in KINDS:
            raise ValueError(f"unknown model.observables {kind!r} ({' | '.join(KINDS)})")
        self.kind = kind
        self.gene_names = np.asarray(gene_names)
        n_genes = len(self.gene_names)
        cells = x[np.asarray(train_rows)]
        variance = cells.var(axis=0)
        rng = np.random.default_rng(model_cfg["observable_seed"])

        if kind == "kegg":
            membership, names = build_prior(config, self.gene_names)
        elif kind == "random":
            # KEGG's row sizes, permuted gene sets. The control for "is it the
            # pathways, or just K sparse projections of this sparsity?"
            reference, _ = build_prior(config, self.gene_names)
            sizes = reference.sum(axis=1).astype(int)
            membership = np.zeros_like(reference)
            for row, size in enumerate(sizes):
                membership[row, rng.choice(n_genes, size=max(size, 1), replace=False)] = 1.0
            names = [f"<random {i}>" for i in range(len(sizes))]
        elif kind == "pca":
            # A dictionary LEARNED from the data, at the same width. If this wins,
            # biology is not load-bearing and the Koopman claim is withdrawn.
            reference, _ = build_prior(config, self.gene_names)
            n_components = min(reference.shape[0], cells.shape[0], n_genes)
            centred = cells - cells.mean(axis=0, keepdims=True)
            # Through the GENE covariance, not an SVD of the cell matrix. The right
            # singular vectors are the eigenvectors of C^T C, and that route costs the
            # same whatever the cell count while a full SVD scales linearly in it:
            # measured on random matrices of the real width, 4,000 x 5,000 took 21.2 s by
            # SVD against 8.6 s here, 8,000 x 5,000 took 43.9 s against 8.8 s, so at the
            # 63,378 cells combosciplex actually has the gap is around 27x. The old call
            # computed all 5,000 singular vectors and kept a few hundred; three queues
            # doing that at once left the GPUs idle for a long stretch of every pca run.
            #
            # The Gram matrix squares the condition number, which is why eigh runs in
            # float64 while the big matmul stays in float32. Agreement with the SVD on the
            # retained subspace, sign-free, is 0.999997 at worst over the top 300.
            gram = (centred.T @ centred).astype(np.float64)
            _, vectors = np.linalg.eigh(gram)
            right = vectors[:, ::-1][:, :n_components].T
            membership = np.ascontiguousarray(right, dtype=np.float32)
            names = [f"<pc {i}>" for i in range(n_components)]
        else:  # genes: no grouping at all
            reference, _ = build_prior(config, self.gene_names)
            keep = np.argsort(-variance)[:reference.shape[0]]
            membership = np.zeros((len(keep), n_genes), dtype=np.float32)
            membership[np.arange(len(keep)), keep] = 1.0
            names = [f"<gene {self.gene_names[c]}>" for c in keep]

        # Pooling applies to the pathway block only. A PCA row is already a unit
        # projection and a `genes`/`random` row is an indicator, so scaling either by
        # its size would be measuring something other than what the name says.
        pathway_rows = (_pool_rows(membership, model_cfg["observable_pool"])
                        if kind in ("kegg", "random") else membership)

        anchors = _select_anchors(model_cfg["anchor_genes"],
                                  int(model_cfg["n_anchor_genes"]),
                                  list(targets), self.gene_names, variance)
        anchor_rows = np.zeros((len(anchors), n_genes), dtype=np.float32)
        anchor_rows[np.arange(len(anchors)), anchors] = 1.0

        self.n_pathways = int(pathway_rows.shape[0])
        self.n_anchors = int(len(anchors))
        self.anchor_columns = anchors
        self.names = list(names) + [f"<anchor {self.gene_names[c]}>" for c in anchors]
        matrix = np.concatenate([pathway_rows, anchor_rows], axis=0).astype(np.float32)
        if matrix.shape[0] == 0:
            raise ValueError("Phi has no coordinates: check the KEGG snapshot and "
                             "data.pathway_min_genes / pathway_max_genes")

        # Standardisation, from training rows only, fitted once.
        raw = cells @ matrix.T
        self._mean = raw.mean(axis=0).astype(np.float32)
        # A coordinate with no variation in the training cells carries no
        # information; clamping rather than dropping keeps the coordinate list
        # aligned with `names` and with the readout scaffold.
        self._std = np.maximum(raw.std(axis=0), 1e-6).astype(np.float32)

        self.register_buffer("matrix", torch.from_numpy(matrix), persistent=True)
        self.register_buffer("mean", torch.from_numpy(self._mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(self._std), persistent=True)
        self.register_buffer("whitener", self._fit_whitener(config, cells),
                             persistent=True)

    # ------------------------------------------------------------------ interface

    @property
    def dim(self) -> int:
        return self.n_pathways + self.n_anchors

    @property
    def pathway_block(self) -> slice:
        return slice(0, self.n_pathways)

    @property
    def anchor_block(self) -> slice:
        """Reported separately from the pathway block, always.

        scPKFM's README made the same demand of its free tokens: a coordinate that is
        not a pathway must not be presented as one. Here the anchor coordinates are
        single genes, so an operator entry touching them says something different from
        one touching a pathway.
        """
        return slice(self.n_pathways, self.dim)

    def _fit_whitener(self, config: dict, cells: np.ndarray) -> torch.Tensor:
        """The [K, K] map that decorrelates the coordinates, or the identity.

        WHAT THIS IS FOR. PCA observables beat KEGG ones by a margin the pre-registered
        rule adopts (-0.0379 +- 0.0167 against -0.0173 +- 0.0136 for KEGG with the same
        dense readout), and KEGG is indistinguishable from RANDOM observables of the same
        width (-0.0155 +- 0.0188). Two explanations fit that, and they have opposite
        consequences for the paper:

          the SPAN is wrong   - the biology does not describe the right subspace
          the BASIS is wrong  - KEGG pathways share genes heavily so the coordinates are
                                strongly correlated, where PCA's are orthogonal by
                                construction

        Whitening separates them, because it changes ONLY the basis: the row space of M is
        untouched, so a model that recovers PCA's performance after whitening says the
        biological subspace was never the problem.

        It is not a no-op even though A_a is a general K x K matrix that could absorb any
        change of basis. Two things here are basis-dependent: A_a is LOW RANK
        (shared_rank + private_rank, so which directions it can reach depends on the
        coordinates), and gradient descent is not invariant to a change of basis whatever
        the parameterisation.

        ZCA rather than PCA whitening - the symmetric inverse square root - because it is
        the decorrelating map closest to the identity, so each whitened coordinate stays as
        close as it can to the pathway it came from. Fitted on TRAINING ROWS ONLY, like the
        standardisation it follows, and stored in the checkpoint.
        """
        mode = config["model"].get("observable_whiten", "none")
        size = self.matrix.shape[0]
        if mode == "none":
            return torch.eye(size, dtype=torch.float32)
        if mode != "zca":
            raise ValueError(f"unknown model.observable_whiten {mode!r} (none | zca)")
        coordinates = (cells @ self._matrix_for_fit().T - self._mean) / self._std
        centred = coordinates - coordinates.mean(axis=0, keepdims=True)
        covariance = (centred.T @ centred).astype(np.float64) / max(len(centred) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        # A floor rather than a pseudo-inverse: KEGG pathways are redundant enough that
        # some directions carry almost no variance, and inverting those would amplify
        # noise into coordinates the operator then has to spend capacity undoing.
        floor = float(config["model"].get("observable_whiten_floor", 1e-3))
        values = np.maximum(values, floor * max(float(values.max()), 1e-12))
        inverse_root = (vectors / np.sqrt(values)) @ vectors.T
        return torch.from_numpy(np.ascontiguousarray(inverse_root, dtype=np.float32))

    def _matrix_for_fit(self) -> np.ndarray:
        return self.matrix.detach().cpu().numpy()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Supply an identity whitener for checkpoints written before it existed.

        `whitener` was added after 78 runs had already been trained, and a new buffer makes
        load_state_dict fail outright under its default strict=True - so every one of those
        runs became unscoreable the moment the buffer landed. An old checkpoint was trained
        with no whitening at all, and the identity is exactly that, so the upgrade is not an
        approximation: it reproduces the model that was saved.

        The same thing happened once in scPKFM when the head's values grew a rank axis, and
        it was fixed the same way. The lesson that did not transfer is that a buffer added
        to a module in a repository full of finished runs needs this hook IN THE SAME
        COMMIT, not after a sweep has failed to score.
        """
        key = prefix + "whitener"
        if key not in state_dict:
            state_dict[key] = torch.eye(self.matrix.shape[0],
                                        dtype=self.matrix.dtype,
                                        device=self.matrix.device)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, G] -> [B, K]. One matmul; no parameters, no gradient of its own."""
        standardised = (x @ self.matrix.T - self.mean) / self.std
        return standardised @ self.whitener.T

    def edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(gene, observable) index pairs W may be non-zero on: the support of M.T.

        A non-additive correction can therefore reach a gene only through a pathway
        that gene belongs to, or through its own anchor coordinate. That is the
        pathway-mediation claim expressed as a sparsity pattern rather than as a
        penalty, so it cannot be traded away by the optimiser.
        """
        observable, gene = torch.nonzero(self.matrix, as_tuple=True)
        return gene, observable

    def summary(self) -> dict:
        gene, _ = self.edges()
        covered = int(torch.unique(gene).numel())
        return {
            "kind": self.kind,
            "K": self.dim,
            "K_pathway": self.n_pathways,
            "K_anchor": self.n_anchors,
            "edges": int(gene.numel()),
            "genes_reachable": covered,
            "genes_total": len(self.gene_names),
            # Genes no observable touches cannot receive a non-additive correction at
            # all. Their prediction is the additive component alone, which is the
            # honest consequence of the claim and has to be reported rather than
            # patched: if this fraction is large the scaffold is too sparse to carry
            # the residual.
            "genes_unreachable": len(self.gene_names) - covered,
        }
