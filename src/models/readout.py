"""W: observable displacement -> gene displacement, on the KEGG scaffold.

    Delta x_residual = W @ (p1 - p0)

W IS WHERE THE ZERO INITIALISATION LIVES, and that is the whole reason the model's
first prediction is exactly the additive baseline. W is linear, so dL/dW is
proportional to the observable-space displacement and is non-zero at step 0: W moves
first and the operators follow once W is non-zero. Putting the zero in the operators
instead would kill every gradient, because the composition is second order in A - see
models/operator.py.

THE SPARSITY IS THE PATHWAY-MEDIATION CLAIM. W may be non-zero exactly where Phi's
matrix is, which observables.edges() returns, so a non-additive correction reaches a
gene only through a pathway that gene belongs to or through its own anchor coordinate.
The additive component meanwhile is free across every gene. As biology: synergy and
antagonism between perturbations happen through shared pathways.

Expressed as a support rather than as a penalty on purpose - a penalty is something the
optimiser can trade away, a support is not.

MEASURED BEFORE BUILDING IT, because it is a go/no-go: KEGG reaches a quarter of the
genes (1,268 of 5,000 on combosciplex) and holds 80.4 % of the residual energy on the
reported genes; on Norman, 2,404 of 5,032 and 82.7 %. Genes no observable touches
receive the additive component alone, which is the honest consequence and puts a floor
under the model: 0.7734 on Table 3, against scDFM's 1.6567. The scaffold is therefore
not the binding constraint.

THE `random` ARM HAS A COVERAGE ADVANTAGE, which is worth knowing before reading the
ablation. KEGG pathways overlap heavily on well-studied genes, so a random scaffold at
the same edge count reaches 2.3x more genes and holds 6 points more residual energy -
per gene reached, KEGG is 2.1x more efficient. If KEGG wins on score while touching
half as many genes, that is a stronger result than the ablation was designed to give.
"""

from __future__ import annotations

import torch
import torch.nn as nn

KINDS = ("kegg", "dense", "random")


class Readout(nn.Module):
    """W, initialised at ZERO. `kegg` and `random` are sparse; `dense` is not."""

    def __init__(self, config: dict, observables, n_genes: int):
        super().__init__()
        model_cfg = config["model"]
        kind = model_cfg["readout"]
        if kind not in KINDS:
            raise ValueError(f"unknown model.readout {kind!r} ({' | '.join(KINDS)})")
        self.kind = kind
        self.n_genes = int(n_genes)
        self.dim = int(observables.dim)

        if kind == "dense":
            # Every (gene, observable) pair learnable: 5,000 x 413 = 2.07 M. The
            # ablation for the scaffold, and the reason it is a separate branch is
            # memory - the edge representation below would materialise [B, G*K].
            self.weight = nn.Parameter(torch.zeros(self.n_genes, self.dim))
        else:
            gene, observable = observables.edges()
            if kind == "random":
                # The same NUMBER of edges, placed uniformly. Asks whether it is KEGG
                # or just sparsity - and note this arm reaches more genes, not fewer.
                rng = torch.Generator().manual_seed(int(model_cfg["observable_seed"]))
                count = gene.numel()
                gene = torch.randint(self.n_genes, (count,), generator=rng)
                observable = torch.randint(self.dim, (count,), generator=rng)
            self.register_buffer("gene", gene.to(torch.long), persistent=True)
            self.register_buffer("observable", observable.to(torch.long), persistent=True)
            self.values = nn.Parameter(torch.zeros(gene.numel()))

        # OFF by default. The additive component already owns every gene's constant
        # shift; a second one would make the two terms fight over it, which is the
        # identifiability failure scPKFM had between its operators and rho - measured
        # as a held-out drug's effect pointing OPPOSITE its true shift (cosine -0.26).
        self.bias = (nn.Parameter(torch.zeros(self.n_genes))
                     if model_cfg["readout_bias"] else None)

    @property
    def n_edges(self) -> int:
        return self.n_genes * self.dim if self.kind == "dense" else int(self.gene.numel())

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """[B, K] -> [B, G]."""
        if self.kind == "dense":
            out = displacement @ self.weight.T
        else:
            # Scatter over edges rather than forming a [G, K] matrix: the support is
            # ~8,000 of 2 million entries, and index_add keeps both the memory and the
            # gradient on the edges that exist.
            contribution = displacement[:, self.observable] * self.values
            out = torch.zeros(displacement.shape[0], self.n_genes,
                              device=displacement.device, dtype=displacement.dtype)
            out = out.index_add(1, self.gene, contribution)
        return out if self.bias is None else out + self.bias

    @torch.no_grad()
    def dense_matrix(self) -> torch.Tensor:
        """W as a dense [G, K]. For inspection and for the interpretation figure.

        Paired with the operator's spectrum it is what makes a mode readable: A_a's
        eigenvector says which pathways move together, and W's column says which genes
        that reaches.
        """
        if self.kind == "dense":
            return self.weight.detach()
        out = torch.zeros(self.n_genes, self.dim,
                          device=self.values.device, dtype=self.values.dtype)
        out[self.gene, self.observable] = self.values.detach()
        return out

    def extra_repr(self) -> str:
        return (f"kind={self.kind}, n_genes={self.n_genes}, dim={self.dim}, "
                f"edges={self.n_edges}, bias={self.bias is not None}")
