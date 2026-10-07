"""Central configuration.

Every value an experiment might vary lives here, never as a literal inside the
code. Runs record the full resolved config next to their results, and any key can
be overridden from the command line with dot notation:

    python scripts/build_data.py --set data.n_hvg=5000 data.hvg_criterion=dispersion
"""

from __future__ import annotations

import copy
import json
from typing import Any

DEFAULTS: dict[str, Any] = {
    "data": {
        "raw_h5ad": "data/norman/norman.h5ad",
        "kegg_dir": "assets/kegg",
        "pathway_min_genes": 10,
        "pathway_max_genes": 300,
        "drop_disease_pathways": True,
        "cache_h5ad": "assets/norman_modeled.h5ad",
        # X in the source file is ALREADY log1p-normalised (values 0.305-6.405,
        # non-integer, no raw-count layer). Never re-apply normalize_total/log1p.
        "assume_prenormalised": True,
        # A layer holding raw counts to rebuild X from, or null to use X as
        # shipped. combosciplex needs "counts": its X is normalised to 10,000 per
        # cell, the reference pipeline to the median library size, and that moves
        # Control L2 from 5.33 to 8.25 on the same cells. See
        # preprocess.normalised_source.
        "normalise_from_counts": None,
        "n_hvg": 3000,  # null selects every gene
        "hvg_criterion": "raw_variance",  # raw_variance | dispersion | scanpy
        # scanpy calls sc.pp.highly_variable_genes with its seurat default,
        # which is what scDFM uses. dispersion is a plain variance/mean ratio
        # and picks a different set - the two differ by 1.6x in Control L2.
        "force_include_targets": True,
        # Exclude the fold's held-out conditions from the cells that CHOOSE the
        # gene space. scDFM does not do this - its HVG call runs over the whole
        # dataset - so turning it on makes our problem strictly harder than the
        # published one, and the cache becomes fold-specific. Off by default for
        # that reason; run.sh turns it on because a zero-shot claim needs it.
        "exclude_test_from_hvg": False,
        # Condition naming. Norman writes 'AHR+FEV' / 'AHR+ctrl' / 'ctrl';
        # other datasets use 'control' or 'DMSO'. Read through
        # src/data/conventions.py, never hardcoded.
        "control_label": "ctrl",
        "condition_separator": "+",
        "chunk_size": 10000,
    },
    "split": {
        # The additive folds ship WITH the dataset, so there is nothing to
        # regenerate and nothing to copy: assets/splits_additive.pkl was
        # byte-identical to this file. The combinations folds are a deterministic
        # function of the additive ones and are derived at load time rather than
        # cached, so no split artifact can drift out of sync with its source.
        # reference_pkl: the file that ships with the dataset (Norman).
        # obs_column:    a column of the cache's obs, for datasets that carry
        #                their split per cell instead (combosciplex).
        # generated:     make one deterministically from a seed, for a dataset
        #                that ships no split at all. Nothing is written to disk.
        # list:          an explicit set of held-out conditions, one fold. With
        #                test_conditions null this is scDFM's seven for
        #                combosciplex (splits.SCDFM_COMBOSCIPLEX_TEST), the set
        #                behind their table; obs_column's 'ood' shares only one
        #                condition with it.
        "source": "reference_pkl",  # reference_pkl | obs_column | generated | list
        "test_conditions": None,  # source=list only; null = scDFM's combosciplex seven
        # source=list only. Removed from training and from gene selection but NOT
        # scored - how a validation run keeps the real test set out of the model
        # while scoring something else.
        "exclude_conditions": None,
        # Development split, so design choices are made without scoring a reported
        # test set. source=list (combosciplex): scores COMBOSCIPLEX_VALIDATION, or
        # COMBOSCIPLEX_VALIDATION_FOLDS[validation_fold], and excludes scDFM's seven.
        # source=reference_pkl (Norman, additive fold 0 only): scores
        # NORMAN_VALIDATION[0] and excludes that fold's test doubles.
        "validation": False,
        "validation_fold": None,  # combosciplex: 0-2, or null for the legacy pair
        # combosciplex only: also score splits.COMBOSCIPLEX_VALIDATION_SINGLES
        # [validation_fold], one held-out training single, so the single-drug block
        # of Table 3 has a validation counterpart. Needs validation_fold.
        "validation_singles": False,
        "reference_pkl": "data/norman/split_results.pkl",
        "obs_key": "split",
        "obs_test_value": "test",
        "generate_scheme": "combinations",  # doubles | combinations | group
        "generate_seed": 0,
        "generate_test_fraction": 0.3,
        "n_folds": 5,
        "group_key": "cell_type",  # for generate_scheme=group
        "method": "additive",  # additive | combinations
        "fold": 0,
        # Keep this fraction of the training COMBINATIONS (every single is kept). The
        # low-data axis: whether a fixed biological dictionary makes the operator more
        # data-efficient than one fitted from the data. Nested across fractions, so the
        # curve is a curve and not a sequence of unrelated samples.
        "train_condition_fraction": 1.0,
        "train_condition_seed": 0,
    },
    "eval": {
        # power=1 is Szekely's energy distance. power=2 collapses to
        # 2*||mean_x - mean_y||^2 and would make the metric blind to everything
        # beyond first moments.
        "edist_power": 1,
        "n_gen_cells": 256,  # control cells the predicted shift is applied to
        # PROVISIONAL. Picking this by looking at test performance would not be
        # legitimate, so it is pinned rather than tuned; select it by inner CV on
        # the training doubles before quoting a final target line.
        "ridge_alpha": 1.0,
        "ridge_weight_by_cells": False,
        # Multiplier on the learned residual, applied at inference. 0 is the additive
        # baseline exactly, 1 is the trained model. SELECTED ON THE VALIDATION FOLDS, and
        # swept from already-trained runs because the residual is linear in it. See
        # models/model.py for the measurement that motivates it: the residual's direction
        # is reliable on some conditions and anti-correlated on others, while its
        # magnitude is the same everywhere.
        "residual_scale": 1.0,
        # The per-condition calibration, s = clip(c ||r||^-p, 0, s_max), applied at
        # inference over the condition's own mean residual. p = 0 reduces it to a constant
        # scale and p = 1 makes the CORRECTION MAGNITUDE constant, keeping only the
        # residual's direction. residual_coefficient = null switches it off and
        # residual_scale alone applies.
        #
        # Both are SELECTED ON THE VALIDATION FOLDS by scripts/dev_rule.py, which fits them
        # in closed form from three numbers per condition and cross-validates by holding out
        # whole folds. Measured there on w6: combosciplex chose p = 0.8 and recovered 58 % of
        # the per-condition oracle where the best transferable constant recovered 19 %;
        # Norman chose p = 0.0, i.e. the family declined the exponent where it could not help.
        "residual_coefficient": None,
        "residual_power": 0.0,
        # LOAD-BEARING, not cosmetic. At p = 1 a condition whose residual is near zero
        # demands an enormous s, and that is the direction the earlier reliability
        # measurement says is WRONG: a small residual's cosine is dominated by noise, so its
        # sign is close to a coin toss (AUC 0.82 on combosciplex, 0.94 on Norman). Verified
        # on a short combosciplex run: at c = 0.5, p = 1 the four scored doubles asked for
        # 1.05, 0.98, 1.27 and 4.0, the last one clipped. The cap is what keeps the 1/||r||
        # shape from amplifying exactly the conditions it should leave alone.
        "residual_scale_max": 4.0,
        # Cap a REALISED cell at the largest value its gene attains in the training
        # cells. Inference-time, and it only touches the sample gate - the soft gate
        # returns the mean and never goes through the realisation path, so every
        # reported L2 is unaffected. Without it the sample gate emits values a hundred
        # times anything observed (ALOX15: realised 178.87, observed maximum 1.77) and
        # cell-eval refuses the export outright.
        "cap_realisation": False,
        # clamped_gaussian | gamma. How a realised magnitude is drawn, at inference only.
        # The clamp in clamped_gaussian biases the realised mean upward by 0.40 of L2,
        # measured and converged; gamma matches the same mean and variance with a
        # distribution that is already positive and so has no clamp to bias.
        "realisation": "clamped_gaussian",
        # Lower bound on the hurdle's detection probability, at inference. None keeps
        # whatever the run was trained with (1e-2). It bounds a realised magnitude at
        # mu/q_floor and provably cannot move the mean, since q * (mu/q) = mu for any q.
        "hurdle_q_floor": None,
        # Global magnitude correction applied AFTER decoding (predict.fit_alpha).
        # The model's predicted displacement is systematically too short - measured
        # ratio 0.646 on training singles, the conditions the loss supervises most
        # directly - and one scalar fitted on TRAINING conditions moved 5-fold
        # Norman L2 2.2482 -> 2.1418 and DS 0.7750 -> 0.8956. The fitted
        # alpha_train (1.16-1.38) was close to the test-optimal alpha, and a
        # per-condition oracle alpha only reaches 1.94, so this is most of what a
        # magnitude correction can buy; what remains is direction.
        #
        #   none  no correction
        #   mean  one shift for every cell, (alpha-1) * mean displacement. The
        #         predicted population's shape is exactly the decoder's.
        #   cell  each cell's own displacement is scaled.
        # They differ by (alpha-1) * the CENTRED displacement, so both give the same
        # population mean and L2 cannot separate them; cell additionally scales how
        # much the displacement varies between cells, which is the smaller part of
        # the spread when a transported population already carries the control
        # population's heterogeneity. Post-hoc and training-free: one checkpoint
        # scores every setting, so this is measured rather than chosen.
        "magnitude_alpha": "none",  # none | mean | cell

        "device": "cuda",
        "seed": 0,
    },
    "model": {
        # =====================================================================
        #   Delta x(S, x) = sum_{a in S} w_a                      additive, closed form
        #                 + 1[|S| >= 2] * W @ B_S @ Phi(x)        residual, learned
        #
        #   Phi   FIXED observables: KEGG pathway activities plus anchor genes
        #   A_a   per-perturbation Koopman operator on those observables
        #   B_S   composition of the operators in S
        #   W     observable -> gene readout, non-zero on KEGG edges only
        #
        # Two properties hold by construction and are asserted in
        # tests/test_structure.py rather than hoped for:
        #
        #   A_a = 0  =>  the prediction IS the additive baseline
        #   |S| = 1  =>  the second term is empty, so a single perturbation is the
        #                additive baseline EXACTLY, at every point in training
        #
        # Both matter because the baseline is not weak. Measured on the reported
        # metric over 5 folds (scripts/baseline_l2.py): Table 1 1.669, Table 2
        # single 1.416 / double 2.251, Table 3 1.858 - ahead of scPKFM on all four
        # and at or above scDFM on the first two. Training can only move down from
        # there, and it cannot touch the single block, which is the block learned
        # models lose 0.203 on.
        # =====================================================================

        # --- the additive component: 88 % of the signal, never learned ---
        # ridge  w_a from a one-hot ridge over the TRAINING conditions,
        #        baselines.fit_ridge_additive. Deterministic, so it contributes no
        #        seed noise - and scPKFM's seed noise (0.28 L2 on combosciplex)
        #        exceeded every effect it was trying to measure.
        # none   the model produces the whole displacement. The control arm, and
        #        what scPKFM was: it starts at zero and has to find 88 % of the
        #        signal before reaching any of the residual.
        "additive": "ridge",  # ridge | none
        # Ridge penalty for w_a. PROVISIONAL for the same reason eval.ridge_alpha is:
        # picking it by looking at test performance is not legitimate, so select it
        # by inner CV over the TRAINING combinations before quoting a final line.
        # Deliberately NOT eval.ridge_alpha - there it sizes a baseline being
        # reported, here it sizes a component of the model, and moving one should
        # not silently move the other.
        "additive_alpha": 1.0,

        # --- the observables: fixed, not learned ---
        # THE KOOPMAN CLAIM. eDMD's central open problem is choosing the dictionary
        # of observables, and the usual answers are polynomials, RBFs, or a learned
        # encoder. KEGG is a biological answer: it comes from outside the data so it
        # cannot overfit it, and it is fixed so there is no reconstruction loss and
        # no autoencoder ceiling. On scPKFM's Table 3 that ceiling was 0.97 of a
        # 2.138 total - 45 % of the error, paid before transport started, and not
        # removable while a latent autoencoder was in the design (widening it made
        # things worse: rank-16 readout took the single block 2.57 -> 3.94).
        #
        #   kegg    Phi(x) = [pool(M x) ; x_anchor]. The claim.
        #   pca     the same dimension from a PCA of the TRAINING cells. Ablation: a
        #           dictionary learned from the data. If it wins, biology is not
        #           load-bearing and claim A is withdrawn.
        #   random  random gene groups at KEGG's sparsity and size distribution.
        #           Ablation: is it the pathways, or just K sparse projections?
        #   genes   the top-K variance genes, no grouping. Ablation: is pooling
        #           doing anything at all?
        "observables": "kegg",  # kegg | pca | random | genes
        # How a pathway's activity is pooled from its member genes. mean is the only
        # one invariant to pathway size; sum makes a 300-gene pathway 30x the scale
        # of a 10-gene one and the operator would have to spend capacity undoing it.
        "observable_pool": "mean",  # mean | sum | l2
        # Anchor coordinates appended to the pathway ones.
        #
        # A PERTURBATION'S OWN TARGET MUST BE OBSERVABLE. 53 of Norman's 101 targets
        # sit in no usable KEGG pathway, and an observable space that cannot see the
        # perturbed gene cannot represent its effect - that is not a tuning issue,
        # it is a rank deficiency. `targets` adds every perturbation target present
        # in the modelled gene space; `variance` tops the set up to n_anchor_genes
        # with the highest-variance genes, which is what gives the residual somewhere
        # to live for a drug (combosciplex perturbs drugs, which have no target gene
        # in the data at all).
        "anchor_genes": "targets+variance",  # targets | variance | targets+variance | none
        "n_anchor_genes": 256,
        # Seed for observables=random and for the random readout scaffold. Fixed so
        # both ablations are reproducible.
        "observable_seed": 0,
        # Decorrelate the observable coordinates after standardising them, within the SAME
        # span. Added to separate two explanations of why PCA observables beat KEGG ones on
        # combosciplex (-0.0379 +- 0.0167 against -0.0173 +- 0.0136 at the same dense
        # readout, with KEGG indistinguishable from RANDOM observables at -0.0155 +-
        # 0.0188): either the biological SPAN is wrong, or the basis is simply
        # ill-conditioned because KEGG pathways share genes and PCA's axes are orthogonal by
        # construction. Whitening leaves the span untouched, so it moves only the second.
        "observable_whiten": "none",
        "observable_whiten_floor": 1e-3,

        # --- the operator: A_a, acting on the observables ---
        #   A_a = U diag(c_a) V + P_a Q_a
        # Most of each operator comes from modes every perturbation shares, and a
        # perturbation chooses how much of each it uses; the private part is small
        # and rank-limited.
        #
        # WHY SHARED RATHER THAN PER-PERTURBATION. combosciplex has 17 drugs, 24
        # training conditions, and 13 drugs that never appear alone. A full K x K
        # operator is ~90,000 parameters fitted from two or three conditions. See
        # the budget in docs/DESIGN.md section 4.
        #
        # m = 0 with p = r is a plain per-perturbation rank-r operator, which
        # tests/test_structure.py asserts as the special case.
        "shared_rank": 32,   # m, the shared modes
        "private_rank": 4,   # p, the per-perturbation private rank. 0 = shared only
        # INITIALISATION, and it is load-bearing - but the ZERO GOES IN THE READOUT,
        # NOT HERE. A_a is initialised small and RANDOM (U, V, P_a, Q_a) with c_a at
        # one, so every perturbation begins using every shared mode equally and the
        # basis first learns what the perturbations have in common.
        #
        # A_a must NOT start at zero. The composition is second order in A, so
        # dB_S/dA_a is proportional to A_b: with every operator at zero the gradient
        # of the loss with respect to every operator is also zero and nothing ever
        # moves. It is the same vanishing-product trap as zeroing both factors of a
        # low-rank product, one level up.
        #
        # What makes the initial prediction exactly the additive baseline is W = 0
        # (models/readout.py). W is LINEAR, so dL/dW is proportional to B_S Phi(x),
        # which is non-zero - W moves first and A_a follows once W is non-zero. That
        # is the zero-initialised-output-layer argument scPKFM used for rho, applied
        # at the layer where it actually holds.
        #
        # U diag(c) V is invariant to U -> kU, c -> c/k, so read A_a as a whole and
        # never U or c_a alone.
        "operator_init_scale": 0.02,

        # --- the composition law ---
        # THE COMPOSITION CLAIM. Summing generators and integrating gives
        #
        #   exp(t(A+B)) - (exp(tA) + exp(tB) - I) = (t^2/2)(AB + BA) + O(t^3)
        #
        # so the leading correction to an additive flow map is the ANTI-commutator.
        # It is symmetric under exchange, which is what a SIMULTANEOUS double
        # perturbation is. The commutator [A,B] is antisymmetric and describes order
        # dependence - Phi_a then Phi_b against the reverse - for which a
        # simultaneous perturbation carries no signal.
        #
        # That is why scPKFM's Lie bracket term lost 5-0 across seven settings and
        # two backbones, and why interaction terms in this literature generally
        # target the wrong symmetry. Verified numerically: at operator scales
        # 0.05 / 0.02 / 0.01 the anticommutator model's relative error is
        # 3.5e-2 / 1.4e-2 / 7.0e-3, falling linearly in the scale as an O(t^3)
        # remainder must, while the cosine against the commutator stays at -0.30.
        #
        #   anticommutator  B_S = sum_{a<b} (A_a A_b + A_b A_a). The claim.
        #   commutator      sum_{a<b} (A_a A_b - A_b A_a). Ablation, PREDICTED to do
        #                   nothing: it is orthogonal to the symmetric signal.
        #   bilinear        an unconstrained symmetric second-order form, still with
        #                   no pair-indexed parameter. Ablation: is it the
        #                   anticommutator specifically, or any symmetric term?
        #   sum             B_S = sum_a A_a, first order only. Ablation: is second
        #                   order needed at all, given that the additive part is
        #                   already handled in gene space?
        #
        # Every option is LINEAR IN Phi(x), so the flow map is exactly
        # exp(B_S) and there is no integrator anywhere in this model.
        "composition": "anticommutator",
        # There is no scalar on B_S and no time embedding. A_a already carries its own
        # scale through U and c_a, and a separate factor initialised at zero would
        # reintroduce the vanishing gradient described above. scPKFM's s(t) MLP was a
        # scalar reparameterisation of a scale the operator already had.
        #
        # Rank of G in composition=bilinear: B_S = sum_{a<b} (A_a G A_b + A_b G A_a)
        # with G = I + U_g V_g and U_g at zero, so that arm STARTS as the
        # anticommutator and the ablation is nested - it asks whether the
        # anticommutator specifically is right, or whether any symmetric second-order
        # form does as well, with the two sharing a starting point.
        "composition_rank": 16,

        # --- the readout: observables back to gene space ---
        # THE PATHWAY-MEDIATION CLAIM. W is non-zero only where KEGG says a gene
        # belongs to a pathway (about 39,600 edges), plus the diagonal of the anchor
        # block. So the additive component is free across all 5,000 genes, while the
        # NON-ADDITIVE correction may only travel along the pathway scaffold.
        #
        # As biology: synergy and antagonism between perturbations happen through
        # shared pathways. That is a claim, and dense falsifies it.
        #
        #   kegg    the scaffold. The claim.
        #   dense   every (gene, observable) pair learnable, 5,000 x K. Ablation: if
        #           dense is not better, the scaffold is load-bearing rather than a
        #           parameter saving.
        #   random  a random scaffold at the same sparsity. Ablation: KEGG, or just
        #           sparsity?
        "readout": "kegg",  # kegg | dense | random
        # Per-gene bias on the readout. OFF: the additive component already owns
        # every gene's constant shift, and a second one would make the two terms
        # fight over it - which is the identifiability failure scPKFM had between
        # its operators and rho, measured as a held-out drug's effect pointing
        # OPPOSITE its true shift (cosine -0.26, -0.31).
        "readout_bias": False,

        # --- the output head: mean prediction -> population ---
        # 41.2 % of entries are exactly zero, an mse head produces exact zeros
        # 0.000 % of the time and loses 24 % of the standard deviation, and
        # realising the binary event by SAMPLING moved energy distance 6.64 -> 1.46.
        # DS is a population statistic and is the axis scPKFM trailed on.
        #
        # The head here is SIMPLER than scPKFM's. There it read a [B, G, d_v] tensor
        # out of a decoder; here there is no decoder, so it reads the predicted mean
        # expression per gene and turns that scalar into a distribution:
        #
        #   gate_logit_g = a_g * xhat_g + b_g
        #   magnitude_g  = softplus(c_g * xhat_g + d_g)
        #   log_scale_g  = e_g
        #
        # Five parameters per gene, ~25 K total, and `affine` is the only link
        # implemented. scPKFM's head_rank ablation does not transfer: rank 16 was
        # harmful there because its DECODER was overfitting, and there is no decoder
        # here.
        "decoder_head": "hurdle",
        "hurdle_link": "affine",
        "hurdle_bce_weight": 1.0,
        # sample is right for a distribution-level metric; soft is the conditional
        # expectation and is what every reported L2 table was scored with.
        "hurdle_gate": "sample",  # soft | hard | sample
        "hurdle_magnitude": "gaussian",  # point <- MSE | gaussian
    },
    "train": {
        # ONE STAGE. There is no autoencoder to pretrain and no latent to
        # standardise, which is what scPKFM's two stages were for. Its stage 1 also
        # had to be frozen for stage 2 because unfreezing gave a trivial optimum -
        # collapse the latent and every field scores perfectly (measured:
        # ||z1 - z0|| fell to 0.019 while ||z0|| stayed near 8). Fixed observables
        # remove that failure mode along with the stage.
        "epochs": 400,
        "batch_size": 256,
        # 0 = a full pass. Set it low for smoke runs so a configuration can be
        # checked end to end in seconds.
        "max_steps_per_epoch": 0,
        "lr": 1e-3,
        "lr_cosine": True,
        "lr_min": 1e-6,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,

        # --- WHAT IS TRAINED ON: TRAINING COMBINATIONS ONLY ---
        # A single perturbation's prediction is the additive component, which is
        # closed form and has nothing to learn, so single conditions enter only
        # through the ridge fit. This is not a limitation to work around - it is
        # exactly why the single block cannot be damaged, and learned models lose
        # 0.203 to ridge on that block (scDFM 1.619 and scPKFM 1.754 against ridge's
        # 1.416, over 5 folds). Both published models have the same problem, so it
        # is the setting and not one architecture.

        # --- flow matching in observable space ---
        # The field is v(p, t) = B_S p: linear AND autonomous. Flow matching is
        # unchanged - same interpolant, same coupling - but the same operator has to
        # work at every t, which is what makes this an ODE rather than an arbitrary
        # field. The endpoint is then available in closed form and there is no
        # integrator: exp(B_S) p0, one matrix exponential per condition.
        "fm_weight": 1.0,
        # exp(B_S) p0 against p1, over the WHOLE BATCH rather than one mean point.
        # scPKFM could only afford the mean point because RK4 cost n_steps * 4 field
        # evaluations per cell, and its own plan recorded the batch version as the
        # one untried item with a measured bound (0.20-0.26). Here it is one
        # matrix_exp, so the bound is reachable at no extra cost.
        "endpoint_weight": 3.0,
        # The same endpoint after the readout and the head, in GENE space, against
        # the condition's mean over the reported genes. This term IS the reported
        # metric, so it is the one loss that cannot be mismatched with what is
        # scored. scPKFM's endpoint loss lived in the latent space and the mismatch
        # was measured: the conditions it supervised most directly came out worst
        # (train singles displacement ratio 0.646 against test doubles' 1.039).
        "gene_endpoint_weight": 1.0,
        # Multi-scale MMD between the predicted and the real population. OFF.
        # scPKFM ran it at weight 10 and measured the estimate at NOISE LEVEL
        # (negative values). It is listed because DS is a population statistic and
        # the axis that was trailed on - but it has to earn its weight on the
        # validation folds rather than be assumed into the default.
        "mmd_weight": 0.0,

        # --- the residual target ---
        # ridge  r_S = m_S - m_ctrl - sum_a w_a, the residual the additive component
        #        leaves. Computable for every training combination WITHOUT reading a
        #        single condition, which is what makes it usable where 13 of 17 drugs
        #        never appear alone.
        # true   r_S = m_S - m_A - m_B + m_ctrl, the textbook interaction. Needs both
        #        singles, so it does not exist for most combosciplex drugs. Kept for
        #        Norman's additive split, where it DOES exist, as a check that the
        #        ridge residual is the right target rather than a convenient one.
        "residual_target": "ridge",  # ridge | true

        # --- minibatch OT coupling; never random pairing ---
        # Computed in OBSERVABLE space (K ~ 550) rather than gene space (G = 5,000).
        # A minibatch OT plan is an ESTIMATE of the true plan, and the estimate is
        # the leading hypothesis for scPKFM's unexplained asymmetry: sharp OT helped
        # Norman (-0.0661 +/- 0.0093 over 3 seeds - the one properly measured
        # positive result in that project) and hurt combosciplex (+0.0194), and the
        # two differ 5x in cells per condition, so batch 48 saw 13 % of a Norman
        # condition against 2.4 % of a combosciplex one. Twenty times fewer
        # dimensions is the other half of that ratio.
        "coupling": "uot",  # uot | ot | random (random is a control only)
        "uot_reg": 0.05,
        "uot_reg_marginal": 1.0,
        # Sinkhorn's convergence threshold. POT defaults to 1e-6, which is far stricter
        # than the PLAN needs: its error measures the change in the scaling vectors, and
        # the plan itself stabilises long before that. At the default uot_reg the sampling
        # law at 1e-2 matches a plan solved to 1e-12 with 20,000 iterations to 3e-16 -
        # identical in float64 - while the solve takes 30-42 % of the time (17.4 -> 5.3 ms
        # at batch 256). That is 24 % of a whole training step.
        #
        # A THRESHOLD RATHER THAN AN ITERATION CAP, because the cap that suffices depends
        # on reg - 50 iterations at reg 0.05, 100 at 0.01, 500 at 0.005 - so a fixed cap
        # would silently truncate a smaller reg.
        #
        # IT IS ONLY FREE AT reg >= 0.05. Measured across batch 64/256 and dimension
        # 32/128/413: at reg 0.05 the plan is identical in every combination, while at
        # 0.01 and 0.005 it depends on the geometry and moves by up to 7e-5. A smaller reg
        # makes a sharper plan and converges more slowly. coupling_plan REFUSES the
        # combination rather than warning - use 1e-6 if you lower uot_reg, which v1's
        # coupling experiment did.
        #
        # Because the plan is numerically unchanged at the default reg, this does not
        # alter what any experiment measures: runs made before and after remain
        # comparable, which is what made it safe to change between experiments.
        "uot_stop_thr": 1e-2,
        # A degenerate plan (non-finite, or no mass) falls back to random pairing for
        # that batch. It is counted per epoch and training STOPS when more than this
        # share of an epoch's batches fell back: the fallback used to be silent, and
        # a reg too small for the cost scale made it happen on every batch (measured
        # 600/600 without cost normalisation at reg 0.1).
        "coupling_fallback_max": 0.05,

        # Epochs between mid-training checkpoints. 0 disables. A long run whose only
        # save is after evaluation loses everything to a kill or an OOM.
        "save_every": 25,
        "device": "cuda",
        "seed": 0,
        "out_dir": "results/runs",
    },
}


def _coerce(text: str) -> Any:
    """Turn a command-line string into the value it obviously denotes."""
    lowered = text.lower()
    # Only "null" spells None. "none" stays a string, because it is a legitimate
    # value for at least one key (model.interaction) and silently turning it into
    # None made that config crash after the run had already started.
    if lowered == "null":
        return None
    # A bracketed value is JSON. model.hidden is a list, and without this it
    # arrived as the STRING "[2048,1024]" and the first Linear was built from
    # whatever len() of that string returned - a config error that only surfaced
    # as a shape mismatch deep in stage 1.
    if text[:1] in "[{":
        import json
        try:
            return json.loads(text)
        except ValueError:
            return text
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b=value` strings onto a copy of `config`.

    Unknown keys raise instead of being silently created, so a typo in a sweep
    script fails loudly rather than running with the default.
    """
    resolved = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        path, raw = item.split("=", 1)
        node = resolved
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in node:
                raise KeyError(f"unknown config section {path!r}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"unknown config key {path!r}")
        node[keys[-1]] = _coerce(raw)
    return resolved


def load(overrides: list[str] | None = None) -> dict:
    return apply_overrides(DEFAULTS, overrides)


def dumps(config: dict) -> str:
    return json.dumps(config, indent=2, sort_keys=True)
