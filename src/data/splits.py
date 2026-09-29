"""Loading and validating the inherited scDFM splits.

These splits are NOT ours. They come from scDFM's src/data_process/data.py so that
"the authors' method won on the authors' split" cannot be raised against us. Two
properties therefore have to hold and are checked here rather than assumed:

  1. the shipped reference is a clean partition per fold (train and test disjoint).
  2. the combinations folds derive deterministically from the additive ones.

There is no cached split artifact. The additive folds ARE the file that ships with
the dataset, and the combinations folds are computed from them at load time - a
copy on disk could drift out of sync with its source and nothing would notice.

The additive folds are five INDEPENDENT random 70/30 draws with different seeds,
not a five-way partition, so test sets overlap between folds. Anything that needs
leak-free cells (e.g. running GRN inference) must intersect across all folds
rather than assume a partition.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

N_TEST_DOUBLES_FOR_COMBINATIONS = 15  # scDFM keeps only the first 15 test doubles
CONTROL_SUFFIX = "ctrl"


def load(path: str) -> list[dict[str, Any]]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def derive_combinations(additive_folds: list[dict], control_suffix: str = CONTROL_SUFFIX) -> list[dict]:
    """Reproduce the combinations split from the additive one.

    scDFM keeps the first 15 test doubles, then moves the singles of every gene
    appearing in those doubles into the test set as well. Those held-out singles
    are the reason a leak-safe cell selection cannot simply keep all singles.
    """
    derived = []
    for fold in additive_folds:
        test_doubles = list(fold["test"][:N_TEST_DOUBLES_FOR_COMBINATIONS])
        held_out_genes = {gene for pair in test_doubles for gene in pair.split("+")}
        held_out_singles = [f"{gene}+{control_suffix}" for gene in held_out_genes]
        derived.append({
            "test_doubles": test_doubles,
            "held_out_genes": held_out_genes,
            "held_out_singles": held_out_singles,
            "test": test_doubles + held_out_singles,
        })
    return derived


def _read_obs(config: dict, needs: tuple[str, ...] = ()):
    """obs from the cache when it can answer, otherwise from the raw file.

    Split validation runs BEFORE the cache is built, so requiring the cache made
    the two steps circular; and a cache built by an earlier run may predate the
    column the split needs, so "exists" is not enough - it has to actually carry
    `needs`. obs is identical in both files apart from the gene axis, so falling
    back to the raw file is always safe.
    """
    import os
    import anndata as ad

    cache = config["data"]["cache_h5ad"]
    if os.path.exists(cache):
        obs = ad.read_h5ad(cache, backed="r").obs
        if all(key in obs.columns for key in needs if key):
            return obs
    return ad.read_h5ad(config["data"]["raw_h5ad"], backed="r").obs


def folds_from_obs(config: dict) -> list[dict[str, Any]]:
    """One fold read out of an obs column, for datasets that ship their split that way.

    combosciplex is the case this exists for: it has no split pickle, but every
    cell carries obs['split'] in {train, test, ood}. Conditions are assigned to
    whichever value holds most of their cells - a condition split across values
    would otherwise land in both train and test.

    There is no k-fold structure here, so the result is a single fold and
    `split.fold` must stay 0.
    """
    split_cfg = config["split"]
    obs = _read_obs(config, (split_cfg["obs_key"],))
    conditions = obs["condition"].astype(str).to_numpy()
    if split_cfg["obs_key"] not in obs.columns:
        raise ValueError(
            f"obs has no column {split_cfg['obs_key']!r}; found {list(obs.columns)[:12]}")
    values = obs[split_cfg["obs_key"]].astype(str).to_numpy()

    assignment: dict[str, str] = {}
    for condition in set(conditions.tolist()):
        mask = conditions == condition
        labels, counts = np.unique(values[mask], return_counts=True)
        assignment[condition] = str(labels[counts.argmax()])

    held = split_cfg["obs_test_value"]
    test = sorted(c for c, v in assignment.items() if v == held)
    train = sorted(c for c, v in assignment.items() if v != held)
    if not test:
        available = sorted(set(assignment.values()))
        available = sorted(set(assignment.values()))
        raise ValueError(
            f"no condition is majority {held!r}. After condition-level "
            f"assignment the available values are {available}. A value that "
            f"exists per CELL but never wins a condition is a within-condition "
            f"cell holdout, not a condition-level split, and this model "
            f"predicts whole conditions - pick one of the values above. "
            f"For combosciplex use split.obs_test_value=ood.")
    return [{"train": train, "test": test, "train_doubles": train}]


# scDFM's held-out conditions for combosciplex, verbatim and in their order, from
# scDFM/src/data_process/data.py (process_data, combosciplex branch). The list is
# hard-coded there and none of their run scripts overrides it, so it IS the test
# set behind their combosciplex table.
SCDFM_COMBOSCIPLEX_TEST = [
    "Panobinostat+Crizotinib",
    "Panobinostat+Curcumin",
    "Panobinostat+SRT1720",
    "Panobinostat+Sorafenib",
    "SRT2104+Alvespimycin",
    "control+Alvespimycin",
    "control+Dacinostat",
]

# Two training combinations held out to make design decisions on combosciplex
# without scoring scDFM's seven. Chosen by the STRUCTURE of the test set, never by
# its scores: every test combination is Panobinostat + X where X appears in
# training only next to Givinostat, another HDAC inhibitor. Both pairs below have
# that shape, and both drugs of each keep training conditions after removal
# (Dasatinib: single, Givinostat+, Dacinostat+; SRT2104: single, Givinostat+;
# Panobinostat keeps four), so they stay combination problems rather than
# unseen-drug problems.
COMBOSCIPLEX_VALIDATION = [
    "Panobinostat+Dasatinib",
    "Panobinostat+SRT2104",
]

# ---- development validation sets (phase 0) ------------------------------------
# Design decisions are made on these and never on a reported test set.
#
# combosciplex: the 12 training combinations whose two drugs both keep a training
# condition when the combination is held out, split into three folds of four by a
# seeded shuffle (random.seed(0)) such that each fold is valid on its own. The two
# test-shaped pairs above sit in folds 0 and 2. The pair itself stays available as
# the legacy set (validation_fold null) so the shared-basis runs reproduce.
COMBOSCIPLEX_VALIDATION_FOLDS = [
    ["Panobinostat+SRT2104", "Panobinostat+Alvespimycin",
     "SRT3025+Cediranib", "Givinostat+Dasatinib"],
    ["Dacinostat+Dasatinib", "Givinostat+Cediranib",
     "Panobinostat+SRT3025", "Panobinostat+PCI-34051"],
    ["Dacinostat+PCI-34051", "Givinostat+SRT2104",
     "Cediranib+PCI-34051", "Panobinostat+Dasatinib"],
]

# The singles-aware folds ("sv"): fold i above plus ONE held-out training single.
# Table 3 scores two single drugs, control+Alvespimycin and control+Dacinostat, and
# neither drug is ever trained alone - each appears only inside 2 and 3 training
# combinations. The folds above score combinations only, so that block, the worst
# one of Table 3, was invisible to validation.
#
# Chosen by structure alone, never by a score: once fold i is also held out, the
# drug must keep exactly 2 training combinations and no training single, as the
# test singles do. Only four drugs have a training single. Givinostat keeps 8
# combinations, far easier than the test; holding out Panobinostat's single would
# unanchor every Panobinostat+X validation combination, which the test
# combinations are not. Dasatinib keeps 2 in every fold, SRT2104 only in fold 1.
# One single per fold keeps three of the four training singles.
#
# Known limits, recorded before any run: two of the three are Dasatinib, whose
# nearest structure is 0.21 ECFP4 Tanimoto (assets/drugs), while both test
# singles have a close analogue; and validation combinations containing Dasatinib
# lose their single anchor, so they are harder than their test counterparts.
COMBOSCIPLEX_VALIDATION_SINGLES = ["control+Dasatinib", "control+SRT2104",
                                   "control+Dasatinib"]

# Norman: the five additive folds are independent 70/30 draws, so nearly every
# double is a test condition somewhere - only PLK4+STIL is scored by no fold of
# Table 1 or Table 2, and no validation set can be clean for every fold. These are
# the 15 training doubles of additive fold 0 (both singles present) that appear the
# FEWEST times as a test double or held-out gene in the other folds of either
# table: 21 appearances in total, 8 in Table 1 and 13 in Table 2. Folds 1-3 tie on
# exposure; fold 1 is avoided because it is the easiest fold and the one earlier
# tuning concentrated on. The residual overlap is a disclosed limitation.
NORMAN_VALIDATION = {
    0: ["PLK4+STIL", "POU3F2+FOXL2", "RHOXF2+ZBTB25", "TMSB4X+BAK1",
        "CDKN1C+CDKN1B", "POU3F2+CBFA2T3", "PRDM1+CBFA2T3", "RHOXF2+SET",
        "BCL2L11+TGFBR2", "MAP2K3+SLC38A2", "TBX3+TBX2", "BCL2L11+BAK1",
        "FOXF1+FOXL2", "FOXL2+MEIS1", "ZNF318+FOXL2"],
}


def norman_validation(reference: list[dict[str, Any]], fold: int) -> list[dict[str, Any]]:
    """The additive folds with `fold` re-split for development.

    The dev fold scores NORMAN_VALIDATION[fold] and excludes its own test doubles
    from training without scoring them, recorded as fold["excluded"]. Every other
    fold is returned untouched.
    """
    if fold not in NORMAN_VALIDATION:
        raise ValueError(f"Norman validation is defined for additive fold(s) "
                         f"{sorted(NORMAN_VALIDATION)}, not fold {fold}")
    validation = list(NORMAN_VALIDATION[fold])
    source = reference[fold]
    missing = [d for d in validation if d not in source["train"]]
    if missing:
        raise ValueError(f"validation doubles are not training doubles of fold {fold}: "
                         f"{missing}")
    held = set(validation)
    derived = list(reference)
    derived[fold] = {"train": [d for d in source["train"] if d not in held],
                     "test": validation, "excluded": list(source["test"])}
    return derived


def folds_from_list(config: dict) -> list[dict[str, Any]]:
    """One fold whose test set is an explicit list of conditions.

    This is scDFM's combosciplex split: a condition is test if it is in the list
    and train otherwise, control always on the train side. The file's own
    obs['split'] column disagrees with that list - six of the seven are 'train'
    there - which is why obs_column cannot reproduce their table.

    `split.test_conditions` null means SCDFM_COMBOSCIPLEX_TEST. A listed condition
    absent from the data raises: scDFM's code would quietly test on fewer
    conditions, and a row averaged over a different set is not comparable.

    Two of scDFM's seven are single drugs, so unlike obs_column this fold holds
    out singles. They are recorded in `held_out_singles`, and
    baselines.training_conditions excludes every test condition explicitly -
    without that, its "every single is trainable" rule would put them back.

    `split.exclude_conditions` are removed from training (and from gene
    selection) without being scored, and recorded as fold["excluded"].
    `split.validation=true` fills the two lists for a design run: the scored set
    becomes COMBOSCIPLEX_VALIDATION and scDFM's seven are excluded, so the model
    never sees the test conditions and nothing scores them.
    `split.validation_singles=true` adds COMBOSCIPLEX_VALIDATION_SINGLES[fold]
    to the scored set of validation fold `split.validation_fold`.
    """
    # Local import: conventions is a leaf module, but keeping it out of module
    # scope means this file's import order can never become circular.
    from .conventions import ConditionNaming

    split_cfg = config["split"]
    with_single = bool(split_cfg.get("validation_singles"))
    if split_cfg.get("validation"):
        chosen = split_cfg.get("validation_fold")
        if chosen is None:
            if with_single:
                raise ValueError("split.validation_singles needs split.validation_fold 0-"
                                 f"{len(COMBOSCIPLEX_VALIDATION_FOLDS) - 1}")
            scored = COMBOSCIPLEX_VALIDATION
        elif 0 <= int(chosen) < len(COMBOSCIPLEX_VALIDATION_FOLDS):
            scored = list(COMBOSCIPLEX_VALIDATION_FOLDS[int(chosen)])
            if with_single:
                scored.append(COMBOSCIPLEX_VALIDATION_SINGLES[int(chosen)])
        else:
            raise ValueError(f"split.validation_fold must be 0-"
                             f"{len(COMBOSCIPLEX_VALIDATION_FOLDS) - 1} or null, "
                             f"got {chosen!r}")
        test = list(split_cfg.get("test_conditions") or scored)
        excluded = list(split_cfg.get("exclude_conditions") or SCDFM_COMBOSCIPLEX_TEST)
    elif with_single:
        raise ValueError("split.validation_singles needs split.validation=true")
    else:
        test = list(split_cfg.get("test_conditions") or SCDFM_COMBOSCIPLEX_TEST)
        excluded = list(split_cfg.get("exclude_conditions") or [])
    for name, values in (("test_conditions", test), ("exclude_conditions", excluded)):
        if len(set(values)) != len(values):
            raise ValueError(f"split.{name} repeats a condition: {values}")
    both = set(test) & set(excluded)
    if both:
        raise ValueError(f"conditions are both scored and excluded: {sorted(both)}")
    present = set(_read_obs(config)["condition"].astype(str).tolist())
    missing = [c for c in test + excluded if c not in present]
    if missing:
        raise ValueError(f"the split names conditions absent from the data: {missing}")
    naming = ConditionNaming.from_config(config)
    held = set(test) | set(excluded)
    train = sorted(c for c in present if c not in held)
    return [{"train": train, "test": test, "excluded": excluded,
             "train_doubles": [c for c in train if naming.is_double(c)],
             "held_out_singles": [c for c in test + excluded if naming.is_single(c)]}]


def folds_generated(config: dict) -> list[dict[str, Any]]:
    """Make a split for a dataset that ships none.

    Deterministic from `split.generate_seed` and written nowhere: the folds are
    recomputed on every load, so there is no artifact that can drift away from the
    seed that produced it - the same reason the combinations folds are derived
    rather than cached.

    Uses the legacy RandomState rather than default_rng. numpy does not guarantee
    that default_rng's stream is stable across versions, and a split that quietly
    changes when numpy is upgraded would invalidate every number measured against
    it.

    Three schemes:

      doubles       hold out a fraction of the double perturbations. The singles
                    stay available, so the additive baselines remain computable.
      combinations  additionally hold out the singles of every held-out double.
                    Harder, and the additive baselines become uncomputable, which
                    is the point.
      group         hold out whole groups named by an obs column - one cell line,
                    one donor, one batch. A different generalisation axis from the
                    two above, and the one to use for "does this transfer to a
                    cell line the model never saw".
    """
    split_cfg = config["split"]
    naming = _naming(config)
    needs = (split_cfg["group_key"],) if split_cfg["generate_scheme"] == "group" else ()
    obs = _read_obs(config, needs)
    conditions = obs["condition"].astype(str).to_numpy()

    scheme = split_cfg["generate_scheme"]
    n_folds = split_cfg["n_folds"]
    fraction = split_cfg["generate_test_fraction"]
    seed = split_cfg["generate_seed"]

    if scheme == "group":
        key = split_cfg["group_key"]
        if key not in obs.columns:
            raise ValueError(
                f"obs has no column {key!r}; set split.group_key to the column that "
                f"names the cell line / donor / batch")
        groups = obs[key].astype(str).to_numpy()
        unique = sorted(set(groups.tolist()))
        if len(unique) < 2:
            raise ValueError(f"obs[{key!r}] has one group; nothing to hold out")
        generated = []
        for fold in range(n_folds):
            order = list(unique)
            np.random.RandomState(seed + fold).shuffle(order)
            n_test = max(1, int(round(len(unique) * fraction)))
            held_groups = set(order[:n_test])
            test_cells = np.isin(groups, list(held_groups))
            generated.append({
                "train": sorted(set(conditions[~test_cells].tolist())),
                "test": sorted(set(conditions[test_cells].tolist())),
                "held_out_groups": sorted(held_groups),
                "group_key": key,
            })
        return generated

    doubles = sorted({c for c in set(conditions.tolist()) if naming.is_double(c)})
    if not doubles:
        raise ValueError("no double-perturbation conditions found; nothing to hold out")

    generated = []
    for fold in range(n_folds):
        order = list(doubles)
        np.random.RandomState(seed + fold).shuffle(order)
        n_test = max(1, int(round(len(doubles) * fraction)))
        test_doubles = sorted(order[:n_test])
        train_doubles = sorted(order[n_test:])

        entry: dict[str, Any] = {"test_doubles": test_doubles,
                                 "train_doubles": train_doubles}
        if scheme == "combinations":
            genes = {g for pair in test_doubles for g in naming.genes(pair)}
            held_singles = [naming.single(g) for g in sorted(genes)]
            entry["held_out_genes"] = sorted(genes)
            entry["held_out_singles"] = held_singles
            entry["test"] = test_doubles + held_singles
            entry["train"] = train_doubles
        elif scheme == "doubles":
            entry["test"] = test_doubles
            entry["train"] = train_doubles
        else:
            raise ValueError(f"unknown split.generate_scheme {scheme!r}")
        generated.append(entry)
    return generated


def _naming(config: dict):
    from .conventions import ConditionNaming
    return ConditionNaming.from_config(config)


def folds(config: dict, method: str | None = None) -> list[dict[str, Any]]:
    """The folds for `method`.

    Two sources, chosen by `split.source`:

      reference_pkl  the file that ships with the dataset (Norman). additive is
                     that file verbatim; combinations is derived from it.
      obs_column     a column of the cache's obs (combosciplex). One fold, and
                     `method` is ignored because the dataset defines its own.

    additive     : exactly the scDFM reference.
    combinations : derived from it on the spot. Caching this to disk would create
                   a second artifact that can silently disagree with its source.
    """
    method = method or config["split"]["method"]
    source = config["split"].get("source", "reference_pkl")
    if source == "obs_column":
        return folds_from_obs(config)
    if source == "list":
        return folds_from_list(config)
    if source == "generated":
        return folds_generated(config)
    reference = load(config["split"]["reference_pkl"])
    if config["split"].get("validation"):
        if method != "additive":
            raise ValueError("split.validation on the reference split is defined for "
                             "method=additive only")
        return norman_validation(reference, int(config["split"]["fold"]))
    if method == "additive":
        return reference
    if method == "combinations":
        derived = derive_combinations(reference)
        for fold, source in zip(derived, reference):
            # combinations holds out only the FIRST 15 test doubles, so every
            # other double is trainable - including the ones the additive fold
            # put in test. Deriving train from the additive train instead would
            # silently shrink it from 110 doubles to 88.
            all_doubles = list(source["train"]) + list(source["test"])
            held = set(fold["test"])
            fold["train_doubles"] = [c for c in all_doubles if c not in held]
            # Singles are added by the caller, which knows which ones the data
            # actually contains; see baselines.training_conditions.
            fold["train"] = list(fold["train_doubles"])
        return derived
    raise ValueError(f"unknown split method {method!r}")


def validate(config: dict) -> dict[str, Any]:
    """Sanity-check the split source. Raises on anything inconsistent."""
    source = config["split"].get("source", "reference_pkl")
    if source == "generated":
        generated = folds_generated(config)
        for i, fold in enumerate(generated):
            overlap = set(fold["train"]) & set(fold["test"])
            if overlap:
                raise ValueError(f"generated fold {i} puts {overlap} in both sides")
        return {"n_folds": len(generated), "reference_ok": True,
                "combinations_derived": False, "source": "generated",
                "additive_sizes": [(len(f["train"]), len(f["test"])) for f in generated],
                "combinations_sizes": [],
                "additive_test_pairwise_overlap": (0, 0),
                "doubles_train_in_every_fold": len(
                    set.intersection(*[set(f["train"]) for f in generated]))}

    if source == "list":
        fold = folds_from_list(config)[0]
        overlap = set(fold["train"]) & (set(fold["test"]) | set(fold["excluded"]))
        if overlap:
            raise ValueError(f"list split puts {overlap} in both train and test")
        return {"n_folds": 1, "reference_ok": True, "combinations_derived": False,
                "additive_sizes": [(len(fold["train"]), len(fold["test"]))],
                "combinations_sizes": [], "source": "list",
                "additive_test_pairwise_overlap": (0, 0),
                "doubles_train_in_every_fold": len(fold["train_doubles"])}

    if source == "obs_column":
        fold = folds_from_obs(config)[0]
        overlap = set(fold["train"]) & set(fold["test"])
        if overlap:
            raise ValueError(f"obs split puts {overlap} in both train and test")
        return {"n_folds": 1, "reference_ok": True, "combinations_derived": False,
                "additive_sizes": [(len(fold["train"]), len(fold["test"]))],
                "combinations_sizes": [], "source": "obs_column",
                "additive_test_pairwise_overlap": (0, 0),
                "doubles_train_in_every_fold": len(fold["train"])}

    reference = load(config["split"]["reference_pkl"])
    report: dict[str, Any] = {"n_folds": len(reference)}

    for i, fold in enumerate(reference):
        if not {"train", "test"} <= set(fold):
            raise ValueError(f"fold {i} is missing train/test")
        if set(fold["train"]) & set(fold["test"]):
            raise ValueError(f"fold {i} has a condition in both train and test")
    report["reference_ok"] = True
    report["additive_sizes"] = [(len(f["train"]), len(f["test"])) for f in reference]

    derived = derive_combinations(reference)
    report["combinations_sizes"] = [
        (len(f["test_doubles"]), len(f["held_out_singles"])) for f in derived
    ]
    for i, fold in enumerate(derived):
        expected = {g for pair in fold["test_doubles"] for g in pair.split("+")}
        if set(fold["held_out_genes"]) != expected:
            raise ValueError(f"fold {i} derivation is inconsistent")
    report["combinations_derived"] = True

    test_sets = [set(f["test"]) for f in reference]
    overlaps = [
        len(test_sets[i] & test_sets[j])
        for i in range(len(test_sets))
        for j in range(i + 1, len(test_sets))
    ]
    report["additive_test_pairwise_overlap"] = (min(overlaps), max(overlaps))
    report["doubles_train_in_every_fold"] = len(
        set.intersection(*[set(f["train"]) for f in reference])
    )
    return report


def held_out_conditions(config: dict) -> set[str]:
    """Every condition held out by ANY fold of ANY split method.

    This is the exclusion set for anything that must not see test information at
    all - GRN inference above all. It is deliberately conservative: one artifact
    that is safe everywhere beats per-split artifacts that can be mixed up.
    """
    excluded: set[str] = set()
    # Validation re-splits one development fold and is undefined for combinations
    # on the reference split, so the tables' own folds are read with it switched
    # off and the development fold is added on top. validation_singles is part of
    # the development split and is switched off with it.
    plain = {**config, "split": {**config["split"], "validation": False,
                                 "validation_singles": False}}
    for method in ("additive", "combinations"):
        for fold in folds(plain, method):
            excluded |= set(fold["test"]) | set(fold.get("excluded", ()))
    if config["split"].get("validation"):
        for fold in folds(config, "additive"):
            excluded |= set(fold["test"]) | set(fold.get("excluded", ()))
    return excluded
