"""The split must be bit-for-bit the one that ships with the dataset.

This is a hard requirement, not a preference: the whole point of inheriting
scDFM's folds is that "the authors' method won on the authors' split" cannot be
raised against us. A split that drifts - by a rename, a re-sort, a cached copy, or
a change in the gene space - silently destroys that argument, and nothing in the
training loop would notice.

The gene space is allowed to differ from scDFM's. The split is not.

    python -m pytest tests/test_splits.py -v
"""

from __future__ import annotations

import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import splits


@pytest.fixture(scope="module")
def config():
    return config_module.load()


@pytest.fixture(scope="module")
def reference(config):
    with open(config["split"]["reference_pkl"], "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------- identity
def test_additive_is_the_shipped_file_verbatim(config, reference):
    """Not "equivalent", not "same set" - the same lists in the same order."""
    loaded = splits.folds(config, "additive")
    assert len(loaded) == len(reference)
    for i, (ours, theirs) in enumerate(zip(loaded, reference)):
        assert list(ours["train"]) == list(theirs["train"]), f"fold {i} train order differs"
        assert list(ours["test"]) == list(theirs["test"]), f"fold {i} test order differs"


def test_no_cached_split_artifact_exists():
    """A copy on disk can disagree with its source and nothing would notice."""
    for stale in ("assets/splits_additive.pkl", "assets/splits_combinations.pkl"):
        assert not os.path.exists(stale), (
            f"{stale} is back; splits must be read from the shipped reference only")


def test_folds_are_not_mutated_between_calls(config):
    """Callers hold real dicts; a mutation must not leak into the next load."""
    first = splits.folds(config, "additive")
    first[0]["train"].append("SENTINEL+SENTINEL")
    second = splits.folds(config, "additive")
    assert "SENTINEL+SENTINEL" not in second[0]["train"]


# ---------------------------------------------------------------- independence
@pytest.mark.parametrize("n_hvg", [1000, 3000, 5000, None])
def test_gene_space_does_not_touch_the_split(n_hvg, reference):
    """The gene space is a free choice; the split must be invariant to it."""
    altered = config_module.load([f"data.n_hvg={'null' if n_hvg is None else n_hvg}"])
    loaded = splits.folds(altered, "additive")
    for ours, theirs in zip(loaded, reference):
        assert list(ours["train"]) == list(theirs["train"])
        assert list(ours["test"]) == list(theirs["test"])


# ---------------------------------------------------------------- structure
def test_train_and_test_are_disjoint(config):
    for method in ("additive", "combinations"):
        for i, fold in enumerate(splits.folds(config, method)):
            overlap = set(fold["train"]) & set(fold["test"])
            assert not overlap, f"{method} fold {i} shares {overlap}"


def test_combinations_holds_out_the_singles_of_its_test_doubles(config):
    """This is what distinguishes the split, and what a naive baseline leaks through."""
    for i, fold in enumerate(splits.folds(config, "combinations")):
        genes = {g for pair in fold["test_doubles"] for g in pair.split("+")}
        assert set(fold["held_out_genes"]) == genes, f"fold {i}"
        assert set(fold["held_out_singles"]) == {f"{g}+ctrl" for g in genes}, f"fold {i}"
        assert set(fold["held_out_singles"]) <= set(fold["test"]), f"fold {i}"
        assert not set(fold["held_out_singles"]) & set(fold["train"]), (
            f"fold {i} keeps a held-out single in train")


def test_combinations_train_keeps_every_double_it_is_allowed(config, reference):
    """combinations holds out only 15 doubles, so 110 of the 125 stay trainable.

    Deriving this from the ADDITIVE train list instead gives 88 and drops the
    singles too, which lowers the ridge-additive target line from 0.1853 to
    0.0487 - i.e. it makes the bar easier without anything looking wrong. The
    numbers below are the ones the shipped combinations split had.
    """
    all_doubles = set(reference[0]["train"]) | set(reference[0]["test"])
    assert len(all_doubles) == 125

    for i, fold in enumerate(splits.folds(config, "combinations")):
        assert len(fold["test_doubles"]) == 15, f"fold {i}"
        assert len(fold["train_doubles"]) == 110, (
            f"fold {i} has {len(fold['train_doubles'])} train doubles, expected 110")
        assert not set(fold["train_doubles"]) & set(fold["test_doubles"]), f"fold {i}"


def test_combinations_training_conditions_include_the_surviving_singles(config):
    """Held-out singles must go, the rest must stay - 101 - held_out of them."""
    cache = config["data"]["cache_h5ad"]
    if not os.path.exists(cache):
        pytest.skip(f"{cache} not built")
    import anndata as ad
    import numpy as np
    from src.eval import baselines

    adata = ad.read_h5ad(cache)
    conditions = adata.obs["condition"].astype(str).to_numpy()
    stats = baselines.ConditionMeans(np.zeros((len(conditions), 1), dtype=np.float32),
                                     conditions)
    n_singles = sum(1 for c in stats.mean if c.endswith("+ctrl"))

    for i, fold in enumerate(splits.folds(config, "combinations")):
        allowed = baselines.training_conditions(stats, fold, "combinations")
        held = set(fold["held_out_singles"])
        assert not set(allowed) & held, f"fold {i} lets a held-out single into train"
        singles_kept = sum(1 for c in allowed if c.endswith("+ctrl"))
        assert singles_kept == n_singles - len(held), (
            f"fold {i} kept {singles_kept} singles, expected {n_singles - len(held)}")


def test_combinations_derives_from_the_same_reference(config, reference):
    derived = splits.folds(config, "combinations")
    for i, (fold, source) in enumerate(zip(derived, reference)):
        assert list(fold["test_doubles"]) == list(source["test"][:15]), f"fold {i}"


def test_folds_overlap_because_they_are_independent_draws(config):
    """Five 70/30 draws, NOT a five-way partition.

    Code that assumes a partition - leak-free cell selection above all - would be
    wrong, so the property is pinned here rather than left as a comment.
    """
    test_sets = [set(f["test"]) for f in splits.folds(config, "additive")]
    overlaps = [len(test_sets[i] & test_sets[j])
                for i in range(len(test_sets)) for j in range(i + 1, len(test_sets))]
    assert min(overlaps) > 0, "folds look like a partition; the split source changed"


# ---------------------------------------------------------------- generated splits
@pytest.mark.parametrize("scheme", ["doubles", "combinations"])
def test_generated_split_is_deterministic(scheme):
    """Same seed, same folds - it is recomputed on every load, never cached."""
    base = config_module.load(["split.source=generated",
                               f"split.generate_scheme={scheme}"])
    first = splits.folds(base)
    second = splits.folds(base)
    for a, b in zip(first, second):
        assert list(a["test"]) == list(b["test"])
        assert list(a["train"]) == list(b["train"])


def test_generated_split_moves_with_the_seed():
    base = config_module.load(["split.source=generated"])
    if not os.path.exists(base["data"]["cache_h5ad"]):
        pytest.skip("cache not built")
    other = config_module.load(["split.source=generated", "split.generate_seed=7"])
    assert splits.folds(base)[0]["test"] != splits.folds(other)[0]["test"]


def test_generated_combinations_holds_out_the_constituent_singles():
    """The scheme that makes the additive baselines uncomputable must actually do it."""
    config = config_module.load(["split.source=generated",
                                 "split.generate_scheme=combinations"])
    if not os.path.exists(config["data"]["cache_h5ad"]):
        pytest.skip("cache not built")
    for i, fold in enumerate(splits.folds(config)):
        genes = {g for pair in fold["test_doubles"] for g in pair.split("+")}
        assert set(fold["held_out_genes"]) == genes, f"fold {i}"
        assert set(fold["held_out_singles"]) <= set(fold["test"]), f"fold {i}"
        assert not set(fold["held_out_singles"]) & set(fold["train"]), f"fold {i}"


def test_generated_split_never_shares_a_condition():
    for scheme in ("doubles", "combinations"):
        config = config_module.load(["split.source=generated",
                                     f"split.generate_scheme={scheme}"])
        if not os.path.exists(config["data"]["cache_h5ad"]):
            pytest.skip("cache not built")
        for i, fold in enumerate(splits.folds(config)):
            assert not set(fold["train"]) & set(fold["test"]), f"{scheme} fold {i}"


# ---------------------------------------------------------------- data agreement
def test_every_split_condition_exists_in_the_data(config):
    """A condition named by the split but absent from the cache is silently dropped."""
    cache = config["data"]["cache_h5ad"]
    if not os.path.exists(cache):
        pytest.skip(f"{cache} not built")
    import anndata as ad
    conditions = set(ad.read_h5ad(cache, backed="r").obs["condition"].astype(str))
    for method in ("additive", "combinations"):
        for i, fold in enumerate(splits.folds(config, method)):
            for key in ("train", "test"):
                missing = set(fold[key]) - conditions
                assert not missing, f"{method} fold {i} {key} names absent conditions: {missing}"


# ---------------------------------------------------------------- explicit list (combosciplex)
COMBOSCIPLEX_RAW = "data/combosciplex/combosciplex.h5ad"
SCDFM_SINGLE_DRUGS = ["control+Alvespimycin", "control+Dacinostat"]


@pytest.fixture(scope="module")
def combosciplex_config():
    if not os.path.exists(COMBOSCIPLEX_RAW):
        pytest.skip(f"{COMBOSCIPLEX_RAW} not present")
    # The cache path points nowhere on purpose, so the split is read from the raw
    # file's obs - the same obs any cache built from it would carry.
    return config_module.load([f"data.raw_h5ad={COMBOSCIPLEX_RAW}",
                               "data.cache_h5ad=assets/__no_cache_for_split_tests__.h5ad",
                               "data.control_label=control",
                               "split.source=list"])


@pytest.fixture(scope="module")
def combosciplex_stats(combosciplex_config):
    import anndata as ad
    import numpy as np
    from src.data.conventions import ConditionNaming
    from src.eval import baselines

    conditions = ad.read_h5ad(COMBOSCIPLEX_RAW, backed="r").obs["condition"].astype(str).to_numpy()
    return baselines.ConditionMeans(np.zeros((len(conditions), 1), dtype=np.float32),
                                    conditions, ConditionNaming.from_config(combosciplex_config))


def test_list_default_is_scdfm_combosciplex_seven_verbatim(combosciplex_config):
    """Their table's test set, in their order - not obs['split'] == 'ood'."""
    loaded = splits.folds(combosciplex_config)
    assert len(loaded) == 1
    assert loaded[0]["test"] == splits.SCDFM_COMBOSCIPLEX_TEST


def test_list_split_partitions_the_data(combosciplex_config):
    import anndata as ad
    present = set(ad.read_h5ad(COMBOSCIPLEX_RAW, backed="r").obs["condition"].astype(str))
    fold = splits.folds(combosciplex_config)[0]
    assert not set(fold["train"]) & set(fold["test"])
    assert set(fold["train"]) | set(fold["test"]) == present


def test_list_records_its_single_drug_holdouts(combosciplex_config):
    fold = splits.folds(combosciplex_config)[0]
    assert fold["held_out_singles"] == SCDFM_SINGLE_DRUGS
    assert set(fold["held_out_singles"]) <= set(fold["test"])
    assert not set(fold["train_doubles"]) & set(fold["test"])


@pytest.mark.parametrize("method", ["additive", "combinations"])
def test_list_training_conditions_never_include_a_test_condition(
        combosciplex_config, combosciplex_stats, method):
    """The leak this source would open without the explicit exclusion.

    run.sh passes no split.method for combosciplex, so training runs under the
    additive rule "train list + every single" - which, before the exclusion,
    handed both of scDFM's single-drug test conditions back to training.
    """
    from src.eval import baselines

    fold = splits.folds(combosciplex_config)[0]
    allowed = baselines.training_conditions(combosciplex_stats, fold, method)
    assert not set(allowed) & set(fold["test"]), f"{method} trains on a test condition"
    assert len(allowed) == len(set(allowed)), f"{method} lists a condition twice"
    naming = combosciplex_stats.naming
    trainable_singles = {c for c in combosciplex_stats.mean
                         if naming.is_single(c) and c not in fold["test"]}
    assert trainable_singles <= set(allowed), f"{method} dropped a trainable single"


def test_list_rejects_a_condition_absent_from_the_data(combosciplex_config):
    import copy
    broken = copy.deepcopy(combosciplex_config)
    broken["split"]["test_conditions"] = ["NotADrug+control"]
    with pytest.raises(ValueError, match="absent from the data"):
        splits.folds(broken)


def test_validate_accepts_the_list_source(combosciplex_config):
    report = splits.validate(combosciplex_config)
    assert report["source"] == "list" and report["n_folds"] == 1


# ---------------------------------------------------------------- norman is unchanged
def test_additive_training_conditions_are_train_doubles_then_every_single(config):
    """The explicit test exclusion must not change Norman's additive list at all."""
    cache = config["data"]["cache_h5ad"]
    if not os.path.exists(cache):
        pytest.skip(f"{cache} not built")
    import anndata as ad
    import numpy as np
    from src.eval import baselines

    conditions = ad.read_h5ad(cache, backed="r").obs["condition"].astype(str).to_numpy()
    stats = baselines.ConditionMeans(np.zeros((len(conditions), 1), dtype=np.float32),
                                     conditions)
    singles = [c for c in stats.mean if stats.naming.is_single(c)]
    for i, fold in enumerate(splits.folds(config, "additive")):
        allowed = baselines.training_conditions(stats, fold, "additive")
        assert allowed == list(fold["train"]) + singles, f"fold {i}"


# ---------------------------------------------------------------- combosciplex validation split
@pytest.fixture(scope="module")
def combosciplex_validation_config(combosciplex_config):
    import copy
    validation = copy.deepcopy(combosciplex_config)
    validation["split"]["validation"] = True
    return validation


def test_validation_scores_the_pair_and_excludes_the_test_seven(combosciplex_validation_config):
    fold = splits.folds(combosciplex_validation_config)[0]
    assert fold["test"] == splits.COMBOSCIPLEX_VALIDATION
    assert fold["excluded"] == splits.SCDFM_COMBOSCIPLEX_TEST
    assert not (set(fold["test"]) | set(fold["excluded"])) & set(fold["train"])


@pytest.mark.parametrize("method", ["additive", "combinations"])
def test_validation_training_never_sees_test_or_validation(
        combosciplex_validation_config, combosciplex_stats, method):
    from src.eval import baselines

    fold = splits.folds(combosciplex_validation_config)[0]
    allowed = baselines.training_conditions(combosciplex_stats, fold, method)
    assert not set(allowed) & (set(fold["test"]) | set(fold["excluded"])), method
    # Both held-out single drugs of the test seven, not just the combinations.
    assert not set(allowed) & set(SCDFM_SINGLE_DRUGS), method


def test_validation_pairs_stay_combination_problems(combosciplex_validation_config):
    """Every drug of a validation pair keeps a training condition, as every drug
    of the test combinations does - otherwise it would be an unseen-drug test."""
    fold = splits.folds(combosciplex_validation_config)[0]
    for pair in fold["test"]:
        for drug in pair.split("+"):
            assert any(drug in c.split("+") for c in fold["train"]), drug


def test_list_rejects_a_condition_both_scored_and_excluded(combosciplex_config):
    import copy
    broken = copy.deepcopy(combosciplex_config)
    broken["split"]["exclude_conditions"] = [splits.SCDFM_COMBOSCIPLEX_TEST[0]]
    with pytest.raises(ValueError, match="both scored and excluded"):
        splits.folds(broken)


def test_held_out_conditions_cover_the_excluded_ones(combosciplex_validation_config):
    held = splits.held_out_conditions(combosciplex_validation_config)
    assert set(splits.SCDFM_COMBOSCIPLEX_TEST) <= held
    assert set(splits.COMBOSCIPLEX_VALIDATION) <= held


# ---------------------------------------------------------------- development validation sets
def _norman_validation_config(config, fold=0):
    import copy
    validation = copy.deepcopy(config)
    validation["split"]["validation"] = True
    validation["split"]["fold"] = fold
    validation["split"]["method"] = "additive"
    return validation


def test_norman_validation_resplits_only_fold_zero(config, reference):
    loaded = splits.folds(_norman_validation_config(config), "additive")
    dev = loaded[0]
    assert dev["test"] == splits.NORMAN_VALIDATION[0]
    assert dev["excluded"] == list(reference[0]["test"])
    assert not set(dev["test"]) & set(dev["train"])
    assert not set(dev["excluded"]) & set(dev["train"])
    assert set(dev["train"]) | set(dev["test"]) == set(reference[0]["train"])
    for i in range(1, len(reference)):
        assert list(loaded[i]["test"]) == list(reference[i]["test"]), f"fold {i} changed"


def test_norman_validation_leaves_the_reference_untouched(config, reference):
    splits.folds(_norman_validation_config(config), "additive")
    again = splits.folds(config, "additive")
    assert list(again[0]["test"]) == list(reference[0]["test"])


def test_norman_validation_is_defined_for_fold_zero_only(config):
    with pytest.raises(ValueError, match="fold"):
        splits.folds(_norman_validation_config(config, fold=1), "additive")


def test_norman_validation_refuses_combinations(config):
    with pytest.raises(ValueError, match="additive only"):
        splits.folds(_norman_validation_config(config), "combinations")


@pytest.mark.parametrize("method", ["additive"])
def test_norman_validation_training_never_sees_validation_or_test(config, method):
    cache = config["data"]["cache_h5ad"]
    if not os.path.exists(cache):
        pytest.skip(f"{cache} not built")
    import anndata as ad
    import numpy as np
    from src.eval import baselines

    conditions = ad.read_h5ad(cache, backed="r").obs["condition"].astype(str).to_numpy()
    stats = baselines.ConditionMeans(np.zeros((len(conditions), 1), dtype=np.float32),
                                     conditions)
    fold = splits.folds(_norman_validation_config(config), method)[0]
    allowed = baselines.training_conditions(stats, fold, method)
    assert not set(allowed) & (set(fold["test"]) | set(fold["excluded"]))


def test_norman_validation_doubles_are_scorable(config):
    """Every validation double and both of its singles exist in the data, so the
    additive reference and resid_R2 are defined for all fifteen."""
    raw = config["data"]["raw_h5ad"]
    if not os.path.exists(raw):
        pytest.skip(f"{raw} not present")
    import anndata as ad
    present = set(ad.read_h5ad(raw, backed="r").obs["condition"].astype(str))
    for double in splits.NORMAN_VALIDATION[0]:
        assert double in present, double
        for gene in double.split("+"):
            assert f"{gene}+ctrl" in present, gene


def test_norman_held_out_conditions_cover_validation_and_tables(config):
    held = splits.held_out_conditions(_norman_validation_config(config))
    assert set(splits.NORMAN_VALIDATION[0]) <= held
    assert splits.held_out_conditions(config) <= held


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_combosciplex_validation_fold_selects_its_set(combosciplex_validation_config, fold):
    import copy
    chosen = copy.deepcopy(combosciplex_validation_config)
    chosen["split"]["validation_fold"] = fold
    loaded = splits.folds(chosen)[0]
    assert loaded["test"] == splits.COMBOSCIPLEX_VALIDATION_FOLDS[fold]
    assert loaded["excluded"] == splits.SCDFM_COMBOSCIPLEX_TEST
    assert not (set(loaded["test"]) | set(loaded["excluded"])) & set(loaded["train"])
    for pair in loaded["test"]:
        for drug in pair.split("+"):
            assert any(drug in c.split("+") for c in loaded["train"]), (fold, drug)


def test_combosciplex_validation_folds_are_disjoint_and_clear_of_the_test_set():
    flat = [c for fold in splits.COMBOSCIPLEX_VALIDATION_FOLDS for c in fold]
    assert len(flat) == len(set(flat)) == 12
    assert not set(flat) & set(splits.SCDFM_COMBOSCIPLEX_TEST)
    assert set(splits.COMBOSCIPLEX_VALIDATION) <= set(flat)


def test_combosciplex_validation_fold_out_of_range(combosciplex_validation_config):
    import copy
    broken = copy.deepcopy(combosciplex_validation_config)
    broken["split"]["validation_fold"] = 3
    with pytest.raises(ValueError, match="validation_fold"):
        splits.folds(broken)


# ---------------------------------------------------------------- singles-aware (sv) folds
def _sv_fold(combosciplex_validation_config, fold):
    import copy
    chosen = copy.deepcopy(combosciplex_validation_config)
    chosen["split"]["validation_fold"] = fold
    chosen["split"]["validation_singles"] = True
    return splits.folds(chosen)[0]


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_sv_fold_is_its_cv_fold_plus_one_single(combosciplex_validation_config, fold):
    loaded = _sv_fold(combosciplex_validation_config, fold)
    single = splits.COMBOSCIPLEX_VALIDATION_SINGLES[fold]
    assert loaded["test"] == splits.COMBOSCIPLEX_VALIDATION_FOLDS[fold] + [single]
    assert loaded["excluded"] == splits.SCDFM_COMBOSCIPLEX_TEST
    assert single in loaded["held_out_singles"]
    assert not (set(loaded["test"]) | set(loaded["excluded"])) & set(loaded["train"])


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_sv_single_has_the_shape_of_a_test_single(combosciplex_validation_config, fold):
    """The pre-registered structure: the held-out drug is never trained alone and
    stays in exactly two training combinations, as Alvespimycin (2) and
    Dacinostat (3) do in Table 3."""
    loaded = _sv_fold(combosciplex_validation_config, fold)
    drug = splits.COMBOSCIPLEX_VALIDATION_SINGLES[fold].split("+")[1]
    combos = [c for c in loaded["train"]
              if drug in c.split("+") and not c.startswith("control+")]
    assert len(combos) == 2, (fold, combos)
    assert f"control+{drug}" not in loaded["train"]


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_sv_training_never_sees_its_single(combosciplex_validation_config,
                                           combosciplex_stats, fold):
    from src.eval import baselines

    loaded = _sv_fold(combosciplex_validation_config, fold)
    for method in ("additive", "combinations"):
        allowed = baselines.training_conditions(combosciplex_stats, loaded, method)
        assert splits.COMBOSCIPLEX_VALIDATION_SINGLES[fold] not in allowed, method
        assert not set(allowed) & (set(loaded["test"]) | set(loaded["excluded"])), method


def test_sv_does_not_change_the_cv_folds(combosciplex_validation_config):
    import copy
    for fold in range(3):
        chosen = copy.deepcopy(combosciplex_validation_config)
        chosen["split"]["validation_fold"] = fold
        assert splits.folds(chosen)[0]["test"] == splits.COMBOSCIPLEX_VALIDATION_FOLDS[fold]
    assert all(len(f) == 4 for f in splits.COMBOSCIPLEX_VALIDATION_FOLDS)


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_held_out_conditions_cover_the_sv_single(combosciplex_validation_config, fold):
    """build_data calls this on the sv config; it must not trip the sv guard and
    must list the held-out single with the rest."""
    import copy
    chosen = copy.deepcopy(combosciplex_validation_config)
    chosen["split"]["validation_fold"] = fold
    chosen["split"]["validation_singles"] = True
    held = splits.held_out_conditions(chosen)
    assert splits.COMBOSCIPLEX_VALIDATION_SINGLES[fold] in held
    assert set(splits.COMBOSCIPLEX_VALIDATION_FOLDS[fold]) <= held
    assert set(splits.SCDFM_COMBOSCIPLEX_TEST) <= held


def test_validation_singles_needs_a_validation_fold(combosciplex_config,
                                                    combosciplex_validation_config):
    import copy
    no_fold = copy.deepcopy(combosciplex_validation_config)
    no_fold["split"]["validation_singles"] = True
    with pytest.raises(ValueError, match="validation_fold"):
        splits.folds(no_fold)
    not_validation = copy.deepcopy(combosciplex_config)
    not_validation["split"]["validation_singles"] = True
    with pytest.raises(ValueError, match="validation=true"):
        splits.folds(not_validation)
