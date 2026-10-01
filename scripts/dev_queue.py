"""Plan a development experiment as one sequential shell queue per GPU.

    python scripts/dev_queue.py --name w1 --datasets combosciplex norman \
        --arms base= g10="train.gene_endpoint_weight=10" e1="train.endpoint_weight=1" \
        --seeds 0 1 2 --gpus 0 1 3

then, one tmux window per GPU:

    sh results/dev/w1/gpu0.sh

EVERY RUN IS A DEVELOPMENT RUN. Norman scores its fold-0 validation set
(splits.NORMAN_VALIDATION), combosciplex each of its validation folds
(COMBOSCIPLEX_VALIDATION_FOLDS), and --combo-set sv adds one held-out SINGLE per fold.
No reported test set is ever scored here. That discipline is the reason this repository
exists: scPKFM scored its test set thirteen times while selecting, and its seed noise
(0.28 L2 on combosciplex) turned out to exceed every effect it was reading.

ARMS ARE CONFIG OVERRIDES, NOT SHELL FLAGS. scPKFM's version emitted run.sh flags which
run.sh then translated into config keys, and the translation layer is what silently
changed a cache name (`_hvgall`) so that reusing an encoder was refused. There is nothing
to translate here: an arm is a `key=value` list handed straight to
`scripts/train.py --set`.

THERE ARE NO ENCODER JOBS. scPKFM trained one encoder per (dataset, fold) and had every
arm load it, so a difference between two arms was a difference in stage 2 alone. This
model has one stage and no autoencoder, so every job is a whole run - and the thing that
made encoder sharing necessary is gone: the additive component every arm starts from is a
closed-form ridge fit, identical across arms by construction rather than by sharing.

WHY THE ARMS ARE LIKELY TO DIFFER BY DATASET. Two smoke runs measured opposite deficits.
On combosciplex the residual was well aligned and badly scaled (cos 0.55, norm ratio 0.15
against an optimum of 0.55); on Norman it was nearly optimally scaled and less aligned
(cos 0.31, ratio 0.22 against an optimum of 0.31). One weight set is unlikely to fix both,
so --arms-for lets an arm be defined per dataset and the score script pools within a
dataset family anyway.
"""

from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = re.compile(r"^[A-Za-z0-9]+$")
COMBO_SETS = ("cv", "sv")
# Which Norman regime a queue scores. `nval` is method=additive: every single is in
# training, which is Table 1 - the table ridge already WINS (1.669 against scDFM's 1.704),
# so a learned model has 0.035 to play for there. `ncomb` is method=combinations, holding
# out the singles of every scored double, which is Table 2 Double - one of the only two
# blocks where scDFM beats ridge at all (+0.220, and Table 3's +0.201 is the other). Every
# Norman run before this option used nval, so the harder of the two had never been measured.
NORMAN_SETS = ("nval", "ncomb")
COMBOSCIPLEX_FOLDS = (0, 1, 2)

# The configuration every arm starts from, as config overrides. Kept here rather than in
# src/config.py's defaults because these are the DATA choices a development run shares,
# not the model's defaults - a reader of a generated queue should see them.
DATASETS = {
    "combosciplex": [
        "data.cache_h5ad=assets/combosciplex_dev.h5ad",
        "data.raw_h5ad=data/combosciplex/combosciplex.h5ad",
        "data.control_label=control",
        "data.normalise_from_counts=counts",
        "data.n_hvg=5000", "data.hvg_criterion=scanpy",
        "split.source=list", "split.fold=0", "split.validation=true",
    ],
    "norman": [
        "data.cache_h5ad=assets/norman_dev.h5ad",
        "data.raw_h5ad=data/norman/norman.h5ad",
        "data.n_hvg=5000", "data.hvg_criterion=scanpy",
        "split.source=reference_pkl", "split.method=additive",
        "split.fold=0", "split.validation=true",
    ],
}


def dataset_groups(datasets: list[str], combo_folds, combo_set: str,
                   norman_set: str = "nval") -> list[tuple]:
    """(dataset, family label, extra overrides) for every validation set requested.

    The family label carries the fold, and scripts/dev_score.py pools within a family
    with the fold number dropped. `cv` scores combinations only; `sv` also scores one
    held-out single per fold - and the two TRAIN ON DIFFERENT CONDITIONS, so they are
    never pooled together.
    """
    if combo_set not in COMBO_SETS:
        raise ValueError(f"unknown --combo-set {combo_set!r} ({' | '.join(COMBO_SETS)})")
    if norman_set not in NORMAN_SETS:
        raise ValueError(f"unknown --norman-set {norman_set!r} "
                         f"({' | '.join(NORMAN_SETS)})")
    groups = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r} ({' | '.join(DATASETS)})")
        if dataset == "norman":
            # The method override comes as a group override so it lands AFTER the dataset's
            # own split.method=additive and replaces it.
            extra = ([] if norman_set == "nval" else ["split.method=combinations"])
            groups.append(("norman", norman_set, extra))
        else:
            for fold in combo_folds:
                extra = [f"split.validation_fold={fold}"]
                if combo_set == "sv":
                    extra.append("split.validation_singles=true")
                groups.append(("combosciplex", f"{combo_set}{fold}", extra))
    return groups


def cache_name(dataset: str, family: str, combo_set: str) -> str:
    """The cache a (dataset, family) pair needs, named after what determines it.

    A validation fold holds out different conditions, and data.exclude_test_from_hvg
    makes the gene space depend on which - so two families must never share a cache
    file. scPKFM learned this the expensive way: an option added later appended a tag to
    the cache name, and a run silently reused or overwrote the wrong one.
    """
    return f"assets/{dataset}_{family}_dev.h5ad"


def plan(name: str, datasets: list[str], arms: dict[str, list[str]], seeds: list[int],
         gpus: list[int], combo_folds=COMBOSCIPLEX_FOLDS, combo_set: str = "cv",
         extra: list[str] = (), norman_set: str = "nval") -> list[dict]:
    for label in [name, *arms]:
        if not NAME.match(label):
            raise ValueError(f"names must be letters and digits only, got {label!r}")
    if not gpus:
        raise ValueError("at least one GPU is needed")

    jobs = []
    for dataset, family, group_overrides in dataset_groups(datasets, combo_folds,
                                                           combo_set, norman_set):
        cache = cache_name(dataset, family, combo_set)
        base = [o for o in DATASETS[dataset] if not o.startswith("data.cache_h5ad=")]
        base = base + [f"data.cache_h5ad={cache}"] + group_overrides + list(extra)
        for arm, arm_overrides in arms.items():
            for seed in seeds:
                jobs.append({
                    # `kind`, `dataset` and `group` are the field names
                    # scripts/dev_score.py reads; `family` is redundant with them and
                    # kept only so a human reading the manifest does not have to join.
                    "kind": "arm",
                    "tag": f"{name}_{arm}_{family}_s{seed}",
                    "dataset": dataset, "group": family,
                    "family": f"{dataset}:{family}",
                    "arm": arm, "seed": seed, "cache": cache,
                    # Arm overrides come LAST, so an arm can override anything the
                    # dataset set - including the cache, if it ever needs its own.
                    "overrides": base + [f"train.seed={seed}"] + arm_overrides,
                })
    # Round-robin over GPUs, so each queue holds a mix of datasets and arms rather than
    # one dataset finishing hours before another.
    for index, job in enumerate(jobs):
        job["gpu"] = gpus[index % len(gpus)]
    return jobs


def write(name: str, jobs: list[dict], gpus: list[int], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    caches = sorted({job["cache"] for job in jobs})
    for gpu in gpus:
        mine = [job for job in jobs if job["gpu"] == gpu]
        lines = ["#!/bin/sh",
                 "# Generated by scripts/dev_queue.py. One queue per GPU, sequential.",
                 f"# {len(mine)} runs on GPU {gpu}.",
                 "set -e", ""]
        if gpu == gpus[0]:
            lines += ["# Caches first, on this queue only: every other queue waits for",
                      "# them to exist. Building the same cache twice in parallel is how",
                      "# two runs end up reading a half-written file.",
                      ""]
            for cache in caches:
                job = next(j for j in jobs if j["cache"] == cache)
                data_overrides = " ".join(
                    o for o in job["overrides"]
                    if o.startswith(("data.", "split.")) )
                lines += [f'if [ ! -f "{cache}" ]; then',
                          f'  python data_prepare.py --set {data_overrides}',
                          "fi", ""]
        else:
            lines += ["# Wait for the caches GPU "
                      f"{gpus[0]}'s queue builds.", ""]
            for cache in caches:
                lines += [f'while [ ! -f "{cache}" ]; do sleep 30; done', ""]
        for job in mine:
            overrides = " ".join(job["overrides"])
            checkpoint = f'results/runs/{job["tag"]}/checkpoint.pt'
            # SKIP WHAT IS ALREADY DONE, so a queue can be interrupted and restarted.
            # scripts/train.py refuses to overwrite a checkpoint and these scripts run
            # under `set -e`, so without this guard a restart aborts on the first finished
            # run - which turns any interruption into "throw the queue away or pass
            # --force and destroy the comparison". A run of w8 took 4,085 s, so a queue is
            # hours long and being unable to resume it is expensive.
            lines += [f'if [ -f "{checkpoint}" ]; then',
                      f'  echo "=== {job["tag"]} (done, skipping) ==="',
                      "else",
                      f'  echo "=== {job["tag"]} ==="',
                      f'  CUDA_VISIBLE_DEVICES={gpu} python scripts/train.py '
                      f'--tag {job["tag"]} --set {overrides}',
                      "fi",
                      ""]
        path = os.path.join(out_dir, f"gpu{gpu}.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines))
        os.chmod(path, 0o755)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"name": name, "jobs": jobs}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="letters and digits only")
    parser.add_argument("--datasets", nargs="+", default=["combosciplex"],
                        choices=sorted(DATASETS))
    parser.add_argument("--arms", nargs="+", required=True,
                        help="label=\"key=value key=value\". An empty value is the "
                             "baseline arm, e.g. base= . Overrides are applied after "
                             "the dataset's own, so an arm can change anything.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="scPKFM's seed noise (0.28 L2 on combosciplex) exceeded "
                             "every effect it measured. Three is the minimum that gives "
                             "a standard error at all.")
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--combo-set", default="cv", choices=COMBO_SETS,
                        help="cv scores combinations only; sv adds one held-out single "
                             "per fold. Table 3 averages five combinations AND two "
                             "singles, and the single block is where published models "
                             "lose to ridge, so sv is the one that matches the target.")
    parser.add_argument("--combo-folds", nargs="+", type=int,
                        default=list(COMBOSCIPLEX_FOLDS))
    parser.add_argument("--norman-set", default="nval", choices=NORMAN_SETS,
                        help="nval is method=additive, the Table 1 regime where ridge "
                             "already beats scDFM; ncomb is method=combinations with the "
                             "scored doubles' singles held out, which is Table 2 Double - "
                             "one of the two blocks that decide the paper.")
    parser.add_argument("--set", dest="extra", nargs="*", default=[],
                        help="overrides added to every job")
    args = parser.parse_args()

    arms: dict[str, list[str]] = {}
    for item in args.arms:
        if "=" not in item:
            raise SystemExit(f"--arms takes label=\"overrides\", got {item!r}")
        label, value = item.split("=", 1)
        arms[label] = value.split()

    jobs = plan(args.name, args.datasets, arms, args.seeds, args.gpus,
                tuple(args.combo_folds), args.combo_set, args.extra, args.norman_set)
    out_dir = os.path.join("results", "dev", args.name)
    write(args.name, jobs, args.gpus, out_dir)

    print(f"{len(jobs)} runs over {len(args.gpus)} GPUs -> {out_dir}")
    for gpu in args.gpus:
        print(f"  sh {os.path.join(out_dir, f'gpu{gpu}.sh')}  "
              f"({sum(1 for j in jobs if j['gpu'] == gpu)} runs)")
    families = sorted({job["family"] for job in jobs})
    print(f"  families: {', '.join(families)}")
    print(f"  arms: {', '.join(arms)}   seeds: {args.seeds}")
    print("\nscore with:")
    print(f"  python scripts/dev_score.py {os.path.join(out_dir, 'manifest.json')} "
          f"--baseline {next(iter(arms))}")


if __name__ == "__main__":
    main()
