"""Plan a development experiment as one sequential shell queue per GPU.

    python scripts/dev_queue.py --name p0n --datasets norman --arms base= \
        --seeds 0 1 2 3 4 --gpus 0 1 2 3
    python scripts/dev_queue.py --name c1 --encoder-name p0c --datasets combosciplex \
        --arms base= reg001="--uot-reg 0.01" ot="--coupling ot" --seeds 0 1 2 --gpus 0 1 2 3

then, on the server, one tmux window per GPU:

    sh results/dev/p0n/gpu0.sh

Every run is a development run: Norman on its fold-0 validation set
(splits.NORMAN_VALIDATION) and combosciplex on each of its validation folds
(splits.COMBOSCIPLEX_VALIDATION_FOLDS; --combo-set sv adds one held-out single per
fold). No reported test set is ever scored.

Norman arms start from exact OT (--norman-coupling ot, decided on experiment c1);
pass --norman-coupling uot to plan them on UOT again. Experiments planned before
that decision (p0n, c1n) used UOT for their base arm.

One ENCODER job per (dataset, validation fold) trains stage 1 and fits the latent
standardisation (--stage2 0); every arm and seed loads it. The runs being compared
therefore share their encoder exactly, and a difference between two arms is a
difference in stage 2 alone. An arm job waits until its encoder's checkpoint.pt
exists, so all queues can be started together. --encoder-name reuses the encoders
of an earlier experiment instead of training new ones.

Runs are identified by tag, and scripts/dev_score.py finds a run as the one
directory matching results/runs/<tag>_*. Names are letters and digits only, so no
tag can be a prefix of another run's directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The fin configuration Tables 1-3 were trained with.
BASE_FLAGS = ["--latent-readout", "pathway", "--generator-rank", "64", "--latent-dim", "64",
              "--kl", "0.001", "--hurdle-bce", "1.0", "--mmd", "10", "--batch", "48",
              "--stage1", "15", "--stage2", "1000", "--stage1-lr", "1e-3",
              "--lr", "5e-5", "--lr-cosine", "--lr-min", "1e-6", "--no-celleval"]
COMBOSCIPLEX_FOLDS = (0, 1, 2)
NAME = re.compile(r"^[A-Za-z0-9]+$")


COMBO_SETS = ("cv", "sv")
COUPLINGS = ("ot", "uot")


def dataset_groups(datasets: list[str], combo_folds,
                   combo_set: str = "cv") -> list[tuple[str, str, list[str]]]:
    """(dataset, group label, split flags) for every validation set requested.

    combo_set cv scores combinations only (splits.COMBOSCIPLEX_VALIDATION_FOLDS);
    sv also scores one held-out single per fold (COMBOSCIPLEX_VALIDATION_SINGLES).
    The two train on different conditions, so they never share an encoder.
    """
    if combo_set not in COMBO_SETS:
        raise ValueError(f"unknown combosciplex set {combo_set!r} ({' | '.join(COMBO_SETS)})")
    groups = []
    for dataset in datasets:
        if dataset == "norman":
            groups.append(("norman", "nval", ["--dataset", "norman", "--method", "additive",
                                              "--fold", "0", "--validation"]))
        elif dataset == "combosciplex":
            for fold in combo_folds:
                flags = ["--dataset", "combosciplex", "--fold", "0",
                         "--validation", "--val-fold", str(fold)]
                if combo_set == "sv":
                    flags.append("--val-singles")
                groups.append(("combosciplex", f"{combo_set}{fold}", flags))
        else:
            raise ValueError(f"unknown dataset {dataset!r} (norman | combosciplex)")
    return groups


def dataset_flags(norman_coupling: str = "ot") -> dict[str, list[str]]:
    """Per-dataset defaults every arm of that dataset starts from.

    Decided 2026-09-17 on the c1 coupling experiment: exact OT for Norman
    (-0.066 L2, ~7 SE), UOT 0.05 for combosciplex (exact OT +0.019, ~4 SE worse).
    Kept a flag, not a code default, so Norman can go back to UOT at any time.
    An arm that passes its own --coupling still wins: run.sh keeps the last value.
    """
    if norman_coupling not in COUPLINGS:
        raise ValueError(f"unknown Norman coupling {norman_coupling!r} ({' | '.join(COUPLINGS)})")
    return {"norman": ["--coupling", norman_coupling]}


def without(flags: list[str], flag: str, takes_value: bool) -> list[str]:
    """`flags` with every occurrence of `flag` (and its value) removed."""
    out, skip = [], False
    for token in flags:
        if skip:
            skip = False
            continue
        if token == flag:
            skip = takes_value
            continue
        out.append(token)
    return out


def plan_jobs(name: str, datasets: list[str], arms: dict[str, str], seeds: list[int],
              gpus: list[int], combo_folds=COMBOSCIPLEX_FOLDS,
              encoder_name: str | None = None, extra: list[str] = (),
              combo_set: str = "cv", norman_coupling: str = "ot") -> list[dict]:
    labels = [name, *arms] + ([encoder_name] if encoder_name else [])
    for label in labels:
        if not NAME.match(label):
            raise ValueError(f"names must be letters and digits only, got {label!r}")
    if not gpus:
        raise ValueError("at least one GPU is needed")
    encoder_name = encoder_name or name
    groups = dataset_groups(datasets, combo_folds, combo_set)
    defaults = dataset_flags(norman_coupling)
    extra = list(extra)

    encoders = []
    if encoder_name == name:
        for dataset, group, split_flags in groups:
            # --stage2 0: stage 1 plus the latent standardisation, nothing else. No
            # cosine schedule, since it would be built over zero epochs.
            # The base --stage2 is removed rather than overridden, so a reader of the
            # generated queue sees one --stage2 per encoder line.
            flags = (without(without(BASE_FLAGS, "--lr-cosine", False), "--stage2", True)
                     + split_flags + without(extra, "--stage2", True)
                     + ["--init-vae-from", "", "--stage2", "0"])
            encoders.append({"kind": "encoder", "tag": f"{name}_enc_{group}",
                             "dataset": dataset, "group": group, "arm": None,
                             "seed": 0, "encoder": None, "flags": flags})

    arm_jobs = []
    for arm, arm_flags in arms.items():
        tokens = shlex.split(arm_flags)
        base = BASE_FLAGS
        if "--generator" in tokens and tokens[tokens.index("--generator") + 1] != "affine":
            # generator_rank sizes the affine generator only; build_generator
            # refuses it for the others.
            base = without(base, "--generator-rank", True)
        for dataset, group, split_flags in groups:
            for seed in seeds:
                arm_jobs.append({"kind": "arm", "tag": f"{name}_{arm}_{group}_s{seed}",
                                 "dataset": dataset, "group": group, "arm": arm,
                                 "seed": seed, "encoder": f"{encoder_name}_enc_{group}",
                                 "flags": (base + split_flags
                                           + defaults.get(dataset, []) + extra
                                           + ["--seed", str(seed)] + tokens)})

    # Encoders spread over the GPUs first so they start together; arms follow
    # round-robin and wait for their encoder wherever it runs.
    for i, job in enumerate(encoders):
        job["gpu"] = gpus[i % len(gpus)]
    for i, job in enumerate(arm_jobs):
        job["gpu"] = gpus[(len(encoders) + i) % len(gpus)]
    jobs = encoders + arm_jobs
    tags = [job["tag"] for job in jobs]
    if len(set(tags)) != len(tags):
        raise ValueError("duplicate run tags in the plan")
    return jobs


HEAD = """#!/bin/sh
# Generated by scripts/dev_queue.py - experiment {name}, GPU {gpu}.
# Its jobs run one after another. Start every gpu*.sh of the experiment at once,
# one per tmux window: arm jobs wait for their encoder wherever it runs.
cd "$(dirname "$0")/../../.." || exit 1

finished() {{ ls results/runs/"$1"_*/checkpoint.pt >/dev/null 2>&1; }}

run_job() {{
  tag="$1"; encoder="$2"; shift 2
  if finished "$tag"; then echo "skip $tag (finished)"; return 0; fi
  if [ -n "$encoder" ]; then
    until finished "$encoder"; do echo "$tag: waiting for encoder $encoder"; sleep 120; done
    set -- "$@" --init-vae-from "$(dirname "$(ls results/runs/"$encoder"_*/checkpoint.pt | head -1)")"
  fi
  echo "=== $tag  $(date)"
  sh run.sh --tag "$tag" "$@" || echo "FAILED $tag"
}}

"""


def write_queues(name: str, jobs: list[dict], out_root: str) -> str:
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    for gpu in sorted({job["gpu"] for job in jobs}):
        lines = [HEAD.format(name=name, gpu=gpu)]
        for job in (j for j in jobs if j["gpu"] == gpu):
            args = ["--gpu_num", str(gpu), *job["flags"]]
            lines.append(f"run_job {job['tag']} {shlex.quote(job['encoder'] or '')} "
                         + " ".join(shlex.quote(a) for a in args) + "\n")
        lines.append('echo "=== queue done  $(date)"\n')
        path = os.path.join(out_dir, f"gpu{gpu}.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(lines)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"name": name, "jobs": jobs}, handle, indent=2)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="experiment name (letters, digits)")
    parser.add_argument("--datasets", nargs="+", default=["norman", "combosciplex"])
    parser.add_argument("--combo-folds", nargs="+", type=int, default=list(COMBOSCIPLEX_FOLDS))
    parser.add_argument("--combo-set", choices=COMBO_SETS, default="cv",
                        help="combosciplex validation folds: cv (combinations only) or "
                             "sv (plus one held-out single per fold)")
    parser.add_argument("--norman-coupling", choices=COUPLINGS, default="ot",
                        help="coupling every Norman arm starts from (decided: ot)")
    parser.add_argument("--arms", nargs="+", required=True,
                        help='name=flags, e.g. base= ot="--coupling ot"')
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--encoder-name", default=None,
                        help="reuse the encoders of this earlier experiment")
    parser.add_argument("--extra-flags", default="",
                        help="appended to every job, e.g. for a smoke test")
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "dev"))
    args = parser.parse_args()

    arms = {}
    for item in args.arms:
        label, _, flags = item.partition("=")
        arms[label] = flags
    jobs = plan_jobs(args.name, args.datasets, arms, args.seeds, args.gpus,
                     args.combo_folds, args.encoder_name, shlex.split(args.extra_flags),
                     args.combo_set, args.norman_coupling)
    out_dir = write_queues(args.name, jobs, args.out)

    def shown(path: str) -> str:
        # relpath raises on Windows when the path is on another drive than the repo.
        try:
            return os.path.relpath(path, ROOT)
        except ValueError:
            return path

    encoders = sum(job["kind"] == "encoder" for job in jobs)
    print(f"{len(jobs)} jobs ({encoders} encoders, {len(jobs) - encoders} arm runs) "
          f"-> {shown(out_dir)}")
    for gpu in sorted({job["gpu"] for job in jobs}):
        count = sum(job["gpu"] == gpu for job in jobs)
        print(f"  gpu {gpu}: {count} jobs   sh {shown(out_dir)}/gpu{gpu}.sh")
    print(f"score with:  python scripts/dev_score.py {shown(out_dir)}/manifest.json "
          f"--baseline <arm> --device cuda")


if __name__ == "__main__":
    main()
