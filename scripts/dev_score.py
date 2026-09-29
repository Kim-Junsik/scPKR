"""Score development runs and decide by the pre-registered rule.

    python scripts/dev_score.py results/dev/p0n/manifest.json results/dev/p0c/manifest.json \
        --baseline base --device cuda
    python scripts/dev_score.py results/dev/c1n/manifest.json results/dev/r1n/manifest.json \
        results/dev/s1c/manifest.json --baseline base --family-baseline norman:nval=ot \
        --arms rho --device cuda

For every finished ARM run of the given manifests: L2 on its validation conditions,
computed the way paper_table.compute_l2 computes the reported metric - over the
scanpy-HVG genes of the validation set plus control, soft gate, 1,024 transported
cells. Encoder runs are not scored. Manifests can be combined, so a later
experiment can pair its arms with an earlier experiment's baseline runs when both
used the same encoders.

FAMILIES. Runs are compared within a family: dataset plus validation set with the
fold number dropped - norman:nval, combosciplex:cv, combosciplex:sv. cv and sv
train on different conditions, so they are never pooled.

METRIC. norman:nval and combosciplex:cv score doubles only, exactly as before (the
p0 and c1 numbers reproduce). combosciplex:sv folds hold out four combinations
and one single, and are scored like Table 3, whose L2 is the mean over five
combinations and two singles:
    L2 = 5/7 x mean L2(combinations) + 2/7 x mean L2(single)
The two blocks are reported separately as well.

Noise: the baseline arm's seed-to-seed SD within each validation fold, pooled.

Comparison: every other arm is paired with the baseline run of the same family,
validation fold and seed; delta = L2(arm) - L2(baseline). Per family, the mean
delta and its standard error over pairs. --family-baseline FAMILY=ARM changes the
baseline for one family, e.g. norman:nval=ot after Norman moved to exact OT.

Decision rule, fixed in the phase-0 design before any arm was run, with the block
gate added on 2026-09-17 before any sv run existed:
  adopt an arm if, on at least one family, mean delta <= -2 SE,
  on no family mean delta > +1 SE,
  and in no block of a blocked family (combinations, single) mean delta > +1 SE.
Ties between adoptable arms go to the simpler, closer-to-default arm, by hand.

Known limit of the single block: three conditions over three folds, two of them
the same drug, so its SE describes these two drugs rather than singles in general.

Required seeds: the pairs needed for 2 SE to drop below --mde, from the observed
SD of the paired deltas - or sqrt(2) x the baseline noise while no pairs exist -
converted to seeds per validation fold.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# Table 3 holds out five combinations and two singles.
TABLE3_WEIGHTS = {"double": 5 / 7, "single": 2 / 7}
BLOCKS = ("double", "single")


def find_run(tag: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(ROOT, "results", "runs", f"{tag}_*", "checkpoint.pt")))
    if len(hits) > 1:
        raise SystemExit(f"tag {tag} matches {len(hits)} finished runs: {hits}")
    return os.path.dirname(hits[0]) if hits else None


def family(dataset: str, group: str) -> str:
    """norman:nval, combosciplex:cv, combosciplex:sv - the fold number dropped."""
    return f"{dataset}:{group.rstrip('0123456789')}"


def weighted_l2(blocks: dict[str, float]) -> float:
    """The decision metric: Table 3's weighting when a single was scored, else doubles."""
    if blocks.get("single") is None:
        return blocks["double"]
    return sum(TABLE3_WEIGHTS[b] * blocks[b] for b in BLOCKS)


def score_run(run_dir: str, device: str, n_cells: int,
              rho_off: bool = False) -> dict[str, float | None]:
    """Mean L2 of the validation doubles and of the validation singles.

    One transport pass, doubles first. measure_transport draws its cells from one
    rng in condition order, so the doubles consume exactly the draws
    paper_table.compute_l2(..., group="double") consumes and their mean is
    bit-identical to it; the singles come after and cannot disturb it.

    `rho_off` scores the same weights with the learned composition switched off
    (v = sum_a u_a for combinations). The rng is reseeded identically, so a
    run scored both ways is compared on the very same cells. Singles never use
    rho, so their scores do not move.
    """
    from src.eval.diagnostics import (condition_groups, load_run, measure_transport,
                                      scdfm_eval_genes)

    config, data, stats, fold, vae, field = load_run(run_dir, device, "soft")
    if rho_off:
        field.composition_kind = "additive"
    rng = np.random.default_rng(config["eval"]["seed"])
    groups = condition_groups(data, stats, fold, config["split"]["method"])
    doubles, singles = groups["test doubles"], groups["test singles"]
    genes = scdfm_eval_genes(data, fold, 1000)
    rows = measure_transport(vae, field, data, stats, doubles + singles,
                             config, rng, device, n_cells, genes=genes)
    by_condition = {r["condition"]: r["l2"] for r in rows}
    out: dict[str, float | None] = {
        "double": float(np.mean([by_condition[c] for c in doubles])) if doubles else float("nan"),
        "single": float(np.mean([by_condition[c] for c in singles])) if singles else None,
    }
    return out


def pooled_sd(groups: list[list[float]]) -> tuple[float, int]:
    """Pooled within-group SD and its degrees of freedom."""
    num, dof = 0.0, 0
    for values in groups:
        if len(values) >= 2:
            num += float(np.var(values, ddof=1)) * (len(values) - 1)
            dof += len(values) - 1
    return (math.sqrt(num / dof) if dof else float("nan")), dof


def summarise(deltas: list[float]) -> dict:
    n = len(deltas)
    mean = float(np.mean(deltas)) if n else float("nan")
    sd = float(np.std(deltas, ddof=1)) if n >= 2 else float("nan")
    se = sd / math.sqrt(n) if n >= 2 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "se": se}


def _usable(stats: dict) -> bool:
    return stats["n"] >= 2 and math.isfinite(stats["se"])


def decide(per_family: dict[str, dict],
           blocks: dict[str, dict[str, dict]] | None = None) -> str:
    """The pre-registered rule. `blocks` maps a family to its per-block statistics."""
    blocks = blocks or {}
    stats = [s for s in per_family.values() if _usable(s)]
    block_stats = [s for family_blocks in blocks.values() for s in family_blocks.values()]
    if (len(stats) < len(per_family) or not stats
            or any(not _usable(s) for s in block_stats)):
        return "insufficient pairs"
    improved = any(s["mean"] <= -2 * s["se"] for s in stats)
    if any(s["mean"] > s["se"] for s in stats):
        return "reject (hurts a dataset)"
    if any(s["mean"] > s["se"] for s in block_stats):
        return "reject (hurts a block)"
    return "ADOPT" if improved else "reject (no 2 SE improvement)"


def parse_family_baselines(items: list[str]) -> dict[str, str]:
    chosen: dict[str, str] = {}
    for item in items:
        name, _, arm = item.partition("=")
        if ":" not in name or not arm:
            raise SystemExit(f"--family-baseline wants FAMILY=ARM, e.g. norman:nval=ot; got {item!r}")
        chosen[name] = arm
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--baseline", required=True, help="arm the others are paired with")
    parser.add_argument("--family-baseline", action="append", default=[],
                        help="FAMILY=ARM overriding --baseline for one family, "
                             "e.g. norman:nval=ot")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="compare only these arms (baselines are always loaded)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=1024)
    parser.add_argument("--mde", type=float, default=0.05,
                        help="smallest L2 improvement the experiment must resolve")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    from src.eval import diagnostics

    overrides = parse_family_baselines(args.family_baseline)

    def baseline_of(fam: str) -> str:
        return overrides.get(fam, args.baseline)

    jobs = []
    for path in args.manifests:
        with open(path, encoding="utf-8") as handle:
            jobs += [job for job in json.load(handle)["jobs"] if job["kind"] == "arm"]
    wanted = set(args.arms) if args.arms else None
    jobs = [job for job in jobs
            if wanted is None or job["arm"] in wanted
            or job["arm"] == baseline_of(family(job["dataset"], job["group"]))]

    records, missing, current, seen = [], [], None, {}
    print(f"{'run tag':34s} {'family':17s} {'group':5s} {'arm':10s} {'seed':>4s} "
          f"{'L2':>8s} {'double':>8s} {'single':>8s}")
    for job in sorted(jobs, key=lambda j: (j["dataset"], j["group"], j["arm"], j["seed"])):
        fam = family(job["dataset"], job["group"])
        key = (fam, job["group"], job["arm"], job["seed"])
        if key in seen:
            # Two manifests with the same arm name on the same fold and seed - e.g. a
            # UOT base and an exact-OT base - would silently overwrite each other.
            raise SystemExit(f"{job['tag']} and {seen[key]} are both arm '{job['arm']}' of "
                             f"{job['group']} seed {job['seed']}; give them different "
                             f"arm names or score them separately")
        seen[key] = job["tag"]
        run_dir = find_run(job["tag"])
        if run_dir is None:
            missing.append(job["tag"])
            continue
        group = (job["dataset"], job["group"])
        if group != current:
            # Validation folds differ in cache AND split; drop both caches so a
            # later fold can never be scored with an earlier fold's split or cells.
            diagnostics._DATASETS.clear()
            diagnostics._FOLDS.clear()
            current = group
        blocks = score_run(run_dir, args.device, args.n_cells)
        l2 = weighted_l2(blocks)
        records.append({"tag": job["tag"], "family": fam, "dataset": job["dataset"],
                        "group": job["group"], "arm": job["arm"], "seed": job["seed"],
                        "run": os.path.basename(run_dir), "l2": l2,
                        "l2_double": blocks["double"], "l2_single": blocks["single"]})
        single = f"{blocks['single']:8.4f}" if blocks["single"] is not None else f"{'-':>8s}"
        print(f"{job['tag'][:33]:34s} {fam:17s} {job['group']:5s} {job['arm']:10s} "
              f"{job['seed']:4d} {l2:8.4f} {blocks['double']:8.4f} {single}")
    if missing:
        print(f"\nnot finished yet ({len(missing)}): {', '.join(missing)}")

    by_key = {(r["family"], r["group"], r["arm"], r["seed"]): r for r in records}
    families = sorted({r["family"] for r in records})
    blocked = {fam for fam in families
               if any(r["l2_single"] is not None for r in records if r["family"] == fam)}

    print("\n=== noise: baseline across seeds, pooled within fold ===")
    noise = {}
    for fam in families:
        base = baseline_of(fam)
        groups = sorted({r["group"] for r in records if r["family"] == fam})
        metrics = ["l2"] + (["l2_double", "l2_single"] if fam in blocked else [])
        parts = []
        for metric in metrics:
            per_group = [[by_key[k][metric] for k in sorted(by_key)
                          if k[0] == fam and k[1] == g and k[2] == base] for g in groups]
            sd, dof = pooled_sd(per_group)
            if metric == "l2":
                noise[fam] = (sd, len(groups))
            parts.append(f"{metric} SD {sd:.4f} (dof {dof})")
        print(f"  {fam:17s} baseline '{base}'   " + "   ".join(parts))

    compared = sorted({r["arm"] for r in records if r["arm"] != baseline_of(r["family"])})
    print("\n=== paired deltas vs baseline (negative = better) ===")
    decisions = {}
    sd_deltas = {fam: [] for fam in families}
    for arm in compared:
        per_family, per_block = {}, {}
        for fam in families:
            base = baseline_of(fam)
            pairs = [(by_key[k], by_key[(k[0], k[1], base, k[3])]) for k in sorted(by_key)
                     if k[0] == fam and k[2] == arm and (k[0], k[1], base, k[3]) in by_key]
            if not pairs and arm not in {r["arm"] for r in records if r["family"] == fam}:
                continue  # this arm was never run on this family
            per_family[fam] = summarise([a["l2"] - b["l2"] for a, b in pairs])
            if math.isfinite(per_family[fam]["sd"]):
                sd_deltas[fam].append(per_family[fam]["sd"])
            if fam in blocked:
                per_block[fam] = {block: summarise([a[f"l2_{block}"] - b[f"l2_{block}"]
                                                    for a, b in pairs])
                                  for block in BLOCKS}
        decisions[arm] = decide(per_family, per_block)
        print(f"  {arm}   -> {decisions[arm]}")
        for fam, s in per_family.items():
            print(f"      {fam:17s} {s['mean']:+.4f} +- {s['se']:.4f} (n={s['n']})")
            for block, b in per_block.get(fam, {}).items():
                print(f"        {block:15s} {b['mean']:+.4f} +- {b['se']:.4f} (n={b['n']})")

    print(f"\n=== seeds needed per validation fold to resolve {args.mde} L2 (2 SE) ===")
    for fam in families:
        sigma, n_groups = noise[fam]
        if sd_deltas[fam]:
            sd, source = max(sd_deltas[fam]), "largest paired-delta SD"
        else:
            sd, source = math.sqrt(2) * sigma, "sqrt(2) x baseline SD"
        if not math.isfinite(sd):
            print(f"  {fam:17s} unknown - need at least two baseline seeds per fold")
            continue
        pairs = math.ceil((2 * sd / args.mde) ** 2)
        print(f"  {fam:17s} SD {sd:.4f} ({source}) -> {pairs} pairs "
              f"= {math.ceil(pairs / max(n_groups, 1))} seeds x {n_groups} fold(s)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["tag", "run", "family", "dataset",
                                                        "group", "arm", "seed", "l2",
                                                        "l2_double", "l2_single"])
            writer.writeheader()
            writer.writerows(records)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
