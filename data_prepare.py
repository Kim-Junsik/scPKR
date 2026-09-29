"""Everything on the data side, in one command.

    python data_prepare.py
    python data_prepare.py --force          # rebuild even if artifacts exist
    python data_prepare.py --set data.n_hvg=5000

Four steps, none of which touch a model:

  1. KEGG snapshot          the pathway mask is built from it
  2. split validation       must be the file that ships with the dataset
  3. modelled-space cache   HVG + perturbation targets
  4. baselines              the target line a model has to beat

Step 4 belongs here rather than in test.sh: it evaluates nothing that was trained.
Ridge-additive is a property of the data, so its numbers are constants of the
dataset, computed once and read back later.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import config as config_module  # noqa: E402


def run(label: str, argv: list[str]) -> None:
    print(f"\n{'=' * 68}\n {label}\n{'=' * 68}")
    started = time.time()
    completed = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"\n[FAILED] {label} (exit {completed.returncode})")
    print(f"  ... {time.time() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="config overrides, e.g. data.n_hvg=5000")
    parser.add_argument("--force", action="store_true",
                        help="re-download KEGG and rebuild the cache")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    overrides = ["--set", *args.overrides] if args.overrides else []

    run("1/4  KEGG snapshot",
        ["scripts/download_kegg.py", *(["--force"] if args.force else []), *overrides])

    run("2/4  split validation",
        ["scripts/build_data.py", "--validate-only", *overrides])

    run("3/4  modelled-space cache",
        ["scripts/build_data.py", "--skip-validation",
         *(["--force"] if args.force else []), *overrides])

    if not args.skip_baselines:
        # `method` selects between the additive and combinations derivations of
        # the shipped split. Other sources define their own folds and ignore it,
        # so running both would print the same table twice.
        config = config_module.load(args.overrides)
        methods = (("additive", "combinations")
                   if config["split"].get("source", "reference_pkl") == "reference_pkl"
                   else (config["split"]["method"],))
        for method in methods:
            run(f"4/4  baselines ({method})",
                ["scripts/run_baselines.py", "--set", f"split.method={method}",
                 *args.overrides])

    print(f"\n{'=' * 68}")
    print(" done. next:  sh train.sh")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
