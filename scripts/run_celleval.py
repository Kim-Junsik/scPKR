"""Score a trained run with cell-eval 0.5.42.

Runs in two steps because cell-eval needs its own interpreter:

  1. this environment  - load the checkpoint, transport control cells, write
                         pred.h5ad / real.h5ad
  2. .env-celleval     - run MetricsEvaluator on those two files

    python scripts/run_celleval.py results/runs/<tag>
    python scripts/run_celleval.py results/runs/<tag> --profile full
    python scripts/run_celleval.py results/runs/<tag> --export-only
    python scripts/run_celleval.py results/runs/<tag> --score-only

Step 1 runs anywhere; step 2 needs the interpreter that holds cell-eval. When the
repo is shared between a Linux container and a Windows host and .env-celleval is
the Windows one, split it: --export-only in the container, --score-only on Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import splits
from src.data.conventions import ConditionNaming
from src.data.dataset import PerturbationData
from src.eval.celleval import CONTROL_LABEL, PERT_COL, export
from src.models.backbones import build_backbone
from src.models.flow import PKFMField

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CELLEVAL_WINDOWS = os.path.join(REPO_ROOT, ".env-celleval")
CELLEVAL_LINUX = os.path.join(REPO_ROOT, ".env-celleval-linux")
SETUP_HINT = "sh scripts/setup_celleval_linux.sh"


def candidate_interpreters() -> list[str]:
    """Interpreters that might hold cell-eval, best first.

    The Linux venv is offered first, and the Windows .exe is offered ONLY on
    Windows. That ordering is the fix for the case that actually bites: with the
    repo mounted into a container, .env-celleval/python.exe is visible, so
    os.path.exists says yes, but exec'ing it there dies with
    `WSL ERROR: UtilBindVsockAnyPort ... socket failed` - a message that names
    neither the file nor the reason.
    """
    found = [os.path.join(CELLEVAL_LINUX, "bin", "python"),
             os.path.join(CELLEVAL_WINDOWS, "bin", "python")]
    if os.name == "nt":
        # Scripts/ first: that is where `python -m venv` and `uv venv` put the
        # interpreter on Windows. The bare python.exe below is the EMBEDDED
        # distribution the repo shipped with - DLLs/, Library/, python311.dll and
        # no Scripts/ - which is a different layout, so looking only for it left
        # a perfectly good venv undiscovered.
        found.append(os.path.join(CELLEVAL_WINDOWS, "Scripts", "python.exe"))
        found.append(os.path.join(CELLEVAL_LINUX, "Scripts", "python.exe"))
        found.append(os.path.join(CELLEVAL_WINDOWS, "python.exe"))
    return found


def imports_cell_eval(interpreter: str) -> bool:
    try:
        return subprocess.run([interpreter, "-c", "import cell_eval"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


def find_celleval_python(override: str | None = None) -> str:
    """The interpreter that holds cell-eval.

    Absolute and OS-normalised: CreateProcess on Windows does not resolve a
    relative forward-slash path even when os.path.exists accepts it.

    Candidates are PROBED, not merely checked for existence. A venv that was
    created but never installed into - an interrupted setup leaves exactly that -
    would otherwise shadow a working interpreter, and the run would die on
    ModuleNotFoundError naming the empty venv rather than falling through to the
    python that does have cell-eval.
    """
    explicit = override or os.environ.get("CELLEVAL_PYTHON")
    if explicit:
        # Returned unprobed on purpose: an explicit choice that does not work
        # should say so (check_runnable) rather than be silently replaced.
        return os.path.normpath(os.path.abspath(explicit))

    tried = []
    for candidate in candidate_interpreters():
        if not os.path.exists(candidate):
            continue
        if imports_cell_eval(candidate):
            return os.path.normpath(candidate)
        tried.append(candidate)
    # This very interpreter. The split environment exists because of a dependency
    # conflict, but an image that already carries cell-eval makes it pointless.
    if imports_cell_eval(sys.executable):
        return sys.executable

    stranded = os.path.exists(os.path.join(CELLEVAL_WINDOWS, "python.exe"))
    detail = ("\n.env-celleval holds a WINDOWS interpreter, which cannot run here."
              if stranded and os.name != "nt" else "")
    empty = ("\nfound but without cell-eval: " + ", ".join(tried) +
             "\n  (an interrupted setup leaves an empty venv - delete it)"
             if tried else "")
    raise SystemExit(
        f"no usable cell-eval interpreter found.{detail}{empty}\n"
        f"Install it here:  pip install 'cell-eval==0.5.42'\n"
        f"Or build a separate env:  {SETUP_HINT}\n"
        f"Or point at one with --celleval-python / $CELLEVAL_PYTHON.")


def check_runnable(interpreter: str) -> None:
    """Fail before the export, not after it.

    Exporting transports every test condition and writes two ~200 MB files; doing
    that first and only then discovering the interpreter cannot start wastes the
    expensive half of the job.
    """
    if interpreter.endswith(".exe") and os.name != "nt":
        raise SystemExit(
            f"{interpreter} is a Windows interpreter and this is not Windows.\n"
            f"Build the Linux one instead:  {SETUP_HINT}")
    probe = subprocess.run([interpreter, "-c", "import cell_eval"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(f"{interpreter} cannot import cell_eval:\n"
                         f"{probe.stderr[-1000:]}\nBuild it with:  {SETUP_HINT}")

# Kept as a string and run by the other interpreter; importing cell_eval here
# would fail, which is the whole reason for the split.
#
# The __main__ guard is REQUIRED, not stylistic. cell-eval parallelises with
# multiprocessing, and on Windows (spawn start method) each child re-imports the
# main module - without the guard every child re-runs the whole script and spawns
# more children, so the run never terminates and never errors either. That is what
# made the first attempts appear to hang.
SCORING_SCRIPT = '''
import sys, json
import multiprocessing

# The console codepage is not always utf-8 - cp949 on a Korean Windows - and
# polars draws its tables with box characters. Without this the process dies on
# UnicodeEncodeError AFTER results.csv and agg_results.csv are already on disk,
# so a finished scoring run is reported as a failure.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main():
    from cell_eval import MetricsEvaluator
    pred, real, outdir, profile, control, pert_col, threads = sys.argv[1:8]
    evaluator = MetricsEvaluator(
        adata_pred=pred, adata_real=real,
        control_pert=control, pert_col=pert_col,
        outdir=outdir, allow_discrete=False, num_threads=int(threads))
    results, agg = evaluator.compute(profile=profile, break_on_error=False)
    # ASCII only, deliberately. polars renders tables with box-drawing characters,
    # and printing those through a cp949 console produced mojibake at best and a
    # UnicodeEncodeError at worst - after both csv files were already written, so
    # a finished run reported itself as failed. The numbers live in results.csv
    # and agg_results.csv; `sh test.sh --summary --celleval` formats them.
    print(json.dumps({"n_rows": int(results.height),
                      "n_metrics": len(results.columns) - 1,
                      "columns": results.columns[:20]}))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
'''


def celleval_dir(gate: str | None, alpha: str | None) -> str:
    """The export folder for one (gate, alpha) reading of a checkpoint.

    scripts/paper_table.py imports THIS function rather than rebuilding the name,
    because a reader that disagrees with the writer reports "no cell-eval yet" for
    an export sitting on disk, or worse joins one reading's L2 to another's columns.
    alpha=none keeps the historical name so exports made before the option existed
    still resolve.
    """
    name = "celleval" if not gate else f"celleval_{gate}"
    if alpha and alpha != "none":
        name = f"{name}_a{alpha}"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="a directory holding checkpoint.pt")
    parser.add_argument("--profile", default="minimal",
                        choices=["full", "minimal", "vcc", "de", "anndata"])
    parser.add_argument("--export-only", action="store_true",
                        help="write the h5ad pair and stop")
    parser.add_argument("--score-only", action="store_true",
                        help="score an export that already exists; skips the "
                             "checkpoint, the data and the transport entirely")
    parser.add_argument("--celleval-python", default=None,
                        help="interpreter holding cell-eval, if not .env-celleval")
    parser.add_argument("--threads", type=int, default=1,
                        help="cell-eval worker threads. The DE pass is the whole "
                             "cost of scoring and it ran single-threaded, which "
                             "is why this takes longer than the training does. "
                             "pdex already builds ONE shared matrix for workers "
                             "to read, so raising this does not raise /dev/shm.")
    parser.add_argument("--alpha", default=None, choices=("none", "mean", "cell"),
                        help="post-hoc magnitude correction (eval.magnitude_alpha), "
                             "fitted on this run's TRAINING conditions. Required for "
                             "any checkpoint written before the option existed: its "
                             "config carries no such key and would score uncorrected.")
    parser.add_argument("--gate", default=None,
                        choices=["soft", "hard", "sample"],
                        help="override model.hurdle_gate for this scoring only. "
                             "The gate decides how the hurdle head realises the "
                             "binary event at INFERENCE and appears in no training "
                             "loss, so this re-reads the same weights rather than "
                             "changing the model. Results go to celleval_<gate>/ "
                             "so the two scorings of one run cannot overwrite each "
                             "other; without it, celleval/ as usual.")
    parser.add_argument("--infer-top-gene", type=int, default=None,
                        help="score on scanpy-HVG genes of the test subset, as "
                             "scDFM does (their run.sh uses 1000). Without it the "
                             "numbers sit on a different gene space than theirs.")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="cap cells per condition; cell-eval runs a DE test per "
                             "condition, so the full export is slow to score")
    args = parser.parse_args()

    # The gate AND the magnitude correction go in the folder name. Both change the
    # predictions this export holds, so an export made under one must never
    # overwrite an export made under another - that would destroy the uncorrected
    # baseline the corrected run is being compared against, silently.
    out_dir = os.path.join(args.run_dir, celleval_dir(args.gate, args.alpha))
    interpreter = None
    if not args.export_only:
        # Resolved and probed BEFORE the export, so a broken environment costs
        # seconds instead of the whole transport.
        interpreter = find_celleval_python(args.celleval_python)
        check_runnable(interpreter)

    if args.score_only:
        paths = {"pred": os.path.join(out_dir, "pred.h5ad"),
                 "real": os.path.join(out_dir, "real.h5ad")}
        absent = [p for p in paths.values() if not os.path.exists(p)]
        if absent:
            raise SystemExit(f"--score-only needs an existing export; missing "
                             f"{', '.join(absent)}")
        print(f"scoring the existing export in {out_dir}")
    else:
        checkpoint_path = os.path.join(args.run_dir, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise SystemExit(f"no checkpoint at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if args.gate:
            config["model"]["hurdle_gate"] = args.gate
            print(f"hurdle gate overridden: {args.gate}")
        if args.alpha:
            config["eval"]["magnitude_alpha"] = args.alpha
            print(f"magnitude alpha overridden: {args.alpha}")
        device = config["train"]["device"]
        if device == "cuda" and not torch.cuda.is_available():
            device = config["train"]["device"] = config["eval"]["device"] = "cpu"

        print(f"loading data ({config['data']['cache_h5ad']}) ...")
        data = PerturbationData(config["data"]["cache_h5ad"],
                                naming=ConditionNaming.from_config(config))
        method = config["split"]["method"]
        fold = splits.folds(config, method)[config["split"]["fold"]]

        vae = build_backbone(config, data.n_genes, data.gene_names).to(device)
        vae.load_state_dict(checkpoint["vae"])
        field = PKFMField(config, data.n_perturbations, vae.latent_dim,
                          data.perturbations).to(device)
        field.load_state_dict(checkpoint["field"])
        # The anchor table and the magnitude correction are not in the checkpoint -
        # both are functions of the training conditions and the condition means, and
        # predict_cells reads them off the field. This script builds its own field
        # rather than going through diagnostics.load_run, so without this it would
        # score an anchored or alpha-corrected run as if it were neither, and five
        # of the paper's columns would silently disagree with the others.
        from src.eval import baselines as _baselines
        from src.eval import predict as _predict
        # Built from the cells already loaded here rather than through
        # diagnostics._dataset, which would open the cache a second time - 2.2 GB
        # on Norman for a table of means.
        _labels = np.empty(data.x.shape[0], dtype=object)
        for _condition, _rows in data.rows.items():
            _labels[_rows] = _condition
        _stats = _baselines.ConditionMeans(data.x, _labels, data.naming)
        _train_conditions = _baselines.training_conditions(_stats, fold, method)
        field.anchor_table = _baselines.anchor_deltas(
            config["model"].get("anchor", "none"), _stats, _train_conditions,
            list(_stats.mean), alpha=config["eval"]["ridge_alpha"])
        vae.eval()
        field.eval()
        _alpha_mode = config["eval"].get("magnitude_alpha", "none")
        field.magnitude_alpha = None
        if _alpha_mode != "none":
            field.magnitude_alpha = (_alpha_mode, _predict.fit_alpha(
                vae, field, data, _stats, _train_conditions, config,
                np.random.default_rng(config["eval"]["seed"]),
                anchor=field.anchor_table))
            print(f"magnitude alpha ({_alpha_mode}): "
                  f"{field.magnitude_alpha[1]:.4f}")

        genes = None
        if args.infer_top_gene:
            from src.eval.diagnostics import scdfm_eval_genes
            genes = scdfm_eval_genes(data, fold, args.infer_top_gene)
            print(f"scoring on {len(genes):,} scanpy-HVG genes of the test subset")

        rng = np.random.default_rng(config["eval"]["seed"])
        print("transporting control cells for every test condition ...")
        paths = export(vae, field, data, fold, config, out_dir, rng,
                       args.max_cells, genes)
        print(f"  wrote {paths['pred']} and {paths['real']}  "
              f"({paths['n_cells']} cells, {paths['n_conditions']} conditions, "
              f"{paths['n_genes']} genes)")

        if args.export_only:
            return

    script_path = os.path.normpath(os.path.abspath(os.path.join(out_dir, "_score.py")))
    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(SCORING_SCRIPT)

    print(f"\nscoring with cell-eval (profile={args.profile}) ...")
    completed = subprocess.run(
        [interpreter, script_path,
         os.path.abspath(paths["pred"]), os.path.abspath(paths["real"]),
         os.path.abspath(out_dir), args.profile, CONTROL_LABEL, PERT_COL,
         str(args.threads)],
        # The child is told to emit utf-8, so the parent must DECODE utf-8. Left
        # to the locale it decodes as cp949 on a Korean Windows and every non-ascii
        # byte comes back mangled.
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"cell-eval failed (exit {completed.returncode})")

    with open(os.path.join(args.run_dir, "celleval_summary.json"), "w") as handle:
        json.dump({"profile": args.profile, "outdir": out_dir}, handle, indent=2)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
