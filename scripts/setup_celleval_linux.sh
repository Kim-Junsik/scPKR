#!/bin/sh
# Build .env-celleval-linux, the interpreter scripts/run_celleval.py scores with.
#
#     sh scripts/setup_celleval_linux.sh
#
# WHY A SEPARATE ENVIRONMENT. cell-eval pins versions of anndata, polars and scipy
# that conflict with the ones this repository trains under, so installing it into the
# training environment silently changes what the model runs on. The two never share an
# interpreter: run_celleval.py exports pred.h5ad / real.h5ad from the training
# environment and shells out to this one to score them.
#
# WHY IT IS PINNED. cell-eval's metric definitions have moved between releases, so a
# different version is a different arithmetic under the same column names and nothing in
# the output says so. 0.5.42 is the version this repository's tooling was written and
# checked against. It is NOT known which version the papers being compared against used -
# that is a limitation of the comparison and belongs in the paper, not a reason to drift.
#
# pdex IS PINNED TOO, AND THAT IS NOT OPTIONAL. cell-eval 0.5.42 asks for pdex>=0.1.20
# with no upper bound, and pdex 0.2+ removed parallel_differential_expression, so a plain
# install of 0.5.42 today builds an environment that imports and then dies on
#   ImportError: cannot import name 'parallel_differential_expression' from 'pdex'
# 0.1.20 is pinned because it is the pdex released on the SAME DAY as cell-eval 0.5.42,
# 2025-07-28, which is the closest thing available to the combination that was tested.
#
# ONE OPERATIONAL TRAP, recorded because it cost a day on the previous model: cell-eval
# runs its differential-expression tests through multiprocessing, which needs shared
# memory. A container started with the Docker default of 64 MB for /dev/shm dies with
# SIGBUS (exit -7) and no useful message. Their own Dockerfile sets --shm-size=8g. Check
# with `df -h /dev/shm` before blaming the export.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR="$ROOT/.env-celleval-linux"
VERSION="0.5.42"
PDEX_VERSION="0.1.20"

if [ -d "$ENV_DIR" ]; then
  echo "$ENV_DIR exists already."
  echo "Delete it to rebuild:  rm -rf $ENV_DIR"
else
  echo "creating $ENV_DIR ..."
  python -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install --quiet --upgrade pip
echo "installing cell-eval==$VERSION with pdex==$PDEX_VERSION ..."
# Both in ONE pip call, so the resolver cannot satisfy cell-eval first with a newer pdex
# and then downgrade it afterwards.
"$ENV_DIR/bin/python" -m pip install --quiet "cell-eval==$VERSION" "pdex==$PDEX_VERSION"

echo
"$ENV_DIR/bin/python" - <<'PY'
# Importing is the test, not pip's exit code: the pdex breakage above installs cleanly
# and only fails when cell_eval reaches for a name that is gone.
from importlib.metadata import version
import cell_eval
print("cell_eval", version("cell-eval"), "+ pdex", version("pdex"), "import cleanly")
PY

echo
echo "shared memory available to cell-eval's DE tests:"
df -h /dev/shm | tail -1
echo "  (64M is the Docker default and is NOT enough - their Dockerfile uses 8g."
echo "   A short /dev/shm shows up as SIGBUS, exit -7, with no other message.)"
echo
echo "done. scripts/run_celleval.py will find this interpreter on its own."
