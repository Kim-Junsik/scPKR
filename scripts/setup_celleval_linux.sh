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
# WHY IT IS PINNED. cell-eval's metric definitions have moved between releases, and the
# reported tables of the papers being compared against were produced with 0.5.42. An
# unpinned install would quietly score a different arithmetic and the comparison would
# be worthless without anything saying so.
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

if [ -d "$ENV_DIR" ]; then
  echo "$ENV_DIR exists already."
  echo "Delete it to rebuild:  rm -rf $ENV_DIR"
else
  echo "creating $ENV_DIR ..."
  python -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install --quiet --upgrade pip
echo "installing cell-eval==$VERSION ..."
"$ENV_DIR/bin/python" -m pip install --quiet "cell-eval==$VERSION"

echo
"$ENV_DIR/bin/python" - <<'PY'
import cell_eval
print("cell_eval", getattr(cell_eval, "__version__", "?"), "imports cleanly")
PY

echo
echo "shared memory available to cell-eval's DE tests:"
df -h /dev/shm | tail -1
echo "  (64M is the Docker default and is NOT enough - their Dockerfile uses 8g."
echo "   A short /dev/shm shows up as SIGBUS, exit -7, with no other message.)"
echo
echo "done. scripts/run_celleval.py will find this interpreter on its own."
