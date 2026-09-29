# syntax=docker/dockerfile:1.7
#
# scPKFM - single-cell Pathway Koopman Flow Matching. Training and testing image.
#
#   docker build -t scpkfm .
#   docker run --gpus all --shm-size=8g --rm -it \
#       -v "$PWD/data:/workspace/data" \
#       -v "$PWD/assets:/workspace/assets" \
#       -v "$PWD/results:/workspace/results" \
#       scpkfm sh run.sh
#
# Three directories are mounted rather than copied, and each for its own reason:
#
#   data/     norman.h5ad is 2.1 GB. Not in the repo, not in the image.
#   assets/   holds the KEGG snapshot AND the per-fold cache the container
#             WRITES (assets/norman_scanpy5000_fold1.h5ad, ~137 MB). Mounting it
#             means data_prepare.py builds the cache once, on the host, instead
#             of once per container.
#   results/  checkpoints and metrics. The whole point of the run; it must
#             outlive the container.
#
# --shm-size is not optional for the scoring half. cell-eval's DE pass (pdex)
# builds ONE dense shared-memory matrix of the whole real set before it computes
# anything, and POSIX shared memory on Linux lives in /dev/shm - which Docker
# caps at 64 MB by default no matter how much RAM the host has. 17,918 cells over
# 1,000 genes in float32 is 68 MiB, so scoring dies with
#
#     cell-eval failed (exit -7)
#
# a second after "Creating shared memory matrix", naming neither pdex nor
# /dev/shm. --ipc=host works too and gives the host's own tmpfs, usually half of
# RAM. Training itself does not need either.
#
# KEGG is not baked in. It is free for academic use but redistributing the files
# is a separate permission, so the image fetches nothing at build time. Either
# mount an assets/kegg you already have, or make one inside the container:
#
#   docker run --rm -v "$PWD/assets:/workspace/assets" scpkfm \
#       python scripts/download_kegg.py
#
# Without it the P-CAB pathway mask cannot be built and training will not start.


# --------------------------------------------------------------------- base
# 3.12, not 3.13, and the constraint comes from both ends: scanpy and anndata
# need >=3.11, cell-eval's metadata says >=3.10,<3.13. 3.12 is the only version
# both halves of this project accept, which is what lets one image hold both
# environments.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

# Why a plain python image and not nvidia/cuda: the torch wheels below carry
# their own CUDA runtime and cuDNN (the nvidia-* packages pip pulls in with
# them), and the host driver is injected by the NVIDIA container toolkit at
# `docker run --gpus all`. A CUDA base image would add a second, unused copy of
# the runtime for about 2 GB.
#
# What that does NOT buy is running without a driver. Check before paying for a
# training run - scripts/train.py falls back to CPU with a one-line [warn] that
# is easy to miss in a log, and stage 2 on CPU is not worth waiting for:
#
#   docker run --gpus all --rm scpkfm \
#       python -c "import torch; print(torch.cuda.is_available())"

LABEL org.opencontainers.image.title="scPKFM" \
      org.opencontainers.image.description="single-cell Pathway Koopman Flow Matching - training and evaluation" \
      org.opencontainers.image.licenses="MIT"

# PYTHONDONTWRITEBYTECODE is not a size choice. src/ is bind-mounted in the
# development flow, and without it the container leaves root-owned __pycache__
# directories in the host checkout that the host user then cannot delete.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential earns its ~200 MB: every pinned wheel below has a manylinux
# build today, but scanpy's transitive set (numba/llvmlite) is the kind that
# occasionally does not for a new interpreter, and a source build failing at
# `docker build` is a better discovery than a slightly larger image is a cost.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        procps; \
    rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------ training environment
WORKDIR /workspace

# Requirements first, source last: editing src/ then rebuilds only the final
# COPY instead of reinstalling torch.
COPY requirements.txt requirements-celleval.txt ./

# torch is installed on its own, from a chosen wheel index. cu118 matches the
# environment the reported numbers were measured in. Override it for a newer
# driver, or for a CPU-only image:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# requirements.txt then sees torch==2.7.1 already satisfied and leaves it alone
# (== ignores the local +cu118 segment, per PEP 440).
#
# The import check at the end is not ceremony. A wheel can install and still
# fail to import when a native dependency is missing from a slim image, and the
# alternative is finding that out after the data is mounted and the cache built.
ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
RUN set -eux; \
    pip install --no-cache-dir --upgrade pip; \
    pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"; \
    pip install --no-cache-dir -r requirements.txt; \
    python -c "import torch, scanpy, anndata, ot, sklearn, h5py; print('torch', torch.__version__)"


# ---------------------------------------------------- cell-eval environment
# Built into the image so a run never has to reach the network, and placed at
# /opt rather than /workspace/.env-celleval-linux so that bind-mounting the repo
# over /workspace cannot hide it. scripts/run_celleval.py reads $CELLEVAL_PYTHON
# before it probes anywhere else.
#
# The symlink is for the other mount style (code from the image, only
# data/assets/results mounted): it makes run.sh's own existence check pass, so
# run.sh does not try to build a second copy on first use.
#
# --build-arg WITH_CELLEVAL=0 skips this. The cost is real - five of the
# reported table's eight columns come from cell-eval and stay '-' without it -
# so it is opt-out, and a failure here fails the build rather than being
# discovered thirty minutes into a scoring pass.
ARG WITH_CELLEVAL=1
ENV CELLEVAL_PYTHON=/opt/celleval/bin/python
RUN set -eux; \
    if [ "${WITH_CELLEVAL}" = "1" ]; then \
        python -m venv /opt/celleval; \
        /opt/celleval/bin/pip install --no-cache-dir --upgrade pip; \
        /opt/celleval/bin/pip install --no-cache-dir -r requirements-celleval.txt; \
        /opt/celleval/bin/python -c "import cell_eval, importlib.metadata as m; print('cell-eval', m.version('cell-eval'))"; \
        ln -s /opt/celleval /workspace/.env-celleval-linux; \
    else \
        echo "WITH_CELLEVAL=0 - the reported table's five cell-eval columns will read '-'"; \
    fi


# ------------------------------------------------------------------- source
COPY . /workspace

# The three mount points, created here so that `docker run` without -v still
# gets a working (if empty) tree rather than a permission error on a root-owned
# directory the daemon would create.
RUN mkdir -p /workspace/data /workspace/assets/kegg /workspace/results/runs


# --------------------------------------------------------------------- user
# Runs as a normal user matching the host's, so checkpoints written into a
# mounted results/ are owned by the person who started the container and not by
# root. On Linux:  --build-arg UID=$(id -u) --build-arg GID=$(id -g)
ARG UID=1000
ARG GID=1000
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" scpkfm; fi; \
    if ! getent passwd "${UID}" >/dev/null; then \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash scpkfm; \
    fi; \
    chown -R "${UID}:${GID}" /workspace; \
    if [ -d /opt/celleval ]; then chown -R "${UID}:${GID}" /opt/celleval; fi
USER ${UID}:${GID}

# Interactive by default. `sh run.sh` overwrites nothing silently - it refuses
# when results/runs/$RUN/checkpoint.pt already exists - but which TAG and FOLD a
# container should train is a per-experiment decision, so it is passed in rather
# than baked into the image.
#
#   docker run ... scpkfm sh run.sh                  build cache + train + evaluate
#   docker run ... scpkfm sh run.sh --no-celleval    skip the ~30 min scoring
#   docker run ... scpkfm sh run.sh --eval-only      re-score a finished run
#   docker run ... scpkfm sh test.sh --summary       read back finished runs
#   docker run ... scpkfm python -m pytest -q        the structural tests
CMD ["bash"]
