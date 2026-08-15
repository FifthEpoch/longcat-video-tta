#!/bin/bash
#SBATCH --job-name=wan_sf_env
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_sf_env_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_sf_env_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# Create /scratch/$USER/conda-envs/self_forcing (Self-Forcing official recipe).
# GPU node: flash-attn needs a CUDA compiler. Do NOT install into longcat.
#
#   sbatch wan_experiment/sbatch/setup_env.sbatch
#   FORCE=1 to rebuild the env from scratch.

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/self_forcing}"
SF_ROOT="${SF_ROOT:-${SCRATCH_BASE}/third_party/Self-Forcing}"
FORCE="${FORCE:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" \
    "${PROJECT_ROOT}/wan_experiment/slurm_log"

echo "=============================================================================="
echo "Wan / Self-Forcing env setup"
echo "  job     : ${SLURM_JOB_ID:-local}  node=$(hostname)  $(date)"
echo "  env     : ${CONDA_ENV}"
echo "  sf_root : ${SF_ROOT}"
echo "  force   : ${FORCE}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
unset PYTHONHOME
unset PYTHONPATH

if [ "${FORCE}" = "1" ] && [ -d "${CONDA_ENV}" ]; then
    echo "FORCE=1 -> removing existing env"
    conda env remove -p "${CONDA_ENV}" -y || rm -rf "${CONDA_ENV}"
fi

if [ ! -x "${CONDA_ENV}/bin/python" ]; then
    echo "Creating conda env at ${CONDA_ENV} (python 3.10)"
    conda create -p "${CONDA_ENV}" python=3.10 -y
else
    echo "Env already exists: ${CONDA_ENV}"
fi

conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_ENV}/bin/python"
PIP="${CONDA_ENV}/bin/pip"

# Clone Self-Forcing (pinned to main; recorded in the healthcheck report).
if [ ! -d "${SF_ROOT}/.git" ]; then
    mkdir -p "$(dirname "${SF_ROOT}")"
    git clone --depth=1 https://github.com/guandeh17/self-forcing.git "${SF_ROOT}"
else
    echo "Self-Forcing clone exists; fetch (not rebase) to stay current"
    git -C "${SF_ROOT}" fetch --depth=1 origin main || true
fi
echo "Self-Forcing HEAD: $(git -C "${SF_ROOT}" rev-parse --short HEAD)  $(git -C "${SF_ROOT}" log -1 --pretty=%s)"

# Official requirements.txt pulls nvidia-pyindex / nvidia-tensorrt / pycuda
# (TensorRT ONNX export + the Self-Forcing demo server). Those need CUDA
# headers; job 15772007 died with `cuda.h: No such file or directory` while
# building pycuda. PyTorch inference (sample / I2V continuation) does not
# use them. Strip before pip; treat the same as optional flash-attn.
REQ_FILTERED="${TMPDIR}/sf_requirements_infer.txt"
grep -vE '^(nvidia-pyindex|nvidia-tensorrt|pycuda)([[:space:]]|$)' \
    "${SF_ROOT}/requirements.txt" > "${REQ_FILTERED}"
echo "pip install -r requirements.txt (TensorRT/pycuda stripped)"
echo "---- filtered-out extras ----"
comm -13 <(sort "${REQ_FILTERED}") <(sort "${SF_ROOT}/requirements.txt") || true
"${PIP}" install --upgrade pip
"${PIP}" install -r "${REQ_FILTERED}"

# flash-attn is in the official recipe but is a frequent compile failure.
# Do not fail the job if it does not build; inference still loads without it.
set +e
"${PIP}" install flash-attn --no-build-isolation
FLASH_RC=$?
set -e
if [ "${FLASH_RC}" -ne 0 ]; then
    echo "WARNING: flash-attn install failed (rc=${FLASH_RC}); continuing without it"
fi

echo "python setup.py develop (Self-Forcing)"
cd "${SF_ROOT}"
"${PYTHON}" setup.py develop

echo "---- env sanity ----"
"${PYTHON}" - <<'PY'
import sys
print("python", sys.version)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
import numpy
print("numpy", numpy.__version__)
import diffusers, transformers
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
try:
    import flash_attn
    print("flash_attn", getattr(flash_attn, "__version__", "present"))
except Exception as e:
    print("flash_attn UNAVAILABLE:", type(e).__name__, e)
PY

echo "=============================================================================="
echo "Env setup complete: $(date)"
echo "=============================================================================="
