#!/bin/bash
#SBATCH --job-name=wan_health
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_healthcheck_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_healthcheck_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# Load Wan + Self-Forcing weights, decode a handful of VBench-I2V images,
# optionally run one short Self-Forcing clip. Writes
#   wan_experiment/results/setup_healthcheck/report.json

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/self_forcing}"
SF_ROOT="${SF_ROOT:-${SCRATCH_BASE}/third_party/Self-Forcing}"
CKPT_ROOT="${CKPT_ROOT:-${SCRATCH_BASE}/wan-checkpoints}"
WAN_DIR="${CKPT_ROOT}/Wan2.1-T2V-1.3B"
SF_CKPT="${CKPT_ROOT}/self_forcing_dmd.pt"
I2V_DIR="${I2V_DIR:-${PROJECT_ROOT}/datasets/vbench_i2v}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/wan_experiment/results/setup_healthcheck}"
SKIP_GEN="${SKIP_GEN:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$OUT_DIR" \
    "${PROJECT_ROOT}/wan_experiment/slurm_log"

echo "=============================================================================="
echo "Wan / Self-Forcing healthcheck"
echo "  job : ${SLURM_JOB_ID:-local}  node=$(hostname)  $(date)"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_ENV}/bin/python"

"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || {
    echo "ERROR: self_forcing env cannot import torch" >&2
    exit 1
}

EXTRA=()
if [ "${SKIP_GEN}" = "1" ]; then
    EXTRA+=(--skip-gen)
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" wan_experiment/scripts/healthcheck_wan_sf.py \
    --wan-dir "${WAN_DIR}" \
    --sf-ckpt "${SF_CKPT}" \
    --sf-root "${SF_ROOT}" \
    --i2v-dir "${I2V_DIR}" \
    --out "${OUT_DIR}/report.json" \
    "${EXTRA[@]}"

echo "=============================================================================="
echo "Healthcheck wrote ${OUT_DIR}/report.json  $(date)"
echo "=============================================================================="
