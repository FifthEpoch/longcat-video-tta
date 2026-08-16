#!/bin/bash
#SBATCH --job-name=wan_i2v_notta
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_i2v_notta_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_i2v_notta_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# NOTTA I2V continuation on Wan2.1-1.3B + Self-Forcing.
# Env vars: HORIZON_S (default 5), N_VIDEOS (default 2), SEED (default 0),
#           SERIES (default i2v_notta_smoke), SHARD_ID / NUM_SHARDS.
#
#   sbatch --export=ALL,HORIZON_S=5,N_VIDEOS=2 wan_experiment/sbatch/run_i2v_notta.sbatch

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

HORIZON_S="${HORIZON_S:-5}"
N_VIDEOS="${N_VIDEOS:-2}"
SEED="${SEED:-0}"
SERIES="${SERIES:-i2v_notta_smoke}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/wan_experiment/results/${SERIES}/h${HORIZON_S}s_shard${SHARD_ID}}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$OUT_DIR" \
    "${PROJECT_ROOT}/wan_experiment/slurm_log"

echo "=============================================================================="
echo "Wan I2V NOTTA continuation"
echo "  job     : ${SLURM_JOB_ID:-local}  node=$(hostname)  $(date)"
echo "  horizon : ${HORIZON_S}s   n=${N_VIDEOS}  seed=${SEED}"
echo "  series  : ${SERIES}  shard=${SHARD_ID}/${NUM_SHARDS}"
echo "  out     : ${OUT_DIR}"
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

cd "${PROJECT_ROOT}"
"${PYTHON}" wan_experiment/scripts/run_i2v_continuation.py \
    --sf-root "${SF_ROOT}" \
    --wan-dir "${WAN_DIR}" \
    --sf-ckpt "${SF_CKPT}" \
    --i2v-dir "${I2V_DIR}" \
    --out-dir "${OUT_DIR}" \
    --horizon-s "${HORIZON_S}" \
    --n "${N_VIDEOS}" \
    --seed "${SEED}" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}"

echo "=============================================================================="
echo "Wrote ${OUT_DIR}/summary.json  $(date)"
echo "=============================================================================="
