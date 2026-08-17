#!/bin/bash
#SBATCH --job-name=wan_i2v_chunk
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_i2v_chunk_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_i2v_chunk_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# Chunked Wan I2V: METHOD=notta|always_bon|gated_bon
#   sbatch --export=ALL,METHOD=gated_bon,HORIZON_S=30,N_VIDEOS=16,SEARCH_K=4 \
#       wan_experiment/sbatch/run_i2v_chunked.sbatch

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

HORIZON_S="${HORIZON_S:-30}"
N_VIDEOS="${N_VIDEOS:-2}"
SEED="${SEED:-0}"
METHOD="${METHOD:-notta}"
SEARCH_K="${SEARCH_K:-4}"
SEARCH_FROM="${SEARCH_FROM:-1}"
GATE_THRESHOLD="${GATE_THRESHOLD:-2.0}"
CHUNK_LATENTS="${CHUNK_LATENTS:-24}"
SERIES="${SERIES:-i2v_chunked_smoke}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/wan_experiment/results/${SERIES}/${METHOD}_h${HORIZON_S}s_shard${SHARD_ID}}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TMPDIR="${SCRATCH_BASE}/tmp"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$OUT_DIR" \
    "${PROJECT_ROOT}/wan_experiment/slurm_log"

echo "=============================================================================="
echo "Wan I2V chunked ${METHOD}"
echo "  job     : ${SLURM_JOB_ID:-local}  node=$(hostname)  $(date)"
echo "  horizon : ${HORIZON_S}s  n=${N_VIDEOS}  seed=${SEED}  k=${SEARCH_K}"
echo "  chunk   : ${CHUNK_LATENTS} latents  search_from=${SEARCH_FROM}"
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
"${PYTHON}" wan_experiment/scripts/run_i2v_chunked.py \
    --sf-root "${SF_ROOT}" \
    --wan-dir "${WAN_DIR}" \
    --sf-ckpt "${SF_CKPT}" \
    --i2v-dir "${I2V_DIR}" \
    --out-dir "${OUT_DIR}" \
    --horizon-s "${HORIZON_S}" \
    --chunk-latents "${CHUNK_LATENTS}" \
    --n "${N_VIDEOS}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    --search-k "${SEARCH_K}" \
    --search-from-chunk "${SEARCH_FROM}" \
    --gate-threshold "${GATE_THRESHOLD}" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}"

echo "=============================================================================="
echo "Wrote ${OUT_DIR}/summary.json  $(date)"
echo "=============================================================================="
