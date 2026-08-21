#!/bin/bash
#SBATCH --job-name=wan_v2v_chunk
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_v2v_chunk_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_v2v_chunk_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# Chunked Wan V2V: METHOD=notta|seed_bon|hinge_bon|late_bon|hist_drop|
#   good_backtrack|cached_bon|sink|motion_bon|shift_search|backtrack|knob_probe
#   sbatch --export=ALL,METHOD=notta,HORIZON_S=30,N_VIDEOS=2 \
#       wan_experiment/sbatch/run_v2v_chunked.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/self_forcing}"
SF_ROOT="${SF_ROOT:-${SCRATCH_BASE}/third_party/Self-Forcing}"
LL_ROOT="${LL_ROOT:-${SCRATCH_BASE}/third_party/LongLive}"
RF_ROOT="${RF_ROOT:-${SCRATCH_BASE}/third_party/RollingForcing}"
CKPT_ROOT="${CKPT_ROOT:-${SCRATCH_BASE}/wan-checkpoints}"
WAN_DIR="${CKPT_ROOT}/Wan2.1-T2V-1.3B"
SF_CKPT="${CKPT_ROOT}/self_forcing_dmd.pt"
LL_BASE="${LL_BASE:-${CKPT_ROOT}/longlive/models/longlive_base.pt}"
LL_LORA="${LL_LORA:-${CKPT_ROOT}/longlive/models/lora.pt}"
RF_CKPT="${RF_CKPT:-${CKPT_ROOT}/rolling_forcing_dmd.pt}"

HORIZON_S="${HORIZON_S:-30}"
N_VIDEOS="${N_VIDEOS:-8}"
SEED="${SEED:-0}"
METHOD="${METHOD:-notta}"
SEARCH_K="${SEARCH_K:-4}"
SEARCH_FROM="${SEARCH_FROM:-0}"
PREFIX_LATENTS="${PREFIX_LATENTS:-9}"
CHUNK_LATENTS="${CHUNK_LATENTS:-21}"
BACKTRACK_THRESHOLD="${BACKTRACK_THRESHOLD:-2.0}"
BACKTRACK_MOTION_FRAC="${BACKTRACK_MOTION_FRAC:-0.4}"
SERIES="${SERIES:-v2v_panda_bakeoff_8v}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
LIVE_MIN="${LIVE_MIN:-0.012}"
SINK_SIZE="${SINK_SIZE:-3}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-12}"
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
echo "Wan V2V chunked ${METHOD}"
echo "  job     : ${SLURM_JOB_ID:-local}  node=$(hostname)  $(date)"
echo "  horizon : ${HORIZON_S}s  n=${N_VIDEOS}  seed=${SEED}  k=${SEARCH_K}"
echo "  prefix  : ${PREFIX_LATENTS} lat  chunk=${CHUNK_LATENTS}  search_from=${SEARCH_FROM}"
echo "  series  : ${SERIES}  shard=${SHARD_ID}/${NUM_SHARDS}"
echo "  videos  : ${VIDEO_DIR}"
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

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: video dir missing: ${VIDEO_DIR}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" wan_experiment/scripts/run_v2v_chunked.py \
    --sf-root "${SF_ROOT}" \
    --wan-dir "${WAN_DIR}" \
    --sf-ckpt "${SF_CKPT}" \
    --ll-root "${LL_ROOT}" \
    --ll-base "${LL_BASE}" \
    --ll-lora "${LL_LORA}" \
    --rf-root "${RF_ROOT}" \
    --rf-ckpt "${RF_CKPT}" \
    --video-dir "${VIDEO_DIR}" \
    --out-dir "${OUT_DIR}" \
    --horizon-s "${HORIZON_S}" \
    --prefix-latents "${PREFIX_LATENTS}" \
    --chunk-latents "${CHUNK_LATENTS}" \
    --n "${N_VIDEOS}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    --search-k "${SEARCH_K}" \
    --search-from-chunk "${SEARCH_FROM}" \
    --backtrack-threshold "${BACKTRACK_THRESHOLD}" \
    --backtrack-motion-frac "${BACKTRACK_MOTION_FRAC}" \
    --live-min "${LIVE_MIN}" \
    --sink-size "${SINK_SIZE}" \
    --local-attn-size "${LOCAL_ATTN_SIZE}" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}"

echo "=============================================================================="
echo "Wrote ${OUT_DIR}/summary.json  $(date)"
echo "=============================================================================="
