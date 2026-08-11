#!/bin/bash
#SBATCH --job-name=lh_drift
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/sbatch/slurm_log/lh_drift_%j.out
#SBATCH --error=delta_experiment/sbatch/slurm_log/lh_drift_%j.err

# ============================================================================
# Long-horizon NOTTA drift diagnostic (see 2026-08-06 problem-difficulty memo).
# Runs true autoregressive rollout (NOTTA, delta-free) for NUM_CHUNKS chunks and
# records per-chunk GT-free drift signals + GT metrics where GT overlaps.
#
# The whole point: does LongCat degrade over a long autoregressive rollout? If
# yes -> headroom exists for TTA/steering/correction. If no -> the base model is
# too strong for this framing and we should switch base models.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ---- config (env-overridable) ---------------------------------------------
NUM_VIDEOS="${NUM_VIDEOS:-24}"
NUM_CHUNKS="${NUM_CHUNKS:-8}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"
START_VIDEO_IDX="${START_VIDEO_IDX:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
# rollout protocol + intervention
METHOD="${METHOD:-notta}"                 # notta | delta | delta_stream | bestof
ROLLOUT_MODE="${ROLLOUT_MODE:-reencode}"  # reencode | native
DELTA_STEPS="${DELTA_STEPS:-10}"
DELTA_LR="${DELTA_LR:-1e-3}"
DELTA_PLACEMENT="${DELTA_PLACEMENT:-adaln}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-0}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-0}"
STREAM_REFIT_STEPS="${STREAM_REFIT_STEPS:-5}"
STREAM_REFIT_LR="${STREAM_REFIT_LR:-0}"
STREAM_BLEND="${STREAM_BLEND:-0.5}"
STREAM_TARGET="${STREAM_TARGET:-generated}"   # generated | clean
# best-of-N test-time search (only used when METHOD=bestof)
SEARCH_K="${SEARCH_K:-4}"
SEARCH_SEAM_WEIGHT="${SEARCH_SEAM_WEIGHT:-1.0}"
# TTC anchored correction (only used when METHOD=ttc|ttc_gated)
TTC_SIGMA_THRESHOLD="${TTC_SIGMA_THRESHOLD:-0.3}"
TTC_CADENCE="${TTC_CADENCE:-1}"
TTC_WEIGHT="${TTC_WEIGHT:-0.1}"
TTC_FULL_LATENT="${TTC_FULL_LATENT:-0}"          # 1 => --ttc-full-latent
TTC_GATE_THRESHOLD="${TTC_GATE_THRESHOLD:-0.15}"

# ---- paths ----------------------------------------------------------------
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/results/diag_longhorizon_drift}"

echo "=============================================================================="
echo "Long-horizon drift diagnostic"
echo "  Job ID     : ${SLURM_JOB_ID:-<local>}   Node: $(hostname)   $(date)"
echo "  Method     : ${METHOD}   rollout-mode=${ROLLOUT_MODE}"
echo "  Videos     : ${NUM_VIDEOS}   chunks=${NUM_CHUNKS}"
echo "  Geometry   : cond=${NUM_COND_FRAMES} frames=${NUM_FRAMES} gsf=${GEN_START_FRAME}"
echo "  Delta      : steps=${DELTA_STEPS} lr=${DELTA_LR} placement=${DELTA_PLACEMENT}"
echo "  Sampler    : steps=${NUM_INFERENCE_STEPS} cfg=${GUIDANCE_SCALE} seed=${SEED}"
echo "  Data dir   : ${DATA_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "=============================================================================="

# ---- environment ----------------------------------------------------------
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "✓ Activated conda environment: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi
unset PYTHONHOME
unset PYTHONPATH
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"
mkdir -p "${OUTPUT_DIR}" "delta_experiment/sbatch/slurm_log"

nvidia-smi --query-gpu=name,memory.free --format=csv || true
echo ""

python delta_experiment/scripts/diag_longhorizon_drift.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${NUM_VIDEOS}" \
    --num-chunks "${NUM_CHUNKS}" \
    --num-cond-frames "${NUM_COND_FRAMES}" \
    --num-frames "${NUM_FRAMES}" \
    --gen-start-frame "${GEN_START_FRAME}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --resolution "${RESOLUTION}" \
    --seed "${SEED}" \
    --start-video-idx "${START_VIDEO_IDX}" \
    --chunk-size "${CHUNK_SIZE}" \
    --method "${METHOD}" \
    --rollout-mode "${ROLLOUT_MODE}" \
    --delta-steps "${DELTA_STEPS}" \
    --delta-lr "${DELTA_LR}" \
    --delta-placement "${DELTA_PLACEMENT}" \
    --tta-total-frames "${TTA_TOTAL_FRAMES}" \
    --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
    --stream-refit-steps "${STREAM_REFIT_STEPS}" \
    --stream-refit-lr "${STREAM_REFIT_LR}" \
    --stream-blend "${STREAM_BLEND}" \
    --stream-target "${STREAM_TARGET}" \
    --search-k "${SEARCH_K}" \
    --search-seam-weight "${SEARCH_SEAM_WEIGHT}" \
    --ttc-sigma-threshold "${TTC_SIGMA_THRESHOLD}" \
    --ttc-cadence "${TTC_CADENCE}" \
    --ttc-weight "${TTC_WEIGHT}" \
    --ttc-gate-threshold "${TTC_GATE_THRESHOLD}" \
    $( [ "${TTC_FULL_LATENT}" = "1" ] && echo "--ttc-full-latent" )

echo ""
echo "=============================================================================="
echo "Drift diagnostic complete: $(date)"
echo "Summary : ${OUTPUT_DIR}/summary.json"
echo "Plots   : python scripts/plot_drift_curves.py --summary ${OUTPUT_DIR}/summary.json --out-dir ${OUTPUT_DIR}/plots"
echo "=============================================================================="
