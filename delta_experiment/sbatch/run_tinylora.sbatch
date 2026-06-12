#!/bin/bash
#SBATCH --job-name=tinylora_tta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=delta_experiment/sbatch/slurm_log/tinylora_%j.out
#SBATCH --error=delta_experiment/sbatch/slurm_log/tinylora_%j.err

# ============================================================================
# TinyLoRA TTA: Ultra-low-rank SVD-based adapters for LongCat-Video
#
# Adapts the TinyLoRA method (Morris et al., 2026) for per-video TTA.
# Typical trainable params: 192 (default) down to ~4 (with weight tying).
#
# Estimated time: ~2-4 hours for 100 videos (training only, no generation)
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS="${NUM_VIDEOS:-100}"
TTA_STEPS="${TTA_STEPS:-20}"
TTA_LR="${TTA_LR:-1e-3}"
SVD_RANK="${SVD_RANK:-2}"
ALPHA="${ALPHA:-1.0}"
N_TIE="${N_TIE:-1}"
TARGET_PRESET="${TARGET_PRESET:-qkv_proj}"
TARGET_MODULES="${TARGET_MODULES:-}"
TARGET_BLOCKS="${TARGET_BLOCKS:-all}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-${GEN_START_FRAME}}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-${NUM_COND_FRAMES}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"
START_VIDEO_IDX="${START_VIDEO_IDX:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
SKIP_GENERATION="${SKIP_GENERATION:-}"
NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS:-}"

# Early stopping configuration
ES_DISABLE="${ES_DISABLE:-}"
ES_CHECK_EVERY="${ES_CHECK_EVERY:-5}"
ES_PATIENCE="${ES_PATIENCE:-3}"
ES_ANCHOR_SIGMAS="${ES_ANCHOR_SIGMAS:-0.25,0.5,0.75}"
ES_NOISE_DRAWS="${ES_NOISE_DRAWS:-2}"
ES_STRATEGY="${ES_STRATEGY:-patience}"
ES_HOLDOUT_FRACTION="${ES_HOLDOUT_FRACTION:-0.25}"

# Data augmentation
AUG_ENABLED="${AUG_ENABLED:-}"
AUG_FLIP="${AUG_FLIP:-}"
AUG_ROTATE_DEG="${AUG_ROTATE_DEG:-10.0}"
AUG_ROTATE_RANDOM_MIN="${AUG_ROTATE_RANDOM_MIN:-5.0}"
AUG_ROTATE_RANDOM_MAX="${AUG_ROTATE_RANDOM_MAX:-15.0}"
AUG_ROTATE_RANDOM_COUNT="${AUG_ROTATE_RANDOM_COUNT:-2}"
AUG_ROTATE_RANDOM_STEP="${AUG_ROTATE_RANDOM_STEP:-1.0}"
AUG_ROTATE_ZOOM="${AUG_ROTATE_ZOOM:-1}"
AUG_SPEED_FACTORS="${AUG_SPEED_FACTORS:-}"

# CLIP gate
CLIP_GATE_ENABLED="${CLIP_GATE_ENABLED:-}"
CLIP_GATE_THRESHOLD="${CLIP_GATE_THRESHOLD:-0.0}"
CLIP_GATE_BACKEND="${CLIP_GATE_BACKEND:-clip}"
CLIP_GATE_MODEL="${CLIP_GATE_MODEL:-openai/clip-vit-large-patch14}"
CLIP_GATE_SAMPLE_FRAMES="${CLIP_GATE_SAMPLE_FRAMES:-4}"
CLIP_GATE_AGGREGATION="${CLIP_GATE_AGGREGATION:-mean}"
CLIP_GATE_SAMPLING_MODE="${CLIP_GATE_SAMPLING_MODE:-full_window}"
CLIP_GATE_LATE_FRACTION="${CLIP_GATE_LATE_FRACTION:-0.4}"
CLIP_GATE_LATE_ONLY="${CLIP_GATE_LATE_ONLY:-}"
CLIP_GATE_FAIL_OPEN="${CLIP_GATE_FAIL_OPEN:-1}"
CLIP_GATE_LOG_ONLY="${CLIP_GATE_LOG_ONLY:-}"

# Caption guard
CAPTION_GUARD_MODE="${CAPTION_GUARD_MODE:-warn}"
FIXED_CAPTION="${FIXED_CAPTION:-}"
FEATURE_FRAME_GUARD_MODE="${FEATURE_FRAME_GUARD_MODE:-warn}"

# TTA caption ablation: when set, drop captions in the TTA training step
# (inference prompt unchanged). Off by default.
TTA_DISABLE_CAPTION="${TTA_DISABLE_CAPTION:-}"

# Anchor-frame x0 consistency auxiliary loss (Modification 1 of
# sweep_experiment/reports/LITERATURE_tta_recipe_modifications_2026-06-12.md).
# Default 0.0 = byte-identical to pre-patch behaviour (the --anchor-x0-weight
# flag is not passed to the runner). Set > 0 to activate.
ANCHOR_X0_WEIGHT="${ANCHOR_X0_WEIGHT:-0.0}"

# Online FVD / FID / VBench++
COMPUTE_FVD="${COMPUTE_FVD:-0}"
COMPUTE_FID="${COMPUTE_FID:-0}"
COMPUTE_VBENCH="${COMPUTE_VBENCH:-0}"
MIN_FVD_VIDEOS="${MIN_FVD_VIDEOS:-256}"

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${SCRATCH_BASE}/longcat-video-tta/datasets/panda_100_480p}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/delta_experiment/results/tinylora_r${SVD_RANK}_ntie${N_TIE}_${TARGET_PRESET}}"

echo "=============================================================================="
echo "TinyLoRA TTA for LongCat-Video"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Configuration:"
echo "  Videos         : ${NUM_VIDEOS}"
echo "  TTA steps      : ${TTA_STEPS}"
echo "  TTA LR         : ${TTA_LR}"
echo "  SVD rank       : ${SVD_RANK}"
echo "  Alpha          : ${ALPHA}"
echo "  Weight tying   : n_tie=${N_TIE}"
echo "  Target preset  : ${TARGET_PRESET}"
echo "  Target modules : ${TARGET_MODULES:-<from preset>}"
echo "  Target blocks  : ${TARGET_BLOCKS}"
echo "  Cond frames    : ${NUM_COND_FRAMES}"
echo "  Num frames     : ${NUM_FRAMES}"
echo "  TTA total      : ${TTA_TOTAL_FRAMES}"
echo "  TTA context    : ${TTA_CONTEXT_FRAMES}"
echo "  Augmentation   : enabled=${AUG_ENABLED:-0}, flip=${AUG_FLIP:-0}"
echo "  CLIP gate      : enabled=${CLIP_GATE_ENABLED:-0}"
echo "  Online FVD     : ${COMPUTE_FVD}  FID: ${COMPUTE_FID}"
echo "  Save videos    : $([ "${NO_SAVE_VIDEOS}" = "1" ] && echo "no" || echo "yes")"
echo "  TTA no-caption : ${TTA_DISABLE_CAPTION:-0}"
echo "  Anchor-x0 wt   : ${ANCHOR_X0_WEIGHT}"
echo "  Checkpoint dir : ${CHECKPOINT_DIR}"
echo "  Data dir       : ${DATA_DIR}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "=============================================================================="

# ============================================================================
# Environment Setup
#
# CRITICAL: module load sets PYTHONHOME / PYTHONPATH to the system anaconda,
# which breaks the conda env's sys.path (wrong Python version, missing
# packages). We MUST unset these after conda activate. See:
#   sweep_experiment/docs/SBATCH_CONDA_BEST_PRACTICES.md
# ============================================================================
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

# CRITICAL: unset PYTHONHOME and PYTHONPATH set by "module load anaconda3/..."
# Without this, even explicit $PYTHON uses the system anaconda's sys.path,
# causing "No module named 'torch'" and similar import errors.
unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

# Verify the environment is correct (catches broken conda activation early)
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed — conda env is broken or PYTHONHOME was not unset" >&2
    exit 1
}

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
export VBENCH_CACHE_DIR="${SCRATCH_BASE}/.cache/vbench"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$VBENCH_CACHE_DIR"

cd "$PROJECT_ROOT"

# Create output and log directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "delta_experiment/sbatch/slurm_log"

# ============================================================================
# GPU Info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Build extra flags
# ============================================================================
EXTRA_FLAGS=""
if [ -n "${TARGET_MODULES}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --target-modules ${TARGET_MODULES}"
fi
if [ "${SKIP_GENERATION}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --skip-generation"
fi

# Early stopping flags
ES_FLAGS=""
if [ "${ES_DISABLE}" = "1" ]; then
    ES_FLAGS="${ES_FLAGS} --es-disable"
fi
ES_FLAGS="${ES_FLAGS} --es-check-every ${ES_CHECK_EVERY}"
ES_FLAGS="${ES_FLAGS} --es-patience ${ES_PATIENCE}"
ES_FLAGS="${ES_FLAGS} --es-anchor-sigmas ${ES_ANCHOR_SIGMAS}"
ES_FLAGS="${ES_FLAGS} --es-noise-draws ${ES_NOISE_DRAWS}"
ES_FLAGS="${ES_FLAGS} --es-strategy ${ES_STRATEGY}"
ES_FLAGS="${ES_FLAGS} --es-holdout-fraction ${ES_HOLDOUT_FRACTION}"

# Save / video flags
SAVE_FLAGS=""
if [ "${NO_SAVE_VIDEOS}" = "1" ]; then
    SAVE_FLAGS="--no-save-videos"
fi

# Augmentation flags
AUG_FLAGS=""
if [ "${AUG_ENABLED}" = "1" ]; then
    AUG_FLAGS="${AUG_FLAGS} --aug-enabled"
fi
if [ "${AUG_FLIP}" = "1" ]; then
    AUG_FLAGS="${AUG_FLAGS} --aug-flip"
fi
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-deg ${AUG_ROTATE_DEG}"
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-random-min ${AUG_ROTATE_RANDOM_MIN}"
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-random-max ${AUG_ROTATE_RANDOM_MAX}"
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-random-count ${AUG_ROTATE_RANDOM_COUNT}"
AUG_FLAGS="${AUG_FLAGS} --aug-rotate-random-step ${AUG_ROTATE_RANDOM_STEP}"
if [ "${AUG_ROTATE_ZOOM}" != "1" ]; then
    AUG_FLAGS="${AUG_FLAGS} --no-aug-rotate-zoom"
fi
if [ -n "${AUG_SPEED_FACTORS}" ]; then
    AUG_FLAGS="${AUG_FLAGS} --aug-speed-factors ${AUG_SPEED_FACTORS}"
fi

# CLIP gate flags
CLIP_FLAGS=""
if [ "${CLIP_GATE_ENABLED}" = "1" ]; then
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-enabled"
fi
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-threshold ${CLIP_GATE_THRESHOLD}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-backend ${CLIP_GATE_BACKEND}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-model ${CLIP_GATE_MODEL}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-sample-frames ${CLIP_GATE_SAMPLE_FRAMES}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-aggregation ${CLIP_GATE_AGGREGATION}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-sampling-mode ${CLIP_GATE_SAMPLING_MODE}"
CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-late-fraction ${CLIP_GATE_LATE_FRACTION}"
if [ "${CLIP_GATE_LATE_ONLY}" = "1" ]; then
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-late-only"
fi
if [ "${CLIP_GATE_LOG_ONLY}" = "1" ]; then
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-log-only"
fi
if [ "${CLIP_GATE_FAIL_OPEN}" = "1" ]; then
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-fail-open"
else
    CLIP_FLAGS="${CLIP_FLAGS} --clip-gate-fail-closed"
fi

# Caption / feature guard flags
CAPTION_FLAGS="--caption-guard-mode ${CAPTION_GUARD_MODE}"
if [ -n "${FIXED_CAPTION}" ]; then
    CAPTION_FLAGS="${CAPTION_FLAGS} --fixed-caption $(printf '%q' "${FIXED_CAPTION}")"
fi
FEATURE_FLAGS="--feature-frame-guard-mode ${FEATURE_FRAME_GUARD_MODE}"

# TTA caption-ablation flag
NOPROMPT_FLAGS=""
if [ "${TTA_DISABLE_CAPTION}" = "1" ]; then
    NOPROMPT_FLAGS="--tta-disable-caption"
fi

# Anchor-frame x0 loss flag — only emitted when > 0 so the byte-identical
# pre-patch guarantee holds for any submit wrapper that doesn't set
# ANCHOR_X0_WEIGHT.
X0_FLAGS=""
case "${ANCHOR_X0_WEIGHT}" in
    0|0.0|0.00|0.000|"")
        : ;;
    *)
        X0_FLAGS="--anchor-x0-weight ${ANCHOR_X0_WEIGHT}"
        ;;
esac

# Online eval flags
EVAL_FLAGS=""
[ "${COMPUTE_FVD}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-fvd"
[ "${COMPUTE_FID}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-fid"
[ "${COMPUTE_VBENCH}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-vbench"
EVAL_FLAGS="${EVAL_FLAGS} --min-fvd-videos ${MIN_FVD_VIDEOS}"

# ============================================================================
# Run TinyLoRA TTA
# ============================================================================
echo "Starting TinyLoRA TTA..."

"$PYTHON" delta_experiment/scripts/run_tinylora.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${NUM_VIDEOS}" \
    --start-video-idx "${START_VIDEO_IDX}" \
    --chunk-size "${CHUNK_SIZE}" \
    --svd-rank "${SVD_RANK}" \
    --alpha "${ALPHA}" \
    --n-tie "${N_TIE}" \
    --target-preset "${TARGET_PRESET}" \
    --target-blocks "${TARGET_BLOCKS}" \
    --tta-steps "${TTA_STEPS}" \
    --tta-lr "${TTA_LR}" \
    --num-cond-frames "${NUM_COND_FRAMES}" \
    --num-frames "${NUM_FRAMES}" \
    --gen-start-frame "${GEN_START_FRAME}" \
    --tta-total-frames "${TTA_TOTAL_FRAMES}" \
    --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --resolution "${RESOLUTION}" \
    --seed "${SEED}" \
    ${SAVE_FLAGS} \
    ${EXTRA_FLAGS} \
    ${ES_FLAGS} \
    ${AUG_FLAGS} \
    ${CLIP_FLAGS} \
    ${CAPTION_FLAGS} \
    ${FEATURE_FLAGS} \
    ${EVAL_FLAGS} \
    ${NOPROMPT_FLAGS} \
    ${X0_FLAGS}

echo ""
echo "=============================================================================="
echo "TinyLoRA TTA Complete"
echo "=============================================================================="
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}/summary.json"
echo "=============================================================================="
