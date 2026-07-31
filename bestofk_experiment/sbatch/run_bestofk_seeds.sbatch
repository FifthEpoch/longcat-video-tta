#!/bin/bash
#SBATCH --job-name=bestofk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=bestofk_experiment/slurm_log/%x_%j.out
#SBATCH --error=bestofk_experiment/slurm_log/%x_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# ============================================================================
# Best-of-k seed generation for one chunk (frozen LongCat, no TTA).
# Generates K continuations per video from K distinct seeds and records
# per-candidate PSNR/SSIM/LPIPS + GT-free selector signals. Same geometry as the
# 1000v preview pool (cond=14, frames=28, gen_start=48) so it lines up 1:1.
#
# Env: START_VIDEO_IDX, CHUNK_SIZE, MAX_VIDEOS, NUM_SEEDS, DATA_DIR, OUTPUT_DIR.
# Submit chunks via bestofk_experiment/sbatch/submit_bestofk_seeds.sh.
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
OUTPUT_DIR="${OUTPUT_DIR:?ERROR: OUTPUT_DIR required}"

MAX_VIDEOS="${MAX_VIDEOS:-200}"
START_VIDEO_IDX="${START_VIDEO_IDX:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
NUM_SEEDS="${NUM_SEEDS:-8}"
SEED_STRIDE="${SEED_STRIDE:-1000}"
SEED="${SEED:-42}"

NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESOLUTION="${RESOLUTION:-480p}"
NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS:-0}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME" bestofk_experiment/slurm_log

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Best-of-k seed generation (chunk)"
echo "  data_dir   : ${DATA_DIR}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "  chunk      : [${START_VIDEO_IDX}, +${CHUNK_SIZE})  max=${MAX_VIDEOS}"
echo "  K seeds    : ${NUM_SEEDS}  stride=${SEED_STRIDE}"
echo "  geometry   : cond=${NUM_COND_FRAMES} frames=${NUM_FRAMES} gsf=${GEN_START_FRAME}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.free --format=csv || true

SAVE_FLAG=""
if [ "${NO_SAVE_VIDEOS}" = "1" ]; then SAVE_FLAG="--no-save-videos"; fi

"$PYTHON" bestofk_experiment/scripts/run_bestofk_seeds.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-videos "${MAX_VIDEOS}" \
  --start-video-idx "${START_VIDEO_IDX}" \
  --chunk-size "${CHUNK_SIZE}" \
  --num-seeds "${NUM_SEEDS}" \
  --seed-stride "${SEED_STRIDE}" \
  --seed "${SEED}" \
  --num-cond-frames "${NUM_COND_FRAMES}" \
  --num-frames "${NUM_FRAMES}" \
  --gen-start-frame "${GEN_START_FRAME}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --resolution "${RESOLUTION}" \
  ${SAVE_FLAG}

echo "Done chunk [${START_VIDEO_IDX}, +${CHUNK_SIZE}). Summary: ${OUTPUT_DIR}/summary.json"
