#!/bin/bash
#SBATCH --job-name=tta_batch_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=sweep_experiment/slurm_log/%x_%j.out
#SBATCH --error=sweep_experiment/slurm_log/%x_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/wc3013}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:?DATA_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/results/batched_tta_benchmark/${SLURM_JOB_ID}}"

METHODS="${METHODS:-adasteer_batched}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16}"
MAX_VIDEOS="${MAX_VIDEOS:-16}"
MAX_GROUPS="${MAX_GROUPS:-1}"

NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"
DELTA_STEPS="${DELTA_STEPS:-10}"
DELTA_LR="${DELTA_LR:-0.005}"
CAPTION_GUARD_MODE="${CAPTION_GUARD_MODE:-warn}"
FEATURE_FRAME_GUARD_MODE="${FEATURE_FRAME_GUARD_MODE:-warn}"
FIXED_CAPTION="${FIXED_CAPTION:-}"

echo "=============================================================================="
echo "Batched Independent TTA Benchmark"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Methods      : ${METHODS}"
echo "Batch sizes  : ${BATCH_SIZES}"
echo "Max videos   : ${MAX_VIDEOS}"
echo "Max groups   : ${MAX_GROUPS}"
echo "Data dir     : ${DATA_DIR}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Frames       : cond=${NUM_COND_FRAMES}, total=${NUM_FRAMES}, gen_start=${GEN_START_FRAME}"
echo "TTA window   : total=${TTA_TOTAL_FRAMES}, context=${TTA_CONTEXT_FRAMES}"
echo "=============================================================================="

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
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${OUTPUT_DIR}" "sweep_experiment/slurm_log"

cd "${PROJECT_ROOT}"
nvidia-smi --query-gpu=name,memory.free --format=csv

fixed_caption_flags=()
if [ -n "${FIXED_CAPTION}" ]; then
  fixed_caption_flags=(--fixed-caption "${FIXED_CAPTION}")
fi

"${PYTHON}" sweep_experiment/scripts/benchmark_batched_tta.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --methods "${METHODS}" \
  --batch-sizes "${BATCH_SIZES}" \
  --max-videos "${MAX_VIDEOS}" \
  --max-groups "${MAX_GROUPS}" \
  --num-cond-frames "${NUM_COND_FRAMES}" \
  --num-frames "${NUM_FRAMES}" \
  --gen-start-frame "${GEN_START_FRAME}" \
  --tta-total-frames "${TTA_TOTAL_FRAMES}" \
  --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --resolution "${RESOLUTION}" \
  --seed "${SEED}" \
  --delta-steps "${DELTA_STEPS}" \
  --delta-lr "${DELTA_LR}" \
  --caption-guard-mode "${CAPTION_GUARD_MODE}" \
  --feature-frame-guard-mode "${FEATURE_FRAME_GUARD_MODE}" \
  "${fixed_caption_flags[@]}"
