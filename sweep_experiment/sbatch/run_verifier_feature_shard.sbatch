#!/bin/bash
#SBATCH --job-name=verifier_feat
#SBATCH --output=sweep_experiment/slurm_log/verifier_feat_%j.out
#SBATCH --error=sweep_experiment/slurm_log/verifier_feat_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
MODE="${MODE:?}"
RUN_ID="${RUN_ID:?}"
SERIES="${SERIES:?}"
OUT_FEATURES="${OUT_FEATURES:?}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/longcat-video-checkpoints}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

SCRATCH_BASE="/scratch/${USER}"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"

cd "${PROJECT_ROOT}"

python3 scripts/extract_probe_learned_features.py \
  --mode "${MODE}" \
  --series-root "${SERIES}" \
  --run-id "${RUN_ID}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output "${OUT_FEATURES}/${RUN_ID}_${MODE}_shard${SHARD_ID}.csv" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --device cuda

echo "Done ${MODE} ${RUN_ID} shard ${SHARD_ID}/${NUM_SHARDS}"
