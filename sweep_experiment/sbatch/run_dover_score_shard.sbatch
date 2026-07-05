#!/bin/bash
#SBATCH --job-name=dover_score
#SBATCH --output=sweep_experiment/slurm_log/dover_score_%j.out
#SBATCH --error=sweep_experiment/slurm_log/dover_score_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DOVER_ROOT="${DOVER_ROOT:-/scratch/${USER}/third_party/DOVER}"
RUN_ID="${RUN_ID:?}"
SERIES="${SERIES:?}"
OUT_SCORES="${OUT_SCORES:?}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

SCRATCH_BASE="/scratch/${USER}"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"

cd "${PROJECT_ROOT}"
export DOVER_ROOT

python3 scripts/score_videos_dover_batch.py \
  --input-dir "${SERIES}/${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --output "${OUT_SCORES}/${RUN_ID}_shard${SHARD_ID}.csv" \
  --dover-root "${DOVER_ROOT}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --device cuda

echo "Done shard ${SHARD_ID}/${NUM_SHARDS} for ${RUN_ID}"
