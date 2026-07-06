#!/bin/bash
#SBATCH --job-name=verifier_score
#SBATCH --output=sweep_experiment/slurm_log/verifier_score_%j.out
#SBATCH --error=sweep_experiment/slurm_log/verifier_score_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
BACKEND="${BACKEND:?}"
RUN_ID="${RUN_ID:?}"
SERIES="${SERIES:?}"
OUT_SCORES="${OUT_SCORES:?}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
VIDEOALIGN_ROOT="${VIDEOALIGN_ROOT:-/scratch/${USER}/third_party/VideoAlign}"
VISIONREWARD_ROOT="${VISIONREWARD_ROOT:-/scratch/${USER}/third_party/VisionReward}"
CAPTIONS_CSV="${CAPTIONS_CSV:-${PROJECT_ROOT}/datasets/panda_ood_budget_pilot_480p/metadata.csv}"

SCRATCH_BASE="/scratch/${USER}"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"

cd "${PROJECT_ROOT}"
export VIDEOALIGN_ROOT VISIONREWARD_ROOT

python3 scripts/score_probe_videos_verifier.py \
  --backend "${BACKEND}" \
  --input-dir "${SERIES}/${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --output "${OUT_SCORES}/${RUN_ID}_${BACKEND}_shard${SHARD_ID}.csv" \
  --captions-csv "${CAPTIONS_CSV}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --device cuda

echo "Done ${BACKEND} ${RUN_ID} shard ${SHARD_ID}/${NUM_SHARDS}"
