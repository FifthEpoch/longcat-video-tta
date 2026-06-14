#!/bin/bash
#SBATCH --job-name=extract_flow_shape
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/logs/extract_flow_shape_%j.out
#SBATCH --error=sweep_experiment/logs/extract_flow_shape_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"

VIDEOS_DIR="${VIDEOS_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/flow_shape_features.csv}"
TTA_VISIBLE_FRAMES="${TTA_VISIBLE_FRAMES:-auto}"
TARGET_SIZE="${TARGET_SIZE:-256x320}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-0}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT_CSV}")" sweep_experiment/logs

EXTRA=()
[ "${RESUME}" = "1" ] && EXTRA+=(--resume)

"${PYTHON}" scripts/extract_flow_shape_features.py \
    --videos-dir "${VIDEOS_DIR}" \
    --output "${OUTPUT_CSV}" \
    --tta-visible-frames "${TTA_VISIBLE_FRAMES}" \
    --target-size "${TARGET_SIZE}" \
    --device "${DEVICE}" \
    "${EXTRA[@]}"

echo "Done: ${OUTPUT_CSV}"
