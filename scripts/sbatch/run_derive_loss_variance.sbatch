#!/bin/bash
#SBATCH --job-name=derive_loss_var
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/logs/derive_loss_var_%j.out
#SBATCH --error=sweep_experiment/logs/derive_loss_var_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"

OOD_CSV="${OOD_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/loss_var_features.csv}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT_CSV}")" sweep_experiment/logs

"${PYTHON}" scripts/derive_loss_variance.py \
    --ood-csv "${OOD_CSV}" \
    --output "${OUTPUT_CSV}"

echo "Done: ${OUTPUT_CSV}"
