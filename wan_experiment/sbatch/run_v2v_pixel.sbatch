#!/bin/bash
#SBATCH --job-name=wan_v2v_pixel
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_v2v_pixel_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_v2v_pixel_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# Paired 30 s PSNR/SSIM/LPIPS on already-generated caption V2V.
# L40S. Do not sit on the H200 cap. Skip-existing per-video json.

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/self_forcing}"
SERIES_DIR="${SERIES_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_128v}"
METHODS="${METHODS:-notta rolling_notta sf_pseudo sf_always_search}"

cd "${PROJECT_ROOT}"
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

PYTHON="${CONDA_ENV}/bin/python"
echo "pixel series=${SERIES_DIR}"
echo "methods=${METHODS}"
"${PYTHON}" -u wan_experiment/scripts/score_v2v_pixel_metrics.py \
    --series-dir "${SERIES_DIR}" \
    --methods ${METHODS} \
    --device cuda \
    --lpips-step "${LPIPS_STEP:-8}"
