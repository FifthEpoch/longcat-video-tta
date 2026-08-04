#!/bin/bash
#SBATCH --job-name=diag_sampler
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=comparison_methods/slurm_log/diag_sampler_%j.out
#SBATCH --error=comparison_methods/slurm_log/diag_sampler_%j.err

# Localize WHY the differentiable LongCat sampler yields PSNR~9/SSIM~0.05 while
# the real pipeline gets ~19 dB. Short (2 videos), so it won't trip the util
# policy. See comparison_methods/scripts/diagnose_sampler.py header for the read.

set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
NUM_VIDEOS="${NUM_VIDEOS:-2}"
STEPS="${STEPS:-50}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH_BASE}/conda-envs/longcat"
unset PYTHONHOME PYTHONPATH
cd "${PROJECT_ROOT}"
mkdir -p comparison_methods/slurm_log

python3 comparison_methods/scripts/diagnose_sampler.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --num-videos "${NUM_VIDEOS}" \
  --num-inference-steps "${STEPS}"
