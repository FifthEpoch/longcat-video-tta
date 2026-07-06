#!/bin/bash
#SBATCH --job-name=verifier_eval
#SBATCH --output=sweep_experiment/slurm_log/verifier_eval_%j.out
#SBATCH --error=sweep_experiment/slurm_log/verifier_eval_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"

SCRATCH_BASE="/scratch/${USER}"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"

cd "${PROJECT_ROOT}"
bash scripts/run_verifier_options_eval.sh
