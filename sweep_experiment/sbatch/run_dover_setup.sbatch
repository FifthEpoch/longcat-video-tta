#!/bin/bash
#SBATCH --job-name=dover_setup
#SBATCH --output=sweep_experiment/slurm_log/dover_setup_%j.out
#SBATCH --error=sweep_experiment/slurm_log/dover_setup_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DOVER_ROOT="${DOVER_ROOT:-/scratch/${USER}/third_party/DOVER}"

cd "${PROJECT_ROOT}"
bash scripts/setup_dover_env.sh
