#!/bin/bash
#SBATCH --job-name=resize_vids
#SBATCH --output=datasets/slurm_log/resize_%j.out
#SBATCH --error=datasets/slurm_log/resize_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

SCRATCH="/scratch/wc3013"
REPO="${SCRATCH}/longcat-video-tta"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH}/conda-envs/longcat"

cd "${REPO}"
mkdir -p datasets/slurm_log

python datasets/resize_videos.py \
    --src-dir datasets/panda_100 \
    --dst-dir datasets/panda_100_480p
