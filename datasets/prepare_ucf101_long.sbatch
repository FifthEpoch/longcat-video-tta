#!/bin/bash
#SBATCH --job-name=prep_ucf_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=datasets/slurm_log/prep_ucf_long_%j.out
#SBATCH --error=datasets/slurm_log/prep_ucf_long_%j.err

set -euo pipefail

cd /scratch/wc3013/longcat-video-tta
mkdir -p datasets/slurm_log

module purge
module load ffmpeg/6.0

echo "Preparing UCF-101 1000-video long-context dataset"
echo "  Min frames: 93 (for 93-frame generation)"
echo "  Start: $(date)"

python datasets/prepare_ucf101_long.py \
    --src-dir /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org \
    --dst-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_1000_long_480p \
    --num-videos 1000 \
    --min-frames 93 \
    --seed 42

echo ""
echo "Done: $(date)"
