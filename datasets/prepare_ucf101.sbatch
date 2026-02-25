#!/bin/bash
#SBATCH --job-name=prep_ucf101
#SBATCH --output=datasets/slurm_log/prep_ucf101_%j.out
#SBATCH --error=datasets/slurm_log/prep_ucf101_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

SCRATCH="/scratch/wc3013"
REPO="${SCRATCH}/longcat-video-tta"
UCF101_SRC="${SCRATCH}/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org"

NUM_VIDEOS="${NUM_VIDEOS:-1000}"
VIDEOS_PER_CATEGORY="${VIDEOS_PER_CATEGORY:-10}"
DST_DIR="${DST_DIR:-${REPO}/datasets/ucf101_${NUM_VIDEOS}_480p}"

echo "=============================================================="
echo "Prepare UCF101 ${NUM_VIDEOS}-video subset (832x480)"
echo "=============================================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Source     : ${UCF101_SRC}"
echo "Dest       : ${DST_DIR}"
echo "Num videos : ${NUM_VIDEOS}"
echo "Per category: ${VIDEOS_PER_CATEGORY}"
echo "Start      : $(date)"
echo "=============================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH}/conda-envs/longcat"

echo "Python : $(which python)"
echo "ffmpeg : $(which ffmpeg)"
ffmpeg -version | head -1

cd "${REPO}"
mkdir -p datasets/slurm_log

python datasets/prepare_ucf101_subset.py \
    --src-dir "${UCF101_SRC}" \
    --dst-dir "${DST_DIR}" \
    --num-videos "${NUM_VIDEOS}" \
    --videos-per-category "${VIDEOS_PER_CATEGORY}" \
    --seed 42

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
