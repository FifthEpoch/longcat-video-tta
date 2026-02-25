#!/bin/bash
#SBATCH --job-name=resize_vids
#SBATCH --output=datasets/slurm_log/resize_%j.out
#SBATCH --error=datasets/slurm_log/resize_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

SCRATCH="/scratch/wc3013"
REPO="${SCRATCH}/longcat-video-tta"

SRC_DIR="${SRC_DIR:-datasets/panda_100}"
DST_DIR="${DST_DIR:-datasets/panda_100_480p}"

echo "=============================================================="
echo "Resize videos to 832x480"
echo "=============================================================="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Source : ${SRC_DIR}"
echo "Dest   : ${DST_DIR}"
echo "Start  : $(date)"
echo "=============================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH}/conda-envs/longcat"

echo "Python : $(which python)"
echo "ffmpeg : $(which ffmpeg 2>/dev/null || echo 'NOT FOUND')"

if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg via conda..."
    conda install -c conda-forge ffmpeg -y --quiet
fi

cd "${REPO}"
mkdir -p datasets/slurm_log

python datasets/resize_videos.py \
    --src-dir "${SRC_DIR}" \
    --dst-dir "${DST_DIR}"

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
