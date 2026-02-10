#!/bin/bash
#SBATCH --job-name=dl_longcat
#SBATCH --output=baseline_experiment/sbatch/slurm_log/dl_model_%j.out
#SBATCH --error=baseline_experiment/sbatch/slurm_log/dl_model_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

SCRATCH="/scratch/wc3013"
MODEL_DIR="${MODEL_DIR:-${SCRATCH}/longcat-video-checkpoints}"

echo "=============================================================="
echo "Download LongCat-Video model weights"
echo "=============================================================="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Dest      : ${MODEL_DIR}"
echo "Start     : $(date)"
echo "=============================================================="

# Activate conda
source /share/apps/NYUAD/miniconda/3-4.11.0/etc/profile.d/conda.sh 2>/dev/null \
  || source "${SCRATCH}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || eval "$(conda shell.bash hook)" 2>/dev/null
conda activate longcat

# Make sure huggingface-cli is available
pip install --quiet "huggingface_hub[cli]"

export TMPDIR="${SCRATCH}/tmp"
export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "${TMPDIR}" "${HF_HOME}"

echo ""
echo "Downloading meituan-longcat/LongCat-Video → ${MODEL_DIR} ..."
echo ""

huggingface-cli download meituan-longcat/LongCat-Video \
    --local-dir "${MODEL_DIR}" \
    --local-dir-use-symlinks False

echo ""
echo "=============================================================="
echo "Download complete: $(date)"
echo "Contents:"
ls -lh "${MODEL_DIR}/"
echo ""
du -sh "${MODEL_DIR}"
echo "=============================================================="
