#!/bin/bash
#SBATCH --job-name=longcat_base
#SBATCH --output=baseline_experiment/sbatch/slurm_log/baseline_%j.out
#SBATCH --error=baseline_experiment/sbatch/slurm_log/baseline_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1

set -euo pipefail

SCRATCH="/scratch/wc3013"
REPO="${SCRATCH}/longcat-video-tta"
MODEL_DIR="${MODEL_DIR:-${SCRATCH}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${REPO}/datasets/panda_100_480p}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/baseline_experiment/results/baseline_480p}"

echo "=============================================================="
echo "LongCat-Video Baseline Inference"
echo "=============================================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model      : ${MODEL_DIR}"
echo "Data       : ${DATA_DIR}"
echo "Output     : ${OUTPUT_DIR}"
echo "Start      : $(date)"
echo "=============================================================="

# Activate conda (same method as working env setup script)
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH}/conda-envs/longcat"

export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

# Verify model exists
if [ ! -d "${MODEL_DIR}/dit" ]; then
    echo "ERROR: Model not found at ${MODEL_DIR}. Run download_model.sbatch first."
    exit 1
fi

echo ""
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
echo ""

cd "${REPO}"

mkdir -p baseline_experiment/sbatch/slurm_log
mkdir -p "${OUTPUT_DIR}"

python baseline_experiment/scripts/run_baseline.py \
    --checkpoint-dir "${MODEL_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-cond-frames 2 \
    --num-gen-frames 14 \
    --resolution 480p \
    --num-inference-steps 50 \
    --guidance-scale 4.0 \
    --seed 42 \
    --max-videos 100 \
    --save-videos

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "Results:  ${OUTPUT_DIR}"
echo "=============================================================="
