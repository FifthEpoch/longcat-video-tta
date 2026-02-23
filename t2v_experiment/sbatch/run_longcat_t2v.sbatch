#!/bin/bash
#SBATCH --job-name=longcat_t2v
#SBATCH --output=t2v_experiment/sbatch/slurm_log/longcat_t2v_%j.out
#SBATCH --error=t2v_experiment/sbatch/slurm_log/longcat_t2v_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1

set -euo pipefail

SCRATCH="/scratch/wc3013"
REPO="${SCRATCH}/longcat-video-tta"
MODEL_DIR="${MODEL_DIR:-${SCRATCH}/longcat-video-checkpoints}"
META_CSV="${META_CSV:-${REPO}/datasets/panda_100/metadata.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/t2v_experiment/results/longcat}"
NUM_FRAMES="${NUM_FRAMES:-29}"
MAX_VIDEOS="${MAX_VIDEOS:-100}"

echo "=============================================================="
echo "LongCat-Video T2V Baseline"
echo "=============================================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model      : ${MODEL_DIR}"
echo "Prompts    : ${META_CSV}"
echo "Output     : ${OUTPUT_DIR}"
echo "Frames     : ${NUM_FRAMES}"
echo "Start      : $(date)"
echo "=============================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH}/conda-envs/longcat"

export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"
mkdir -p t2v_experiment/sbatch/slurm_log

cd "${REPO}"

python t2v_experiment/scripts/run_longcat_t2v.py \
    --checkpoint-dir "${MODEL_DIR}" \
    --meta-csv "${META_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-frames "${NUM_FRAMES}" \
    --num-inference-steps 50 \
    --guidance-scale 4.0 \
    --height 480 \
    --width 832 \
    --seed 42 \
    --max-videos "${MAX_VIDEOS}" \
    --fps 15

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "Results:  ${OUTPUT_DIR}"
echo "=============================================================="
