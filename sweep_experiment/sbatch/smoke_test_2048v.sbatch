#!/bin/bash
#SBATCH --job-name=smoke_2048v
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=sweep_experiment/slurm_log/smoke_2048v_%j.out
#SBATCH --error=sweep_experiment/slurm_log/smoke_2048v_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# ============================================================================
# Pre-flight smoke test for a freshly-built 2048-video dataset.
#
# Runs NoTTA (delta_steps=0) on the first SMOKE_N (default 20) videos of
# DATASET_DIR. Confirms the inference pipeline can load, decode, and
# produce per-video metrics for every video in the new dataset BEFORE
# the full 6-method x N-chunk batch is submitted.
#
# Submit:
#   sbatch --account=torch_pr_36_mren \
#          --export=ALL,DATASET_DIR=/scratch/wc3013/longcat-video-tta/datasets/panda_2048_480p,SMOKE_N=20 \
#          sweep_experiment/sbatch/smoke_test_2048v.sbatch
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"

DATASET_DIR="${DATASET_DIR:?ERROR: DATASET_DIR env var is required}"
SMOKE_N="${SMOKE_N:-20}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"

echo "=============================================================================="
echo "2048-video dataset smoke test"
echo "=============================================================================="
echo "  Dataset      : ${DATASET_DIR}"
echo "  N videos     : ${SMOKE_N}"
echo "  Frame geom   : cond=${NUM_COND_FRAMES} num=${NUM_FRAMES} gen_start=${GEN_START_FRAME}"
echo "  Checkpoint   : ${CHECKPOINT_DIR}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
if [ ! -d "${CONDA_ENV}" ]; then
    echo "ERROR: conda env not found at ${CONDA_ENV}" >&2
    exit 1
fi
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

"${PYTHON}" scripts/smoke_test_2048_dataset.py \
    --dataset-dir "${DATASET_DIR}" \
    --num-videos "${SMOKE_N}" \
    --num-cond-frames "${NUM_COND_FRAMES}" \
    --num-frames "${NUM_FRAMES}" \
    --gen-start-frame "${GEN_START_FRAME}" \
    --checkpoint-dir "${CHECKPOINT_DIR}"

echo ""
echo "=============================================================================="
echo "Smoke test exit code: $?"
echo "=============================================================================="
