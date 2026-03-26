#!/bin/bash
#SBATCH --job-name=opensora_t2v
#SBATCH --output=t2v_experiment/sbatch/slurm_log/opensora_t2v_%j.out
#SBATCH --error=t2v_experiment/sbatch/slurm_log/opensora_t2v_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1

set -euo pipefail

SCRATCH="/scratch/wc3013"
LONGCAT_REPO="${SCRATCH}/longcat-video-tta"
OPENSORA_REPO="${SCRATCH}/open-sora-v2.0-experiment"
PROMPTS_CSV="${PROMPTS_CSV:-${LONGCAT_REPO}/t2v_experiment/opensora_prompts.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${LONGCAT_REPO}/t2v_experiment/results/opensora}"
NUM_FRAMES="${NUM_FRAMES:-29}"

echo "=============================================================="
echo "Open-Sora v2.0 T2V Baseline"
echo "=============================================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Prompts    : ${PROMPTS_CSV}"
echo "Output     : ${OUTPUT_DIR}"
echo "Frames     : ${NUM_FRAMES}"
echo "Start      : $(date)"
echo "=============================================================="

# ── Environment ───────────────────────────────────────────────────────
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH}/conda-envs/opensora20"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"
mkdir -p t2v_experiment/sbatch/slurm_log

cd "${OPENSORA_REPO}"

# Ensure opensora package is importable
pip install -e . --no-deps -q 2>/dev/null || true
export PYTHONPATH="${OPENSORA_REPO}:${PYTHONPATH:-}"

# ── Step 1: Prepare prompts CSV (if not already created) ──────────
if [ ! -f "${PROMPTS_CSV}" ]; then
    echo "Preparing prompts CSV..."
    cd "${LONGCAT_REPO}"
    python t2v_experiment/scripts/prepare_opensora_prompts.py \
        --meta-csv datasets/panda_100/metadata.csv \
        --output "${PROMPTS_CSV}" \
        --max-videos 100
    cd "${OPENSORA_REPO}"
fi
echo "Using prompts: ${PROMPTS_CSV}"
echo ""

# ── Step 2: Run Open-Sora T2V inference ──────────────────────────
mkdir -p "${OUTPUT_DIR}"

MASTER_PORT=$((29500 + RANDOM % 1000))

torchrun --nproc_per_node 1 --standalone --master_port ${MASTER_PORT} \
    scripts/diffusion/inference.py \
    configs/diffusion/inference/256px.py \
    --save-dir "${OUTPUT_DIR}" \
    --dataset.data_path "${PROMPTS_CSV}" \
    --sampling_option.num_frames "${NUM_FRAMES}" \
    --sampling_option.num_steps 50 \
    --cond_type t2v \
    --fps_save 15 \
    --seed 42

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "Results:  ${OUTPUT_DIR}"
echo "=============================================================="
