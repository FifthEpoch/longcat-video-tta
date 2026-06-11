#!/bin/bash
#SBATCH --job-name=compute_tier3_probes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/logs/compute_tier3_probes_%j.out
#SBATCH --error=sweep_experiment/logs/compute_tier3_probes_%j.err

# ============================================================================
# Per-video Tier-3 mini-TTA probes against the LongCat-Video base model.
#
# Runs scripts/compute_tier3_probes.py over a directory of real videos and
# emits a per-video CSV with H-T3-1 grad_norm_θ0 (LoRA-param gradient L2
# norm of the visible-window flow-matching loss at the unadapted weights)
# and H-T3-2 single_step_loss_drop (fractional drop after ONE Adam step on
# the LoRA r=8 adapter at the headline LORA_R8_TTA learning rate) per the
# gating-experiment plan (sweep_experiment/reports/PLAN_gating_experiment_
# 2026-06-11.md §2.4 / §3.1).
#
# Resource profile mirrors run_compute_diffusion_ood.sbatch (same base
# model load, ~6 forward passes per video + 3 backwards + 3 Adam steps —
# about 2-3x the OOD scorer's per-video wallclock, hence the same 4 h cap):
# h200 GPU, 192 G memory, 8 CPU.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

trap 'echo "ERROR on line ${LINENO}: command \"${BASH_COMMAND}\" failed with exit code $?" >&2' ERR

# ============================================================================
# Path setup
# ============================================================================
SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"

# ============================================================================
# Inputs / outputs (override via --export=ALL,VAR=value when submitting)
# ============================================================================
VIDEOS_DIR="${VIDEOS_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
CAPTIONS_CSV="${CAPTIONS_CSV:-${VIDEOS_DIR}/metadata.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features.csv}"
TTA_VISIBLE_FRAMES="${TTA_VISIBLE_FRAMES:-auto}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
TIMESTEPS="${TIMESTEPS:-100,500,900}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
RESUME="${RESUME:-0}"

# ============================================================================
# LoRA probe recipe (defaults mirror headline LORA_R8_TTA from
# sweep_experiment/sbatch/submit_standard_1000v_chunked.sh §2)
# ============================================================================
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_LR="${LORA_LR:-5.0e-5}"
LORA_WEIGHT_DECAY="${LORA_WEIGHT_DECAY:-0.01}"
LORA_TARGETS="${LORA_TARGETS:-qkv,proj}"
LORA_TARGET_BLOCKS="${LORA_TARGET_BLOCKS:-all}"
LORA_TARGET_FFN="${LORA_TARGET_FFN:-0}"

# ============================================================================
# Banner
# ============================================================================
echo "=============================================================================="
echo "Per-video Tier-3 mini-TTA probes (LoRA-r8 grad-norm + single-step loss drop)"
echo "=============================================================================="
echo "Job ID         : ${SLURM_JOB_ID:-<unset>}"
echo "Node           : $(hostname)"
echo "Start time     : $(date)"
echo "=============================================================================="
echo "Resolved configuration:"
echo "  USER               : ${USER}"
echo "  SCRATCH_BASE       : ${SCRATCH_BASE}"
echo "  PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "  CONDA_ENV          : ${CONDA_ENV}"
echo "  CHECKPOINT_DIR     : ${CHECKPOINT_DIR}"
echo "  VIDEOS_DIR         : ${VIDEOS_DIR}"
echo "  CAPTIONS_CSV       : ${CAPTIONS_CSV}"
echo "  OUTPUT_CSV         : ${OUTPUT_CSV}"
echo "  TTA_VISIBLE_FRAMES : ${TTA_VISIBLE_FRAMES}"
echo "  TTA_CONTEXT_FRAMES : ${TTA_CONTEXT_FRAMES}"
echo "  TIMESTEPS          : ${TIMESTEPS}"
echo "  SEED               : ${SEED}"
echo "  DEVICE             : ${DEVICE}"
echo "  MAX_VIDEOS         : ${MAX_VIDEOS}  (0 = all)"
echo "  RESUME             : ${RESUME}"
echo "  LORA_RANK          : ${LORA_RANK}"
echo "  LORA_ALPHA         : ${LORA_ALPHA}"
echo "  LORA_LR            : ${LORA_LR}"
echo "  LORA_WEIGHT_DECAY  : ${LORA_WEIGHT_DECAY}"
echo "  LORA_TARGETS       : ${LORA_TARGETS}"
echo "  LORA_TARGET_BLOCKS : ${LORA_TARGET_BLOCKS}"
echo "  LORA_TARGET_FFN    : ${LORA_TARGET_FFN}"
echo "=============================================================================="

# ============================================================================
# Environment setup
#
# CRITICAL: module load anaconda3/... sets PYTHONHOME / PYTHONPATH to the
# system anaconda, which shadows the conda env's sys.path and reproduces
# the "No module named 'torch'" symptom. We MUST unset both AFTER
# `conda activate`. See sweep_experiment/docs/SBATCH_CONDA_BEST_PRACTICES.md.
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

if [ -d "${CONDA_ENV}" ]; then
    conda activate "${CONDA_ENV}"
    echo "Activated conda: ${CONDA_ENV}"
else
    echo "ERROR: Conda environment not found at ${CONDA_ENV}" >&2
    exit 1
fi

unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found at ${PYTHON}" >&2
    exit 1
fi
echo "Using: ${PYTHON} ($("${PYTHON}" --version 2>&1))"

"${PYTHON}" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed — conda env is broken or PYTHONHOME was not unset" >&2
    exit 1
}

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT_CSV}")"
mkdir -p "sweep_experiment/logs"

# ============================================================================
# Sanity-check inputs
# ============================================================================
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: CHECKPOINT_DIR not found: ${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [ ! -d "${VIDEOS_DIR}" ]; then
    echo "ERROR: VIDEOS_DIR not found: ${VIDEOS_DIR}" >&2
    exit 1
fi
if [ ! -f "${CAPTIONS_CSV}" ]; then
    echo "WARN: CAPTIONS_CSV not found: ${CAPTIONS_CSV} — rows will use empty captions" >&2
fi

# ============================================================================
# GPU info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Build extra flags
# ============================================================================
EXTRA_FLAGS=""
if [ "${RESUME}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --resume"
fi
if [ "${MAX_VIDEOS}" != "0" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --max-videos ${MAX_VIDEOS}"
fi
if [ "${LORA_TARGET_FFN}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --lora-target-ffn"
fi

# ============================================================================
# Run Tier-3 probe
# ============================================================================
echo "Starting Tier-3 probe computation..."

"${PYTHON}" scripts/compute_tier3_probes.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --videos-dir "${VIDEOS_DIR}" \
    --captions-csv "${CAPTIONS_CSV}" \
    --output "${OUTPUT_CSV}" \
    --tta-visible-frames "${TTA_VISIBLE_FRAMES}" \
    --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
    --timesteps "${TIMESTEPS}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-lr "${LORA_LR}" \
    --lora-weight-decay "${LORA_WEIGHT_DECAY}" \
    --lora-targets "${LORA_TARGETS}" \
    --lora-target-blocks "${LORA_TARGET_BLOCKS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    ${EXTRA_FLAGS}

echo ""
echo "=============================================================================="
echo "Tier-3 probe computation complete"
echo "=============================================================================="
echo "End time   : $(date)"
echo "Output CSV : ${OUTPUT_CSV}"
echo "=============================================================================="
