#!/bin/bash
#SBATCH --job-name=savi_dno_lc
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=comparison_methods/slurm_log/savi_dno_longcat_%j.out
#SBATCH --error=comparison_methods/slurm_log/savi_dno_longcat_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "=============================================================================="
echo "SAVi-DNO Evaluation (Noise Optimization on LongCat Backbone)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Euler steps: ${SAVI_EULER_STEPS:-10}"
echo "Num GPUs: ${NUM_GPUS:-2}"
echo "LR: ${SAVI_LR:-1e-4}"
echo "Lambda: ${SAVI_LAM:-0.0012}"
echo "p: ${SAVI_P:-0.7}"
echo "Guidance: ${SAVI_GUIDANCE:-4.0}"
echo "Loss mode: ${SAVI_LOSS_MODE:-latent} (default: latent loss for LongCat)"
echo "Rollout steps: ${SAVI_ROLLOUT_STEPS:-10}"
echo "Max grad norm: ${SAVI_MAX_GRAD_NORM:-1.0}"
echo "No-optimize: ${SAVI_NO_OPTIMIZE:-}"
echo "Oracle-leak: ${SAVI_ORACLE_LEAK:-0} (1=leaky oracle upper bound; default 0=fair leakage-free)"
echo "Method: ${NOISE_OPT_METHOD:-savi_dno} (savi_dno=default; dno=Karunratanakul CVPR24; direct_noise_opt=Tang ICML25)"
echo "Reg weight: ${NOISE_OPT_REG_WEIGHT:--1} (<0 = method default)"
echo "Geometry: cond=${SAVI_NUM_COND_FRAMES:-14} total=${SAVI_NUM_FRAMES:-28} gen_start=${SAVI_GEN_START_FRAME:-48}"
echo "=============================================================================="

NOISE_OPT_METHOD="${NOISE_OPT_METHOD:-savi_dno}"

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${SAVI_LC_DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
# Default output dir is keyed on Euler steps AND the variant (optimized / no-opt
# / oracle) so distinct runs can NEVER silently overwrite each other's
# summary.json when SAVI_LC_OUTPUT_DIR is not set explicitly.
_SAVI_VARIANT=""
if [ "${SAVI_NO_OPTIMIZE:-}" = "1" ]; then
    _SAVI_VARIANT="_noopt"
elif [ "${SAVI_ORACLE_LEAK:-}" = "1" ]; then
    _SAVI_VARIANT="_oracle"
fi
# Output dir is also keyed on the noise-opt method so dno / direct_noise_opt
# never collide with savi_dno. Default (savi_dno) keeps the original path.
case "${NOISE_OPT_METHOD}" in
    dno)              _METHOD_BASE="dno_longcat" ;;
    direct_noise_opt) _METHOD_BASE="direct_noise_opt_longcat" ;;
    *)                _METHOD_BASE="savi_dno_longcat" ;;
esac
OUTPUT_DIR="${SAVI_LC_OUTPUT_DIR:-${PROJECT_ROOT}/comparison_methods/results/${_METHOD_BASE}_s${SAVI_EULER_STEPS:-10}${_SAVI_VARIANT}}"
MAX_VIDEOS="${SAVI_LC_MAX_VIDEOS:-1000}"

# ============================================================================
# Environment Setup (LongCat conda env)
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi

unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed" >&2
    exit 1
}

cd "${PROJECT_ROOT}"

# ============================================================================
# Build flags
# ============================================================================
GT_CACHE_FLAG=""
if [ -n "${GT_FEATURES_CACHE:-}" ] && [ -f "${GT_FEATURES_CACHE}" ]; then
    GT_CACHE_FLAG="--gt-features-cache ${GT_FEATURES_CACHE}"
fi
SAVE_LIST_FLAG=""
if [ -n "${SAVE_ONLY_LIST:-}" ] && [ -f "${SAVE_ONLY_LIST}" ]; then
    SAVE_LIST_FLAG="--save-only-list ${SAVE_ONLY_LIST}"
fi
NO_OPT_FLAG=""
if [ -n "${SAVI_NO_OPTIMIZE:-}" ] && [ "${SAVI_NO_OPTIMIZE}" = "1" ]; then
    NO_OPT_FLAG="--no-optimize"
fi
LOSS_FLAG=""
if [ "${SAVI_LOSS_MODE:-latent}" = "pixel" ]; then
    LOSS_FLAG="--pixel-loss"
fi
# Fair protocol is the DEFAULT (adapt noise on observed history, predict unseen
# future). Set SAVI_ORACLE_LEAK=1 only to produce a labelled oracle upper bound
# (optimizes noise against the scored future = ground-truth leakage).
ORACLE_FLAG=""
if [ -n "${SAVI_ORACLE_LEAK:-}" ] && [ "${SAVI_ORACLE_LEAK}" = "1" ]; then
    ORACLE_FLAG="--oracle-leak"
fi
# Save every generated mp4 by default (bank videos for the figure bank / future
# experiments). Set SAVI_NO_SAVE_VIDEOS=1 for a metrics-only run.
SAVE_FLAG="--save-videos"
if [ "${SAVI_NO_SAVE_VIDEOS:-0}" = "1" ]; then
    SAVE_FLAG=""
fi
# VBench++ (7 dims) on the saved mp4s so SAVi-DNO matches AdaSteer's metric set.
# On by default; needs saved videos. Set SAVI_COMPUTE_VBENCH=0 to skip.
VBENCH_FLAG="--compute-vbench"
if [ "${SAVI_COMPUTE_VBENCH:-1}" != "1" ] || [ -z "${SAVE_FLAG}${SAVE_LIST_FLAG}" ]; then
    VBENCH_FLAG=""
fi
echo "Save videos  : ${SAVE_FLAG:-NO (metrics only)}"
echo "VBench       : ${VBENCH_FLAG:-OFF}"

# ============================================================================
# Run SAVi-DNO with LongCat backbone
# ============================================================================
"$PYTHON" comparison_methods/scripts/savi_dno_longcat.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${MAX_VIDEOS}" \
    --num-inference-steps "${SAVI_EULER_STEPS:-10}" \
    --num-gpus "${NUM_GPUS:-2}" \
    --guidance-scale "${SAVI_GUIDANCE:-4.0}" \
    --lr "${SAVI_LR:-1e-4}" \
    --lam "${SAVI_LAM:-0.0012}" \
    --p "${SAVI_P:-0.7}" \
    --rollout-steps "${SAVI_ROLLOUT_STEPS:-10}" \
    --max-grad-norm "${SAVI_MAX_GRAD_NORM:-1.0}" \
    --num-cond-frames "${SAVI_NUM_COND_FRAMES:-14}" \
    --num-frames "${SAVI_NUM_FRAMES:-28}" \
    --gen-start-frame "${SAVI_GEN_START_FRAME:-48}" \
    --method "${NOISE_OPT_METHOD}" \
    --reg-weight "${NOISE_OPT_REG_WEIGHT:--1}" \
    --seed 42 \
    ${NO_OPT_FLAG} \
    ${LOSS_FLAG} \
    ${ORACLE_FLAG} \
    ${SAVE_FLAG} \
    ${VBENCH_FLAG} \
    ${GT_CACHE_FLAG} \
    ${SAVE_LIST_FLAG}

echo ""
echo "=============================================================================="
echo "SAVi-DNO LongCat Complete"
echo "End time: $(date)"
echo "=============================================================================="
