#!/bin/bash
#SBATCH --job-name=ttc_lc
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --output=comparison_methods/slurm_log/ttc_longcat_%j.out
#SBATCH --error=comparison_methods/slurm_log/ttc_longcat_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "=============================================================================="
echo "Pathwise TTC (LongCat backbone) -- LONG-HORIZON comparison only"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
echo "Geometry: cond=${TTC_NUM_COND_FRAMES:-14} total=${TTC_NUM_FRAMES:-93} gen_start=${TTC_GEN_START_FRAME:-14}"
echo "sigma<=: ${TTC_SIGMA_THRESHOLD:-0.3}  cadence: ${TTC_CADENCE:-1}  weight: ${TTC_WEIGHT:-0.1}"
echo "Full-latent: ${TTC_FULL_LATENT:-0}  No-correction(baseline): ${TTC_NO_CORRECTION:-0}"
echo "=============================================================================="

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${TTC_DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
OUTPUT_DIR="${TTC_OUTPUT_DIR:-${PROJECT_ROOT}/comparison_methods/results/ttc_longctx_1000v}"
MAX_VIDEOS="${TTC_MAX_VIDEOS:-1000}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "$CONDA_ENV"
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
cd "${PROJECT_ROOT}"
mkdir -p comparison_methods/slurm_log

GT_CACHE_FLAG=""
if [ -n "${GT_FEATURES_CACHE:-}" ] && [ -f "${GT_FEATURES_CACHE}" ]; then
    GT_CACHE_FLAG="--gt-features-cache ${GT_FEATURES_CACHE}"
fi
SAVE_LIST_FLAG=""
if [ -n "${SAVE_ONLY_LIST:-}" ] && [ -f "${SAVE_ONLY_LIST}" ]; then
    SAVE_LIST_FLAG="--save-only-list ${SAVE_ONLY_LIST}"
fi
NO_SAVE_FLAG=""
if [ "${TTC_NO_SAVE_VIDEOS:-0}" = "1" ]; then
    NO_SAVE_FLAG="--no-save-videos"
fi
FULL_LATENT_FLAG=""
if [ "${TTC_FULL_LATENT:-0}" = "1" ]; then
    FULL_LATENT_FLAG="--ttc-full-latent"
fi
NO_CORR_FLAG=""
if [ "${TTC_NO_CORRECTION:-0}" = "1" ]; then
    NO_CORR_FLAG="--no-correction"
fi
EVAL_FLAGS="--min-fvd-videos ${MIN_FVD_VIDEOS:-256}"
[ "${COMPUTE_FVD:-1}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-fvd"
[ "${COMPUTE_FID:-1}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-fid"
[ "${COMPUTE_VBENCH:-1}" = "1" ] && EVAL_FLAGS="${EVAL_FLAGS} --compute-vbench"

"$PYTHON" comparison_methods/scripts/ttc_longcat.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${MAX_VIDEOS}" \
    --num-cond-frames "${TTC_NUM_COND_FRAMES:-14}" \
    --num-frames "${TTC_NUM_FRAMES:-93}" \
    --gen-start-frame "${TTC_GEN_START_FRAME:-14}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS:-50}" \
    --guidance-scale "${GUIDANCE_SCALE:-4.0}" \
    --ttc-sigma-threshold "${TTC_SIGMA_THRESHOLD:-0.3}" \
    --ttc-cadence "${TTC_CADENCE:-1}" \
    --ttc-weight "${TTC_WEIGHT:-0.1}" \
    --seed "${SEED:-42}" \
    ${FULL_LATENT_FLAG} \
    ${NO_CORR_FLAG} \
    ${NO_SAVE_FLAG} \
    ${EVAL_FLAGS} \
    ${GT_CACHE_FLAG} \
    ${SAVE_LIST_FLAG}

echo ""
echo "=============================================================================="
echo "TTC LongCat Complete  End: $(date)"
echo "=============================================================================="
