#!/bin/bash
#SBATCH --job-name=tta_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=sweep_experiment/slurm_log/eval_%x_%j.out
#SBATCH --error=sweep_experiment/slurm_log/eval_%x_%j.err

# ============================================================================
# Evaluation dispatcher: FVD, VBench++, or find_best_configs
#
#   EVAL_MODE = fvd | vbench | best_configs
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"

EVAL_MODE="${EVAL_MODE:?ERROR: EVAL_MODE env var required (fvd|vbench|best_configs)}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "$PROJECT_ROOT"

case "${EVAL_MODE}" in

  fvd)
    GEN_DIR="${GEN_DIR:?ERROR: GEN_DIR required for FVD}"
    REF_DIR="${REF_DIR:?ERROR: REF_DIR required for FVD}"
    OUTPUT="${OUTPUT:-${GEN_DIR}/fvd_score.json}"
    NUM_FRAMES="${NUM_FRAMES:-16}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    MIN_VIDEOS="${MIN_VIDEOS:-256}"
    FORCE_FVD="${FORCE_FVD:-0}"
    COMPUTE_FID="${COMPUTE_FID:-0}"
    SELF_CHECK="${SELF_CHECK:-0}"

    EXTRA_FLAGS=""
    [ "${FORCE_FVD}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --force"
    [ "${COMPUTE_FID}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --compute-fid"
    [ "${SELF_CHECK}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --self-check"

    echo "Computing FVD: ${GEN_DIR} vs ${REF_DIR}"
    echo "  num_frames=${NUM_FRAMES} batch_size=${BATCH_SIZE} min_videos=${MIN_VIDEOS}"
    echo "  force=${FORCE_FVD} compute_fid=${COMPUTE_FID} self_check=${SELF_CHECK}"
    python sweep_experiment/scripts/eval_fvd.py \
        --gen-dir "${GEN_DIR}" \
        --ref-dir "${REF_DIR}" \
        --num-frames "${NUM_FRAMES}" \
        --batch-size "${BATCH_SIZE}" \
        --min-videos "${MIN_VIDEOS}" \
        --output "${OUTPUT}" \
        ${EXTRA_FLAGS}
    ;;

  vbench)
    VIDEO_DIR="${VIDEO_DIR:?ERROR: VIDEO_DIR required for VBench++}"
    OUTPUT="${OUTPUT:-${VIDEO_DIR}/vbench_scores.json}"

    echo "Running VBench++ on: ${VIDEO_DIR}"
    python sweep_experiment/scripts/eval_vbench.py \
        --video-dir "${VIDEO_DIR}" \
        --output "${OUTPUT}"
    ;;

  best_configs)
    RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/sweep_experiment/results}"
    OUTPUT="${OUTPUT:-${RESULTS_ROOT}/best_configs.json}"

    echo "Finding best configs across: ${RESULTS_ROOT}"
    python sweep_experiment/scripts/find_best_configs.py \
        --results-dir "${RESULTS_ROOT}" \
        --output "${OUTPUT}"
    ;;

  *)
    echo "ERROR: Unknown EVAL_MODE '${EVAL_MODE}'. Use: fvd|vbench|best_configs" >&2
    exit 1
    ;;
esac

echo ""
echo "Evaluation complete: ${EVAL_MODE}"
echo "End time: $(date)"
