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

    echo "Computing FVD: ${GEN_DIR} vs ${REF_DIR}"
    python sweep_experiment/scripts/eval_fvd.py \
        --gen-dir "${GEN_DIR}" \
        --ref-dir "${REF_DIR}" \
        --output "${OUTPUT}"
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
