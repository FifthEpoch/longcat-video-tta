#!/bin/bash
# ============================================================================
# Submit diffusion-OOD scoring on datasets/panda_segment_pool with --resume.
#
# Bakes in VIDEOS_DIR / OUTPUT_CSV so a fresh login shell cannot silently
# fall back to panda_1000_480p (see job 13281092 failure 2026-07-10).
#
# Usage (cluster):
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash scripts/sbatch/submit_segment_pool_ood.sh
#
# Override walltime (sbatch flag overrides #SBATCH in the .sbatch file):
#   SBATCH_TIME=24:00:00 bash scripts/sbatch/submit_segment_pool_ood.sh
#
# Smoke test:
#   MAX_VIDEOS=10 RESUME=0 bash scripts/sbatch/submit_segment_pool_ood.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
OOD_SBATCH="${OOD_SBATCH:-scripts/sbatch/run_compute_diffusion_ood.sbatch}"
SBATCH_TIME="${SBATCH_TIME:-}"

export VIDEOS_DIR="${VIDEOS_DIR:-${PROJECT_ROOT}/datasets/panda_segment_pool}"
export CAPTIONS_CSV="${CAPTIONS_CSV:-${VIDEOS_DIR}/metadata.csv}"
export OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-07-10/diffusion_ood_scores_segment_pool.csv}"
export MAX_VIDEOS="${MAX_VIDEOS:-0}"
export RESUME="${RESUME:-1}"

cd "${PROJECT_ROOT}"

TIME_ARGS=()
if [ -n "${SBATCH_TIME}" ]; then
    TIME_ARGS=(--time="${SBATCH_TIME}")
fi

echo "Submitting segment-pool OOD scoring:"
echo "  VIDEOS_DIR  = ${VIDEOS_DIR}"
echo "  OUTPUT_CSV  = ${OUTPUT_CSV}"
echo "  MAX_VIDEOS  = ${MAX_VIDEOS}"
echo "  RESUME      = ${RESUME}"
if [ -n "${SBATCH_TIME}" ]; then
    echo "  SBATCH_TIME = ${SBATCH_TIME}"
fi

sbatch "${TIME_ARGS[@]}" --export=ALL "${OOD_SBATCH}"
