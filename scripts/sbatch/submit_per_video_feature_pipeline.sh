#!/bin/bash
# ============================================================================
# Submit the per-video feature-extraction -> correlation pipeline.
#
# Stage 1 (GPU): scripts/extract_video_features_for_tta.py
#                via scripts/sbatch/run_extract_video_features.sbatch
# Stage 2 (CPU): scripts/correlate_tta_gain_with_features.py
#                via scripts/sbatch/run_correlate_tta_gain.sbatch
#                (scheduled with --dependency=afterok on stage 1)
#
# Login-node compute is not allowed on the cluster, so this wrapper merely
# chains two sbatch submissions; the heavy lifting runs inside Slurm.
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/$USER/longcat-video-tta
#   bash scripts/sbatch/submit_per_video_feature_pipeline.sh
#
# Override any input/output path without editing this script, e.g.:
#   VIDEOS_DIR=/scratch/$USER/longcat-video-tta/datasets/panda_2048_480p \
#   OUTPUT_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features_2048.csv \
#   FEATURES_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features_2048.csv \
#       bash scripts/sbatch/submit_per_video_feature_pipeline.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"

EXTRACT_SBATCH="${EXTRACT_SBATCH:-scripts/sbatch/run_extract_video_features.sbatch}"
CORR_SBATCH="${CORR_SBATCH:-scripts/sbatch/run_correlate_tta_gain.sbatch}"

# ============================================================================
# Stage 1 (extraction) defaults — passed through to run_extract_video_features.sbatch
# ============================================================================
VIDEOS_DIR="${VIDEOS_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
CAPTIONS_CSV="${CAPTIONS_CSV:-${VIDEOS_DIR}/metadata.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv}"
TTA_VISIBLE_FRAMES="${TTA_VISIBLE_FRAMES:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"

# ============================================================================
# Stage 2 (correlation) defaults — passed through to run_correlate_tta_gain.sbatch
# Default FEATURES_CSV to match stage 1's OUTPUT_CSV so the chain is consistent
# when neither is overridden.
# ============================================================================
GAINS_CSV="${GAINS_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv}"
FEATURES_CSV="${FEATURES_CSV:-${OUTPUT_CSV}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation}"

# ============================================================================
# Banner
# ============================================================================
echo "============================================================"
echo "Per-video feature extraction -> correlation pipeline"
echo "============================================================"
echo "  PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "  EXTRACT_SBATCH     : ${EXTRACT_SBATCH}"
echo "  CORR_SBATCH        : ${CORR_SBATCH}"
echo ""
echo "Stage 1 (extraction, GPU):"
echo "  VIDEOS_DIR         : ${VIDEOS_DIR}"
echo "  CAPTIONS_CSV       : ${CAPTIONS_CSV}"
echo "  OUTPUT_CSV         : ${OUTPUT_CSV}"
echo "  TTA_VISIBLE_FRAMES : ${TTA_VISIBLE_FRAMES}"
echo "  BATCH_SIZE         : ${BATCH_SIZE}"
echo "  DEVICE             : ${DEVICE}"
echo ""
echo "Stage 2 (correlation, CPU, depends on stage 1):"
echo "  GAINS_CSV          : ${GAINS_CSV}"
echo "  FEATURES_CSV       : ${FEATURES_CSV}"
echo "  OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "============================================================"

# ============================================================================
# Pre-flight sanity checks (cheap, login-node-safe)
# ============================================================================
if [ ! -f "${EXTRACT_SBATCH}" ]; then
    echo "ERROR: extraction sbatch not found: ${EXTRACT_SBATCH}" >&2
    echo "  Are you running this from ${PROJECT_ROOT}?" >&2
    exit 1
fi
if [ ! -f "${CORR_SBATCH}" ]; then
    echo "ERROR: correlation sbatch not found: ${CORR_SBATCH}" >&2
    exit 1
fi
if [ ! -d "${VIDEOS_DIR}" ]; then
    echo "WARN: VIDEOS_DIR does not exist on the login node: ${VIDEOS_DIR}"
    echo "      (this may still be fine if the compute node sees /scratch differently,"
    echo "      but usually indicates a typo or wrong dataset path)"
fi
if [ ! -f "${GAINS_CSV}" ]; then
    echo "WARN: GAINS_CSV does not exist yet: ${GAINS_CSV}"
    echo "      Stage 2 will fail unless this file appears before stage 1 finishes."
fi

# ============================================================================
# Stage 1: feature extraction
# ============================================================================
EXTRACT_JID=$(sbatch --parsable \
    --export="ALL,VIDEOS_DIR=${VIDEOS_DIR},CAPTIONS_CSV=${CAPTIONS_CSV},OUTPUT_CSV=${OUTPUT_CSV},TTA_VISIBLE_FRAMES=${TTA_VISIBLE_FRAMES},BATCH_SIZE=${BATCH_SIZE},DEVICE=${DEVICE}" \
    "${EXTRACT_SBATCH}")
echo "Submitted feature extraction: ${EXTRACT_JID}"

# ============================================================================
# Stage 2: correlation (afterok on stage 1)
# ============================================================================
CORR_JID=$(sbatch --parsable \
    --dependency="afterok:${EXTRACT_JID}" \
    --export="ALL,GAINS_CSV=${GAINS_CSV},FEATURES_CSV=${FEATURES_CSV},OUTPUT_DIR=${OUTPUT_DIR}" \
    "${CORR_SBATCH}")
echo "Submitted correlation (depends on ${EXTRACT_JID}): ${CORR_JID}"

echo ""
echo "============================================================"
echo "Submitted 2 jobs."
echo ""
echo "Monitor:"
echo "  squeue -u \$USER -j ${EXTRACT_JID},${CORR_JID}"
echo "  squeue -u \$USER | grep -E 'extract_video_features|correlate_tta_gain'"
echo ""
echo "Logs:"
echo "  ${PROJECT_ROOT}/sweep_experiment/logs/extract_video_features_${EXTRACT_JID}.{out,err}"
echo "  ${PROJECT_ROOT}/sweep_experiment/logs/correlate_tta_gain_${CORR_JID}.{out,err}"
echo "============================================================"
