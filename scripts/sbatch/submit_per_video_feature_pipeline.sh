#!/bin/bash
# ============================================================================
# Submit the per-video feature-extraction + diffusion-OOD → correlation
# pipeline.
#
# Stage 1a (GPU): scripts/extract_video_features_for_tta.py
#                 via scripts/sbatch/run_extract_video_features.sbatch
#                 (CLIP / DINO / cuts / texture / colour Tier-1 features)
# Stage 1b (GPU): scripts/compute_diffusion_ood_score.py
#                 via scripts/sbatch/run_compute_diffusion_ood.sbatch
#                 (per-video flow-matching MSE against LongCat-Video base;
#                  the OOD-score proxy for TTA gain)
# Stage 2  (CPU): scripts/correlate_tta_gain_with_features.py
#                 via scripts/sbatch/run_correlate_tta_gain.sbatch
#                 (scheduled with --dependency=afterok:1a:1b so the
#                  combined ρ(ΔPSNR, X) report covers BOTH feature CSVs)
#
# Stages 1a and 1b are independent of each other (both depend only on the
# dataset) so they fan out in parallel. Stage 2 chains behind both via
# Slurm's `--dependency=afterok:<JID1>:<JID2>` syntax (colon-separated
# jobids === "after ALL listed jobs succeed").
#
# Login-node compute is not allowed on the cluster, so this wrapper merely
# chains sbatch submissions; the heavy lifting runs inside Slurm.
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/$USER/longcat-video-tta
#   bash scripts/sbatch/submit_per_video_feature_pipeline.sh
#
# Skip the OOD job for a quick re-run (falls back to the original
# single-dependency form so stage 2 only waits on extraction):
#   SKIP_OOD=1 bash scripts/sbatch/submit_per_video_feature_pipeline.sh
#
# Override any input/output path without editing this script, e.g.:
#   VIDEOS_DIR=/scratch/$USER/longcat-video-tta/datasets/panda_2048_480p \
#   OUTPUT_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features_2048.csv \
#   OOD_OUTPUT_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores_2048.csv \
#   FEATURES_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features_2048.csv \
#       bash scripts/sbatch/submit_per_video_feature_pipeline.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"

EXTRACT_SBATCH="${EXTRACT_SBATCH:-scripts/sbatch/run_extract_video_features.sbatch}"
OOD_SBATCH="${OOD_SBATCH:-scripts/sbatch/run_compute_diffusion_ood.sbatch}"
CORR_SBATCH="${CORR_SBATCH:-scripts/sbatch/run_correlate_tta_gain.sbatch}"

# ============================================================================
# Stage 1a (feature extraction) defaults — passed through to
# run_extract_video_features.sbatch
# ============================================================================
VIDEOS_DIR="${VIDEOS_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
CAPTIONS_CSV="${CAPTIONS_CSV:-${VIDEOS_DIR}/metadata.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv}"
TTA_VISIBLE_FRAMES="${TTA_VISIBLE_FRAMES:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"

# ============================================================================
# Stage 1b (diffusion-OOD) defaults — passed through to
# run_compute_diffusion_ood.sbatch
# ============================================================================
SKIP_OOD="${SKIP_OOD:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/longcat-video-checkpoints}"
OOD_OUTPUT_CSV="${OOD_OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
TIMESTEPS="${TIMESTEPS:-100,500,900}"
SEED="${SEED:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
RESUME="${RESUME:-0}"

# ============================================================================
# Stage 2 (correlation) defaults — passed through to run_correlate_tta_gain.sbatch
# Default FEATURES_CSV to match stage 1a's OUTPUT_CSV so the chain is
# consistent when neither is overridden. OOD_CSV defaults to stage 1b's
# OOD_OUTPUT_CSV; if SKIP_OOD=1 we pass an empty OOD_CSV.
# ============================================================================
GAINS_CSV="${GAINS_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv}"
FEATURES_CSV="${FEATURES_CSV:-${OUTPUT_CSV}}"
OOD_CSV="${OOD_CSV:-${OOD_OUTPUT_CSV}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation}"

# ============================================================================
# Banner
# ============================================================================
echo "============================================================"
echo "Per-video feature + OOD extraction -> correlation pipeline"
echo "============================================================"
echo "  PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "  EXTRACT_SBATCH     : ${EXTRACT_SBATCH}"
echo "  OOD_SBATCH         : ${OOD_SBATCH}  ($([ "${SKIP_OOD}" = "1" ] && echo "SKIPPED" || echo "submitted"))"
echo "  CORR_SBATCH        : ${CORR_SBATCH}"
echo ""
echo "Stage 1a (feature extraction, GPU):"
echo "  VIDEOS_DIR         : ${VIDEOS_DIR}"
echo "  CAPTIONS_CSV       : ${CAPTIONS_CSV}"
echo "  OUTPUT_CSV         : ${OUTPUT_CSV}"
echo "  TTA_VISIBLE_FRAMES : ${TTA_VISIBLE_FRAMES}"
echo "  BATCH_SIZE         : ${BATCH_SIZE}"
echo "  DEVICE             : ${DEVICE}"
echo ""
if [ "${SKIP_OOD}" = "1" ]; then
echo "Stage 1b (diffusion-OOD, GPU): SKIPPED (SKIP_OOD=1)"
else
echo "Stage 1b (diffusion-OOD, GPU):"
echo "  CHECKPOINT_DIR     : ${CHECKPOINT_DIR}"
echo "  VIDEOS_DIR         : ${VIDEOS_DIR}"
echo "  CAPTIONS_CSV       : ${CAPTIONS_CSV}"
echo "  OOD_OUTPUT_CSV     : ${OOD_OUTPUT_CSV}"
echo "  TTA_VISIBLE_FRAMES : ${TTA_VISIBLE_FRAMES}"
echo "  TTA_CONTEXT_FRAMES : ${TTA_CONTEXT_FRAMES}"
echo "  TIMESTEPS          : ${TIMESTEPS}"
echo "  SEED               : ${SEED}"
echo "  MAX_VIDEOS         : ${MAX_VIDEOS}"
echo "  RESUME             : ${RESUME}"
fi
echo ""
echo "Stage 2 (correlation, CPU, depends on stage 1a$([ "${SKIP_OOD}" = "1" ] || echo " + 1b")):"
echo "  GAINS_CSV          : ${GAINS_CSV}"
echo "  FEATURES_CSV       : ${FEATURES_CSV}"
if [ "${SKIP_OOD}" = "1" ]; then
echo "  OOD_CSV            : (skipped)"
else
echo "  OOD_CSV            : ${OOD_CSV}"
fi
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
if [ "${SKIP_OOD}" != "1" ] && [ ! -f "${OOD_SBATCH}" ]; then
    echo "ERROR: OOD-score sbatch not found: ${OOD_SBATCH}" >&2
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
if [ "${SKIP_OOD}" != "1" ] && [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "WARN: CHECKPOINT_DIR does not exist on the login node: ${CHECKPOINT_DIR}"
    echo "      Stage 1b will fail unless this resolves on the compute node."
fi
if [ ! -f "${GAINS_CSV}" ]; then
    echo "WARN: GAINS_CSV does not exist yet: ${GAINS_CSV}"
    echo "      Stage 2 will fail unless this file appears before stage 1 finishes."
fi

# ============================================================================
# Stage 1a: feature extraction
# ============================================================================
EXTRACT_JID=$(sbatch --parsable \
    --export="ALL,VIDEOS_DIR=${VIDEOS_DIR},CAPTIONS_CSV=${CAPTIONS_CSV},OUTPUT_CSV=${OUTPUT_CSV},TTA_VISIBLE_FRAMES=${TTA_VISIBLE_FRAMES},BATCH_SIZE=${BATCH_SIZE},DEVICE=${DEVICE}" \
    "${EXTRACT_SBATCH}")
echo "Submitted feature extraction: ${EXTRACT_JID}"

# ============================================================================
# Stage 1b: diffusion-OOD scoring (parallel with stage 1a)
# ============================================================================
OOD_JID=""
if [ "${SKIP_OOD}" != "1" ]; then
    OOD_JID=$(sbatch --parsable \
        --export="ALL,CHECKPOINT_DIR=${CHECKPOINT_DIR},VIDEOS_DIR=${VIDEOS_DIR},CAPTIONS_CSV=${CAPTIONS_CSV},OUTPUT_CSV=${OOD_OUTPUT_CSV},TTA_VISIBLE_FRAMES=${TTA_VISIBLE_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},TIMESTEPS=${TIMESTEPS},SEED=${SEED},DEVICE=${DEVICE},MAX_VIDEOS=${MAX_VIDEOS},RESUME=${RESUME}" \
        "${OOD_SBATCH}")
    echo "Submitted diffusion-OOD computation: ${OOD_JID}"
fi

# ============================================================================
# Stage 2: correlation (afterok on stage 1a AND stage 1b when present)
#
# Slurm dependency grammar: `--dependency=afterok:A:B` means "after BOTH A
# AND B succeed". When SKIP_OOD=1 we fall back to the single-dependency form
# so stage 2 only waits on extraction.
# ============================================================================
if [ -n "${OOD_JID}" ]; then
    DEP_SPEC="afterok:${EXTRACT_JID}:${OOD_JID}"
    DEP_DESC="${EXTRACT_JID} + ${OOD_JID}"
    CORR_EXPORT="ALL,GAINS_CSV=${GAINS_CSV},FEATURES_CSV=${FEATURES_CSV},OOD_CSV=${OOD_CSV},OUTPUT_DIR=${OUTPUT_DIR}"
else
    DEP_SPEC="afterok:${EXTRACT_JID}"
    DEP_DESC="${EXTRACT_JID}"
    # Explicitly clear OOD_CSV when SKIP_OOD=1 so any externally-set OOD_CSV
    # leaking through ALL does not silently re-enable the OOD join in stage 2.
    CORR_EXPORT="ALL,GAINS_CSV=${GAINS_CSV},FEATURES_CSV=${FEATURES_CSV},OOD_CSV=,OUTPUT_DIR=${OUTPUT_DIR}"
fi

CORR_JID=$(sbatch --parsable \
    --dependency="${DEP_SPEC}" \
    --export="${CORR_EXPORT}" \
    "${CORR_SBATCH}")
echo "Submitted correlation (depends on ${DEP_DESC}): ${CORR_JID}"

echo ""
echo "============================================================"
if [ -n "${OOD_JID}" ]; then
    echo "Submitted 3 jobs."
    echo ""
    echo "Monitor:"
    echo "  squeue -u \$USER -j ${EXTRACT_JID},${OOD_JID},${CORR_JID}"
    echo "  squeue -u \$USER | grep -E 'extract_video_features|compute_diffusion_ood|correlate_tta_gain'"
    echo ""
    echo "Logs:"
    echo "  ${PROJECT_ROOT}/sweep_experiment/logs/extract_video_features_${EXTRACT_JID}.{out,err}"
    echo "  ${PROJECT_ROOT}/sweep_experiment/logs/compute_diffusion_ood_${OOD_JID}.{out,err}"
    echo "  ${PROJECT_ROOT}/sweep_experiment/logs/correlate_tta_gain_${CORR_JID}.{out,err}"
else
    echo "Submitted 2 jobs (OOD skipped via SKIP_OOD=1)."
    echo ""
    echo "Monitor:"
    echo "  squeue -u \$USER -j ${EXTRACT_JID},${CORR_JID}"
    echo "  squeue -u \$USER | grep -E 'extract_video_features|correlate_tta_gain'"
    echo ""
    echo "Logs:"
    echo "  ${PROJECT_ROOT}/sweep_experiment/logs/extract_video_features_${EXTRACT_JID}.{out,err}"
    echo "  ${PROJECT_ROOT}/sweep_experiment/logs/correlate_tta_gain_${CORR_JID}.{out,err}"
fi
echo "============================================================"
