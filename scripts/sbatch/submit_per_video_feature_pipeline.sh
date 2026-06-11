#!/bin/bash
# ============================================================================
# Submit the per-video feature-extraction + diffusion-OOD + Tier-3 probe
# → correlation pipeline.
#
# Stage 1a (GPU): scripts/extract_video_features_for_tta.py
#                 via scripts/sbatch/run_extract_video_features.sbatch
#                 (CLIP / DINO / cuts / texture / colour Tier-1 features)
# Stage 1b (GPU): scripts/compute_diffusion_ood_score.py
#                 via scripts/sbatch/run_compute_diffusion_ood.sbatch
#                 (per-video flow-matching MSE against LongCat-Video base;
#                  Tier-2 OOD-score proxy for TTA gain)
# Stage 1c (GPU): scripts/compute_tier3_probes.py
#                 via scripts/sbatch/run_compute_tier3_probes.sbatch
#                 (H-T3-1 grad_norm_θ0 + H-T3-2 single_step_loss_drop against
#                  a fresh LoRA r=8 adapter; the gating-plan §3.1 Tier-3
#                  authorisation from Decision 4)
# Stage 2  (CPU): scripts/correlate_tta_gain_with_features.py
#                 via scripts/sbatch/run_correlate_tta_gain.sbatch
#                 (scheduled with --dependency=afterok:1a:1b:1c so the
#                  combined ρ(ΔPSNR, X) report covers ALL three feature CSVs)
#
# Stages 1a, 1b, and 1c are independent of each other (all three depend
# only on the dataset) so they fan out in parallel. Stage 2 chains behind
# all three via Slurm's `--dependency=afterok:<JID1>:<JID2>:<JID3>` syntax
# (colon-separated jobids === "after ALL listed jobs succeed").
#
# Login-node compute is not allowed on the cluster, so this wrapper merely
# chains sbatch submissions; the heavy lifting runs inside Slurm.
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/$USER/longcat-video-tta
#   bash scripts/sbatch/submit_per_video_feature_pipeline.sh
#
# Skip the OOD job for a quick re-run (falls back to a shorter dependency
# list so stage 2 only waits on the remaining jobs):
#   SKIP_OOD=1 bash scripts/sbatch/submit_per_video_feature_pipeline.sh
#
# Skip the Tier-3 probe job (mirrors SKIP_OOD pattern; same dependency
# fallback):
#   SKIP_TIER3=1 bash scripts/sbatch/submit_per_video_feature_pipeline.sh
#
# Skip BOTH the OOD and Tier-3 jobs (fastest re-run path; correlation only
# joins the feature-extractor CSV):
#   SKIP_OOD=1 SKIP_TIER3=1 bash scripts/sbatch/submit_per_video_feature_pipeline.sh
#
# Override any input/output path without editing this script, e.g.:
#   VIDEOS_DIR=/scratch/$USER/longcat-video-tta/datasets/panda_2048_480p \
#   OUTPUT_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features_2048.csv \
#   OOD_OUTPUT_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores_2048.csv \
#   TIER3_OUTPUT_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features_2048.csv \
#   FEATURES_CSV=/scratch/$USER/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features_2048.csv \
#       bash scripts/sbatch/submit_per_video_feature_pipeline.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"

EXTRACT_SBATCH="${EXTRACT_SBATCH:-scripts/sbatch/run_extract_video_features.sbatch}"
OOD_SBATCH="${OOD_SBATCH:-scripts/sbatch/run_compute_diffusion_ood.sbatch}"
TIER3_SBATCH="${TIER3_SBATCH:-scripts/sbatch/run_compute_tier3_probes.sbatch}"
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
# Stage 1c (Tier-3 probes) defaults — passed through to
# run_compute_tier3_probes.sbatch
# ============================================================================
SKIP_TIER3="${SKIP_TIER3:-0}"
TIER3_OUTPUT_CSV="${TIER3_OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features.csv}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_LR="${LORA_LR:-5.0e-5}"
LORA_WEIGHT_DECAY="${LORA_WEIGHT_DECAY:-0.01}"
LORA_TARGETS="${LORA_TARGETS:-qkv,proj}"
LORA_TARGET_BLOCKS="${LORA_TARGET_BLOCKS:-all}"
LORA_TARGET_FFN="${LORA_TARGET_FFN:-0}"

# ============================================================================
# Stage 2 (correlation) defaults — passed through to run_correlate_tta_gain.sbatch
# Default FEATURES_CSV to match stage 1a's OUTPUT_CSV so the chain is
# consistent when neither is overridden. OOD_CSV defaults to stage 1b's
# OOD_OUTPUT_CSV; TIER3_CSV defaults to stage 1c's TIER3_OUTPUT_CSV. If
# SKIP_OOD=1 / SKIP_TIER3=1 we pass an empty value so the correlation
# script silently drops that join (mirrors the existing OOD-fallback path).
# ============================================================================
GAINS_CSV="${GAINS_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv}"
FEATURES_CSV="${FEATURES_CSV:-${OUTPUT_CSV}}"
OOD_CSV="${OOD_CSV:-${OOD_OUTPUT_CSV}}"
TIER3_CSV="${TIER3_CSV:-${TIER3_OUTPUT_CSV}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation}"

# ============================================================================
# Banner
# ============================================================================
echo "============================================================"
echo "Per-video feature + OOD + Tier-3 probe -> correlation pipeline"
echo "============================================================"
echo "  PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "  EXTRACT_SBATCH     : ${EXTRACT_SBATCH}"
echo "  OOD_SBATCH         : ${OOD_SBATCH}    ($([ "${SKIP_OOD}"   = "1" ] && echo "SKIPPED" || echo "submitted"))"
echo "  TIER3_SBATCH       : ${TIER3_SBATCH}  ($([ "${SKIP_TIER3}" = "1" ] && echo "SKIPPED" || echo "submitted"))"
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
if [ "${SKIP_TIER3}" = "1" ]; then
echo "Stage 1c (Tier-3 probes, GPU): SKIPPED (SKIP_TIER3=1)"
else
echo "Stage 1c (Tier-3 probes, GPU):"
echo "  CHECKPOINT_DIR     : ${CHECKPOINT_DIR}"
echo "  VIDEOS_DIR         : ${VIDEOS_DIR}"
echo "  CAPTIONS_CSV       : ${CAPTIONS_CSV}"
echo "  TIER3_OUTPUT_CSV   : ${TIER3_OUTPUT_CSV}"
echo "  TTA_VISIBLE_FRAMES : ${TTA_VISIBLE_FRAMES}"
echo "  TTA_CONTEXT_FRAMES : ${TTA_CONTEXT_FRAMES}"
echo "  TIMESTEPS          : ${TIMESTEPS}"
echo "  SEED               : ${SEED}"
echo "  MAX_VIDEOS         : ${MAX_VIDEOS}"
echo "  RESUME             : ${RESUME}"
echo "  LORA_RANK / ALPHA  : ${LORA_RANK} / ${LORA_ALPHA}"
echo "  LORA_LR            : ${LORA_LR}"
echo "  LORA_WEIGHT_DECAY  : ${LORA_WEIGHT_DECAY}"
echo "  LORA_TARGETS       : ${LORA_TARGETS}"
echo "  LORA_TARGET_BLOCKS : ${LORA_TARGET_BLOCKS}"
echo "  LORA_TARGET_FFN    : ${LORA_TARGET_FFN}"
fi
echo ""
dep_desc="stage 1a"
[ "${SKIP_OOD}"   = "1" ] || dep_desc="${dep_desc} + 1b"
[ "${SKIP_TIER3}" = "1" ] || dep_desc="${dep_desc} + 1c"
echo "Stage 2 (correlation, CPU, depends on ${dep_desc}):"
echo "  GAINS_CSV          : ${GAINS_CSV}"
echo "  FEATURES_CSV       : ${FEATURES_CSV}"
if [ "${SKIP_OOD}" = "1" ]; then
echo "  OOD_CSV            : (skipped)"
else
echo "  OOD_CSV            : ${OOD_CSV}"
fi
if [ "${SKIP_TIER3}" = "1" ]; then
echo "  TIER3_CSV          : (skipped)"
else
echo "  TIER3_CSV          : ${TIER3_CSV}"
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
if [ "${SKIP_TIER3}" != "1" ] && [ ! -f "${TIER3_SBATCH}" ]; then
    echo "ERROR: Tier-3 probe sbatch not found: ${TIER3_SBATCH}" >&2
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
if { [ "${SKIP_OOD}" != "1" ] || [ "${SKIP_TIER3}" != "1" ]; } && [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "WARN: CHECKPOINT_DIR does not exist on the login node: ${CHECKPOINT_DIR}"
    echo "      Stage 1b / 1c will fail unless this resolves on the compute node."
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
# Stage 1b: diffusion-OOD scoring (parallel with stage 1a + 1c)
# ============================================================================
OOD_JID=""
if [ "${SKIP_OOD}" != "1" ]; then
    OOD_JID=$(sbatch --parsable \
        --export="ALL,CHECKPOINT_DIR=${CHECKPOINT_DIR},VIDEOS_DIR=${VIDEOS_DIR},CAPTIONS_CSV=${CAPTIONS_CSV},OUTPUT_CSV=${OOD_OUTPUT_CSV},TTA_VISIBLE_FRAMES=${TTA_VISIBLE_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},TIMESTEPS=${TIMESTEPS},SEED=${SEED},DEVICE=${DEVICE},MAX_VIDEOS=${MAX_VIDEOS},RESUME=${RESUME}" \
        "${OOD_SBATCH}")
    echo "Submitted diffusion-OOD computation: ${OOD_JID}"
fi

# ============================================================================
# Stage 1c: Tier-3 probes (parallel with stage 1a + 1b)
# ============================================================================
TIER3_JID=""
if [ "${SKIP_TIER3}" != "1" ]; then
    TIER3_JID=$(sbatch --parsable \
        --export="ALL,CHECKPOINT_DIR=${CHECKPOINT_DIR},VIDEOS_DIR=${VIDEOS_DIR},CAPTIONS_CSV=${CAPTIONS_CSV},OUTPUT_CSV=${TIER3_OUTPUT_CSV},TTA_VISIBLE_FRAMES=${TTA_VISIBLE_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},TIMESTEPS=${TIMESTEPS},SEED=${SEED},DEVICE=${DEVICE},MAX_VIDEOS=${MAX_VIDEOS},RESUME=${RESUME},LORA_RANK=${LORA_RANK},LORA_ALPHA=${LORA_ALPHA},LORA_LR=${LORA_LR},LORA_WEIGHT_DECAY=${LORA_WEIGHT_DECAY},LORA_TARGETS=${LORA_TARGETS},LORA_TARGET_BLOCKS=${LORA_TARGET_BLOCKS},LORA_TARGET_FFN=${LORA_TARGET_FFN}" \
        "${TIER3_SBATCH}")
    echo "Submitted Tier-3 probe computation: ${TIER3_JID}"
fi

# ============================================================================
# Stage 2: correlation (afterok on every stage-1 job that was submitted)
#
# Slurm dependency grammar: `--dependency=afterok:A:B:C` means "after ALL of
# A, B, AND C succeed". We build the colon-separated list dynamically from
# whichever stage-1 jobs were actually submitted so SKIP_OOD / SKIP_TIER3
# fall back to a shorter dependency without breaking the chain.
# ============================================================================
DEP_JIDS="${EXTRACT_JID}"
DEP_DESC="${EXTRACT_JID}"
if [ -n "${OOD_JID}" ]; then
    DEP_JIDS="${DEP_JIDS}:${OOD_JID}"
    DEP_DESC="${DEP_DESC} + ${OOD_JID}"
fi
if [ -n "${TIER3_JID}" ]; then
    DEP_JIDS="${DEP_JIDS}:${TIER3_JID}"
    DEP_DESC="${DEP_DESC} + ${TIER3_JID}"
fi
DEP_SPEC="afterok:${DEP_JIDS}"

# Explicitly clear OOD_CSV / TIER3_CSV when skipped so any externally-set
# value leaking through ALL does not silently re-enable the join.
EFFECTIVE_OOD_CSV="${OOD_CSV}"
[ "${SKIP_OOD}"   = "1" ] && EFFECTIVE_OOD_CSV=""
EFFECTIVE_TIER3_CSV="${TIER3_CSV}"
[ "${SKIP_TIER3}" = "1" ] && EFFECTIVE_TIER3_CSV=""

CORR_EXPORT="ALL,GAINS_CSV=${GAINS_CSV},FEATURES_CSV=${FEATURES_CSV},OOD_CSV=${EFFECTIVE_OOD_CSV},TIER3_CSV=${EFFECTIVE_TIER3_CSV},OUTPUT_DIR=${OUTPUT_DIR}"

CORR_JID=$(sbatch --parsable \
    --dependency="${DEP_SPEC}" \
    --export="${CORR_EXPORT}" \
    "${CORR_SBATCH}")
echo "Submitted correlation (depends on ${DEP_DESC}): ${CORR_JID}"

echo ""
echo "============================================================"
N_SUBMITTED=1
[ -n "${OOD_JID}" ]   && N_SUBMITTED=$((N_SUBMITTED + 1))
[ -n "${TIER3_JID}" ] && N_SUBMITTED=$((N_SUBMITTED + 1))
N_SUBMITTED=$((N_SUBMITTED + 1))  # +correlation
echo "Submitted ${N_SUBMITTED} jobs."
echo ""

# Build the squeue list dynamically so monitor commands only reference jobs
# we actually fired.
SQ_LIST="${EXTRACT_JID}"
[ -n "${OOD_JID}" ]   && SQ_LIST="${SQ_LIST},${OOD_JID}"
[ -n "${TIER3_JID}" ] && SQ_LIST="${SQ_LIST},${TIER3_JID}"
SQ_LIST="${SQ_LIST},${CORR_JID}"

# And the squeue grep filter, again restricted to jobs we submitted.
GREP_RE="extract_video_features"
[ -n "${OOD_JID}" ]   && GREP_RE="${GREP_RE}|compute_diffusion_ood"
[ -n "${TIER3_JID}" ] && GREP_RE="${GREP_RE}|compute_tier3_probes"
GREP_RE="${GREP_RE}|correlate_tta_gain"

echo "Monitor:"
echo "  squeue -u \$USER -j ${SQ_LIST}"
echo "  squeue -u \$USER | grep -E '${GREP_RE}'"
echo ""
echo "Logs:"
echo "  ${PROJECT_ROOT}/sweep_experiment/logs/extract_video_features_${EXTRACT_JID}.{out,err}"
[ -n "${OOD_JID}" ]   && echo "  ${PROJECT_ROOT}/sweep_experiment/logs/compute_diffusion_ood_${OOD_JID}.{out,err}"
[ -n "${TIER3_JID}" ] && echo "  ${PROJECT_ROOT}/sweep_experiment/logs/compute_tier3_probes_${TIER3_JID}.{out,err}"
echo "  ${PROJECT_ROOT}/sweep_experiment/logs/correlate_tta_gain_${CORR_JID}.{out,err}"
echo "============================================================"
