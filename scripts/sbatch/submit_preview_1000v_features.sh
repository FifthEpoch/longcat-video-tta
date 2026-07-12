#!/usr/bin/env bash
# Extract router feature CSVs for the 1000v OOD-preview set.
#
# 1) GPU: 9-d video/caption features (extract_video_features_for_tta.py)
# 2) CPU: filter segment-pool OOD CSV → preview retain list (no re-GPU)
# 3) GPU: VAE latent profile (~130-d Block C)
#
# Prereqs:
#   bash scripts/sample_segment_pool_ood_preview_1000v.sh
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash scripts/sbatch/submit_preview_1000v_features.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/preview_1000v_env.sh
source "${SCRIPT_DIR}/../preview_1000v_env.sh"

PROJECT_ROOT="${REPO}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

cd "${PROJECT_ROOT}"
mkdir -p "${PREVIEW_FEATURE_DIR}" sweep_experiment/slurm_log

if [ ! -d "${PREVIEW_DATASET_DIR}" ]; then
    echo "ERROR: preview dataset missing: ${PREVIEW_DATASET_DIR}" >&2
    echo "Run: bash scripts/sample_segment_pool_ood_preview_1000v.sh" >&2
    exit 1
fi
if [ ! -f "${REPO}/${PREVIEW_JSON}" ]; then
    echo "ERROR: retain list missing: ${REPO}/${PREVIEW_JSON}" >&2
    exit 1
fi
if [ ! -f "${SEGMENT_OOD_CSV}" ]; then
    echo "ERROR: segment-pool OOD CSV missing: ${SEGMENT_OOD_CSV}" >&2
    exit 1
fi

VIDEO_OUT="${PREVIEW_FEATURE_DIR}/video_features.csv"
OOD_OUT="${PREVIEW_FEATURE_DIR}/diffusion_ood_scores.csv"
VAE_OUT="${PREVIEW_FEATURE_DIR}/vae_latent_profile_features.csv"

echo "============================================================"
echo "Preview 1000v feature extraction"
echo "============================================================"
echo "  dataset      : ${PREVIEW_DATASET_DIR}"
echo "  feature dir  : ${PREVIEW_FEATURE_DIR}"
echo "  segment OOD  : ${SEGMENT_OOD_CSV}"
echo "============================================================"

# OOD filter is CPU-only — no dependency on video/VAE jobs.
OOD_JID=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --cpus-per-task=2 \
    --mem=4G \
    --time=00:15:00 \
    --job-name=filter_ood_preview \
    --output="sweep_experiment/slurm_log/filter_ood_preview_%j.out" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},PREVIEW_FEATURE_DATE=${PREVIEW_FEATURE_DATE}" \
    --wrap="cd ${PROJECT_ROOT} && source scripts/preview_1000v_env.sh && python3 scripts/filter_ood_csv_for_retain_list.py --ood-csv \"\${SEGMENT_OOD_CSV}\" --retain-json \"\${REPO}/\${PREVIEW_JSON}\" --output \"\${PREVIEW_FEATURE_DIR}/diffusion_ood_scores.csv\"")

VIDEO_JID=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --time=04:00:00 \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},VIDEOS_DIR=${PREVIEW_DATASET_DIR},CAPTIONS_CSV=${PREVIEW_DATASET_DIR}/metadata.csv,OUTPUT_CSV=${VIDEO_OUT}" \
    scripts/sbatch/run_extract_video_features.sbatch)

VAE_JID=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --time=06:00:00 \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},VIDEOS_DIR=${PREVIEW_DATASET_DIR},OUTPUT_CSV=${VAE_OUT},RESUME=1" \
    scripts/sbatch/run_extract_vae_latent_profile.sbatch)

echo "Filter OOD     : ${OOD_JID}"
echo "Video features : ${VIDEO_JID}"
echo "VAE profile    : ${VAE_JID}"
echo ""
echo "When done, check:"
echo "  wc -l ${VIDEO_OUT} ${OOD_OUT} ${VAE_OUT}"
echo "  # expect ~1001 lines each (1000 + header)"
echo ""
echo "Next (after budget sweep + VBench backfill):"
echo "  bash sweep_experiment/sbatch/submit_deploy_router_1000v_preview.sh"
