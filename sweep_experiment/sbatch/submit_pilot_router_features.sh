#!/usr/bin/env bash
# Phase-0 features on budget **pilot 200v** dataset → full router N=200.
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
FEAT_DIR="${REPO}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}"

export VIDEOS_DIR="${REPO}/datasets/panda_ood_budget_pilot_480p"
export CAPTIONS_CSV="${VIDEOS_DIR}/metadata.csv"
export OUTPUT_CSV="${FEAT_DIR}/video_features.csv"
export OOD_OUTPUT_CSV="${FEAT_DIR}/diffusion_ood_scores.csv"
export TIER3_OUTPUT_CSV="${FEAT_DIR}/tier3_probe_features.csv"
export FLOW_OUTPUT_CSV="${FEAT_DIR}/flow_shape_features.csv"
export BPP_OUTPUT_CSV="${FEAT_DIR}/bpp_features.csv"
export FFT_OUTPUT_CSV="${FEAT_DIR}/fft_features.csv"
export VAE_RECERR_OUTPUT_CSV="${FEAT_DIR}/vae_recerr_features.csv"
export MOTION_OUTPUT_CSV="${FEAT_DIR}/latent_motion_features.csv"
export LOSS_VAR_OUTPUT_CSV="${FEAT_DIR}/loss_variance_features.csv"

mkdir -p "${FEAT_DIR}"

echo "Pilot router features → ${FEAT_DIR}"
echo "  VIDEOS_DIR=${VIDEOS_DIR}"

# OOD + tier3 only (fast path for router); add full pipeline if time allows.
SKIP_TIER3="${SKIP_TIER3:-0}" \
bash "${REPO}/scripts/sbatch/submit_per_video_feature_pipeline.sh"

echo ""
echo "When jobs finish, retrain router:"
echo "  FEATURE_DATE=${DATE_TAG} DATE_TAG=2026-07-02 \\"
echo "    FEAT_DIR override via run_vbench_headroom_router.sh env vars"
echo "  # Or merge pilot CSVs into 2026-06-09 and re-run:"
echo "  DATE_TAG=2026-07-02 bash scripts/run_vbench_headroom_router.sh"
