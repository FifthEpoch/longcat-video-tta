#!/usr/bin/env bash
# Re-score budget router: VAE inference embedding ONLY (CPU @ N=200).
#
# Router input: vae_latent_profile_features.csv ONLY (~130-d from encode_video).
# NO video_features, NO OOD/Tier-3/probe/TTA metrics.
#
# Prereqs:
#   .../2026-07-06/vae_latent_profile_features.csv
#   sweep_experiment/results/panda_ood_budget_pilot/  (12-config VBench — labels only)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
DATE_TAG="${DATE_TAG:-2026-07-06}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

JID=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --cpus-per-task=4 \
  --mem=16G \
  --time=00:30:00 \
  --job-name=deploy_router \
  --output="sweep_experiment/slurm_log/deploy_strict_router_%j.out" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},FEATURE_DATE=${FEATURE_DATE},DATE_TAG=${DATE_TAG}" \
  --wrap="cd ${PROJECT_ROOT} && bash scripts/run_deploy_strict_router.sh")

echo "Deploy-strict router eval: ${JID}"
echo ""
echo "When done:"
echo "  cat sweep_experiment/reports/per_video_analysis/${DATE_TAG}/deploy_strict_router/summary.md"
