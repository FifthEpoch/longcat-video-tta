#!/usr/bin/env bash
# Re-score deploy routers with structured feature blocks (CPU @ N=200).
#
# Blocks: A=video/caption (9) · B=diffusion-OOD (~20) · C=VAE profile (~130)
# Headline when OOD allowed: video_caption_ood (A+B, ~29-d)
#
# Prereqs: video_features.csv, diffusion_ood_scores.csv, pilot VBench results
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
