#!/usr/bin/env bash
# PSNR-targeted deploy router: same 9-d Block A, predict PSNR per config.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log
JID=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --cpus-per-task=4 \
  --mem=16G \
  --time=00:30:00 \
  --job-name=deploy_psnr_router \
  --output="sweep_experiment/slurm_log/deploy_psnr_router_%j.out" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},FEATURE_DATE=${FEATURE_DATE}" \
  --wrap="cd ${PROJECT_ROOT} && bash scripts/run_deploy_psnr_router.sh")
echo "Deploy PSNR router: ${JID}"
echo "  cat sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/deploy_psnr_router/summary.md"
