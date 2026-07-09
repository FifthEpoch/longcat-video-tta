#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/deploy_psnr_router}"
cd "$REPO"
python3 scripts/run_deploy_psnr_router.py \
  --series-root "$REPO/sweep_experiment/results/panda_ood_budget_pilot" \
  --feature-date "$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}" \
  --output-dir "$OUT"
echo "Done: $OUT/summary.md"
