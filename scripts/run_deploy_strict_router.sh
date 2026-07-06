#!/usr/bin/env bash
# Deploy-strict ridge routers: video/caption (+ VAE encode path only).
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/deploy_strict_router}"

cd "$REPO"

python3 scripts/run_deploy_strict_router_experiments.py \
  --series-root "$REPO/sweep_experiment/results/panda_ood_budget_pilot" \
  --feature-date "$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}" \
  --output-dir "$OUT" \
  --run-all

echo "Done: $OUT/summary.md"
