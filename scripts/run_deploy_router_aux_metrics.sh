#!/usr/bin/env bash
# Cross-metric analysis for deploy routers (CPU + optional GPU FVD).
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/deploy_router_aux_metrics}"
GT_CACHE="${GT_CACHE:-$REPO/gt_caches/panda_1000_longcat.npz}"
RUN_FVD="${RUN_FVD:-0}"

cd "$REPO"

EXTRA=()
if [ "$RUN_FVD" = "1" ]; then
  EXTRA+=(--run-fvd --gt-cache "$GT_CACHE" --device cuda)
fi

python3 scripts/analyze_deploy_router_aux_metrics.py \
  --series-root "$REPO/sweep_experiment/results/panda_ood_budget_pilot" \
  --feature-date "$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}" \
  --output-dir "$OUT" \
  "${EXTRA[@]}"

echo "Done: $OUT/summary.md"
