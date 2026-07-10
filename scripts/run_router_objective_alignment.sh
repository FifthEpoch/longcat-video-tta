#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/router_objective_alignment}"
VB_CSV="${VB_CSV:-$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/deploy_router_aux_metrics/router_runs/video_caption_only/budget_config_oof_predictions.csv}"
PSNR_CSV="${PSNR_CSV:-$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/deploy_psnr_router/budget_config_oof_predictions.csv}"
cd "$REPO"
EXTRA=()
if [ -f "$VB_CSV" ]; then EXTRA+=(--vb-picks-csv "$VB_CSV"); fi
if [ -f "$PSNR_CSV" ]; then EXTRA+=(--psnr-picks-csv "$PSNR_CSV"); fi
python3 scripts/analyze_router_objective_alignment.py \
  --series-root "$REPO/sweep_experiment/results/panda_ood_budget_pilot" \
  --feature-date "$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}" \
  --output-dir "$OUT" \
  "${EXTRA[@]}"
echo "Done: $OUT/summary.md"
