#!/usr/bin/env bash
# Wave-1 VBench predictor screen (CPU, N=200 pilot).
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/wave1_predictor_experiments}"

cd "$REPO"

ARGS=(
  --series-root "$REPO/sweep_experiment/results/panda_ood_budget_pilot"
  --feature-date "$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}"
  --output-dir "$OUT"
)

if [ -n "${EXPERIMENT:-}" ]; then
  python3 scripts/run_vbench_predictor_wave1_experiments.py "${ARGS[@]}" --experiment "$EXPERIMENT"
else
  python3 scripts/run_vbench_predictor_wave1_experiments.py "${ARGS[@]}" --run-all
fi

echo "Done: $OUT"
