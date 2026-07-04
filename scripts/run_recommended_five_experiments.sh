#!/usr/bin/env bash
# Recommended 5-experiment routing program (pilot N=200).
#
#   bash scripts/run_recommended_five_experiments.sh
#   EXPERIMENT=exp2_dyn_delta_router bash scripts/run_recommended_five_experiments.sh
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-05}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/recommended_five_experiments}"

cd "$REPO"

ARGS=(
  --series-root "$REPO/sweep_experiment/results/panda_ood_budget_pilot"
  --feature-date "$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}"
  --output-dir "$OUT"
)

if [ -n "${EXPERIMENT:-}" ]; then
  python3 scripts/run_recommended_five_experiments.py "${ARGS[@]}" --experiment "$EXPERIMENT"
else
  python3 scripts/run_recommended_five_experiments.py "${ARGS[@]}" --run-all
fi

echo "Done: $OUT/recommended_five_experiments_summary.md"
