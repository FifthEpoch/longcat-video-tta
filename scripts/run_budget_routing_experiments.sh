#!/usr/bin/env bash
# Run one routing experiment or the full suite (CPU-only, pilot N=200).
#
# Usage:
#   bash scripts/run_budget_routing_experiments.sh              # all experiments
#   EXPERIMENT=dim_dynamic_degree bash scripts/run_budget_routing_experiments.sh
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-05}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
SERIES="${SERIES:-$REPO/sweep_experiment/results/panda_ood_budget_pilot}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/budget_routing_experiments}"

cd "$REPO"

ARGS=(
  --series-root "$SERIES"
  --feature-date "$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}"
  --output-dir "$OUT"
)

if [ -n "${EXPERIMENT:-}" ]; then
  python3 scripts/run_budget_routing_experiments.py "${ARGS[@]}" --experiment "$EXPERIMENT"
else
  python3 scripts/run_budget_routing_experiments.py "${ARGS[@]}" --run-all
  python3 scripts/aggregate_budget_routing_results.py --input-dir "$OUT"
  python3 scripts/eval_budget_routing_on_999v.py \
    --output "$OUT/999v_proxy_bestof3.md" || true
fi

echo "Done: $OUT"
