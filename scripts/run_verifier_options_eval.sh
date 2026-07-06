#!/usr/bin/env bash
# CPU eval wrapper for all four verifier routing options.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
SCORES_DIR="${SCORES_DIR:-${PROJECT_ROOT}/sweep_experiment/reports/verifier_scores}"
FEATURES_DIR="${FEATURES_DIR:-${PROJECT_ROOT}/sweep_experiment/reports/verifier_features}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/verifier_options_eval}"

cd "${PROJECT_ROOT}"
python3 scripts/run_verifier_options_eval.py \
  --feature-date "${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}" \
  --scores-dir "${SCORES_DIR}" \
  --features-dir "${FEATURES_DIR}" \
  --output-dir "${OUT_DIR}" \
  --run-all "$@"

echo ""
echo "Results: ${OUT_DIR}/summary.md"
cat "${OUT_DIR}/summary.md"
