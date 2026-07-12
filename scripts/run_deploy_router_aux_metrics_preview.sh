#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/preview_1000v_env.sh
source "${SCRIPT_DIR}/preview_1000v_env.sh"

OUT="${OUT:-${PREVIEW_FEATURE_DIR}/deploy_router_aux_metrics}"
GT_CACHE="${GT_CACHE:-${REPO}/gt_caches/panda_1000_longcat.npz}"
RUN_FVD="${RUN_FVD:-0}"

cd "${REPO}"

EXTRA=()
if [ "${RUN_FVD}" = "1" ]; then
  EXTRA+=(--run-fvd --gt-cache "${GT_CACHE}" --device cuda)
fi

python3 scripts/analyze_deploy_router_aux_metrics.py \
  --series-root "${PREVIEW_SERIES_ROOT}" \
  --feature-date "${PREVIEW_FEATURE_DIR}" \
  --output-dir "${OUT}" \
  "${EXTRA[@]}"

echo "Done: ${OUT}/summary.md"
