#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/preview_1000v_env.sh
source "${SCRIPT_DIR}/preview_1000v_env.sh"

OUT="${OUT:-${PREVIEW_FEATURE_DIR}/router_objective_alignment}"
VB_CSV="${VB_CSV:-${PREVIEW_FEATURE_DIR}/deploy_router_aux_metrics/router_runs/video_caption_only/budget_config_oof_predictions.csv}"
PSNR_CSV="${PSNR_CSV:-${PREVIEW_FEATURE_DIR}/deploy_psnr_router/budget_config_oof_predictions.csv}"

cd "${REPO}"

EXTRA=()
if [ -f "${VB_CSV}" ]; then EXTRA+=(--vb-picks-csv "${VB_CSV}"); fi
if [ -f "${PSNR_CSV}" ]; then EXTRA+=(--psnr-picks-csv "${PSNR_CSV}"); fi

if ((${#EXTRA[@]} > 0)); then
  python3 scripts/analyze_router_objective_alignment.py \
    --series-root "${PREVIEW_SERIES_ROOT}" \
    --feature-date "${PREVIEW_FEATURE_DIR}" \
    --output-dir "${OUT}" \
    "${EXTRA[@]}"
else
  python3 scripts/analyze_router_objective_alignment.py \
    --series-root "${PREVIEW_SERIES_ROOT}" \
    --feature-date "${PREVIEW_FEATURE_DIR}" \
    --output-dir "${OUT}"
fi

echo "Done: ${OUT}/summary.md"
