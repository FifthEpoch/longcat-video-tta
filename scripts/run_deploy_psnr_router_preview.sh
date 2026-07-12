#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/preview_1000v_env.sh
source "${SCRIPT_DIR}/preview_1000v_env.sh"

OUT="${OUT:-${PREVIEW_FEATURE_DIR}/deploy_psnr_router}"

cd "${REPO}"

python3 scripts/run_deploy_psnr_router.py \
  --series-root "${PREVIEW_SERIES_ROOT}" \
  --feature-date "${PREVIEW_FEATURE_DIR}" \
  --output-dir "${OUT}"

echo "Done: ${OUT}/summary.md"
