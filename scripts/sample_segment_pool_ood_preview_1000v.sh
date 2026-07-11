#!/bin/bash
# ============================================================================
# Sample 1000 OOD-quintile-stratified videos from the *partial* segment-pool
# OOD CSV (while full 29K scoring is still in flight).
#
# Preview set: 200 videos × 5 quintiles from whatever rows exist in the OOD
# CSV today. Quintile edges are defined on this scored subset only — NOT the
# final full-pool distribution. Scoring order is canonical video_id sort, so
# the pool is an early prefix of the segment pool (selection bias vs final).
#
# Still useful to preview routers at N=1000 train/OOF vs N=200 pilot.
#
# Usage (cluster, while OOD job 13325919 runs):
#   cd /scratch/wc3013/longcat-video-tta
#   bash scripts/sample_segment_pool_ood_preview_1000v.sh
#
# Dry-run (print counts only):
#   DRY_RUN=1 bash scripts/sample_segment_pool_ood_preview_1000v.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
OOD_CSV="${OOD_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-07-10/diffusion_ood_scores_segment_pool.csv}"
SOURCE_DATASET="${SOURCE_DATASET:-${PROJECT_ROOT}/datasets/panda_segment_pool}"
OUTPUT_JSON="${OUTPUT_JSON:-${PROJECT_ROOT}/sweep_experiment/lists/panda_ood_budget_1000v_preview_videos.json}"
CREATE_DATASET="${CREATE_DATASET:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
PER_QUINTILE="${PER_QUINTILE:-200}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
MIN_SCORED="${MIN_SCORED:-5000}"

cd "${PROJECT_ROOT}"

if [ ! -f "${OOD_CSV}" ]; then
    echo "ERROR: OOD CSV not found: ${OOD_CSV}" >&2
    exit 1
fi

N_SCORED=$(( $(wc -l < "${OOD_CSV}") - 1 ))
MIN_PER_BIN=$(( N_SCORED / 5 ))

echo "============================================================"
echo "Segment-pool OOD preview 1000v sampling"
echo "============================================================"
echo "  OOD CSV        : ${OOD_CSV}"
echo "  scored videos  : ${N_SCORED}"
echo "  per quintile   : ${PER_QUINTILE} (target total ${PER_QUINTILE}×5)"
echo "  min per bin    : ~${MIN_PER_BIN} (need >= ${PER_QUINTILE})"
echo "  source dataset : ${SOURCE_DATASET}"
echo "  output json    : ${OUTPUT_JSON}"
echo "  create dataset : ${CREATE_DATASET}"
echo "  seed           : ${SEED}"
echo ""
echo "  NOTE: quintiles are on scored-prefix only; not final 29K pool."
echo "============================================================"

if [ "${N_SCORED}" -lt "${MIN_SCORED}" ]; then
    echo "WARN: only ${N_SCORED} scored videos (< ${MIN_SCORED}); proceeding anyway." >&2
fi

if [ "${MIN_PER_BIN}" -lt "${PER_QUINTILE}" ]; then
    echo "ERROR: not enough scored videos per quintile (${MIN_PER_BIN} < ${PER_QUINTILE})." >&2
    echo "Wait for more OOD scoring or lower PER_QUINTILE." >&2
    exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY] Would run sample_ood_quintile_videos.py"
    exit 0
fi

python3 scripts/sample_ood_quintile_videos.py \
    --ood-csv "${OOD_CSV}" \
    --source-dataset "${SOURCE_DATASET}" \
    --per-quintile "${PER_QUINTILE}" \
    --seed "${SEED}" \
    --output-json "${OUTPUT_JSON}" \
    --create-dataset "${CREATE_DATASET}"

echo ""
echo "Done. Next:"
echo "  bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh"
echo "  # after sweep merges: router feature extract + deploy on preview series"
