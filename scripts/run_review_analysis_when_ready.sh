#!/usr/bin/env bash
# CPU analysis to run after review GPU sweeps merge (login node OK).
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
BASE="${REPO}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}"

cd "${REPO}"

echo "=== Merge + VBench refresh (standard + review series) ==="
for series in \
    sweep_experiment/results/panda_1000v_standard/LORA_R1_TTA \
    sweep_experiment/results/panda_1000v_adasteer_budget_vbench; do
    if [ -d "${REPO}/${series}" ]; then
        python3 sweep_experiment/scripts/merge_chunks.py --results-dir "${REPO}/${series}" --recursive || true
        for d in "${REPO}/${series}"/*/; do
            [ -d "$d" ] && python3 scripts/update_merged_with_vbench.py --method-dir "$d" 2>/dev/null || true
        done
    fi
done

echo ""
echo "=== Refresh 999v VBench agreement (includes LORA_R1 if merged) ==="
DATE_TAG="${DATE_TAG}" bash scripts/run_panda_vbench_agreement.sh || true

echo ""
echo "=== Budget VBench oracle @ 999v (if series complete) ==="
if [ -d "${REPO}/sweep_experiment/results/panda_1000v_adasteer_budget_vbench" ]; then
    python3 scripts/analyze_adasteer_budget_vbench_oracle.py --bootstrap \
        --series-root "${REPO}/sweep_experiment/results/panda_1000v_adasteer_budget_vbench" \
        --ood-csv "${REPO}/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv" \
        --output "${BASE}/adasteer_budget_vbench_1000v.md"
fi

echo ""
echo "=== Router (use pilot features if DATE_TAG=${FEATURE_DATE} ready) ==="
FEATURE_DATE="${FEATURE_DATE}" DATE_TAG="${DATE_TAG}" \
    FEAT="${REPO}/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}" \
    bash -c '
if [ -f "$FEAT/video_features.csv" ]; then
  export FEATURE_DATE
  bash scripts/run_vbench_headroom_router.sh
else
  echo "SKIP router — pilot features not at $FEAT"
fi
'

echo ""
echo "Done. Key outputs under ${BASE}/"
