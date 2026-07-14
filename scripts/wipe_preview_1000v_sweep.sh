#!/usr/bin/env bash
# Remove chunk artifacts so a full 12-config resubmit starts clean.
#
# Use when per-run PSNR looks OK but cross-config intersection is low (mixed-era
# chunks from symlink/missing-video failures).
#
#   bash scripts/wipe_preview_1000v_sweep.sh          # dry-run
#   CONFIRM=1 bash scripts/wipe_preview_1000v_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/preview_1000v_env.sh
source "${SCRIPT_DIR}/preview_1000v_env.sh"

CONFIRM="${CONFIRM:-0}"
ONLY_RUNS="${ONLY_RUNS:-}"

_in_filter() {
    local needle="$1"
    [ -z "${ONLY_RUNS}" ] && return 0
    for m in ${ONLY_RUNS}; do
        if [ "${m}" = "${needle}" ]; then return 0; fi
    done
    return 1
}

RUNS=(
    S2_LR1e3 S2_LR5e3 S2_LR1e2
    S5_LR1e3 S5_LR5e3 S5_LR1e2
    S10_LR1e3 S10_LR5e3 S10_LR1e2
    S20_LR1e3 S20_LR5e3 S20_LR1e2
)

echo "Series: ${PREVIEW_SERIES_ROOT}"
echo "CONFIRM=${CONFIRM}  ONLY_RUNS=${ONLY_RUNS:-<all>}"
echo ""

removed=0
for run_id in "${RUNS[@]}"; do
    if ! _in_filter "${run_id}"; then continue; fi
    run_dir="${PREVIEW_SERIES_ROOT}/${run_id}"
    [ -d "${run_dir}" ] || continue

    if [ -f "${run_dir}/merged_summary.json" ]; then
        echo "  rm ${run_dir}/merged_summary.json"
        [ "${CONFIRM}" = "1" ] && rm -f "${run_dir}/merged_summary.json"
        removed=$((removed + 1))
    fi

    for chunk_dir in "${run_dir}"/chunk_*; do
        [ -d "${chunk_dir}" ] || continue
        for f in summary.json checkpoint.json fvd_checkpoint.npz; do
            if [ -f "${chunk_dir}/${f}" ]; then
                echo "  rm ${chunk_dir}/${f}"
                [ "${CONFIRM}" = "1" ] && rm -f "${chunk_dir}/${f}"
                removed=$((removed + 1))
            fi
        done
    done
done

if [ "${CONFIRM}" != "1" ]; then
    echo ""
    echo "[dry-run] Would remove ${removed} artifact files. Re-run with CONFIRM=1 to apply."
    echo "Then: bash scripts/run_preview_1000v_pipeline.sh sweep"
else
    echo ""
    echo "Removed ${removed} artifact files."
    echo "Next: bash scripts/run_preview_1000v_pipeline.sh sweep"
fi
