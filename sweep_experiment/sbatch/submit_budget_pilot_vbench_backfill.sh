#!/usr/bin/env bash
# VBench backfill for H9 budget pilot (all 7 dims — runs had COMPUTE_VBENCH=0).
#
# Prerequisites:
#   - Mp4s saved (NO_SAVE_VIDEOS=0 re-run finished): chunk_*/videos/*.mp4
#   - bash scripts/setup_vbench_backfill_env.sh  (once)
#
# Submit:
#   bash sweep_experiment/sbatch/submit_budget_pilot_vbench_backfill.sh
#
# After jobs finish:
#   bash scripts/run_budget_vbench_sliding_analysis.sh
# Or submit backfill + analysis together:
#   bash sweep_experiment/sbatch/submit_budget_vbench_sliding_analysis.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_ood_budget_pilot}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sweep_experiment/sbatch/run_vbench_backfill.sbatch}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_RUNS="${ONLY_RUNS:-}"

# Budget pilot had zero inline VBench — backfill all 7 dims.
ALL_DIMS="subject_consistency background_consistency aesthetic_quality motion_smoothness dynamic_degree imaging_quality temporal_flickering"

PILOT_RUNS=(
    S2_LR1e3 S2_LR5e3 S2_LR1e2
    S5_LR1e3 S5_LR5e3 S5_LR1e2
    S10_LR1e3 S10_LR5e3 S10_LR1e2
    S20_LR1e3 S20_LR5e3 S20_LR1e2
)

_in_filter() {
    local needle="$1"
    [ -z "${ONLY_RUNS}" ] && return 0
    for m in ${ONLY_RUNS}; do
        [ "${m}" = "${needle}" ] && return 0
    done
    return 1
}

_exec() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] $*"
    else
        "$@"
    fi
}

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

count=0
skipped=0
for run_id in "${PILOT_RUNS[@]}"; do
    _in_filter "${run_id}" || continue
    METHOD_DIR="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${run_id}"
    if [ ! -d "${METHOD_DIR}" ]; then
        echo "WARN: missing ${METHOD_DIR}" >&2
        skipped=$((skipped + 1))
        continue
    fi
    n_mp4=$(find "${METHOD_DIR}" -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n_mp4}" = "0" ]; then
        echo "WARN: no mp4s under ${METHOD_DIR} — skip (wait for NO_SAVE_VIDEOS=0 jobs)" >&2
        skipped=$((skipped + 1))
        continue
    fi
    job_name="vb_budget_${run_id}"
    _exec sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --export="ALL,METHOD_DIR=${METHOD_DIR},DIMS=${ALL_DIMS},PROJECT_ROOT=${PROJECT_ROOT}" \
        "${SBATCH_SCRIPT}"
    count=$((count + 1))
done

echo ""
echo "Submitted ${count} VBench backfill jobs (${skipped} skipped)."
echo "After completion: bash scripts/run_budget_pilot_vbench_oracle.sh"
