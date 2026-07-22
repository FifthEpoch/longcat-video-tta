#!/usr/bin/env bash
# VBench backfill for 1000v OOD-preview budget grid (COMPUTE_VBENCH=0 during sweep).
#
# Prereqs:
#   - Sweep finished with saved mp4s (NO_SAVE_VIDEOS=0), e.g.:
#       NO_SAVE_VIDEOS=0 bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh
#   - bash scripts/setup_vbench_backfill_env.sh  (once)
#
# Submit:
#   bash sweep_experiment/sbatch/submit_budget_1000v_preview_vbench_backfill.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/preview_1000v_env.sh
source "${SCRIPT_DIR}/../../scripts/preview_1000v_env.sh"

PROJECT_ROOT="${REPO}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/${PREVIEW_SERIES}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-sweep_experiment/sbatch/run_vbench_backfill.sbatch}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_RUNS="${ONLY_RUNS:-}"

# GENEVAL=1 -> evaluate the GENERATED-ONLY clips (conditioning frames trimmed by
# scripts/make_geneval_clips.py) and write to a separate results dir so the old
# full-clip results remain for audit. This is the CORRECT window for VBench.
GENEVAL="${GENEVAL:-0}"
if [ "${GENEVAL}" = "1" ]; then
    VIDEOS_SUBDIR="${VIDEOS_SUBDIR:-videos_geneval}"
    OUT_SUBDIR="${OUT_SUBDIR:-vbench_results_geneval}"
else
    VIDEOS_SUBDIR="${VIDEOS_SUBDIR:-videos}"
    OUT_SUBDIR="${OUT_SUBDIR:-vbench_results}"
fi

ALL_DIMS="subject_consistency background_consistency aesthetic_quality motion_smoothness dynamic_degree imaging_quality temporal_flickering"

# NOTTA first so the baseline is ready for downstream vs-NOTTA analysis.
PREVIEW_RUNS=(
    NOTTA
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
for run_id in "${PREVIEW_RUNS[@]}"; do
    _in_filter "${run_id}" || continue
    METHOD_DIR="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${run_id}"
    if [ ! -d "${METHOD_DIR}" ]; then
        echo "WARN: missing ${METHOD_DIR}" >&2
        skipped=$((skipped + 1))
        continue
    fi
    n_mp4=$(find "${METHOD_DIR}" -path "*/${VIDEOS_SUBDIR}/*.mp4" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n_mp4}" = "0" ]; then
        echo "WARN: no mp4s under ${METHOD_DIR}/*/${VIDEOS_SUBDIR} — skip" >&2
        if [ "${GENEVAL}" = "1" ]; then
            echo "      (run scripts/make_geneval_clips.py --method-dir ${METHOD_DIR} first)" >&2
        fi
        skipped=$((skipped + 1))
        continue
    fi
    job_name="vb_prev1k_${run_id}"
    _exec sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --export="ALL,METHOD_DIR=${METHOD_DIR},DIMS=${ALL_DIMS},PROJECT_ROOT=${PROJECT_ROOT},VIDEOS_SUBDIR=${VIDEOS_SUBDIR},OUT_SUBDIR=${OUT_SUBDIR},FORCE=${FORCE:-0}" \
        "${SBATCH_SCRIPT}"
    count=$((count + 1))
done

echo ""
echo "Submitted ${count} VBench backfill jobs (${skipped} skipped)."
echo "After completion:"
echo "  bash sweep_experiment/sbatch/submit_deploy_router_1000v_preview.sh"
