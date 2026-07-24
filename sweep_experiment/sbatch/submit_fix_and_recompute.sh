#!/usr/bin/env bash
# ============================================================================
# One-command orchestrator: fix eval windows + recompute EVERYTHING as chained
# sbatch jobs. Submit once, then log out — SLURM dependencies run the pipeline
# unattended in the correct order.
#
# Job graph:
#
#   TRIM (array 0-12, CPU) ---> VBENCH (13 jobs, GPU) ---> FOLD (CPU)
#                                                              \
#   FVD  (matched, GPU) ----------------------------------------> ANALYSES (CPU)
#
#   - TRIM     : trim cond frames -> videos_geneval/ (self-healing, idempotent)
#   - VBENCH   : VBench on gen-only clips -> vbench_results_geneval/
#   - FOLD     : fold gen-only VBench into merged_summary (deprecate old)
#   - FVD      : matched offline FVD/FID (frozen cache, common N) — independent
#                of trim (uses videos/), so it runs in parallel
#   - ANALYSES : oracle + router matrix + chart JSON on corrected data
#
# Usage:
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_fix_and_recompute.sh
#
#   DRY_RUN=1 bash ...   # print the sbatch plan without submitting
#   SKIP_FVD=1 bash ...  # skip the matched-FVD leg (VBench pipeline only)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_FVD="${SKIP_FVD:-0}"
JOBID_FILE="${JOBID_FILE:-${PROJECT_ROOT}/sweep_experiment/slurm_log/vbench_geneval_jobids.txt}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

_run() {  # echo + (maybe) execute, capturing stdout
    if [ "${DRY_RUN}" = "1" ]; then echo "[DRY] $*" >&2; echo "DRYRUN_JOBID"; else "$@"; fi
}

echo "=============================================================="
echo "Fix + recompute orchestrator"
echo "  project_root : ${PROJECT_ROOT}"
echo "  account      : ${ACCOUNT}"
echo "  dry_run      : ${DRY_RUN}   skip_fvd: ${SKIP_FVD}"
echo "=============================================================="

# ---- 1. TRIM (array over 13 arms) -----------------------------------------
JID_TRIM=$(_run sbatch --parsable --account="${ACCOUNT}" \
    sweep_experiment/sbatch/run_geneval_trim.sbatch)
echo "[1] TRIM array   -> ${JID_TRIM}"

# ---- 2. VBENCH on gen-only clips (depends on TRIM) ------------------------
# submit script writes the colon-joined VBench job ids to JOBID_FILE.
if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY] GENEVAL=1 FORCE=1 DEP_AFTEROK=${JID_TRIM} bash sweep_experiment/sbatch/submit_budget_1000v_preview_vbench_backfill.sh" >&2
    VB_IDS="DRYRUN_VBIDS"
else
    GENEVAL=1 FORCE=1 DEP_AFTEROK="${JID_TRIM}" JOBID_FILE="${JOBID_FILE}" \
        bash sweep_experiment/sbatch/submit_budget_1000v_preview_vbench_backfill.sh
    VB_IDS="$(cat "${JOBID_FILE}")"
fi
echo "[2] VBENCH jobs  -> ${VB_IDS}"

# ---- 3. FOLD gen-only VBench into merged_summary (depends on all VBench) ---
JID_FOLD=$(_run sbatch --parsable --account="${ACCOUNT}" --dependency=afterok:"${VB_IDS}" \
    sweep_experiment/sbatch/run_vbench_geneval_fold.sbatch)
echo "[3] FOLD         -> ${JID_FOLD}  (afterok:${VB_IDS})"

# ---- 4. FVD matched recompute (independent; runs in parallel) -------------
DEP_ANALYSES="afterok:${JID_FOLD}"
if [ "${SKIP_FVD}" = "1" ]; then
    echo "[4] FVD          -> SKIPPED (SKIP_FVD=1)"
else
    JID_FVD=$(_run sbatch --parsable --account="${ACCOUNT}" \
        --export=ALL,INTERSECT_NOTTA=1,SKIP_GT_CACHE=1 \
        sweep_experiment/sbatch/run_preview_1000v_matched_fvd.sbatch)
    echo "[4] FVD matched  -> ${JID_FVD}  (INTERSECT_NOTTA=1)"
    DEP_ANALYSES="afterok:${JID_FOLD}:${JID_FVD}"
fi

# ---- 5. ANALYSES on corrected data (depends on FOLD [+ FVD]) --------------
JID_ANALYSES=$(_run sbatch --parsable --account="${ACCOUNT}" --dependency="${DEP_ANALYSES}" \
    sweep_experiment/sbatch/run_geneval_analyses.sbatch)
echo "[5] ANALYSES     -> ${JID_ANALYSES}  (dependency=${DEP_ANALYSES})"

echo ""
echo "Submitted. Monitor with:  squeue -u ${USER}"
echo "Final artifacts appear under sweep_experiment/reports/per_video_analysis/<today>/"
echo "When ANALYSES finishes, paste back chart_data_1000v_geneval.json + the"
echo "matched FVD summary for local chart rendering + table refresh."
