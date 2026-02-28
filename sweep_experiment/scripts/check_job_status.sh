#!/bin/bash
# =============================================================================
# Check status of all TTA/sweep jobs and result directories.
#
# Run this on the cluster (where results and SLURM are available):
#   cd /scratch/wc3013/longcat-video-tta   # or your PROJECT_ROOT
#   PROJECT_ROOT=$PWD bash sweep_experiment/scripts/check_job_status.sh
#
# Run locally (if you have synced results):
#   cd /path/to/LongCat-Video-Experiment
#   PROJECT_ROOT=$PWD bash sweep_experiment/scripts/check_job_status.sh
#
# Output: counts (complete / in-progress / failed), failed run list,
#         squeue/sacct summary, and next steps (export+figures vs investigate+resubmit).
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"

SWEEP_RESULTS="${PROJECT_ROOT}/sweep_experiment/results"
SWEEP_NO_PREEMPT="${PROJECT_ROOT}/sweep_experiment/results_no_preempt"
BASE_RESULTS="${PROJECT_ROOT}/baseline_experiment/results"
LOG_DIR="${PROJECT_ROOT}/sweep_experiment/slurm_log"

echo "================================================================================"
echo "  JOB & RESULTS STATUS — $(date)"
echo "  PROJECT_ROOT = ${PROJECT_ROOT}"
echo "================================================================================"

# -----------------------------------------------------------------------------
# 1. SLURM queue (if available)
# -----------------------------------------------------------------------------
echo ""
echo "=== CURRENT QUEUE (squeue) ==="
if command -v squeue &>/dev/null; then
  squeue -u "${USER}" 2>/dev/null | head -50
  N_QUEUED=$(squeue -u "${USER}" -h 2>/dev/null | wc -l)
  echo "(Total: ${N_QUEUED} jobs for ${USER})"
else
  echo "(squeue not available — run this script on the cluster)"
fi

# -----------------------------------------------------------------------------
# 2. Recent job exit status (sacct)
# -----------------------------------------------------------------------------
echo ""
echo "=== RECENT JOB EXITS (sacct, last 7 days) ==="
if command -v sacct &>/dev/null; then
  START="2020-01-01T00:00"
  if date -d '7 days ago' +%Y-%m-%dT%H:%M &>/dev/null; then
    START=$(date -d '7 days ago' +%Y-%m-%dT%H:%M)
  elif date -v-7d +%Y-%m-%dT%H:%M &>/dev/null; then
    START=$(date -v-7d +%Y-%m-%dT%H:%M)
  fi
  for name in tta_exp3 tta_exp4 tta_exp5 tta_adas tta_full tta_lora tta_delt opensora setup_co panda70m resize_v; do
    COUNT=$(sacct --name="${name}" --starttime="${START}" --format=JobID,State -n 2>/dev/null | wc -l) || COUNT=0
    if [ "${COUNT}" -gt 0 ]; then
      echo "--- ${name} ---"
      sacct --name="${name}" --starttime="${START}" --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End 2>/dev/null | head -25
    fi
  done
else
  echo "(sacct not available — run this script on the cluster)"
fi

# -----------------------------------------------------------------------------
# 3. Scan result directories: complete / in-progress / failed
# -----------------------------------------------------------------------------
echo ""
echo "=== RESULT DIRECTORIES ==="

complete=0
in_progress=0
failed_dirs=()
failed_series=()
# Track failed keys to avoid duplicates when scanning both results and results_no_preempt
declare -A failed_seen
# Avoid set -e triggering on arithmetic when value is 0
inc_complete() { complete=$((complete + 1)); }
inc_in_progress() { in_progress=$((in_progress + 1)); }

scan_dir() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  for series_dir in "$dir"/*; do
    [ -d "$series_dir" ] || continue
    local series_name="${series_dir##*/}"
    if [ -f "${series_dir}/summary.json" ]; then
      inc_complete
      continue
    fi
    for run_dir in "${series_dir}"/*; do
      [ -d "$run_dir" ] || continue
      local run_name="${run_dir##*/}"
      # Baseline runs use cond*_gen*/generated_videos for output; not a run dir
      [ "$run_name" = "generated_videos" ] && continue
      if [ -f "${run_dir}/summary.json" ]; then
        inc_complete
      elif [ -f "${run_dir}/checkpoint.json" ]; then
        inc_in_progress
      else
        local key="${series_name}/${run_name}"
        [ -n "${failed_seen[$key]:-}" ] && continue
        failed_seen[$key]=1
        failed_dirs+=("${run_dir}")
        failed_series+=("${series_name}/${run_name}")
      fi
    done
  done
}

scan_dir "${SWEEP_RESULTS}"
scan_dir "${SWEEP_NO_PREEMPT}"
scan_dir "${BASE_RESULTS}"

echo "  Complete (summary.json):    ${complete}"
echo "  In-progress (checkpoint):   ${in_progress}"
echo "  Failed/empty (no summary): ${#failed_dirs[@]}"

if [ ${#failed_dirs[@]} -gt 0 ]; then
  echo ""
  echo "  Failed or empty run dirs (candidates for investigate + resubmit):"
  for i in "${!failed_series[@]}"; do
    echo "    - ${failed_series[$i]}"
  done
fi

# -----------------------------------------------------------------------------
# 4. Recent SLURM logs (if present)
# -----------------------------------------------------------------------------
echo ""
echo "=== RECENT SLURM LOGS (sweep_experiment/slurm_log) ==="
if [ -d "${LOG_DIR}" ]; then
  ls -lt "${LOG_DIR}" 2>/dev/null | head -20
else
  echo "  (log directory not found)"
fi

# -----------------------------------------------------------------------------
# 5. Next steps
# -----------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "  NEXT STEPS"
echo "================================================================================"
echo ""
echo "1) For SUCCESSFUL runs (complete):"
echo "   - Export results and regenerate graphics:"
echo "     cd ${PROJECT_ROOT}"
echo "     PROJECT_ROOT=${PROJECT_ROOT} python3 sweep_experiment/scripts/export_all_results.py > all_results.json"
echo "     python3 paper_figures/generate_figures.py"
echo ""
echo "2) For FAILED or missing runs:"
echo "   - Inspect SLURM exit status and logs:"
echo "     bash sweep_experiment/scripts/investigate_failed_jobs.sh"
echo "   - For specific job IDs (from sacct above):"
echo "     bash sweep_experiment/scripts/investigate_failed_jobs.sh <JOBID> [JOBID2 ...]"
echo "   - Docs: sweep_experiment/docs/INVESTIGATE_EXP3_FAILURES.md (exp3/sweep)"
echo "           sweep_experiment/docs/INVESTIGATE_EXP5_UCF101_FAILURES.md (exp5 batch-size UCF-101)"
echo "   - After fixing (config/env/resources), resubmit with:"
echo "     python3 sweep_experiment/scripts/run_sweep.py --config <config>.yaml --account torch_pr_36_mren"
echo "     (Use --run-ids to resubmit only failed runs; see INVESTIGATE_EXP5_UCF101_FAILURES.md for exp5.)"
echo ""
echo "================================================================================"
