#!/bin/bash
# Run this on the cluster (e.g. torch-login-1) to investigate why exp3 or other
# sweep jobs left the queue quickly (failed or completed).
# Usage: bash sweep_experiment/scripts/investigate_failed_jobs.sh [job_id ...]
#   With no args: show sacct for recent tta_exp3* and tta_full* jobs and list their logs.
#   With job IDs: show sacct for those jobs and tail their .err logs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/sweep_experiment/slurm_log"
cd "$REPO_ROOT"

if [ $# -gt 0 ]; then
  echo "=== Sacct for specified job IDs ==="
  sacct -j "$(echo "$*" | tr ' ' ',')" --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End
  echo ""
  for j in "$@"; do
    # Job ID may have .0 suffix in sacct; log files use the integer part
    jid="${j%%.*}"
    found=0
    for f in "${LOG_DIR}"/*_"${jid}.err"; do
      if [ -f "$f" ]; then
        echo "=== Last 80 lines of $f ==="
        tail -80 "$f"
        echo ""
        found=1
      fi
    done
    if [ "$found" -eq 0 ]; then
      echo "No .err log found for job $jid in ${LOG_DIR}"
      ls -la "${LOG_DIR}"/ 2>/dev/null | head -20
    fi
  done
  exit 0
fi

echo "=== Recent sacct: tta_exp3*, tta_full*, tta_exp5* (last 24h) ==="
START_24H=""
if date -d '24 hours ago' +%Y-%m-%dT%H:%M &>/dev/null; then
  START_24H=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M)
elif date -v-24H +%Y-%m-%dT%H:%M &>/dev/null; then
  START_24H=$(date -v-24H +%Y-%m-%dT%H:%M)
fi
for name in tta_exp3 tta_full tta_exp5; do
  echo "--- ${name} ---"
  if [ -n "$START_24H" ]; then
    sacct --name="${name}" --starttime="${START_24H}" --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End 2>/dev/null | head -25
  else
    sacct --name="${name}" --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End 2>/dev/null | head -25
  fi
  echo ""
done

echo "=== Slurm log directory: ${LOG_DIR} ==="
if [ -d "$LOG_DIR" ]; then
  echo "Recent .out / .err (by mtime):"
  ls -lt "${LOG_DIR}"/ 2>/dev/null | head -25
  echo ""
  echo "Any tta_exp3*, tta_full*, or tta_exp5* logs:"
  ls "${LOG_DIR}"/tta_exp3* "${LOG_DIR}"/tta_full* "${LOG_DIR}"/tta_exp5* 2>/dev/null || true
else
  echo "Log directory not found."
fi

echo ""
echo "To inspect a specific job's error log, run:"
echo "  bash sweep_experiment/scripts/investigate_failed_jobs.sh <JOBID>"
echo "Example: bash sweep_experiment/scripts/investigate_failed_jobs.sh 2864549 2864551"
