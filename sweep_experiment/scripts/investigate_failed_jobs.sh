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

echo "=== Recent sacct: tta_exp3* and tta_full* (last 24h) ==="
sacct --name=tta_exp3 --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M) \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End 2>/dev/null || \
  sacct --name=tta_exp3 --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End | head -30
echo ""
sacct --name=tta_full --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M) \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End 2>/dev/null || \
  sacct --name=tta_full --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End | head -30

echo ""
echo "=== Slurm log directory: ${LOG_DIR} ==="
if [ -d "$LOG_DIR" ]; then
  echo "Recent .out / .err (by mtime):"
  ls -lt "${LOG_DIR}"/ 2>/dev/null | head -25
  echo ""
  echo "Any tta_exp3* or tta_full* logs:"
  ls "${LOG_DIR}"/tta_exp3* "${LOG_DIR}"/tta_full* 2>/dev/null || true
else
  echo "Log directory not found."
fi

echo ""
echo "To inspect a specific job's error log, run:"
echo "  bash sweep_experiment/scripts/investigate_failed_jobs.sh <JOBID>"
echo "Example: bash sweep_experiment/scripts/investigate_failed_jobs.sh 2864549 2864551"
