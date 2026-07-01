#!/usr/bin/env bash
# Submit budget-pilot VBench backfill (GPU, parallel) + sliding-config analysis (CPU).
#
# Safe to run alongside adb_pilot mp4 jobs — backfill uses vbench-backfill env on
# H200; analysis is CPU-only and waits on backfill via Slurm dependency.
#
# Usage:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull --ff-only origin main
#   bash sweep_experiment/sbatch/submit_budget_vbench_sliding_analysis.sh
#
# Dry-run backfill only:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_budget_vbench_sliding_analysis.sh
#
# Skip backfill (analysis only, if VBench already complete):
#   SKIP_BACKFILL=1 bash sweep_experiment/sbatch/submit_budget_vbench_sliding_analysis.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BACKFILL="${SKIP_BACKFILL:-0}"
FORCE_BACKFILL="${FORCE_BACKFILL:-0}"
WAIT_MINUTES="${WAIT_MINUTES:-180}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

BACKFILL_JOBS=()

if [ "${SKIP_BACKFILL}" != "1" ]; then
  echo "========== Phase A: VBench backfill (incomplete configs only) =========="
  # Re-use pilot submitter logic inline so we can collect job IDs.
  RESULTS_SUBDIR="sweep_experiment/results/panda_ood_budget_pilot"
  SBATCH_SCRIPT="sweep_experiment/sbatch/run_vbench_backfill.sbatch"
  ALL_DIMS="subject_consistency background_consistency aesthetic_quality motion_smoothness dynamic_degree imaging_quality temporal_flickering"
  PILOT_RUNS=(
    S2_LR1e3 S2_LR5e3 S2_LR1e2
    S5_LR1e3 S5_LR5e3 S5_LR1e2
    S10_LR1e3 S10_LR5e3 S10_LR1e2
    S20_LR1e3 S20_LR5e3 S20_LR1e2
  )

  _needs_backfill() {
    local run_id="$1"
    local method_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${run_id}"
    [ -d "${method_dir}" ] || return 1
    local n_mp4
    n_mp4=$(find "${method_dir}" -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    [ "${n_mp4}" -gt 0 ] || return 1
    if [ "${FORCE_BACKFILL}" = "1" ]; then
      return 0
    fi
    # Complete when imaging_quality eval exists in every chunk that has mp4s.
    local n_chunks n_iq
    n_chunks=$(find "${method_dir}" -mindepth 1 -maxdepth 1 -type d -name 'chunk_*' 2>/dev/null | wc -l | tr -d ' ')
    n_iq=$(find "${method_dir}" -path '*/vbench_results/vbench_imaging_quality_eval_results.json' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n_chunks}" -gt 0 ] && [ "${n_iq}" -ge "${n_chunks}" ]; then
      return 1
    fi
    return 0
  }

  for run_id in "${PILOT_RUNS[@]}"; do
    if ! _needs_backfill "${run_id}"; then
      echo "  skip (complete): ${run_id}"
      continue
    fi
    METHOD_DIR="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${run_id}"
    job_name="vb_budget_${run_id}"
    if [ "${DRY_RUN}" = "1" ]; then
      echo "[DRY] sbatch --account=${ACCOUNT} --job-name=${job_name} ..."
    else
      jid=$(sbatch --parsable \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --export="ALL,METHOD_DIR=${METHOD_DIR},DIMS=${ALL_DIMS},PROJECT_ROOT=${PROJECT_ROOT},FORCE=${FORCE_BACKFILL}" \
        "${SBATCH_SCRIPT}")
      echo "  submitted ${run_id}: job ${jid}"
      BACKFILL_JOBS+=("${jid}")
    fi
  done
  echo "  backfill jobs submitted: ${#BACKFILL_JOBS[@]}"
else
  echo "========== Phase A: SKIP_BACKFILL=1 =========="
fi

echo ""
echo "========== Phase B: VBench sliding-config analysis (CPU) =========="
DEP_ARGS=()
if [ "${#BACKFILL_JOBS[@]}" -gt 0 ]; then
  dep="afterany:$(IFS=:; echo "${BACKFILL_JOBS[*]}")"
  DEP_ARGS=(--dependency="${dep}")
  echo "  dependency: ${dep} (analysis runs even if some backfill jobs fail)"
else
  echo "  no backfill dependency (immediate or dry-run)"
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "[DRY] sbatch ${DEP_ARGS[*]} --export=ALL,DATE_TAG=${DATE_TAG},WAIT_MINUTES=${WAIT_MINUTES} run_budget_vbench_sliding_analysis.sbatch"
else
  AN_JOB=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    "${DEP_ARGS[@]}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG},WAIT_MINUTES=${WAIT_MINUTES}" \
    sweep_experiment/sbatch/run_budget_vbench_sliding_analysis.sbatch)
  echo "  submitted analysis: job ${AN_JOB}"
  echo ""
  echo "Monitor:"
  echo "  tail -f sweep_experiment/slurm_log/vb_slide_an_${AN_JOB}.out"
  echo "  tail -f sweep_experiment/slurm_log/vbench_backfill_*.out"
  echo ""
  echo "Output:"
  echo "  sweep_experiment/reports/per_video_analysis/${DATE_TAG}/adasteer_budget_vbench_oracle_pilot.md"
fi
