#!/bin/bash
# Submit batched independent TTA throughput benchmarks.
#
# Safe default:
#   bash sweep_experiment/sbatch/submit_batched_tta_benchmark.sh
#
# Real submission:
#   DRY_RUN=0 bash sweep_experiment/sbatch/submit_batched_tta_benchmark.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-1}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/sweep_experiment/results/batched_tta_benchmark}"
SBATCH="${PROJECT_ROOT}/sweep_experiment/sbatch/run_batched_tta_benchmark.sbatch"
METHODS="${METHODS:-adasteer_batched,lora_serial,tinylora_serial}"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_ROOT}"

submit_profile() {
  local profile="$1"
  local data_dir="$2"
  local num_frames="$3"
  local gen_start="$4"
  local tta_total="$5"
  local tta_context="$6"
  local max_videos="$7"
  local batch_sizes="$8"
  local time_limit="$9"

  local output_dir="${RESULTS_ROOT}/${profile}"
  local export_vars
  export_vars="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_DIR=${data_dir},OUTPUT_DIR=${output_dir},METHODS=${METHODS},BATCH_SIZES=${batch_sizes},MAX_VIDEOS=${max_videos},MAX_GROUPS=${MAX_GROUPS:-1},NUM_COND_FRAMES=14,NUM_FRAMES=${num_frames},GEN_START_FRAME=${gen_start},TTA_TOTAL_FRAMES=${tta_total},TTA_CONTEXT_FRAMES=${tta_context},DELTA_STEPS=10,DELTA_LR=0.005,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn"

  echo "============================================================"
  echo "Profile    : ${profile}"
  echo "Data       : ${data_dir}"
  echo "Methods    : ${METHODS}"
  echo "Batch sizes: ${batch_sizes}"
  echo "Output     : ${output_dir}"
  echo "Dry-run    : ${DRY_RUN}"
  echo "============================================================"

  if [ "${DRY_RUN}" = "1" ]; then
    echo sbatch --account="${ACCOUNT}" --job-name="bench_${profile}" --time="${time_limit}" --export="${export_vars}" "${SBATCH}"
  else
    sbatch --account="${ACCOUNT}" --job-name="bench_${profile}" --time="${time_limit}" --export="${export_vars}" "${SBATCH}"
  fi
}

PANDA="${PROJECT_ROOT}/datasets/panda_1000_480p"
UCF="${PROJECT_ROOT}/datasets/ucf101_100_480p"

submit_profile "panda_standard" "${PANDA}" 28 48 48 14 "${MAX_VIDEOS_STANDARD:-16}" "${BATCH_SIZES_STANDARD:-1,2,4,8,16}" "${TIME_STANDARD:-04:00:00}"
submit_profile "panda_longctx" "${PANDA}" 93 14 14 14 "${MAX_VIDEOS_LONG:-8}" "${BATCH_SIZES_LONG:-1,2,4,8}" "${TIME_LONG:-04:00:00}"
submit_profile "ucf_longctx" "${UCF}" 61 14 14 14 "${MAX_VIDEOS_LONG:-8}" "${BATCH_SIZES_LONG:-1,2,4,8}" "${TIME_LONG:-04:00:00}"

echo "Done. Set DRY_RUN=0 to submit."
