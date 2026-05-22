#!/bin/bash
# Submit next-stage validation and retrieval-batch discovery sweeps.
#
# Defaults to DRY_RUN=1 so it is safe to run locally or on the cluster.
# To submit for real:
#   DRY_RUN=0 bash sweep_experiment/sbatch/submit_validation_and_retrieval.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-1}"
TIME_VALIDATION="${TIME_VALIDATION:-24:00:00}"
TIME_RETRIEVAL="${TIME_RETRIEVAL:-12:00:00}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
PYTHON="${PYTHON:-python3}"

cd "${PROJECT_ROOT}"

dry_flag=()
if [ "${DRY_RUN}" = "1" ]; then
  dry_flag=(--dry-run)
fi

run_sweep() {
  local config="$1"
  local data_dir="$2"
  local time_limit="$3"

  echo "============================================================"
  echo "Config : ${config}"
  echo "Data   : ${data_dir}"
  echo "Time   : ${time_limit}"
  echo "Dry-run: ${DRY_RUN}"
  echo "============================================================"
  "${PYTHON}" sweep_experiment/scripts/run_sweep.py \
    --config "${config}" \
    --data-dir "${data_dir}" \
    --account "${ACCOUNT}" \
    --time "${time_limit}" \
    "${dry_flag[@]}"
}

PANDA_1000="${PROJECT_ROOT}/datasets/panda_1000_480p"
PANDA_200="${PROJECT_ROOT}/datasets/panda_200_480p"
UCF_1000="${PROJECT_ROOT}/datasets/ucf101_1000_480p"
UCF_200="${PROJECT_ROOT}/datasets/ucf101_200_480p"

echo "Submitting 1000-video validation sweeps..."
run_sweep sweep_experiment/configs/panda_1000v_s10_lr005_validation.yaml \
  "${PANDA_1000}" "${TIME_VALIDATION}"
run_sweep sweep_experiment/configs/ucf101_1000v_s5_lr0025_validation.yaml \
  "${UCF_1000}" "${TIME_VALIDATION}"

echo "Submitting 200-video retrieval-batch discovery sweeps..."
run_sweep sweep_experiment/configs/panda_200_batch_retrieval_delta_a.yaml \
  "${PANDA_200}" "${TIME_RETRIEVAL}"
run_sweep sweep_experiment/configs/ucf101_200_batch_retrieval_delta_a.yaml \
  "${UCF_200}" "${TIME_RETRIEVAL}"

echo "Done. Set DRY_RUN=0 to submit these jobs."
