#!/bin/bash
# Submit paper-aligned retrieval-augmented batch-level TTA sweeps.
#
# Defaults to DRY_RUN=1 so it is safe to run locally or on the cluster.
# To submit for real:
#   DRY_RUN=0 bash sweep_experiment/sbatch/submit_batch_retrieval.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-1}"
TIME_STANDARD="${TIME_STANDARD:-24:00:00}"
TIME_LONG="${TIME_LONG:-24:00:00}"
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

PANDA="${PROJECT_ROOT}/datasets/panda_1000_480p"
UCF="${PROJECT_ROOT}/datasets/ucf101_100_480p"

run_sweep sweep_experiment/configs/panda_batch_retrieval_delta_a.yaml "${PANDA}" "${TIME_STANDARD}"
run_sweep sweep_experiment/configs/panda_batch_retrieval_lora.yaml "${PANDA}" "${TIME_STANDARD}"

run_sweep sweep_experiment/configs/panda_longctx_batch_retrieval_delta_a.yaml "${PANDA}" "${TIME_LONG}"
run_sweep sweep_experiment/configs/panda_longctx_batch_retrieval_lora.yaml "${PANDA}" "${TIME_LONG}"

run_sweep sweep_experiment/configs/ucf_longctx_batch_retrieval_delta_a.yaml "${UCF}" "${TIME_LONG}"
run_sweep sweep_experiment/configs/ucf_longctx_batch_retrieval_lora.yaml "${UCF}" "${TIME_LONG}"

echo "Done. Set DRY_RUN=0 to submit these jobs."
