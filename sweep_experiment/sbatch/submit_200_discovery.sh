#!/bin/bash
# Submit 200-video short/standard-horizon AdaSteer discovery sweeps.
#
# Defaults to DRY_RUN=1 so it is safe to run locally or on the cluster.
# To submit for real:
#   DRY_RUN=0 bash sweep_experiment/sbatch/submit_200_discovery.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-1}"
TIME_STANDARD="${TIME_STANDARD:-08:00:00}"
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

  echo "============================================================"
  echo "Config : ${config}"
  echo "Data   : ${data_dir}"
  echo "Time   : ${TIME_STANDARD}"
  echo "Dry-run: ${DRY_RUN}"
  echo "============================================================"
  "${PYTHON}" sweep_experiment/scripts/run_sweep.py \
    --config "${config}" \
    --data-dir "${data_dir}" \
    --account "${ACCOUNT}" \
    --time "${TIME_STANDARD}" \
    "${dry_flag[@]}"
}

PANDA="${PROJECT_ROOT}/datasets/panda_200_480p"
UCF="${PROJECT_ROOT}/datasets/ucf101_200_480p"

run_sweep sweep_experiment/configs/panda_200_adasteer_steps_lr.yaml "${PANDA}"
run_sweep sweep_experiment/configs/ucf101_200_adasteer_steps_lr.yaml "${UCF}"

echo "Done. Set DRY_RUN=0 to submit these jobs."
