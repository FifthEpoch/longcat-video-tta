#!/bin/bash
# Submit 200-video combined-tricks AdaSteer sweeps for Panda-70M and UCF-101.
#
# Sweeps S10 / S20 x AREG x retrieval, holding the dataset-specific best
# anchor_reg_noise_draws (Panda=1, UCF=2) and anchor_reg_weight=0.2 fixed.
#
# Defaults to DRY_RUN=1 so it is safe to run locally or on the cluster.
# To submit for real:
#   DRY_RUN=0 bash sweep_experiment/sbatch/submit_combined_tta_tricks_200.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-1}"
# S20 + AREG + retrieval is the most expensive run we have queued; allocate
# generously. AREG02 took ~85s/video so S20 + AREG + K=1 is ~170s * 200 ~= 9.5h.
TIME_STANDARD="${TIME_STANDARD:-18:00:00}"
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

run_sweep sweep_experiment/configs/panda_200_combined_tta_tricks.yaml \
  "${PROJECT_ROOT}/datasets/panda_200_480p"
run_sweep sweep_experiment/configs/ucf101_200_combined_tta_tricks.yaml \
  "${PROJECT_ROOT}/datasets/ucf101_200_480p"

echo "Done. Set DRY_RUN=0 to submit these jobs."
