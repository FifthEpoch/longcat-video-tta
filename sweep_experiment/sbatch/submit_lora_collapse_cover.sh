#!/bin/bash
# Regenerate No-TTA / collapsed-LoRA / AdaSteer videos on the 30-video
# LoRA-collapse cover-image subset.
#
# Run AFTER scripts/build_lora_collapse_subset.py has created the
# datasets/ucf500_lora_collapse_candidates_480p subset dataset.
#
# Defaults to DRY_RUN=1. To submit for real:
#   DRY_RUN=0 bash sweep_experiment/sbatch/submit_lora_collapse_cover.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-1}"
# 30 videos, no FVD/FID computation; standard horizon. ~30 * 130s = ~65 min
# for AdaSteer; collapsed-LoRA at 50 steps is heavier. Allocate 6h to be safe.
TIME_STANDARD="${TIME_STANDARD:-06:00:00}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/ucf500_lora_collapse_candidates_480p}"

cd "${PROJECT_ROOT}"

if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: subset dataset missing at ${DATA_DIR}." >&2
  echo "Run scripts/build_lora_collapse_subset.py first." >&2
  exit 1
fi

dry_flag=()
if [ "${DRY_RUN}" = "1" ]; then
  dry_flag=(--dry-run)
fi

run_sweep() {
  local config="$1"

  echo "============================================================"
  echo "Config : ${config}"
  echo "Data   : ${DATA_DIR}"
  echo "Time   : ${TIME_STANDARD}"
  echo "Dry-run: ${DRY_RUN}"
  echo "============================================================"
  "${PYTHON}" sweep_experiment/scripts/run_sweep.py \
    --config "${config}" \
    --data-dir "${DATA_DIR}" \
    --account "${ACCOUNT}" \
    --time "${TIME_STANDARD}" \
    "${dry_flag[@]}"
}

run_sweep sweep_experiment/configs/ucf500_lora_collapse_cover_notta.yaml
run_sweep sweep_experiment/configs/ucf500_lora_collapse_cover_lora.yaml
run_sweep sweep_experiment/configs/ucf500_lora_collapse_cover_adasteer.yaml

echo "Done. Set DRY_RUN=0 to submit these jobs."
