#!/bin/bash
# Create deterministic 200-video discovery subsets from existing 1000-video pools.
#
# Usage on cluster:
#   cd /scratch/wc3013/longcat-video-tta
#   bash datasets/create_discovery_subsets.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
PYTHON="${PYTHON:-python3}"
SEED="${SEED:-42}"
NUM_VIDEOS="${NUM_VIDEOS:-200}"

cd "${PROJECT_ROOT}"

echo "Creating Panda-70M ${NUM_VIDEOS}-video discovery subset..."
"${PYTHON}" datasets/create_video_subset.py \
  --src-dir "${PROJECT_ROOT}/datasets/panda_1000_480p" \
  --dst-dir "${PROJECT_ROOT}/datasets/panda_${NUM_VIDEOS}_480p" \
  --num-videos "${NUM_VIDEOS}" \
  --seed "${SEED}"

echo "Creating UCF-101 ${NUM_VIDEOS}-video discovery subset..."
"${PYTHON}" datasets/create_video_subset.py \
  --src-dir "${PROJECT_ROOT}/datasets/ucf101_1000_480p" \
  --dst-dir "${PROJECT_ROOT}/datasets/ucf101_${NUM_VIDEOS}_480p" \
  --num-videos "${NUM_VIDEOS}" \
  --seed "${SEED}" \
  --stratify-by category

echo "Done."
