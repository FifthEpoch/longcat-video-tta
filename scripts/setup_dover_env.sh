#!/usr/bin/env bash
# One-time DOVER install for probe-video scoring (track C).
#
#   cd /scratch/wc3013/longcat-video-tta
#   bash scripts/setup_dover_env.sh
#
# Sets DOVER_ROOT=/scratch/$USER/third_party/DOVER by default.
set -euo pipefail

DOVER_ROOT="${DOVER_ROOT:-/scratch/${USER}/third_party/DOVER}"
mkdir -p "$(dirname "${DOVER_ROOT}")"

if [ ! -d "${DOVER_ROOT}/.git" ]; then
  git clone --depth=1 https://github.com/QualityAssessment/DOVER.git "${DOVER_ROOT}"
fi

cd "${DOVER_ROOT}"
pip install -e . --quiet

mkdir -p pretrained_weights
if [ ! -f pretrained_weights/DOVER.pth ]; then
  wget -q -O pretrained_weights/DOVER.pth \
    https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
fi

echo "DOVER ready at ${DOVER_ROOT}"
echo "export DOVER_ROOT=${DOVER_ROOT}"
