#!/bin/bash
# ============================================================================
# Submitter for the long-horizon NOTTA drift diagnostic.
#
# Decisive cheap test from the 2026-08-06 problem-difficulty memo: does LongCat
# degrade over a long autoregressive rollout? Run it, then plot:
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash delta_experiment/sbatch/submit_longhorizon_drift.sh
#   # after it finishes:
#   python scripts/plot_drift_curves.py \
#       --summary sweep_experiment/results/diag_longhorizon_drift/summary.json \
#       --out-dir sweep_experiment/results/diag_longhorizon_drift/plots
#
# Overridable via env, e.g.  NUM_VIDEOS=40 NUM_CHUNKS=10 bash <this>
# ============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SBATCH="${PROJECT_ROOT}/delta_experiment/sbatch/run_longhorizon_drift.sbatch"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

NUM_VIDEOS="${NUM_VIDEOS:-24}"
NUM_CHUNKS="${NUM_CHUNKS:-8}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
SEED="${SEED:-42}"
SERIES="${SERIES:-diag_longhorizon_drift}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/results/${SERIES}}"

echo "============================================================"
echo "Long-horizon drift diagnostic (NOTTA)"
echo "  account : ${ACCOUNT}"
echo "  series  : ${SERIES}"
echo "  data    : ${DATA_DIR}"
echo "  N=${NUM_VIDEOS}  chunks=${NUM_CHUNKS}  seed=${SEED}"
echo "  geometry: cond=${NUM_COND_FRAMES} frames=${NUM_FRAMES} gsf=${GEN_START_FRAME}"
echo "  output  : ${OUTPUT_DIR}"
echo "============================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 -> not submitting."
  exit 0
fi

jid=$(sbatch --parsable --account="${ACCOUNT}" \
  --export=ALL,"NUM_VIDEOS=${NUM_VIDEOS},NUM_CHUNKS=${NUM_CHUNKS},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},SEED=${SEED},DATA_DIR=${DATA_DIR},CHECKPOINT_DIR=${CHECKPOINT_DIR},OUTPUT_DIR=${OUTPUT_DIR}" \
  "${SBATCH}")
echo "submitted job ${jid}  -> ${OUTPUT_DIR}/summary.json"
echo ""
echo "After it finishes, build drift-curve PNGs:"
echo "  python scripts/plot_drift_curves.py --summary ${OUTPUT_DIR}/summary.json --out-dir ${OUTPUT_DIR}/plots"
