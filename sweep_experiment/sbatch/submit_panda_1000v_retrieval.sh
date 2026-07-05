#!/usr/bin/env bash
# Track B: Panda 1000v retrieval sweep (4 methods × 10 chunks = 40 GPU jobs).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash sweep_experiment/sbatch/submit_panda_1000v_retrieval.sh
#
# Optional: use segment pool if embeddings exist:
#   PANDA_POOL=$PWD/datasets/panda_segment_pool bash sweep_experiment/sbatch/submit_panda_1000v_retrieval.sh
#
# Dry-run:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_panda_1000v_retrieval.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
PANDA_POOL="${PANDA_POOL:-${PROJECT_ROOT}/datasets/panda_2048_480p}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

echo "========== Track B preflight =========="
EVAL="${PROJECT_ROOT}/datasets/panda_1000_480p"
if [ ! -d "${EVAL}" ]; then
  echo "ERROR: eval set missing: ${EVAL}" >&2
  exit 2
fi
if [ ! -d "${PANDA_POOL}" ]; then
  echo "ERROR: pool missing: ${PANDA_POOL}" >&2
  exit 2
fi
if [ ! -f "${PANDA_POOL}/caption_embeddings.npy" ]; then
  echo "WARN: ${PANDA_POOL}/caption_embeddings.npy missing."
  echo "  Submitting anyway (SIM will encode on-the-fly). Recommended:"
  echo "  sbatch --account=${ACCOUNT} --export=ALL,POOL_DIR=${PANDA_POOL} \\"
  echo "    delta_experiment/sbatch/precompute_pool_embeddings.sbatch"
fi
n_eval=$(find "${EVAL}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
echo "  eval clips : ${EVAL} (${n_eval} mp4 at top level; runner uses MAX_VIDEOS=1000)"
echo "  pool       : ${PANDA_POOL}"
echo ""

export ONLY_DATASET=panda
export PANDA_POOL
export PROJECT_ROOT
export ACCOUNT

bash sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh

echo ""
echo "Track B submitted. Monitor: squeue -u \$USER | grep t1kr_panda"
echo "Results: sweep_experiment/results/panda_1000v_retrieval/{K5_RAND,K10_RAND,K5_SIM,K10_SIM}/"
echo ""
echo "After all chunks complete (~3 days at 2-GPU cap):"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "    --results-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_retrieval --recursive"
echo "  python scripts/update_merged_with_vbench.py \\"
echo "    --series-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_retrieval --force"
