#!/bin/bash
# One-shot yes/no: live_bon N=32 vs existing confirm_32v notta.
# Do not retune live_min. Do not submit notta again. Do not scancel
# lineage 16140812–816 or ideas 16145125–131.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_live32.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_live_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
LIVE_MIN="${LIVE_MIN:-0.012}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},LIVE_MIN=${LIVE_MIN}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing." >&2
    exit 1
fi
if [[ ! -d "${NOTTA_DIR}" ]]; then
    echo "ERROR: confirm notta missing: ${NOTTA_DIR}" >&2
    exit 1
fi

J=$(sbatch --parsable --account="${ACCOUNT}" --time="${SEARCH_WALL}" \
    --export=ALL,METHOD=live_bon,SEARCH_K=4,${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${SERIES} live_bon n=${N_VIDEOS} job ${J}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/live_bon_h30s_shard0 ${NOTTA_DIR}"

VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${J}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${J}"

echo "YES/NO test: live_bon N=32 vs confirm notta. live_min=${LIVE_MIN}."
echo "Do not retune. Do not scancel lineage/ideas."
echo "When generate finishes (sidecars, not unpaired summary):"
echo "  python wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --allow-partial"
echo "Cancel this wave only:  scancel ${J} ${VB}"
