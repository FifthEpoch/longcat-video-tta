#!/bin/bash
# Resubmit only rf_chunk N=32 after the kv_cache1 alias fix, then VBench
# that dir (other host-split joined.json already exist and will skip).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_rf_chunk.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_host_split_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
GEN_WALL="${GEN_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

J=$(sbatch --parsable --account="${ACCOUNT}" --time="${GEN_WALL}" \
    --export=ALL,METHOD=rf_chunk,SEARCH_K=1,${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${SERIES} rf_chunk n=${N_VIDEOS} job ${J}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
OUT="${ROOT}/rf_chunk_h30s_shard0"
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${J}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${OUT}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench rf_chunk-only job ${VB} afterany ${J}"
echo "Cancel this wave only:  scancel ${J} ${VB}"
