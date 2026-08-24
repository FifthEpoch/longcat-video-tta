#!/bin/bash
# Same-wave always-search on both hosts. k=4 = family width and
# CachedSearch's BoN-4 budget on Wan 1.3B. Same motion+trust pick,
# no gate.
#
# If SF always is already in the queue, use submit_v2v_rf_always_search.sh
# instead (do not start a second SF job).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_always_search_wave.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SF_SERIES="${SF_SERIES:-v2v_panda_sf_always_32v}"
RF_SERIES="${RF_SERIES:-v2v_panda_rf_always_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"
ROLL_BASE="${ROLL_BASE:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v/rolling_notta_h30s_shard0}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"
PSEUDO_DIR="${PSEUDO_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_sf_family_32v/sf_pseudo_h30s_shard0}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},VIDEO_WORKERS=1"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -f /scratch/${USER}/wan-checkpoints/self_forcing_dmd.pt ]]; then
    echo "ERROR: Self-Forcing ckpt missing." >&2
    exit 1
fi
if [[ ! -f /scratch/${USER}/wan-checkpoints/rolling_forcing_dmd.pt ]]; then
    echo "ERROR: Rolling Forcing ckpt missing." >&2
    exit 1
fi

SF=$(sbatch --parsable --account="${ACCOUNT}" --time="${SEARCH_WALL}" \
    --export=ALL,METHOD=sf_always_search,SEARCH_K=4,SERIES="${SF_SERIES}",${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${SF_SERIES} sf_always_search k=4 n=${N_VIDEOS} job ${SF}"

RF=$(sbatch --parsable --account="${ACCOUNT}" --time="${SEARCH_WALL}" \
    --export=ALL,METHOD=rf_always_search,SEARCH_K=4,SERIES="${RF_SERIES}",${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${RF_SERIES} rf_always_search k=4 n=${N_VIDEOS} job ${RF}"

SF_ROOT="${PROJECT_ROOT}/wan_experiment/results/${SF_SERIES}"
RF_ROOT="${PROJECT_ROOT}/wan_experiment/results/${RF_SERIES}"
VIDEO_DIRS="${SF_ROOT}/sf_always_search_h30s_shard0 ${RF_ROOT}/rf_always_search_h30s_shard0 ${PSEUDO_DIR} ${NOTTA_DIR} ${ROLL_BASE}"
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterok:${SF}:${RF}" \
    --export=ALL,SERIES_DIR="${SF_ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterok ${SF}:${RF}"
echo "k=4 on both hosts. Cite vs SF notta, vs sf_pseudo, vs rolling_notta."
