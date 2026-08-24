#!/bin/bash
# RF always-search only. Use this if SF always is already queued
# (do not start a second SF job). Set SF_JOB so VBench waits on both.
#
#   SF_JOB=<sf_always_jobid> bash wan_experiment/sbatch/submit_v2v_rf_always_search.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
RF_SERIES="${RF_SERIES:-v2v_panda_rf_always_32v}"
SF_SERIES="${SF_SERIES:-v2v_panda_sf_always_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"
ROLL_BASE="${ROLL_BASE:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v/rolling_notta_h30s_shard0}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"
PSEUDO_DIR="${PSEUDO_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_sf_family_32v/sf_pseudo_h30s_shard0}"
SF_JOB="${SF_JOB:-}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},VIDEO_WORKERS=1"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -f /scratch/${USER}/wan-checkpoints/rolling_forcing_dmd.pt ]]; then
    echo "ERROR: Rolling Forcing ckpt missing." >&2
    exit 1
fi

RF=$(sbatch --parsable --account="${ACCOUNT}" --time="${SEARCH_WALL}" \
    --export=ALL,METHOD=rf_always_search,SEARCH_K=4,SERIES="${RF_SERIES}",${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${RF_SERIES} rf_always_search k=4 n=${N_VIDEOS} job ${RF}"

SF_ROOT="${PROJECT_ROOT}/wan_experiment/results/${SF_SERIES}"
RF_ROOT="${PROJECT_ROOT}/wan_experiment/results/${RF_SERIES}"
VIDEO_DIRS="${SF_ROOT}/sf_always_search_h30s_shard0 ${RF_ROOT}/rf_always_search_h30s_shard0 ${PSEUDO_DIR} ${NOTTA_DIR} ${ROLL_BASE}"
DEP="${RF}"
if [[ -n "${SF_JOB}" ]]; then
    if squeue -j "${SF_JOB}" -h -o '%i' 2>/dev/null | grep -q .; then
        DEP="${RF}:${SF_JOB}"
    else
        echo "SF_JOB ${SF_JOB} not in squeue; VBench waits on RF only"
    fi
fi
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterok:${DEP}" \
    --export=ALL,SERIES_DIR="${RF_ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterok ${DEP}"
echo "Set SF_JOB if SF always is still running so VBench waits for both."
