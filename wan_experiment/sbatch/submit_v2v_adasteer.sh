#!/bin/bash
# AdaSteer confirmation on Wan SF V2V. Caption-conditioned. N=8 first.
# Three update rules that actually differ (not the LongCat knob museum).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_adasteer.sh
#   N_VIDEOS=32 bash wan_experiment/sbatch/submit_v2v_adasteer.sh   # after N=8
#
# Cite vs caption notta (WAVE=1 16310318) when that dir exists.
# Queues behind caption WAVE=1 QOS. No TTC. No I2V.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
N_VIDEOS="${N_VIDEOS:-8}"
SERIES="${SERIES:-v2v_panda_adasteer_${N_VIDEOS}v}"
GEN_WALL="${GEN_WALL:-04:00:00}"
ADA_WALL="${ADA_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"
ADA_STEPS="${ADA_STEPS:-10}"
ADA_LR="${ADA_LR:-5e-3}"
ADA_BLEND="${ADA_BLEND:-0.5}"
ADA_REFIT_STEPS="${ADA_REFIT_STEPS:-5}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/notta_h30s_shard0}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -f "${VIDEO_DIR}/metadata.csv" ]]; then
    echo "ERROR: ${VIDEO_DIR}/metadata.csv missing." >&2
    exit 1
fi

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},VIDEO_WORKERS=1,ADA_STEPS=${ADA_STEPS},ADA_LR=${ADA_LR},ADA_BLEND=${ADA_BLEND},ADA_REFIT_STEPS=${ADA_REFIT_STEPS}"

JOBS=()
submit_method() {
    local method="$1"
    local wall="$2"
    local J
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${wall}" \
        --export=ALL,METHOD="${method}",SEARCH_K=1,${COMMON} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${SERIES} ${method} n=${N_VIDEOS} job ${J}"
    JOBS+=("${J}")
}

submit_method ada_fixed "${ADA_WALL}"
submit_method ada_stream "${ADA_WALL}"
submit_method ada_resid "${ADA_WALL}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/ada_fixed_h30s_shard0 ${ROOT}/ada_stream_h30s_shard0 ${ROOT}/ada_resid_h30s_shard0"
if [[ -d "${NOTTA_DIR}" ]]; then
    VIDEO_DIRS="${VIDEO_DIRS} ${NOTTA_DIR}"
fi
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterok:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench ${SERIES} job ${VB} afterok ${DEPS}"
JOBS+=("${VB}")

echo "AdaSteer N=${N_VIDEOS}. S=${ADA_STEPS} LR=${ADA_LR} blend=${ADA_BLEND}."
echo "ada_fixed · ada_stream · ada_resid. Caption metadata.csv."
echo "Cite vs caption notta, not stem notta. No TTC. No I2V."
echo "Cancel this wave only:  scancel ${JOBS[*]}"
