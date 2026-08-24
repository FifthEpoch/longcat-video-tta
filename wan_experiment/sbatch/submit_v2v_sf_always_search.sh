#!/bin/bash
# Ablation: always k=4 motion+trust on SF chunked. Same pick as
# sf_pseudo, no prefix gate. Splits gate vs pick on the +37% win.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_sf_always_search.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_sf_always_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"
ROLL_BASE="${ROLL_BASE:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v/rolling_notta_h30s_shard0}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"
PSEUDO_DIR="${PSEUDO_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_sf_family_32v/sf_pseudo_h30s_shard0}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},VIDEO_WORKERS=1"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing." >&2
    exit 1
fi
if [[ ! -f /scratch/${USER}/wan-checkpoints/self_forcing_dmd.pt ]]; then
    echo "ERROR: Self-Forcing ckpt missing." >&2
    exit 1
fi

J=$(sbatch --parsable --account="${ACCOUNT}" --time="${SEARCH_WALL}" \
    --export=ALL,METHOD=sf_always_search,SEARCH_K=4,${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${SERIES} sf_always_search k=4 n=${N_VIDEOS} job ${J}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/sf_always_search_h30s_shard0 ${PSEUDO_DIR} ${NOTTA_DIR} ${ROLL_BASE}"
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterok:${J}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterok ${J}"
echo "Host = SF chunked. Same pick as sf_pseudo. No prefix gate."
echo "Cite vs SF notta AND vs sf_pseudo. Do not scale."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_sf_family_dissect.py \\"
echo "    --family-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_sf_family_32v \\"
echo "    --also-dir ${ROOT} \\"
echo "    --notta-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --rolling-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v"
