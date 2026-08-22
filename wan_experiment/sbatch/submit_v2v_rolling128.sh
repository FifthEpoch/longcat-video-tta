#!/bin/bash
# Scale the one N=32 host that passed the locked bars.
#   notta         Self-Forcing, first 128 of panda_1000_480p
#   rolling_notta Rolling Forcing, same 128 (prefix of the N=32 set)
# No search. No appear. No live_min retune. No TTC.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_rolling128.sh
#
# 2-way H200: both generate jobs run together (~2 h). VBench afterany.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_rolling_128v}"
N_VIDEOS="${N_VIDEOS:-128}"
GEN_WALL="${GEN_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing." >&2
    exit 1
fi
if [[ ! -f /scratch/${USER}/wan-checkpoints/rolling_forcing_dmd.pt ]]; then
    echo "ERROR: Rolling Forcing ckpt missing." >&2
    exit 1
fi

JOBS=()
J=$(sbatch --parsable --account="${ACCOUNT}" --time="${GEN_WALL}" \
    --export=ALL,METHOD=notta,SEARCH_K=1,${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${SERIES} notta n=${N_VIDEOS} job ${J}"
JOBS+=("${J}")

J=$(sbatch --parsable --account="${ACCOUNT}" --time="${GEN_WALL}" \
    --export=ALL,METHOD=rolling_notta,SEARCH_K=1,${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${SERIES} rolling_notta n=${N_VIDEOS} job ${J}"
JOBS+=("${J}")

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",METHODS="notta rolling_notta",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "rolling-128: SF notta + RF rolling_notta. Fresh pair, not confirm_32v."
echo "When generate finishes (sidecars):"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} --allow-partial"
echo "  python3 -u wan_experiment/scripts/pair_v2v_tails.py \\"
echo "    --baseline-dir ${ROOT} --series-dir ${ROOT}"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
