#!/bin/bash
# One wave: Family A/B/D N=32 on RF + Family C sink probe N=32.
# Offline chunk-trace (all N) should run on the login node in the same paste.
# Paper baseline stays SF notta. Ablation zero is rolling_notta.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   python3 -u wan_experiment/scripts/resim_v2v_rf_chunk_trace.py --only all
#   bash wan_experiment/sbatch/submit_v2v_family_wave.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_family_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
GEN_WALL="${GEN_WALL:-08:00:00}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"
ROLL_BASE="${ROLL_BASE:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v/rolling_notta_h30s_shard0}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"

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
submit_method() {
    local method="$1"
    local k="$2"
    local wall="$3"
    local J
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${wall}" \
        --export=ALL,METHOD="${method}",SEARCH_K="${k}",${COMMON} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${SERIES} ${method} k=${k} n=${N_VIDEOS} job ${J}"
    JOBS+=("${J}")
}

submit_method rf_rewind 1 "${GEN_WALL}"
submit_method rf_sick_search 4 "${SEARCH_WALL}"
submit_method rf_pseudo 4 "${SEARCH_WALL}"
submit_method rf_sink 1 "${GEN_WALL}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/rf_rewind_h30s_shard0 ${ROOT}/rf_sick_search_h30s_shard0 ${ROOT}/rf_pseudo_h30s_shard0 ${ROOT}/rf_sink_h30s_shard0 ${ROLL_BASE} ${NOTTA_DIR}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "Family wave N=32 on RF. 2-way H200: these queue."
echo "A rewind k=1 · B sick_search k=4 · D pseudo k=4 · C sink k=1 (HG-f is not this)."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --allow-partial"
echo "  python3 -u wan_experiment/scripts/pair_v2v_tails.py \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v \\"
echo "    --series-dir ${ROOT}"
echo "Cite vs SF notta AND vs rolling_notta. Cancel this wave only:  scancel ${JOBS[*]}"
