#!/bin/bash
# N=8 Rolling Forcing leftovers that were parked until the host passed N=32.
#   rolling_rho_lo  idea 4, ρ=0.5 always (more early noise)
#   rolling_rho_hi  idea 4, ρ=2.0 always (cleaner near-future)
#   rolling_adapt   idea 4, ρ from prefix_motion
#   rolling_look    idea 6+7, k=4 lookahead + trust reject
# Baseline: lineage rolling_notta (same 8). Not SF notta.
# Do not scale from this table. No TTC. Do not retune live_min.
#
# STEM-PROMPT AUDIT ONLY. Do not use this for new generates.
# Caption replay:
#   bash wan_experiment/sbatch/submit_v2v_caption_leftovers.sh
#
#   FORCE_STEM=1 bash wan_experiment/sbatch/submit_v2v_rolling_leftovers.sh

set -euo pipefail
if [[ "${FORCE_STEM:-0}" != "1" ]]; then
    echo "ERROR: this script writes stem leftover dirs (pandas in the tail)." >&2
    echo "Use: bash wan_experiment/sbatch/submit_v2v_caption_leftovers.sh" >&2
    echo "Override only with FORCE_STEM=1 (audit)." >&2
    exit 2
fi

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_rolling_leftovers_8v}"
N_VIDEOS="${N_VIDEOS:-8}"
ROLL_WALL="${ROLL_WALL:-04:00:00}"
LOOK_WALL="${LOOK_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"
ROLL_BASE="${ROLL_BASE:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_lineage_8v/rolling_notta_h30s_shard0}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_bakeoff_8v/notta_h30s_shard0}"

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
    echo "V2V ${SERIES} ${method} n=${N_VIDEOS} job ${J}"
    JOBS+=("${J}")
}

submit_method rolling_rho_lo 1 "${ROLL_WALL}"
submit_method rolling_rho_hi 1 "${ROLL_WALL}"
submit_method rolling_adapt 1 "${ROLL_WALL}"
submit_method rolling_look 4 "${LOOK_WALL}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/rolling_rho_lo_h30s_shard0 ${ROOT}/rolling_rho_hi_h30s_shard0 ${ROOT}/rolling_adapt_h30s_shard0 ${ROOT}/rolling_look_h30s_shard0 ${ROLL_BASE} ${NOTTA_DIR}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "Leftovers N=8 vs lineage rolling_notta. 2-way H200: queues behind 128."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_bakeoff_8v \\"
echo "    --allow-partial"
echo "  python3 -u wan_experiment/scripts/pair_v2v_tails.py \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_bakeoff_8v \\"
echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_lineage_8v \\"
echo "    --series-dir ${ROOT}"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
