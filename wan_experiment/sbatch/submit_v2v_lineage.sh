#!/bin/bash
# Submit every remaining Self-Forcing-lineage V2V test at once.
# SF live_bon / live_hist start immediately. LongLive + Rolling Forcing
# wait on the CPU download. One VBench job waits on all generate.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_lineage.sh
#
# 2-way H200 cap: extras queue. No TTC. No hist_drop-32.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_lineage_8v}"
N_VIDEOS="${N_VIDEOS:-8}"
LIVE_MIN="${LIVE_MIN:-0.012}"
NOTTA_WALL="${NOTTA_WALL:-04:00:00}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
DL_WALL="${DL_WALL:-04:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},LIVE_MIN=${LIVE_MIN}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing." >&2
    exit 1
fi

JOBS=()

submit_method() {
    local method="$1"
    local k="$2"
    local wall="$3"
    local dep="${4:-}"
    local extra="${5:-}"
    local depflag=()
    if [[ -n "${dep}" ]]; then
        depflag=(--dependency="afterok:${dep}")
    fi
    local J
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${wall}" \
        "${depflag[@]}" \
        --export=ALL,METHOD="${method}",SEARCH_K="${k}",${COMMON}${extra} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${SERIES} ${method} n=${N_VIDEOS} job ${J}${dep:+ dep=${dep}}"
    JOBS+=("${J}")
}

# --- SF student, no download ---
submit_method live_bon 4 "${SEARCH_WALL}"
submit_method live_hist 4 "${SEARCH_WALL}"

# --- LongLive / Rolling Forcing weights ---
DL=$(sbatch --parsable --account="${ACCOUNT}" --time="${DL_WALL}" \
    "${SB}/download_lineage.sbatch")
echo "lineage download job ${DL}"
JOBS+=("${DL}")

submit_method longlive_notta 1 "${NOTTA_WALL}" "${DL}"
submit_method longlive_sink 1 "${NOTTA_WALL}" "${DL}"
submit_method longlive_prefix_sink 1 "${NOTTA_WALL}" "${DL}"
submit_method longlive_live_bon 4 "${SEARCH_WALL}" "${DL}"
submit_method rolling_notta 1 "${SEARCH_WALL}" "${DL}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/live_bon_h30s_shard0 ${ROOT}/live_hist_h30s_shard0 ${ROOT}/longlive_notta_h30s_shard0 ${ROOT}/longlive_sink_h30s_shard0 ${ROOT}/longlive_prefix_sink_h30s_shard0 ${ROOT}/longlive_live_bon_h30s_shard0 ${ROOT}/rolling_notta_h30s_shard0 ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_bakeoff_8v/notta_h30s_shard0"

GEN_DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${GEN_DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${GEN_DEPS}"
JOBS+=("${VB}")

echo "Submitted ${#JOBS[@]} jobs: ${JOBS[*]}"
echo "2-way H200 cap: extras queue. No TTC. No hist_drop-32."
echo "When generate finishes (sidecars, do not trust unpaired summary):"
echo "  python wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_bakeoff_8v \\"
echo "    --allow-partial"
echo "Cancel:  scancel ${JOBS[*]}"
