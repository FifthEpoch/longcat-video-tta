#!/bin/bash
# Submit sampling-space ideas 1 / 5 / 3 on the same N=8 Panda prefix set.
# Does NOT scancel the lineage suite. 2-way H200 cap: these queue behind it.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_ideas.sh
#
# Methods:
#   appear_bon     idea 5 always-on appearance pick
#   live_appear    idea 5 + live prefix gate
#   pseudo_gate    idea 1: held-out B MAE gate, then two-sided seed_bon
#   pseudo_appear  idea 1 + idea 5
#   noise_probe    idea 3: k=1, log first-step residual U_t
#   noise_bon      idea 3: search iff cand0 U_t >= tau; appear pick
#
# No TTC. No hist_drop-32. No LongLive/RF in this wave.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_ideas_8v}"
N_VIDEOS="${N_VIDEOS:-8}"
LIVE_MIN="${LIVE_MIN:-0.012}"
PSEUDO_GAMMA="${PSEUDO_GAMMA:-0.0}"
NOISE_TAU="${NOISE_TAU:-0.04}"
NOTTA_WALL="${NOTTA_WALL:-04:00:00}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},LIVE_MIN=${LIVE_MIN},PSEUDO_GAMMA=${PSEUDO_GAMMA},NOISE_TAU=${NOISE_TAU}"

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
    local J
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${wall}" \
        --export=ALL,METHOD="${method}",SEARCH_K="${k}",${COMMON} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${SERIES} ${method} n=${N_VIDEOS} job ${J}"
    JOBS+=("${J}")
}

submit_method appear_bon 4 "${SEARCH_WALL}"
submit_method live_appear 4 "${SEARCH_WALL}"
submit_method pseudo_gate 4 "${SEARCH_WALL}"
submit_method pseudo_appear 4 "${SEARCH_WALL}"
submit_method noise_probe 1 "${NOTTA_WALL}"
submit_method noise_bon 4 "${SEARCH_WALL}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/appear_bon_h30s_shard0 ${ROOT}/live_appear_h30s_shard0 ${ROOT}/pseudo_gate_h30s_shard0 ${ROOT}/pseudo_appear_h30s_shard0 ${ROOT}/noise_probe_h30s_shard0 ${ROOT}/noise_bon_h30s_shard0 ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_bakeoff_8v/notta_h30s_shard0"

GEN_DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${GEN_DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${GEN_DEPS}"
JOBS+=("${VB}")

echo "Submitted ${#JOBS[@]} jobs: ${JOBS[*]}"
echo "2-way H200 cap: extras queue behind lineage. Do not scancel 16140808-816."
echo "When generate finishes (sidecars, do not trust unpaired summary):"
echo "  python wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_bakeoff_8v \\"
echo "    --allow-partial"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
