#!/bin/bash
# Family A/B/C/D on Self-Forcing native chunked sampler. N=32.
# Paper claim: method-on-SF vs SF notta. RF do-nothing is a comparison
# row only — not the host. Do NOT use the rolling sampler (H1 twitch).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_sf_family.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_sf_family_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
GEN_WALL="${GEN_WALL:-08:00:00}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"
ROLL_BASE="${ROLL_BASE:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v/rolling_notta_h30s_shard0}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"

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

submit_method sf_rewind 1 "${GEN_WALL}"
submit_method sf_sick_search 4 "${SEARCH_WALL}"
submit_method sf_pseudo 4 "${SEARCH_WALL}"
submit_method sf_sink 1 "${GEN_WALL}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/sf_rewind_h30s_shard0 ${ROOT}/sf_sick_search_h30s_shard0 ${ROOT}/sf_pseudo_h30s_shard0 ${ROOT}/sf_sink_h30s_shard0 ${NOTTA_DIR} ${ROLL_BASE}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterok:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterok ${DEPS}"
JOBS+=("${VB}")

echo "SF family N=32. Host = Self-Forcing chunked. VIDEO_WORKERS=1. VBench afterok L40S."
echo "A sf_rewind k=1 · B sf_sick_search k=4 · D sf_pseudo k=4 · C sf_sink k=1."
echo "Cite vs SF notta (the claim). RF rolling is comparison only. No sf_roll."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --allow-partial"
echo "  python3 -u wan_experiment/scripts/pair_v2v_tails.py \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v \\"
echo "    --series-dir ${ROOT}"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
