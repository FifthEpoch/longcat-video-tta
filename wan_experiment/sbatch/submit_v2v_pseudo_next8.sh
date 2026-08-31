#!/bin/bash
# Cheapen + re-gate Pseudo-future Search. Caption N=8. Same-wave twins.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   SMOKE=1 bash wan_experiment/sbatch/submit_v2v_pseudo_next8.sh
#   bash wan_experiment/sbatch/submit_v2v_pseudo_next8.sh
#   WAVE=cached bash wan_experiment/sbatch/submit_v2v_pseudo_next8.sh
#   WAVE=repseudo bash wan_experiment/sbatch/submit_v2v_pseudo_next8.sh
#
# γ stays 0.0. k stays 4. Pick stays motion+trust.
# Do not remake caption-32 / caption-128 videos.
# Do not scancel 16674378 (Always 128 VBench) or 16678705 (pixel).
# No TTC. No I2V. VIDEO_WORKERS=1.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_caption_pseudo_next_8v}"
N_VIDEOS="${N_VIDEOS:-8}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    N_VIDEOS=2
    SERIES="${SERIES}_smoke"
fi
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
ALWAYS_WALL="${ALWAYS_WALL:-10:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"
SF_CAP="${SF_CAP:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/notta_h30s_shard0}"
SF_PSEUDO="${SF_PSEUDO:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/sf_pseudo_h30s_shard0}"
SF_ALWAYS="${SF_ALWAYS:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/sf_always_search_h30s_shard0}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},VIDEO_WORKERS=1"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing." >&2
    exit 1
fi
if [[ ! -f "${VIDEO_DIR}/metadata.csv" ]]; then
    echo "ERROR: ${VIDEO_DIR}/metadata.csv missing." >&2
    exit 1
fi

JOBS=()
METHODS_RUN=()
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
    METHODS_RUN+=("${method}")
}

WAVE="${WAVE:-all}"
if [[ "${WAVE}" == "cached" || "${WAVE}" == "all" ]]; then
    submit_method sf_pseudo_cached 4 "${SEARCH_WALL}"
    submit_method sf_always_cached 4 "${ALWAYS_WALL}"
fi
if [[ "${WAVE}" == "repseudo" || "${WAVE}" == "all" ]]; then
    submit_method sf_repseudo 4 "${SEARCH_WALL}"
    submit_method sf_repseudo_cached 4 "${SEARCH_WALL}"
fi
if [[ "${#JOBS[@]}" -eq 0 ]]; then
    echo "ERROR: WAVE=${WAVE} (use all|cached|repseudo)" >&2
    exit 1
fi

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS=""
for m in "${METHODS_RUN[@]}"; do
    VIDEO_DIRS="${VIDEO_DIRS} ${ROOT}/${m}_h30s_shard0"
done
VIDEO_DIRS="${VIDEO_DIRS} ${SF_CAP}"
if [[ -d "${SF_PSEUDO}" ]]; then
    VIDEO_DIRS="${VIDEO_DIRS} ${SF_PSEUDO}"
fi
if [[ -d "${SF_ALWAYS}" ]]; then
    VIDEO_DIRS="${VIDEO_DIRS} ${SF_ALWAYS}"
fi
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "Pseudo-next N=${N_VIDEOS} caption. γ=0 k=4. Cheapen + re-gate."
echo "Cite vs caption-32 SF notta / sf_pseudo / sf_always_search first ${N_VIDEOS}."
echo "Do not scancel 16674378 or 16678705."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v \\"
echo "    --allow-partial"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
