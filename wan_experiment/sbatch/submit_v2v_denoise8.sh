#!/bin/bash
# In-chunk denoise hooks. Caption N=8. Same-wave twins.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   SMOKE=1 WAVE=lastmix bash wan_experiment/sbatch/submit_v2v_denoise8.sh
#   WAVE=lastmix bash wan_experiment/sbatch/submit_v2v_denoise8.sh
#   WAVE=bpseudo bash wan_experiment/sbatch/submit_v2v_denoise8.sh
#   WAVE=restep  bash wan_experiment/sbatch/submit_v2v_denoise8.sh
#   WAVE=all     bash wan_experiment/sbatch/submit_v2v_denoise8.sh
#
# lastmix  — last DMD step 0.5-mix with step-3 if appear-punch / always
# bpseudo  — hide last committed block; extra seed writes the next if it wins
# restep   — redo last 2 of 4 DMD steps if appear-punch / always
#
# lastmix is NO (2026-08-29). Paste WAVE=bpseudo and WAVE=restep.
# Do not relaunch RF intra. Do not scancel 16545806.
# Thresholds pre-registered. Do not retune 1.5×. VIDEO_WORKERS=1.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_caption_denoise_8v}"
N_VIDEOS="${N_VIDEOS:-8}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    N_VIDEOS=2
    SERIES="${SERIES}_smoke"
fi
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
ALWAYS_WALL="${ALWAYS_WALL:-12:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"
SF_CAP="${SF_CAP:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/notta_h30s_shard0}"

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

WAVE="${WAVE:-bpseudo}"
if [[ "${WAVE}" == "lastmix" ]]; then
    submit_method sf_lastmix 4 "${SEARCH_WALL}"
    submit_method sf_lastmix_always 4 "${ALWAYS_WALL}"
    submit_method rf_lastmix 4 "${SEARCH_WALL}"
    submit_method rf_lastmix_always 4 "${ALWAYS_WALL}"
elif [[ "${WAVE}" == "bpseudo" ]]; then
    submit_method sf_bpseudo 4 "${SEARCH_WALL}"
    submit_method sf_bpseudo_always 4 "${ALWAYS_WALL}"
    submit_method rf_bpseudo 4 "${SEARCH_WALL}"
elif [[ "${WAVE}" == "restep" ]]; then
    submit_method sf_restep 4 "${SEARCH_WALL}"
    submit_method sf_restep_always 4 "${ALWAYS_WALL}"
    submit_method rf_restep 4 "${SEARCH_WALL}"
    submit_method rf_restep_always 4 "${ALWAYS_WALL}"
elif [[ "${WAVE}" == "all" ]]; then
    submit_method sf_lastmix 4 "${SEARCH_WALL}"
    submit_method sf_lastmix_always 4 "${ALWAYS_WALL}"
    submit_method rf_lastmix 4 "${SEARCH_WALL}"
    submit_method rf_lastmix_always 4 "${ALWAYS_WALL}"
    submit_method sf_bpseudo 4 "${SEARCH_WALL}"
    submit_method sf_bpseudo_always 4 "${ALWAYS_WALL}"
    submit_method rf_bpseudo 4 "${SEARCH_WALL}"
    submit_method sf_restep 4 "${SEARCH_WALL}"
    submit_method sf_restep_always 4 "${ALWAYS_WALL}"
    submit_method rf_restep 4 "${SEARCH_WALL}"
    submit_method rf_restep_always 4 "${ALWAYS_WALL}"
else
    echo "ERROR: WAVE=${WAVE} (use lastmix|bpseudo|restep|all)" >&2
    exit 1
fi

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS=""
for m in "${METHODS_RUN[@]}"; do
    VIDEO_DIRS="${VIDEO_DIRS} ${ROOT}/${m}_h30s_shard0"
done
VIDEO_DIRS="${VIDEO_DIRS} ${SF_CAP}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "Denoise hooks WAVE=${WAVE} N=${N_VIDEOS} caption. Appear punch 1.5×."
echo "Cite vs caption SF notta. Same-wave: gated + always-on + RF host."
echo "No rf_bpseudo_always (that is rf_intra_always, already NO)."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v \\"
echo "    --allow-partial"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
