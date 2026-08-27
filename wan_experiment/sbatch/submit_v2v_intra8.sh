#!/bin/bash
# Intra-chunk motion+appear probe. Caption N=8. Same-wave twins.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   SMOKE=1 bash wan_experiment/sbatch/submit_v2v_intra8.sh
#   bash wan_experiment/sbatch/submit_v2v_intra8.sh
#   WAVE=sf bash wan_experiment/sbatch/submit_v2v_intra8.sh   # SF only after KV fix
#
# sf_intra         — after each 3-latent block, resample if freeze OR
#                    sharp/color/sat punch vs prefix (1.5× / 0.8×)
# sf_intra_always  — k=4 every block (no gate)
# rf_intra         — RF 21-latent span rewind if motion OR appear sick
# rf_intra_always  — RF always try an alt seed on every span
#
# Thresholds pre-registered. Do not retune after N=8.
# No TTC. No I2V. No WAVE=2 leftovers. VIDEO_WORKERS=1.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_caption_intra_8v}"
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

WAVE="${WAVE:-all}"
if [[ "${WAVE}" == "sf" ]]; then
    submit_method sf_intra 4 "${SEARCH_WALL}"
    submit_method sf_intra_always 4 "${ALWAYS_WALL}"
elif [[ "${WAVE}" == "all" ]]; then
    submit_method sf_intra 4 "${SEARCH_WALL}"
    submit_method sf_intra_always 4 "${ALWAYS_WALL}"
    submit_method rf_intra 4 "${SEARCH_WALL}"
    submit_method rf_intra_always 4 "${ALWAYS_WALL}"
else
    echo "ERROR: WAVE=${WAVE} (use all|sf)" >&2
    exit 1
fi

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/sf_intra_h30s_shard0 ${ROOT}/sf_intra_always_h30s_shard0 ${ROOT}/rf_intra_h30s_shard0 ${ROOT}/rf_intra_always_h30s_shard0 ${SF_CAP}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "Intra-chunk N=${N_VIDEOS} caption. Thresholds 0.8 motion / 1.5 appear."
echo "Cite vs caption SF notta. Same-wave: gated + always-on + RF host."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v \\"
echo "    --allow-partial"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
