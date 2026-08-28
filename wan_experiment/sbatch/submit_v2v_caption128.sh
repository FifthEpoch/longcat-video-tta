#!/bin/bash
# Caption V2V paper-size N=128. Same metadata.csv protocol as N=32.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   WAVE=hosts bash wan_experiment/sbatch/submit_v2v_caption128.sh
#   WAVE=cite  bash wan_experiment/sbatch/submit_v2v_caption128.sh
#
# WAVE=hosts — Self Forcing + Rolling Forcing do-nothing (baselines).
# WAVE=cite  — Pseudo + Always only. Reuses hosts notta. Do not resubmit notta.
# WAVE=all   — hosts + cite in one paste (one notta).
#
# First 128 of panda_1000_480p = prefix of the caption-32 set.
# Do not mix with stem `v2v_panda_rolling_128v`.
# Do not scancel lastmix 16505827–837. No TTC. No I2V. VIDEO_WORKERS=1.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_caption_128v}"
N_VIDEOS="${N_VIDEOS:-128}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    N_VIDEOS=2
    SERIES="${SERIES}_smoke"
fi
HOST_WALL="${HOST_WALL:-16:00:00}"
SEARCH_WALL="${SEARCH_WALL:-18:00:00}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},VIDEO_WORKERS=1,PSEUDO_GAMMA=0.0"

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

WAVE="${WAVE:-hosts}"
if [[ "${WAVE}" == "hosts" ]]; then
    submit_method notta 1 "${HOST_WALL}"
    submit_method rolling_notta 1 "${HOST_WALL}"
elif [[ "${WAVE}" == "cite" ]]; then
    submit_method sf_pseudo 4 "${SEARCH_WALL}"
    submit_method sf_always_search 4 "${SEARCH_WALL}"
elif [[ "${WAVE}" == "all" ]]; then
    submit_method notta 1 "${HOST_WALL}"
    submit_method rolling_notta 1 "${HOST_WALL}"
    submit_method sf_pseudo 4 "${SEARCH_WALL}"
    submit_method sf_always_search 4 "${SEARCH_WALL}"
else
    echo "ERROR: WAVE=${WAVE} (use hosts|cite|all)" >&2
    exit 1
fi

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS=""
for m in "${METHODS_RUN[@]}"; do
    VIDEO_DIRS="${VIDEO_DIRS} ${ROOT}/${m}_h30s_shard0"
done
if [[ "${WAVE}" == "cite" ]]; then
    VIDEO_DIRS="${VIDEO_DIRS} ${ROOT}/notta_h30s_shard0"
fi
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "Caption N=${N_VIDEOS} WAVE=${WAVE}. Sidecar must be prompt_source=metadata_csv."
echo "Cite vs this series notta, not stem rolling-128."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} --allow-partial"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
echo "Do not scancel lastmix 16505827-837."
