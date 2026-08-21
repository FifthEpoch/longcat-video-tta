#!/bin/bash
# V2V sampling-space bake-off on Panda prefixes.
#   SMOKE=1    → N=2 NOTTA only
#   PROBE=1    → N=2 knob_probe (shift × cfg)
#   CONFIRM=1  → N=32 notta vs seed_bon only (N=8 promote confirm)
#   default    → N=8 wave-1 methods
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   CONFIRM=1 bash wan_experiment/sbatch/submit_v2v_bakeoff.sh
#
# No TTC. Do not scale I2V-32. 2-way H200 cap: extras queue.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"

SMOKE="${SMOKE:-0}"
PROBE="${PROBE:-0}"
CONFIRM="${CONFIRM:-0}"
SKIP_SHIFT="${SKIP_SHIFT:-0}"
SKIP_BACKTRACK="${SKIP_BACKTRACK:-0}"
NOTTA_WALL="${NOTTA_WALL:-04:00:00}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"

if [[ "${SMOKE}" == "1" ]]; then
    SERIES="${SERIES:-v2v_panda_smoke}"
    N_VIDEOS="${N_VIDEOS:-2}"
    METHODS=(notta)
    NOTTA_WALL="${WALL:-02:00:00}"
    SEARCH_WALL="${WALL:-02:00:00}"
elif [[ "${PROBE}" == "1" ]]; then
    SERIES="${SERIES:-v2v_panda_probe}"
    N_VIDEOS="${N_VIDEOS:-2}"
    METHODS=(knob_probe)
    NOTTA_WALL="${WALL:-01:00:00}"
    SEARCH_WALL="${WALL:-01:00:00}"
elif [[ "${CONFIRM}" == "1" ]]; then
    SERIES="${SERIES:-v2v_panda_confirm_32v}"
    N_VIDEOS="${N_VIDEOS:-32}"
    METHODS=(notta seed_bon)
    NOTTA_WALL="${NOTTA_WALL:-04:00:00}"
    SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
else
    SERIES="${SERIES:-v2v_panda_bakeoff_8v}"
    N_VIDEOS="${N_VIDEOS:-8}"
    METHODS=(notta seed_bon motion_bon)
    if [[ "${SKIP_SHIFT}" != "1" ]]; then
        METHODS+=(shift_search)
    fi
    if [[ "${SKIP_BACKTRACK}" != "1" ]]; then
        METHODS+=(backtrack)
    fi
    NOTTA_WALL="${WALL:-06:00:00}"
    SEARCH_WALL="${WALL:-06:00:00}"
fi

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing. Panda 1000 eval set should already be on cluster." >&2
    exit 1
fi

JOBS=()
for METHOD in "${METHODS[@]}"; do
    K=1
    T="${NOTTA_WALL}"
    case "${METHOD}" in
        seed_bon|motion_bon) K=4; T="${SEARCH_WALL}" ;;
        shift_search) K=3; T="${SEARCH_WALL}" ;;
        knob_probe) T="${SEARCH_WALL}" ;;
    esac
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${T}" \
        --export=ALL,METHOD="${METHOD}",SEARCH_K=${K},${COMMON} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${SERIES} ${METHOD} n=${N_VIDEOS} job ${J}"
    JOBS+=("${J}")
done

echo "Submitted ${#JOBS[@]} jobs: ${JOBS[*]}"
echo "2-way H200 cap: extras queue. No TTC."
echo "When they finish:"
if [[ "${PROBE}" == "1" ]]; then
    echo "  python wan_experiment/scripts/analyze_v2v_probe.py \\"
    echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
else
    echo "  python wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
    echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
    echo "  # then official VBench on each method dir (vbench-backfill env):"
    echo "  python wan_experiment/scripts/score_i2v_vbench.py --clip full \\"
    echo "    --video-dir ${PROJECT_ROOT}/wan_experiment/results/${SERIES}/notta_h30s_shard0"
fi
echo "Cancel:  scancel ${JOBS[*]}"
