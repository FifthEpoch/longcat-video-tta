#!/bin/bash
# Forward the two leftovers that are not the dead live-gate:
#   rolling_notta N=32  (host; N=8 passed motion+Dyn+IQ)
#   appear_bon    N=32  (appearance pick; N=8 Dyn 0.5 + IQ hold)
# Reuse confirm_32v notta. Do not retune live_min. Do not submit
# live_hist / prefix_sink / pseudo / noise_bon (duplicates or mixed).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_forward.sh
#
# First, on the login node, run the mixed audit (no GPU):
#   python3 -u wan_experiment/scripts/diagnose_v2v_mixed.py \
#     --baseline-dir wan_experiment/results/v2v_panda_bakeoff_8v \
#     --series-dir wan_experiment/results/v2v_panda_lineage_8v \
#     --series-dir wan_experiment/results/v2v_panda_ideas_8v

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_forward_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
ROLL_WALL="${ROLL_WALL:-04:00:00}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing." >&2
    exit 1
fi
if [[ ! -d "${NOTTA_DIR}" ]]; then
    echo "ERROR: confirm notta missing: ${NOTTA_DIR}" >&2
    exit 1
fi
if [[ ! -f /scratch/${USER}/wan-checkpoints/rolling_forcing_dmd.pt ]]; then
    echo "ERROR: Rolling Forcing ckpt missing." >&2
    exit 1
fi

JOBS=()
J=$(sbatch --parsable --account="${ACCOUNT}" --time="${ROLL_WALL}" \
    --export=ALL,METHOD=rolling_notta,SEARCH_K=1,${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${SERIES} rolling_notta n=${N_VIDEOS} job ${J}"
JOBS+=("${J}")

J=$(sbatch --parsable --account="${ACCOUNT}" --time="${SEARCH_WALL}" \
    --export=ALL,METHOD=appear_bon,SEARCH_K=4,${COMMON} \
    "${SB}/run_v2v_chunked.sbatch")
echo "V2V ${SERIES} appear_bon n=${N_VIDEOS} job ${J}"
JOBS+=("${J}")

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/rolling_notta_h30s_shard0 ${ROOT}/appear_bon_h30s_shard0 ${NOTTA_DIR}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "Forward: rolling_notta + appear_bon N=32 vs confirm notta."
echo "prefix_sink / LongLive quality stay on the N=8 diagnose — not this submit."
echo "When generate finishes (sidecars):"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --allow-partial"
echo "Cancel this wave only:  scancel ${JOBS[*]}"
