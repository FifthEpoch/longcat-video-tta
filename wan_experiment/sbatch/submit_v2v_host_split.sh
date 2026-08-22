#!/bin/bash
# Four cheap host hypotheses. First run is N=32 (same first 32 as
# confirm/forward). N=8 has been a PROMOTE trap all week. No LoRA. No TTC.
#
#   H1 GPU: sf_roll (SF θ + RF window) and rf_chunk (RF θ + SF chunks)
#   H4 GPU: sf_recache / rf_recache (VAE re-encode last ~2 s, reset KV)
#   H2/H3:  login CPU, no GPU — resim_v2v_host_switch.py on existing mp4s
#
# Baselines: confirm_32v SF notta AND forward_32v rolling_notta.
# H1 vs both. H4 vs its own host. Do not scale to 128 from a weak 32.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   python3 -u wan_experiment/scripts/resim_v2v_host_switch.py
#   bash wan_experiment/sbatch/submit_v2v_host_split.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_host_split_32v}"
N_VIDEOS="${N_VIDEOS:-32}"
GEN_WALL="${GEN_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"
ROLL_BASE="${ROLL_BASE:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v/rolling_notta_h30s_shard0}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"

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
    local J
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${GEN_WALL}" \
        --export=ALL,METHOD="${method}",SEARCH_K=1,${COMMON} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${SERIES} ${method} n=${N_VIDEOS} job ${J}"
    JOBS+=("${J}")
}

submit_method sf_roll
submit_method rf_chunk
submit_method sf_recache
submit_method rf_recache

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/sf_roll_h30s_shard0 ${ROOT}/rf_chunk_h30s_shard0 ${ROOT}/sf_recache_h30s_shard0 ${ROOT}/rf_recache_h30s_shard0 ${ROLL_BASE} ${NOTTA_DIR}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterany:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterany ${DEPS}"
JOBS+=("${VB}")

echo "H1/H4 N=32. Same first 32 as confirm/forward. 2-way H200: queues behind 128 VBench."
echo "H2/H3 is login CPU (already in the paste block above)."
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --allow-partial"
echo "  python3 -u wan_experiment/scripts/pair_v2v_tails.py \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v \\"
echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v \\"
echo "    --series-dir ${ROOT}"
echo "H1 vs SF notta AND vs rolling_notta. H4 vs its own host."
echo "Do not scale to 128 from a weak 32. Cancel this wave only:  scancel ${JOBS[*]}"
