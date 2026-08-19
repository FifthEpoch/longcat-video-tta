#!/bin/bash
# T2V 30 s three-way on first 128 MovieGen prompts.
#   NOTTA | always-BoN k=4 | gated-BoN hybrid (same gate as I2V-32)
# Optional compare to Relax Forcing / Self-Forcing++ / FreqForcing.
# Not a task lock. No TTC. I2V-32 scale-up stays closed.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   # recommended first (2 prompts, 3 methods, ~15 min):
#   SMOKE=1 bash wan_experiment/sbatch/submit_t2v_bon128.sh
#   # then the 128:
#   bash wan_experiment/sbatch/submit_t2v_bon128.sh
#
# 2-way H200 cap: extra jobs queue. 4 shards × 3 methods = 12 generate jobs.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SF_ROOT="${SF_ROOT:-${SCRATCH_BASE}/third_party/Self-Forcing}"
PYTHON="${PYTHON:-${SCRATCH_BASE}/conda-envs/self_forcing/bin/python}"

SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
    SERIES="${SERIES:-t2v_bon_smoke}"
    N_VIDEOS="${N_VIDEOS:-2}"
    NUM_SHARDS="${NUM_SHARDS:-1}"
    NOTTA_TIME="${NOTTA_TIME:-02:00:00}"
    SEARCH_TIME="${SEARCH_TIME:-02:00:00}"
else
    SERIES="${SERIES:-t2v_bon_128v_vbenchlong}"
    N_VIDEOS="${N_VIDEOS:-128}"
    NUM_SHARDS="${NUM_SHARDS:-4}"
    NOTTA_TIME="${NOTTA_TIME:-04:00:00}"
    SEARCH_TIME="${SEARCH_TIME:-08:00:00}"
fi

PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/datasets/moviegen_128_resolved.txt}"
GATE="GATE_THRESHOLD=2.0,GATE_CH1_THRESHOLD=0.8,GATE_DELTA=0.5,GATE_DELTA_PREV_MIN=0.5"
COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=1,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=${NUM_SHARDS},PROMPT_FILE=${PROMPT_FILE},${GATE}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log" \
    "${PROJECT_ROOT}/datasets"

echo "Resolving MovieGen prompts -> ${PROMPT_FILE}"
"${PYTHON}" "${PROJECT_ROOT}/wan_experiment/scripts/prepare_t2v_prompts.py" \
    --sf-root "${SF_ROOT}" \
    --vendor "${PROJECT_ROOT}/datasets/moviegen_128.txt" \
    --n "${N_VIDEOS}" \
    --out "${PROMPT_FILE}"

JOBS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    J1=$(sbatch --parsable --account="${ACCOUNT}" --time="${NOTTA_TIME}" \
        --export=ALL,METHOD=notta,SEARCH_K=1,SHARD_ID=${SHARD_ID},${COMMON} \
        "${SB}/run_t2v_chunked.sbatch")
    echo "T2V ${SERIES} NOTTA          shard ${SHARD_ID} job ${J1}"
    JOBS+=("${J1}")

    J2=$(sbatch --parsable --account="${ACCOUNT}" --time="${SEARCH_TIME}" \
        --export=ALL,METHOD=always_bon,SEARCH_K=4,SHARD_ID=${SHARD_ID},${COMMON} \
        "${SB}/run_t2v_chunked.sbatch")
    echo "T2V ${SERIES} always-BoN k=4 shard ${SHARD_ID} job ${J2}"
    JOBS+=("${J2}")

    J3=$(sbatch --parsable --account="${ACCOUNT}" --time="${SEARCH_TIME}" \
        --export=ALL,METHOD=gated_bon,SEARCH_K=4,SHARD_ID=${SHARD_ID},${COMMON} \
        "${SB}/run_t2v_chunked.sbatch")
    echo "T2V ${SERIES} gated-BoN k=4  shard ${SHARD_ID} job ${J3}"
    JOBS+=("${J3}")
done

echo "Submitted ${#JOBS[@]} jobs: ${JOBS[*]}"
echo "2-way H200 cap: extras queue. No TTC."
echo "When they finish:"
echo "  python wan_experiment/scripts/analyze_i2v_bon.py \\"
echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
echo "Cancel:  scancel ${JOBS[*]}"
