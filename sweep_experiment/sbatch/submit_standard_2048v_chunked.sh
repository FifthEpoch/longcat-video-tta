#!/bin/bash
# ============================================================================
# Submit the headline 2048-video sweep: 6 methods x 2 datasets, chunked.
#
# Methods per dataset:
#   1. NOTTA           (delta_a, delta_steps=0)
#   2. ADA             (delta_a, AdaSteer — S10/LR=5e-3 on Panda, S5/LR=2.5e-3 on UCF)
#   3. LORA_R1         (lora,    rank=1, 5 steps, lr=2e-4)
#   4. LORA_R8         (lora,    rank=8, 10 steps, lr=5e-5)
#   5. TL_BARE_R2      (tinylora, rank=2, n_tie=1, qkv_proj, all blocks)
#   6. TL_TIED_R2      (tinylora, rank=2, n_tie=48, qkv_proj, all blocks)
#
# Chunking strategy: 2048 videos / CHUNK_SIZE = NUM_CHUNKS jobs per method.
# Default 16 chunks x 128 videos at ~14h each keeps every job comfortably
# under the 24h preemption wall.
#
# Each chunk writes its own summary.json and fvd_fid_stats.npz under
#   sweep_experiment/results/<series>_2048v/<RUN_ID>/chunk_<i>/
# (or delta_experiment/results/tinylora_2048v/<RUN_ID>/chunk_<i>/ for TL).
# After completion, merge with sweep_experiment/scripts/merge_chunks.py.
#
# COMPUTE_VBENCH=1 is set so VBench++ runs inline per chunk and writes
# its scores into each chunk's summary.json. Set NO_SAVE_VIDEOS=0 so
# generated mp4 files are retained on disk for any post-hoc analysis;
# total disk footprint is ~250 GB across both datasets and is cleaned
# up by the standard scratch GC policy.
#
# Submit (after `git pull` and dataset builds):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_standard_2048v_chunked.sh
#
# Submit a dry run that prints sbatch lines without firing them:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_standard_2048v_chunked.sh
#
# Submit only one dataset:
#   ONLY_DATASET=panda bash sweep_experiment/sbatch/submit_standard_2048v_chunked.sh
#   ONLY_DATASET=ucf   bash sweep_experiment/sbatch/submit_standard_2048v_chunked.sh
#
# Submit only specific run IDs:
#   ONLY_METHODS="NOTTA ADA" bash sweep_experiment/sbatch/submit_standard_2048v_chunked.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
TL_SBATCH="${TL_SBATCH:-delta_experiment/sbatch/run_tinylora.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Per-dataset wall-time per chunk.
# - delta_a (NOTTA / ADA): 10-step TTA -> ~10h/chunk at 128 vids => 14h with buffer
# - lora R1/R8           : 5-10 step TTA -> ~10h/chunk        => 14h
# - tinylora (20 steps)  : higher TTA cost, slower backward    => 18h
TIME_SHORT="${TIME_SHORT:-14:00:00}"
TIME_TL="${TIME_TL:-18:00:00}"

NUM_CHUNKS="${NUM_CHUNKS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
MAX_VIDEOS="${MAX_VIDEOS:-2048}"

# Frame geometry — standard 28-frame horizon.
NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

DRY_RUN="${DRY_RUN:-0}"
ONLY_DATASET="${ONLY_DATASET:-both}"
ONLY_METHODS="${ONLY_METHODS:-}"

count=0
declare -a submitted_jobs

# Lower-case for matching against the optional ONLY_METHODS filter.
_in_filter() {
    local needle="$1"
    [ -z "${ONLY_METHODS}" ] && return 0
    for m in ${ONLY_METHODS}; do
        if [ "${m}" = "${needle}" ]; then return 0; fi
    done
    return 1
}

_exec_or_dry() {
    # Print and optionally execute an sbatch command.
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] $*"
        return 0
    fi
    "$@"
}

submit_sweep_chunks() {
    # Args: $1=dataset_tag (panda|ucf101)  $2=method (lora|delta_a)
    #       $3=run_id  $4=results_subdir  $5=data_dir  $6+=extra,KV pairs
    local dataset_tag="$1"; shift
    local method="$1"; shift
    local run_id="$1"; shift
    local results_subdir="$1"; shift
    local data_dir="$1"; shift
    local extra="$*"

    if ! _in_filter "${run_id}"; then return 0; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${PROJECT_ROOT}/${results_subdir}/${run_id}/chunk_${chunk}"
        local job_name="t2k_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_SHORT}" \
            --export="ALL,METHOD=${method},RUN_ID=${run_id},SERIES_NAME=${dataset_tag}_2048v,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn${extra}" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

submit_tinylora_chunks() {
    # Args: $1=dataset_tag  $2=run_id  $3=data_dir  $4=results_subdir
    #       $5+=extra KV
    local dataset_tag="$1"; shift
    local run_id="$1"; shift
    local data_dir="$1"; shift
    local results_subdir="$1"; shift
    local extra="$*"

    if ! _in_filter "${run_id}"; then return 0; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${PROJECT_ROOT}/${results_subdir}/${run_id}/chunk_${chunk}"
        local job_name="t2k_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_TL}" \
            --export="ALL,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},NUM_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn${extra}" \
            "${TL_SBATCH}"
        count=$((count + 1))
    done
}

submit_dataset() {
    # Args: $1=dataset_tag (panda|ucf101)
    #       $2=data_dir
    #       $3=sweep_results_subdir
    #       $4=tinylora_results_subdir
    #       $5=adasteer_steps
    #       $6=adasteer_lr
    local dataset_tag="$1"
    local data_dir="$2"
    local sweep_subdir="$3"
    local tl_subdir="$4"
    local ada_steps="$5"
    local ada_lr="$6"

    echo "############################################################"
    echo "Submitting dataset: ${dataset_tag}"
    echo "  data_dir     : ${data_dir}"
    echo "  sweep results: ${sweep_subdir}"
    echo "  TL results   : ${tl_subdir}"
    echo "  AdaSteer     : steps=${ada_steps} lr=${ada_lr}"
    echo "############################################################"

    # 1) NoTTA
    submit_sweep_chunks "${dataset_tag}" "delta_a" "NOTTA" \
        "${sweep_subdir}" "${data_dir}" \
        ",DELTA_STEPS=0,DELTA_LR=${ada_lr}"

    # 2) AdaSteer
    submit_sweep_chunks "${dataset_tag}" "delta_a" "ADA" \
        "${sweep_subdir}" "${data_dir}" \
        ",DELTA_STEPS=${ada_steps},DELTA_LR=${ada_lr}"

    # 3) LoRA R1 (tiny LoRA recipe)
    # Note: TARGET_MODULES default in run_sweep.sbatch is already "qkv,proj"
    # — we don't pass it here because SLURM --export treats commas as
    # variable separators, so embedded commas in values get split.
    submit_sweep_chunks "${dataset_tag}" "lora" "LORA_R1" \
        "${sweep_subdir}" "${data_dir}" \
        ",LORA_RANK=1,LORA_ALPHA=2,LORA_TARGET_BLOCKS=all,NUM_STEPS=5,LEARNING_RATE=2.0e-4,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,TARGET_FFN=0"

    # 4) LoRA R8 (industry-default LoRA recipe)
    submit_sweep_chunks "${dataset_tag}" "lora" "LORA_R8" \
        "${sweep_subdir}" "${data_dir}" \
        ",LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,TARGET_FFN=0"

    # 5) TinyLoRA BARE (rank=2, no tying, qkv+proj, all blocks)
    submit_tinylora_chunks "${dataset_tag}" "TL_BARE_R2" "${data_dir}" "${tl_subdir}" \
        ",SVD_RANK=2,N_TIE=1,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=all,TTA_STEPS=20,TTA_LR=1e-3"

    # 6) TinyLoRA TIED (rank=2, n_tie=48, qkv+proj, all blocks => 4 params)
    submit_tinylora_chunks "${dataset_tag}" "TL_TIED_R2" "${data_dir}" "${tl_subdir}" \
        ",SVD_RANK=2,N_TIE=48,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=all,TTA_STEPS=20,TTA_LR=1e-3"

    echo ""
}

echo "============================================================"
echo "2048-video headline sweep submission"
echo "============================================================"
echo "  account     : ${ACCOUNT}"
echo "  num chunks  : ${NUM_CHUNKS}  x ${CHUNK_SIZE} videos = ${MAX_VIDEOS}"
echo "  dry run     : ${DRY_RUN}"
echo "  only dataset: ${ONLY_DATASET}"
echo "  only methods: ${ONLY_METHODS:-<all>}"
echo "============================================================"
echo ""

if [ "${ONLY_DATASET}" = "panda" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "panda" \
        "${PROJECT_ROOT}/datasets/panda_2048_480p" \
        "sweep_experiment/results/panda_2048v" \
        "delta_experiment/results/tinylora_panda_2048v" \
        "10" "5.0e-3"
fi

if [ "${ONLY_DATASET}" = "ucf" ] || [ "${ONLY_DATASET}" = "ucf101" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "ucf101" \
        "${PROJECT_ROOT}/datasets/ucf101_2048_480p" \
        "sweep_experiment/results/ucf101_2048v" \
        "delta_experiment/results/tinylora_ucf101_2048v" \
        "5" "2.5e-3"
fi

echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* t2k_'"
echo ""
echo "After completion, merge chunks:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_2048v --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/ucf101_2048v --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/delta_experiment/results/tinylora_panda_2048v --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/delta_experiment/results/tinylora_ucf101_2048v --recursive"
echo "============================================================"
