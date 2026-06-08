#!/bin/bash
# ============================================================================
# Ablation: standard-horizon TTA WITHOUT text prompt.
#
# Hypothesis: visual-only TTA may produce different gains than visual+text
# TTA. The model is updated on the conditioning video at TTA time WITHOUT a
# caption (caption replaced with the empty string ""), but the final
# inference / generation step still receives the real caption from the
# eval set. Quality / VBench metrics therefore see the same prompt as the
# headline standard-horizon table; the ONLY changing variable is the TTA
# caption.
#
# This is a sibling of submit_standard_1000v_chunked.sh — same datasets,
# same chunking, same hyperparameters, same series dirs. It differs only in:
#   1. Run IDs are suffixed with `_NOPROMPT` (e.g. ADA -> ADA_NOPROMPT).
#   2. Each job is exported with TTA_DISABLE_CAPTION=1, which both
#      run_sweep.sbatch and run_tinylora.sbatch translate into the
#      `--tta-disable-caption` runner flag.
#   3. NOTTA is omitted because there is no TTA step to disable the
#      caption for; NOTTA_NOPROMPT would be byte-identical to NOTTA.
#
# Methods per dataset (all 4 from the headline table):
#   1. ADA_NOPROMPT          (delta_a, AdaSteer with no TTA caption)
#   2. LORA_R8_TTA_NOPROMPT  (lora,    LoRA TTA with no TTA caption)
#   3. TL_BARE_R2_NOPROMPT   (tinylora, bare with no TTA caption)
#   4. TL_TIED_R2_NOPROMPT   (tinylora, tied with no TTA caption)
#
# Datasets:
#   Panda: datasets/panda_1000_480p
#   UCF  : datasets/ucf101_1000_480p
#
# Series dirs (SAME as the headline standard-horizon runs — the
# _NOPROMPT methods land alongside ADA, LORA_R8_TTA, etc.):
#   sweep_experiment/results/panda_1000v_standard/<METHOD>_NOPROMPT/
#   sweep_experiment/results/ucf101_1000v_standard/<METHOD>_NOPROMPT/
#   delta_experiment/results/tinylora_panda_1000v_standard/<METHOD>_NOPROMPT/
#   delta_experiment/results/tinylora_ucf101_1000v_standard/<METHOD>_NOPROMPT/
#
# Total submission: 4 methods × 2 datasets × 10 chunks = 80 jobs.
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh
#
# Dry-run that prints sbatch lines without firing them:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh
#
# Single dataset:
#   ONLY_DATASET=panda bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh
#   ONLY_DATASET=ucf   bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh
#
# Subset of methods:
#   ONLY_METHODS="ADA_NOPROMPT" bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh
#
# Smoke-test single chunk × single method (recommended before firing the
# full 80-job sweep):
#   DRY_RUN=0 NUM_CHUNKS=1 ONLY_DATASET=panda ONLY_METHODS="ADA_NOPROMPT" \
#       bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
TL_SBATCH="${TL_SBATCH:-delta_experiment/sbatch/run_tinylora.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Match headline standard-horizon walls exactly.
TIME_SHORT="${TIME_SHORT:-12:00:00}"
TIME_TL="${TIME_TL:-16:00:00}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"

# Frame geometry — standard 28-frame horizon (matches headline table).
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

_in_filter() {
    local needle="$1"
    [ -z "${ONLY_METHODS}" ] && return 0
    for m in ${ONLY_METHODS}; do
        if [ "${m}" = "${needle}" ]; then return 0; fi
    done
    return 1
}

_exec_or_dry() {
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
        local job_name="t1knp_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_SHORT}" \
            --export="ALL,METHOD=${method},RUN_ID=${run_id},SERIES_NAME=${dataset_tag}_1000v,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,TTA_DISABLE_CAPTION=1${extra}" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

submit_tinylora_chunks() {
    local dataset_tag="$1"; shift
    local run_id="$1"; shift
    local data_dir="$1"; shift
    local results_subdir="$1"; shift
    local extra="$*"

    if ! _in_filter "${run_id}"; then return 0; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${PROJECT_ROOT}/${results_subdir}/${run_id}/chunk_${chunk}"
        local job_name="t1knp_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_TL}" \
            --export="ALL,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},NUM_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,TTA_DISABLE_CAPTION=1${extra}" \
            "${TL_SBATCH}"
        count=$((count + 1))
    done
}

submit_dataset() {
    # Args: $1=dataset_tag  $2=data_dir  $3=sweep_results_subdir
    #       $4=tinylora_results_subdir  $5=adasteer_steps  $6=adasteer_lr
    local dataset_tag="$1"
    local data_dir="$2"
    local sweep_subdir="$3"
    local tl_subdir="$4"
    local ada_steps="$5"
    local ada_lr="$6"

    echo "############################################################"
    echo "Submitting dataset: ${dataset_tag}  (NO-PROMPT TTA ablation)"
    echo "  data_dir     : ${data_dir}"
    echo "  sweep results: ${sweep_subdir}"
    echo "  TL results   : ${tl_subdir}"
    echo "  AdaSteer     : steps=${ada_steps} lr=${ada_lr}"
    echo "  TTA caption  : DISABLED  (TTA_DISABLE_CAPTION=1)"
    echo "############################################################"

    # 1) AdaSteer — no TTA caption.
    submit_sweep_chunks "${dataset_tag}" "delta_a" "ADA_NOPROMPT" \
        "${sweep_subdir}" "${data_dir}" \
        ",DELTA_STEPS=${ada_steps},DELTA_LR=${ada_lr}"

    # 2) LoRA TTA baseline — no TTA caption.
    submit_sweep_chunks "${dataset_tag}" "lora" "LORA_R8_TTA_NOPROMPT" \
        "${sweep_subdir}" "${data_dir}" \
        ",LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,TARGET_FFN=0"

    # 3) TinyLoRA BARE — no TTA caption.
    submit_tinylora_chunks "${dataset_tag}" "TL_BARE_R2_NOPROMPT" "${data_dir}" "${tl_subdir}" \
        ",SVD_RANK=2,N_TIE=1,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=all,TTA_STEPS=20,TTA_LR=1e-3"

    # 4) TinyLoRA TIED — no TTA caption.
    submit_tinylora_chunks "${dataset_tag}" "TL_TIED_R2_NOPROMPT" "${data_dir}" "${tl_subdir}" \
        ",SVD_RANK=2,N_TIE=48,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=all,TTA_STEPS=20,TTA_LR=1e-3"

    echo ""
}

echo "============================================================"
echo "1000-video standard-horizon NO-PROMPT TTA ablation"
echo "============================================================"
echo "  account     : ${ACCOUNT}"
echo "  num chunks  : ${NUM_CHUNKS}  x ${CHUNK_SIZE} videos = ${MAX_VIDEOS}"
echo "  dry run     : ${DRY_RUN}"
echo "  only dataset: ${ONLY_DATASET}"
echo "  only methods: ${ONLY_METHODS:-<all>}"
echo "  TTA caption : DISABLED (TTA_DISABLE_CAPTION=1)"
echo "============================================================"
echo ""

if [ "${ONLY_DATASET}" = "panda" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "panda" \
        "${PROJECT_ROOT}/datasets/panda_1000_480p" \
        "sweep_experiment/results/panda_1000v_standard" \
        "delta_experiment/results/tinylora_panda_1000v_standard" \
        "10" "5.0e-3"
fi

if [ "${ONLY_DATASET}" = "ucf" ] || [ "${ONLY_DATASET}" = "ucf101" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "ucf101" \
        "${PROJECT_ROOT}/datasets/ucf101_1000_480p" \
        "sweep_experiment/results/ucf101_1000v_standard" \
        "delta_experiment/results/tinylora_ucf101_1000v_standard" \
        "5" "2.5e-3"
fi

echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* t1knp_'"
echo ""
echo "After completion, merge chunks (NO-PROMPT methods land in the same"
echo "series dir as the headline runs — the merge command is identical):"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_standard --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/ucf101_1000v_standard --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/delta_experiment/results/tinylora_panda_1000v_standard --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/delta_experiment/results/tinylora_ucf101_1000v_standard --recursive"
echo ""
echo "After merge, rebuild the standard-horizon paper table to incorporate"
echo "the *_NOPROMPT rows alongside the existing ADA / LORA_R8_TTA / TL_*"
echo "headline rows:"
echo "  python scripts/build_paper_tables.py --regime panda_std \\"
echo "      --output sweep_experiment/reports/paper_tables/\$(date +%Y-%m-%d)_headline_1000v_noprompt.md"
echo "  python scripts/build_paper_tables.py --regime ucf_std \\"
echo "      --output sweep_experiment/reports/paper_tables/\$(date +%Y-%m-%d)_headline_1000v_noprompt.md"
echo "============================================================"
