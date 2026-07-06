#!/usr/bin/env bash
# Submit all four verifier routing options on the budget pilot (N=200).
#
# Pipeline:
#   1. (optional) setup_verifier_models.sh — clone VideoAlign + pip deps
#   2. GPU: VideoScore + VideoReward on S2/S10 probe mp4s (Options 1–2)
#   3. GPU: VAE + ResNet embeddings on probe mp4s (Options 3–4)
#   4. CPU: OOF routing eval for all four options
#
# Prerequisites:
#   - Probe mp4s for S2_LR5e3 and S10_LR5e3 under panda_ood_budget_pilot
#   - Phase-0 features under per_video_analysis/${FEATURE_DATE}/
#   - VideoReward checkpoint in ${VIDEOALIGN_ROOT}/checkpoints (Option 2)
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash scripts/setup_verifier_models.sh
#   bash sweep_experiment/sbatch/submit_verifier_options_pilot.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
SERIES="${SERIES:-${PROJECT_ROOT}/sweep_experiment/results/panda_ood_budget_pilot}"
OUT_SCORES="${OUT_SCORES:-${PROJECT_ROOT}/sweep_experiment/reports/verifier_scores}"
OUT_FEATURES="${OUT_FEATURES:-${PROJECT_ROOT}/sweep_experiment/reports/verifier_features}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/longcat-video-checkpoints}"
VIDEOALIGN_ROOT="${VIDEOALIGN_ROOT:-/scratch/${USER}/third_party/VideoAlign}"
NUM_SHARDS="${NUM_SHARDS:-4}"
PROBE_RUNS=(S2_LR5e3 S10_LR5e3)
BACKENDS=(videoscore videoreward)
MODES=(vae resnet)

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log "${OUT_SCORES}" "${OUT_FEATURES}"

echo "========== Verifier options preflight =========="
missing=0
for rid in "${PROBE_RUNS[@]}"; do
  n=$(find "${SERIES}/${rid}" -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  echo "  ${rid}: ${n} mp4s"
  if [ "${n}" -eq 0 ]; then
    echo "  ERROR: no mp4s — rerun pilot with NO_SAVE_VIDEOS=0 for probes" >&2
    missing=1
  fi
done
feat="${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/video_features.csv"
if [ ! -f "${feat}" ]; then
  echo "  WARN: missing ${feat} — eval may fail"
fi
if [ "${missing}" -eq 1 ]; then
  exit 2
fi

# Optional setup job if VideoAlign not present
DEPS=""
if [ ! -d "${VIDEOALIGN_ROOT}" ]; then
  echo "Submitting verifier model setup..."
  SETUP=$(sbatch --parsable --account="${ACCOUNT}" \
    --job-name=verifier_setup \
    --cpus-per-task=2 --mem=8G --time=01:00:00 \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT}" \
    --wrap="cd ${PROJECT_ROOT} && bash scripts/setup_verifier_models.sh")
  DEPS="afterok:${SETUP}"
  echo "  setup job: ${SETUP}"
fi

ALL_JOBS=()

echo ""
echo "Submitting Option 1–2 verifier scoring (${NUM_SHARDS} shards × ${#PROBE_RUNS[@]} runs × ${#BACKENDS[@]} backends)..."
for backend in "${BACKENDS[@]}"; do
  for rid in "${PROBE_RUNS[@]}"; do
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      dep_args=()
      if [ -n "${DEPS}" ]; then dep_args=(--dependency="${DEPS}"); fi
      jid=$(sbatch --parsable "${dep_args[@]}" \
        --account="${ACCOUNT}" \
        --job-name="vsc_${backend}_${rid}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},BACKEND=${backend},RUN_ID=${rid},SERIES=${SERIES},OUT_SCORES=${OUT_SCORES},SHARD_ID=${shard},NUM_SHARDS=${NUM_SHARDS},VIDEOALIGN_ROOT=${VIDEOALIGN_ROOT}" \
        sweep_experiment/sbatch/run_verifier_score_shard.sbatch)
      ALL_JOBS+=("${jid}")
    done
  done
done

echo ""
echo "Submitting Option 3–4 feature extraction (${NUM_SHARDS} shards × ${#PROBE_RUNS[@]} runs × ${#MODES[@]} modes)..."
for mode in "${MODES[@]}"; do
  for rid in "${PROBE_RUNS[@]}"; do
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      jid=$(sbatch --parsable \
        --account="${ACCOUNT}" \
        --job-name="vfeat_${mode}_${rid}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},MODE=${mode},RUN_ID=${rid},SERIES=${SERIES},OUT_FEATURES=${OUT_FEATURES},CHECKPOINT_DIR=${CHECKPOINT_DIR},SHARD_ID=${shard},NUM_SHARDS=${NUM_SHARDS}" \
        sweep_experiment/sbatch/run_verifier_feature_shard.sbatch)
      ALL_JOBS+=("${jid}")
    done
  done
done

if [ ${#ALL_JOBS[@]} -eq 0 ]; then
  echo "ERROR: no GPU jobs submitted" >&2
  exit 2
fi

dep_csv="afterok:$(IFS=:; echo "${ALL_JOBS[*]}")"
EVAL=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=verifier_eval \
  --dependency="${dep_csv}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},FEATURE_DATE=${FEATURE_DATE}" \
  sweep_experiment/sbatch/run_verifier_options_eval.sbatch)

echo ""
echo "========== Submitted =========="
echo "  GPU jobs: ${#ALL_JOBS[@]} (${ALL_JOBS[*]:0:3}...)"
echo "  CPU eval: ${EVAL}"
echo ""
echo "When done:"
echo "  cat sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/verifier_options_eval/summary.md"
echo "  cat sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/verifier_options_eval/verifier_options_decision.json"
