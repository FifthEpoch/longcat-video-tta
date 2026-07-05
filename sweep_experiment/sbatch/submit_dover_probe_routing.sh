#!/usr/bin/env bash
# Track C: DOVER scores on budget-pilot probe mp4s (S2/S10) → exp13 routing eval.
#
# Prerequisites:
#   1. Probe mp4s exist for S2_LR5e3 and S10_LR5e3 under panda_ood_budget_pilot
#   2. DOVER installed once: bash scripts/setup_dover_env.sh
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash sweep_experiment/sbatch/submit_dover_probe_routing.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DOVER_ROOT="${DOVER_ROOT:-/scratch/${USER}/third_party/DOVER}"
SERIES="${SERIES:-${PROJECT_ROOT}/sweep_experiment/results/panda_ood_budget_pilot}"
OUT_SCORES="${OUT_SCORES:-${PROJECT_ROOT}/sweep_experiment/reports/dover_scores}"
OUT_EVAL="${OUT_EVAL:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-07-05/dover_probe_routing}"
NUM_SHARDS="${NUM_SHARDS:-4}"

PROBE_RUNS=(S2_LR5e3 S10_LR5e3)

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log "${OUT_SCORES}"

echo "========== Track C preflight =========="
if [ ! -f "${DOVER_ROOT}/pretrained_weights/DOVER.pth" ]; then
  echo "DOVER not installed. Submitting setup job first..."
  SETUP=$(sbatch --parsable --account="${ACCOUNT}" \
    --job-name=dover_setup \
    --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --export="ALL,DOVER_ROOT=${DOVER_ROOT}" \
    sweep_experiment/sbatch/run_dover_setup.sbatch)
  echo "  setup job: ${SETUP}"
  DEPS="afterok:${SETUP}"
else
  DEPS=""
fi

missing=0
for rid in "${PROBE_RUNS[@]}"; do
  n=$(find "${SERIES}/${rid}" -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  echo "  ${rid}: ${n} mp4s"
  if [ "${n}" -eq 0 ]; then
    echo "  ERROR: no mp4s for ${rid}. Run probe mp4 re-run first:" >&2
    echo "    ONLY_RUNS='S2_LR5e3 S10_LR5e3' NO_SAVE_VIDEOS=0 FORCE_MP4_RERUN=1 \\" >&2
    echo "      bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh" >&2
    missing=1
  fi
done
if [ "${missing}" -eq 1 ]; then
  exit 2
fi

echo ""
echo "Submitting DOVER scoring: ${#PROBE_RUNS[@]} configs × ${NUM_SHARDS} shards..."

SCORE_JOBS=()
for rid in "${PROBE_RUNS[@]}"; do
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    dep_args=()
    if [ -n "${DEPS}" ]; then dep_args=(--dependency="${DEPS}"); fi
    jid=$(sbatch --parsable "${dep_args[@]}" \
      --account="${ACCOUNT}" \
      --job-name="dover_${rid}" \
      --cpus-per-task=4 --mem=32G --time=04:00:00 \
      --gres=gpu:h200:1 \
      --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DOVER_ROOT=${DOVER_ROOT},RUN_ID=${rid},SERIES=${SERIES},OUT_SCORES=${OUT_SCORES},SHARD_ID=${shard},NUM_SHARDS=${NUM_SHARDS}" \
      sweep_experiment/sbatch/run_dover_score_shard.sbatch)
    SCORE_JOBS+=("${jid}")
  done
done

if [ ${#SCORE_JOBS[@]} -eq 0 ]; then
  echo "ERROR: no scoring jobs submitted" >&2
  exit 2
fi

dep_csv="afterok:$(IFS=:; echo "${SCORE_JOBS[*]}")"
EVAL=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=dover_eval \
  --dependency="${dep_csv}" \
  --cpus-per-task=2 --mem=8G --time=00:15:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},OUT_SCORES=${OUT_SCORES},OUT_EVAL=${OUT_EVAL}" \
  sweep_experiment/sbatch/run_dover_probe_eval.sbatch)

echo ""
echo "Track C submitted."
echo "  scoring shards: ${SCORE_JOBS[*]}"
echo "  eval job: ${EVAL}"
echo "  scores: ${OUT_SCORES}/"
echo "  results: ${OUT_EVAL}/exp13_dover_probe_route.md"
