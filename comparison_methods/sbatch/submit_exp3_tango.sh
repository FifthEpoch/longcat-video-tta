#!/bin/bash
# ============================================================================
# EXP3 — TANGO predicted-noise-gaussianity guidance (training-free FVD lever)
#
# Motivation: AdaSteer's per-video delta does NOT move FVD (EXP2 null on FVD +
# all VBench dims). TANGO instead nudges the SAMPLING TRAJECTORY so the per-step
# predicted noise eps_hat stays ~ N(0, I) — a distribution-level intervention
# that directly targets FVD. See sweep_experiment/reports/
# 2026-08-04_literature_v2v_tta_directions.md.
#
# Design (clean isolation): every arm uses the SAME differentiable Euler sampler
# with NO noise optimization (--no-optimize) and the FAITHFUL LongCat prediction
# config (CFG on + 50 steps; the PVDM no-CFG/10-step recipe is garbage on this
# backbone). The ONLY difference between arms is the TANGO guidance:
#   control      : no guidance                     (the sampler's own FVD)
#   tango_lXX    : --tango-guidance --tango-lambda  (gaussianity guidance ON)
# So control-vs-tango isolates the guidance effect. All arms share the OOD
# preview pool + geometry (cond=14/frames=28/gsf=48, seed=42), so results line
# up with the AdaSteer / placement runs.
#
# Pilot (default N=80): screen lambda for numerical stability + pixel/VBench
# direction (N=80 FVD is NOT reliable). Then re-run the 1-2 best lambdas at
# EXP3_N=512 for a trustworthy FVD (matches the placement-FVD scale-up logic).
#
# Usage (repo root on the cluster):
#   bash comparison_methods/sbatch/submit_exp3_tango.sh                 # submit
#   DRY_RUN=1 bash comparison_methods/sbatch/submit_exp3_tango.sh       # print only
#   EXP3_N=512 EXP3_LAMBDAS="0.05" bash comparison_methods/sbatch/submit_exp3_tango.sh
# ============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SBATCH="${PROJECT_ROOT}/comparison_methods/sbatch/run_savi_dno_longcat.sbatch"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# --- shared config (held fixed across arms) --------------------------------
SERIES="${SERIES:-exp3_tango_panda_preview}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
EXP3_N="${EXP3_N:-80}"
SEED="${SEED:-42}"

# geometry — MUST match the budget preview series
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"

# faithful LongCat prediction: CFG on + 50 steps (custom sampler is garbage
# without CFG). No noise-opt (isolate TANGO).
GEN_STEPS="${GEN_STEPS:-50}"
GUIDANCE="${GUIDANCE:-4.0}"

# TANGO guidance-strength sweep (pilot). Comma/space separated.
EXP3_LAMBDAS="${EXP3_LAMBDAS:-0.02 0.05 0.1}"
TANGO_SIGMA_HI="${TANGO_SIGMA_HI:-0.9}"
TANGO_SIGMA_LO="${TANGO_SIGMA_LO:-0.0}"
TANGO_KURTOSIS="${TANGO_KURTOSIS:-0.0}"

# optional preview GT cache (for online FVD/FID; N=80 online FVD is NOT reliable
# — real FVD is scored later on saved videos at scale). Leave empty to skip.
GT_FEATURES_CACHE="${GT_FEATURES_CACHE:-}"

RESULTS_ROOT="${PROJECT_ROOT}/comparison_methods/results/${SERIES}"

echo "============================================================"
echo "EXP3 TANGO gaussianity guidance"
echo "  account   : ${ACCOUNT}"
echo "  series    : ${SERIES}"
echo "  data_dir  : ${DATA_DIR}"
echo "  N         : ${EXP3_N}   seed=${SEED}"
echo "  geometry  : cond=${NUM_COND_FRAMES} frames=${NUM_FRAMES} gsf=${GEN_START_FRAME}"
echo "  sampler   : no-opt, CFG on, ${GEN_STEPS} steps, guidance=${GUIDANCE}"
echo "  lambdas   : ${EXP3_LAMBDAS}   sigma=[${TANGO_SIGMA_LO},${TANGO_SIGMA_HI}] kurt=${TANGO_KURTOSIS}"
echo "  gt_cache  : ${GT_FEATURES_CACHE:-<none>}"
echo "============================================================"

submit () {
  # $1=run_id  $2=tango(0/1)  $3=lambda
  local run_id="$1" tango="$2" lam="$3"
  local out="${RESULTS_ROOT}/${run_id}"
  local -a envs=(
    "SAVI_LC_DATA_DIR=${DATA_DIR}" "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    "SAVI_LC_OUTPUT_DIR=${out}" "SAVI_LC_MAX_VIDEOS=${EXP3_N}"
    "SAVI_NUM_COND_FRAMES=${NUM_COND_FRAMES}" "SAVI_NUM_FRAMES=${NUM_FRAMES}"
    "SAVI_GEN_START_FRAME=${GEN_START_FRAME}"
    "SAVI_NO_OPTIMIZE=1" "SAVI_ROLLOUT_STEPS=0"
    "SAVI_GENERATION_CFG=1" "SAVI_GENERATION_STEPS=${GEN_STEPS}"
    "SAVI_GUIDANCE=${GUIDANCE}"
    "SAVI_COMPUTE_VBENCH=1"
    "SAVI_TANGO_GUIDANCE=${tango}" "SAVI_TANGO_LAMBDA=${lam}"
    "SAVI_TANGO_SIGMA_HI=${TANGO_SIGMA_HI}" "SAVI_TANGO_SIGMA_LO=${TANGO_SIGMA_LO}"
    "SAVI_TANGO_KURTOSIS=${TANGO_KURTOSIS}"
  )
  if [ -n "${GT_FEATURES_CACHE}" ]; then
    envs+=("GT_FEATURES_CACHE=${GT_FEATURES_CACHE}")
  fi
  echo "-> arm ${run_id}: tango=${tango} lambda=${lam}"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "   DRY: sbatch --account=${ACCOUNT} --job-name=exp3_${run_id}"
    echo "        ${envs[*]}"
  else
    local jid
    jid=$(sbatch --parsable --account="${ACCOUNT}" \
           --export=ALL,"$(IFS=,; echo "${envs[*]}")" \
           --job-name="exp3_${run_id}" "${SBATCH}")
    echo "   job ${jid}  (-> ${out})"
  fi
}

# control arm (no guidance) — the sampler's own FVD baseline. Skip with
# EXP3_CONTROL=0 when re-running only the guided arms (control is unaffected
# by TANGO-code changes, so no need to regenerate it).
if [ "${EXP3_CONTROL:-1}" = "1" ]; then
  submit "control" 0 0.0
else
  echo "-> control: SKIPPED (EXP3_CONTROL=0)"
fi

# TANGO lambda sweep
for lam in ${EXP3_LAMBDAS}; do
  slug=$(echo "${lam}" | tr -d '.' )
  submit "tango_l${slug}" 1 "${lam}"
done

echo ""
echo "After the arms finish, score FVD control-vs-TANGO on the saved gen clips:"
echo "  # gen-only already (savi runner saves gen frames), so eval_fvd directly on videos/"
echo "  for arm in control ${EXP3_LAMBDAS//./}; do :; done"
echo "  (see comparison_methods/results/${SERIES}/<arm>/{summary.json,videos/,vbench_results/})"
echo ""
echo "Pilot read (N=${EXP3_N}): confirm TANGO is (a) numerically stable and (b) does"
echo "not tank PSNR/VBench vs control. Then re-run the best lambda(s) at EXP3_N=512"
echo "for a reliable FVD:  EXP3_N=512 EXP3_LAMBDAS=\"<best>\" bash $0"
