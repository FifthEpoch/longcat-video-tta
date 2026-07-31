#!/bin/bash
# ============================================================================
# Submit the two PUBLISHED noise-optimization TTA baselines on LongCat, on the
# same 1000v OOD preview pool + shared GT cache as AdaSteer / SAVi-DNO, so the
# only thing that differs is the TTA strategy (apples-to-apples).
#
#   dno              -> Karunratanakul et al., "Optimizing Diffusion Noise Can
#                       Serve As Universal Motion Priors", CVPR 2024
#                       (decorrelation regularizer).
#   direct_noise_opt -> Tang et al., "Inference-Time Alignment of Diffusion
#                       Models with Direct Noise Optimization", ICML 2025
#                       (Gaussian-shell probability regularizer).
#
# Both reuse comparison_methods/scripts/savi_dno_longcat.py (differentiable
# LongCat Euler sampler) with the leakage-free prediction protocol: optimize
# the initial noise on an OBSERVED history segment, then apply it to predict
# the UNSEEN future. No future GT enters optimization.
#
# IMPORTANT (matched comparison): DATA_DIR must contain the SAME videos as the
# AdaSteer 1000v preview pool (panda_ood_budget_1000v_preview) and GT_CACHE must
# be that pool's cache, or the FVD/PSNR are not comparable to the AdaSteer rows.
#
# SCHEDULING NOTE (why the first attempt, jobs 15044153/54, died at ~2h):
#   They ran 2 GPUs with a naive pipeline split, so each H200 sat at ~50% util
#   and Torch's aggressive Low-GPU-Utilization policy (auto-cancel below 60% on
#   gh* nodes) killed them -- it was NOT the 48h wall clock. This script now
#   defaults to ONE GPU (keeps it ~100% busy) and the run checkpoints after
#   every video, so a cancelled/preempted job can be resubmitted (same
#   OUTPUT_DIR) and resumes exactly where it stopped without corrupting the
#   pooled FVD/FID. At ~413 s/video a single 48h window covers ~400 videos; for
#   the full 1000 just resubmit the same command 2-3x (it will skip done ones).
#
# Usage:
#   cd /scratch/wc3013/longcat-video-tta
#   bash comparison_methods/sbatch/run_noise_opt_baselines.sh
#   # optional overrides:
#   DATA_DIR=... GT_CACHE=... MAX_VIDEOS=1000 EULER_STEPS=10 NUM_GPUS=1 \
#     bash comparison_methods/sbatch/run_noise_opt_baselines.sh
# ============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
cd "${PROJECT_ROOT}"

# Same pool + cache as the AdaSteer 1000v preview (override if your dataset dir
# for that exact pool lives elsewhere).
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"
# ONE GPU by default -> ~100% util -> not killed by the Low-GPU-Util policy.
EULER_STEPS="${EULER_STEPS:-10}"
NUM_GPUS="${NUM_GPUS:-1}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10}"

SBATCH="comparison_methods/sbatch/run_savi_dno_longcat.sbatch"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
mkdir -p comparison_methods/slurm_log

if [ ! -d "${DATA_DIR}" ]; then
  echo "WARNING: DATA_DIR does not exist: ${DATA_DIR}" >&2
  echo "         Set DATA_DIR to the 1000v preview pool's video dataset dir." >&2
fi
if [ ! -f "${GT_CACHE}" ]; then
  echo "WARNING: GT_CACHE missing: ${GT_CACHE} (FVD/FID will be computed against" >&2
  echo "         per-run references instead of the shared preview cache)." >&2
fi

echo "============================================================"
echo "Submitting published noise-opt TTA baselines"
echo "  data_dir  : ${DATA_DIR}"
echo "  gt_cache  : ${GT_CACHE}"
echo "  max_videos: ${MAX_VIDEOS}   euler_steps: ${EULER_STEPS}   gpus: ${NUM_GPUS}"
echo "============================================================"

for METHOD in dno direct_noise_opt; do
  echo "  -> submitting ${METHOD}"
  NOISE_OPT_METHOD="${METHOD}" \
  SAVI_LC_DATA_DIR="${DATA_DIR}" \
  GT_FEATURES_CACHE="${GT_CACHE}" \
  SAVI_LC_MAX_VIDEOS="${MAX_VIDEOS}" \
  SAVI_EULER_STEPS="${EULER_STEPS}" \
  NUM_GPUS="${NUM_GPUS}" \
  SAVI_ROLLOUT_STEPS="${ROLLOUT_STEPS}" \
  sbatch --account="${ACCOUNT}" "${SBATCH}"
done

echo ""
echo "Submitted. Monitor: squeue -u ${USER}"
echo "Results:"
echo "  comparison_methods/results/dno_longcat_s${EULER_STEPS}/summary.json"
echo "  comparison_methods/results/direct_noise_opt_longcat_s${EULER_STEPS}/summary.json"
