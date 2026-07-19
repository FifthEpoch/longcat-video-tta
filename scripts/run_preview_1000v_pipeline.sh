#!/usr/bin/env bash
# End-to-end commands for the OOD-preview 1000v router pipeline.
#
# Usage:
#   bash scripts/run_preview_1000v_pipeline.sh <phase>
#
# Phases:
#   sample     — OOD-quintile 1000v retain list + symlink dataset
#   sweep      — 12-config budget grid (metrics-only default)
#   sweep-mp4  — same with NO_SAVE_VIDEOS=0 (needed for VBench backfill)
#   notta      — NO-TTA baseline on the SAME pool (router's 13th candidate)
#   merge      — merge chunk summaries per run
#   features   — video + filtered OOD + VAE profile (sbatch chain)
#   vbench     — VBench backfill on saved mp4s (12 GPU jobs)
#   routers    — deploy router CPU suite (sbatch chain)
#   audit      — PSNR/chunk coverage check (run before routers)
#   scope      — per-config overlap vs reference; scopes which configs to rerun
#   resweep    — wipe stale chunk artifacts + resubmit (metrics-only, NO mp4s)
#   resweep-mp4— wipe + resubmit WITH mp4s saved (VBench + downstream reuse)
#   diagnose   — classify per-video NaN PSNR failure modes
#   status     — print paths + CSV line counts
#
# Full happy path (metrics-first, then mp4+vbench+routers):
#   bash scripts/run_preview_1000v_pipeline.sh sample
#   bash scripts/run_preview_1000v_pipeline.sh sweep
#   bash scripts/run_preview_1000v_pipeline.sh merge
#   bash scripts/run_preview_1000v_pipeline.sh features
#   # after sweep metrics OK, rerun for mp4s OR run sweep-mp4 from scratch:
#   bash scripts/run_preview_1000v_pipeline.sh sweep-mp4
#   bash scripts/run_preview_1000v_pipeline.sh vbench
#   bash scripts/run_preview_1000v_pipeline.sh routers
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/preview_1000v_env.sh
source "${SCRIPT_DIR}/preview_1000v_env.sh"

PHASE="${1:-status}"

case "${PHASE}" in
  sample)
    bash "${REPO}/scripts/sample_segment_pool_ood_preview_1000v.sh"
    ;;
  sweep)
    bash "${REPO}/sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh"
    ;;
  sweep-mp4)
    NO_SAVE_VIDEOS=0 bash "${REPO}/sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh"
    ;;
  notta)
    # NO-TTA baseline on the SAME OOD-preview pool (router's 13th candidate).
    # Saves frames by default so it matches the grid arms + enables VBench.
    bash "${REPO}/sweep_experiment/sbatch/submit_notta_1000v_preview.sh"
    ;;
  merge)
    cd "${REPO}"
    python3 sweep_experiment/scripts/merge_chunks.py \
      --results-dir "${PREVIEW_SERIES_ROOT}" \
      --recursive
    ;;
  features)
    bash "${REPO}/scripts/sbatch/submit_preview_1000v_features.sh"
    ;;
  vbench)
    bash "${REPO}/sweep_experiment/sbatch/submit_budget_1000v_preview_vbench_backfill.sh"
    ;;
  routers)
    bash "${REPO}/sweep_experiment/sbatch/submit_deploy_router_1000v_preview.sh"
    ;;
  audit)
    cd "${REPO}"
    python3 scripts/audit_preview_1000v_sweep.py --series-root "${PREVIEW_SERIES_ROOT}"
    ;;
  resweep)
    CONFIRM=1 bash "${REPO}/scripts/wipe_preview_1000v_sweep.sh"
    bash "${REPO}/sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh"
    ;;
  resweep-mp4)
    # Wipe stale chunk artifacts + resubmit WITH frames saved (NO_SAVE_VIDEOS=0).
    # Use this — not plain `resweep` — whenever you need the generated videos
    # (VBench backfill, downstream predictor experiments, etc.).
    CONFIRM=1 bash "${REPO}/scripts/wipe_preview_1000v_sweep.sh"
    NO_SAVE_VIDEOS=0 bash "${REPO}/sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh"
    ;;
  diagnose)
    cd "${REPO}"
    python3 scripts/diagnose_preview_psnr_nan.py --series-root "${PREVIEW_SERIES_ROOT}"
    ;;
  scope)
    cd "${REPO}"
    python3 scripts/diagnose_preview_intersection.py \
      --series-root "${PREVIEW_SERIES_ROOT}" \
      --retain-json "${REPO}/${PREVIEW_JSON}" \
      --per-chunk
    ;;
  status)
    echo "PREVIEW_SERIES_ROOT=${PREVIEW_SERIES_ROOT}"
    echo "PREVIEW_DATASET_DIR=${PREVIEW_DATASET_DIR}"
    echo "PREVIEW_FEATURE_DIR=${PREVIEW_FEATURE_DIR}"
    echo "SEGMENT_OOD_CSV=${SEGMENT_OOD_CSV}"
    if [ -f "${SEGMENT_OOD_CSV}" ]; then
      echo -n "segment_pool OOD lines: "
      wc -l < "${SEGMENT_OOD_CSV}"
    fi
    for f in video_features.csv diffusion_ood_scores.csv vae_latent_profile_features.csv; do
      p="${PREVIEW_FEATURE_DIR}/${f}"
      if [ -f "${p}" ]; then
        echo -n "${f}: "
        wc -l < "${p}"
      else
        echo "${f}: (missing)"
      fi
    done
    if [ -d "${PREVIEW_SERIES_ROOT}" ]; then
      echo "series runs: $(find "${PREVIEW_SERIES_ROOT}" -maxdepth 1 -type d -name 'S*' | wc -l | tr -d ' ')"
    else
      echo "series: (missing)"
    fi
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    echo "Use: sample | sweep | sweep-mp4 | notta | merge | features | vbench | routers | audit | scope | resweep | resweep-mp4 | diagnose | status" >&2
    exit 1
    ;;
esac
