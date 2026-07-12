#!/usr/bin/env bash
# Shared paths for the OOD-preview 1000v router pipeline (source before other scripts).
set -euo pipefail

export REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
export PREVIEW_FEATURE_DATE="${PREVIEW_FEATURE_DATE:-2026-07-12}"
export PREVIEW_DATE_TAG="${PREVIEW_DATE_TAG:-${PREVIEW_FEATURE_DATE}}"

export PREVIEW_SERIES="${PREVIEW_SERIES:-panda_ood_budget_1000v_preview}"
export PREVIEW_DATASET="${PREVIEW_DATASET:-panda_ood_budget_1000v_preview_480p}"
export PREVIEW_JSON="${PREVIEW_JSON:-sweep_experiment/lists/panda_ood_budget_1000v_preview_videos.json}"

export PREVIEW_SERIES_ROOT="${PREVIEW_SERIES_ROOT:-${REPO}/sweep_experiment/results/${PREVIEW_SERIES}}"
export PREVIEW_DATASET_DIR="${PREVIEW_DATASET_DIR:-${REPO}/datasets/${PREVIEW_DATASET}}"
export PREVIEW_FEATURE_DIR="${PREVIEW_FEATURE_DIR:-${REPO}/sweep_experiment/reports/per_video_analysis/${PREVIEW_FEATURE_DATE}}"

export SEGMENT_OOD_CSV="${SEGMENT_OOD_CSV:-${REPO}/sweep_experiment/reports/per_video_analysis/2026-07-10/diffusion_ood_scores_segment_pool.csv}"
