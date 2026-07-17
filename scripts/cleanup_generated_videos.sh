#!/usr/bin/env bash
# ============================================================================
# Delete generated *.mp4 files to reclaim space — SAFELY, and only after the
# durable metrics/manifests are captured.
#
# This is the LAST step of the cleanup pipeline:
#   1. build_run_manifest.py     (document every run + pool fingerprints)
#   2. VBench backfill on any paper-relevant run missing it
#   3. curate_figure_bank.py     (keep matched example clips)
#   4. THIS script               (delete the bulk mp4s)
#
# Hard protections (never deleted, regardless of KEEP list):
#   - datasets/**                      (source GT inputs)
#   - **/figure_bank/**                (curated qualitative examples)
#   - baseline_experiment/results/gt_clips_*  (FVD/FID reference clips)
#   - LongCat-Video/**                 (vendored code assets)
#
# Refuses to run unless a manifest directory exists (proof the runs are
# documented). Dry-run by default; set CONFIRM=1 to actually delete.
#
# Usage:
#   # dry-run, keep only the two active budget series:
#   KEEP="sweep_experiment/results/panda_ood_budget_1000v_preview
#   sweep_experiment/results/panda_ood_budget_pilot" \
#     bash scripts/cleanup_generated_videos.sh
#
#   # confirm:
#   CONFIRM=1 KEEP="..." bash scripts/cleanup_generated_videos.sh
# ============================================================================
set -euo pipefail

REPO="${REPO:-$(pwd)}"
cd "$REPO"

CONFIRM="${CONFIRM:-0}"
MANIFEST_GLOB="${MANIFEST_GLOB:-sweep_experiment/reports/cleanup_manifests/*/MANIFEST.md}"

# Series (relative paths) whose mp4s to KEEP entirely. Newline/space separated.
KEEP="${KEEP:-sweep_experiment/results/panda_ood_budget_1000v_preview
sweep_experiment/results/panda_ood_budget_pilot}"

# ---- guard: require a manifest so we never delete undocumented runs --------
if ! ls ${MANIFEST_GLOB} >/dev/null 2>&1; then
    echo "ERROR: no manifest found matching ${MANIFEST_GLOB}" >&2
    echo "Run first: python3 scripts/build_run_manifest.py" >&2
    exit 1
fi
echo "Manifest present: $(ls -1 ${MANIFEST_GLOB} | tail -1)"

# ---- build the find prune expression --------------------------------------
# Always-protected path fragments.
PROTECT=(
    "./datasets/*"
    "*/figure_bank/*"
    "./baseline_experiment/results/gt_clips_*"
    "./LongCat-Video/*"
)
for k in ${KEEP}; do
    PROTECT+=("./${k}/*")
done

PRUNE=()
for p in "${PROTECT[@]}"; do
    PRUNE+=(-not -path "${p}")
done

echo "Protected (kept):"
for p in "${PROTECT[@]}"; do echo "  ${p}"; done
echo ""

LIST="$(mktemp -t cleanup-mp4-XXXXX.txt)"
find . -type f -name '*.mp4' "${PRUNE[@]}" -printf '%s %p\n' 2>/dev/null > "${LIST}"

n=$(wc -l < "${LIST}")
gb=$(awk '{s+=$1} END{printf "%.1f", s/1073741824}' "${LIST}")
echo "Deletion set: ${n} files, ${gb} GB"
echo "By series:"
awk '{ k=split($2,a,"/"); print a[2]"/"a[3]"/"a[4] }' "${LIST}" | sort | uniq -c | sort -rn
echo ""

if [ "${CONFIRM}" != "1" ]; then
    echo "[dry-run] Nothing deleted. Full list: ${LIST}"
    echo "Re-run with CONFIRM=1 to delete these ${n} files (${gb} GB)."
    exit 0
fi

# ---- delete ----------------------------------------------------------------
cut -d' ' -f2- "${LIST}" | while IFS= read -r f; do
    rm -f -- "${f}"
done
echo "Deleted ${n} mp4 files (${gb} GB reclaimed)."
echo "Deleted-file list preserved at: ${LIST}"
