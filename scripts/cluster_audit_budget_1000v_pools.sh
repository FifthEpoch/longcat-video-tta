#!/usr/bin/env bash
# Audit candidate pools + embeddings for N=1000 OOD-stratified budget/router sweep.
# Run on cluster login node; paste full output back for planning.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash scripts/cluster_audit_budget_1000v_pools.sh | tee /tmp/budget_1000v_audit.txt
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
cd "$REPO"

echo "========== Repo =========="
echo "PWD=$PWD"
git log -1 --oneline 2>/dev/null || true
echo ""

echo "========== Candidate pools (existence + size) =========="
POOLS=(
  "datasets/panda_segment_pool"
  "datasets/panda_pool_10k"
  "datasets/panda_2048_480p"
  "datasets/panda_1000_480p"
  "datasets/panda_ood_budget_pilot_480p"
)
for p in "${POOLS[@]}"; do
  echo "--- $p ---"
  if [ ! -d "$p" ]; then
    echo "  MISSING"
    continue
  fi
  n_meta=0
  [ -f "$p/metadata.csv" ] && n_meta=$(tail -n +2 "$p/metadata.csv" | wc -l | tr -d ' ')
  n_vid=0
  if [ -d "$p/videos" ]; then
    n_vid=$(find "$p/videos" -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  else
    n_vid=$(find "$p" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  fi
  du -sh "$p" 2>/dev/null || true
  echo "  metadata rows: $n_meta"
  echo "  mp4 count:     $n_vid"
  if [ -f "$p/caption_embeddings.npy" ]; then
    python3 - <<PY
import json, numpy as np
from pathlib import Path
p = Path("$p")
e = np.load(p / "caption_embeddings.npy")
print(f"  caption_embeddings.npy: shape={e.shape} dtype={e.dtype}")
j = p / "caption_embeddings.json"
if j.exists():
    print(f"  caption_embeddings.json: {j.read_text()[:200]}...")
PY
  else
    echo "  caption_embeddings.npy: MISSING"
  fi
  if [ -f "$p/validation_report.json" ]; then
    echo "  validation_report.json:"
    head -c 400 "$p/validation_report.json"; echo "..."
  fi
  echo ""
done

echo "========== OOD score CSVs (for stratified sampling) =========="
find sweep_experiment/reports/per_video_analysis -name 'diffusion_ood_scores.csv' 2>/dev/null \
  | sort | while read -r f; do
  n=$(tail -n +2 "$f" 2>/dev/null | wc -l | tr -d ' ')
  echo "  $f  rows=$n"
done
echo ""

echo "========== Existing budget / router series =========="
for d in sweep_experiment/results/panda_ood_budget_pilot \
         sweep_experiment/results/panda_1000v_adasteer_budget \
         sweep_experiment/results/panda_ood_budget_1000v; do
  echo "--- $d ---"
  if [ ! -d "$d" ]; then
    echo "  MISSING"
    continue
  fi
  ls -d "$d"/S* 2>/dev/null | wc -l | xargs echo "  S* run dirs:"
  for rid in S10_LR5e3 NOTTA; do
    if [ -d "$d/$rid" ]; then
      m=$(find "$d/$rid" -name 'merged_summary.json' 2>/dev/null | head -1)
      if [ -n "$m" ]; then
        python3 -c "import json; b=json.load(open('$m')); print('  $rid merged: n=', b.get('num_successful', b.get('num_videos')), ' psnr=', b.get('psnr'))"
      fi
    fi
  done
  echo ""
done

echo "========== VAE / feature artifacts (router inputs) =========="
FEAT_DIRS=(
  "sweep_experiment/reports/per_video_analysis/2026-07-06"
  "sweep_experiment/reports/per_video_analysis/2026-06-09"
)
for fd in "${FEAT_DIRS[@]}"; do
  echo "--- $fd ---"
  if [ ! -d "$fd" ]; then
    echo "  MISSING"; continue
  fi
  for f in video_features.csv diffusion_ood_scores.csv vae_latent_profile_features.csv; do
    if [ -f "$fd/$f" ]; then
      n=$(tail -n +2 "$fd/$f" | wc -l | tr -d ' ')
      echo "  $f: $n rows"
    else
      echo "  $f: MISSING"
    fi
  done
  echo ""
done

echo "========== VAE latent cache (cross-config reuse) =========="
CACHE_DIRS=(
  "datasets/panda_ood_budget_pilot_480p/vae_latent_cache"
  "datasets/panda_ood_budget_1000v_480p/vae_latent_cache"
  "datasets/panda_1000_480p/vae_latent_cache"
)
found=0
for c in "${CACHE_DIRS[@]}"; do
  if [ -d "$c" ]; then
    n=$(find "$c" -name '*.pt' -o -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    echo "  $c: $n cached files"
    found=1
  fi
done
if [ "$found" = "0" ]; then
  echo "  No vae_latent_cache dirs found (expected — not implemented yet in run_delta_a)"
fi
echo ""

echo "========== Pilot retain list =========="
for j in sweep_experiment/lists/panda_ood_budget_pilot_videos.json \
         sweep_experiment/lists/panda_ood_budget_1000v_videos.json; do
  if [ -f "$j" ]; then
    python3 -c "import json; b=json.load(open('$j')); print('$j: n=', b.get('n_selected', len(b.get('videos',[]))), ' quintiles=', b.get('quintile_counts'))"
  else
    echo "  $j: MISSING"
  fi
done
echo ""

echo "========== Slurm hints (recent pool/build jobs) =========="
ls -lt datasets/slurm_log/build_panda* 2>/dev/null | head -3 || echo "  (no build_panda logs)"
ls -lt sweep_experiment/slurm_log/deploy_*router* 2>/dev/null | head -3 || true
echo ""
echo "DONE — paste /tmp/budget_1000v_audit.txt"
