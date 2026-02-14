#!/bin/bash
# ==========================================================================
# Prune & Evaluate UCF101 Baseline Sweep Results
# ==========================================================================
# Run on the cluster login node with the longcat conda env active:
#   conda activate /scratch/wc3013/conda-envs/longcat
#   cd /scratch/wc3013/longcat-video-tta
#   bash baseline_experiment/scripts/prune_ucf101_sweep.sh
# ==========================================================================
set -euo pipefail

REPO="/scratch/wc3013/longcat-video-tta"
RESULTS="${REPO}/baseline_experiment/results"
KEEP_LIST="${RESULTS}/ucf101_cond2_gen14/keep_videos.txt"

cd "${REPO}"

# ── Verify keep list exists ──────────────────────────────────────────────
if [ ! -f "${KEEP_LIST}" ]; then
    echo "ERROR: keep_videos.txt not found at ${KEEP_LIST}"
    echo "       Run prune_and_summarize.py --create-keep-list on ucf101_cond2_gen14 first."
    exit 1
fi
echo "Using keep list: ${KEEP_LIST}"
echo "Contents: $(wc -l < "${KEEP_LIST}") video indices"
echo ""

# ── Step 1: Prune the 4 remaining runs ──────────────────────────────────
echo "=============================================================="
echo "Step 1: Pruning 4 UCF101 sweep configs"
echo "=============================================================="

for CFG in ucf101_cond2_gen7 ucf101_cond2_gen28 ucf101_cond14_gen14 ucf101_cond28_gen14; do
    DIR="${RESULTS}/${CFG}"
    if [ ! -d "${DIR}" ]; then
        echo "  SKIP: ${CFG} -- directory not found"
        continue
    fi
    if [ ! -f "${DIR}/per_video_metrics.csv" ]; then
        echo "  SKIP: ${CFG} -- per_video_metrics.csv not found"
        continue
    fi
    echo ""
    echo "  Pruning ${CFG} ..."
    python baseline_experiment/scripts/prune_and_summarize.py \
        --results-dir "${DIR}" \
        --keep-list "${KEEP_LIST}"
done

# Also regenerate RESULTS.md for the first run (ucf101_cond2_gen14)
echo ""
echo "  Regenerating RESULTS.md for ucf101_cond2_gen14 ..."
python baseline_experiment/scripts/prune_and_summarize.py \
    --results-dir "${RESULTS}/ucf101_cond2_gen14" \
    --keep-list "${KEEP_LIST}"

echo ""
echo "Step 1 complete."
echo ""

# ── Step 2: Prune GT clips (if they exist) ──────────────────────────────
echo "=============================================================="
echo "Step 2: Pruning GT clips (if they exist)"
echo "=============================================================="

python3 << 'PYEOF'
from pathlib import Path
import re

keep_path = '/scratch/wc3013/longcat-video-tta/baseline_experiment/results/ucf101_cond2_gen14/keep_videos.txt'
keep = set(int(l.strip()) for l in open(keep_path) if l.strip())
gt_dir = Path('/scratch/wc3013/longcat-video-tta/baseline_experiment/results/gt_clips_ucf101')
if gt_dir.exists():
    deleted = 0
    for mp4 in sorted(gt_dir.glob('*.mp4')):
        m = re.match(r'^(\d+)_', mp4.name)
        if m and int(m.group(1)) not in keep:
            mp4.unlink()
            deleted += 1
    remaining = len(list(gt_dir.glob('*.mp4')))
    print(f'  Pruned GT clips: deleted {deleted}, kept {remaining}')
else:
    print('  No GT clips directory found -- skipping')
PYEOF

echo ""

# ── Step 3: Print aggregate metrics for all 5 configs ───────────────────
echo "=============================================================="
echo "Step 3: Aggregate metrics for all 5 UCF101 configs"
echo "=============================================================="
echo ""

python3 << 'PYEOF'
import json
from pathlib import Path

configs = [
    'ucf101_cond2_gen7',
    'ucf101_cond2_gen14',
    'ucf101_cond2_gen28',
    'ucf101_cond14_gen14',
    'ucf101_cond28_gen14',
]
root = Path('/scratch/wc3013/longcat-video-tta/baseline_experiment/results')

header = f"{'Config':<25s}  {'PSNR':>16s}  {'SSIM':>16s}  {'LPIPS':>16s}"
print(header)
print('-' * 80)

def fmt(d):
    if not d:
        return '       N/A      '
    mean = d.get('mean', 0)
    std = d.get('std', 0)
    return f'{mean:7.2f} +/- {std:5.2f}'

for cfg in configs:
    sp = root / cfg / 'summary.json'
    if not sp.exists():
        print(f'{cfg:<25s}  summary.json not found')
        continue
    with open(sp) as f:
        s = json.load(f)
    m = s.get('metrics', {})
    psnr = m.get('psnr', {})
    ssim = m.get('ssim', {})
    lpips = m.get('lpips', {})
    print(f'{cfg:<25s}  {fmt(psnr)}  {fmt(ssim)}  {fmt(lpips)}')
PYEOF

echo ""

# ── Step 4: Generate UCF101 sweep plot ──────────────────────────────────
echo "=============================================================="
echo "Step 4: Generating UCF101 sweep comparison plots"
echo "=============================================================="

python baseline_experiment/scripts/plot_baseline_sweep.py \
    --results-root "${RESULTS}" \
    --prefix ucf101

echo ""
echo "=============================================================="
echo "All done!"
echo "=============================================================="
