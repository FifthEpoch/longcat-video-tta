# Spec — 16v NOTTA 5 s vs 30 s VBench++ (first-16 / last-16)

**Date:** 2026-08-19
**Series:** `i2v_notta_16v` (already generated; no new videos)
**Status:** SUBMIT-READY. No numbers in this file.

## Why

Handpicked drift (`score_i2v_drift.py`) already compared first 16
frames vs last 16 frames (skip cond frame 0) at both horizons:
5 s median sharp +11% / motion −14%; 30 s sharp +167% / motion −60%.
Those are GT-free signals, not VBench++. This job asks whether the
official quality dimensions show the same 5 s vs 30 s stress.

## Windows

| Clip | Frames | Role |
|---|---|---|
| `full` | all (~85 / ~481) | Official comparable number at 5 s. 30 s full is long-clip diagnostic. |
| `first5` | first 5 s (includes f0) | Same-duration pair: 5 s full vs 30 s first5. |
| `first1` | frames `[1:17]` (16 fr, skip f0) | Same head as the handpicked table. Diagnostic. |
| `last1` | last 16 frames | Same tail as the handpicked table. Diagnostic. |

`first1` / `last1` are **not** official VBench++. VBench quality
dims are designed for ~5 s clips. 16-frame scores can be noisy,
especially `dynamic_degree` (0/1 RAFT) and `motion_smoothness`.

5 s and 30 s were **separate generates** (same 16 images, seed 0).
`first1` is not a shared prefix.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_vbench_notta16.sh
```

Env: `vbench-backfill`. Existing `joined.json` is skipped.

## Analyze (after the job)

```bash
python wan_experiment/scripts/analyze_i2v_vbench_horizon.py \
    --series-dir wan_experiment/results/i2v_notta_16v \
    --out sweep_experiment/reports/paper_tables/$(date +%F)_wan_i2v_notta16_vbench_headtail.md
```

Cite medians. Do not invent PSNR. Do not replace the hybrid-32
full-clip VBench++ table with these 16v diagnostics.
