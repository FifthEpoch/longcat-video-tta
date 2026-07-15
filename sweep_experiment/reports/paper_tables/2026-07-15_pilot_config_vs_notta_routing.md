# Pilot: config vs no-TTA vs routing — pixel + VBench (N=200, same pool)

**Date:** 2026-07-15
**Series:** `panda_ood_budget_pilot` (200 OOD-stratified Panda clips, 40/quintile × 5)
**NOTTA:** joined by canonical id from `panda_1000v_standard/NOTTA` — **same 200 videos**
**Script:** `scripts/analyze_pilot_config_vs_notta_full.py` (commit `c413900`)
**Cluster path:** `per_video_analysis/2026-07-14/pilot_config_vs_notta_full.md`

All rows are on the identical 200 videos. FVD/FID are per-config population
values (`merged_summary.json`); NOTTA-subset / routing FVD need saved frames
(pilot ran `NO_SAVE_VIDEOS=1`). Routing rows are the **oracle** (per-video best
config) — deployable upper bound; the learned OOF router realizes ≈7.2% (PSNR)
/ ≈20.8% (VBench-raw) of the oracle gap.

> **VBench caveat:** the `VBench` column is the unweighted mean of 7 **raw**
> dims (dominated by imaging_quality/MUSIQ 0–100), **not** normalized VBench++
> (~0.77). NOTTA > TTA on this scale reflects imaging_quality dropping under
> adaptation — do not cross-compare with the slide's 0.772/0.802.

## Table 1 — Population (N=200)

| Policy | PSNR (dB) | SSIM | LPIPS | VBench(raw) | FVD | FID |
|---|---:|---:|---:|---:|---:|---:|
| No-TTA (NOTTA) | 17.798 | 0.6348 | 0.3436 | 9.993 | — | — |
| Fixed AdaSteer (S10/5e-3) | 17.996 | 0.6355 | 0.3474 | 9.398 | 331.2 | 63.4 |
| Best single config (S2/5e-3) | 18.120 | 0.6380 | 0.3450 | 9.392 | 320.9 | 61.8 |
| **Oracle — PSNR routing** | **18.744** | **0.6503** | **0.3297** | 9.414 | — | — |
| Oracle — VBench routing | 18.023 | 0.6343 | 0.3480 | 9.538 | — | — |

**Paired Δ vs oracle-PSNR routing (N=200):** +0.946 dB over no-TTA, +0.748 over
fixed S10, +0.62–0.86 over every one of the 12 fixed configs.

## Table 2 — PSNR by OOD quintile (Q1 in-dist → Q5 most OOD)

| Quintile | N | No-TTA | Best fixed (mean) | Oracle route | Δ route−NOTTA | Δ route−bestfixed |
|---|---:|---:|---|---:|---:|---:|
| Q1 | 40 | 18.546 | `S20_LR1e2` (18.816) | 19.161 | +0.615 | +0.345 |
| Q2 | 40 | 18.967 | `S10_LR1e2` (19.130) | 19.419 | +0.452 | +0.290 |
| Q3 | 40 | 18.490 | `S2_LR1e2` (19.794) | 20.171 | +1.682 | +0.377 |
| Q4 | 40 | 18.508 | `S2_LR5e3` (18.313) | 18.749 | +0.241 | +0.436 |
| Q5 | 40 | 14.478 | `S2_LR1e2` (15.466) | 16.221 | +1.743 | +0.755 |
| All | 200 | 17.798 | — | 18.744 | +0.946 | — |

In Q4 the best fixed config (18.313) is *below* no-TTA (18.508) — several fixed
configs actively hurt — yet oracle routing still nets +0.241. The best fixed
config **rotates by quintile** (S20→S10→S2→S2→S2): no single budget wins.

## Headline findings

1. **The AdaSteer gain lives in per-video config selection, not any fixed
   budget.** Fixed S10 is only +0.20 dB over no-TTA; per-video oracle routing is
   **+0.95 dB over no-TTA** and **+0.75 dB over the best fixed config**.
2. **PSNR routing co-improves SSIM (+0.0155) and LPIPS (−0.0139)** vs no-TTA —
   not a metric trade-off.
3. **No single config is best across OOD strata**, and mis-budgeted fixed TTA
   can underperform no-TTA (Q4) — motivating a router.
4. **Objective still matters:** VBench(raw) routing recovers VBench-raw headroom
   but leaves PSNR near fixed (18.023), consistent with the PSNR/VBench oracle
   decoupling seen at 999v.

## Reproduce

```bash
git pull --ff-only origin main  # >= c413900
python3 scripts/analyze_pilot_config_vs_notta_full.py \
    --series-root sweep_experiment/results/panda_ood_budget_pilot \
    --baseline-series-root sweep_experiment/results/panda_1000v_standard \
    --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \
    --output sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/pilot_config_vs_notta_full.md
```
