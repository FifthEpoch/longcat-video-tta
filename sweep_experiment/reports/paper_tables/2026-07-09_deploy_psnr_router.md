# Deploy PSNR router @ N=200 — 9-d Block A, PSNR target

**Date:** 2026-07-09  
**Series:** `panda_ood_budget_pilot`  
**Cluster path:** `per_video_analysis/2026-07-06/deploy_psnr_router/summary.md`  
**Script:** `scripts/run_deploy_psnr_router.py` (commit `90c2ead`)

## Setup

Same **9-d** deploy features as `video_caption_only` (cuts, CLIP, DINO, Laplacian, RGB).
**Change:** ridge labels = **PSNR per config** (not VBench total). Deploy: argmax predicted PSNR → one AdaSteer. 5-fold OOF @ N=200.

## Results

| Metric | PSNR-targeted router | VBench-targeted router (prior) |
|--------|---------------------|-------------------------------|
| **Δ PSNR vs fixed S10** | **+0.0539 dB** | +0.009 dB |
| **PSNR oracle captured %** | **7.2%** | 1.2% |
| **VBench oracle captured %** | 5.6% (side effect) | **20.8%** |
| Oracle-config match rate | 15.5% (PSNR oracle) | 18.5% (VB oracle) |
| Mean PSNR (policy / fixed / oracle) | 18.050 / 17.996 / 18.744 dB | — |

## Conclusions

1. **Objective was the bottleneck for PSNR, not the 9-d input format.** Same features, PSNR target → **6×** PSNR lift vs VBench target (+0.054 vs +0.009 dB; 7.2% vs 1.2% PSNR cap).

2. **You cannot optimize both with one 9-d router.** PSNR routing sacrifices VBench capture (5.6% vs 20.8%). Pick the deploy objective explicitly.

3. **9-d PSNR routing is still weak in absolute terms.** 7.2% of +0.748 dB PSNR oracle headroom; 15.5% PSNR-oracle match. Features carry *some* PSNR signal but not enough for a strong PSNR headline.

4. **Paper recommendation:** Keep **VBench-targeted** Block A as headline (perceptual deploy story). PSNR-targeted router is an ablation showing objective tradeoff — not a replacement.

## Reproduce

```bash
git pull --ff-only origin main  # ≥ 90c2ead
bash sweep_experiment/sbatch/submit_deploy_psnr_router.sh
```
