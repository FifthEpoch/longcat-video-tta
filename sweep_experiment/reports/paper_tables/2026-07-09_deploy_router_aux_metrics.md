# Deploy router cross-metric analysis @ N=200

**Date:** 2026-07-09  
**Series:** `panda_ood_budget_pilot` (200 OOD-stratified videos, 12 AdaSteer configs)  
**Cluster path:** `per_video_analysis/2026-07-06/deploy_router_aux_metrics/summary.md`  
**Script:** `scripts/analyze_deploy_router_aux_metrics.py` (commit `7eed702`)

## Method

5-fold OOF ridge config pickers (same as deploy router). For each video, look up
PSNR/SSIM/LPIPS from **existing** outputs for the router-selected config.
**No new generation.** VBench labels used for routing only.

**Captured % (per metric)** = (policy − fixed) / (oracle − fixed) on that metric.

## Results — PSNR / SSIM / LPIPS

| Policy | VB captured % | PSNR (dB) | Δ vs fixed | PSNR cap % | SSIM Δ | LPIPS Δ | ρ(VB gain, ΔPSNR) |
|--------|-------------:|----------:|-----------:|-----------:|-------:|--------:|------------------:|
| Fixed S10_LR5e3 | 0% | 17.996 | — | 0% | 0 | 0 | — |
| Oracle VBench | — | 18.023 | +0.027 | 3.5% | −0.0012 | +0.0006 | — |
| Oracle PSNR | — | 18.744 | +0.748 | 100% | +0.0148 | −0.0178 | — |
| **Router video/caption (A)** | **20.8%** | 18.005 | **+0.009** | **1.2%** | −0.0019 | +0.0013 | **0.10** |
| Router VAE pooled (C) | 9.7% | 17.950 | −0.046 | −6.2% | −0.0001 | +0.0012 | 0.04 |

Population FVD/FID (fixed S10 only in pilot merged_summary): FVD **331.2**, FID **63.4**, N=200.
Router FVD pending (`RUN_FVD=1` + saved mp4s).

## Headline findings

1. **VBench routing decouples from PSNR.** Block A recovers **20.8%** of VBench oracle headroom but only **1.2%** of PSNR oracle headroom (+0.009 dB vs fixed).

2. **Even the VBench oracle barely moves PSNR** (+0.027 dB, 3.5% PSNR cap). PSNR oracle headroom is **+0.748 dB** — the 12-config grid optimizes different objectives.

3. **Per-video VBench gains do not track ΔPSNR** (Spearman ρ = **0.10** Block A, 0.04 Block C).

4. **VAE router hurts reconstruction** (−0.046 dB PSNR vs fixed) while still gaining some VBench (9.7%).

5. **SSIM/LPIPS:** routers slightly worse than fixed on both; oracle PSNR improves SSIM (+0.0148) and LPIPS (−0.0178) — config choice trades metrics.

## Paper narrative (honest)

- **Claim:** Pre-adaptation routing improves **VBench total** with one AdaSteer run (~+3.8% vs fixed in VB units).
- **Do not claim:** Pixel fidelity gains — routing is not a PSNR play in this pilot.
- **Mechanism:** Step×LR grid shifts perceptual/aesthetic VBench dimensions; PSNR headroom lives in a different config subset (oracle PSNR ≠ oracle VBench).

## Reproduce

```bash
git pull --ff-only origin main  # ≥ 7eed702
bash sweep_experiment/sbatch/submit_deploy_router_aux_metrics.sh
```
