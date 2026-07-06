# Deploy router — structured feature blocks @ N=200

**Date:** 2026-07-07  
**Cluster:** `sweep_experiment/reports/per_video_analysis/2026-07-06/deploy_strict_router/`  
**Script:** `run_deploy_strict_router_experiments.py` · 5-fold OOF ridge · VBench total

## Feature blocks (concatenated)

| Block | Name | # dims (pilot CSV) | Source |
|-------|------|-------------------:|--------|
| A | `video_caption` | 9 | `video_features.csv` |
| B | `diffusion_ood` | **12** | `diffusion_ood_scores.csv` (pilot has 12 numeric cols, not full 20) |
| C | `vae_inference` | 130 | `vae_latent_profile_features.csv` |

**Excluded:** Tier-3, probe PSNR/SSIM, NOTTA/AdaSteer eval metrics, bpp/FFT aux.

## Results

| Experiment | Blocks | # feat | Captured % | Match % | Δ vs fixed S10 |
|---|---|---:|---:|---:|---:|
| **`video_caption_only`** | A | 9 | **20.8** | 18.5 | **+0.0291** |
| `video_caption_ood` | A+B | 21 | **18.9** | **21.0** | **+0.0265** |
| `vae_inference_embedding` | C | 130 | 9.7 | 16.5 | +0.0136 |
| `video_caption_ood_vae` | A+B+C | 151 | 10.1 | 19.5 | +0.0142 |
| `diffusion_ood_only` | B | 12 | 4.9 | 18.0 | +0.0069 |

Oracle headroom vs fixed @ N=200: **+0.140**.

## Conclusions

1. **Best captured headroom:** **Block A alone** (9-d video/caption stats) @ **20.8%** / **+0.0291** — **~2×** VAE-only (9.7%) and **~2×** the 51-d lab router (9.0%).
2. **When OOD is allowed (A+B):** **18.9%** / **+0.0265** — strong, but **slightly below A-only** on captured %; OOD **raises oracle-config match** (21.0% vs 18.5%).
3. **OOD alone (B):** weak (4.9%) — needs video/caption context.
4. **Stacking VAE (A+B+C):** **overfits** @ N=200 (10.1%) — do not deploy full stack.
5. Internal **>25%** bar: still **not met**; **20.8%** is closest so far.

## Deploy recommendation

| Policy | Blocks | When |
|--------|--------|------|
| **Default (cheapest)** | A only | No extra DiT forward; best captured % |
| **OOD-permitted** | A+B | Accept frozen DiT OOD pass; best match rate |
| ~~VAE-only~~ | C | Superseded by A for routing @ N=200 |
