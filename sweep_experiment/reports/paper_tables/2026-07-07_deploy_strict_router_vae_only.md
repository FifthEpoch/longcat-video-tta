# Deploy-strict router — VAE inference embedding only @ N=200

**Date:** 2026-07-07  
**Series:** `panda_ood_budget_pilot` (12 AdaSteer configs, 200 OOD-stratified videos)  
**Cluster path:** `sweep_experiment/reports/per_video_analysis/2026-07-06/deploy_strict_router/`  
**Script:** `scripts/run_deploy_strict_router_experiments.py` · experiment `vae_inference_embedding`

## Deploy rule

| Router input | Allowed at inference |
|--------------|----------------------|
| `vae_latent_profile_features.csv` (~130-d) | **Yes** — pooled stats from LongCat `encode_video` on TTA-visible window (same as inference) |
| CLIP/DINO/cuts/bpp/FFT/OOD/Tier-3/probe/TTA metrics | **No** |

Offline labels: pilot 12-config VBench matrix (calibration only; not router features).

## Result (5-fold OOF ridge, VBench total)

| Experiment | # feat | Captured % | Oracle match % | Δ vs fixed S10 |
|---|---:|---:|---:|---:|
| **`vae_inference_embedding`** | **130** | **9.7** | **16.5** | **+0.0136** |

Oracle headroom vs fixed S10 @ N=200: **+0.140** → recovered gap ≈ **0.097 × 0.140 ≈ +0.0136**.

## Comparison

| Router | Deploy inputs | Captured % | Δ vs fixed |
|--------|---------------|------------|------------|
| Lab Phase-0 linear (`baseline_linear_total`, 51-d) | Video + OOD DiT + Tier-3 LoRA + aux | 9.0 | +0.013 |
| **Deploy VAE-only (`vae_inference_embedding`)** | **VAE encode profile only** | **9.7** | **+0.0136** |
| Prior `vae_profile_probe` (Jul 6) | VAE + AdaSteer probe PSNR/SSIM | 12.2 | — (not deploy-strict) |
| Prior `vae_profile_full` stacked | Phase-0 + VAE + probe | 4.2 | overfit |

## Conclusion

**Headline deploy router:** cache VAE latent profile at inference → ridge pick config → one AdaSteer. No extra DiT/LoRA/probe passes beyond what inference already requires. Performance **≥** the heavier 51-d lab bundle while meeting the strict deploy bar. Internal >25% headroom bar still **not met** (9.7%).
