# Deploy-strict router @ N=200 — VAE inference embedding ONLY (pending)

**Date:** 2026-07-06  
**Status:** **PENDING** — run on cluster after `git pull`

## Deploy rule (strict)

| Allowed at inference | Blocked |
|----------------------|---------|
| **LongCat VAE latent profile** (~130-d) from `encode_video` on input mp4 — same tensor AdaSteer already needs | CLIP/DINO/cuts/bpp/FFT/OOD/Tier-3 |
| Cached in `vae_latent_profile_features.csv` | AdaSteer/NOTTA metrics, probe PSNR/SSIM |
| Ridge → pick config → **one AdaSteer** | VAE decode rec-error, any DiT forward for features |

**Offline only (labels, not router inputs):** pilot VBench matrix over 12 configs — lab calibration, not deployed as features.

## Run on cluster

```bash
cd /scratch/wc3013/longcat-video-tta && git pull

# Extract VAE profiles if missing (encode-only, no TTA):
sbatch scripts/sbatch/run_extract_vae_latent_profile.sbatch
# OR: bash sweep_experiment/sbatch/submit_vae_latent_profile_pilot.sh  (extract job only)

bash sweep_experiment/sbatch/submit_deploy_strict_router.sh

cat sweep_experiment/reports/per_video_analysis/2026-07-06/deploy_strict_router/summary.md
```

## Result

| Experiment | # feat | Captured % | Match % | Δ vs fixed S10 |
|---|---:|---:|---:|---:|
| `vae_inference_embedding` | ~130 | *(pending)* | | |

Script: `scripts/run_deploy_strict_router_experiments.py`
