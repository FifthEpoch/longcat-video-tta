# Deploy router — structured blocks (video/caption + OOD) — PENDING

**Date:** 2026-07-07  
**Run on cluster after pull:** `bash sweep_experiment/sbatch/submit_deploy_strict_router.sh`

## Feature space (ordered concatenation → ridge)

| Block | Name | # dims | Source | Contents |
|-------|------|-------:|--------|----------|
| **A** | `video_caption` | 9 | `video_features.csv` | cuts (×3), CLIP text–image sim (×3), DINO temporal L2, Laplacian var, RGB entropy |
| **B** | `diffusion_ood` | ~20 | `diffusion_ood_scores.csv` | per-t caption/uncond loss + score norm @ t∈{100,500,900}; mean losses, Δ cap−uncond, latent norm stats, mean score norms |
| **C** | `vae_inference` | ~130 | `vae_latent_profile_features.csv` | ctx/tgt/full VAE latent pools (optional third block) |

**Excluded:** Tier-3 LoRA, probe PSNR/SSIM, NOTTA/AdaSteer eval metrics, bpp/FFT/motion aux.

## Experiments

| ID | Blocks | Role |
|----|--------|------|
| `video_caption_only` | A | Video+caption baseline |
| `diffusion_ood_only` | B | OOD-only ablation |
| **`video_caption_ood`** | **A+B** | **Headline when OOD allowed (~29-d)** |
| `vae_inference_embedding` | C | Prior result: 9.7%, +0.0136 |
| `video_caption_ood_vae` | A+B+C | Full stack (~159-d) |

## Results

*(paste from `deploy_strict_router/summary.md` after cluster run)*
