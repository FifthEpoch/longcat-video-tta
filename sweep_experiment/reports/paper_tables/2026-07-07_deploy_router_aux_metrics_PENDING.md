# Deploy router cross-metric analysis — PENDING cluster run

**Date:** 2026-07-07  
**Script:** `scripts/analyze_deploy_router_aux_metrics.py`  
**Submit:** `bash sweep_experiment/sbatch/submit_deploy_router_aux_metrics.sh`

## Question

VBench routers pick step×LR from pre-adaptation features. Do the **selected configs**
also improve PSNR / SSIM / LPIPS (and FVD/FID) vs fixed S10?

## Method

1. Re-run 5-fold OOF ridge config pickers (`video_caption_only`, `vae_inference_embedding`).
2. For each video, look up metrics from **existing** pilot outputs for the picked config.
3. Report population mean vs fixed / NOTTA / oracles.
4. **PSNR/SSIM/LPIPS captured %** = (policy − fixed) / (oracle − fixed) on that metric.
5. **FVD/FID:** build symlink policy dirs → `eval_fvd.py` (requires saved mp4s).

## Expected output

`per_video_analysis/2026-07-06/deploy_router_aux_metrics/summary.md`

## Hypotheses to test

| If true | Implication |
|---------|-------------|
| Video/caption router captures PSNR headroom | VBench routing aligns with reconstruction |
| VBench captured % ≫ PSNR captured % | Router optimizes perceptual/aesthetic dims, not pixels |
| LPIPS moves opposite PSNR | Config grid trades sharpness vs fidelity |
| Router FVD ≈ fixed, < oracle PSNR FVD | Honest deploy ceiling on distributional quality |

**Results:** *(fill after cluster run)*
