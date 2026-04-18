# Experiment Metrics Log

**Purpose:** Running record of all experiment outcomes, configurations, and observations.
Intended for paper writing, debugging metric regressions, and cross-referencing results.

**Model:** LongCat-Video (13.6B DiT + AutoencoderKLWan VAE)
**Resolution:** 480p (480x832), 50 inference steps, seed=42
**Cluster:** NYU Greene HPC, H200 GPUs

---

## Table of Contents

1. [Datasets](#datasets)
2. [Known Issues and Caveats](#known-issues-and-caveats)
3. [Apr 2026 - Delta Vector Ablation (Panda-70M)](#apr-2026---delta-vector-ablation-panda-70m)
4. [Apr 2026 - LoRA Step Sweep](#apr-2026---lora-step-sweep)
5. [Apr 2026 - CLIP-Gate Threshold Sweep](#apr-2026---clip-gate-threshold-sweep)
6. [Apr 2026 - LoRA Timing and LR Sanity](#apr-2026---lora-timing-and-lr-sanity)
7. [Feb 2026 - Panda-70M (panda_100_480p)](#feb-2026---panda-70m-panda_100_480p)
8. [Feb 2026 - UCF-101 (100 videos)](#feb-2026---ucf-101-100-videos)
9. [Comparison Methods](#comparison-methods)
10. [Metric Regression Analysis](#metric-regression-analysis)
11. [TinyLoRA Experiments - Status](#tinylora-experiments---status)

---

## Datasets

| Dataset | Path on Cluster | Videos | Notes |
|---------|----------------|--------|-------|
| Panda-70M (1000) | datasets/panda_1000_480p | 1000 | Current primary set. max_videos=100 selects first 100. |
| Panda-70M (100) | datasets/panda_100_480p | 100 | Original small subset. DIFFERENT videos from panda_1000_480p. |
| UCF-101 (100) | datasets/ucf101_test_480p | 100 | Action recognition clips. |
| UCF-101 (500) | datasets/ucf101_test_480p | 500 | Larger UCF eval (Phase 2). |

CRITICAL: panda_1000_480p and panda_100_480p contain DIFFERENT videos with different difficulty distributions. Results are NOT directly comparable across these datasets.

---

## Known Issues and Caveats

### Gen-Start-Frame Misalignment Bug (fixed Feb 13, 2026)

Commit 3ef900f fixed a critical bug where the baseline experiment loaded frames starting at index 0 instead of using the fixed anchor (gen_start_frame=32). This caused:
- Pre-fix No-TTA baseline on panda_100_480p: PSNR=14.98 (INCORRECT -- metrics computed on wrong GT frames)
- Post-fix No-TTA baseline on panda_100_480p: PSNR=22.07 (correct)

The paper figures generated in late February used the pre-fix baseline (14.98) compared against post-fix TTA results (~22.6), creating an artificial +7.6 dB PSNR improvement. The actual improvement was ~+0.5 dB. See Metric Regression Analysis for details.

### gen_start_frame Change: 32 to 48

Early experiments used gen_start_frame=32. Later configs (including the current ablation study) use gen_start_frame=48 and tta_total_frames=48. This gives the model more context but changes the evaluation window. Results across these settings are not directly comparable.

---

## Apr 2026 - Delta Vector Ablation (Panda-70M)

**Dataset:** panda_1000_480p, first 100 videos
**Config base:** gen_start_frame=48, tta_total_frames=48, tta_context_frames=14, num_cond_frames=14, num_frames=28, guidance_scale=4.0
**Results path:** sweep_experiment/results/sanity_100/panda_adasteer_ablation/
**Config file:** sweep_experiment/configs/panda_adasteer_ablation.yaml
**Date:** April 16-17, 2026

All runs: delta_steps=10, delta_lr=0.005, method=delta_a

| Run ID | Features | PSNR | SSIM | LPIPS | FVD | FID | Train(s) | Total(s) | dFVD |
|--------|----------|------|------|-------|-----|-----|----------|----------|------|
| NOTTA_G4 | No TTA (baseline) | 18.612 | 0.6824 | 0.3201 | 641.1 | 76.1 | 0.0 | 79.9 | -- |
| AS_BARE | Bare (no ES/CLIP/Aug) | 18.604 | 0.6837 | 0.3166 | 561.1 | 76.2 | 54.1 | 133.6 | -80.0 |
| AS_ES1 | Early stopping (check=1, pat=2) | 18.487 | 0.6812 | 0.3186 | 555.8 | 75.9 | 106.4 | 186.1 | -85.3 |
| AS_AUG | Augmentation (flip+rotate) | 18.663 | 0.6816 | 0.3182 | 587.2 | 76.6 | 57.3 | 137.1 | -53.9 |
| AS_CLIP | CLIP gate (threshold=0.25) | 18.600 | 0.6816 | 0.3201 | 647.9 | 76.6 | 5.4 | 85.5 | +6.8 |
| AS_ES_CLIP | ES + CLIP (0.25) | 18.591 | 0.6816 | 0.3201 | 645.5 | 76.7 | 10.8 | 90.6 | +4.4 |
| AS_FULL | All features (ES+CLIP+Aug) | 18.598 | 0.6818 | 0.3204 | 666.4 | 76.4 | 11.1 | 91.0 | +25.3 |
| AS_GA2 | Gradient accum 2x | 18.507 | 0.6808 | 0.3195 | 662.1 | 76.1 | 108.8 | 188.3 | +21.0 |
| AS_GA4 | Gradient accum 4x | -- | -- | -- | -- | -- | -- | TIMEOUT 8h | -- |

Key findings:
- AS_BARE is the best config: FVD -80.0, cleanest cost/quality tradeoff
- ES1 gains marginal FVD (-85 vs -80) but doubles training time (106s vs 54s)
- Augmentation helps PSNR (+0.05) but hurts FVD (587 vs 561)
- CLIP gate at threshold=0.25 skips 90% of videos, worse than baseline
- Gradient accumulation adds cost with no benefit; GA4 timed out at 8 hours

### Previous Best Delta Vector (different result path, same dataset)

Results path: sweep_experiment/results/sanity_100/panda_adasteer_g0/S10_LR005/
Date: April 16, 2026

| Run | Steps | LR | G | PSNR | SSIM | LPIPS | FVD | FID | Train(s) | Total(s) |
|-----|-------|----|---|------|------|-------|-----|-----|----------|----------|
| S10_LR005 | 10 | 0.005 | 4.0 | 18.590 | 0.6841 | 0.3164 | 568.7 | 74.1 | 54.6 | 134.4 |
| S5_LR005 | 5 | 0.005 | 4.0 | 18.588 | 0.6828 | 0.3194 | 571.2 | 75.2 | 27.3 | 107.3 |

---

## Apr 2026 - LoRA Step Sweep

**Dataset:** panda_1000_480p, first 100 videos
**Config:** lora_rank=8, lora_alpha=16, target_modules=qkv,proj, all blocks, no ES/CLIP
**Results path:** sweep_experiment/results/sanity_100/panda_lora_step_sweep/
**Date:** April 17, 2026

| Run ID | Steps | LR | PSNR | SSIM | LPIPS | FVD | FID | Train(s) | Total(s) | dFVD |
|--------|-------|----|------|------|-------|-----|-----|----------|----------|------|
| LORA_S5_LR1e-5 | 5 | 1e-5 | 18.591 | 0.6823 | 0.3207 | 651.0 | 77.3 | 9.2 | 88.7 | +9.9 |
| LORA_S5_LR5e-5 | 5 | 5e-5 | 18.609 | 0.6818 | 0.3211 | 651.3 | 76.4 | 9.3 | 89.5 | +10.2 |
| LORA_S10_LR1e-5 | 10 | 1e-5 | 18.604 | 0.6823 | 0.3205 | 660.0 | 77.4 | 18.4 | 99.3 | +18.9 |
| LORA_S10_LR5e-5 | 10 | 5e-5 | 18.616 | 0.6818 | 0.3205 | 644.6 | 76.9 | 18.5 | 99.2 | +3.5 |

Key finding: LoRA at 5/10 steps does not meaningfully improve FVD. Best LoRA (S10 LR=5e-5, FVD=644.6) barely matches baseline (641.1). Delta Vector clearly outperforms LoRA regardless of step count.

### Previous LoRA Results (20 steps)

Results path: sweep_experiment/results/sanity_100/panda_lora_lr_sanity/
Date: April 7, 2026

| Run | Steps | LR | PSNR | SSIM | LPIPS | FVD | FID | Train(s) | Total(s) |
|-----|-------|----|------|------|-------|-----|-----|----------|----------|
| LR1e5_S20 | 20 | 1e-5 | 18.569 | 0.6821 | 0.3201 | 641.5 | 77.2 | 37.3 | 118.5 |
| LR5e5_S20 | 20 | 5e-5 | 18.540 | 0.6808 | 0.3216 | 676.6 | 77.7 | 36.9 | 118.4 |
| LR2e4_S20 | 20 | 2e-4 | 17.441 | 0.6332 | 0.3582 | 672.5 | 85.7 | 36.8 | 117.0 |

Also: panda_lora_ablation/LORA_noES_noCLIP (10 steps, lr=2e-5): PSNR=18.575, FVD=668.4, train=55.9s

---

## Apr 2026 - CLIP-Gate Threshold Sweep

**Dataset:** panda_1000_480p, first 100 videos
**Config base:** Delta Vector, delta_steps=10, delta_lr=0.005, guidance_scale=4.0
**Results paths:**
- sanity_100/panda_adasteer_ablation/AS_CLIP/ (threshold=0.25)
- panda_adasteer_ablation/AS_CLIP_T15/ (threshold=0.15)
- panda_adasteer_ablation/AS_CLIP_T10/ (threshold=0.10)
**Date:** April 17, 2026

| Run | Threshold | Skipped | PSNR | FVD | FID | Train(s) | Total(s) | dFVD |
|-----|-----------|---------|------|-----|-----|----------|----------|------|
| AS_BARE | None | 0% | 18.604 | 561.1 | 76.2 | 54.1 | 133.6 | -80.0 |
| AS_CLIP_T10 | 0.10 | 7% | 18.615 | 598.6 | 74.7 | 50.4 | 130.2 | -42.5 |
| AS_CLIP_T15 | 0.15 | 33% | 18.603 | 578.0 | 77.1 | 36.6 | 116.8 | -63.1 |
| AS_CLIP | 0.25 | 90% | 18.600 | 647.9 | 76.6 | 5.4 | 85.5 | +6.8 |

Key finding: No gate is best for FVD. Threshold=0.15 saves ~17s train time with modest FVD loss (578 vs 561). Threshold=0.25 was far too aggressive.

---

## Feb 2026 - Panda-70M (panda_100_480p)

**Dataset:** panda_100_480p, 99 videos (1 failed)
**Config:** gen_start_frame=32, num_cond_frames=14, num_frames=28, guidance_scale=4.0, 20 steps
**Date:** February 14-28, 2026
**Source:** all_results.json (removed from git Mar 1 in commit 393d7c4)

### Best per-method (standard config: 20 steps, 14 cond, 28 gen)

| Method | Series/Run | Params | PSNR | SSIM | LPIPS | Train(s) | Baseline PSNR |
|--------|-----------|--------|------|------|-------|----------|---------------|
| No TTA | panda_no_tta_continuation/NOTTA | -- | 22.067 | 0.768 | 0.236 | 0.0 | -- |
| Delta Vector (B) | delta_b_low_lr/DBL4 | 512 | 22.588 | 0.746 | 0.221 | 147.1 | 22.067 |
| LoRA | lora_ultra_constrained/LB4 | 73,728 | 22.641 | -- | -- | 37.1 | 22.067 |
| Full-model | full_lr_sweep/F3 | 13.6B | 22.073 | -- | -- | 24.0 | 22.067 |

Actual PSNR gains over correct baseline:
- Delta Vector: +0.52 PSNR
- LoRA: +0.57 PSNR
- Full-model: +0.01 PSNR

FVD/FID were NOT computed for these February runs.

---

## Feb 2026 - UCF-101 (100 videos)

**Dataset:** ucf101_test_480p, 100 videos
**Config:** gen_start_frame=32, num_cond_frames=14, num_frames=28, guidance_scale=4.0
**Results path:** sweep_experiment/results/sanity_100/ucf_adasteer_diagnostic/
**Date:** April 10, 2026

| Run | Steps | LR | PSNR | SSIM | LPIPS | FVD | FID | dPSNR | dFVD |
|-----|-------|----|------|------|-------|-----|-----|-------|------|
| NOTTA (baseline) | 0 | -- | 19.937 | 0.724 | 0.252 | 590.7 | 47.3 | -- | -- |
| ADA_5s_lr1e3 | 5 | 0.001 | 19.930 | 0.724 | 0.252 | 583.3 | 47.6 | -0.007 | -7.4 |
| ADA_5s_lr5e3 | 5 | 0.005 | 19.950 | 0.724 | 0.252 | 585.7 | 47.4 | +0.013 | -5.0 |
| ADA_5s_lr1e2 | 5 | 0.01 | 19.892 | 0.722 | 0.254 | 594.5 | 48.2 | -0.045 | +3.8 |

Key finding: TTA gains on UCF-101 are much smaller than on Panda-70M. Best is -7.4 FVD (vs -80 on Panda).

---

## Comparison Methods

### SAVi-DNO on LongCat Backbone

Dataset: panda_1000_480p, 100 videos
Results path: comparison_methods/results/savi_dno_longcat_panda100_s10/
Date: April 11, 2026

PSNR=7.451, SSIM=0.046, LPIPS=0.980, FVD=4724.2, FID=430.7

Catastrophic failure. Hyperparameter mismatch (designed for PVDM 87M UNet, not LongCat 13.6B DiT). Fix implemented, awaiting re-run with lr=1e-4, latent loss, gradient clipping, 10 rollout steps.

### SAVi-DNO on PVDM (original backbone)

Dataset: Original PVDM dataset, 500 videos
Results path: comparison_methods/results/savi_dno_s10/, savi_dno_s50/

| Steps | PSNR | SSIM |
|-------|------|------|
| 10 | 18.766 | 0.612 |
| 50 | 18.252 | 0.590 |

---

## Metric Regression Analysis

Date of analysis: April 17, 2026

### The Problem

Paper figures from late February showed:
- No-TTA PSNR = 14.98, AdaSteer PSNR = 22.59 -> apparent +7.6 dB gain
- Current results show No-TTA = 18.61, Delta Vector = 18.60 -> essentially 0 gain

### Root Cause: Two Separate Issues

Issue 1: Misleading baseline in old figures (primary cause of apparent regression)

The generate_figures.py script selected No-TTA from baseline_experiment/results/cond14_gen14 (PSNR=14.98), which was computed BEFORE the gen_start_frame alignment fix on Feb 13 (3ef900f). The TTA results were computed AFTER the fix. The correct No-TTA baseline was panda_no_tta_continuation/NOTTA (PSNR=22.07). Real gain was +0.52 dB, not +7.6 dB.

Issue 2: Different dataset

Old results used panda_100_480p (No-TTA PSNR=22.07). Current results use panda_1000_480p (No-TTA PSNR=18.61). The current dataset is harder. On the current dataset, per-video PSNR/SSIM gains are near zero, but FVD improves by -80 points (distributional quality gain).

### Summary

| Comparison | Dataset | NoTTA PSNR | Best DV PSNR | dPSNR | dFVD |
|-----------|---------|------------|-------------|-------|------|
| Old figures (misleading) | panda_100_480p | 14.98 (buggy) | 22.59 | +7.61 | N/A |
| Old data (correct baseline) | panda_100_480p | 22.07 | 22.59 | +0.52 | N/A |
| Current data | panda_1000_480p | 18.61 | 18.60 | +0.01 | -80.0 |

Recommendation: Report FVD as the primary metric. Delta Vector's value is in improving the generation distribution, not individual frame fidelity.

---

## TinyLoRA Experiments - Status

Date submitted: April 17, 2026
Status: ALL 13 FAILED (exit code 2)
Root cause: Sbatch script passed --aug-rotate-zoom but the CLI flag is --no-aug-rotate-zoom.
Fix: Applied locally in run_tinylora.sbatch. Awaiting push + resubmit.

Planned runs:

| Run ID | SVD Rank | n_tie | Target | Blocks | Steps | LR | Aug | Notes |
|--------|----------|-------|--------|--------|-------|----|-----|-------|
| TL_BARE_R2 | 2 | 1 | qkv_proj | all | 20 | 1e-3 | off | Baseline TinyLoRA |
| TL_BARE_R1 | 1 | 1 | qkv_proj | all | 20 | 1e-3 | off | Rank 1 |
| TL_BARE_R4 | 4 | 1 | qkv_proj | all | 20 | 1e-3 | off | Rank 4 |
| TL_TIED_R2 | 2 | 48 | qkv_proj | all | 20 | 1e-3 | off | Full weight tying |
| TL_ALLATTN_R2 | 2 | 1 | all_attn | all | 20 | 1e-3 | off | All attention targets |
| TL_ALL_R2 | 2 | 1 | all | all | 20 | 1e-3 | off | All targets incl FFN |
| TL_AUG_R2 | 2 | 1 | qkv_proj | all | 20 | 1e-3 | on | With augmentation |
| TL_STEPS10 | 2 | 1 | qkv_proj | all | 10 | 1e-3 | off | Fewer steps |
| TL_STEPS10_AUG | 2 | 1 | qkv_proj | all | 10 | 1e-3 | on | Fewer steps + aug |
| TL_LR5E3_R2 | 2 | 1 | qkv_proj | all | 20 | 5e-3 | off | Higher LR |
| TL_LAST5 | 2 | 1 | qkv_proj | last_5 | 20 | 1e-3 | off | Only last 5 blocks |
| TL_LAST10 | 2 | 1 | qkv_proj | last_10 | 20 | 1e-3 | off | Only last 10 blocks |
| TL_LAST24 | 2 | 1 | qkv_proj | last_24 | 20 | 1e-3 | off | Only last 24 blocks |

---

## 1000-Video Evaluation - Status

Date submitted: Feb 16, 2026
Config files:
- sweep_experiment/configs/panda_1000v_best_methods.yaml (NOTTA + DV_BARE)
- sweep_experiment/configs/panda_1000v_lora.yaml (LORA_R4_S20 + LORA_R8_S10)
Dataset: panda_1000_480p, ALL 1000 videos
Submit script: sweep_experiment/sbatch/submit_1000v_best.sh

### LoRA Config Selection Rationale

Ranked all LoRA results on panda_1000_480p (100 videos) by FVD:

| Config | Rank | Blocks | Steps | LR | FVD | PSNR | Train(s) |
|--------|------|--------|-------|----|-----|------|----------|
| NOTTA (baseline) | -- | -- | 0 | -- | 641.1 | 18.612 | 0.0 |
| LR1e5_S20 | 4 | last_4 | 20 | 1e-5 | 641.5 | 18.569 | 37.3 |
| LORA_S10_LR5e-5 | 8 | all | 10 | 5e-5 | 644.6 | 18.616 | 18.5 |
| LORA_S5_LR1e-5 | 8 | all | 5 | 1e-5 | 651.0 | 18.591 | 9.2 |
| LORA_S5_LR5e-5 | 8 | all | 5 | 5e-5 | 651.3 | 18.609 | 9.3 |
| S4_GN10_LR2e5 | 1 | all | 20 | 2e-5 | 653.6 | 18.593 | 114.4 |
| LORA_S10_LR1e-5 | 8 | all | 10 | 1e-5 | 660.0 | 18.604 | 18.4 |
| LORA_noES_noCLIP | 4 | last_4 | 10 | 2e-5 | 668.4 | 18.575 | 55.9 |

No LoRA config meaningfully beats baseline FVD (641.1) at 100 videos. Selected two for 1000v:
- LORA_R4_S20: best FVD (rank=4, last_4, 20 steps, lr=1e-5) -- FVD=641.5
- LORA_R8_S10: best PSNR (rank=8, all, 10 steps, lr=5e-5) -- PSNR=18.616

### Planned 1000v Runs

| Run ID | Method | Key Config | 100v FVD | 100v PSNR | Status |
|--------|--------|-----------|----------|-----------|--------|
| NOTTA | No TTA | guidance=4.0 | 641.1 | 18.612 | PENDING |
| DV_BARE | Delta Vector | steps=10, lr=0.005 | 561.1 | 18.604 | PENDING |
| LORA_R4_S20 | LoRA | R4, last_4, 20s, lr=1e-5 | 641.5 | 18.569 | PENDING |
| LORA_R8_S10 | LoRA | R8, all, 10s, lr=5e-5 | 644.6 | 18.616 | PENDING |

---

## 1000-Video Evaluation - Status

Date submitted: Feb 16, 2026
Config files:
- sweep_experiment/configs/panda_1000v_best_methods.yaml (NOTTA + DV_BARE)
- sweep_experiment/configs/panda_1000v_lora.yaml (LORA_R4_S20 + LORA_R8_S10)
Dataset: panda_1000_480p, ALL 1000 videos
Submit script: sweep_experiment/sbatch/submit_1000v_best.sh

### LoRA Config Selection Rationale

Ranked all LoRA results on panda_1000_480p (100 videos) by FVD:

| Config | Rank | Blocks | Steps | LR | FVD | PSNR | Train(s) |
|--------|------|--------|-------|----|-----|------|----------|
| NOTTA (baseline) | -- | -- | 0 | -- | 641.1 | 18.612 | 0.0 |
| LR1e5_S20 | 4 | last_4 | 20 | 1e-5 | 641.5 | 18.569 | 37.3 |
| LORA_S10_LR5e-5 | 8 | all | 10 | 5e-5 | 644.6 | 18.616 | 18.5 |
| LORA_S5_LR1e-5 | 8 | all | 5 | 1e-5 | 651.0 | 18.591 | 9.2 |
| LORA_S5_LR5e-5 | 8 | all | 5 | 5e-5 | 651.3 | 18.609 | 9.3 |
| S4_GN10_LR2e5 | 1 | all | 20 | 2e-5 | 653.6 | 18.593 | 114.4 |
| LORA_S10_LR1e-5 | 8 | all | 10 | 1e-5 | 660.0 | 18.604 | 18.4 |
| LORA_noES_noCLIP | 4 | last_4 | 10 | 2e-5 | 668.4 | 18.575 | 55.9 |

No LoRA config meaningfully beats baseline FVD (641.1) at 100 videos. Selected two for 1000v:
- LORA_R4_S20: best FVD (rank=4, last_4, 20 steps, lr=1e-5) -- FVD=641.5
- LORA_R8_S10: best PSNR (rank=8, all, 10 steps, lr=5e-5) -- PSNR=18.616

### Planned 1000v Runs

| Run ID | Method | Key Config | 100v FVD | 100v PSNR | Status |
|--------|--------|-----------|----------|-----------|--------|
| NOTTA | No TTA | guidance=4.0 | 641.1 | 18.612 | PENDING |
| DV_BARE | Delta Vector | steps=10, lr=0.005 | 561.1 | 18.604 | PENDING |
| LORA_R4_S20 | LoRA | R4, last_4, 20s, lr=1e-5 | 641.5 | 18.569 | PENDING |
| LORA_R8_S10 | LoRA | R8, all, 10s, lr=5e-5 | 644.6 | 18.616 | PENDING |
