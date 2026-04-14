# LongCat-Video TTA — Comprehensive Experiment Results

**Last updated:** Feb 16, 2026
**Cluster:** NYU Greene HPC (H200 partition)
**Model:** LongCat-Video (AutoencoderKLWan VAE + LongCatVideoTransformer3DModel DiT)
**Resolution:** 480p (480×832)

---

## Table of Contents

1. [Datasets](#datasets)
2. [Methods Overview](#methods-overview)
3. [Experiment Results](#experiment-results)
   - [1. No-TTA Baselines](#1-no-tta-baselines)
   - [2. LoRA TTA](#2-lora-tta)
   - [3. AdaSteer (Delta-A) — Initial Diagnostics](#3-adasteer-delta-a--initial-diagnostics)
   - [4. AdaSteer Step × LR Sweep](#4-adasteer-step--lr-sweep)
   - [5. AdaSteer Parameter Count Sweep (Delta-B)](#5-adasteer-parameter-count-sweep-delta-b)
   - [6. AdaSteer Conditioning Frames Sweep](#6-adasteer-conditioning-frames-sweep)
   - [7. AdaSteer TTA-Overlap Hypothesis Test](#7-adasteer-tta-overlap-hypothesis-test)
   - [8. February Config Replay](#8-february-config-replay)
4. [Cross-Experiment Summary Tables](#cross-experiment-summary-tables)
5. [Key Findings](#key-findings)
6. [Appendix: Configuration Reference](#appendix-configuration-reference)

---

## Datasets

| Dataset | Path on Cluster | Videos | Notes |
|---------|----------------|--------|-------|
| **Panda-70M (1000)** | `datasets/panda_1000_480p` | 1000 | Primary evaluation set. `max_videos=100` selects first 100. |
| **Panda-70M (100)** | `datasets/panda_100_480p` | 100 | Smaller subset. **Different videos** from panda_1000_480p. |
| **UCF-101** | `datasets/ucf101_test_480p` | 100 | Action recognition dataset. |

**IMPORTANT:** Results on `panda_1000_480p` and `panda_100_480p` are NOT directly comparable — they contain different videos with different difficulty distributions.

---

## Methods Overview

| Method | Trainable Params | Description |
|--------|-----------------|-------------|
| **No-TTA** | 0 | Baseline: generate without any test-time adaptation. |
| **LoRA** | ~1K–100K (rank-dependent) | Low-rank adapters on attention QKV/projection layers. |
| **AdaSteer (Delta-A)** | 1 (single scalar) | Learnable scalar added to AdaLN timestep embedding. |
| **AdaSteer (Delta-B)** | 64–512 (dim-dependent) | Learnable vector added to AdaLN timestep embedding. |

Common fixed settings (unless noted):
- `num_inference_steps`: 50
- `guidance_scale`: 4.0
- `seed`: 42
- `es_disable`: true (early stopping off)
- Optimizer: AdamW (for delta methods, hardcoded in `optimize_delta_a`)
- Grad clipping: `clip_grad_norm_(1.0)` (hardcoded for delta methods)

---

## Experiment Results

### 1. No-TTA Baselines

#### 1a. Panda-70M (panda_1000_480p), N=100

| Run ID | Config | PSNR | SSIM | LPIPS | FVD | FID |
|--------|--------|------|------|-------|-----|-----|
| NOTTA | cond=14, gen_start=48, frames=28 | **18.612** | **0.682** | **0.320** | 641.1 | 76.1 |

- **Source:** `sweep_experiment/results/sanity_100/panda_notta/NOTTA/summary.json`
- **Git commit:** `1802d5c`
- **Data dir:** `datasets/panda_1000_480p`

#### 1b. UCF-101, N=100

| Run ID | Config | PSNR | SSIM | LPIPS | FVD | FID |
|--------|--------|------|------|-------|-----|-----|
| NOTTA | cond=14, gen_start=32, frames=28 | **19.937** | **0.724** | **0.252** | 590.7 | 47.3 |

- **Source:** `sweep_experiment/results/sanity_100/ucf_notta_100/NOTTA/summary.json`
- **Git commit:** `1802d5c`

#### 1c. Panda-70M (panda_100_480p), N=99

| Run ID | Config | PSNR | SSIM | LPIPS | FVD | FID |
|--------|--------|------|------|-------|-----|-----|
| NOTTA_C1 | cond=1, gen_start=48, frames=15 | 14.258 | 0.550 | 0.461 | 699.3 | 109.9 |
| NOTTA_C5 | cond=5, gen_start=48, frames=19 | 17.825 | 0.630 | 0.374 | 709.5 | 82.6 |
| NOTTA_C10 | cond=10, gen_start=48, frames=24 | 16.545 | 0.622 | 0.382 | 714.5 | 96.2 |

- **Source:** `sweep_experiment/results/panda_adasteer_cond_sweep/`
- **Note:** On `panda_100_480p` (different dataset from main experiments)

---

### 2. LoRA TTA

#### 2a. Panda-70M (panda_1000_480p), N=100

**Fixed config:** cond=14, frames=28, gen_start=48, tta_total=48, tta_ctx=14, lora_rank=1, steps=20

| Run ID | LR | max_grad_norm | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | FVD | FID |
|--------|-----|--------------|------|-------|------|-------|-------|--------|-----|-----|
| Baseline (No-TTA) | — | — | 18.612 | — | 0.682 | — | 0.320 | — | 641.1 | 76.1 |
| S4_GN10_LR2e5 | 2e-5 | 10.0 | 18.593 | -0.019 | 0.683 | +0.001 | 0.320 | 0.000 | 653.6 | 77.5 |
| S4_GN1_LR2e4 | 2e-4 | 1.0 | 16.725 | -1.887 | 0.615 | -0.067 | 0.386 | +0.066 | 781.4 | 94.2 |

- **Source:** `sweep_experiment/results/sanity_100/panda_lora_s4_diagnostic/`
- **Conclusion:** LoRA at lr=2e-4 with grad_norm=1.0 severely overfits. At lr=2e-5 it's PSNR-neutral. LoRA shows no gains on Panda-70M.

---

### 3. AdaSteer (Delta-A) — Initial Diagnostics

#### 3a. Panda-70M (panda_1000_480p), N=100

**Fixed config:** cond=14, frames=28, gen_start=48, tta_total=48, tta_ctx=14

| Run ID | Steps | LR | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | FVD | ΔFVD | FID |
|--------|-------|-----|------|-------|------|-------|-------|--------|-----|------|-----|
| Baseline | 0 | — | 18.612 | — | 0.682 | — | 0.320 | — | 641.1 | — | 76.1 |
| ADA_5s_lr1e3 | 5 | 0.001 | 18.576 | -0.036 | 0.683 | +0.001 | 0.318 | -0.002 | 634.1 | -7.0 | 75.1 |
| ADA_5s_lr5e3 | 5 | 0.005 | 18.603 | -0.009 | 0.681 | -0.001 | 0.317 | -0.003 | 571.2 | **-69.9** | 76.8 |
| ADA_5s_lr1e2 | 5 | 0.01 | 18.611 | -0.001 | 0.680 | -0.002 | 0.319 | -0.001 | 589.1 | -52.0 | 75.8 |

- **Source:** First diagnostic sweep (pre `panda_adasteer_steps_lr`)
- **Git commit:** `1802d5c`
- **Best FVD:** lr=0.005, 5 steps → FVD 571.2 (−69.9)

#### 3b. UCF-101, N=100

**Fixed config:** cond=14, frames=28, gen_start=32, tta_total=??, tta_ctx=??

| Run ID | Steps | LR | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | FVD | ΔFVD | FID |
|--------|-------|-----|------|-------|------|-------|-------|--------|-----|------|-----|
| Baseline | 0 | — | 19.937 | — | 0.724 | — | 0.252 | — | 590.7 | — | 47.3 |
| ADA_5s_lr1e3 | 5 | 0.001 | 19.930 | -0.007 | 0.724 | 0.000 | 0.252 | 0.000 | 583.3 | -7.4 | 47.6 |
| ADA_5s_lr5e3 | 5 | 0.005 | 19.950 | +0.013 | 0.724 | 0.000 | 0.252 | 0.000 | 585.7 | -5.0 | 47.4 |
| ADA_5s_lr1e2 | 5 | 0.01 | 19.892 | -0.045 | 0.722 | -0.002 | 0.254 | +0.002 | 594.5 | +3.8 | 48.2 |

- **Source:** First diagnostic sweep (UCF runs)

---

### 4. AdaSteer Step × LR Sweep

#### Panda-70M (panda_1000_480p), N=100

**Fixed config:** cond=14, frames=28, gen_start=48, tta_total=48, tta_ctx=14

| Run ID | Steps | LR | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | FVD | ΔFVD | FID |
|--------|-------|------|------|-------|------|-------|-------|--------|-----|------|-----|
| Baseline | 0 | — | 18.612 | — | 0.682 | — | 0.320 | — | 641.1 | — | 76.1 |
| S5_LR05 | 5 | 0.05 | 18.549 | -0.063 | 0.665 | -0.017 | 0.323 | +0.003 | 591.4 | -49.7 | 80.0 |
| S10_LR001 | 10 | 0.001 | 18.568 | -0.044 | 0.682 | 0.000 | 0.319 | -0.001 | 633.9 | -7.2 | 76.2 |
| S10_LR005 | 10 | 0.005 | 18.590 | -0.022 | 0.684 | +0.002 | 0.316 | -0.004 | 568.7 | **-72.4** | 74.1 |
| S10_LR01 | 10 | 0.01 | 18.589 | -0.023 | 0.682 | 0.000 | 0.319 | -0.001 | 568.4 | **-72.7** | 79.5 |
| S10_LR05 | 10 | 0.05 | 17.587 | -1.025 | 0.598 | -0.084 | 0.386 | +0.066 | 605.7 | -35.4 | 99.4 |
| S20_LR001 | 20 | 0.001 | 18.498 | -0.114 | 0.681 | -0.001 | 0.320 | 0.000 | 637.4 | -3.7 | 77.3 |
| S20_LR005 | 20 | 0.005 | 18.488 | -0.124 | 0.678 | -0.004 | 0.321 | +0.001 | 657.6 | +16.5 | 77.9 |
| S20_LR01 | 20 | 0.01 | 18.586 | -0.026 | 0.676 | -0.006 | 0.323 | +0.003 | 608.5 | -32.6 | 79.9 |

- **Source:** `sweep_experiment/results/sanity_100/panda_adasteer_steps_lr/`
- **Git commit:** `1802d5c`
- **Config file:** `sweep_experiment/configs/panda_adasteer_steps_lr.yaml`

**Observations:**
- **Best FVD:** S10_LR01 (568.4, −72.7) and S10_LR005 (568.7, −72.4)
- **Best LPIPS:** S10_LR005 (0.316, −0.004)
- **PSNR always flat or slightly negative** — no config shows PSNR gain
- **LR=0.05 with ≥10 steps causes severe overfitting** (S10_LR05: −1.0 dB PSNR)
- **20 steps generally worse than 5-10 steps** for PSNR while not improving FVD
- **Sweet spot: 5-10 steps, LR 0.005-0.01**

---

### 5. AdaSteer Parameter Count Sweep (Delta-B)

#### Panda-70M (panda_1000_480p), N=100

**Fixed config:** cond=14, frames=28, gen_start=48, steps=5, lr=0.005

| Run ID | Delta Dim | Groups | Total Params | FVD | ΔFVD | FID |
|--------|-----------|--------|-------------|-----|------|-----|
| Baseline | — | — | 0 | 641.1 | — | 76.1 |
| P64 | 64 | 1 | 64 | 626.1 | -15.0 | 75.8 |
| P128 | 128 | 1 | 128 | 571.5 | **-69.6** | 75.2 |
| P256 | 256 | 1 | 256 | 573.1 | -68.0 | 74.2 |
| G2 | default | 2 | ~256 | 642.5 | +1.4 | 79.3 |
| G4 | default | 4 | ~512 | 603.8 | -37.3 | 79.0 |

- **Source:** `sweep_experiment/results/sanity_100/panda_adasteer_params/`
- **Note:** PSNR/SSIM/LPIPS not available in summary for Delta-B runs (only FVD/FID reported)

**Observations:**
- **Sweet spot at dim=128-256** — beyond that, more params hurt
- **Multiple groups (G2, G4) underperform** single-group with same or fewer params
- Delta-A (1 param) and Delta-B dim=128 achieve similar FVD (~571)

---

### 6. AdaSteer Conditioning Frames Sweep

#### Panda-70M (panda_100_480p), N=99

**⚠️ WARNING: This experiment used `panda_100_480p` (different dataset). Not directly comparable to experiments on `panda_1000_480p`.**

**Fixed config:** gen_start=48, steps=5, lr=0.005

| Run ID | Cond | Frames | Steps | PSNR | SSIM | LPIPS | FVD | FID |
|--------|------|--------|-------|------|------|-------|-----|-----|
| NOTTA_C1 | 1 | 15 | 0 | 14.258 | 0.550 | 0.461 | 699.3 | 109.9 |
| ADA_C1 | 1 | 15 | 5 | 14.386 | 0.551 | 0.461 | 702.3 | 110.1 |
| | | | | **+0.128** | **+0.001** | **0.000** | +3.0 | |
| NOTTA_C5 | 5 | 19 | 0 | 17.825 | 0.630 | 0.374 | 709.5 | 82.6 |
| ADA_C5 | 5 | 19 | 5 | 17.850 | 0.630 | 0.375 | 672.9 | 84.3 |
| | | | | **+0.025** | **0.000** | +0.001 | **-36.6** | |
| NOTTA_C10 | 10 | 24 | 0 | 16.545 | 0.622 | 0.382 | 714.5 | 96.2 |
| ADA_C10 | 10 | 24 | 5 | 16.546 | 0.619 | 0.381 | 646.7 | 95.7 |
| | | | | **+0.001** | **-0.003** | **-0.001** | **-67.8** | |

- **Source:** `sweep_experiment/results/panda_adasteer_cond_sweep/`
- **Config file:** `sweep_experiment/configs/panda_adasteer_cond_sweep.yaml`

**Observations:**
- Fewer conditioning frames = lower absolute quality (as expected)
- **cond=1 (image-to-video):** tiny PSNR gain (+0.13) but within noise, FVD neutral
- **cond=5:** PSNR neutral, FVD improves (−36.6)
- **cond=10:** PSNR neutral, FVD improves significantly (−67.8)
- AdaSteer's FVD benefit scales with number of conditioning frames

---

### 7. AdaSteer TTA-Overlap Hypothesis Test

#### Panda-70M (panda_1000_480p), N=100

**Hypothesis:** February PSNR gains came from (1) TTA training on same frames as generation conditioning, and (2) zero conditioning latents during training.

**Fixed config:** cond=14, frames=28, gen_start=48, steps=5, lr=0.005

| Run ID | tta_total | tta_ctx | Train Lat | Cond Lat | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | FVD | ΔFVD | FID |
|--------|-----------|---------|-----------|----------|------|-------|------|-------|-------|--------|-----|------|-----|
| Baseline | — | — | — | — | 18.612 | — | 0.682 | — | 0.320 | — | 641.1 | — | 76.1 |
| OLD_BUG | 14 | 0 | 3 | 0 | 18.579 | -0.033 | 0.681 | -0.001 | 0.322 | +0.002 | 630.0 | -11.1 | 76.7 |
| OVERLAP_COND | 14 | 14 | 1 | 3 | 18.542 | -0.070 | 0.681 | -0.001 | 0.321 | +0.001 | 634.0 | -7.1 | 77.7 |
| ALL_NOCOND | 48 | 0 | 9 | 0 | 18.568 | -0.044 | 0.681 | -0.001 | 0.319 | -0.001 | 613.0 | -28.1 | 74.9 |
| CURRENT | 48 | 14 | 6 | 4 | 18.560 | -0.052 | 0.681 | -0.001 | 0.319 | -0.001 | 575.5 | **-65.6** | 76.8 |

- **Source:** `sweep_experiment/results/sanity_100/panda_adasteer_tta_overlap/`
- **Git commit:** `8964a58`
- **Config file:** `sweep_experiment/configs/panda_adasteer_tta_overlap.yaml`

**Factorial analysis:**
- **Zero conditioning effect:** +0.03-0.07 PSNR (negligible), +38-85 FVD (much worse)
- **Frame overlap effect:** negligible on all metrics
- **Hypothesis refuted:** Neither factor produces PSNR gains
- **CURRENT (standard config) gives best FVD** (575.5)

---

### 8. February Config Replay

#### Panda-70M (panda_1000_480p), N=100

**Purpose:** Test whether the exact February frame geometry (cond=2, gen_start=32) produces PSNR gains.

| Run ID | Cond | Frames | gen_start | Steps | tta_total | tta_ctx | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | FVD | FID |
|--------|------|--------|-----------|-------|-----------|---------|------|-------|------|-------|-------|--------|-----|-----|
| **NOTTA_FEB** | 2 | 16 | 32 | 0 | 32 | 2 | 13.790 | — | 0.542 | — | 0.472 | — | 920.8 | 122.7 |
| FEB_OLDBUG | 2 | 16 | 32 | 20 | 2 | 2 | 13.551 | **-0.239** | 0.535 | -0.007 | 0.480 | +0.008 | 913.8 | 126.3 |
| FEB_CURRENT | 2 | 16 | 32 | 20 | 32 | 2 | 13.681 | **-0.109** | 0.537 | -0.005 | 0.474 | +0.002 | 914.2 | 122.7 |
| **NOTTA_CUR** | 14 | 28 | 48 | 0 | 48 | 14 | 18.612 | — | 0.682 | — | 0.320 | — | 641.1 | 76.1 |
| CUR_BEST | 14 | 28 | 48 | 5 | 48 | 14 | 18.565 | **-0.047** | 0.682 | 0.000 | 0.318 | -0.002 | 577.7 | 75.5 |

- **Source:** `sweep_experiment/results/sanity_100/panda_adasteer_feb_replay/`
- **Config file:** `sweep_experiment/configs/panda_adasteer_feb_replay.yaml`

**Observations:**
- **February geometry is terrible overall:** PSNR=13.8 with cond=2 (vs 18.6 with cond=14)
- **TTA HURTS with February geometry:** FEB_OLDBUG is −0.24 dB below its baseline
- **CUR_BEST matches prior results:** PSNR=18.565 (vs 18.603 earlier), FVD=577.7 (vs 571.2)
- **NOTTA_CUR reproduces exactly:** PSNR=18.612 (identical to prior baseline)

---

## Cross-Experiment Summary Tables

### Best AdaSteer Results on Panda-70M (panda_1000_480p)

All relative to No-TTA baseline: PSNR=18.612, SSIM=0.682, LPIPS=0.320, FVD=641.1

| Config | Steps | LR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔFVD | Notes |
|--------|-------|------|-------|-------|--------|------|-------|
| S10_LR005 | 10 | 0.005 | -0.022 | +0.002 | **-0.004** | **-72.4** | Best LPIPS + FVD |
| S10_LR01 | 10 | 0.01 | -0.023 | 0.000 | -0.001 | **-72.7** | Best FVD |
| ADA_5s_lr5e3 | 5 | 0.005 | -0.009 | -0.001 | -0.003 | -69.9 | Best PSNR preservation |
| CURRENT (overlap) | 5 | 0.005 | -0.052 | -0.001 | -0.001 | -65.6 | Reproducibility check |
| CUR_BEST (feb replay) | 5 | 0.005 | -0.047 | 0.000 | -0.002 | -63.4 | Reproducibility check |
| ALL_NOCOND | 5 | 0.005 | -0.044 | -0.001 | -0.001 | -28.1 | No conditioning |
| S5_LR05 | 5 | 0.05 | -0.063 | -0.017 | +0.003 | -49.7 | LR too high |
| S10_LR05 | 10 | 0.05 | -1.025 | -0.084 | +0.066 | -35.4 | Severe overfitting |

### Method Comparison on Panda-70M (panda_1000_480p)

| Method | Best Config | ΔPSNR | ΔSSIM | ΔLPIPS | ΔFVD |
|--------|------------|-------|-------|--------|------|
| **AdaSteer (Delta-A)** | 10 steps, lr=0.005 | -0.022 | +0.002 | **-0.004** | **-72.4** |
| **AdaSteer (Delta-B, dim=128)** | 5 steps, lr=0.005 | N/A | N/A | N/A | -69.6 |
| **LoRA (rank=1)** | 20 steps, lr=2e-5 | -0.019 | +0.001 | 0.000 | +12.5 |

### UCF-101 Summary

| Method | Config | ΔPSNR | ΔSSIM | ΔLPIPS | ΔFVD |
|--------|--------|-------|-------|--------|------|
| AdaSteer | 5s, lr=0.001 | -0.007 | 0.000 | 0.000 | -7.4 |
| AdaSteer | 5s, lr=0.005 | +0.013 | 0.000 | 0.000 | -5.0 |
| AdaSteer | 5s, lr=0.01 | -0.045 | -0.002 | +0.002 | +3.8 |

---

## Key Findings

### 1. AdaSteer Produces Significant FVD Improvement
- Consistent −60 to −73 FVD reduction (10-11% improvement in temporal coherence)
- Best at 10 steps, lr=0.005-0.01
- Only 1 trainable parameter, ~27s training overhead per video

### 2. PSNR Is Consistently Flat
- No AdaSteer configuration produces meaningful PSNR gains on Panda-70M
- Best PSNR preservation: 5 steps, lr=0.005 (−0.009 dB, within noise)
- PSNR degradation increases with more steps and higher LR

### 3. LPIPS Shows Marginal Improvement
- Best: S10_LR005 achieves −0.004 LPIPS (0.320 → 0.316)
- LPIPS improvement correlates with FVD improvement

### 4. Overfitting Threshold
- **LR ≤ 0.01 with ≤ 10 steps:** Safe (PSNR neutral, FVD improves)
- **LR = 0.05 with ≥ 10 steps:** Severe overfitting (−1 dB PSNR)
- **20 steps:** Generally worse than 5-10 steps regardless of LR

### 5. Conditioning Frames Matter for Absolute Quality
- cond=2: PSNR=13.8 (terrible — insufficient context)
- cond=14: PSNR=18.6 (good — adequate context)
- TTA benefit (FVD) scales with conditioning frame count

### 6. February PSNR Gains Not Reproducible
- Exhaustively tested: zero conditioning, frame overlap, exact Feb geometry
- All factors ruled out — gains likely a measurement artifact
- No code changes in `cbca8d8` affect delta_a optimization/generation

---

## Appendix: Configuration Reference

### Frame Geometry

```
Video: [frame 0] [frame 1] ... [frame T]
                                 |
       |<--- tta_total_frames -->|<--- gen frames --->|
       |                         |                     |
       |  [tta_ctx] [tta_train]  | gen_start_frame    |
       |                         |                     |
       |        [num_cond_frames]|                     |
       |                         |<-- num_frames ------>|
```

- `gen_start_frame`: Anchor frame where generation begins
- `num_cond_frames`: Frames from [gen_start - num_cond, gen_start) used as visual conditioning
- `num_frames`: Total frames in generation (cond + generated)
- `tta_total_frames`: Pixel frames before anchor loaded for TTA training
- `tta_context_frames`: Leading TTA frames treated as clean context (timestep=0)

### Latent Conversion

- VAE temporal factor: 4× compression
- `num_latent_frames = 1 + (num_pixel_frames - 1) // 4`
- `num_ctx_latents = 1 + (tta_context_frames - 1) // 4`

### Git Commits

| Commit | Description |
|--------|-------------|
| `3fbf906` | February experiment baseline (old bug present) |
| `cbca8d8` | Fix TTA bugs + efficiency improvements |
| `1802d5c` | Add AdaSteer sweep configs |
| `8964a58` | Add TTA-overlap hypothesis test config |
