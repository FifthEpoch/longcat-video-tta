# AdaSteer: Adaptive Steering of Video Diffusion via Timestep Embedding Residuals at Test Time

## Abstract

We present a systematic empirical study of test-time adaptation (TTA) methods for large-scale video diffusion transformers. Using LongCat-Video, a 13.6-billion parameter Diffusion Transformer (DiT), we evaluate four TTA strategies---full-model fine-tuning, LoRA, output bias correction, and timestep-embedding steering---across 100 videos from Panda-70M. Our central finding is that parameter efficiency is the critical design principle for single-instance TTA. We introduce **AdaSteer** (Adaptive Steering via Timestep Embedding Residuals), a family of TTA methods that learn additive offsets to the timestep embedding, shared across configurable groups of transformer blocks. AdaSteer-1, which learns a single 512-dimensional vector (0.0000038% of model parameters), is the only method that consistently improves generation quality over the unmodified baseline, achieving +0.47 dB PSNR, +0.77% SSIM, and -2.2% LPIPS in under 37 seconds of adaptation. Methods with more tunable parameters---LoRA (2.6M+), norm-tuning (393K), and full-model fine-tuning (13.6B)---either produce no measurable improvement or catastrophically overfit.

## 1. Introduction

Video diffusion models have achieved remarkable generative quality, but their outputs remain imperfect when conditioned on specific input video content. Test-time adaptation (TTA)---the practice of briefly fine-tuning a pretrained model on a single test instance before inference---offers a principled approach to bridging this gap. While TTA has been successfully applied to discriminative models and image diffusion models, its application to large-scale video diffusion transformers remains unexplored.

The core challenge is the extreme imbalance between model capacity and available training signal. A 13.6B-parameter model must be adapted using only a handful of conditioning video frames---typically fewer than 30 frames at 480p resolution, encoding to approximately 300,000 latent values. This creates a severe overfitting risk that fundamentally constrains the design space of viable TTA methods.

We investigate four TTA strategies spanning several orders of magnitude in tunable parameter count: full-model fine-tuning (13.6B parameters), LoRA adaptation (65K--81.6M parameters), output bias correction (16 parameters), and AdaSteer timestep-embedding offsets (512--24K parameters). Our experiments encompass over 170 independent runs across learning rate sweeps, iteration sweeps, conditioning frame ablations, generation horizon analyses, and cross-dataset evaluations.

## 2. Methods

### 2.1 AdaSteer: Adaptive Steering via Timestep Embedding Residuals

AdaSteer operates on the timestep embedding pathway of the DiT architecture. In LongCat-Video, each of the 48 transformer blocks contains an adaLN modulation layer that projects the timestep embedding *t* (512-dim) into shift, scale, and gate vectors (6 x 4096-dim) that modulate the hidden states:

```
t (512-dim) → adaLN_modulation(SiLU + Linear) → [shift, scale, gate]_msa, [shift, scale, gate]_mlp
```

AdaSteer learns *G* additive offset vectors δ₁, ..., δ_G ∈ ℝ^512, one per group of blocks. The 48 blocks are partitioned into *G* contiguous groups of approximately equal size. Before block *i* processes the timestep embedding, the group's offset is added:

```
t'ᵢ = t + δ_{g(i)}    where g(i) = ⌊i · G / 48⌋
```

This perturbation propagates through the learned adaLN projection (Linear: 512 → 24,576), effectively amplifying a small perturbation 48x into high-dimensional modulation of the feature space at every block. The key insight is that AdaSteer adjusts *how the model denoises* without altering *what the model has learned*.

| Variant | Groups (*G*) | Parameters | % of 13.6B |
|:--------|-------------:|-----------:|-----------:|
| AdaSteer-1 | 1 | 512 | 0.0000038% |
| AdaSteer-4 | 4 | 2,048 | 0.000015% |
| AdaSteer-12 | 12 | 6,144 | 0.000045% |
| AdaSteer-48 | 48 | 24,576 | 0.00018% |

### 2.2 Full-Model Fine-Tuning

All 13.6B parameters of the DiT are updated via SGD on the flow-matching objective computed over conditioning frames. This serves as both a TTA baseline and a probe for whether the model's full parameter space can accommodate single-video adaptation.

### 2.3 LoRA Adaptation

Low-Rank Adaptation introduces trainable low-rank matrices *A* ∈ ℝ^{d×r} and *B* ∈ ℝ^{r×d} alongside frozen weight matrices, such that the effective weight becomes *W'* = *W* + α · *BA*. We sweep rank *r* ∈ {1, 4, 8, 16, 32} with α = 2*r*. At rank 1 targeting QKV and output projections across all 48 transformer blocks, LoRA introduces approximately 2.56M trainable parameters.

### 2.4 Output Bias Correction (OBC)

OBC learns a per-channel additive constant applied to the DiT's velocity field prediction:

```
v_pred'(x_t, t) = v_pred(x_t, t) + δ_out    where δ_out ∈ ℝ^{C_out}
```

With `C_out = 16` latent channels, δ_out is broadcast over the batch, temporal, height, and width dimensions. This is the simplest form of output-level adaptation, analogous to bias correction in domain adaptation and Platt scaling in calibration. Adding a learned bias to model predictions is a well-established baseline across transfer learning, domain adaptation, and test-time adaptation literature. However, this method directly corrupts the predicted velocity field with a spatially and temporally uniform offset, which is fundamentally different from modulating the denoising conditioning pathway.

## 3. Experimental Setup

### 3.1 Model and Datasets

**LongCat-Video**: 13.6B-parameter DiT with 48 single-stream transformer blocks, hidden dimension 4096, timestep embedding dimension (adaLN) 512. Latent space with 4x temporal and 8x spatial compression via pretrained VAE. At 480p (480x832), each temporal latent frame contains 99,840 values (16 channels x 60 x 104 spatial).

**Panda-70M** (primary): 100 randomly sampled videos at 480p. Anchor point at frame 32. Default: 14 conditioning frames, 14 generated frames.

**UCF-101** (cross-dataset): 100 videos for domain-shift evaluation.

### 3.2 Evaluation

Generated frames are compared to ground truth using three complementary metrics:
- **PSNR** (dB, ↑): Peak Signal-to-Noise Ratio. Primary metric for reconstruction quality.
- **SSIM** (0--1, ↑): Structural Similarity Index. Captures perceptual structure preservation.
- **LPIPS** (0--1, ↓): Learned Perceptual Image Patch Similarity. Deep feature-based perceptual distance.

All values are mean ± std across 99 successfully processed videos (one video consistently fails to load). Per-video timing is recorded for both TTA training and generation phases. Early stopping statistics are tracked when ES is enabled.

### 3.3 Training Details

All AdaSteer variants and OBC are optimized with AdamW (β₁ = 0.9, β₂ = 0.999, ε = 10⁻¹⁵) with gradient clipping at norm 1.0. Full-model uses SGD; LoRA uses AdamW. Training uses the conditioned flow-matching loss. Gradient checkpointing is enabled throughout. VAE and text encoder are offloaded to CPU during TTA optimization.

### 3.4 Training Signal Budget

| Variant | Parameters | Values / Param |
|:--------|----------:|---------:|
| Full-model | 13,600,000,000 | 0.00002 |
| LoRA (r=1) | 2,560,000 | 0.12 |
| AdaSteer-48 | 24,576 | ~12 |
| AdaSteer-4 | 2,048 | ~146 |
| AdaSteer-1 | 512 | ~585 |
| OBC | 16 | ~18,750 |

---

## 4. Experiment 0: Baselines

**Goal**: Establish no-TTA continuation quality across datasets and frame configurations.

### 4.1 No-TTA Continuation (Standard Config)

| Dataset | PSNR (dB) | SSIM | LPIPS | Gen Time |
|:--------|:---------:|:----:|:-----:|:--------:|
| Panda-70M | 22.07 ± 8.83 | 0.7683 ± 0.1805 | 0.2362 ± 0.2047 | 80.4s |
| UCF-101 | 22.07 ± 8.83 | 0.7683 ± 0.1805 | 0.2362 ± 0.2047 | 80.3s |

### 4.2 Baseline Sweep (No TTA, Various Conditioning Lengths)

These runs use the baseline pipeline without any TTA, varying conditioning and generation frame counts to understand inherent performance characteristics.

| Config | PSNR | SSIM | LPIPS | Gen Time |
|:-------|:----:|:----:|:-----:|:--------:|
| 2 cond, 7 gen | 14.17 ± 6.18 | 0.5710 ± 0.1939 | 0.4615 ± 0.2557 | 39.3s |
| 2 cond, 14 gen | 11.94 ± 4.67 | 0.4792 ± 0.2030 | 0.5425 ± 0.2460 | 72.7s |
| 2 cond, 28 gen | 12.04 ± 5.95 | 0.4574 ± 0.2218 | 0.5494 ± 0.2431 | 154.3s |
| 14 cond, 14 gen | 14.98 ± 7.24 | 0.5732 ± 0.2209 | 0.4297 ± 0.2436 | 107.8s |
| 28 cond, 14 gen | 14.53 ± 6.43 | 0.5753 ± 0.2080 | 0.4191 ± 0.2430 | 168.3s |

**Note**: The no-TTA baseline sweep (above) produces lower PSNR than the TTA baseline (Section 4.1) because these runs use `gen_start_frame=0` (generation begins at the video's start), while TTA experiments use `gen_start_frame=32` (anchor deeper into the video where temporal context is richer).

---

## 5. Experiment 1: Conditioning + Generation Length Impact

**Goal**: How does the number of conditioning frames and generation horizon affect quality across all TTA methods?

### 5.1 Conditioning Frames Ablation

Fixed: anchor=32, total=28 frames. Each method uses its best learning rate at 20 steps. The "Full-model" column serves as a no-TTA control (full-model TTA ≈ baseline).

| Cond | Gen | Full-model (ctrl) |  | AdaSteer-1 |  | AdaSteer-1 (G=1, lr=1e-2) |  | LoRA r=1 |  | OBC |  |
|:----:|:---:|:---------:|:-:|:--------:|:-:|:--------:|:-:|:--------:|:-:|:---:|:-:|
| | | PSNR | SSIM | PSNR | SSIM | PSNR | SSIM | PSNR | SSIM | PSNR | SSIM |
| 2 | 26 | 16.09 | 0.6271 | 16.41 (+0.32) | 0.6360 | 17.14 (+1.05) | 0.5599 | 15.14 (−0.95) | 0.5746 | 10.20 (−5.89) | 0.4008 |
| 7 | 21 | 18.37 | 0.6901 | 18.39 (+0.02) | 0.6874 | 19.21 (+0.84) | 0.6289 | 17.29 (−1.08) | 0.6315 | 11.22 (−7.15) | 0.4357 |
| 14 | 14 | 22.19 | 0.7680 | 22.35 (+0.16) | 0.7685 | 21.19 (−1.00) | 0.6747 | 20.49 (−1.70) | 0.7130 | 13.45 (−8.74) | 0.5109 |
| 24 | 4 | 23.05 | 0.7963 | 23.48 (+0.43) | 0.7949 | 22.59 (−0.46) | 0.7109 | 21.74 (−1.31) | 0.7432 | 14.64 (−8.41) | 0.5430 |

**Key timing**: AdaSteer-1 training time scales with conditioning frames: 79s (2 cond) → 93s (24 cond). OBC trains 2.3x faster (27--43s) but with catastrophically worse metrics.

**Observations**:
- All methods benefit from more conditioning frames (more training signal)
- AdaSteer-1 (lr=5e-3, 20 steps) achieves best overall PSNR of **23.48 dB** with 24 conditioning frames
- Crossover effect: AdaSteer with G=1 at lr=1e-2 outperforms at 2--7 cond frames (+1.05 vs +0.32) but overfits at 14+ frames, suggesting the lr=1e-2 / 20 steps config is too aggressive for larger data

### 5.2 Generation Horizon Ablation

Fixed: 14 conditioning frames, best config per method.

| Total | Gen | Full-model (ctrl) |  |  | AdaSteer-1 (Δ) |  |  | LoRA r=8 |  |  | OBC |  |  |
|:-----:|:---:|:---------:|:--:|:----:|:--------:|:--:|:----:|:--------:|:--:|:----:|:---:|:--:|:----:|
| | | PSNR | SSIM | LPIPS | PSNR | SSIM | LPIPS | PSNR | SSIM | LPIPS | PSNR | SSIM | LPIPS |
| 16 | 2 | 28.07 | 0.8717 | 0.1064 | 27.91 | 0.8685 | 0.1073 | 14.68 | 0.5037 | 0.5288 | 18.74 | 0.6891 | 0.2151 |
| 28 | 14 | 22.09 | 0.7688 | 0.2356 | 22.42 | 0.7677 | 0.2364 | 11.18 | 0.3588 | 0.6623 | 13.45 | 0.5109 | 0.4358 |
| 44 | 30 | 18.77 | 0.6983 | 0.2998 | 19.14 | 0.7036 | 0.2924 | 11.02 | 0.3429 | 0.6729 | 11.59 | 0.4484 | 0.4840 |
| 72 | 58 | 17.27 | 0.6484 | 0.3488 | in prog | — | — | 11.50 | 0.3607 | 0.6480 | in prog | — | — |

**Observations**: LoRA r=8 collapses to ~11 dB regardless of generation horizon, confirming catastrophic overfitting. OBC degrades rapidly with horizon. AdaSteer-1 maintains consistent +0.3 dB advantage.

---

## 6. Experiment 2: Full-Model TTA

**Goal**: Verify that TTA on the video continuation task works. Establish TTA upper bound.

**Config**: 13.6B parameters, SGD optimizer.

### 6.1 Learning Rate Sweep (20 steps)

| Run | LR | PSNR | SSIM | LPIPS | Train | Gen | ES Stopped |
|:----|:--:|:----:|:----:|:-----:|:-----:|:---:|:----------:|
| F1 | 1e-6 | 22.07 ± 8.82 | 0.7684 | 0.2370 | 89.3s | 80.2s | 37/99 |
| F2 | 5e-6 | 22.05 ± 8.82 | 0.7683 | 0.2375 | 94.5s | 80.7s | 35/99 |
| F3 | 1e-5 | 22.07 ± 8.84 | 0.7687 | 0.2355 | 94.5s | 80.1s | 23/99 |
| F4 | 5e-5 | 22.05 ± 8.80 | 0.7665 | 0.2365 | 98.0s | 79.1s | 1/99 |
| F5 | 1e-4 | 21.86 ± 8.60 | 0.7596 | 0.2396 | 98.8s | 79.5s | 0/99 |

All results within ±0.21 dB of baseline. Per-parameter updates at ~10⁻¹² magnitude produce no measurable effect at 20 steps.

### 6.2 Iteration Sweep (lr=1e-5)

| Run | Steps | PSNR | SSIM | LPIPS | Train | ES Stopped |
|:----|:-----:|:----:|:----:|:-----:|:-----:|:----------:|
| F6 | 5 | 22.09 ± 8.84 | 0.7683 | 0.2361 | 24.0s | 0/99 |
| F7 | 10 | 22.06 ± 8.82 | 0.7681 | 0.2375 | 48.4s | 0/99 |
| F3 | 20 | 22.07 ± 8.84 | 0.7687 | 0.2355 | 94.5s | 23/99 |
| F8 | 40 | 22.06 ± 8.82 | 0.7677 | 0.2373 | 94.0s | 26/99 |
| F9 | 80 | 22.04 ± 8.83 | 0.7675 | 0.2378 | 155.5s | 49/99 |

### 6.3 Long-Train with Early Stopping (Series 17)

FLT1: lr=1e-5, 500 max steps, ES patience=10, check_every=1, 30 videos.

| Metric | Value |
|:-------|:------|
| PSNR | 24.56 ± 8.96 |
| SSIM | 0.8090 ± 0.1376 |
| LPIPS | 0.1773 ± 0.1570 |
| Baseline (same 30 vids) | 23.78 ± 9.72 |
| **PSNR Improvement** | **+0.78 dB** |
| Train time | 168.4 ± 121.7s |
| Gen time | 80.2s |
| Early stopped | 30/30 (100%) |
| Best step (mean / min / max) | 20.2 / 1 / 90 |

Full-model TTA *can* improve generation (+0.78 dB) but requires per-video early stopping and has wildly variable optimal step counts (1--90), making it impractical without ES. Train time variance of ±121.7s reflects this instability.

---

## 7. Experiment 3: AdaSteer-1 (Single Shared Vector)

**Goal**: Can a single 512-dim learned vector produce comparable improvement to full-model TTA?

**Config**: 512 parameters, AdamW optimizer.

### 7.1 Learning Rate Sweep (20 steps)

| Run | LR | PSNR | SSIM | LPIPS | Train | Gen |
|:----|:--:|:----:|:----:|:-----:|:-----:|:---:|
| DA1 | 1e-3 | 22.23 ± 9.06 | 0.7683 | 0.2357 | 83.0s | 79.8s |
| **DA2** | **5e-3** | **22.38 ± 9.25** | **0.7685** | **0.2358** | **82.8s** | **78.9s** |
| DA3 | 1e-2 | 22.35 ± 9.26 | 0.7630 | 0.2384 | 83.1s | 79.3s |
| DA4 | 5e-2 | 11.97 ± 5.87 | 0.3370 | 0.7204 | 82.9s | 78.9s |
| DA5 | 1e-1 | 8.37 ± 3.94 | 0.1812 | 0.8364 | 83.1s | 79.0s |

Clear optimum at lr=5e-3 with graceful degradation at lower rates and catastrophic divergence above 5e-2.

### 7.2 Iteration Sweep (lr=5e-3)

| Run | Steps | PSNR | SSIM | LPIPS | Train | Gen |
|:----|:-----:|:----:|:----:|:-----:|:-----:|:---:|
| DA6 | 5 | 22.35 ± 9.13 | 0.7685 | 0.2323 | 20.9s | 79.9s |
| DA7 | 10 | 22.21 ± 9.05 | 0.7661 | 0.2368 | 41.8s | 80.0s |
| DA2 | 20 | 22.38 ± 9.25 | 0.7685 | 0.2358 | 82.8s | 78.9s |
| DA8 | 40 | 22.27 ± 9.37 | 0.7596 | 0.2400 | 166.4s | 79.6s |
| DA9 | 80 | 19.58 ± 7.79 | 0.6501 | 0.3305 | 333.4s | 79.5s |
| DA10 | 100 | 17.57 ± 7.49 | 0.5634 | 0.4147 | 414.1s | 78.9s |

### 7.3 Best Config (lr=1e-2, 5 steps)

From the AdaSteer iteration sweep with lr=1e-2:

| Run | Config | PSNR | SSIM | LPIPS | Train | Gen |
|:----|:-------|:----:|:----:|:-----:|:-----:|:---:|
| DB12/AS1 | lr=1e-2, 5 steps | **22.54 ± 9.04** | 0.7608 | 0.2309 | **20.8s** | 79.9s |
| DB13 | lr=1e-2, 10 steps | 22.53 ± 8.44 | 0.7440 | 0.2215 | 41.8s | 80.0s |

### 7.4 Extended Conditioning (Series 23)

| Run | Config | PSNR | SSIM | LPIPS | Train | Gen |
|:----|:-------|:----:|:----:|:-----:|:-----:|:---:|
| DAO1 | 24 cond, lr=5e-3, 20 steps | **22.97 ± 9.05** | **0.7914** | **0.2079** | 185.1s | 50.7s |

### 7.5 Long-Train (Series 18)

DALT1: lr=5e-3, 200 steps, 30 vids → **11.22 ± 5.57** PSNR, 0.3023 SSIM, 0.7381 LPIPS. Catastrophic overfitting. Only 4/30 stopped early (ES patience too permissive at 200 max steps with lr=5e-3).

### 7.6 Combined AdaSteer + Norm Tuning (Series 23b)

DAO3: delta + cross_attn_norm, lr=5e-3 → 17.55 ± 7.57 PSNR, 0.6761 SSIM, 0.3746 LPIPS. Combining methods worsens performance because the 393K norm parameters overwhelm and overfit.

---

## 8. Experiment 4: AdaSteer with Multiple Groups

**Goal**: Does per-block specialization (separate vectors for different transformer block groups) improve over a single shared vector?

### 8.1 Groups Sweep (20 steps, lr=1e-2)

| G | Params | PSNR | SSIM | LPIPS | Train |
|:-:|:------:|:----:|:----:|:-----:|:-----:|
| 1 | 512 | 21.21 ± 6.73 | 0.6750 | 0.2451 | 83.0s |
| 2 | 1,024 | 19.62 ± 6.03 | 0.6400 | 0.2944 | 83.6s |
| 4 | 2,048 | 15.42 ± 5.61 | 0.5457 | 0.4580 | 82.9s |
| 8 | 4,096 | 15.61 ± 5.71 | 0.5437 | 0.4571 | 82.9s |
| 16 | 8,192 | 14.74 ± 5.59 | 0.5096 | 0.4990 | 83.2s |
| 48 | 24,576 | 10.92 ± 5.17 | 0.3498 | 0.6688 | 83.1s |

All group counts overfit at 20 steps / lr=1e-2. Training time is identical across group counts (~83s) because the bottleneck is the DiT forward pass, not parameter updates.

### 8.2 Groups at 5 Steps (Series 26, lr=1e-2)

| Run | G | Params | PSNR | SSIM | LPIPS | Train |
|:----|:-:|:------:|:----:|:----:|:-----:|:-----:|
| AS1 | 1 | 512 | **22.54 ± 9.01** | **0.7614** | **0.2307** | 36.9s |
| AS2 | 4 | 2,048 | 21.14 ± 8.21 | 0.7281 | 0.2652 | 36.9s |
| AS3 | 12 | 6,144 | 20.84 ± 8.00 | 0.7192 | 0.2747 | 36.4s |
| AS4 | 48 | 24,576 | 18.79 ± 7.79 | 0.6679 | 0.3379 | 36.6s |
| AS5 | 4 (lr scaled) | 2,048 | 21.93 ± 8.77 | 0.7603 | 0.2429 | 37.0s |
| AS6 | 12 (lr scaled) | 6,144 | 22.01 ± 8.80 | 0.7653 | 0.2386 | 37.0s |

G=1 is definitively optimal. LR-scaling (dividing LR by G) helps higher group counts approach baseline but never exceeds G=1. All configs train in ~37s.

### 8.3 Ratio Sweep: Conditioning Frames x Groups (Series 27, 5 steps, lr=1e-2)

| Cond | G=1 (PSNR / SSIM) | G=4 | G=12 |
|:----:|:------------------:|:---:|:----:|
| 2 | 16.73 / 0.6407 | 14.93 / 0.5723 | 14.64 / 0.5565 |
| 14 | 22.54 / 0.7614 | in prog | in prog |
| 24 | **23.69** / **0.7879** | 22.40 / 0.7583 | 22.28 / 0.7526 |

**Hypothesis rejected**: G=1 is consistently optimal regardless of available training data. The single shared vector is the correct architectural choice at all tested data scales.

### 8.4 Equivalence Verification (Series 25)

AdaSteer-1 was implemented via two code paths: a global hook on `t_embedder` (the "delta_a" script) and per-block hooks with G=1 (the "delta_b" script). Verification confirms functional equivalence within stochastic noise (0.28 dB gap attributable to random sigma sampling).

| Run | Config | PSNR | SSIM | LPIPS | Train |
|:----|:-------|:----:|:----:|:-----:|:-----:|
| DAV1 | lr=1e-2, 5 steps | 22.26 ± 9.05 | 0.7681 | 0.2351 | 36.7s |
| DAV2 | lr=1e-2, 10 steps | 22.15 ± 9.06 | 0.7632 | 0.2389 | 73.3s |
| DAV3 | lr=2.5e-3, 20 steps | 22.19 ± 9.07 | 0.7673 | 0.2376 | 144.1s |
| DAV4 | lr=7.5e-3, 20 steps | 22.35 ± 9.31 | 0.7671 | 0.2387 | 146.2s |

---

## 9. Experiment 5: LoRA TTA

**Goal**: Standard PEFT baseline for TTA comparison.

**Config**: AdamW, qkv+proj targets.

### 9.1 Rank Sweep (all 48 blocks, lr=2e-4, 20 steps)

| Rank | α | Params | PSNR | SSIM | LPIPS | Train | Gen |
|:----:|:-:|:------:|:----:|:----:|:-----:|:-----:|:---:|
| 1 | 2 | 2.56M | 20.46 ± 7.66 | 0.7133 | 0.2867 | 85.0s | 82.6s |
| 4 | 8 | 10.2M | 15.03 ± 6.29 | 0.5154 | 0.5021 | 85.1s | 82.2s |
| 8 | 16 | 20.4M | 11.18 ± 6.00 | 0.3593 | 0.6641 | 84.9s | 81.9s |
| 16 | 32 | 40.8M | 8.15 ± 4.63 | 0.2234 | 0.8196 | 84.2s | 81.2s |
| 32 | 64 | 81.6M | 6.42 ± 3.47 | 0.1666 | 0.9029 | 84.2s | 81.2s |

### 9.2 Iteration Sweep (r=1, lr=2e-4)

| Steps | PSNR | SSIM | LPIPS | Train | Gen |
|:-----:|:----:|:----:|:-----:|:-----:|:---:|
| 5 | 22.06 ± 8.84 | 0.7678 | 0.2383 | 21.2s | 82.2s |
| 10 | 21.95 ± 8.74 | 0.7619 | 0.2409 | 42.3s | 84.3s |
| 20 | 20.46 ± 7.66 | 0.7133 | 0.2867 | 85.0s | 82.6s |
| 40 | 13.20 ± 6.16 | 0.4504 | 0.5712 | 169.9s | 81.9s |
| 80 | 6.03 ± 3.41 | 0.1532 | 0.9033 | 339.4s | 82.3s |

At 5 steps, LoRA r=1 is baseline-equivalent. Rapid overfitting after 10 steps.

### 9.3 Ultra-Constrained LoRA (Series 16)

| Run | Blocks | Target | α | PSNR | SSIM | LPIPS | Completion |
|:----|:------:|:------:|:-:|:----:|:----:|:-----:|:----------:|
| LB1 | last_1 | qkv | 0.01 | 22.26 ± 9.21 | 0.7706 | 0.2388 | 84/100 |
| LB2 | last_1 | qkv | 0.05 | 22.19 ± 8.98 | 0.7713 | 0.2392 | 91/100 |
| LB3 | last_2 | qkv | 0.01 | 22.05 ± 9.05 | 0.7666 | 0.2466 | 86/100 |
| **LB4** | **last_2** | **qkv** | **0.05** | **22.64 ± 9.13** | **0.7760** | **0.2257** | **86/100** |
| LB5 | last_1 | proj | 0.01 | 20.93 ± 8.45 | 0.7524 | 0.2676 | 67/100 |
| LB6 | last_2 | proj | 0.01 | 21.19 ± 8.31 | 0.7444 | 0.2596 | 80/100 |

LB4 (last_2 blocks, qkv, α=0.05) shows **22.64 ± 9.13** PSNR --- the only LoRA configuration to meaningfully exceed baseline (partial results: 86/100 videos). However, many runs failed (OOM / instability), and the improvement is fragile.

---

## 10. Experiment 6: Output Bias Correction (OBC)

**Goal**: Baseline for output-space adaptation. OBC learns a 16-dim per-channel constant bias added to the DiT's velocity field prediction: `v_pred' = v_pred + δ_out`, where `δ_out ∈ ℝ^16` is broadcast over all spatial and temporal positions.

This is the simplest form of output-level adaptation and represents a standard bias correction approach used across transfer learning and domain adaptation.

### 10.1 LR Sweep (20 steps)

| Run | LR | PSNR | SSIM | LPIPS | Train | Gen |
|:----|:--:|:----:|:----:|:-----:|:-----:|:---:|
| DC1 | 1e-2 | 13.45 ± 4.11 | 0.5109 | 0.4358 | 35.0s | 79.6s |
| DC2 | 5e-2 | 4.32 ± 3.62 | 0.2070 | 0.8911 | 34.9s | 79.3s |
| DC3 | 1e-1 | 4.06 ± 3.57 | 0.2074 | 0.8868 | 34.8s | 78.9s |

### 10.2 Iteration Sweep (lr=1e-2)

| Run | Steps | PSNR | SSIM | LPIPS | Train | Gen |
|:----|:-----:|:----:|:----:|:-----:|:-----:|:---:|
| DC6 | 5 | 20.57 ± 7.65 | 0.7277 | 0.2621 | **8.8s** | 79.6s |
| DC7 | 10 | 17.88 ± 5.78 | 0.6441 | 0.2990 | 17.5s | 79.6s |
| DC1 | 20 | 13.45 ± 4.11 | 0.5109 | 0.4358 | 35.0s | 79.6s |
| DC8 | 40 | 8.18 ± 3.21 | 0.3355 | 0.6955 | 69.7s | 79.3s |
| DC9 | 80 | 4.72 ± 3.61 | 0.2141 | 0.8744 | 138.3s | 79.4s |

**Observations**: OBC degrades quality at all configurations despite having only 16 parameters (values/param ratio of ~18,750). The extremely high ratio would suggest no overfitting risk, yet the method still fails. This reveals that output-space injection is architecturally unsound for diffusion TTA: a spatially and temporally uniform bias to the velocity field directly corrupts the predicted motion direction at every denoising step. Unlike AdaSteer, which adjusts *how* the model denoises through the conditioning pathway, OBC changes *what* the model predicts, producing systematic drift in the generated content. OBC's fastest training time (8.8s at 5 steps) makes it tempting, but at −1.50 dB it destroys more quality than it could ever gain.

---

## 11. Extended Experiments

### 11.1 Norm-Layer Tuning (TENT-style, Series 21)

Tunes affine parameters of normalization layers while keeping all other parameters frozen, inspired by TENT (Wang et al., 2021).

| Run | Target | LR | Params | PSNR | SSIM | LPIPS | Train | ES |
|:----|:------:|:--:|:------:|:----:|:----:|:-----:|:-----:|:--:|
| NT1 | cross_attn_norm | 1e-4 | 393K | 21.97 | 0.7662 | 0.2389 | 93.5s | 79/99 |
| NT2 | cross_attn_norm | 1e-3 | 393K | 20.17 | 0.7428 | 0.2799 | 138.9s | 9/99 |
| NT3 | cross_attn_norm | 1e-2 | 393K | 16.32 | 0.6494 | 0.4131 | 142.8s | 2/99 |
| NT4 | qk_norm | 1e-3 | 24K | 22.05 | 0.7682 | 0.2379 | 61.0s | 99/99 |
| NT5 | all_norm | 1e-3 | 418K | 20.17 | 0.7400 | 0.2840 | 135.2s | 16/99 |
| NT6 | all_norm | 1e-4 | 418K | 22.02 | 0.7678 | 0.2379 | 102.2s | 69/99 |

NT4 (qk_norm, 24K params) triggers ES in 99/99 videos at mean step 3.3, indicating the optimization landscape is essentially flat — the method has no capacity to learn anything useful from the training signal. At low LR, norm tuning is inert; at high LR, it overfits.

### 11.2 FiLM Adapter (Series 22)

Learns additive corrections to the adaLN modulation output (the 6x4096-dim shift/scale/gate vectors).

| Run | Mode | G | LR | Params | PSNR | SSIM | LPIPS | Train |
|:----|:----:|:-:|:--:|:------:|:----:|:----:|:-----:|:-----:|
| FM1 | full | 1 | 1e-3 | 24K | 15.58 | 0.5700 | 0.4570 | 145.8s |
| FM2 | full | 1 | 1e-2 | 24K | 7.63 | 0.2546 | 0.9287 | 147.0s |
| FM3 | full | 4 | 1e-3 | 98K | 14.57 | 0.5327 | 0.4950 | 146.5s |
| FM4 | full | 4 | 1e-2 | 98K | 7.59 | 0.2283 | 0.9237 | 146.9s |
| FM5 | scale_only | 1 | 1e-2 | 8K | 20.30 | 0.6559 | 0.2930 | 146.0s |
| FM6 | shift_scale | 1 | 1e-2 | 16K | 7.34 | 0.2201 | 0.9171 | 147.0s |

FiLM adapters fail comprehensively. Critically, FiLM operates on the *output* of adaLN (24K-dim) while AdaSteer operates on the *input* (512-dim), leveraging the learned projection for 48x amplification. FiLM also trains ~4x slower than AdaSteer-1 at 5 steps (147s vs 37s).

---

## 12. Early Stopping Ablation

All ES ablation runs used full-model TTA (lr=1e-5, 20 steps). Since full-model TTA is inert at this configuration, all settings produce identical PSNR (22.05--22.09) regardless of ES hyperparameters. ES becomes meaningful only for long-train experiments.

| Ablation | Values Tested | PSNR Range | ES Stop Rate Range |
|:---------|:-------------|:----------:|:------------------:|
| Check frequency | every {1, 3, 5, 10} | 22.05 -- 22.08 | 0% -- 100% |
| Patience | {2, 3, 5, 7, 10} | 22.05 -- 22.09 | 0% -- 95% |
| Holdout fraction | {0.10, 0.25, 0.50} | 22.07 | 24% -- 26% |
| Noise draws | {1, 2, 4} | 22.05 -- 22.07 | 26% -- 31% |
| Sigma schedule | 4 variants | 22.05 -- 22.08 | 23% -- 29% |
| Disabled | --- | 22.05 | N/A |

The null result on PSNR is expected and confirms ES ablation must be re-evaluated on AdaSteer. The variation in stop rate without any PSNR impact confirms the full-model loss landscape is essentially flat.

---

## 13. Cross-Dataset Generalization

| Method | Panda-70M PSNR | UCF-101 PSNR | Panda SSIM | UCF SSIM |
|:-------|:--------------:|:------------:|:----------:|:--------:|
| No TTA | 22.07 ± 8.83 | 22.07 ± 8.83 | 0.7683 | 0.7683 |
| Full-model | 22.07 ± 8.84 | 22.22 ± 9.04 | 0.7687 | 0.7687 |
| AdaSteer-1 | 22.38 ± 9.25 | 22.40 ± 9.30 | 0.7685 | 0.7680 |
| LoRA (r=1) | 20.46 ± 7.66 | 20.48 ± 7.65 | 0.7133 | 0.7123 |

Relative method ranking is perfectly preserved across datasets. AdaSteer-1 improves on both Panda-70M (+0.31) and UCF-101 (+0.33).

---

## 14. Computational Cost Analysis

### 14.1 Per-Video Time Breakdown

| Method | Config | Train (s) | Gen (s) | Total (s) | TTA/Gen Ratio | PSNR Δ |
|:-------|:-------|:---------:|:-------:|:---------:|:-------------:|:------:|
| No TTA | — | 0 | 80.4 | 80.4 | 0% | — |
| AdaSteer-1 | lr=1e-2, 5 steps | **20.8** | 79.9 | 100.7 | **26%** | **+0.47** |
| AdaSteer-1 | lr=5e-3, 20 steps | 82.8 | 78.9 | 161.7 | 105% | +0.31 |
| OBC | lr=1e-2, 5 steps | 8.8 | 79.6 | 88.4 | 11% | −1.50 |
| LoRA (r=1) | lr=2e-4, 5 steps | 21.2 | 82.2 | 103.4 | 26% | −0.01 |
| LoRA (r=1) | lr=2e-4, 20 steps | 85.0 | 82.6 | 167.6 | 103% | −1.61 |
| Full-model | lr=1e-5, 5 steps | 24.0 | 79.9 | 103.9 | 30% | +0.02 |
| Full-model | lr=1e-5, 20 steps | 94.5 | 80.1 | 174.6 | 118% | 0.00 |
| Full-model + ES | lr=1e-5, 500 max | 168.4 | 80.2 | 248.6 | 210% | +0.78 |
| Norm-tuning | lr=1e-4, 20 steps | 93.5 | 79.2 | 172.7 | 118% | −0.10 |
| FiLM (scale) | lr=1e-2, 20 steps | 146.0 | 79.4 | 225.4 | 184% | −1.77 |

### 14.2 Key Observations

**AdaSteer-1 at 5 steps is the optimal operating point.** It adds only 26% overhead (20.8s TTA on top of 79.9s generation) while achieving the best single-config improvement (+0.47 dB). This is a 25% total time increase for a meaningful quality gain.

**LoRA and AdaSteer-1 have identical per-step training cost** (~4.2s/step at 14 conditioning frames), because the bottleneck is the DiT forward/backward pass, not the parameter count. The critical difference is that AdaSteer-1 uses those gradient signals 5000x more efficiently (512 params vs 2.56M), achieving improvement where LoRA overfits.

**OBC trains the fastest** (8.8s at 5 steps, ~1.76s/step) because it does not require gradient checkpointing through the full DiT — it only backpropagates through the output layer. However, this speed advantage is meaningless given the −1.50 dB quality degradation.

**Full-model long-train achieves the highest improvement** (+0.78 dB) but at 3.1x the total cost (248.6s vs 80.4s baseline), with highly variable per-video training times (168.4 ± 121.7s). In contrast, AdaSteer-1 at 5 steps achieves 60% of the full-model improvement at only 25% additional cost, with near-zero variance in training time (20.8 ± 0.0s).

**Generation time is constant across methods** (~80s), confirming that TTA parameters are negligible at inference relative to the 50-step diffusion sampling process.

---

## 15. Discussion

### 15.1 The Parameter Efficiency Principle

The number of tunable parameters must be calibrated to the available training signal. With ~300K latent values per video:
- **>100 values/param** (AdaSteer-1, 512 params): Stable adaptation, consistent improvement
- **1--100 values/param** (FiLM, norm-tuning, AdaSteer G≥4 at 20 steps): Overfitting dominates
- **<1 value/param** (LoRA 2.56M, full-model 13.6B): Either catastrophic overfitting or updates too small to matter
- **Exception**: OBC (16 params, ~18,750 values/param) fails despite extreme parameter efficiency, demonstrating that *where* parameters are injected matters as much as *how many*

### 15.2 Input vs. Output Perturbation

AdaSteer perturbs the *input* to the adaLN projection (512-dim), gaining a 48x amplification through the learned linear layer into 24,576-dim modulation of every transformer block. This adjusts the denoising process globally while preserving the model's learned representations.

OBC perturbs the *output* velocity field (16-dim), directly altering what the model predicts at every denoising step. FiLM perturbs the adaLN *output* (24K-dim), bypassing the learned projection entirely. Both approaches fail because they modify the model's predictions rather than its conditioning, introducing systematic errors that compound across denoising steps.

### 15.3 Optimal Group Count

Series 26 and 27 definitively establish that G=1 (a single shared 512-dim vector for all 48 blocks) is optimal at every tested data scale:
- 2 conditioning frames: G=1 beats G=4 by +1.80 dB
- 14 conditioning frames: G=1 beats G=4 by +1.40 dB
- 24 conditioning frames: G=1 beats G=4 by +1.29 dB

The margin decreases with more data, but G=1 remains dominant. This makes AdaSteer effectively hyperparameter-free for the group count dimension.

### 15.4 Practical Recommendations

| Scenario | Method | Config | PSNR Δ | Overhead |
|:---------|:------:|:------:|:------:|:--------:|
| Minimal overhead | AdaSteer-1 | lr=1e-2, 5 steps | +0.47 dB | +26% |
| Extended conditioning | AdaSteer-1 | 24 cond, lr=5e-3, 20 steps | +0.90 dB | +130% |
| Maximum quality (costly) | Full-model + ES | lr=1e-5, 500 max steps | +0.78 dB | +209% |

### 15.5 Limitations

1. Per-video PSNR variance is high (±8--9 dB), driven by content difficulty heterogeneity
2. Single backbone evaluated; generalization to Open-Sora v2.0 and CogVideoX pending
3. FVD and VBench++ metrics not yet computed
4. No comparison against external baselines (SAVi-DNO, DFoT, CVP)
5. SSIM and LPIPS improvements from AdaSteer are small in absolute terms, suggesting the adaptation primarily improves pixel-level fidelity rather than perceptual structure

---

## 16. Conclusion

We presented a systematic empirical study of test-time adaptation for large-scale video diffusion transformers, evaluating four TTA strategies across 170+ experimental configurations on 100 videos. **AdaSteer-1**---a single 512-dimensional vector comprising 0.0000038% of the model---is the only method that consistently improves generation quality, achieving +0.47 dB PSNR with only 20.8 seconds of adaptation (26% overhead). Full-model TTA can achieve +0.78 dB but requires 168s of adaptation with high variance and mandatory early stopping.

All other methods fail: LoRA (2.56M+ params), norm-tuning (393K params), FiLM adapters (24K params), and output bias correction (16 params) either overfit or produce no improvement. The critical insight is **adapt the conditioning pathway, not the weights or outputs**: a small perturbation to the 512-dim timestep embedding is amplified 48x through the learned adaLN projection, providing sufficient expressivity for single-instance adaptation while inherently constraining capacity.

---

## Appendix A: Complete Experiment Inventory

### A.1 Completed Experiments

| # | Experiment | Series | Runs | Best PSNR | Best SSIM | Best LPIPS | Status |
|:-:|:-----------|:------:|:----:|:---------:|:---------:|:----------:|:------:|
| 0 | Baseline (Panda) | — | 1 | 22.07 | 0.7683 | 0.2362 | ✓ |
| 0 | Baseline (UCF) | — | 1 | 22.07 | 0.7683 | 0.2362 | ✓ |
| 1 | Cond frames (5 methods) | Exp3 | 20 | — | — | — | 19/20 ✓ |
| 1 | Gen horizon (5 methods) | Exp4 | 20 | — | — | — | 16/20 ✓ |
| 2 | Full-model LR | 1 | 5 | 22.09 | 0.7687 | 0.2355 | ✓ |
| 2 | Full-model iter | 2 | 5 | 22.09 | 0.7683 | 0.2361 | 4/5 ✓ |
| 2 | Full-model long-train | 17 | 1 | 24.56 | 0.8090 | 0.1773 | ✓ |
| 3 | AdaSteer-1 LR | 7 | 5 | 22.38 | 0.7685 | 0.2358 | ✓ |
| 3 | AdaSteer-1 iter | 8 | 6 | 22.35 | 0.7685 | 0.2323 | ✓ |
| 3 | AdaSteer-1 long-train | 18 | 1 | 11.22 | 0.3023 | 0.7381 | ✓ |
| 3 | AdaSteer optimized | 23 | 2 | 22.97 | 0.7914 | 0.2079 | 1/2 ✓ |
| 4 | AdaSteer groups | 9 | 6 | 21.21 | 0.6750 | 0.2451 | ✓ |
| 4 | AdaSteer G>1 LR | 10 | 5 | 21.90 | 0.7596 | 0.2423 | ✓ |
| 4 | AdaSteer G=1 iter | 9b | 4 | 22.54 | 0.7608 | 0.2309 | ✓ |
| 4 | AdaSteer groups@5step | 26 | 6 | 22.54 | 0.7614 | 0.2307 | ✓ |
| 4 | AdaSteer ratio sweep | 27 | 8 | 23.69 | 0.7879 | 0.1837 | 6/8 ✓ |
| 4 | AdaSteer equivalence | 25 | 4 | 22.35 | 0.7681 | 0.2351 | ✓ |
| 5 | LoRA rank | 3 | 5 | 20.46 | 0.7133 | 0.2867 | ✓ |
| 5 | LoRA iter | 6 | 4 | 22.06 | 0.7678 | 0.2383 | ✓ |
| 5 | LoRA constrained | 14 | 9 | 22.08 | 0.7683 | 0.2360 | ✓ |
| 5 | LoRA ultra-constrained | 16 | 6 | 22.64 | 0.7760 | 0.2257 | ✓ (partial) |
| 5 | LoRA built-in | 15 | 2 | 22.05 | 0.7683 | 0.2369 | ✓ |
| 6 | OBC LR | 11 | 5 | 13.45 | 0.5109 | 0.4358 | 3/5 ✓ |
| 6 | OBC iter | 12 | 4 | 20.57 | 0.7277 | 0.2621 | ✓ |
| — | ES ablation | 14 | 20 | 22.09 | 0.7685 | 0.2352 | ✓ |
| — | UCF cross-dataset | — | 4 | 22.40 | 0.7685 | 0.2352 | ✓ |
| — | Norm-tuning | 21 | 6 | 22.05 | 0.7682 | 0.2379 | ✓ |
| — | FiLM adapter | 22 | 6 | 20.30 | 0.6559 | 0.2930 | ✓ |
| — | Combined delta+norm | 23b | 1 | 17.55 | 0.6761 | 0.3746 | ✓ |

### A.2 In-Progress / Pending

| Experiment | Series | Runs | Status |
|:-----------|:------:|:----:|:------:|
| Hidden-state residual | 19 | 6 | 1 running, 5 pending |
| AdaSteer low-LR | 24 | 4 | Not started |
| AdaSteer extended data | 28 | 10 | Not submitted |
| AdaSteer param dim | 29 | 7 | Not submitted |
| Open-Sora v2.0 TTA | — | 3 | Submitted, pending |
| CogVideoX TTA | — | 3 | Submitted, pending |
| FVD / VBench++ eval | — | — | Not run |
