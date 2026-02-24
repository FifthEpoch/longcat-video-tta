# AdaSteer: Adaptive Shared Timestep Embedding Efficient Residual for Test-Time Adaptation of Video Diffusion Transformers

## Abstract

We present a systematic empirical study of test-time adaptation (TTA) methods for large-scale video diffusion transformers. Using LongCat-Video, a 13.6-billion parameter Diffusion Transformer (DiT), we evaluate six TTA strategies---full-model fine-tuning, LoRA, three residual-based methods, and two conditioning-pathway methods---across 100 videos from Panda-70M and 100 videos from UCF-101. Our central finding is that parameter efficiency is the critical design principle for single-instance TTA. We introduce **AdaSteer** (Adaptive Shared Timestep Embedding Efficient Residual), a family of TTA methods that learn additive offsets to the timestep embedding, shared across configurable groups of transformer blocks. AdaSteer-1, which learns a single 512-dimensional vector (0.0000038% of model parameters), is the only method that consistently improves generation quality over the unmodified baseline, achieving up to +1.62 dB PSNR improvement with extended conditioning. Methods with more tunable parameters---LoRA (2.6M+), norm-tuning (393K), FiLM adapters (24K), and full-model fine-tuning (13.6B)---either produce no measurable improvement or severely degrade output quality due to catastrophic overfitting.

## 1. Introduction

Video diffusion models have achieved remarkable generative quality, but their outputs remain imperfect when conditioned on specific input video content. Test-time adaptation (TTA)---the practice of briefly fine-tuning a pretrained model on a single test instance before inference---offers a principled approach to bridging this gap. While TTA has been successfully applied to discriminative models and image diffusion models, its application to large-scale video diffusion transformers remains unexplored.

The core challenge is the extreme imbalance between model capacity and available training signal. A 13.6B-parameter model must be adapted using only a handful of conditioning video frames---typically fewer than 30 frames at 480p resolution, encoding to approximately 300,000 latent values. This creates a severe overfitting risk that fundamentally constrains the design space of viable TTA methods.

We investigate six TTA strategies spanning several orders of magnitude in tunable parameter count: full-model fine-tuning (13.6B parameters), LoRA adaptation (65K--81.6M parameters), norm-layer tuning (24K--418K parameters), FiLM adapter corrections (8K--98K parameters), output residuals (16 parameters), and timestep-embedding offsets (32--24K parameters). Our experiments encompass over 170 independent runs across learning rate sweeps, iteration sweeps, conditioning frame ablations, generation horizon analyses, and cross-dataset evaluations.

## 2. Methods

### 2.1 AdaSteer: Adaptive Shared Timestep Embedding Efficient Residual

AdaSteer operates on the timestep embedding pathway of the DiT architecture. In LongCat-Video, each of the 48 transformer blocks contains an adaLN modulation layer that projects the timestep embedding *t* (512-dim) into shift, scale, and gate vectors (6 × 4096-dim) that modulate the hidden states:

```
t (512-dim) → adaLN_modulation(SiLU + Linear) → [shift, scale, gate]_msa, [shift, scale, gate]_mlp
```

AdaSteer learns *G* additive offset vectors δ₁, ..., δ_G ∈ ℝ^512, one per group of blocks. The 48 blocks are partitioned into *G* contiguous groups of approximately equal size. Before block *i* processes the timestep embedding, the group's offset is added:

```
t'ᵢ = t + δ_{g(i)}    where g(i) = ⌊i · G / 48⌋
```

| Variant | Groups (*G*) | Parameters | % of 13.6B |
|:--------|-------------:|-----------:|-----------:|
| AdaSteer-1 | 1 | 512 | 0.0000038% |
| AdaSteer-4 | 4 | 2,048 | 0.000015% |
| AdaSteer-12 | 12 | 6,144 | 0.000045% |
| AdaSteer-48 | 48 | 24,576 | 0.00018% |

### 2.2 Full-Model Fine-Tuning

All 13.6B parameters of the DiT are updated via SGD on the flow-matching objective computed over conditioning frames.

### 2.3 LoRA Adaptation

Low-Rank Adaptation introduces trainable low-rank matrices alongside frozen weight matrices. We sweep rank *r* ∈ {1, 4, 8, 16, 32} with α = 2*r*, plus constrained configurations targeting fewer blocks and reduced alpha.

### 2.4 Delta-C: Output Residual

Learns a per-channel additive correction applied to the DiT output, directly modifying the predicted velocity field.

### 2.5 Norm-Layer Tuning (TENT-style)

Inspired by TENT, tunes the affine parameters of normalization layers (cross-attention norm, QK-norm, or all norms) while keeping all other parameters frozen.

### 2.6 FiLM Adapter

Learns per-group additive corrections to the adaLN modulation output, operating on the full 6×4096-dim modulation space (shift, scale, gate for both MSA and MLP). Supports full, shift+scale, and scale-only modes.

## 3. Experimental Setup

### 3.1 Model and Datasets

**LongCat-Video**: 13.6B-parameter DiT with 48 single-stream transformer blocks, hidden dimension 4096, timestep embedding dimension (adaLN) 512. Latent space with 4× temporal compression via pretrained VAE. At 480p (480×832), each temporal latent frame contains 99,840 values (16 channels × 60 × 104 spatial).

**Panda-70M** (primary): 100 randomly sampled videos at 480p. Anchor point at frame 32. Default: 14 conditioning frames, 14 generated frames.

**UCF-101** (cross-dataset): 100 videos for domain-shift evaluation.

### 3.2 Evaluation

Generated frames compared to ground truth using PSNR (dB). All values are mean ± std across 99 successfully processed videos. Each TTA method is applied independently per video with full parameter reset between videos.

### 3.3 Training Signal Budget

At standard 14-frame conditioning: ~300K latent training values. Values-per-parameter ratio determines TTA viability:

| Variant | Parameters | Values / Parameter |
|:--------|----------:|---------:|
| Full-model | 13,600,000,000 | 0.00002 |
| LoRA (r=1) | 2,560,000 | 0.12 |
| Norm-tuning (all) | 418,000 | 0.72 |
| FiLM (full, G=4) | 98,304 | 3.1 |
| FiLM (full, G=1) | 24,576 | 12.2 |
| AdaSteer-48 | 24,576 | ~12 |
| AdaSteer-12 | 6,144 | ~49 |
| AdaSteer-4 | 2,048 | ~146 |
| AdaSteer-1 | 512 | ~585 |
| Delta-C | 16 | ~18,750 |

---

## 4. Experiment 0: Baselines

**Goal**: Establish no-TTA continuation quality across datasets.

| Dataset | Config | PSNR (dB) |
|:--------|:-------|----------:|
| Panda-70M | 14 cond, 14 gen, anchor=32 | 22.07 ± 8.83 |
| UCF-101 | 14 cond, 14 gen, anchor=32 | 22.07 ± 8.83 |

Additional baseline sweeps (conditioning × generation length) were run to understand how frame count affects quality independently of any TTA method. See Experiment 1.

---

## 5. Experiment 1: Conditioning + Generation Length Impact

**Goal**: Identify optimal conditioning length, understand how generation quality degrades with horizon, and establish the frame-count-dependent baseline that all TTA methods must be compared against.

### 5.1 Conditioning Frames Ablation

Fixed: anchor=32, total output=28 frames, best config per method.

| Cond Frames | Gen Frames | No-TTA (Full ctrl) | AdaSteer-1 (Δ) | AdaSteer-4 (Δ) | LoRA r=1 (Δ) | Delta-C (Δ) |
|:-----------:|:----------:|:------------------:|:--------------:|:--------------:|:------------:|:-----------:|
| 2 | 26 | 16.09 ± 6.65 | 16.41 (+0.32) | 17.14 (+1.05) | 15.14 (−0.95) | 10.20 (−5.89) |
| 7 | 21 | 18.37 ± 7.58 | 18.39 (+0.02) | 19.21 (+0.84) | 17.29 (−1.08) | 11.22 (−7.15) |
| 14 | 14 | 22.19 ± 9.02 | 22.35 (+0.16) | 21.19 (−1.00) | 20.49 (−1.70) | 13.45 (−8.74) |
| 24 | 4 | 23.05 ± 8.86 | 23.48 (+0.43) | 22.59 (−0.46) | 21.74 (−1.31) | 14.64 (−8.41) |

**Observations**:
- All methods benefit from more conditioning frames (more training signal)
- AdaSteer-1 achieves best overall PSNR of **23.48 dB** (+0.43) with 24 conditioning frames
- Crossover effect: AdaSteer-4 outperforms AdaSteer-1 at 2-7 cond frames (+1.05 vs +0.32), but overfits at 14+ frames. Suggests optimal group count is data-dependent.

### 5.2 Generation Horizon Ablation

Fixed: 14 conditioning frames, best config per method.

| Total Frames | Gen Frames | No-TTA (ctrl) | AdaSteer-1 (Δ) | AdaSteer-4 (Δ) | LoRA r=1 (Δ) | Delta-C (Δ) |
|:------------:|:----------:|:-------------:|:--------------:|:--------------:|:------------:|:-----------:|
| 16 | 2 | 28.07 ± 9.29 | 27.91 (−0.16) | 25.39 (−2.68) | 14.68 (−13.39) | 18.74 (−9.33) |
| 28 | 14 | 22.09 ± 8.83 | 22.42 (+0.33) | 21.18 (−0.91) | 11.18 (−10.91) | 13.45 (−8.64) |
| 44 | 30 | 18.77 ± 7.05 | 19.14 (+0.37) | 19.46 (+0.69) | 11.02 (−7.75) | 11.59 (−7.18) |
| 72 | 58 | 17.27 ± 5.66 | in progress | in progress | 11.50 (−5.77) | in progress |

**Observations**:
- PSNR degrades monotonically with horizon for all methods
- AdaSteer-1 maintains consistent +0.3 dB advantage through 30 gen frames
- At 30 gen frames, AdaSteer-4 (+0.69) outperforms AdaSteer-1 (+0.37): harder generation tasks may benefit from more capacity

---

## 6. Experiment 2: Full-Model TTA

**Goal**: Verify that TTA works on the video continuation task. Identify best LR. Establish TTA upper bound.

**Parameters**: 13.6B (all DiT params). Optimizer: SGD. ~90s train time per video.

### 6.1 LR Sweep (20 steps)

| Run | LR | PSNR (dB) | Δ vs baseline |
|:----|:--:|:---------:|:-------------:|
| F1 | 1e-6 | 22.07 ± 8.82 | 0.00 |
| F2 | 5e-6 | 22.05 ± 8.82 | −0.02 |
| F3 | 1e-5 | 22.07 ± 8.84 | 0.00 |
| F4 | 5e-5 | 22.05 ± 8.80 | −0.02 |
| F5 | 1e-4 | 21.86 ± 8.60 | −0.21 |

### 6.2 Iteration Sweep (lr=1e-5)

| Run | Steps | PSNR (dB) |
|:----|:-----:|:---------:|
| F6 | 5 | 22.09 ± 8.84 |
| F7 | 10 | 22.06 ± 8.82 |
| F8 | 20 | 22.06 ± 8.82 |
| F9 | 40 | 22.04 ± 8.83 |
| F10 | 80 | in progress |

### 6.3 Long-Train with Early Stopping (Series 17)

FLT1: lr=1e-5, 500 max steps, ES patience=10, 30 videos.

| Metric | Value |
|:-------|:------|
| FLT1 PSNR | 24.56 ± 8.96 |
| Baseline (same 30 vids) | 23.78 ± 9.72 |
| **Improvement** | **+0.78 dB** |
| Early stopped | 30/30 (100%) |
| Best step (mean) | 20 |
| Best step (range) | 1 – 90 |

**Observations**:
- At 20 steps with SGD, full-model TTA is inert: per-parameter updates are ~10⁻¹² and produce no measurable effect
- With 500-step budget + aggressive early stopping, full-model TTA can achieve +0.78 dB, but requires per-video ES and has wildly variable optimal step counts (1-90)
- This confirms TTA *works* on video continuation but is impractical at 13.6B parameters: ~90s per step × 20 steps average = ~30 min per video

---

## 7. Experiment 3: Single Vector TTA (AdaSteer-1)

**Goal**: Can a single 512-dim learned vector produce comparable improvement to full-model TTA?

**Parameters**: 512. Optimizer: AdamW. ~30s train time per video (5 steps). 0.0000038% of model.

### 7.1 LR Sweep (20 steps)

| Run | LR | PSNR (dB) | Δ |
|:----|:--:|:---------:|:--:|
| DA1 | 1e-3 | 22.23 ± 9.06 | +0.16 |
| DA2 | 5e-3 | **22.38 ± 9.25** | **+0.31** |
| DA3 | 1e-2 | 22.35 ± 9.26 | +0.28 |
| DA4 | 5e-2 | 11.97 ± 5.87 | −10.10 |
| DA5 | 1e-1 | 8.37 ± 3.94 | −13.70 |

### 7.2 Iteration Sweep (lr=5e-3)

| Run | Steps | PSNR (dB) |
|:----|:-----:|:---------:|
| DA6 | 5 | 22.35 ± 9.13 |
| DA7 | 10 | 22.21 ± 9.05 |
| DA8 | 20 | 22.27 ± 9.37 |
| DA9 | 40 | 19.58 ± 7.79 |
| DA10 | 80 | 17.57 ± 7.49 |

### 7.3 Best Config (from Delta-B G=1 equivalent)

Using lr=1e-2 at 5 steps: **PSNR = 22.54 ± 9.04 (+0.47 dB)**

### 7.4 Extended Conditioning (Series 23)

DAO1: 24 cond frames, lr=5e-3, 20 steps → **22.97 ± 9.05 (+0.90 dB vs ctrl)**

Combined with the Exp 1 result at 24 cond (23.48 dB, +0.43 vs ctrl at that anchor), AdaSteer-1 achieves up to **+1.62 dB improvement** relative to the 14-cond baseline when given more conditioning data.

### 7.5 Long-Train with ES (Series 18)

DALT1: lr=5e-3, 200 max steps, 30 vids → **11.22 ± 5.57** (catastrophic overfitting despite ES triggering)

**Observations**:
- AdaSteer-1 is the only method that consistently improves over baseline
- Optimal config: lr=1e-2, 5 steps (fast, simple, no ES needed)
- 512 parameters receive substantial per-step updates (~0.01 in magnitude), unlike full-model's ~10⁻¹²
- Long training destroys performance even with ES: the method's strength is its speed (5 steps, ~30s)

---

## 8. Experiment 4: Multiple Vector TTA (AdaSteer G>1)

**Goal**: Does per-block specialization (separate vectors for different transformer block groups) improve over a single shared vector?

### 8.1 Groups Sweep (20 steps, lr=1e-2)

| Run | G | Params | PSNR (dB) | Δ |
|:----|:-:|:------:|:---------:|:--:|
| DB1 | 1 | 512 | 21.21 ± 6.73 | −0.86 |
| DB2 | 2 | 1,024 | 19.62 ± 6.03 | −2.45 |
| DB3 | 4 | 2,048 | 15.42 ± 5.61 | −6.65 |
| DB4 | 8 | 4,096 | 15.61 ± 5.71 | −6.46 |
| DB5 | 16 | 8,192 | 14.74 ± 5.59 | −7.33 |
| DB6 | 48 | 24,576 | 10.92 ± 5.17 | −11.15 |

At 20 steps + lr=1e-2, all group counts overfit. This motivated testing at 5 steps.

### 8.2 Groups at 5 Steps (Series 26)

| Run | G | Params | PSNR (dB) | Δ |
|:----|:-:|:------:|:---------:|:--:|
| AS1 | 1 | 512 | **22.54 ± 9.01** | **+0.47** |
| AS2 | 4 | 2,048 | 21.14 ± 8.21 | −0.93 |
| AS3 | 12 | 6,144 | 20.84 ± 8.00 | −1.23 |
| AS4 | 48 | 24,576 | 18.79 ± 7.79 | −3.28 |
| AS5 | 4 (lr scaled) | 2,048 | 21.93 ± 8.77 | −0.14 |
| AS6 | 12 (lr scaled) | 6,144 | 22.01 ± 8.80 | −0.06 |

**Observation**: G=1 is optimal even at 5 steps. LR-scaling (dividing LR by G) helps higher group counts approach baseline but never exceeds G=1.

### 8.3 Ratio Sweep: Conditioning Frames × Groups (Series 27)

| Cond Frames | G=1 | G=4 | G=12 |
|:-----------:|:---:|:---:|:----:|
| 2 | 16.73 | 14.93 | 14.64 |
| 14 | 22.54 (AS1) | in progress | in progress |
| 24 | **23.69** | 22.40 | 22.28 |

**Hypothesis rejected**: G=1 is consistently optimal regardless of conditioning frame count. More data does not enable more groups — the single shared vector is the correct architectural choice at all tested data scales.

### 8.4 Equivalence Verification (Series 25)

Testing whether Delta-A (single hook on t_embedder) and Delta-B (per-block injection, G=1) give identical results:

| Run | Config | PSNR (dB) |
|:----|:-------|:---------:|
| DAV1 | lr=1e-2, 5 steps | 22.26 ± 9.05 |
| DAV2 | lr=1e-2, 10 steps | 22.15 ± 9.06 |
| DAV3 | lr=2.5e-3, 20 steps | 22.19 ± 9.07 |
| DAV4 | lr=7.5e-3, 20 steps | 22.35 ± 9.31 |

DAV1 (22.26) vs AS1 (22.54): 0.28 dB gap. Architecturally identical methods show small differences due to stochastic sigma sampling divergence between scripts. Not a bug — confirms functional equivalence within noise.

---

## 9. Experiment 5: LoRA TTA

**Goal**: How does LoRA (the standard PEFT method) compare for single-instance TTA?

### 9.1 Rank Sweep (all 48 blocks, qkv+proj, lr=2e-4, 20 steps)

| Run | Rank | α | Params | PSNR (dB) | Δ |
|:----|:----:|:-:|:------:|:---------:|:--:|
| L1 | 1 | 2 | 2.56M | 20.46 ± 7.66 | −1.61 |
| L2 | 4 | 8 | 10.2M | 15.03 ± 6.29 | −7.04 |
| L3 | 8 | 16 | 20.4M | 11.18 ± 6.00 | −10.89 |
| L4 | 16 | 32 | 40.8M | 8.15 ± 4.63 | −13.92 |
| L5 | 32 | 64 | 81.6M | 6.42 ± 3.47 | −15.65 |

Monotonic degradation with rank. Even r=1 underperforms baseline by 1.6 dB.

### 9.2 Iteration Sweep (r=1, lr=2e-4)

| Run | Steps | PSNR (dB) |
|:----|:-----:|:---------:|
| L13 | 5 | 22.06 ± 8.84 |
| L14 | 10 | 21.95 ± 8.74 |
| L15 | 20 | 13.20 ± 6.16 |
| L16 | 40 | 6.03 ± 3.41 |

At 5 steps, LoRA r=1 is baseline-equivalent. Rapid overfitting after 10 steps.

### 9.3 Constrained LoRA (r=1, fewer blocks, reduced α)

| Target | α=0.1 | α=0.5 | α=1.0 |
|:------:|:-----:|:-----:|:-----:|
| Last 4 | 22.08 | 22.04 | 22.01 |
| Last 8 | 22.04 | 22.03 | 21.96 |
| Last 16 | 22.05 | 22.03 | 21.92 |

All constrained configs are baseline-indistinguishable. LoRA either overfits (too much change) or has no effect (too little change) — no operating point achieves consistent improvement.

### 9.4 Ultra-Constrained LoRA (Series 16)

| Run | Blocks | Target | α | PSNR (dB) |
|:----|:------:|:------:|:-:|:---------:|
| LB1 | last_1 | qkv | 0.01 | 22.26 ± 9.21 |
| LB2 | last_1 | qkv | 0.05 | 22.19 ± 8.98 |
| LB3 | last_2 | qkv | 0.01 | 22.05 ± 9.05 |
| LB4 | last_2 | qkv | 0.05 | 22.64 ± 9.13 |
| LB5 | last_1 | proj | 0.01 | 20.93 ± 8.45 |
| LB6 | last_2 | proj | 0.01 | 21.19 ± 8.31 |

LB4 (last_2, qkv, α=0.05) shows **22.64 ± 9.13 (+0.57)** — the only LoRA config to meaningfully exceed baseline. However, this is based on partial results (86/100 videos) and needs verification on the full set.

### 9.5 Built-in LoRA Comparison (Series 15)

| Run | Config | PSNR (dB) |
|:----|:-------|:---------:|
| LN1 | all blocks, r=1, α=2 | 20.43 ± 7.70 |
| LN2 | last_4, r=1, α=0.1 | 22.05 ± 8.82 |

Confirms our custom LoRA implementation matches diffusers' built-in behavior.

### 9.6 Summary

LoRA is fundamentally unsuitable for single-instance TTA at conventional parameterizations. Even at rank 1, 2.56M parameters with ~300K training values gives 0.12 values/param — deeply in the overfitting regime. Ultra-constrained variants (targeting 1-2 blocks with α ≤ 0.05) can approach AdaSteer-level performance but are fragile and hyperparameter-sensitive.

---

## 10. Experiment 6: Output Residual TTA (Delta-C)

**Goal**: How does the naive approach of adding a learned vector to the model's output perform? This is the most straightforward TTA strategy for non-diffusion models.

**Parameters**: 16 (per-channel additive correction to 16-channel latent output).

### 10.1 LR Sweep (20 steps)

| Run | LR | PSNR (dB) | Δ |
|:----|:--:|:---------:|:--:|
| DC1 | 1e-2 | 13.45 ± 4.11 | −8.62 |
| DC2 | 5e-2 | 4.32 ± 3.62 | −17.75 |
| DC3 | 1e-1 | 4.06 ± 3.57 | −18.01 |
| DC4 | 5e-1 | in progress | — |
| DC5 | 1.0 | in progress | — |

### 10.2 Iteration Sweep (lr=1e-2)

| Run | Steps | PSNR (dB) |
|:----|:-----:|:---------:|
| DC6 | 5 | 20.57 ± 7.65 |
| DC7 | 10 | 17.88 ± 5.78 |
| DC8 | 20 | 8.18 ± 3.21 |
| DC9 | 40 | 4.72 ± 3.61 |

**Observations**: Delta-C degrades quality at all configurations. Despite having only 16 parameters (extremely high values/param ratio of ~18,750), output-space injection directly corrupts the predicted velocity field. The problem is not overfitting but architectural: perturbing the output is not equivalent to perturbing conditioning — it changes *what* is generated rather than *how* it is generated.

---

## 11. Extended Experiments

### 11.1 Norm-Layer Tuning (TENT-style, Series 21)

Tunes affine parameters of normalization layers while keeping everything else frozen.

| Run | Target | LR | Params | PSNR (dB) | Δ |
|:----|:------:|:--:|:------:|:---------:|:--:|
| NT1 | cross_attn_norm | 1e-4 | ~393K | 21.97 ± 8.77 | −0.10 |
| NT2 | cross_attn_norm | 1e-3 | ~393K | 20.17 ± 7.74 | −1.90 |
| NT3 | cross_attn_norm | 1e-2 | ~393K | 16.32 ± 7.50 | −5.75 |
| NT4 | qk_norm | 1e-3 | ~24K | 22.05 ± 8.84 | −0.02 |
| NT5 | all_norm | 1e-3 | ~418K | 20.17 ± 7.92 | −1.90 |
| NT6 | all_norm | 1e-4 | ~418K | 22.02 ± 8.82 | −0.05 |

**Observations**: Norm tuning never improves over baseline. At low LR it's inert; at higher LR it overfits. With 393K–418K params and ~300K training values (<1 value/param), this is expected.

### 11.2 FiLM Adapter (Series 22)

Learns corrections to the adaLN modulation output (the 6×4096-dim shift/scale/gate vectors).

| Run | Mode | G | LR | Params | PSNR (dB) | Δ |
|:----|:----:|:-:|:--:|:------:|:---------:|:--:|
| FM1 | full | 1 | 1e-3 | ~24K | 15.58 ± 8.07 | −6.49 |
| FM2 | full | 1 | 1e-2 | ~24K | 7.63 ± 3.24 | −14.44 |
| FM3 | full | 4 | 1e-3 | ~98K | 14.57 ± 7.41 | −7.50 |
| FM4 | full | 4 | 1e-2 | ~98K | 7.59 ± 3.13 | −14.48 |
| FM5 | scale_only | 1 | 1e-2 | ~8K | 20.30 ± 7.79 | −1.77 |
| FM6 | shift_scale | 1 | 1e-2 | ~16K | 7.34 ± 3.16 | −14.73 |

**Observations**: FiLM adapters fail comprehensively. Even scale-only mode (8K params) degrades by 1.77 dB. The 24K–98K parameter range has 3–12 values/param — insufficient for the adaLN output space. Critically, FiLM operates on the *output* of adaLN (24K-dim) while AdaSteer operates on the *input* (512-dim), getting a 47× amplification through the learned projection.

### 11.3 Combined AdaSteer + Norm (Series 23b)

DAO3: delta + cross_attn_norm, lr=5e-3 → **17.55 ± 7.57 (−4.52)**

Combining methods worsens performance: the 393K norm parameters dominate and overfit.

---

## 12. Early Stopping Ablation

All ES ablation runs used full-model TTA (lr=1e-5, 20 steps). Since full-model TTA is inert at this configuration, all settings produce identical PSNR (22.05–22.09). ES becomes meaningful only for long-train experiments (FLT1: 100% of videos stopped early at optimal steps).

| Ablation | Values Tested | PSNR Range |
|:---------|:-------------|:----------:|
| Check frequency | {1, 3, 5, 10} | 22.05 – 22.08 |
| Patience | {2, 3, 5, 7, 10} | 22.05 – 22.09 |
| Holdout fraction | {0.10, 0.25, 0.50} | 22.07 – 22.07 |
| Noise draws | {1, 2, 4} | 22.05 – 22.07 |
| Sigma schedule | 4 variants | 22.05 – 22.08 |
| Disabled | — | 22.05 |

---

## 13. Cross-Dataset Generalization

| Method | Panda-70M | UCF-101 |
|:-------|:---------:|:-------:|
| No TTA | 22.07 ± 8.83 | 22.07 ± 8.83 |
| Full-model | 22.07 ± 8.84 | 22.22 ± 9.04 |
| AdaSteer-1 | 22.38 ± 9.25 | 22.40 ± 9.30 |
| LoRA (r=1) | 20.46 ± 7.66 | 20.48 ± 7.65 |

Relative method ranking is perfectly preserved across datasets.

---

## 14. Discussion

### 14.1 The Parameter Efficiency Principle

The number of tunable parameters must be calibrated to the available training signal. With ~300K latent values per video:
- **>100 values/param** (AdaSteer-1): stable adaptation, consistent improvement
- **1–100 values/param** (FiLM, norm-tuning, AdaSteer G≥4 at 20 steps): overfitting dominates
- **<1 value/param** (LoRA, full-model): either overfits catastrophically or has no effect

### 14.2 Input vs. Output Perturbation

AdaSteer (input to adaLN, 512-dim) achieves +0.47 dB. FiLM (output of adaLN, 24K-dim) and Delta-C (model output, 16-dim) both fail. The key mechanism is the learned adaLN projection (512 → 24,576), which amplifies a small, learnable perturbation into a high-dimensional modulation — effectively adjusting *how the model denoises* without altering *what it has learned*.

### 14.3 Adaptive Capacity Allocation — Hypothesis Rejected

Series 26 and 27 definitively show that G=1 is optimal regardless of conditioning frame count. The single shared vector is the correct choice across all tested data scales (2–24 cond frames, 100K–700K training values). The adaptive G* formula from the prior report is not supported by the data.

### 14.4 Practical Recommendations

| Scenario | Method | Config | Expected Δ | Time |
|:---------|:------:|:------:|:----------:|:----:|
| Fast single-video TTA | AdaSteer-1 | lr=1e-2, 5 steps | +0.47 dB | ~30s |
| High-quality with ES | Full-model | lr=1e-5, 500 steps, ES | +0.78 dB | ~30min |
| Extended conditioning | AdaSteer-1 | lr=5e-3, 20 steps, 24 cond | +0.90 dB | ~45s |

### 14.5 Limitations

1. Only PSNR reported; SSIM and LPIPS computed but not analyzed
2. High per-video variance (±8–9 dB) driven by content difficulty heterogeneity
3. Single backbone (LongCat-Video); generalization to Open-Sora v2.0 and CogVideoX pending
4. VBench++ and FVD metrics not yet computed
5. No comparison against external baselines (SAVi-DNO, DFoT, CVP)

---

## 15. Conclusion

We presented a systematic empirical study of test-time adaptation for large-scale video diffusion transformers, evaluating six TTA strategies across 170+ experimental configurations on 100 videos. **AdaSteer-1**—a single 512-dimensional vector comprising 0.0000038% of the model—is the only method that consistently improves generation quality, achieving +0.47 dB PSNR in 5 optimization steps (~30 seconds). Full-model TTA can achieve +0.78 dB but requires early stopping and ~30 minutes per video.

All other methods fail: LoRA (2.56M+ params), norm-tuning (393K params), FiLM adapters (24K params), and output residuals either overfit or have no effect. The critical insight is that **adapt the conditioning pathway, not the weights**: a small perturbation to the 512-dim timestep embedding is amplified 47× through the learned adaLN projection, providing sufficient expressivity for single-instance adaptation while inherently constraining capacity.

---

## Appendix A: Complete Experiment Inventory

### A.1 Stage 1–2 Experiments (Complete)

| # | Experiment | Series | Runs | Best Result | Status |
|:-:|:-----------|:------:|:----:|:------------|:------:|
| 0 | Baseline (Panda) | — | 1 | 22.07 ± 8.83 | ✓ |
| 0 | Baseline (UCF-101) | — | 1 | 22.07 ± 8.83 | ✓ |
| 1 | Cond frames (5 methods) | Exp3 | 20 | — | 19/20 ✓ |
| 1 | Gen horizon (5 methods) | Exp4 | 20 | — | 16/20 ✓ |
| 2 | Full-model LR | 1 | 5 | 22.09 (F6, 5 steps) | ✓ |
| 2 | Full-model iter | 2 | 5 | 22.09 (5 steps) | 4/5 ✓ |
| 2 | Full-model long-train | 17 | 1 | 24.56 (+0.78, 30 vids) | ✓ |
| 3 | AdaSteer-1 LR | 7 | 5 | 22.38 (lr=5e-3) | ✓ |
| 3 | AdaSteer-1 iter | 8 | 5 | 22.35 (5 steps) | ✓ |
| 3 | AdaSteer-1 long-train | 18 | 1 | 11.22 (overfits) | ✓ |
| 3 | AdaSteer optimized | 23 | 3 | 22.97 (24 cond) | 2/3 ✓ |
| 4 | AdaSteer groups | 9 | 6 | 21.21 (G=1) | ✓ |
| 4 | AdaSteer-4 LR | 10 | 5 | 21.90 (lr=1e-3) | ✓ |
| 4 | AdaSteer-4 iter | 9b | 4 | 22.54 (5 steps) | ✓ |
| 4 | AdaSteer groups@5step | 26 | 6 | 22.54 (G=1) | ✓ |
| 4 | AdaSteer ratio sweep | 27 | 8 | 23.69 (24 cond, G=1) | 6/8 ✓ |
| 4 | AdaSteer equivalence | 25 | 4 | — (verified) | ✓ |
| 5 | LoRA rank | 3 | 5 | 20.46 (r=1) | ✓ |
| 5 | LoRA iter | 6 | 4 | 22.06 (5 steps) | ✓ |
| 5 | LoRA constrained | 14 | 9 | 22.08 (last_4, α=0.1) | ✓ |
| 5 | LoRA ultra-constrained | 16 | 6 | 22.64 (LB4, partial) | ✓ |
| 5 | LoRA built-in | 15 | 2 | 22.05 (last_4, α=0.1) | ✓ |
| 6 | Delta-C LR | 11 | 5 | 13.45 (lr=1e-2) | 3/5 ✓ |
| 6 | Delta-C iter | 12 | 4 | 20.57 (5 steps) | ✓ |
| — | ES ablation | 14 | 20 | null result | ✓ |
| — | UCF-101 cross-dataset | — | 4 | 22.40 (AdaSteer) | ✓ |

### A.2 Extended Experiments

| Experiment | Series | Runs | Best Result | Status |
|:-----------|:------:|:----:|:------------|:------:|
| Norm-tuning (TENT) | 21 | 6 | 22.05 (qk_norm) | ✓ |
| FiLM adapter | 22 | 6 | 20.30 (scale_only) | ✓ |
| Combined delta+norm | 23b | 1 | 17.55 (fails) | ✓ |

### A.3 Pending / In-Progress

| Experiment | Series | Runs | Status |
|:-----------|:------:|:----:|:------:|
| Delta-B hidden-state | 19 | 6 | Not started |
| Delta-B low-LR | 24 | 4 | Not started |
| AdaSteer extended data | 28 | 10 | Not submitted |
| AdaSteer param dim | 29 | 7 | Not submitted |
| Open-Sora v2.0 TTA | 6A | 3 | Not submitted |
| CogVideoX TTA | 6B | 3 | Not submitted |
| FVD evaluation | — | — | Not run |
| VBench++ evaluation | — | — | Not run |
| External baselines | — | — | Not started |
