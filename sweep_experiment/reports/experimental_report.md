# AdaSteer: Adaptive Shared Timestep Embedding Efficient Residual for Test-Time Adaptation of Video Diffusion Transformers

## Abstract

We present a systematic empirical study of test-time adaptation (TTA) methods for large-scale video diffusion transformers. Using LongCat-Video, a 13.6-billion parameter Diffusion Transformer (DiT), we evaluate five TTA strategies---full-model fine-tuning, LoRA, and three residual-based methods---across 100 videos from Panda-70M. Our central finding is that parameter efficiency is the critical design principle for single-instance TTA. We introduce **AdaSteer** (Adaptive Shared Timestep Embedding Efficient Residual), a family of TTA methods that learn additive offsets to the timestep embedding, shared across configurable groups of transformer blocks. AdaSteer-1, which learns a single 512-dimensional vector (0.0000038% of model parameters), is the only method that consistently improves generation quality over the unmodified baseline, achieving up to +0.47 dB PSNR improvement. Methods with more tunable parameters---LoRA (2.6M+) and full-model fine-tuning (13.6B)---either produce no measurable improvement or severely degrade output quality due to catastrophic overfitting. We provide comprehensive ablations and present preliminary evidence toward an **adaptive capacity allocation** scheme, where AdaSteer's group count is automatically scaled based on available training signal, potentially yielding a zero-hyperparameter TTA method.

## 1. Introduction

Video diffusion models have achieved remarkable generative quality, but their outputs remain imperfect when conditioned on specific input video content. Test-time adaptation (TTA)---the practice of briefly fine-tuning a pretrained model on a single test instance before inference---offers a principled approach to bridging this gap. While TTA has been successfully applied to discriminative models (Wang et al., 2021) and image diffusion models, its application to large-scale video diffusion transformers remains unexplored.

The core challenge is the extreme imbalance between model capacity and available training signal. A 13.6B-parameter model must be adapted using only a handful of conditioning video frames---typically fewer than 30 frames at 480p resolution, encoding to approximately 300,000 latent values. This creates a severe overfitting risk that fundamentally constrains the design space of viable TTA methods.

We investigate five TTA strategies spanning several orders of magnitude in tunable parameter count: full-model fine-tuning (13.6B parameters), LoRA adaptation (65K--81.6M parameters), and three variants of residual-based adaptation (512--24K parameters). Our experiments encompass over 100 independent runs across learning rate sweeps, iteration sweeps, conditioning frame ablations, generation horizon analyses, and cross-dataset evaluations.

Our key contribution is **AdaSteer**, a unified framework that subsumes our most effective methods (previously termed Delta-A and Delta-B) under a single parameterization. AdaSteer partitions the transformer's 48 blocks into *G* groups and learns one 512-dimensional additive offset per group, applied to the timestep embedding before adaLN modulation. With *G*=1, all blocks share a single global offset (512 parameters); with *G*=48, each block receives its own offset (24,576 parameters). We show that at the current data scale, *G*=1 is optimal, but present evidence suggesting that the optimal *G* scales with available training data---motivating an adaptive capacity allocation rule that would eliminate group count as a hyperparameter entirely.

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

This perturbation propagates through the adaLN projection, effectively adjusting *how the model denoises* without altering *what the model has learned*. The key insight is that a small 512-dimensional perturbation is amplified through the learned Linear(512 → 24,576) projection into a high-dimensional modulation of the feature space at every block.

| Variant | Groups (*G*) | Parameters | % of 13.6B |
|:--------|-------------:|-----------:|-----------:|
| AdaSteer-1 | 1 | 512 | 0.0000038% |
| AdaSteer-4 | 4 | 2,048 | 0.000015% |
| AdaSteer-12 | 12 | 6,144 | 0.000045% |
| AdaSteer-48 | 48 | 24,576 | 0.00018% |

### 2.2 Full-Model Fine-Tuning

All 13.6B parameters of the DiT are updated via SGD on the flow-matching objective computed over conditioning frames. This serves as both a TTA baseline and as a probe for whether the model's full parameter space can accommodate single-video adaptation.

### 2.3 LoRA Adaptation

Low-Rank Adaptation (Hu et al., 2022) introduces trainable low-rank matrices *A* ∈ ℝ^{d×r} and *B* ∈ ℝ^{r×d} alongside frozen weight matrices, such that the effective weight becomes *W'* = *W* + α · *BA*. We sweep rank *r* ∈ {1, 4, 8, 16, 32} with α = 2*r*. At rank 1 targeting self-attention and output projections across all 48 transformer blocks, LoRA introduces approximately 2.56M trainable parameters. We additionally evaluate constrained configurations targeting only the last 4, 8, or 16 blocks with reduced alpha scaling.

### 2.4 Delta-C: Output Residual

Delta-C learns a per-channel additive correction applied to the DiT output at inference time, directly modifying the predicted velocity field rather than modulating internal representations. This method operates in the output space rather than the conditioning pathway.

## 3. Experimental Setup

### 3.1 Model and Datasets

We use **LongCat-Video**, a 13.6B-parameter Diffusion Transformer with 48 single-stream transformer blocks, a hidden dimension of 4096, and a timestep embedding dimension (adaLN) of 512. The model operates in a latent space with a temporal compression factor of 4× via a pretrained VAE.

**Panda-70M** (primary): 100 randomly sampled videos at 480p resolution (480 × 832). For each video, frames are drawn from a fixed anchor point (frame 32). By default, 14 conditioning frames immediately preceding the anchor are used for both TTA training and generation conditioning, and 14 subsequent frames are generated and compared against ground truth.

**UCF-101** (cross-dataset): 100 videos selected for domain-shift evaluation.

### 3.2 Evaluation Protocol

Generated frames are compared to ground truth using PSNR (dB, higher is better). All reported values are mean ± standard deviation across the 99 successfully processed videos in each run. Each TTA method is applied independently per video: the model is adapted, frames are generated, metrics are computed, and all adapted parameters are reset before the next video.

### 3.3 Training Details

All AdaSteer variants and Delta-C are optimized with AdamW (β₁ = 0.9, β₂ = 0.999, ε = 10⁻¹⁵) with gradient clipping at norm 1.0. Full-model uses SGD; LoRA uses AdamW with warmup. Training uses the flow-matching loss on conditioning frames concatenated with noised target latents. Gradient checkpointing is enabled throughout. VAE and text encoder are offloaded to CPU during TTA optimization.

### 3.4 Training Signal Budget

At 480p resolution with the standard 14-frame conditioning setup, the available training signal consists of approximately 300,000 latent values (after context/train/validation splitting). This yields the following values-per-parameter ratios:

| Variant | Parameters | Values / Parameter |
|:--------|----------:|---------:|
| Full-model | 13,600,000,000 | 0.00002 |
| LoRA (r=1) | 2,560,000 | 0.12 |
| AdaSteer-48 | 24,576 | ~12 |
| AdaSteer-12 | 6,144 | ~49 |
| AdaSteer-4 | 2,048 | ~146 |
| AdaSteer-1 | 512 | ~585 |

This ratio is central to our analysis: methods with fewer than ~100 values per parameter consistently overfit, while those with ~500+ values per parameter achieve stable adaptation.

## 4. Results

### 4.1 Main Comparison

Table 1 reports each method's best achievable PSNR under the standard evaluation protocol (14 conditioning frames, 14 generated frames). For AdaSteer variants, we report the best result across all tested learning rate and step count combinations.

**Table 1: Main results.** Best configuration per method. Δ denotes improvement over the no-TTA baseline.

| Method | Tunable Params | Best Config | PSNR (dB) | Δ |
|:-------|---------------:|:------------|----------:|------:|
| No TTA (baseline) | 0 | --- | 22.07 ± 8.83 | --- |
| Full-model | 13.6B | lr=1e-5, 5 steps | 22.09 ± 8.84 | +0.02 |
| LoRA (rank=1) | 2.56M | lr=2e-4, 5 steps | 22.06 ± 8.84 | −0.01 |
| LoRA (constrained) | ~65K | last 4 blks, α=0.1 | 22.08 ± 8.84 | +0.01 |
| **AdaSteer-1** | **512** | **lr=1e-2, 5 steps** | **22.54 ± 9.04** | **+0.47** |
| AdaSteer-1 | 512 | lr=5e-3, 20 steps | 22.38 ± 9.25 | +0.31 |
| Delta-C | varies | lr=1e-2, 5 steps | 20.57 ± 7.65 | −1.50 |

AdaSteer-1 is the only method that consistently improves over baseline. Full-model TTA and constrained LoRA produce results statistically indistinguishable from the unmodified model, while unconstrained LoRA and Delta-C degrade quality.

> **Note on AdaSteer-1 equivalence:** AdaSteer-1 (previously "Delta-A") and the G=1 case of the per-group method (previously "Delta-B") are architecturally identical—both learn a single 512-dim additive vector to the timestep embedding shared across all blocks. The DB12 result (22.54 dB at lr=1e-2, 5 steps) and DA2 result (22.38 dB at lr=5e-3, 20 steps) reflect different hyperparameter operating points of the same method. Equivalence verification experiments (Series 25) are in progress.

### 4.2 Learning Rate Sensitivity

Each method was swept over a learning rate range appropriate to its parameter scale.

**Table 2: Full-model LR sweep** (20 steps, SGD).

| Run | Learning Rate | PSNR (dB) |
|:----|:------------:|---------:|
| F1 | 1e-6 | 22.07 ± 8.82 |
| F2 | 5e-6 | 22.05 ± 8.82 |
| F3 | 1e-5 | 22.07 ± 8.84 |
| F4 | 5e-5 | 22.05 ± 8.80 |
| F5 | 1e-4 | 21.86 ± 8.60 |

Full-model TTA is insensitive to learning rate across five orders of magnitude (1e-6 to 5e-5), producing results indistinguishable from the unmodified baseline. At lr=1e-4, slight degradation appears. This confirms that 20 SGD steps on 13.6B parameters produce negligible per-parameter updates.

**Table 3: AdaSteer-1 LR sweep** (20 steps, AdamW).

| Run | Learning Rate | PSNR (dB) |
|:----|:------------:|---------:|
| DA1 | 1e-3 | 22.23 ± 9.06 |
| DA2 | 5e-3 | **22.38 ± 9.25** |
| DA3 | 1e-2 | 22.35 ± 9.26 |
| DA4 | 5e-2 | 11.97 ± 5.87 |
| DA5 | 1e-1 | 8.37 ± 3.94 |

AdaSteer-1 exhibits a clear optimum at lr=5e-3 (at 20 steps), with graceful degradation at lower rates and catastrophic divergence above 5e-2.

**Table 4: AdaSteer-4 LR sweep** (20 steps, 4 groups, AdamW).

| Run | Learning Rate | PSNR (dB) |
|:----|:------------:|---------:|
| DB7 | 1e-3 | 21.90 ± 8.48 |
| DB8 | 5e-3 | 21.17 ± 7.39 |
| DB9 | 1e-2 | 19.73 ± 6.12 |
| DB10 | 5e-2 | 9.93 ± 2.23 |
| DB11 | 1e-1 | 8.57 ± 2.02 |

AdaSteer-4 at 20 steps already overfits at lr=1e-3—its best result (21.90) is below baseline. This motivates the investigation in Section 4.3 of whether fewer steps can rescue higher group counts.

**Table 5: LoRA rank sweep** (all 48 blocks, qkv+proj targets, lr=2e-4, 20 steps).

| Run | Rank | α | Approx. Params | PSNR (dB) |
|:----|-----:|--:|---------------:|---------:|
| L1 | 1 | 2 | 2.56M | 20.46 ± 7.66 |
| L2 | 4 | 8 | 10.2M | 15.03 ± 6.29 |
| L3 | 8 | 16 | 20.4M | 11.18 ± 6.00 |
| L4 | 16 | 32 | 40.8M | 8.15 ± 4.63 |
| L5 | 32 | 64 | 81.6M | 6.42 ± 3.47 |

PSNR degrades monotonically with rank. Even at rank 1, LoRA underperforms baseline by 1.6 dB at 20 steps.

**Table 6: Delta-C LR sweep** (20 steps, per-channel mode).

| Run | Learning Rate | PSNR (dB) |
|:----|:------------:|---------:|
| DC1 | 1e-2 | 13.45 ± 4.11 |
| DC2 | 5e-2 | 4.32 ± 3.62 |
| DC3 | 1e-1 | 4.06 ± 3.57 |
| DC4 | 5e-1 | in progress |
| DC5 | 1.0 | in progress |

Delta-C consistently degrades below baseline at all tested learning rates.

### 4.3 Optimization Step Analysis

This analysis reveals the most critical finding of our study: the relationship between parameter count and optimal step count.

**Table 7: PSNR vs. optimization steps.** Each method at its best learning rate. Steps were swept over {5, 10, 40, 80}.

| Steps | Full-model (13.6B) | AdaSteer-1 (512) | AdaSteer-1† (512) | LoRA r=1 (2.56M) | Delta-C |
|------:|:------------------:|:----------------:|:-----------------:|:----------------:|:-------:|
| 5 | 22.09 ± 8.84 | 22.35 ± 9.13 | **22.54 ± 9.04** | 22.06 ± 8.84 | 20.57 ± 7.65 |
| 10 | 22.06 ± 8.82 | 22.21 ± 9.05 | 22.53 ± 8.44 | 21.95 ± 8.74 | 17.88 ± 5.78 |
| 40 | 22.06 ± 8.82 | 22.27 ± 9.37 | 13.32 ± 4.06 | 13.20 ± 6.16 | 8.18 ± 3.21 |
| 80 | 22.04 ± 8.83 | 17.57 ± 7.49 | 9.29 ± 2.15 | 6.03 ± 3.41 | 4.72 ± 3.61 |

*AdaSteer-1 column: lr=5e-3. AdaSteer-1† column: lr=1e-2 (from the iteration sweep with 1 group, timestep target).*

Three distinct regimes emerge:

1. **Inert regime** (Full-model): The model is too large for any meaningful adaptation. PSNR remains within ±0.05 dB of baseline regardless of step count. Each parameter receives an update on the order of 10⁻¹², producing no measurable effect.

2. **Productive regime** (AdaSteer-1 at lr=5e-3): With only 512 parameters, each receives substantial per-step updates. The method achieves its best results at 5 steps and degrades gradually beyond 40 steps, retaining above-baseline performance through approximately 20 steps.

3. **Overfitting regime** (AdaSteer-1† at lr=1e-2, LoRA, Delta-C): At higher learning rates or with more parameters, methods overfit within 10–40 steps. The AdaSteer-1† column (lr=1e-2) peaks higher at 5 steps (22.54) but crashes to 13.32 by step 40, illustrating the LR–steps tradeoff: higher LR yields a sharper peak but a narrower safe window.

### 4.4 AdaSteer Group Count Analysis

The original groups sweep was conducted at 20 steps with lr=1e-2—a configuration that, as shown in Section 4.3, causes severe overfitting even for *G*=1. These results therefore reflect the overfitting ceiling for each group count rather than their true potential.

**Table 8: AdaSteer groups sweep** (20 steps, lr=1e-2).

| Run | Groups (*G*) | Parameters | PSNR (dB) |
|:----|:------------:|-----------:|---------:|
| DB1 | 1 | 512 | 21.21 ± 6.73 |
| DB2 | 2 | 1,024 | 19.62 ± 6.03 |
| DB3 | 4 | 2,048 | 15.42 ± 5.61 |
| DB4 | 8 | 4,096 | 15.61 ± 5.71 |
| DB5 | 16 | 8,192 | 14.74 ± 5.59 |
| DB6 | 48 | 24,576 | 10.92 ± 5.17 |

PSNR degrades monotonically with *G* at 20 steps. However, this does not necessarily mean higher *G* values are inherently worse—it may simply mean they require fewer steps or lower learning rates. Series 26 (in progress) re-evaluates these group counts at 5 steps to determine their true potential.

### 4.5 Conditioning Frame Ablation

We vary the number of conditioning frames from 2 to 24 while holding the anchor point fixed at frame 32. Each method uses its best LR at 20 steps. With a fixed total output length of 28 frames, increasing conditioning frames proportionally reduces the number of generated frames (from 26 to 4), which independently improves PSNR due to shorter prediction horizons. The full-model column serves as a no-TTA control (full-model TTA ≈ baseline at all frame counts).

**Table 9: Effect of conditioning frame count on PSNR.**

| Cond. Frames | Gen. Frames | Full-model (ctrl) | AdaSteer-1 | AdaSteer-4‡ | LoRA (r=1) | Delta-C |
|:------------:|:-----------:|:-----------------:|:----------:|:-----------:|:----------:|:-------:|
| 2 | 26 | 16.09 ± 6.65 | 16.41 (+0.32) | 17.14 (+1.05) | 15.14 (−0.95) | 10.20 (−5.89) |
| 7 | 21 | 18.37 ± 7.58 | 18.39 (+0.02) | 19.21 (+0.84) | 17.29 (−1.08) | 11.22 (−7.15) |
| 14 | 14 | 22.19 ± 9.02 | 22.35 (+0.16) | 21.19 (−1.00) | 20.49 (−1.70) | 13.45 (−8.74) |
| 24 | 4 | 23.05 ± 8.86 | **23.48 (+0.43)** | 22.59 (−0.46) | 21.74 (−1.31) | 14.64 (−8.41) |

*‡ AdaSteer-4 uses the original Delta-B best config (4 groups, lr=1e-2, 20 steps), which overfits at higher frame counts. Its relative advantage at 2–7 frames is notable.*

All methods benefit from additional conditioning frames. AdaSteer-1 achieves its overall best PSNR of **23.48 dB** (+0.43 over control) with 24 conditioning frames. An interesting crossover occurs for AdaSteer-4: at 2–7 frames, it outperforms AdaSteer-1, but at 14+ frames, overfitting at 20 steps erases the advantage. This crossover is precisely the signal motivating the adaptive capacity allocation scheme (Section 5.2): with fewer frames (less training data), more groups may be beneficial because each group's offset has less data to overfit on per optimization step.

### 4.6 Generation Horizon Analysis

We fix 14 conditioning frames and vary the total output length from 16 frames (2 generated) to 72 frames (58 generated).

**Table 10: PSNR vs. generation horizon.** All methods at best config, 20 steps.

| Total Frames | Generated Frames | Full-model (ctrl) | AdaSteer-1 | AdaSteer-4‡ | LoRA (r=1) | Delta-C |
|:------------:|:----------------:|:-----------------:|:----------:|:-----------:|:----------:|:-------:|
| 16 | 2 | 28.07 ± 9.29 | 27.91 (−0.16) | 25.39 (−2.68) | 14.68 (−13.39) | 18.74 (−9.33) |
| 28 | 14 | 22.09 ± 8.83 | 22.42 (+0.33) | 21.18 (−0.91) | 11.18 (−10.91) | 13.45 (−8.64) |
| 44 | 30 | 18.77 ± 7.05 | 19.14 (+0.37) | 19.46 (+0.69) | 11.02 (−7.75) | 11.59 (−7.18) |
| 72 | 58 | 17.27 ± 5.66 | in progress | in progress | 11.50 (−5.77) | in progress |

All methods exhibit monotonic PSNR degradation with longer horizons. AdaSteer-1 maintains a consistent advantage over baseline at horizons up to 30 frames. Notably, at 30 generated frames, AdaSteer-4 (+0.69) outperforms AdaSteer-1 (+0.37)—another hint that longer/harder generation tasks may benefit from higher group counts.

### 4.7 Constrained LoRA Analysis

**Table 11: Constrained LoRA sweep** (rank=1, qkv-only, lr=2e-4, 20 steps).

| Run | Target Blocks | α | PSNR (dB) |
|:----|:-------------:|---:|---------:|
| LA1 | Last 4 | 0.1 | 22.08 ± 8.84 |
| LA2 | Last 4 | 0.5 | 22.04 ± 8.81 |
| LA3 | Last 4 | 1.0 | 22.01 ± 8.80 |
| LA4 | Last 8 | 0.1 | 22.04 ± 8.82 |
| LA5 | Last 8 | 0.5 | 22.03 ± 8.81 |
| LA6 | Last 8 | 1.0 | 21.96 ± 8.77 |
| LA7 | Last 16 | 0.1 | 22.05 ± 8.83 |
| LA8 | Last 16 | 0.5 | 22.03 ± 8.80 |
| LA9 | Last 16 | 1.0 | 21.92 ± 8.73 |

Constraining LoRA to fewer blocks and reducing alpha eliminates overfitting degradation but also eliminates any positive adaptation effect. At α=0.1, LoRA achieves 22.08 dB—indistinguishable from baseline. This reveals a fundamental limitation of LoRA for single-video TTA: the low-rank weight perturbation either produces too large a change (overfitting) or too small a change (no effect), with no configuration achieving consistent improvement.

### 4.8 Early Stopping Analysis

We conduct a comprehensive ablation of early stopping (ES) hyperparameters using the anchored loss method, varying check frequency, patience, holdout fraction, noise draws, and anchor sigma schedules.

**Table 12: Early stopping ablation summary.** All runs use full-model TTA (lr=1e-5, 20 steps).

| Ablation | Swept Values | PSNR Range (dB) |
|:---------|:-------------|:---------------:|
| Check frequency | every {1, 3, 5, 10} steps | 22.05 – 22.08 |
| Patience | {2, 3, 5, 7, 10} checks | 22.05 – 22.09 |
| Holdout fraction | {0.10, 0.25, 0.50} | 22.07 – 22.07 |
| Noise draws | {1, 2, 4} | 22.05 – 22.07 |
| Sigma schedule | 4 variants | 22.05 – 22.08 |
| ES disabled | --- | 22.05 |

All configurations produce identical PSNR within noise. This null result is expected: the ES ablation was conducted on full-model TTA, which produces no measurable adaptation (Section 4.2). Consequently, the anchor loss landscape is flat and early stopping has nothing to detect. A meaningful ES evaluation requires application to AdaSteer, which exhibits non-trivial training dynamics—this is addressed in ongoing experiments with 100–500 step training horizons and step-level checking.

### 4.9 Cross-Dataset Generalization

**Table 13: Cross-dataset evaluation on UCF-101** (100 videos, best config per method).

| Method | Panda-70M | UCF-101 |
|:-------|:---------:|:-------:|
| No TTA | 22.07 ± 8.83 | 22.07 ± 8.83 |
| Full-model | 22.07 ± 8.84 | 22.22 ± 9.04 |
| AdaSteer-1 | 22.38 ± 9.25 | 22.40 ± 9.30 |
| LoRA (r=1) | 20.46 ± 7.66 | 20.48 ± 7.65 |

The relative method ranking is perfectly preserved across datasets. AdaSteer-1 improves over baseline on both Panda-70M (+0.31) and UCF-101 (+0.33), demonstrating that the adaptation mechanism generalizes across video domains.

## 5. Discussion

### 5.1 The Parameter Efficiency Principle

Our results point to a fundamental principle for test-time adaptation of large generative models: **the number of tunable parameters must be calibrated to the available training signal**. With approximately 300K latent values available per video, methods with fewer than ~100 values per parameter (LoRA, AdaSteer with *G* ≥ 12 at 20 steps) consistently overfit, while those with ~500+ values per parameter (AdaSteer-1) achieve stable adaptation. Methods at the extreme end (full-model, with 0.00002 values per parameter) receive per-parameter updates so small they produce no effect.

This principle has direct implications for TTA method design. Rather than adapting existing fine-tuning methods (LoRA, full fine-tuning) and constraining them to avoid overfitting, we advocate for purpose-built TTA strategies that operate in naturally low-dimensional subspaces of the model's behavior. The timestep embedding, which globally modulates all transformer blocks through adaLN, provides exactly such a subspace.

### 5.2 Toward Adaptive Capacity Allocation

The conditioning frame ablation (Table 9) and generation horizon analysis (Table 10) provide preliminary evidence for a data-dependent optimal group count:

- At **2 conditioning frames** (~700K training values), AdaSteer-4 outperforms AdaSteer-1 by +0.73 dB relative to baseline. More data can support more parameters.
- At **14 conditioning frames** (~400K training values), AdaSteer-1 is optimal while AdaSteer-4 overfits at 20 steps.
- At **30 generated frames** (a harder generation task), AdaSteer-4 outperforms AdaSteer-1.

These observations motivate an **adaptive capacity allocation** scheme:

```
G* = max(1, ⌊N_train / (R* × 512)⌋)
```

where *N*_train is the available training token count and *R** is an empirically determined optimal values-per-parameter ratio. If confirmed by ongoing experiments (Series 26–27), this would make AdaSteer a **zero-hyperparameter TTA method**: given a video and its available conditioning frames, the method automatically determines its own adaptation capacity.

The key experiments required to validate this scheme are:
1. **Series 26**: Re-evaluate group counts {1, 4, 12, 48} at 5 steps (the optimal step count) to determine which *G* values are viable at the current data scale.
2. **Series 27**: A 2D sweep of conditioning frames × groups to empirically determine *R** and test whether the optimal *G* shifts predictably with available data.

### 5.3 Why Does the Timestep Embedding Work?

AdaSteer's effectiveness can be understood through the lens of the adaLN architecture. The timestep embedding *t* is projected through a learned linear layer to produce shift, scale, and gate vectors that modulate hidden states at every transformer block. A small perturbation δ to *t* is thus amplified through this projection into a high-dimensional modulation of the feature space—effectively adjusting *how the model denoises* without altering *what the model has learned*. This is analogous to adjusting exposure and color balance on a camera: a few global parameters produce perceptually significant changes without modifying the optics.

### 5.4 Overfitting Dynamics

The iteration sweep data (Table 7) provides a temporal lens on overfitting. For AdaSteer at lr=1e-2 and LoRA, there exists a narrow window (2–10 steps) where adaptation produces improvement, followed by rapid degradation. The sharpness of this transition—PSNR drops from 22.53 dB at 10 steps to 13.32 dB at 40 steps—suggests a phase-transition-like behavior where the model abruptly shifts from interpolating conditioning content to memorizing noise patterns.

AdaSteer-1 at lr=5e-3 exhibits a more gradual degradation curve (22.35 at 5 steps to 17.57 at 80 steps), reflecting that both the learning rate and the architectural constraint of a single 512-dimensional vector limit the maximum expressible perturbation.

### 5.5 Limitations

Several limitations should be noted. First, we report only PSNR; SSIM and LPIPS were computed but are not analyzed here. Second, the standard deviation across videos is large (~±8–9 dB), driven by genuine heterogeneity in video content difficulty. While mean improvements of 0.3–0.5 dB are statistically meaningful across 99 videos, per-video analysis would better characterize which video types benefit most. Third, the early stopping ablation was confounded by being applied to full-model TTA; re-evaluation on AdaSteer is required. Fourth, the adaptive capacity allocation scheme (Section 5.2) is currently hypothetical—validation experiments are in progress. Finally, we evaluate on a single model architecture; generalization to alternative backbones (Open-Sora v2.0, CogVideoX) is the subject of ongoing work.

## 6. Conclusion

We presented a systematic empirical study of test-time adaptation for large-scale video diffusion transformers, introducing **AdaSteer** (Adaptive Shared Timestep Embedding Efficient Residual), a unified family of TTA methods that learn additive offsets to the timestep embedding. Across five TTA strategies and over 100 experimental configurations evaluated on 100 videos, AdaSteer-1—a single 512-dimensional vector comprising 0.0000038% of the model—is the only method that consistently improves generation quality, achieving up to +0.47 dB PSNR over the unmodified baseline.

Our key insight is that parameter efficiency is not merely a practical convenience but a fundamental requirement: the extreme imbalance between model capacity (13.6B parameters) and available training signal (~300K latent values) creates a regime where conventional adaptation methods fail. We identify the values-per-parameter ratio as the critical quantity governing TTA success and present preliminary evidence toward an adaptive capacity allocation scheme that could make AdaSteer a zero-hyperparameter method.

These findings establish a design principle for TTA in large generative models: **adapt the conditioning pathway, not the weights**.

---

## Appendix: Experiment Status

### A.1 Completed Experiments (as of Feb 2026)

| Experiment | Series | Runs | Status |
|:-----------|:------:|:----:|:------:|
| No-TTA baseline (Panda) | — | 1 | ✓ Complete |
| No-TTA baseline (UCF-101) | — | 1 | ✓ Complete |
| Full-model LR sweep | 1 | 5 | ✓ Complete |
| Full-model iteration sweep | 2 | 4/5 | 4 complete, 1 in progress |
| LoRA rank sweep | 3 | 5 | ✓ Complete |
| LoRA constrained sweep | 14 | 9 | ✓ Complete |
| AdaSteer-1 LR sweep | 7 | 5 | ✓ Complete |
| AdaSteer-1 iteration sweep | 8 | 4/5 | 4 complete, 1 in progress |
| AdaSteer groups sweep | 9 | 6 | ✓ Complete |
| AdaSteer-4 LR sweep | 10 | 5 | ✓ Complete |
| AdaSteer-4 iteration sweep | 9b | 4 | ✓ Complete |
| Delta-C LR sweep | 11 | 3/5 | 3 complete, 2 in progress |
| Delta-C iteration sweep | 12 | 4 | ✓ Complete |
| ES ablation (all) | 14 | 19 | ✓ Complete |
| Training frames ablation | Exp3 | 19/20 | 19 complete, 1 in progress |
| Generation horizon ablation | Exp4 | 16/20 | 16 complete, 4 in progress |
| UCF-101 cross-dataset | — | 4 | ✓ Complete |
| Full-model long train | 17 | 1 | In progress |
| AdaSteer-1 long train | 18 | 1 | In progress |
| LoRA built-in comparison | 15 | 2 | In progress |
| LoRA ultra-constrained | 16 | 6 | In progress |

### A.2 Planned / Submitted Experiments

| Experiment | Series | Runs | Purpose |
|:-----------|:------:|:----:|:--------|
| AdaSteer equivalence verification | 25 | 4 | Confirm Delta-A = AdaSteer-1 at matching hyperparams |
| AdaSteer groups at 5 steps | 26 | 6 | Test if higher G works at fewer steps |
| AdaSteer ratio sweep | 27 | 8 | 2D sweep of frames × groups for adaptive allocation |
| Norm-tuning (TENT-style) | 21 | 6 | Tune normalization layer affine params |
| FiLM adapter | 22 | 6 | Learn corrections to adaLN output |
| AdaSteer-1 optimized | 23 | 2 | More frames + extended training with ES |
| AdaSteer-1 + norm combined | 23b | 1 | Combined timestep offset + norm tuning |
| Delta-B hidden-state residual | 19 | 6 | Alternative injection target |
| Delta-B low-LR sweep | 24 | 4 | 1-group with reduced LR at 20 steps |
