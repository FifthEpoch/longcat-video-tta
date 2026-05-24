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
| Panda-70M (200 discovery) | datasets/panda_200_480p | 200 | Planned deterministic subset from panda_1000_480p for cheap standard-horizon parameter sweeps. |
| Panda-70M (100) | datasets/panda_100_480p | 100 | Original small subset. DIFFERENT videos from panda_1000_480p. |
| UCF-101 (200 discovery) | datasets/ucf101_200_480p | 200 | Planned stratified subset from ucf101_1000_480p for discovery sweeps. |
| UCF-101 (100) | datasets/ucf101_test_480p | 100 | Action recognition clips. |
| UCF-101 (500) | datasets/ucf101_test_480p | 500 | Larger UCF eval (Phase 2). |

CRITICAL: panda_1000_480p and panda_100_480p contain DIFFERENT videos with different difficulty distributions. Results are NOT directly comparable across these datasets.

### May 2026 Discovery Funnel

Initial parameter sweeps should run on 200-video discovery subsets before any new 1000-video paper run. The first sweep stage uses the standard 28-frame setting, not long-context Panda, because full-scale standard Panda already shows the clearest FVD gain and is cheaper than 93-frame long-context generation. Long-context Panda should be reserved for validating winners or testing horizon-specific objectives after the short-horizon sweep identifies stable hyperparameters.

Planned configs:
- `sweep_experiment/configs/panda_200_adasteer_steps_lr.yaml`
- `sweep_experiment/configs/ucf101_200_adasteer_steps_lr.yaml`

Promotion rule: only scale configs that improve FVD and do not regress PSNR/SSIM/LPIPS on the 200-video discovery set.

---

## May 2026 - 200-Video Standard-Horizon AdaSteer Discovery Sweep

Date pasted/logged: May 18-19, 2026

Purpose: cheap standard-horizon parameter sweep on new 200-video discovery subsets before promoting any config to 1000-video paper runs.

Shared config:
- `num_cond_frames=14`
- `num_frames=28`
- `gen_start_frame=48`
- `tta_total_frames=48`
- `tta_context_frames=14`
- `num_inference_steps=50`
- `guidance_scale=4.0`
- `resolution=480p`
- `seed=42`
- `max_videos=200`
- `compute_fvd=true`
- `compute_fid=true`
- `es_disable=true`
- `no_save_videos=true`

Configs:
- `sweep_experiment/configs/panda_200_adasteer_steps_lr.yaml`
- `sweep_experiment/configs/ucf101_200_adasteer_steps_lr.yaml`

Status: complete. All 10 runs finished for both Panda and UCF after resuming the checkpointed `S10_*` jobs.

### Panda-70M 200 Discovery

Dataset: `datasets/panda_200_480p`

Use the in-series `NOTTA` row as the matched baseline for this subset. The exporter baseline matched against an older Panda no-TTA run and should not be used for deltas here.

| Run ID | Status | N | PSNR | dPSNR vs in-series NOTTA | SSIM | dSSIM | LPIPS | dLPIPS | FVD | dFVD | FID |
|--------|--------|---:|-----:|-------------------------:|-----:|------:|------:|-------:|----:|-----:|----:|
| NOTTA | complete | 200 | 18.3676 | -- | 0.6564 | -- | 0.3290 | -- | 333.70 | -- | 54.13 |
| S3_LR001 | complete | 200 | 18.4009 | +0.0332 | 0.6564 | -0.0001 | 0.3285 | -0.0005 | 337.46 | +3.76 | 54.49 |
| S3_LR0025 | complete | 200 | 18.3792 | +0.0116 | 0.6549 | -0.0015 | 0.3297 | +0.0007 | 327.55 | -6.15 | 54.53 |
| S3_LR005 | complete | 200 | 18.3524 | -0.0153 | 0.6559 | -0.0005 | 0.3299 | +0.0009 | 328.17 | -5.53 | 53.72 |
| S5_LR001 | complete | 200 | 18.3804 | +0.0127 | 0.6552 | -0.0013 | 0.3298 | +0.0008 | 338.51 | +4.81 | 54.29 |
| S5_LR0025 | complete | 200 | 18.3957 | +0.0281 | 0.6567 | +0.0002 | 0.3280 | -0.0010 | 348.08 | +14.38 | 54.79 |
| S5_LR005 | complete | 200 | 18.4057 | +0.0380 | 0.6560 | -0.0005 | 0.3288 | -0.0002 | 339.15 | +5.45 | 55.46 |
| S10_LR001 | complete | 200 | 18.3980 | +0.0304 | 0.6577 | +0.0013 | 0.3282 | -0.0008 | 339.07 | +5.37 | 54.46 |
| S10_LR0025 | complete | 200 | 18.3977 | +0.0301 | 0.6563 | -0.0001 | 0.3291 | +0.0001 | 339.15 | +5.45 | 54.56 |
| S10_LR005 | complete | 200 | 18.4196 | +0.0520 | 0.6572 | +0.0008 | 0.3272 | -0.0018 | 316.34 | -17.36 | 53.59 |

Takeaways:
- `S10_LR005` is the clear Panda winner: FVD improves 333.70 -> 316.34 (-17.36 / -5.2%), FID improves 54.13 -> 53.59, and pointwise metrics also improve (PSNR +0.0520, SSIM +0.0008, LPIPS -0.0018).
- This is the only completed Panda config that satisfies the promotion rule across FVD and pointwise metrics.
- Candidate promotion: run `S10_LR005` on the full 1000-video standard Panda setting.

### UCF-101 200 Discovery

Dataset: `datasets/ucf101_200_480p`

Raw summary-level PSNR/SSIM/LPIPS fields were `nan` in the direct audit output, but the exporter table computed pointwise metrics from per-video records. Use the FVD/FID table below for promotion decisions until the summaries are audited.

| Run ID | Status | N | FVD | dFVD vs in-series NOTTA | FID | Exporter pointwise note |
|--------|--------|---:|----:|------------------------:|----:|-------------------------|
| NOTTA | complete | 200 | 359.80 | -- | 32.70 | exporter: PSNR 20.4417, SSIM 0.7356, LPIPS 0.2340 |
| S3_LR001 | complete | 200 | 357.92 | -1.88 | 32.73 | near-neutral pointwise |
| S3_LR0025 | complete | 200 | 366.58 | +6.78 | 32.63 | near-neutral pointwise |
| S3_LR005 | complete | 200 | 363.61 | +3.81 | 32.77 | near-neutral pointwise |
| S5_LR001 | complete | 200 | 347.09 | -12.71 | 32.78 | exporter: PSNR 20.4330, SSIM 0.7354, LPIPS 0.2342 |
| S5_LR0025 | complete | 200 | 353.30 | -6.50 | 32.72 | exporter: best completed SSIM/LPIPS among pasted rows |
| S5_LR005 | complete | 200 | 361.99 | +2.19 | 32.89 | exporter: best completed PSNR among pasted rows |
| S10_LR001 | complete | 200 | 360.40 | +0.60 | 32.62 | exporter: PSNR 20.4484, SSIM 0.7356, LPIPS 0.2336 |
| S10_LR0025 | complete | 200 | 359.83 | +0.03 | 32.73 | exporter: PSNR 20.4465, SSIM 0.7353, LPIPS 0.2336 |
| S10_LR005 | complete | 200 | 362.88 | +3.08 | 32.56 | exporter: PSNR 20.4759, SSIM 0.7353, LPIPS 0.2330 |

Takeaways:
- Best UCF FVD is `S5_LR001` (359.80 -> 347.09, -12.71 / -3.5%), but exporter pointwise metrics are slightly worse than in-series `NOTTA`.
- The best UCF pointwise tradeoff is `S5_LR0025`: FVD improves 359.80 -> 353.30 (-6.50 / -1.8%), while exporter pointwise metrics improve over in-series `NOTTA` (PSNR +0.0185, SSIM +0.0008, LPIPS -0.0009).
- `S10_*` improves PSNR/LPIPS but does not improve FVD, so 10 steps is not preferred for UCF.
- Candidate promotion: `S5_LR0025` is the safer UCF config if we require both FVD and pointwise gains; `S5_LR001` is the FVD-only winner.

Next action:
- Prepare 1000-video validation configs for Panda `S10_LR005` and UCF `S5_LR0025`, after confirming we want to spend full-scale compute.
- Audit why UCF summary-level pointwise means are `nan` while exporter table pointwise values are populated.

---

## May 2026 - 200-Video Anchor-Loss Gate Sweep

Date pasted/logged: May 22, 2026

Purpose: test whether the existing anchor-loss validation signal can gate AdaSteer generation-time deltas. This reuses the early-stopping anchor loss, but changes the action from "stop/restore best step" to "use/skip/scale delta at generation time."

Configs:
- `sweep_experiment/configs/panda_200_anchor_gate.yaml`
- `sweep_experiment/configs/ucf101_200_anchor_gate.yaml`

Implementation commit: `b57f362 Add anchor-loss gating for AdaSteer`

Shared setup:
- Panda base config: `delta_steps=10`, `delta_lr=0.005` (winner from 200-video step/LR sweep).
- UCF base config: `delta_steps=5`, `delta_lr=0.0025` (balanced candidate from 200-video step/LR sweep).
- ES/anchor validation enabled: `es_disable=false`, `es_check_every=5`, `es_patience=3`, `es_anchor_sigmas=0.25,0.5,0.75`, `es_noise_draws=2`, `es_holdout_fraction=0.25`.
- Gate modes tested: `G_OFF`, `G_LOG`, `G_BIN_0`, `G_BIN_001`, `G_SOFT_001`.

SLURM status:
- Panda `G_OFF` job `9218476` failed quickly (`FAILED`, exit `2:0`, elapsed `00:01:14`).
- UCF `G_OFF` job `9218487` failed quickly (`FAILED`, exit `2:0`, elapsed `00:01:17`).
- All non-off gate jobs completed.
- Because `G_OFF` was intended as the ES-enabled/no-generation-gate control, compare against the previous non-ES 200-video winners with caution.

### Panda-70M 200 Anchor Gate

Matched references:
- In-series no-TTA from step/LR sweep: FVD `333.70`, FID `54.13`, PSNR `18.3676`, SSIM `0.6564`, LPIPS `0.3290`.
- Previous non-ES winner `S10_LR005`: FVD `316.34`, FID `53.59`, PSNR `18.4196`, SSIM `0.6572`, LPIPS `0.3272`.

| Run ID | Status | N | PSNR | SSIM | LPIPS | FVD | dFVD vs NOTTA | dFVD vs S10_LR005 | FID | Use | Skip | Avg scale | Avg rel anchor impr |
|--------|--------|---:|-----:|-----:|------:|----:|--------------:|------------------:|----:|----:|-----:|----------:|--------------------:|
| G_OFF | failed | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| G_LOG | complete | 200 | 18.4356 | 0.6580 | 0.3272 | 321.59 | -12.11 | +5.24 | 53.76 | 200 | 0 | 1.000 | 0.00259 |
| G_BIN_0 | complete | 200 | 18.4260 | 0.6585 | 0.3269 | 325.54 | -8.16 | +9.20 | 54.40 | 199 | 1 | 0.995 | 0.00262 |
| G_BIN_001 | complete | 200 | 18.4020 | 0.6574 | 0.3270 | 321.62 | -12.08 | +5.28 | 53.92 | 189 | 11 | 0.945 | 0.00262 |
| G_SOFT_001 | complete | 200 | 18.3493 | 0.6553 | 0.3293 | 321.80 | -11.90 | +5.46 | 53.92 | 199 | 1 | 0.260 | 0.00260 |

Takeaways:
- All completed anchor-gate variants improve FVD vs in-series no-TTA, but none beats the previous non-ES `S10_LR005` FVD.
- `G_LOG` has the best Panda pointwise metrics among completed runs and improves FVD vs no-TTA, but FVD is worse than `S10_LR005` by +5.24.
- Binary gating barely skips videos at threshold `0.0` (1/200) and only skips 11/200 at threshold `0.001`; the anchor signal is too weakly selective in this range.
- Soft gating with `soft_scale=0.01` over-damps the delta (`avg_scale=0.26`) and hurts pointwise metrics.
- Do not promote anchor gating for Panda unless the paper values pointwise gains over the stronger FVD from `S10_LR005`.

### UCF-101 200 Anchor Gate

Matched references:
- In-series no-TTA from step/LR sweep: FVD `359.80`, FID `32.70`.
- Previous balanced candidate `S5_LR0025`: FVD `353.30`, FID `32.72`.
- Previous FVD-only winner `S5_LR001`: FVD `347.09`, FID `32.78`.

Raw summary-level PSNR/SSIM/LPIPS fields remained `nan`; exporter pointwise metrics were finite. FVD/FID below are from raw summaries.

| Run ID | Status | N | FVD | dFVD vs NOTTA | dFVD vs S5_LR0025 | FID | Use | Skip | Avg scale | Avg rel anchor impr |
|--------|--------|---:|----:|--------------:|------------------:|----:|----:|-----:|----------:|--------------------:|
| G_OFF | failed | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| G_LOG | complete | 200 | 356.06 | -3.74 | +2.76 | 32.58 | 200 | 0 | 1.000 | 0.00055 |
| G_BIN_0 | complete | 200 | 363.78 | +3.97 | +10.47 | 32.56 | 192 | 8 | 0.960 | 0.00054 |
| G_BIN_001 | complete | 200 | 358.42 | -1.38 | +5.12 | 32.71 | 10 | 190 | 0.050 | 0.00054 |
| G_SOFT_001 | complete | 200 | 361.86 | +2.06 | +8.56 | 32.52 | 196 | 4 | 0.055 | 0.00055 |

Takeaways:
- `G_LOG` is the best UCF anchor-gate run by FVD, but it is worse than the previous balanced candidate `S5_LR0025` and much worse than the FVD-only `S5_LR001`.
- Threshold `0.001` is too strict for UCF: it skips 190/200 videos because the average relative anchor improvement is only about `0.00054`.
- Soft gating heavily damps the delta (`avg_scale=0.055`) and does not help FVD.
- Do not promote anchor gating for UCF based on this sweep.

Overall conclusion:
- Anchor-loss validation is useful as a diagnostic/logging signal, but the simple binary/soft gate did not improve the 200-video Pareto frontier.
- The best next promotion candidates remain Panda `S10_LR005` and UCF `S5_LR0025` (balanced) or UCF `S5_LR001` (FVD-only).
- If revisiting anchor gating, the next experiment should tune thresholds from observed relative-improvement quantiles rather than fixed `0.0`/`0.001`, and should include a successful ES-enabled `G_OFF` control.

---

## May 2026 - Active Validation, Retrieval, and Anchor-Regularization Batch

Date pasted/logged: May 23, 2026

Implementation commits:
- `759bc0e Add validation and retrieval submission batch`
- `1d7d3a1 Add anchor-regularized AdaSteer objective`

### 1000-Video Standard-Horizon Validation

Status: still running/checkpointed as of May 23.

Configs:
- `sweep_experiment/configs/panda_1000v_s10_lr005_validation.yaml`
- `sweep_experiment/configs/ucf101_1000v_s5_lr0025_validation.yaml`

SLURM/job status:
- `9424554` Panda `NOTTA`: running, checkpoint `next_idx=431`, `ok=431`.
- `9424555` Panda `S10_LR005`: running, checkpoint `next_idx=271`, `ok=271`.
- `9424556` UCF `NOTTA`: running, checkpoint `next_idx=435`, `ok=435`.
- `9424557` UCF `S5_LR0025`: running, checkpoint `next_idx=322`, `ok=322`.

No final metrics yet. Wait for summaries before judging full-scale validation.

### 200-Video Anchor-Regularized AdaSteer Objective

Purpose: test whether adding differentiable fixed-sigma heldout anchor loss to the TTA objective improves robustness over the previous 200-video winners.

Configs:
- `sweep_experiment/configs/panda_200_anchor_reg.yaml`
- `sweep_experiment/configs/ucf101_200_anchor_reg.yaml`

Reference baselines:
- Panda in-series no-TTA from step/LR sweep: FVD `333.70`, FID `54.13`, PSNR `18.3676`, SSIM `0.6564`, LPIPS `0.3290`.
- Panda previous winner `S10_LR005`: FVD `316.34`, FID `53.59`, PSNR `18.4196`, SSIM `0.6572`, LPIPS `0.3272`.
- UCF in-series no-TTA from step/LR sweep: FVD `359.80`, FID `32.70`.
- UCF balanced candidate `S5_LR0025`: FVD `353.30`, FID `32.72`.
- UCF FVD-only winner `S5_LR001`: FVD `347.09`, FID `32.78`.

Panda results:

| Run ID | Anchor reg weight | N | PSNR | SSIM | LPIPS | FVD | dFVD vs NOTTA | dFVD vs S10_LR005 | FID | Train(s) |
|--------|------------------:|---:|-----:|-----:|------:|----:|--------------:|------------------:|----:|---------:|
| AREG0 | 0.00 | 200 | 18.4435 | 0.6579 | 0.3266 | 329.36 | -4.34 | +13.01 | 53.93 | 57.5 |
| AREG005 | 0.05 | 200 | 18.3876 | 0.6575 | 0.3285 | 317.97 | -15.73 | +1.63 | 55.00 | 86.5 |
| AREG01 | 0.10 | 200 | 18.4139 | 0.6581 | 0.3276 | 316.50 | -17.20 | +0.16 | 53.75 | 85.5 |
| AREG02 | 0.20 | 200 | 18.4285 | 0.6569 | 0.3264 | 310.97 | -22.73 | -5.37 | 53.35 | 84.2 |

Panda takeaways:
- `AREG02` is a new 200-video Panda best by FVD: `333.70 -> 310.97` (-22.73 / -6.8%), beating previous `S10_LR005` by 5.37 FVD points.
- `AREG02` also improves PSNR and LPIPS vs both no-TTA and previous winner, but SSIM is slightly below previous `S10_LR005`.
- Anchor regularization costs ~84s train/video vs ~57s for unregularized 10-step AdaSteer.
- Candidate follow-up: consider 500-video or 1000-video validation of Panda `AREG02` only after current 1000-video `S10_LR005` validation completes, because compute cost is higher.

UCF results:

Raw summary-level PSNR/SSIM/LPIPS fields remained `nan`; exporter pointwise metrics were finite. FVD/FID below are from raw summaries.

| Run ID | Anchor reg weight | N | FVD | dFVD vs NOTTA | dFVD vs S5_LR0025 | dFVD vs S5_LR001 | FID | Train(s) |
|--------|------------------:|---:|----:|--------------:|------------------:|-----------------:|----:|---------:|
| AREG0 | 0.00 | 200 | 361.84 | +2.04 | +8.54 | +14.75 | 32.60 | 30.1 |
| AREG005 | 0.05 | 200 | 360.61 | +0.81 | +7.31 | +13.52 | 32.66 | 44.5 |
| AREG01 | 0.10 | 200 | 356.21 | -3.59 | +2.91 | +9.12 | 32.51 | 43.8 |
| AREG02 | 0.20 | 200 | 353.95 | -5.85 | +0.65 | +6.86 | 32.70 | 44.6 |

UCF takeaways:
- `AREG02` is the best anchor-reg UCF run, but it does not beat prior `S5_LR0025` or `S5_LR001` on FVD.
- Anchor regularization is not a UCF promotion candidate from this sweep.

### 200-Video Retrieval-Batch AdaSteer

Purpose: test retrieval-augmented shared delta quality, not independent TTA throughput.

Configs:
- `sweep_experiment/configs/panda_200_batch_retrieval_delta_a.yaml`
- `sweep_experiment/configs/ucf101_200_batch_retrieval_delta_a.yaml`

SLURM status:
- Panda `K1` job `9424558`: completed.
- Panda `K5` job `9424559`: failed quickly (`FAILED`, exit `1:0`, elapsed `00:03:29`).
- Panda `K10` job `9424560`: failed quickly (`FAILED`, exit `1:0`, elapsed `00:02:15`).
- UCF `K1` job `9424561`: completed.
- UCF `K5` job `9424562`: failed quickly (`FAILED`, exit `1:0`, elapsed `00:03:38`).
- UCF `K10` job `9424563`: failed quickly (`FAILED`, exit `1:0`, elapsed `00:02:17`).

Completed `K1` results:

| Series | Run ID | N | PSNR | SSIM | LPIPS | FVD | FID | Train(s) |
|--------|--------|---:|-----:|-----:|------:|----:|----:|---------:|
| Panda retrieval | K1 | 200 | 18.4519 | 0.6577 | 0.3269 | 319.26 | 54.58 | 58.2 |
| UCF retrieval | K1 | 200 | nan | nan | nan | 350.74 | 32.74 | 30.2 |

Retrieval-batch takeaways:
- `K1` is effectively a same-method control and completed.
- Retrieval settings `K5`/`K10` failed before producing summaries, so no retrieval quality conclusion yet.
- Need to inspect failed SLURM logs before resubmitting retrieval; likely causes include retrieval pool/path handling, memory during pool embedding, or runtime dependency issues.

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

## May 2026 - Panda-70M Long-Context 999-Video Corrected Evaluation

Date logged: May 14, 2026

Purpose: corrected full-scale long-context evaluation after the earlier partial run had a crashed No-TTA baseline and invalid per-chunk FVD averaging. This run uses proper global FVD/FID from merged sufficient statistics across chunks.

Dataset: `panda_1000_480p`, 999 successful videos

Shared config:
- `num_cond_frames=14`
- `num_frames=93` (14 conditioning + 79 generated)
- `gen_start_frame=14`
- `tta_total_frames=14`
- `tta_context_frames=14`
- `num_inference_steps=50`
- `guidance_scale=4.0`
- `resolution=480p`
- 10 chunks, 100 videos per chunk except final chunk with 99 videos

Result paths:
- `sweep_experiment/results/panda_longctx_1000v/NOTTA/merged_summary.json`
- `sweep_experiment/results/panda_longctx_1000v/ADA_S10/merged_summary.json`
- `sweep_experiment/results/panda_longctx_1000v/LORA_R8/merged_summary.json`
- `delta_experiment/results/tinylora_longctx_1000v/PANDA_TL_LAST24/merged_summary.json`

Chunk/FVD audit:
- All four methods have 10 chunk summaries and 10 `fvd_fid_stats.npz` files.
- All four methods have the same 999 unique videos, with no duplicates and no missing/extra videos versus No-TTA.
- Each method has `gen_count=999`, `ref_count=999`, `gt_cached=False`.
- Feature dimensions are consistent: I3D feature sum `(400,)`, second moment `(400, 400)`.
- FVD/FID values below are global Frechet distances from merged sufficient statistics, not averages of per-chunk FVD/FID.

| Method | Config | PSNR | SSIM | LPIPS | FVD | dFVD vs No-TTA | FID | Train(s) | Gen(s) | Total(s) | VBench aesthetic | VBench background | VBench subject |
|--------|--------|------|------|-------|-----|---------------:|-----|----------|--------|----------|------------------|-------------------|----------------|
| No-TTA | `delta_steps=0` | 12.769 | 0.4744 | 0.5469 | 278.7 | -- | 29.9 | 0.9 | 553.9 | 554.8 | 0.440 | 0.848 | 0.774 |
| AdaSteer S10 | `delta_steps=10`, `delta_lr=5e-3` | 12.787 | 0.4762 | 0.5436 | 284.1 | +5.4 | 29.5 | 18.4 | 552.9 | 571.3 | 0.440 | 0.848 | 0.775 |
| LoRA R8 | rank=8, alpha=16, all blocks, 10 steps, lr=5e-5 | 12.734 | 0.4726 | 0.5480 | 282.4 | +3.7 | 30.3 | 18.3 | 567.9 | 586.1 | 0.485 | 0.848 | 0.757 |
| TinyLoRA LAST24 | rank=2, `n_tie=1`, `qkv_proj`, last 24 blocks, 20 steps, lr=1e-3 | 12.773 | 0.4744 | 0.5468 | 278.6 | -0.1 | 30.1 | 23.0 | 562.2 | 585.2 | 0.440 | 0.848 | 0.774 |

Key findings:
- The earlier 50-video Panda long-context FVD gain did **not** hold at 999 videos.
- AdaSteer slightly improves pointwise metrics over No-TTA (PSNR +0.018, SSIM +0.0018, LPIPS -0.0033) and improves FID (29.9 -> 29.5), but worsens global FVD (278.7 -> 284.1).
- LoRA R8 is worse than No-TTA on PSNR/SSIM/LPIPS/FVD/FID, though its VBench aesthetic score is higher.
- TinyLoRA LAST24 is effectively tied with No-TTA on global FVD and pointwise metrics, with extra train/generation time.
- Interpretation: standard 28-frame Panda remains the strongest full-scale AdaSteer result; long-context Panda at 999 videos does not currently support a distributional-quality claim for AdaSteer.

---

## May 2026 - Batch Experiments Planned

Date planned: May 4, 2026

Purpose: separate two batching questions that were previously conflated.

### A. Retrieval-Augmented Batch-Level TTA

This tests whether training one shared TTA update on an eval video plus retrieved neighbours changes quality. It does NOT test GPU parallel throughput, because the current batch-level implementations cycle through one video per optimizer step.

New paper-aligned configs:

| Config | Method | Setting | K values | Status |
|--------|--------|---------|----------|--------|
| `panda_batch_retrieval_delta_a.yaml` | AdaSteer | Panda standard, 28f, gen_start=48 | 1, 5, 10 | READY |
| `panda_batch_retrieval_lora.yaml` | LoRA R8 | Panda standard, 28f, gen_start=48 | 1, 5, 10 | READY |
| `panda_longctx_batch_retrieval_delta_a.yaml` | AdaSteer | Panda long, 93f, gen_start=14 | 1, 5, 10 | READY |
| `panda_longctx_batch_retrieval_lora.yaml` | LoRA R8 | Panda long, 93f, gen_start=14 | 1, 5, 10 | READY |
| `ucf_longctx_batch_retrieval_delta_a.yaml` | AdaSteer | UCF long, 61f, gen_start=14 | 1, 5, 10 | READY |
| `ucf_longctx_batch_retrieval_lora.yaml` | LoRA R8 | UCF long, 61f, gen_start=14 | 1, 5, 10 | READY |

Submit helper: `sweep_experiment/sbatch/submit_batch_retrieval.sh` (defaults to dry-run; set `DRY_RUN=0` to submit).

### B. Batched Independent TTA Throughput

This tests the deployment claim: AdaSteer can train multiple independent per-video residuals in one batched forward/backward pass, while LoRA/TinyLoRA currently require serial independent adapters.

New benchmark script: `sweep_experiment/scripts/benchmark_batched_tta.py`

Outputs: `batched_tta_benchmark.json` with requested batch size, status/OOM, train seconds/video, total encode+train seconds/video, and peak H200 memory.

Initial profiles:

| Profile | Dataset | Frames | AdaSteer B sweep | Baselines |
|---------|---------|--------|------------------|-----------|
| panda_standard | Panda-70M | 28 | 1, 2, 4, 8, 16 | serial LoRA, serial TinyLoRA |
| panda_longctx | Panda-70M | 93 | 1, 2, 4, 8 | serial LoRA, serial TinyLoRA |
| ucf_longctx | UCF-101 | 61 | 1, 2, 4, 8 | serial LoRA, serial TinyLoRA |

Submit helper: `sweep_experiment/sbatch/submit_batched_tta_benchmark.sh` (defaults to dry-run; set `DRY_RUN=0` to submit).

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

## May 23, 2026 - Long-Horizon Failure-Mode Diagnostic (Plan)

Following the locked Related-Works framing (option A: treat long-context Panda 999v as an honest caveat), the PI requested a diagnostic pass to understand *why* AdaSteer regresses on long-context Panda before designing horizon-aware configs.

Run inputs:
- No-TTA: `sweep_experiment/results/panda_longctx_1000v/NOTTA/` (10 chunks).
- AdaSteer S10: `sweep_experiment/results/panda_longctx_1000v/ADA_S10/` (10 chunks).
- Captions: `datasets/panda_1000_480p/metadata.csv` (filename, caption, [category]).

Diagnostic script: `scripts/diagnose_long_horizon_failures.py` (stdlib only, runs on cluster). It produces:
1. `sweep_experiment/reports/long_horizon_failure_panda_1000v.csv` -- per-video PSNR/SSIM/LPIPS for both methods, deltas, caption, and a coarse theme label (sport, dance_music, cooking, nature, animal, vehicle, talking_head, crowd, indoor_misc, other).
2. `sweep_experiment/reports/long_horizon_failure_panda_1000v.txt` -- stdout summary with per-theme mean deltas, quintile buckets on No-TTA PSNR, and top-25 worst/best videos for qualitative spot-check.

Hypotheses to test:
- H1 (motion mismatch): AdaSteer hurts most on high-motion themes (sport, dance, vehicle) because the conditioning frames under-represent future motion magnitude. Predicts strongly negative dPSNR on `sport` and `vehicle`.
- H2 (scene cuts): AdaSteer hurts on talking_head/news clips that contain a cut into a new scene. Predicts negative dSSIM in talking_head despite high No-TTA PSNR.
- H3 (overfit to clean cond): AdaSteer hurts worst on the highest-PSNR quintile, i.e., it overfits to already-good conditioning frames and adds bias.
- H4 (scale-dependent regression): AdaSteer hurts most on the lowest-PSNR quintile, i.e., when conditioning information is weak the residual extrapolates poorly.

Decision flow:
- If a theme has strongly negative mean dPSNR and large N: design a theme-gated AdaSteer or theme-aware step schedule (Phase B).
- If quintile structure dominates: design a quality-conditioned step schedule (less steps for clean conditioning, more for noisy).
- If neither is clean: fall back to anchor-regularization at long horizon (already implemented; just needs cluster jobs).

Once the CSV+txt are produced, results will be pasted back here and the Phase B sweep design will be discussed before any cluster submission.


## May 23, 2026 - FVD/FID Chunked-Merge Validation

PI flagged a concern that chunked 100-video FVD/FID computation could have introduced an error producing the +5.4 long-context Panda FVD regression.

### Audit of the chunked merge math

The chunked path stores per-chunk sufficient statistics:
- `gen_sum = sum(f_i for i in chunk)` (sum of 400-D I3D features)
- `gen_cov = sum(f_i f_i^T for i in chunk)` (sum of outer products, NOT a covariance yet)
- `gen_count = N`

`sweep_experiment/scripts/merge_chunks.py:_compute_frechet_distance` computes the final FVD only at the end, from totals over all 999 videos:

  mu = sum_total / N_total
  Sigma = cov_total / N_total - outer(mu, mu)

By linearity of sums and sum-of-outer-products, this is exactly the single-pass FVD over all 999 features. There is no per-chunk averaging anywhere. The reference distribution is also chunk-shared (each chunk stores its own ref_sum/ref_cov/ref_count and these are summed identically).

### Numerical validation

Added `scripts/test_chunked_fvd_equivalence.py`. Generates synthetic 400-D (FVD) and 2048-D (FID) feature streams, computes Frechet distance two ways (single pass vs chunked sufficient-statistics merge), asserts agreement to <1e-6 relative tolerance.

Ran locally on May 23, 2026. Result: 7/7 cases passed.

  [PASS] FVD shape, 999 vs 999, chunk=100, distributions match:   rel_diff = 7.8e-15
  [PASS] FVD shape, mean shift +0.10 (typical TTA-vs-NoTTA scale): rel_diff = 1.2e-15
  [PASS] FVD shape, smaller chunks:                                 rel_diff = 7.9e-15
  [PASS] FVD shape, larger chunks:                                  rel_diff = 5.3e-15
  [PASS] FID shape (Inception-2048), no shift:                      rel_diff = 1.6e-11
  [PASS] FID shape (Inception-2048), small shift:                   rel_diff = 1.1e-13
  [PASS] Uneven chunk sizes (500 / 37 leaves remainder):            rel_diff = 0.0e+00

Largest relative error across all FVD-shape cases is ~1e-15, the float64 round-off floor. Largest across FID-shape cases is ~1.6e-11, also within numerical noise. Chunked merge is mathematically identical to single-pass.

### End-to-end production validation (cluster, pending)

`scripts/recompute_fvd_fid_from_stats.py` re-walks all `chunk_*/fvd_fid_stats.npz` under a run directory, sums the sufficient statistics independently, recomputes FVD/FID, and compares against `merged_summary.json`. Numpy + scipy only, no torch, no I3D, no GPU.

Cluster command for the four long-context Panda 999v runs:

```
cd $LONGCAT_REPO
git pull origin main
for METHOD in NOTTA ADA_S10 LORA_R8; do
  python scripts/recompute_fvd_fid_from_stats.py \
    --run-dir sweep_experiment/results/panda_longctx_1000v/$METHOD \
    | tee sweep_experiment/reports/fvd_recompute_panda_longctx_1000v_$METHOD.txt
done
python scripts/recompute_fvd_fid_from_stats.py \
  --run-dir delta_experiment/results/tinylora_longctx_1000v/PANDA_TL_LAST24 \
  | tee sweep_experiment/reports/fvd_recompute_panda_longctx_1000v_TL.txt

python scripts/recompute_fvd_fid_from_stats.py \
  --run-dir sweep_experiment/results/panda_longctx_1000v/NOTTA \
  --compare-dir sweep_experiment/results/panda_longctx_1000v/ADA_S10 \
  | tee sweep_experiment/reports/fvd_recompute_panda_longctx_pairwise.txt
```

Acceptance criterion: recomputed FVD/FID must agree with `merged_summary.json` to <1e-4 relative tolerance for every method. If any method disagrees, that disagreement (not the +5.4 FVD regression itself) becomes the next thing to debug.

### Conclusion (math)

The +5.4 FVD long-context Panda regression for AdaSteer S10 is not a chunked-merge artifact. The local unit test removes the only legitimate concern about the merge math. The cluster recomputation validates the implementation against the actual stored sufficient statistics; if it agrees with the existing numbers (expected), the regression is a real property of the generated distributions and Phase A diagnostics in `scripts/diagnose_long_horizon_failures.py` is the right next step.

## May 23, 2026 - FVD/FID Chunked-Merge Validation (Result)

Cluster ran `scripts/recompute_fvd_fid_from_stats.py` against the four long-context Panda 1000-video runs (one corrupted video per draw, 999 successfully evaluated per run as expected). All four methods passed the validation step at `rel_diff` between `3e-10` and `1.5e-8`, well below the `1e-4` acceptance criterion. The chunked sufficient-statistics merge is implementation-correct, the local unit test was correct, and the +5.4 FVD regression is **not** a numerical or merge artifact.

### Validated numbers (long-context Panda 1000v, 93 frames)

| Method               | FVD       | FID     | rel_diff vs stored (FVD / FID) |
|----------------------|----------:|--------:|--------------------------------|
| No-TTA               | 278.7059  | 29.8970 | 5.9e-10 / 1.5e-8               |
| AdaSteer S10         | 284.1372  | 29.5325 | 1.2e-9  / 2.6e-9               |
| LoRA R8              | 282.4030  | 30.3201 | 1.4e-9  / 5.0e-9               |
| TinyLoRA LAST24      | 278.6105  | 30.0854 | 3.0e-10 / 6.0e-9               |

### Pairwise deltas vs No-TTA

| Method            |  dFVD   | %FVD    |  dFID   | %FID    |
|-------------------|--------:|--------:|--------:|--------:|
| AdaSteer S10      | +5.4314 | +1.95%  | -0.3644 | -1.22%  |
| LoRA R8           | +3.6971 | +1.33%  | +0.4231 | +1.41%  |
| TinyLoRA LAST24   | -0.0954 | -0.03%  | +0.1884 | +0.63%  |

### Cross-method reading of the result

1. **The +5.4 FVD regression for AdaSteer is real.** Now confirmed against the actual stored sufficient statistics, not just against `merged_summary.json` summaries.
2. **AdaSteer S10 is the only TTA method that improves FID** while regressing FVD. FID uses Inception per-frame statistics, FVD uses I3D 3D spatiotemporal features; the FVD-up / FID-down divergence is consistent with a hypothesis that the test-time residual sharpens per-frame appearance to match the conditioning window at the cost of temporal coherence the I3D backbone reads. This is paper-useful because it makes the long-horizon caveat more refined: "AdaSteer trades distributional FVD for per-frame FID at long horizon" rather than just "FVD regressed."
3. **LoRA R8 is strictly worse than No-TTA on this setting** — it regresses both FVD (+3.70) and FID (+0.42). Reinforces the LoRA-overfits-on-single-clip-TTA narrative at the 1000-video scale, not just on 200-video sweeps.
4. **TinyLoRA LAST24 is essentially a no-op** (|dFVD| < 0.1, |dFID| < 0.2). On long-context Panda, the SVD adaptation surface neither helps nor hurts the I3D distribution. Useful contrast for the parameter-count / effect Pareto framing.

### Next step (Phase A)

The chunked-merge investigation is closed. Phase A failure diagnostics (`scripts/diagnose_long_horizon_failures.py`) is unblocked and is the next action; the goal is to localize the regression to specific caption themes / PSNR quintiles so the horizon-aware AdaSteer v2 design (Phase B) targets the actual failure modes rather than guessing. Cluster command and theme-classification notes are in `experiment_tracker/next_actions.md` under "Phase A".
