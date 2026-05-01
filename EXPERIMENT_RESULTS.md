# TTA Evaluation Results (Apr 16 - May 1, 2026)

## Methods Under Evaluation

| Method | Description | Key Hyperparams |
|---|---|---|
| **No-TTA** | Baseline, direct generation from conditioning frames | `delta_steps=0` |
| **AdaSteer (S10)** | Delta Vector TTA | 10 steps, LR=5e-3 |
| **LoRA R8** | Rank-8 LoRA adaptation | 10 steps, LR=5e-5, alpha=16, all blocks, warmup=3 |
| **TinyLoRA (LAST24)** | SVD-compressed LoRA, partial blocks | rank=2, n_tie=1, last 24 blocks, 20 steps, LR=1e-3 |

## Shared Config (Long-Context)

- **Conditioning frames:** 14 (frames 0-13)
- **Generation start:** frame 14
- **TTA training window:** frames 0-13 (no leakage)
- **Inference steps:** 50, **Guidance scale:** 4.0, **Resolution:** 480p (832x480)

---

## Standard Horizon: Panda-70M (28 frames, 1000 videos)

| Method | PSNR | SSIM | LPIPS | FVD | FID |
|---|---|---|---|---|---|
| No-TTA | 18.134 | 0.6601 | 0.3282 | 150.09 | 23.89 |
| AdaSteer (DV_BARE) | 18.119 | 0.6592 | 0.3302 | **142.32** (-5.2%) | 24.61 |

---

## Long Context: UCF-101 (61 frames = 14 cond + 47 gen, 50 videos)

| Method | PSNR | SSIM | LPIPS | FVD | FID | TTA | Gen | Total |
|---|---|---|---|---|---|---|---|---|
| No-TTA | 17.606 | 0.6744 | 0.3168 | 1336.7 | 53.10 | 0.9s | 276.1s | 277.1s |
| AdaSteer S10 | **17.719** | **0.6806** | **0.3122** | **1275.5** (-4.6%) | **51.74** | 18.4s | 277.8s | 296.2s |
| LoRA R8 | 17.613 | 0.6731 | 0.3169 | 1295.5 (-3.1%) | 52.60 | 18.2s | 290.3s | 308.5s |

---

## Long Context: Panda-70M (93 frames = 14 cond + 79 gen, 50 videos)

| Method | PSNR | SSIM | LPIPS | FVD | FID | TTA | Gen | Total |
|---|---|---|---|---|---|---|---|---|
| No-TTA | 14.090 | 0.5245 | 0.4920 | 1378.1 | 127.9 | 0.9s | 549.0s | 549.9s |
| AdaSteer S10 | **14.108** | **0.5255** | 0.4929 | **1292.1** (-6.2%) | **123.3** | 18.3s | 553.7s | 572.0s |
| LoRA R8 | 14.059 | 0.5231 | 0.4931 | 1359.0 (-1.4%) | 129.5 | 18.2s | 569.3s | 587.5s |

---

## Cross-Horizon: AdaSteer FVD Improvement

| Setting | Frames | Videos | No-TTA FVD | AdaSteer FVD | Delta | % |
|---|---|---|---|---|---|---|
| Standard (Panda) | 28 | 999 | 150.09 | 142.32 | -7.8 | **-5.2%** |
| Long (UCF) | 61 | 50 | 1336.7 | 1275.5 | -61.2 | **-4.6%** |
| Long (Panda) | 93 | 50 | 1378.1 | 1292.1 | -86.0 | **-6.2%** |

AdaSteer FVD improvement is consistent (4.6-6.2%) across horizons and datasets. Absolute FVD gap grows with horizon (-7.8 at 28f vs -86.0 at 93f).

---

## 1000-Video Panda-70M Long Context (93f) -- Old Run, Partial

> These are pre-audit results (NOTTA crashed, FVD is per-chunk avg not global). Corrected run in progress.

| Method | Videos | PSNR | SSIM | LPIPS | FVD (chunk avg +/- std) | FID (chunk avg) |
|---|---|---|---|---|---|---|
| No-TTA | 0/1000 | -- | -- | -- | -- | -- |
| AdaSteer S10 | 999/1000 | 12.776 | 0.4758 | 0.5447 | 964.8 +/- 143.3 | 99.6 +/- 4.2 |
| LoRA R8 | 899/1000 | 12.756 | 0.4713 | 0.5487 | 953.2 +/- 88.9 | 100.8 +/- 4.1 |
| TinyLoRA LAST24 | 600/1000 | 12.799 | -- | -- | -- | -- |

---

## Current Status (May 1, 2026)

1000-video Panda-70M resubmitted with all pipeline fixes (4 methods x 10 chunks = 40 jobs). Awaiting results ~May 2.
