# Caption 128 — all official metrics, four methods (2026-09-04)

Supersedes the open LPIPS/FVD cells in
`2026-08-31_wan_v2v_cite128_all_metrics.md`. Do not edit that
file. Prompt = `metadata_csv`. n=128. Visual Benchmark
(VBench) = medians except Dynamic Degree = **percent of clips**.
Pixels = paired 30 s tail vs leftover. Mean seconds = job/96.

| | Self Forcing | Rolling | Pseudo | Always |
|---|---:|---:|---:|---:|
| tail motion | 0.0119 | **0.0158** | 0.0157 | 0.0168 |
| mean s / clip (N=128) | 108 | **47** | 294 | 354 |
| Subject Consistency | 0.666 | **0.685** | 0.660 | 0.661 |
| Background Consistency | 0.801 | **0.802** | 0.792 | 0.790 |
| Aesthetic Quality | 0.499 | **0.529** | 0.510 | 0.503 |
| Imaging Quality | 72.07 | 71.52 | **72.38** | 72.19 |
| Motion Smoothness | **0.992** | 0.991 | 0.991 | 0.990 |
| Dynamic Degree | 32.8% (42) | 28.9% (37) | 47.7% (61) | **50.8% (65)** |
| Temporal Flickering | **0.987** | 0.983 | 0.984 | 0.982 |
| PSNR ↑ | **9.25** | 7.98 | 9.22 | 9.21 |
| SSIM ↑ | **0.279** | 0.250 | 0.268 | 0.266 |
| LPIPS ↓ | **0.745** | 0.762 | 0.753 | 0.751 |
| FVD ↓ (30 s tails) | 410 | 436 | **405** | 425 |
| last16 FVD ↓ | 1397 | **1108** | 1324 | 1273 |

LPIPS + FVD: job **16738784**. Table:
[`2026-09-04_wan_v2v_cite128_lpips_fvd.md`](2026-09-04_wan_v2v_cite128_lpips_fvd.md).

Headline stays VBench + Dyn%. Search ≈ Self Forcing on
PSNR / SSIM / LPIPS. Pseudo slightly wins full-tail FVD.
Rolling loses those four and wins last-16 FVD + subject +
aesthetic. Do not mix leftover ρ or schedule8 N=8 into this
grid.
