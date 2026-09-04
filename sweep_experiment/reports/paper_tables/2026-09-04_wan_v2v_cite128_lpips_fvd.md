# Caption 128 LPIPS + aligned-tail FVD — DONE (2026-09-04)

**Series:** `v2v_panda_caption_128v`. Prompt = `metadata_csv`.
n=128. Job **16738784** COMPLETED 0:0 1h20 (L40S).
**16737041** CANCELLED (second torch). Do not remake videos.

Learned Perceptual Image Patch Similarity (LPIPS): AlexNet,
every 8th aligned frame + last. Lower is closer to leftover.
Fréchet Video Distance (FVD): I3D on **aligned 30 s tails**
only. `fvd_last16` = last 16-frame window only.

| Method | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | last16 FVD ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Self Forcing | 128 | **9.25** | **0.279** | **0.745** | 410.3 | 1397 |
| Rolling Forcing | 128 | 7.98 | 0.250 | 0.762 | 436.3 | **1108** |
| Pseudo-future Search | 128 | 9.22 | 0.268 | 0.753 | **405.4** | 1324 |
| Always-search | 128 | 9.21 | 0.266 | 0.751 | 425.2 | 1273 |

Raw LPIPS: 0.745452 / 0.762388 / 0.753015 / 0.750944.
Raw FVD: 410.312 / 436.338 / 405.380 / 425.244.
Raw last16: 1397.127 / 1107.522 / 1324.204 / 1273.244.

## Read

Search ≈ Self Forcing on reconstruction: LPIPS +0.006 / +0.008.
Pseudo is the best full-tail FVD (**405** vs Self Forcing 410).
Always pays +15 FVD for +4 Dynamic Degree clips.

Rolling Forcing is worse on PSNR / SSIM / LPIPS / full-tail
FVD, and **better** on last-16 FVD (1108 vs 1397). Do not
promote last-16 over the 30 s tail. Headline stays full-clip
Visual Benchmark (VBench) + Dynamic Degree percent.

Pixel suite is **complete**. Grid:
[`2026-09-04_wan_v2v_cite128_all_metrics.md`](2026-09-04_wan_v2v_cite128_all_metrics.md).
