# Caption 128 official row — Always in (2026-08-31)

VBench **16674378** COMPLETED 0:0 in 1h22. Skip-existing resume
after **16615750** preemption. Do not remake videos.

Cite Dyn as **percent of clips**, not the median. All four
`joined.json` have dynamic **median 0**.

**Series:** `v2v_panda_caption_128v`. Prompt = `metadata_csv`.

| Method | tail | subject | IQ | Dyn% | flicker | mean s / clip |
|---|---:|---:|---:|---:|---:|---:|
| Self Forcing | 0.0119 | 0.666 | 72.07 | 32.8% (42/128) | 0.987 | 196 |
| Rolling Forcing | 0.0158 | **0.685** | 71.52 | 28.9% (37/128) | 0.983 | **45** |
| Pseudo-future Search | 0.0157 | 0.660 | **72.38** | 47.7% (61/128) | 0.984 | **304** |
| Always-search | **0.0168** | 0.661 | 72.19 | **50.8% (65/128)** | 0.982 | **348** |

`0.5078125 × 128 = 65`.

Quality is caption-128 official. **Mean seconds / clip** is
sidecar `seconds` on caption-32 (n=32), the locked cost sample.
Cite the **mean**, not the median: Pseudo’s median (357 s) is a
fired clip; the 9 skips sit at ~113 s and pull the mean down.
Always is flat 348 s (k=4 every chunk). Source:
[`2026-08-25_wan_v2v_caption_wall_time_mean.md`](2026-08-25_wan_v2v_caption_wall_time_mean.md).

## Average time per clip (Pseudo vs Always)

| Method | n | mean s | median s | vs Always mean |
|---|---:|---:|---:|---|
| Self Forcing | 32 | 196.1 | 113.1 | 0.56× |
| Rolling Forcing | 32 | 44.7 | 44.6 | 0.13× |
| **Pseudo-future Search** | 32 | **303.6** | 357.0 | **0.87×** |
| **Always-search** | 32 | **348.1** | 348.1 | — |

Pseudo is **44 s cheaper on average** than Always (~13%). That
is the 38-skip / 9-skip gate. It is still **~6.8× Rolling**.

Caption-128 generate wall (first 32 hardlinked, ~96 new clips):
Pseudo **16615748** 7h51 ≈ **294 s / new clip**; Always
**16615749** 9h26 ≈ **354 s / new clip**. Same sign. Do not
treat job-wall / 128 as the mean (the copies are free).

## Gate vs pick

Always − Pseudo = **+4 dynamic clips** (+3.1 pp). Subject
**0.661 vs 0.660**. IQ **72.19 vs 72.38**. Same pattern as
caption N=32 (Always 43.8% / 14 vs Pseudo 40.6% / 13).

The once-on-opening gate (90 fire / 38 skip) is the **cost cut**.
It is almost free on official quality. Do not loosen γ. Do not
retune k on this 128.

vs Self Forcing: Pseudo still the cite row (tail +32%, Dyn%
+15 pp, subject holds, IQ up). Always is the ablation, not the
method.

vs Rolling: Pseudo still better Dyn% and IQ, tied tail, worse
subject, much more expensive. Always does not close the cost
gap.

## Pixel

**16678705** CANCELLED by 0 at 2h20 (preemption). Only Self
Forcing `pixel_full/summary.json` landed: n=128 PSNR **9.25**
SSIM **0.279** LPIPS **None** (`lpips` missing in env). Do not
cite that single row as a method compare. Resubmit skip-existing.
Headline stays VBench + Dyn%. Full VBench++ × pixel × FVD
grid (with holes):
[`2026-08-31_wan_v2v_cite128_all_metrics.md`](2026-08-31_wan_v2v_cite128_all_metrics.md).
