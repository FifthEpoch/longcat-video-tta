# Caption 128 official row — Always in (2026-08-31)

VBench **16674378** COMPLETED 0:0 in 1h22. Skip-existing resume
after **16615750** preemption. Do not remake videos.

Cite Dyn as **percent of clips**, not the median. All four
`joined.json` have dynamic **median 0**.

**Series:** `v2v_panda_caption_128v`. Prompt = `metadata_csv`.

| Method | tail | subject | IQ | Dyn% | flicker |
|---|---:|---:|---:|---:|---:|
| Self Forcing | 0.0119 | 0.666 | 72.07 | 32.8% (42/128) | 0.987 |
| Rolling Forcing | 0.0158 | **0.685** | 71.52 | 28.9% (37/128) | 0.983 |
| Pseudo-future Search | 0.0157 | 0.660 | **72.38** | 47.7% (61/128) | 0.984 |
| Always-search | **0.0168** | 0.661 | 72.19 | **50.8% (65/128)** | 0.982 |

`0.5078125 × 128 = 65`.

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
Headline stays VBench + Dyn%.
