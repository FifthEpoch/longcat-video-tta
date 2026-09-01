# Caption 128 — all official metrics, four methods (2026-08-31)

**Series:** `v2v_panda_caption_128v`. Prompt = `metadata_csv`.
n=128. VBench from `joined.json` dump 21:06. Cite **medians**
except Dynamic Degree = **percent of clips**. PSNR/SSIM =
paired 30 s tail vs real leftover (median). Headline stays
VBench + Dyn%.

| | Self Forcing | Rolling | Pseudo | Always |
|---|---:|---:|---:|---:|
| tail motion | 0.0119 | **0.0158** | 0.0157 | 0.0168 |
| mean s / clip (N=32) | 196 | **45** | 304 | 348 |
| subject_consistency | 0.666 | **0.685** | 0.660 | 0.661 |
| background_consistency | 0.801 | **0.802** | 0.792 | 0.790 |
| aesthetic_quality | 0.499 | **0.529** | 0.510 | 0.503 |
| imaging_quality | 72.07 | 71.52 | **72.38** | 72.19 |
| motion_smoothness | **0.992** | 0.991 | 0.991 | 0.990 |
| dynamic_degree % | 32.8 (42) | 28.9 (37) | 47.7 (61) | **50.8 (65)** |
| temporal_flickering | **0.987** | 0.983 | 0.984 | 0.982 |
| PSNR | **9.25** | 7.98 | 9.22 | — |
| SSIM | **0.279** | 0.250 | 0.268 | — |
| LPIPS | — | — | — | — |
| FVD (aligned tails) | — | — | — | — |

Raw medians: SF 0.66633 / 0.80110 / 0.49880 / 72.074 / 0.99209 /
0.328125 / 0.98749. RF 0.68504 / 0.80180 / 0.52911 / 71.524 /
0.99099 / 0.28906 / 0.98339. Pseudo 0.65976 / 0.79244 / 0.51013
/ 72.377 / 0.99075 / 0.47656 / 0.98374. Always 0.66083 /
0.79048 / 0.50270 / 72.190 / 0.99009 / 0.50781 / 0.98221.

## Read

VBench++ is **complete** for all four. Background is a host
tie (0.801 / 0.802); Pseudo / Always are **−0.01**. Aesthetic
is Rolling’s win (0.529); Pseudo 0.510 sits between the hosts;
Always 0.503 ≈ Self Forcing. Smoothness is a four-way tie at
0.990–0.992 (Always lowest by 0.002 — more living motion).
Flicker same story.

None of the new dims move the cite: Pseudo vs Self Forcing
still wins Dyn% and IQ and holds subject; vs Rolling still
wins Dyn% and IQ, ties the tail, loses subject and aesthetic.
Always ≈ Pseudo on every imaging dim and +4 Dyn clips.

## Pixels (partial, 23:12)

**16694796** wrote Rolling and Pseudo. Always `summary.json`
still **MISSING**. Do not cite a four-way.

Medians n=128, paired 30 s tail vs real leftover:

| | PSNR | SSIM |
|---|---:|---:|
| Self Forcing | **9.25** | **0.279** |
| Rolling | 7.98 | 0.250 |
| Pseudo | 9.22 | 0.268 |
| Always | — | — |

Pseudo ≈ Self Forcing on reconstruction. Rolling is **worse**
(−1.3 dB / −0.03 SSIM) while winning subject and the tail on
VBench. That is the LongCat lesson: more invented motion leaves
the literal leftover. Headline stays VBench + Dyn%.

## Still missing

- **Always PSNR/SSIM:** **16694796** CANCELLED by 0 at 2h20
  (same slot as 705). 50 per-video jsons on disk. Resubmit
  `submit_v2v_pixel128.sh` — skip-existing finishes Always only.
- **LPIPS:** `lpips` missing in env.
- **FVD:** not run. Aligned tails only, `--force`.
