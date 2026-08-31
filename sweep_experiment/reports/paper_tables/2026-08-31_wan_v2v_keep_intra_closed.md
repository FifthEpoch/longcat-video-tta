# Keep-picture + in-chunk: family closed (2026-08-31)

Caption N=8. Cite vs caption Self Forcing first 8
(subject **0.700** / IQ **71.54** / tail **0.0129**).
Pre-registered keep letter: subject ≥ **0.68**, IQ ≥ **70.5**,
and tail or Dyn% beats Self Forcing. Do not retune.

Analyzer `PROMOTE` is vs that Self Forcing row. Every keep /
intra / denoise arm that moved the tail **failed subject**.

## Call

**NO. Mid-chunk rewrite is closed.** Keep-picture was the last
chance (10% nudge, residual, next-seed, latent-travel pick +
first-latent lock). All 14 arms miss subject 0.68. Rolling
keep also drops IQ to 66–67. Intra gated ≡ always (gate dead).
Denoise already NO; restep is n=5 and subject 0.575.

Do not scale. Do not remake `sf_restep`. Do not loosen 0.68 /
70.5.

## Keep-picture (8/8, official scores on disk)

| Method | tail | subject | IQ | flicker | Call |
|---|---:|---:|---:|---:|---|
| Self Forcing (same 8) | 0.0129 | **0.700** | **71.54** | 0.989 | bar |
| sf_nudge / always | 0.0145 / 0.0134 | 0.642 / 0.644 | 69.66 / 70.87 | 0.986 / 0.987 | **NO** |
| sf_nextseed / always | 0.0139 / 0.0130 | 0.629 / 0.640 | 70.11 / 69.81 | 0.987 / 0.987 | **NO** |
| sf_wiggle / always | 0.0143 / 0.0114 | 0.633 / 0.646 | 70.65 / 68.11 | 0.986 / 0.989 | **NO** |
| sf_latmot / always | 0.0193 / 0.0200 | 0.640 / 0.627 | 70.03 / 69.75 | 0.982 / 0.981 | **NO** (hot + twitch) |
| rf_nudge / always | 0.0134 / 0.0129 | 0.659 / 0.657 | 67.10 / 66.61 | 0.986 / 0.987 | **NO** |
| rf_wiggle / always | 0.0135 / 0.0138 | 0.659 / 0.659 | 66.73 / 67.17 | 0.986 / 0.986 | **NO** |
| rf_latmot / always | 0.0142 / 0.0168 | 0.658 / 0.656 | 66.85 / 65.78 | 0.986 / 0.983 | **NO** |

`sf_latmot_always` Dyn median 1 and flicker 0.981 is the old
twitch neighborhood, not living motion. `sf_wiggle_always`
lost the tail.

## Intra (8/8, job 743)

| Method | tail | subject | IQ | Dyn med | flicker | Call |
|---|---:|---:|---:|---:|---:|---|
| sf_intra / always | 0.0207 / 0.0207 | 0.632 / 0.632 | 68.19 / 68.45 | 1 | 0.981 | **NO.** Gate ≡ always |
| rf_intra / always | 0.0169 / 0.0169 | 0.645 / 0.645 | 66.33 / 66.33 | 1 | 0.983 | **NO.** Gate ≡ always |

CPU-snap rerun wrote the Self Forcing videos that OOM’d before.
Same identity tax. Experiment paragraph, not a method.

## Denoise (lastmix / bpseudo n=8; restep n=5)

Unchanged **NO**. `rf_bpseudo` now exists: subject 0.655 / IQ
**64.98**. `sf_restep` n=5 subject **0.575** — do not finish
the three missing clips.

## Why (for the paper paragraph)

Four denoising steps means a “small” rewrite is a new picture.
A 10% blend and a first-latent lock still moved who the clip
is (subject −0.05 to −0.07). The sick gate did not spare
healthy blocks (intra gated ≡ always). Rolling’s last three
frames are not a safer edit surface — IQ fell further.

Jobs: keep SF VBench **16620372**; RF scores already on disk
after **16616188** failed 2:0 on a missing dir. Intra **743**.
Resume **16674379** is skip-existing, leave it.
