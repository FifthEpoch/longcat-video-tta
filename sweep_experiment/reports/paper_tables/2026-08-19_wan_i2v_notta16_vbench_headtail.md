# Wan I2V 5 s vs 30 s VBench++ — `i2v_notta_16v`

Paired videos: **16**. Higher is better. `first1` / `last1` are **16-frame diagnostics** (skip cond frame 0), the same windows as `score_i2v_drift.py`. They are not official VBench++. Cite **5 s full** vs **30 s first5** for a same-duration quality pair. 5 s and 30 s were **separate generates** (same images/seed); first1 is not a shared prefix.

**Source:** job 16010032, `analyze_i2v_vbench_horizon.py`, 2026-08-19.

## Head vs tail (16 frames)

| Dimension | 5 s first16 | 5 s last16 | 5 s Δrel | 30 s first16 | 30 s last16 | 30 s Δrel |
|---|---:|---:|---:|---:|---:|---:|
| subject_consistency | 0.951 / 0.948 | 0.992 / 0.988 | 0.043 | 0.955 / 0.948 | 0.992 / 0.984 | 0.038 |
| background_consistency | 0.943 / 0.934 | 0.964 / 0.964 | 0.023 | 0.941 / 0.941 | 0.971 / 0.968 | 0.032 |
| aesthetic_quality | 0.598 / 0.610 | 0.609 / 0.616 | 0.018 | 0.602 / 0.608 | 0.532 / 0.524 | -0.115 |
| imaging_quality | 72.742 / 69.995 | 73.837 / 72.667 | 0.015 | 72.357 / 70.654 | 70.679 / 70.226 | -0.023 |
| motion_smoothness | 0.984 / 0.984 | 0.992 / 0.992 | 0.008 | 0.985 / 0.983 | 0.992 / 0.990 | 0.007 |
| dynamic_degree | 0.000 / 0.125 | 0.000 / 0.125 | — | 0.000 / 0.250 | 0.000 / 0.188 | — |
| temporal_flickering | 0.978 / 0.976 | 0.980 / 0.981 | 0.003 | 0.977 / 0.975 | 0.990 / 0.988 | 0.013 |

Δrel = (last16 − first16) / first16 on **medians**. Same formula as the handpicked drift table. Sign meaning differs: VBench higher is better, so negative Δrel is a quality drop. Handpicked sharpness *up* is oversharpening (bad), not a VBench win.

## Same-duration 5 s window

| Dimension | 5 s full | 30 s first5 | 30 s − 5 s (med) |
|---|---:|---:|---:|
| subject_consistency | 0.932 / 0.928 | 0.931 / 0.926 | -0.001 |
| background_consistency | 0.948 / 0.943 | 0.935 / 0.933 | -0.013 |
| aesthetic_quality | 0.626 / 0.624 | 0.605 / 0.610 | -0.021 |
| imaging_quality | 73.362 / 72.002 | 71.931 / 72.714 | -1.431 |
| motion_smoothness | 0.991 / 0.991 | 0.991 / 0.990 | -0.000 |
| dynamic_degree | 0.000 / 0.250 | 0.000 / 0.250 | 0.000 |
| temporal_flickering | 0.979 / 0.980 | 0.979 / 0.979 | -0.000 |

## Full clip (unequal length; not a duration-matched pair)

| Dimension | 5 s full (~85 fr) | 30 s full (~481 fr) |
|---|---:|---:|
| subject_consistency | 0.932 / 0.928 | 0.842 / 0.849 |
| background_consistency | 0.948 / 0.943 | 0.903 / 0.900 |
| aesthetic_quality | 0.626 / 0.624 | 0.583 / 0.572 |
| imaging_quality | 73.362 / 72.002 | 72.299 / 72.680 |
| motion_smoothness | 0.991 / 0.991 | 0.992 / 0.991 |
| dynamic_degree | 0.000 / 0.250 | 0.000 / 0.438 |
| temporal_flickering | 0.979 / 0.980 | 0.985 / 0.983 |

## How to read

- Official comparable VBench++ on these 16 is **5 s full**. 30 s full is a long-clip diagnostic, not VBench-I2V's 5 s recipe.
- If 16-frame Δrel at 30 s is much worse than at 5 s, VBench sees the same horizon stress the handpicked table reported (sharp +167% / motion −60%).
- `dynamic_degree` is 0/1 RAFT. Means are the fraction of clips called dynamic; medians are often 0.
- Do not invent PSNR. These stills have no paired 30 s GT.
