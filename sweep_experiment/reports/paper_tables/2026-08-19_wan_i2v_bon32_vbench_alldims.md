# All VBench++ quality dims — hybrid 32 windows (medians)

Paired videos: **32**. Cells are **median** (mean in the per-dim source
tables). Higher is better. Diagnostics; official number remains the
**full clip**.

**Source:** job 16009916 ·
[`2026-08-19_wan_i2v_bon32_vbench_trend.md`](2026-08-19_wan_i2v_bon32_vbench_trend.md)
(median / mean). This file is the same numbers, all 7 dims in one grid.

`dynamic_degree` median is 0 in every cell; the mean (fraction of clips
RAFT calls dynamic) is in parentheses.

## Do-nothing

| Dimension | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 25–30 − 0–5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| subject_consistency | 0.934 | 0.967 | 0.974 | 0.969 | 0.969 | 0.969 | +0.035 |
| background_consistency | 0.942 | 0.957 | 0.951 | 0.946 | 0.950 | 0.957 | +0.015 |
| aesthetic_quality | 0.651 | 0.631 | 0.607 | 0.576 | 0.555 | 0.538 | **−0.113** |
| imaging_quality | 72.87 | 74.92 | 73.06 | 72.20 | 70.05 | 68.14 | **−4.73** |
| motion_smoothness | 0.990 | 0.992 | 0.992 | 0.993 | 0.992 | 0.993 | +0.003 |
| dynamic_degree | 0 (0.250) | 0 (0.312) | 0 (0.344) | 0 (0.188) | 0 (0.281) | 0 (0.188) | 0 (−0.062) |
| temporal_flickering | 0.981 | 0.984 | 0.986 | 0.986 | 0.987 | 0.989 | +0.008 |

## Always-search

| Dimension | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 25–30 − 0–5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| subject_consistency | 0.934 | 0.976 | 0.971 | 0.960 | 0.961 | 0.968 | +0.034 |
| background_consistency | 0.943 | 0.958 | 0.947 | 0.946 | 0.946 | 0.950 | +0.007 |
| aesthetic_quality | 0.650 | 0.631 | 0.604 | 0.570 | 0.565 | 0.545 | **−0.105** |
| imaging_quality | 72.88 | 74.43 | 73.02 | 71.21 | 70.22 | 66.41 | **−6.47** |
| motion_smoothness | 0.990 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | +0.002 |
| dynamic_degree | 0 (0.250) | 0 (0.281) | 0 (0.281) | 0 (0.312) | 0 (0.312) | 0 (0.281) | 0 (+0.031) |
| temporal_flickering | 0.981 | 0.984 | 0.986 | 0.987 | 0.987 | 0.988 | +0.007 |

## Gated-search

| Dimension | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 25–30 − 0–5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| subject_consistency | 0.934 | 0.970 | 0.974 | 0.966 | 0.967 | 0.966 | +0.032 |
| background_consistency | 0.942 | 0.957 | 0.951 | 0.947 | 0.944 | 0.947 | +0.005 |
| aesthetic_quality | 0.650 | 0.626 | 0.604 | 0.558 | 0.540 | 0.520 | **−0.130** |
| imaging_quality | 72.87 | 74.72 | 73.12 | 71.63 | 69.63 | 66.07 | **−6.80** |
| motion_smoothness | 0.990 | 0.991 | 0.992 | 0.992 | 0.992 | 0.992 | +0.002 |
| dynamic_degree | 0 (0.250) | 0 (0.281) | 0 (0.281) | 0 (0.188) | 0 (0.312) | 0 (0.219) | 0 (−0.031) |
| temporal_flickering | 0.981 | 0.984 | 0.987 | 0.987 | 0.986 | 0.988 | +0.007 |

## Tail vs opening (all methods)

0–5 s is shared (piece 0). Cells at 0–5 s match across methods to 3 dp
except background always 0.943 vs 0.942 and aes 0.650 vs 0.651.

| Dimension | 0–5 (do-nothing) | 25–30 do-nothing | 25–30 always | 25–30 gated |
|---|---:|---:|---:|---:|
| subject_consistency | 0.934 | 0.969 | 0.968 | 0.966 |
| background_consistency | 0.942 | 0.957 | 0.950 | 0.947 |
| aesthetic_quality | 0.651 | 0.538 | 0.545 | 0.520 |
| imaging_quality | 72.87 | **68.14** | 66.41 | 66.07 |
| motion_smoothness | 0.990 | 0.993 | 0.992 | 0.992 |
| dynamic_degree (median / mean) | 0 / 0.250 | 0 / 0.188 | 0 / 0.281 | 0 / 0.219 |
| temporal_flickering | 0.981 | 0.989 | 0.988 | 0.988 |

Only aesthetic and imaging fall. Subject, background, smoothness, and
flicker stay flat or rise. Dynamic median is 0 from the first window.

IQ rounded to 2 dp from source 72.872 / 74.915 / 73.058 / 72.204 /
70.045 / 68.139 (do-nothing). Deltas use those source medians.
