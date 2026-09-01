# Caption leftover ρ / look harvest (2026-09-01)

Series `v2v_panda_caption_leftovers_8v`. Prompt =
`prompt_source=metadata_csv` (truck hood, not `panda 0000`).
**Do not mix into** `2026-08-22_wan_v2v_leftovers8_verdict.md`
(stem / T5-panda). Do not remake cite-128.

Analyzer `FAIL` / `HOLD` in the bake-off script is vs Self
Forcing (`notta`). The leftover question is vs caption Rolling
Forcing first-8. This file cites the host.

## Jobs

| Job | Method | State | Elapsed |
|---|---|---|---|
| **16734909** | `rolling_rho_lo` | COMPLETED 0:0 | 10m 47s |
| **16734910** | `rolling_rho_hi` | COMPLETED 0:0 | 10m 03s |
| **16734911** | `rolling_adapt` | COMPLETED 0:0 | 9m 00s |
| **16734912** | `rolling_look` | COMPLETED 0:0 | 45m 52s |
| **16734913** | Visual Benchmark (VBench) full clip | COMPLETED 0:0 | 31m 16s |

8/8 mp4 + 8 sidecar each. First sidecar prompt: “A close up of
a truck with its hood open.”

## Tails vs caption Rolling Forcing first-8

Host column is `v2v_panda_caption_32v/rolling_notta` videos
0000–0007 (pair script last Rolling column). Median tail
motion.

| Method | tail med | vs host | W/L/T | Call |
|---|---:|---:|---|---|
| Rolling Forcing host (first 8) | 0.0128 | — | — | host |
| Self Forcing first 8 | 0.0129 | +1% | — | this 8-slice is not the N=32 +22% |
| `rolling_rho_lo` | 0.0206 | **+61%** | 7/1/0 | tail yes |
| `rolling_rho_hi` | 0.0252 | **+97%** | 8/0/0 | tail yes |
| `rolling_adapt` | 0.0252 | **+97%** | 6/0/2 | copies hi on most clips |
| `rolling_look` | 0.0119 | **−7%** | 2/6/0 | no motion gain |

`rolling_adapt` still matches the stem rule: 0001 / 0006 =
host, 0007 = `rho_lo`, the rest = `rho_hi`.

## Official Visual Benchmark (VBench) Quality Score (N=8 leftover)

Dynamic Degree = **0/8** clips (median 0). Other dims = median.
Self Forcing / Rolling Forcing N=32 rows are **not** paired N=8
— quality letter uses the leftover N=8 numbers against the
locked N=8 bars (Imaging Quality not worse by ≥1.0, Subject
Consistency not worse by ≥0.02 vs the relevant host).

| Method | Subject Consistency | Background Consistency | Aesthetic Quality | Imaging Quality | Motion Smoothness | Dynamic Degree | Temporal Flickering |
|---|---:|---:|---:|---:|---:|---:|---:|
| Self Forcing N=32 (baseline dir) | 0.700 | 0.839 | 0.502 | 71.54 | 0.992 | 0 | 0.989 |
| Rolling Forcing N=32 (WAVE=1) | 0.694 | — | — | 70.22 | — | 0 | 0.985 |
| `rolling_rho_lo` N=8 | 0.653 | 0.813 | 0.545 | **68.09** | 0.990 | **0/8** | 0.980 |
| `rolling_rho_hi` N=8 | 0.630 | 0.809 | 0.555 | **64.44** | 0.986 | **0/8** | 0.976 |
| `rolling_adapt` N=8 | 0.630 | 0.809 | 0.537 | **67.47** | 0.986 | **0/8** | 0.976 |
| `rolling_look` N=8 | 0.666 | 0.800 | 0.550 | **69.51** | 0.993 | **0/8** | 0.987 |

Vs Rolling Forcing N=32 Imaging Quality 70.22 / Subject 0.694:

| Method | Δ Imaging Quality | Δ Subject Consistency | Letter |
|---|---:|---:|---|
| `rolling_rho_lo` | **−2.13** | **−0.041** | **NO** |
| `rolling_rho_hi` | **−5.78** | **−0.064** | **NO** |
| `rolling_adapt` | **−2.75** | **−0.064** | **NO** |
| `rolling_look` | −0.71 | **−0.028** | **NO** (tail also lost) |

A first-8 Rolling Forcing Visual Benchmark (VBench) subset
would not flip `rho_*` (Imaging Quality 64–68). `look` already
loses tail and Subject Consistency.

## Vs stem leftover (audit only)

Stem tails were panda-infected. Caption replay asked whether ρ
still kills Imaging Quality when T5 hears the scene.

| Method | Stem vs host | Caption vs host | Caption letter |
|---|---|---|---|
| `rolling_rho_lo` | IQ −1.66 **NO** | IQ −2.13 **NO** | **NO** |
| `rolling_rho_hi` | tail +40%, IQ −3.77 **NO** | tail +97%, IQ −5.78 **NO** | **NO** |
| `rolling_adapt` | IQ −1.39 **NO** | IQ −2.75 **NO** | **NO** |
| `rolling_look` | HOLD n=8 only | tail −7%, subject −0.028 | **NO** |

The knob still moves pixels. Real captions did **not** save
Imaging Quality. Aesthetic Quality went **up** while Imaging
Quality fell (paint / morph, not “more scene motion”).

## Closed

All four leftover knobs **NO**. Do not scale. Keep the native
Rolling Forcing noise list. Open Rolling-shaped lever stays
window-exit (context noise / next-block bump / FIFO lookahead),
not global ρ.
