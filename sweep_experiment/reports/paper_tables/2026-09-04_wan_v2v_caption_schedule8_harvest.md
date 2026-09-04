# Caption schedule8 linger / dump harvest (2026-09-04)

Series `v2v_panda_caption_schedule_8v`. Prompt =
`prompt_source=metadata_csv` (truck hood). Existing Rolling
Forcing student. Same T=5 as the live list. **Not leftover ρ.**

Analyzer FAIL / HOLD is versus Self Forcing. Cite versus
caption Rolling Forcing first-8.

## Jobs

| Job | Method | State | Elapsed |
|---|---|---|---|
| **16855778** | `rolling_linger` | COMPLETED 0:0 | 12m 02s |
| **16855779** | `rolling_dump` | COMPLETED 0:0 | 9m 28s |
| **16855780** | Visual Benchmark (VBench) full clip | COMPLETED 0:0 | 15m 22s |

8/8 mp4 + sidecar each.

## Live list (not the paper’s linear five-step)

The checkpoint is **T=5** but **not** `[1000, 800, 600, 400, 200]`.
Native: `[1000, 952.4, 882.4, 769.2, 555.6]`. Floor is 556, not
200. We used the endpoint-preserving warp:

| Arm | used |
|---|---|
| linger-high | 1000, 972.2, 888.9, 750.0, 555.6 |
| dump-early | 1000, 777.8, 685.7, 615.1, 555.6 |

## Tails vs caption Rolling Forcing first-8

Host median tail = **0.0134** (last Rolling column, videos
0000–0007). Leftover harvest wrote 0.0128 for the same host
slice — do not edit that file; letters here do not flip.

| Method | tail med | vs host | W/L/T | Call |
|---|---:|---:|---|---|
| Rolling Forcing host (first 8) | 0.0134 | — | — | host |
| `rolling_linger` | 0.0121 | **−10%** | 3/5/0 | no motion |
| `rolling_dump` | 0.0187 | **+39%** | 7/1/0 | tail yes |

## Official Visual Benchmark (VBench) Quality Score (N=8)

Dynamic Degree: linger median 0 → **0/8**. Dump median 0.5 →
**4/8**. Other dims = median. Rolling Forcing N=32 Imaging
Quality 70.22 / Subject Consistency 0.694 is **not** paired N=8.

| Method | Subject Consistency | Background Consistency | Aesthetic Quality | Imaging Quality | Motion Smoothness | Dynamic Degree | Temporal Flickering |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rolling_linger` | 0.668 | 0.812 | 0.583 | **66.34** | 0.993 | **0/8** | 0.987 |
| `rolling_dump` | 0.658 | 0.801 | 0.563 | **68.14** | 0.990 | **4/8** | 0.981 |

Vs Rolling Forcing N=32 (70.22 / 0.694): linger Imaging
Quality **−3.88**, Subject **−0.026**. Dump Imaging Quality
**−2.08**, Subject **−0.036**. Aesthetic Quality went **up**
while Imaging Quality fell (same paint signature as leftover ρ).

## Letter

Both **NO**. Linger does not move the tail and kills Imaging
Quality. Dump moves pixels and still kills Imaging Quality /
Subject Consistency.

**Distribution Matching Distillation (DMD):** the
inference-only list failed the quality bar. That is the
evidence that a new schedule needs a student — or we keep
the native list. Do not scale N=8. Do not start 8-GPU DMD
tonight. Paper lock stays test-time. Do not remake cite-128.
