# Caption FIFO + lock-score harvest (2026-09-04)

Series `v2v_panda_caption_fifo_tscore_8v`. Prompt =
`prompt_source=metadata_csv` (truck hood). FIFO is Kim et al.
on the Rolling host. Lock-score is the **1.3B student** as a
freeze-score, not Wan-14B.

Analyzer FAIL / HOLD is versus Self Forcing. Cite Rolling arms
versus caption Rolling first-8. Cite Self Forcing arms versus
caption Self Forcing first-8. Quality letter versus caption-32
N=32 hosts: Rolling **0.694 / 70.22**, Self Forcing **0.700 /
71.54**.

## Jobs

| Job | Method | State | Elapsed |
|---|---|---|---|
| **16931441** | `rolling_fifo` | COMPLETED 0:0 | 12m 01s |
| **16931442** | `rolling_fifo_sick` | COMPLETED 0:0 | 12m 05s |
| **16931443** | `rf_tscore` | COMPLETED 0:0 | 16m 03s |
| **16931444** | `rf_tscore_always` | COMPLETED 0:0 | 25m 16s |
| **16931445** | `sf_tscore` | COMPLETED 0:0 | 21m 23s |
| **16931446** | `sf_tscore_always` | COMPLETED 0:0 | 33m 29s |
| **16931447** | VBench full clip | COMPLETED 0:0 | 52m 51s |

8/8 mp4 + sidecar each. `fifo_n` / `rewind_logs` live under
`chunk_logs` (top-level keys were empty in the naive print).

## Tails vs matching first-8 host

Rolling first-8 median **0.0134**. Self Forcing first-8 **0.0129**.

| Method | Host | tail med | vs host | W/L/T | Call |
|---|---|---:|---:|---|---|
| Rolling first-8 | — | 0.0134 | — | — | host |
| Self Forcing first-8 | — | 0.0129 | — | — | host |
| `rolling_fifo` | Rolling | 0.0162 | **+21%** | 6/2/0 | tail yes |
| `rolling_fifo_sick` | Rolling | 0.0132 | **−1%** | 1/5/2 | ≈ host |
| `rf_tscore` | Rolling | 0.0134 | **0%** | 0/0/8 | **identity** |
| `rf_tscore_always` | Rolling | 0.0149 | **+11%** | 3/5/0 | weak / loses clips |
| `sf_tscore` | Self Forcing | 0.0129 | **0%** | 0/0/8 | **identity** |
| `sf_tscore_always` | Self Forcing | 0.0120 | **−7%** | 1/6/1 | worse tail |

`rf_tscore` matches the Rolling host on every clip. `sf_tscore`
matches Self Forcing on every clip (same last-chunk drift
102.09). The 1.2× worse-than-previous-span gate **never
redrew**. Always-on draws a second seed; Rolling keeps a
slightly hotter pick and still loses 5/8; Self Forcing gets
colder.

## Official VBench (N=8)

Dynamic Degree median 0 → **0/8** every arm.

| Method | Subject | Background | Aesthetic | Imaging | Smooth | Dyn | Flicker |
|---|---:|---:|---:|---:|---:|---:|---:|
| Self Forcing N=32 | 0.700 | 0.839 | 0.502 | 71.54 | 0.992 | 0 | 0.989 |
| Rolling N=32 | 0.694 | — | — | 70.22 | — | 0 | 0.985 |
| `rolling_fifo` | 0.659 | 0.824 | 0.556 | **68.23** | 0.992 | 0/8 | 0.983 |
| `rolling_fifo_sick` | 0.658 | 0.815 | 0.546 | **66.75** | 0.993 | 0/8 | 0.987 |
| `rf_tscore` | 0.658 | 0.812 | 0.554 | **66.70** | 0.993 | 0/8 | 0.986 |
| `rf_tscore_always` | 0.665 | 0.809 | 0.558 | **65.90** | 0.993 | 0/8 | 0.985 |
| `sf_tscore` | 0.658 | 0.820 | 0.560 | 70.62 | 0.993 | 0/8 | 0.987 |
| `sf_tscore_always` | 0.653 | 0.822 | 0.546 | 70.64 | 0.993 | 0/8 | 0.988 |

`rf_tscore` identity ⇒ first-8 Rolling VBench is IQ **66.70** /
subject 0.658. `sf_tscore` identity ⇒ first-8 Self Forcing
VBench is IQ **70.62** / subject 0.658. Use those as the 8-slice
read, not as a new N=32 host.

Vs N=32 host (IQ ≥1.0 / subject ≥0.02):

| Method | Δ IQ | Δ Subject | Letter |
|---|---:|---:|---|
| `rolling_fifo` | **−1.99** | **−0.035** | **NO** |
| `rolling_fifo_sick` | **−3.47** | **−0.036** | **NO** |
| `rf_tscore` | **−3.53** | **−0.036** | identity; 8-slice ≠ N=32 |
| `rf_tscore_always` | **−4.32** | **−0.029** | **NO** |
| `sf_tscore` | −0.92 | **−0.042** | identity; 8-slice ≠ N=32 |
| `sf_tscore_always` | −0.90 | **−0.047** | **NO** (tail down) |

## Letter

All six **NO**. FIFO moves Rolling pixels (+21%) and still
fails Imaging Quality / Subject vs the N=32 host. Sick FIFO
sleeps. The freeze-score gate never fires on this N=8; always-on
is a second seed that does not beat the host on quality or
(Self Forcing) tail. Not Wan-14B. Do not swap in the 14B
teacher. Do not scale. Do not remake cite-128.
