# Caption mix lock + context noise harvest (2026-09-04)

Series `v2v_panda_caption_mixctx_8v`. Prompt =
`prompt_source=metadata_csv` (truck hood). Diagnostic. Mixed
inference is Liu et al. Appendix E (named, not run). Context
noise is a KV write timestep, not leftover ρ.

Analyzer FAIL is versus Self Forcing. Cite Rolling arms versus
caption Rolling first-8. Cite Self Forcing arms versus caption
Self Forcing first-8. Quality letter versus caption-32 N=32
hosts (same bars as leftover / schedule8): Rolling **0.694 /
70.22**, Self Forcing **0.700 / 71.54**.

## Jobs

| Job | Method | State | Elapsed |
|---|---|---|---|
| **16931124** | `rf_mix` | COMPLETED 0:0 | 14m 26s |
| **16931125** | `rf_mix_always` | COMPLETED 0:0 | 13m 33s |
| **16931126** | `sf_mix` | COMPLETED 0:0 | 22m 05s |
| **16931127** | `sf_mix_always` | COMPLETED 0:0 | 19m 56s |
| **16931128** | `rolling_ctx` | COMPLETED 0:0 | 8m 15s |
| **16931129** | `sf_ctx` | COMPLETED 0:0 | 16m 52s |
| **16931130** | VBench full clip | COMPLETED 0:0 | 49m 43s |

8/8 mp4 + sidecar each. `rolling_ctx` / `sf_ctx` sidecar
`context_noise=50`. Pair `c0_gate` shows `rf_mix n_chunked` 0–2
(mix fired). Top-level `mix_logs` is nested under `chunk_logs`.

## Tails vs matching first-8 host

Host medians from the pair last Rolling column / `notta` column,
videos 0000–0007. Rolling first-8 = **0.0134** (same slice as
schedule8; leftover wrote 0.0128 — do not edit that file).
Self Forcing first-8 = **0.0129**.

| Method | Host | tail med | vs host | W/L/T | Call |
|---|---|---:|---:|---|---|
| Rolling first-8 | — | 0.0134 | — | — | host |
| Self Forcing first-8 | — | 0.0129 | — | — | host |
| `rf_mix` | Rolling | 0.0169 | **+26%** | 8/0/0 | tail yes |
| `rf_mix_always` | Rolling | 0.0226 | **+69%** | 7/1/0 | tail yes |
| `rolling_ctx` | Rolling | 0.0142 | **+6%** | 5/3/0 | weak |
| `sf_mix` | Self Forcing | 0.0133 | **+3%** | 6/2/0 | weak |
| `sf_mix_always` | Self Forcing | 0.0145 | **+12%** | 7/1/0 | tail yes |
| `sf_ctx` | Self Forcing | 0.0143 | **+11%** | 6/2/0 | 0004 tail **0.190** twitch |

## Official VBench (N=8)

Dynamic Degree: median 0 → 0/8; 0.5 → 4/8; 1.0 → 8/8.
Other dims = median.

| Method | Subject | Background | Aesthetic | Imaging | Smooth | Dyn | Flicker |
|---|---:|---:|---:|---:|---:|---:|---:|
| Self Forcing N=32 | 0.700 | 0.839 | 0.502 | 71.54 | 0.992 | 0 | 0.989 |
| Rolling N=32 | 0.694 | — | — | 70.22 | — | 0 | 0.985 |
| `rf_mix` | 0.649 | 0.800 | 0.566 | **68.30** | 0.990 | **4/8** | 0.983 |
| `rf_mix_always` | **0.632** | 0.806 | 0.502 | **67.71** | 0.989 | **8/8** | **0.978** |
| `sf_mix` | 0.660 | 0.810 | 0.514 | **70.51** | 0.993 | 0/8 | 0.987 |
| `sf_mix_always` | 0.661 | 0.807 | 0.560 | 70.74 | 0.992 | **4/8** | 0.986 |
| `rolling_ctx` | 0.664 | 0.810 | 0.546 | **67.55** | 0.992 | 0/8 | 0.986 |
| `sf_ctx` | **0.596** | 0.802 | 0.527 | **69.21** | 0.991 | 0/8 | 0.985 |

Vs N=32 host (IQ ≥1.0 / subject ≥0.02):

| Method | Δ IQ | Δ Subject | Letter |
|---|---:|---:|---|
| `rf_mix` | **−1.92** | **−0.045** | **NO** |
| `rf_mix_always` | **−2.51** | **−0.062** | **NO** (twitch) |
| `rolling_ctx` | **−2.68** | **−0.030** | **NO** |
| `sf_mix` | **−1.03** | **−0.040** | **NO** |
| `sf_mix_always` | −0.80 | **−0.039** | **NO** (subject) |
| `sf_ctx` | **−2.33** | **−0.104** | **NO** (twitch) |

`rf_tscore` in the FIFO wave is pixel-identical to Rolling
first-8, so that row is the first-8 Rolling VBench: IQ **66.70**
/ subject 0.658. A softer first-8 bar would not flip
`rf_mix_always` (flicker 0.978, Dyn 8/8) or `sf_ctx`. It would
not make mix a paper method (Liu Appendix E).

## Letter

All six **NO**. Gated mix moves the tail and still kills
Imaging Quality / Subject vs the N=32 host. Always-on Rolling
mix is the crossed-host twitch (Dyn 8/8, flicker 0.978).
`context_noise=50` is on and still paints (Rolling IQ 67.55;
Self Forcing 0004 explodes). Do not scale. Do not remake
cite-128. Do not start 8-GPU DMD.
