# Caption leftover-flow HIWYN extras harvest (2026-09-06)

Series `v2v_panda_caption_nwarp_8v` (+ `_smoke`). Prompt =
`prompt_source=metadata_csv` (truck hood). Extra-only: pass 1
stays white; later extras are a carried snow field mixed at
γ=0.5 along leftover Farneback mean flow. **Not** moving the
guessed picture (`pred`). **Not** Go-with-the-Flow (they
fine-tune the video model).

Analyzer FAIL is versus Self Forcing. Cite tails versus
caption Self Forcing first-8 (**0.0129**). Quality letter
versus caption-32 N=32 host **0.700 / 71.54**. First-8 Self
Forcing Visual Benchmark (VBench) slice from the
`sf_tscore` identity row: IQ **70.62** / subject **0.658**.

## Jobs

| Job | Role | State | Elapsed |
|---|---|---|---|
| **17028867** | smoke generate | COMPLETED 0:0 | 7m 02s |
| **17028870** | smoke generate | COMPLETED 0:0 | 6m 17s |
| **17028871** | smoke VBench | COMPLETED 0:0 | 6m 32s |
| **17028874** | N=8 generate (always, 23 min) | COMPLETED 0:0 | 22m 53s |
| **17028875** | N=8 generate (live, 17 min) | COMPLETED 0:0 | 16m 47s |
| **17028876** | N=8 VBench | COMPLETED 0:0 | 19m 11s |

8/8 mp4 + sidecar each; smoke 2/2. `nwarp enabled=true` on
always-on. `nwarp.enabled=false` on live skips. γ=0.5,
Farneback, 16 pairs. Truck-hood leftover flow is tiny
(`vy_px=0.0079`, `vx_px=−0.0013`); integer shift on 0000 is
`dy=0, dx=0`. The extras still change: a **carried** field
mixed with fresh snow is not ordinary white.

Live gate `prefix_motion >= 0.012` fired **0001 / 0006 /
0007** (3/8). The other five tails match Self Forcing to
printed precision.

## Tails vs caption Self Forcing first-8

| Method | Host | tail med | vs host | W/L/T | Call |
|---|---|---:|---:|---|---|
| Self Forcing first-8 | — | 0.0129 | — | — | host |
| `sf_nwarp` | Self Forcing | 0.0157 | **+22%** | 6/2/0 | tail yes |
| `sf_nwarp_live` | Self Forcing | 0.0145 | **+12%** | 3/0/5 | gate works; fire = always |

Last-chunk motion went the other way (`sf_nwarp` −3.05 vs
host −0.94). Handcrafted tail up is not official Dynamic
Degree.

## Official VBench (N=8)

Dynamic Degree: median 0 → 0/8; 0.5 → 4/8; 1.0 → 8/8.
Other dims = median.

| Method | Subject | Background | Aesthetic | Imaging | Smooth | Dyn | Flicker |
|---|---:|---:|---:|---:|---:|---:|---|
| Self Forcing N=32 | 0.700 | 0.839 | 0.502 | 71.54 | 0.992 | 7/32 | 0.989 |
| Self Forcing first-8 (`sf_tscore` identity) | 0.658 | — | — | 70.62 | — | 0/8 | — |
| `sf_nwarp` | **0.594** | 0.857 | **0.399** | **49.18** | 0.988 | **0/8** | 0.985 |
| `sf_nwarp_live` | **0.628** | 0.807 | 0.458 | **54.42** | 0.991 | 2/8 | 0.986 |

`sf_nwarp` flicker **mean 0.978** is the twitch band; median
0.985 is not.

Vs N=32 host (IQ ≥1.0 / subject ≥0.02):

| Method | Δ IQ | Δ Subject | Letter |
|---|---:|---:|---|
| `sf_nwarp` | **−22.36** | **−0.106** | **NO** (paint) |
| `sf_nwarp_live` | **−17.12** | **−0.072** | **NO** (paint when it fires) |

Vs first-8 Self Forcing slice (IQ 70.62 / subject 0.658) the
same letter holds. Analyzer: **FAIL (motion win, quality
collapse)** on both. Smoke already showed always-on IQ
**44.97**.

## Letter

Both **NO**. Extra-only leftover-flow extras raise the last
seconds’ pixel wiggle and wreck the photograph (Imaging
Quality 49 / 54, AdaSteer-class). Official Dynamic Degree
does not rise on always-on (0/8). Live Dyn 2/8 is Dyn-only
and still fails Imaging Quality / subject. Do not retune
γ. Do not scale. Do not silently switch to moving `pred`.
Do not remake cite-128. Do not start 8-GPU DMD.

This is the Go-with-the-Flow video lesson on a frozen
student: temporally locked extras are a new noise prior.
The leftover mean was too small to shift a latent cell on
the truck-hood clip; the carry + mix was enough to break
the picture.
