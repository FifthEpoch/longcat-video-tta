# Caption 128 official Dynamic Degree — percent of clips (2026-08-31)

VBench authors: each clip is 0/1 (RAFT). Cite
`population.dynamic_degree.mean`, not the median. All four rows
below that have a join have **median 0**.

**Series:** `v2v_panda_caption_128v`. Prompt = `metadata_csv`.
Hosts from **16545806**. Pseudo join from **750** (preempted after
writing Pseudo). Always **no joined.json** — **16674378** still
running. Do not remake videos.

| Method | n dynamic | Dyn% | median |
|---|---:|---:|---:|
| Self Forcing | 42/128 | **32.8%** | 0 |
| Rolling Forcing | 37/128 | 28.9% | 0 |
| Pseudo-future Search | **61/128** | **47.7%** | 0 |
| Always-search | — | **no join** | — |

`0.328125 × 128 = 42`. `0.2890625 × 128 = 37`.
`0.4765625 × 128 = 61`.

## Same table with the rest of the official row

| Method | tail | subject | IQ | Dyn% | flicker |
|---|---:|---:|---:|---:|---:|
| Self Forcing | 0.0119 | 0.666 | 72.07 | 32.8% (42) | 0.988 |
| Rolling Forcing | 0.0158 | **0.685** | 71.52 | 28.9% (37) | 0.983 |
| Pseudo-future Search | **0.0157** | 0.660 | **72.38** | **47.7% (61)** | 0.984 |
| Always-search | 0.0168 | — | — | — | — |

## vs Self Forcing (paper baseline)

Tail **+32%**. Dyn% **+14.8 pp** (61 vs 42). Subject −0.006
(holds −0.02). IQ **+0.30**. Flicker −0.004.

This is the N=32 sign at paper size: N=32 was Dyn 40.6% (13/32)
vs Self Forcing 21.9% (7/32). Do not cite N=32 subject 0.701
here. The 128 Self Forcing subject is 0.666; Pseudo followed it.

## vs Rolling Forcing (success bar)

Tail **≈ tie**. Dyn% **+18.8 pp** (61 vs 37). IQ **+0.85**.
Subject **−0.025** (misses −0.02). Cost is still ~5 min / clip
vs Rolling 45 s.

Honest line: Pseudo is **better than Rolling on official
Dynamic Degree and image quality**, matches the tail, and is
not Rolling on subject or wall time. Always official still
needed to say how much of the Dyn% is the pick vs the gate.

N=32 Always was 43.8% (14/32) vs Pseudo 40.6% (13/32). If 128
Always lands near Pseudo, the gate is the cost cut. If it lands
clearly higher, the gate is skipping live openings.

## Gate rate (chunks[0].pseudo_fire)

**90 fire / 38 skip** (70% / 30%). Caption N=32 was 23 fire / 9
skip (72% / 28%). The 38 skips are exact Self Forcing — that is
why Always tail is hotter (0.0168 vs 0.0157).

`gate_reason` on fired clips is `sick_motion`. That is the
motion+trust **pick** after the hold-out fired. The planner
label `pseudo_fire` is overwritten. Cite the boolean, not that
string.

## Source duration

`stream=nb_frames` is empty on all 128 paths. Not a length
measurement. Still no paired PSNR/SSIM/LPIPS until
`format=duration` (or fps×seconds) is in.

Do not scancel **16674378**. Mid-chunk rewrite stays closed.
