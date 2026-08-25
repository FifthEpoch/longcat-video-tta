# Caption WAVE=1 outcomes (2026-08-24, generate so far)

Series `v2v_panda_caption_32v`. Prompt = first-segment Panda
caption (`prompt_source=metadata_csv`). Do **not** mix with
stem-prompt tables. Official quality is VBench / VBench++ on the
**full clip** after job **16310330**. That job is still waiting on
the remaining generate arms.

This file starts filling from finished generate sidecars. Subject /
IQ / dynamic degree / flickering stay blank until VBench.

## Jobs (22:41 squeue)

| Job | Method | Generate | n (last seen) |
|---|---|---|---|
| **16310318** | notta | COMPLETED 0:0 1h47 | **32/32** |
| **16310319** | rolling_notta | COMPLETED 0:0 27m | **32/32** |
| **16310320** | sf_rewind | COMPLETED 0:0 1h13 | **32/32** |
| **16310321** | sf_sick_search | left squeue — harvest now | was 6/32 at 20:46 |
| **16310322** | sf_pseudo | R 1h49 gh107 | was 3/32 |
| **16310323** | sf_sink | R 27m | — |
| **16310324–329** | SF always + RF family | R 27m on h200_cds | — |
| **16310330** | VBench full clip | PD (Dependency) | — |
| **16314667–670** | AdaSteer 8v | **not in this squeue** | sacct |

## Caption generate (handcrafted tail, N=32 paired)

Internal diagnostic only. Not a VBench dimension.

| Method | tail | vs caption SF | W/L/tie | Δ med | Official VBench |
|---|---:|---:|---|---:|---|
| notta (SF) | **0.01164** | — | — | — | pending 16310330 |
| rolling_notta | 0.01423 | +22% | 23/9/0 | +0.00273 | pending |
| sf_rewind | 0.01262 | +8% | 23/5/4 | +0.00075 | pending |
| sf_sick_search | — | — | — | — | harvest 321 |
| sf_pseudo | — | — | — | — | running |
| sf_sink | — | — | — | — | running |
| always / RF family | — | — | — | — | running |

Stem audit (same methods, wrong text): SF 0.0135 / RF 0.0178 +31% /
rewind 0.0143 +6% 19/5/8 / sick −1%. Caption SF baseline is lower
(−14% vs stem). Rolling gap shrank (+22% vs +31%); win set wider
(23 vs 21). Rewind still a small typical plus.

## Official VBench (full clip) — not yet

Columns will be subject consistency, imaging quality, dynamic
degree, temporal flickering. Same suite as Self-Forcing / Rolling
Forcing / LongLive. Fill only from **16310330**.

## Do not

Call HOLD/NO from tails. Submit WAVE=2. Mix stem numbers into this
table. Claim AdaSteer finished because it left `squeue`.
