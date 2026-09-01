# Pseudo-next N=8 harvest — CachedSearch + re-gate (2026-08-31)

Smoke **16679371–375** COMPLETED 2/2. N=8 **16679376–379**
8/8. VBench **16679380** 25m. Series
`v2v_panda_caption_pseudo_next_8v`. Prompt = `metadata_csv`.
γ=0 k=4. Cite first 8 of caption-32.

Analyzer `FAIL (quality collapse)` compares n=8 method VBench
to caption-32 **n=32** notta (0.700 / 71.54). Do not use that
string as the CachedSearch quality call. Tails of
`sf_pseudo_cached` / `sf_always_cached` **match** caption-32
`sf_pseudo` / `sf_always_search` first 8 to 6 decimals.

## Average time per clip (caption-32, cite the mean)

| Method | n | mean s / clip | median s |
|---|---:|---:|---:|
| Self Forcing | 32 | 196.1 | 113.1 |
| **Pseudo-future Search** | 32 | **303.6** | 357.0 |
| **Always-search** | 32 | **348.1** | 348.1 |

Pseudo **304 s** vs Always **348 s** (0.87×). First-8 of this
wave is a **median** (n=8 is too small to prefer the mean):
Pseudo 360 s, Always 349 s — both are fired-clip walls.

## Numbers (cite medians; Dyn% = percent of clips)

| Method | tail | wall s (median) | subject | IQ | Dyn% | open fire |
|---|---:|---:|---:|---:|---:|---|
| caption-32 notta first 8 | 0.0129 | 119 | 0.700* | 71.54* | — | — |
| caption-32 `sf_pseudo` first 8 | 0.0145 | **360** | 0.701* | 71.66* | — | 6/8† |
| caption-32 `sf_always_search` first 8 | 0.0149 | **349** | 0.687* | 71.16* | — | — |
| `sf_pseudo_cached` | 0.0145 | 389 | 0.640 | 69.83 | 50% (4/8) | 6/8 |
| `sf_always_cached` | 0.0149 | 393 | 0.640 | 69.83 | 62.5% (5/8) | always |
| `sf_repseudo` | 0.0145 | 552 | 0.640 | 69.77 | 50% (4/8) | 6/8 |
| `sf_repseudo_cached` | 0.0145 | 584 | 0.640 | 69.77 | 50% (4/8) | 6/8 |

\* n=32 official medians, not first-8 VBench. † caption-128 /
N=32 opening rate was ~70%; this 8 is 6/8.

Re-gate per-chunk fire: **6 / 5 / 6 / 7 / 8 / 6**. Alive
(turns off on chunk 1, on for all 8 on chunk 4).

Job wall: cached 45m / always-cached 57m / re-gate 76m /
stacked 80m.

## Calls

| Arm | Call | Why |
|---|---|---|
| CachedSearch cheapen | **NO** | Same tail as full search. Wall **higher** (389 vs 360; 393 vs 349). CPU KV snap is a tax, not a save. |
| Re-gate | **NO** | Gate lives, but tail ≈ opening Pseudo and +53% wall. Always 128 already says the opening gate only drops 4 Dyn clips. |
| Stacked | **NO** | Quality identity with `sf_repseudo`. Slowest. |

Do not scale any of the four. Do not remake caption-32 / 128.
Do not retune γ or k. Next cheapen is **not** this KV snap
(search only early chunks, or prune k on the fired path).

Keep letter vs 0.68 / 70.5: n=8 subject 0.640 / IQ 69.8 miss.
That is this 8-video VBench, not proof CachedSearch broke
caption-32 Pseudo (tails match). Still no promote.

Smoke 2/2 is a crash pass only.
