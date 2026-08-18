# Wan I2V 32v hybrid-gate last-chunk (2026-08-17)

**Source:** cluster `analyze_i2v_bon.py` on
`wan_experiment/results/i2v_bon_32v_hybrid/` (user paste 23:10).
**Methods:** NOTTA | always-BoN k=4 | gated-BoN hybrid
(ch1 incoming>0.8 / late>2.0 / Δ>0.5 ∧ prev>0.5).
**Horizon:** 30 s, 5×24 latents, seed 0.
**Lower composite = better.** Do not cite the raw means — video 26
(spiral galaxy) is 85.63 for both search methods vs NOTTA 5.06.

Regenerate: `python wan_experiment/scripts/analyze_i2v_bon.py --series-dir wan_experiment/results/i2v_bon_32v_hybrid`
Robust cuts below are from the pasted per-video table (documented
arithmetic; not `build_paper_tables.py`).

## Cite this: medians + exclude-26

| Cut | N | NOTTA mean / med | always mean / med | gated mean / med | gated−always mean / med | gated cheaper |
|---|---|---|---|---|---|---|
| **Median (all 32)** | 32 | 3.953 / **3.679** | 5.662 / **2.966** | 5.621 / **3.036** | **−0.041 / 0.000** | **33%** (173 vs 258 s) |
| Exclude video 26 | 31 | 3.917 / 3.577 | **3.082** / 2.911 | **3.040** / 3.006 | −0.042 / 0.000 | same jobs |
| First 16 (paired w/ `i2v_bon_16v`) | 16 | 4.429 / 3.964 | 3.226 / 3.408 | **3.108** / 2.952 | **−0.118 / 0.000** | — |
| Videos 16–31 | 16 | 3.476 / 3.292 | 8.097 / 2.654 | 8.134 / 3.037 | +0.036 / 0.000 | — |

Win counts (all 32):

| Contrast | Mean Δ | Median Δ | better | tie | worse | better-or-tie |
|---|---|---|---|---|---|---|
| always − NOTTA | +1.709 | **−0.434** | 25 | 0 | 7 | 25/32 |
| gated − NOTTA | +1.668 | **−0.269** | 22 | 5 | 5 | 27/32 |
| gated − always | **−0.041** | **0.000** | 9 | 10 | 13 | 19/32 |

Exclude-26 always−NOTTA mean flips to **−0.835** (25/31 better). Search
works. The +1.7 raw mean is one clip.

Wall: NOTTA 91.7 s · always 258.1 s · gated **172.8 s** (33% vs always,
1.88× NOTTA). Gate fired **66/128** searchable chunks (51%).

## Hybrid vs T=2.0 on the original 16

First-16 NOTTA / always last-chunks match `i2v_bon_16v` to three
decimals (4.429 / 3.226). Pairing held.

| Gated rule | gated mean | gated−always | Note |
|---|---|---|---|
| T=2.0 only (`i2v_bon_16v`) | 3.378 | **+0.152** | missed 05/02/09/12 |
| Hybrid (this run) | **3.108** | **−0.118** | sign flip; 05/02/09 match always; 12 almost (2.494 vs 2.375); 06/07 saved |
| Hybrid, drop video 03 | 3.126 | **−0.210** | 03 is the leftover miss (ch1 coin-flip then skip) |

Hypothesis (+0.15 → ~−0.2) is **directionally confirmed**. Video 03
ate the rest (+1.260 gated−always).

## Always-on hurt; did the gate save it?

| i | always−NOTTA | gated−NOTTA | Read |
|---|---|---|---|
| 06 | +0.609 | **−0.135** | saved (early skip) |
| 07 | +0.657 | **0** | saved (early skip; last-chunk trend picked cand0) |
| 16 | +0.272 | **−2.119** | gated won big |
| 19 | +0.820 | +0.820 | followed always; both hurt |
| 26 | **+80.567** | **+80.567** | search catastrophe; gate followed |
| 28 | +0.595 | **0** | saved |
| 30 | +0.244 | **0** | saved |

Gating is doing the job it was hired for on the always-hurt cases,
except when it copies a bad search (19, 26).

## Remaining gated misses vs always (Δ>0.2)

| i | gated−always | Why |
|---|---|---|
| 17 | +1.453 | **never fired** (incoming 0.76–1.20; Δ max 0.40) |
| 03 | +1.260 | ch1 coin-flip then skip; always kept searching |
| 24 | +0.864 | ch1 tiny pick then skip |
| 18 | +0.707 | fire only on last chunk |
| 25 | +0.524 | fired; gated **hurt** NOTTA (+0.513) |
| 20 / 23 / 13 | +0.29–0.30 | late / partial search |

## Gate-reason counts (gated method)

`skip` 62 · `level+trend` 24 · `ch1` 15 · `trend` 13 · `level` 12 · `ch1+level` 2.

## Locked read

- **Not a quality win vs always-on.** Median gated−always is 0; win
  rate 9–13–10. Do not write “gated beats always-on.”
- Honest line is still **tie + cheaper** (33% less wall, keeps the
  median search gain: 3.04 vs 2.97 vs 3.68).
- Hybrid was the right next gate: original-16 sign flip, 05/02/09/12
  caught, 06/07/28/30 saved. Do not drop gating.
- Do **not** cite raw 32v means. Video 26 is a search failure (incoming
  0.84→24.6), not a threshold miss.
- Next lever is **stay-on hysteresis** (once fired, keep searching —
  would target 03/24) and a wake-up for never-fire 17. Not another
  global-T sweep. Not TTC.

## Per-video last-chunk

| i | key | NOTTA | always | gated | always−N | gated−N | gated−A |
|---|---|---|---|---|---|---|---|
| 00 | A black and white abstract video | 4.948 | 4.427 | 4.427 | −0.522 | −0.522 | 0.000 |
| 01 | A boiling pot cooking vegetables | 2.129 | 2.104 | 1.936 | −0.025 | −0.193 | −0.167 |
| 02 | A budding and blossoming flower | 5.275 | 4.164 | 4.164 | −1.111 | −1.111 | 0.000 |
| 03 | A bunch of cars on a highway | 2.798 | 1.567 | 2.827 | −1.231 | +0.030 | **+1.260** |
| 04 | a bald eagle flying | 8.874 | 4.696 | 4.696 | −4.178 | −4.178 | 0.000 |
| 05 | a bar with chairs and a television | 4.990 | 3.908 | 3.908 | −1.082 | −1.082 | 0.000 |
| 06 | a basket of french fries | 2.729 | 3.338 | 2.594 | +0.609 | −0.135 | −0.743 |
| 07 | a beach with a lot of buildings | 1.963 | 2.620 | 1.963 | +0.657 | 0.000 | −0.657 |
| 08 | a beautiful woman in a blue sari | 4.145 | 3.641 | 3.745 | −0.504 | −0.400 | +0.104 |
| 09 | a bicycle leaning against a fence | 5.648 | 4.681 | 4.681 | −0.967 | −0.967 | 0.000 |
| 10 | a bird with a fish in its beak | 3.782 | 3.477 | 3.477 | −0.304 | −0.304 | 0.000 |
| 11 | a blue and white smoke | 11.192 | 4.319 | 2.157 | −6.872 | −9.035 | −2.162 |
| 12 | a blue car driving down a dirt road | 5.494 | 2.375 | 2.494 | −3.119 | −2.999 | +0.119 |
| 13 | a blue fishing boat | 2.263 | 2.044 | 2.336 | −0.218 | +0.073 | +0.292 |
| 14 | a blue train | 1.242 | 1.234 | 1.242 | −0.008 | 0.000 | +0.008 |
| 15 | a boat sits on the shore | 3.393 | 3.021 | 3.077 | −0.372 | −0.315 | +0.056 |
| 16 | a book on fire | 4.776 | 5.047 | 2.656 | +0.272 | −2.119 | −2.391 |
| 17 | a bridge in the middle of a river | 3.006 | 1.553 | 3.006 | −1.453 | 0.000 | **+1.453** |
| 18 | a bridge over a body of water | 2.844 | 2.080 | 2.786 | −0.764 | −0.058 | +0.707 |
| 19 | a brown and white cow eating hay | 2.247 | 3.067 | 3.067 | +0.820 | +0.820 | 0.000 |
| 20 | a brown bear in the water | 4.158 | 2.911 | 3.210 | −1.248 | −0.949 | +0.299 |
| 21 | a building on the side | 2.401 | 1.904 | 1.644 | −0.497 | −0.756 | −0.260 |
| 22 | a bunch of food on a grill | 3.804 | 3.458 | 3.274 | −0.346 | −0.530 | −0.184 |
| 23 | a bunch of houses on a hillside | 2.760 | 2.228 | 2.526 | −0.532 | −0.234 | +0.298 |
| 24 | a busy street with cars | 4.036 | 2.315 | 3.179 | −1.721 | −0.857 | +0.864 |
| 25 | a butterfly on a purple flower | 2.663 | 2.652 | 3.176 | −0.010 | +0.513 | +0.524 |
| 26 | a spiral galaxy | 5.063 | **85.630** | **85.630** | **+80.567** | **+80.567** | 0.000 |
| 27 | a castle on a snowy hill | 6.630 | 6.525 | 6.525 | −0.105 | −0.105 | 0.000 |
| 28 | a chair in a room | 2.061 | 2.656 | 2.061 | +0.595 | 0.000 | −0.595 |
| 29 | a chef preparing mushrooms | 4.149 | 2.308 | 2.419 | −1.840 | −1.730 | +0.110 |
| 30 | a church on a hill | 1.444 | 1.688 | 1.444 | +0.244 | 0.000 | −0.244 |
| 31 | a city bus on a snowy street | 3.577 | 3.535 | 3.535 | −0.042 | −0.042 | 0.000 |
