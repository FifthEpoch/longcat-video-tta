# SF-family N=32 verdict (2026-08-24)

Series `v2v_panda_sf_family_32v`. Host = Self-Forcing chunked.
Paper baseline = SF notta. RF `rolling_notta` is comparison only.
Cite **medians**. Official VBench = full clip. Playbook cells
from `2026-08-24_wan_v2v_sf_family_dissect.md`.

Generate: **16266879** sick, **880** pseudo, **881** sink;
rewind **16266878 FAILED** 8/32, resume **16267992**.
VBench **16268053** afterok 992:879:880. All four **n=32**
+ locked VBench dims present.

## Headline

| Method | tail | vs SF | W/L/tie | exact-SF | fire/act | vs RF | cell | Call |
|---|---:|---:|---|---:|---|---|---|---|
| notta (SF) | 0.0135 | — | — | — | — | — | — | baseline |
| rolling_notta | 0.0178 | +31% | 21/11/0 | — | — | — | — | host |
| `sf_rewind` | 0.0143 | **+6%** | 19/5/8 | 8 | 29/24 | 13/19 | F | **HOLD** |
| `sf_sick_search` | 0.0134 | **−1%** | 20/5/7 | 7 | 29/29 | 13/19 | C | **NO** |
| `sf_pseudo` | **0.0186** | **+37%** | **25/2/5** | 5 | 27/27 | **19/13** | F | **HOLD** (lead) |
| `sf_sink` | **0.0232** | **+72%** | **30/2/0** | 0 | 32/32 | 26/6 | F letter / D tax | **HOLD / no-scale** |

Locked bars vs **SF** (tail ↑, IQ ≥ SF−1, subject ≥ SF−0.02):
rewind / pseudo / sink **pass**. sick **fails** tail.

## Official VBench (full clip)

| Method | subject | IQ | Dyn | flicker | IQ | subject |
|---|---:|---:|---:|---:|---|---|
| notta (SF) | 0.6652 | 69.65 | 0 | 0.9863 | — | — |
| rolling_notta | 0.7018 | 70.44 | 0 | 0.9825 | — | — |
| sf_rewind | 0.6801 | 69.44 | 0 | 0.9853 | hold (−0.21) | hold (+0.015) |
| sf_sick_search | 0.6685 | 69.13 | 0 | 0.9862 | hold (−0.52) | hold |
| sf_pseudo | **0.6911** | **69.83** | **0.50** | 0.9821 | hold (+0.18) | hold (+0.026) |
| sf_sink | 0.6457 | 69.98 | 0 | **0.9774** | hold (+0.33) | **on the line (−0.0195)** |

Flicker 0.977 is the RF-sink tax, not H1 (those were 0.972 + subject
down + Dyn 1). Pseudo Dyn 0.50 + flicker 0.982 is recovered motion,
not a crossed sampler.

## Mechanism (required for any F)

**Rewind.** Sensor lives (29/32 sick). Accept 24/32. Fired videos
+11% tail vs SF; quiet exact. **12 later-freeze after accept** —
one-shot rewind, then SF freezes again. Recovers **0027**
(0.035→0.042). Damps **0004** slightly (0.031→0.028). Loses to RF
13/19. Small 32-wide win, not a host-gap close.

**Sick-search.** Same sick sensor, k=4 motion+trust after a freeze.
29/32 fire. 20 tiny wins / 5 losses / 7 ties, **median −1%**.
Conditional +4% does not lift the typical video. Same pick that
pseudo uses, applied **after** collapse instead of from the prefix.
Actuator is late and anti-aligned with 30 s tail.

**Pseudo.** Prefix hold-out fires **27/32** (loose gate, not RF’s
18/32 dead gate). Fired +29% tail. 23/32 beat SF by >10%. Beats
**RF median** (0.0186 vs 0.0178, 19/13). Not seed_bon-32: that
always-searched with a **drift** pick and **damped** tail −9% /
Dyn 0. This searches on a prefix fire with a **motion** pick and
Dyn 0→0.5. Gate is loose enough that the next experiment must
split **gate vs pick**.

**Sink.** Always-on. +72% tail, 30/2. Wakes stills (**0004**
0.031→0.040, **0015** 0.009→0.029, **0029** 0.009→0.027). Damps
**0027** (0.035→0.029). Subject −0.0195 (letter holds by 0.0005).
Flicker 0.977. Pixel-move probe, **not HG-f**. Same tax as RF sink.

## Named wounds

| Video | SF | RF | rewind | sick | pseudo | sink |
|---|---:|---:|---:|---:|---:|---:|
| 0004 | 0.031 | 0.010 | 0.028 | 0.031 | 0.031 | **0.040** |
| 0027 | 0.035 | 0.018 | **0.042** | 0.038 | **0.043** | 0.029 |

Pseudo covers both named RF wounds without sink’s identity tax.
Rewind is the 0027 story. Sink is the 0004 story.

## Win-set Jaccard (beat SF by >10%)

rewind∩pseudo 14/24 = 0.58 (rewind ⊂ pseudo).
pseudo∩sink 18/32 = 0.56.
rewind∩sink 10/32 = 0.31 (combine *allowed*, not advised tonight:
sink tax + pseudo already has 0004∪0027).
sick n_win=6 — drop.

## Invention sentences

- **Pseudo:** On Self-Forcing chunked, a prefix hold-out (last 3
  latents) that fires k=4 motion+trust search on 27/32 videos
  raises median tail **+37%** vs SF notta (25/2/5), IQ/subject
  hold, Dyn 0→0.5. Not seed_bon-32 (wrong pick, tail down).
- **Rewind:** Resampling a sick chunk (+6%, 19/5/8) works when it
  fires (+11%); 12/24 later re-freeze.
- **Sink:** `sink_size` on SF is a still-waker (+72%) with an
  identity/flicker tax. Not a quality method.

## What we will not do tonight

- Scale any arm to 128.
- Combine sink+pseudo (shared wins + tax).
- Keep sick-search as a standalone.
- Retune `DROP=0.8`.
- Add TTC / LoRA / I2V.
- Cite RF-hosted family +X vs SF as this claim.

## Next (one ablation)

`sf_always_search`: **always** k=4, **same pick** as pseudo
(max temporal motion + trust 0.8), no prefix gate. Same first 32.

| If always ≈ pseudo | The gate is fake. Invention is **motion pick on SF**. |
| If always ≪ pseudo (seed_bon-like) | The 5 skips + prefix fire are the method. |
| If always ≫ pseudo with a tax | Gate is a quality brake. Keep gated. |

Rewind stay-on (search 1–2 chunks after accept) is the second
candidate after that split. Spec:
`2026-08-24_wan_v2v_sf_always_search_spec.md`.

Source: dissect harvest 2026-08-24 11:22. Regenerable from
`v2v_panda_sf_family_32v/*_h30s_shard0/{*.json,vbench_full/joined.json}`
plus confirm/forward sidecars.
