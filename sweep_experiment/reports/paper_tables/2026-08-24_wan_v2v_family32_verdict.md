# Family wave N=32 verdict (2026-08-24)

Jobs: pack-2 **16261273–276 FAILED**; resume workers=1
**16263007–010** and **16263080–086** COMPLETED 0:0 (skip-existing);
both VBench **16263011 / 16263087** COMPLETED 0:0 ~1.5 h.
All four methods **32/32 mp4** + official 7-dim **n=32**.

Cite **medians**. Paper baseline = SF notta (`confirm_32v`).
Ablation zero = `rolling_notta` (`forward_32v`). Analyzer
`PROMOTE` is vs SF and is the **wrong call** for RF controllers.

## Tails (paired N=32)

| Method | tail med | vs SF | W/L vs SF | vs RF | W/L/tie vs RF | exact RF |
|---|---:|---:|---|---:|---|---:|
| notta (SF) | 0.0135 | — | — | −24% | 11/21 | 0 |
| rolling_notta | 0.0178 | +31% | 21/11 | — | — | — |
| rf_rewind | 0.0191 | +41% | 24/8 | **+7.7%** | **23/2/7** | 7 |
| rf_sick_search | 0.0189 | +40% | 22/10 | **+6.5%** | **15/9/8** | 8 |
| rf_pseudo | 0.0180 | +33% | 22/10 | **+1.3%** | **11/3/18** | **18** |
| rf_sink | 0.0220 | +62% | 27/5 | **+24%** | **27/5/0** | 0 |

## Official VBench (full clip)

| Method | subject | IQ | Dyn | flicker | vs RF subj | vs RF IQ |
|---|---:|---:|---:|---:|---:|---:|
| notta (SF) | 0.665 | 69.65 | 0 | 0.986 | — | — |
| rolling_notta | **0.702** | **70.44** | 0 | — | — | — |
| rf_rewind | 0.697 | 70.70 | **1** | 0.981 | −0.005 | +0.26 |
| rf_sick_search | **0.703** | 70.63 | 0.5 | 0.982 | +0.001 | +0.19 |
| rf_pseudo | **0.704** | 69.94 | 0 | 0.982 | +0.002 | −0.50 |
| rf_sink | 0.686 | 70.17 | 0 | 0.977 | −0.016 | −0.27 |

Locked bars vs RF (tail > host, IQ ≥ host−1, subject ≥ host−0.02):
**all four pass the letter.** That is not a scale-up.

## Calls

| Family | Method | Call | Why |
|---|---|---|---|
| A | `rf_rewind` | **HOLD N=32** | +8% vs RF, 23/2. Recovers **0027** (0.018→0.033). Dyn 0→1 is a flag, not H1 twitch (flicker 0.981 not 0.972; subject holds). Do not scale tonight. |
| B | `rf_sick_search` | **HOLD N=32** | +6.5%, 15/9/8. Small. 8/32 exact host. Dyn 0.5. |
| D | `rf_pseudo` | **NO** | +1.3%. **18/32 exact rolling** — gate rarely fires. This is the host with a coin-flip extra seed. |
| C | `rf_sink` | **HOLD / no scale** | +24% vs RF, 27/5. Subject −0.016 (near the 0.02 line), flicker 0.977. Pixel-move probe, **not HG-f**. Helps **0004** (0.010→0.022) more than rewind. Identity/flicker cost. |

Do **not** cite analyzer PROMOTE. Beating SF is the RF host, not our widget.

## Named RF wounds

| Video | SF | RF | rewind | sick | pseudo | sink |
|---|---:|---:|---:|---:|---:|---:|
| 0004 | 0.031 | 0.010 | 0.013 | 0.010 | 0.015 | **0.022** |
| 0027 | 0.035 | 0.018 | **0.033** | 0.021 | 0.018 | 0.019 |

Rewind is the 0027 story (Family A). Sink is the 0004 story.
Neither is a clean 32-wide quality method.

## What this is not

- Not a reason to scale to 128.
- Not H1 `sf_roll` / `rf_chunk` (those were tail 0.028 / Dyn 1 / flicker 0.972 / subject fail).
- Not TTC / LoRA / I2V scale-up.
- Not true HG-f.

Source: harvest 2026-08-24 00:43. Regenerable from
`v2v_panda_family_32v/*_h30s_shard0/{summary.json,vbench_full/joined.json}`
plus confirm/forward pair scripts.
