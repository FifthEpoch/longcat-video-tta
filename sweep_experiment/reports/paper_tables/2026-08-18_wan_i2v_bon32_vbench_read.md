# Wan I2V official VBench — locked read (2026-08-18)

**Series:** `i2v_bon_32v_hybrid` (do-nothing / always-search / gated-search).
**Jobs:** 15959601 (notta) + 15984561 (always + gated). Exit 0.
**Cite:** last5 outcome window
[`2026-08-18_wan_i2v_bon32_vbench_last5.md`](2026-08-18_wan_i2v_bon32_vbench_last5.md).
Full 30 s is prefix-diluted:
[`2026-08-18_wan_i2v_bon32_vbench_full.md`](2026-08-18_wan_i2v_bon32_vbench_full.md).
No PSNR (no 30 s GT on these stills).

This is the first official scorecard for the controller. The
handcrafted last-piece composite is **not** the quality claim.

---

## Headline

Official VBench does **not** say search or gating improved the videos.

On the last 5 seconds (where methods diverge):

| What we hoped | What VBench says |
|---|---|
| Always-search beats do-nothing | **No.** Do-nothing has the best imaging (68.2 vs 66.4) and background (0.957 vs 0.952). Always is better only on aesthetic (0.548 vs 0.535). |
| Gated ties always and is cheaper | **Mostly a tie, slightly worse on Aes/IQ.** Aes 0.522 vs 0.548; IQ 66.1 vs 66.4. Subject/motion/flicker are ties. Gated is still 33% cheaper (173 vs 258 s). |
| Freeze is the headroom search fixes | **No.** `dynamic_degree` median is **0.0** for all three methods. Always mean 0.250 vs 0.188 (about two extra “dynamic” clips of 32). Gated mean = do-nothing. |
| Handcrafted score tracks official quality | **No.** Last5 ρ(last-chunk, imaging_quality) is **+0.23 / +0.24 / +0.33**. We punish sharpness deviation; MUSIQ rewards sharpness. Most other |ρ| < 0.3. |

Gating is **not** a quality win on official metrics. The efficiency
story survives only as “about the same VBench, 33% cheaper, slightly
worse last5 aesthetic.” That is weaker than the handcrafted-score
efficiency line we had been citing.

---

## last5 medians (higher is better)

| Dimension | do-nothing | always | gated | gated−always |
|---|---:|---:|---:|---:|
| subject_consistency | **0.969** | 0.967 | **0.969** | +0.002 |
| background_consistency | **0.957** | 0.952 | 0.952 | −0.001 |
| aesthetic_quality | 0.535 | **0.548** | 0.522 | **−0.026** |
| imaging_quality | **68.17** | 66.43 | 66.11 | −0.32 |
| motion_smoothness | 0.992 | 0.991 | 0.991 | 0 |
| dynamic_degree | 0 | 0 | 0 | 0 |
| temporal_flickering | 0.989 | 0.988 | 0.988 | 0 |

Per-video gated vs always: subject 14/10/8 (gated edge); aesthetic
8/10/14 and imaging 9/10/13 (always edge); motion and flicker split;
dynamic 30/32 exact ties.

---

## Why the handcrafted “search works” finding died

Handcrafted last-chunk medians were do-nothing 3.68 / always 2.97 /
gated 3.04. That looked like search helping. Official last5 IQ and
background go the **other way**: do-nothing wins.

That is consistent with the 11/16 diagnosis. The composite punishes
sharpen-vs-first-second. Search then prefers less-sharp tails. MUSIQ
imaging_quality treats those tails as worse. Positive Spearman on IQ
is the receipt.

Motion smoothness last5 ρ is the only dim with a useful negative
correlation under search (always −0.26, gated −0.30). Not enough to
carry a quality claim.

---

## Full 30 s

Medians collapse toward each other (shared first ~6 s). Still no
quality win. `dynamic_degree` median still 0. Do not cite full-clip
as the outcome table.

---

## What this does to the paper

- **Drop** “search improves quality” until a metric that actually
  moves says so. Official VBench quality dims do not.
- **Keep** gating as an efficiency controller only if we are willing
  to say last5 Aes/IQ are slightly worse. If Aes matters, hybrid
  gated lost to always-search.
- **Do not** treat sick’s better handcrafted median (2.764) as a
  quality win. If the composite is anti-aligned with IQ, sick may
  look worse on VBench. Score sick only if we need that check.
- Controller loop stays GT-free. Outcome eval stays official.
- No test-time training. A Panda-prefix pixel audit is still a
  different series, not a rescoring of these 32.

Regen: `python wan_experiment/scripts/analyze_i2v_vbench.py --series-dir wan_experiment/results/i2v_bon_32v_hybrid --clip last5`
