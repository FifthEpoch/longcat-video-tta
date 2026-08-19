# Wan I2V 32v search-while-sick last-chunk (2026-08-18)

**Source:** job **15959146** finished 14:24 EDT. User paste of
`analyze_i2v_bon.py --series-dir i2v_bon_32v_sick --baseline-dir i2v_bon_32v_hybrid`.
**Methods:** do-nothing and always-search reused from hybrid; gated-search
is search-while-sick (hybrid alarms + stay-on, off if recovery>0.5 or
outgoing<1.0).
**Horizon:** 30 s, 5×24 latents, seed 0, n_ok=32/32.
**Lower composite = better.** Cite medians. Video 26 is 85.63 for both
search methods. These numbers are still the **handcrafted** score.
Official VBench is not a three-way yet (job 15959601 scored do-nothing
only).

Regenerate:
```
python wan_experiment/scripts/analyze_i2v_bon.py \
    --series-dir wan_experiment/results/i2v_bon_32v_sick \
    --baseline-dir wan_experiment/results/i2v_bon_32v_hybrid
```

## Cite this

| Cut | N | do-nothing med | always med | sick med | sick−always mean / med | sick wall |
|---|---|---|---|---|---|---|
| **Median (all 32)** | 32 | 3.679 | 2.966 | **2.764** | **−0.155 / 0.000** | **204 s** (21% vs always 258; hybrid was 173) |
| Exclude video 26 | 31 | 3.577 | 2.911 | **~2.76** | mean sick 2.922 vs always 3.082 / hybrid gated 3.040 | same jobs |

Exclude-26 sick mean = (5.507×32 − 85.630) / 31 = **2.922** (documented
arithmetic). Hybrid exclude-26 gated mean was 3.040.

Win counts (all 32):

| Contrast | Mean Δ | Median Δ | better | tie | worse | better-or-tie |
|---|---|---|---|---|---|---|
| always − do-nothing | +1.709 | **−0.434** | 25 | 0 | 7 | 25/32 |
| sick − do-nothing | +1.554 | **−0.310** | 25 | 4 | 3 | 29/32 |
| sick − always | **−0.155** | **0.000** | 9 | 14 | 9 | 23/32 |

Fired **84/128** (66%). Reasons: skip 44 · already_on 17 · ch1 15 ·
trend 13 · level 13 · level+trend 24 · ch1+level 2.
Hybrid fired 66/128. Forever-sticky fired 96/128.

## Pass / fail vs the spec

| Check | Target | Sick | Result |
|---|---|---|---|
| 11 smoke | near hybrid 2.16, not sticky 4.32 | **1.830** (hybrid 2.157, always 4.319) | **PASS**, beat hybrid |
| 16 book on fire | near hybrid 2.66, not sticky 5.05 | **2.656** (exact hybrid) | **PASS** |
| 03 highway | near always 1.57 | **1.755** (+0.188 vs always; hybrid miss was +1.260) | **NEAR** (piece 4 turned off after 1.674→1.019 recovery, as predicted) |
| 24 busy street | near always 2.32 | **2.315** exact always | **PASS** |
| 06 / 07 piece 1 | still skipped | skip, then later alarms | **PASS** |
| 30 church | back to 1.44, not 1.69 | **1.444** (ch1 then skip×3) | **PASS** |
| Wall | between 173 and 256 s | **204.1 s** | **PASS** |

17 still never wakes (3.006 = do-nothing, always 1.553). 26 still
copies the 85.63 catastrophe.

## Vs hybrid and forever-sticky (handcrafted)

| Method | Median | sick−always better/tie/worse | Wall | 11 | 16 | 03 | 24 |
|---|---|---|---|---|---|---|---|
| Hybrid | 3.036 | 9 / 10 / 13 | 173 s | 2.157 | 2.656 | miss (+1.26) | miss (+0.86) |
| Forever-sticky | 2.99 | 6 / 21 / 5 | 256 s | 4.319 | 5.047 | exact always | exact always |
| **Search-while-sick** | **2.764** | 9 / 14 / 9 | **204 s** | **1.830** | **2.656** | 1.755 | **2.315** |

First gated rule that keeps hybrid’s unique wins **and** catches 24
(and almost 03) without spending the whole always-search budget.

Not a strict quality win vs always-search: 9–9 split, median 0 delta.
It **is** the best handcrafted median so far and 21% cheaper. Official
hybrid VBench later showed the composite is **anti-aligned with
imaging_quality**. Do not treat this sick median as a quality win.
Score the sick mp4s on VBench only if that check is still needed.

## Locked read

Search-while-sick did the job it was hired for on the handcrafted
score. Hybrid remains the cheapest. Sick is the best *typical*
handcrafted ending. Official VBench is still the scorecard that
decides whether any of this helped by common standards.

No test-time training.
