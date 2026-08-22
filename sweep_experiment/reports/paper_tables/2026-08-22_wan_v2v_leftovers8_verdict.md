# Leftovers N=8 + rolling-128 tails (2026-08-22 17:28)

Do **not** cite analyzer PROMOTE. That table is vs SF `notta`.
The leftover question is vs lineage **`rolling_notta`** (IQ 68.68,
subj 0.630, Dyn 0.50, tail 0.0215).

Jobs: leftovers **16209129–132** 9–11 min 0:0, VBench **16209133**
27 min 0:0. 128 generate **16209126** 3h48 / **16209127** 2h18 0:0.
128 VBench **16209128** still PENDING.

## Vs the RF host (paired n=8)

| Method | tail med | vs host | W/L/T | IQ vs host | Subj | Dyn | Flicker | Call |
|---|---:|---:|---|---:|---:|---:|---:|---|
| rolling_notta | 0.0215 | — | — | 68.68 | 0.630 | 0.50 | — | host |
| rolling_rho_lo | 0.0241 | +12% | 6/2/0 | **67.02 (−1.66)** | 0.644 | 1.00 | 0.974 | **NO** |
| rolling_rho_hi | 0.0301 | +40% | 7/1/0 | **64.91 (−3.77)** | 0.633 | 0.50 | 0.971 | **NO** |
| rolling_adapt | 0.0254 | +18% | 5/1/2 | **67.29 (−1.39)** | 0.630 | 0.50 | 0.974 | **NO** |
| rolling_look | 0.0229 | +6% | 5/3/0 | 69.97 (+1.29) | 0.622 | 1.00 | 0.976 | **HOLD, do not scale** |

`rolling_adapt` bit-matches the rule: stills = `rho_hi`, mid (0001/0006)
= native, hot 0007 = `rho_lo`. The ρ knob **moves pixels** (not SF
shift). It applies the IQ-destroying schedule to the stills native RF
already won. Close idea 4.

`rho_hi` is prefix_sink-class: tail↑, IQ/flicker collapse.
`look` holds IQ vs host and damps live 0001/0007 vs native RF.
N=8 only. Same trap as live_bon-8.

## 128 tails (VBench not in)

| Method | N | tail median |
|---|---:|---:|
| notta | 128 | 0.0136 |
| rolling_notta | 128 | **0.0177 (+30%)** |

N=32 was 0.0135 → 0.0178 (+31%). Median **holds**. Mean / win-rate /
IQ / subject still required. Do not call YES until **16209128**.

## Closed

Idea 4 (adaptive ρ): knob live, quality fail. Keep native RF schedule.
Idea 6/7 (`rolling_look`): N=8 HOLD only. No 32.
Appear / live / seed stay closed. Do not gate RF on live prefixes.
