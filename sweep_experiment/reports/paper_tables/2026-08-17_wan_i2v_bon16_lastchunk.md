# Wan I2V 16v last-chunk verifier — 2026-08-17

**Source:** `wan_experiment/results/i2v_bon_16v/{notta,always_bon,gated_bon}_h30s_shard0/summary.json`
Chosen candidate score on chunk 4 (last ~6 s). Lower = closer to the
first-1s-after-cond reference. N=16, seed 0.

## Population

| Method | Mean | Median | vs NOTTA mean |
|---|---|---|---|
| NOTTA | 4.429 | 3.963 | — |
| always-BoN k=4 | **3.226** | 3.408 | **−1.203** |
| gated-BoN k=4 | 3.378 | **3.277** | −1.051 |

gated − always: mean **+0.152**, median **−0.131**.
gated better-or-tie vs always: **6 / 16**.
always better than NOTTA: **14 / 16**. always *worse* than NOTTA: 06, 07.

## Per-video last-chunk score

| i | NOTTA | always | gated | always−NOTTA | gated−NOTTA | gated−always |
|---|---|---|---|---|---|---|
| 00 | 4.948 | 4.427 | 4.427 | −0.522 | −0.522 | 0 |
| 01 | 2.129 | 2.104 | 2.129 | −0.025 | 0 | +0.025 |
| 02 | 5.275 | 4.164 | 5.275 | −1.111 | 0 | +1.111 |
| 03 | 2.798 | 1.567 | 2.688 | −1.231 | −0.110 | +1.121 |
| 04 | 8.874 | 4.696 | 4.696 | −4.178 | −4.178 | 0 |
| 05 | 4.990 | 3.908 | 4.328 | −1.082 | −0.663 | +0.420 |
| 06 | 2.729 | 3.338 | 2.594 | **+0.609** | −0.135 | −0.743 |
| 07 | 1.963 | 2.620 | 1.963 | **+0.657** | 0 | −0.657 |
| 08 | 4.145 | 3.641 | 3.745 | −0.504 | −0.400 | +0.104 |
| 09 | 5.648 | 4.681 | 5.015 | −0.967 | −0.633 | +0.334 |
| 10 | 3.782 | 3.477 | 3.477 | −0.304 | −0.304 | 0 |
| 11 | 11.192 | 4.319 | 2.157 | −6.872 | −9.035 | −2.162 |
| 12 | 5.494 | 2.375 | 4.974 | −3.119 | −0.520 | +2.599 |
| 13 | 2.263 | 2.044 | 2.263 | −0.218 | 0 | +0.218 |
| 14 | 1.242 | 1.234 | 1.242 | −0.008 | 0 | +0.008 |
| 15 | 3.393 | 3.021 | 3.077 | −0.372 | −0.315 | +0.056 |

## Locked claim

User lock (2026-08-16): if gating is the novelty, **gated must not lose
to always-on**. At N=16 last-chunk composite:

- Quality win vs always-on: **no** (mean +0.152, 6/16 better-or-tie).
- Tie + cheaper: **closest fit**. Gated is 21% cheaper (211 vs 267 s)
  and the mean gap is small vs the 1.2 point win over NOTTA. Median
  actually favors gated.
- Quality loss: not a collapse. Gated still beats NOTTA by ~1.05.
  On the two videos where always-on *hurt* NOTTA (06, 07), gated was
  better — 07 never fired.

Honest paper line at this N: **search works; gating is an efficiency
controller that does not erase the search gain and can block
over-search.** Do not write “gated beats always-on.” Do not drop
gating. Score endpoint sharp/motion next if we want a second metric.
No TTC until that call is made.
