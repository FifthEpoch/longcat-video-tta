# Caption official VBench — complete (2026-08-25 20:26)

Prompt = `metadata_csv`. Cite **medians**. Full-clip VBench n=32.
Locked bars vs caption SF **0.700 / 71.54 / 0 / 0.989**: tail ↑,
IQ ≥ 70.54, subject ≥ 0.680. RF-hosted rows also vs RF host
**0.694 / 70.22 / 0 / 0.985**.

`rf_sink` VBench **16358585** COMPLETED 0:0 in 10m48s. Caption
official set is complete. Do not mix stem-prompt numbers.

## SF-hosted (the claim)

| Method | tail | vs SF | W/L | subject | IQ | Dyn | flicker | Call |
|---|---:|---:|---|---:|---:|---:|---:|---|
| notta (SF) | 0.01164 | — | — | **0.700** | **71.54** | 0 | 0.989 | baseline |
| rolling_notta | 0.01423 | +22% | 23/9 | 0.694 | **70.22** | 0 | 0.985 | host; IQ fail vs SF |
| sf_rewind | 0.01262 | +8% | 23/5/4 | 0.698 | 70.89 | 0 | 0.988 | **HOLD** small |
| sf_sick_search | 0.01164 | +0% | 19/4/9 | 0.697 | 71.54 | 0 | 0.988 | **NO** |
| sf_pseudo | 0.01492 | +28% | 23/0/9 | 0.701 | 71.66 | **0** | 0.985 | **HOLD tail.** Not Dyn |
| sf_always_search | 0.01623 | +39% | 30/2/0 | 0.687 | 71.16 | 0 | 0.984 | **HOLD** ablation |
| sf_sink | 0.01907 | +64% | 31/1/0 | **0.672** | 70.89 | 0 | 0.982 | **NO** subject bar |
| seed_bon | 0.00954 | **−18%** | 11/21 | **0.746** | 70.54 | 0 | 0.990 | **NO** motion |
| live_bon | 0.01187 | +2% | 6/5/21 | 0.723 | 71.43 | 0 | 0.989 | **NO** (21 ties) |
| appear_bon | 0.01117 | −4% | 13/19 | 0.723 | 71.23 | 0 | 0.989 | **NO** tail |

## RF-hosted (vs that host)

| Method | tail vs RF | W/L vs RF | subject | IQ | Dyn | flicker | Call vs RF | vs SF |
|---|---:|---|---:|---:|---:|---:|---|---|
| rolling_notta | — | — | 0.694 | 70.22 | 0 | 0.985 | host | IQ fail |
| rf_rewind | +6% | 16/9/7 | 0.692 | 70.32 | 0 | 0.984 | **HOLD** small | IQ fail |
| rf_sick_search | −1% | 12/11/9 | 0.695 | 70.16 | 0 | 0.985 | **NO** | IQ fail |
| rf_pseudo | +8% | 9/3/20 | 0.701 | 70.22 | 0 | 0.984 | **NO** (20 ties) | IQ fail |
| rf_always_search | +25% | 23/9/0 | 0.695 | 70.24 | 0 | 0.983 | tail up; ≈ host | IQ fail |
| rf_sink | +42% | 29/3/0 | **0.709** | 70.15 | 0 | 0.980 | tail + subject vs host | IQ **−1.39** fail |

`rf_sink` vs SF: tail +73% (29/3/0), subject **0.709** (+0.009),
IQ **70.15** (−1.39), Dyn 0, flicker 0.980. Subject holds; IQ
does not. Opposite of SF sink (subject 0.672). Do not cite stem
flicker 0.977. Not ours. **NO** as a claim vs SF.

Every RF row fails the IQ−1 bar **vs caption SF** (host already
does). Cite vs RF for those controllers.

## Crossed host (not ours)

| Method | tail vs SF | subject | IQ | Dyn | flicker | Call |
|---|---:|---:|---:|---:|---:|---|
| sf_roll | +44% | 0.659 | 70.04 | **1** | 0.983 | **NO** |
| rf_chunk | +121% | 0.673 | **66.84** | **1** | **0.975** | **NO** H1-like |

## Status

Caption official N=32 is complete. Queue empty. No WAVE=2. No
AdaSteer N=32. No I2V. No TTC.
