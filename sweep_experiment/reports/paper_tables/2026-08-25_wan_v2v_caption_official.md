# Caption official VBench — almost complete (2026-08-25 19:08)

Prompt = `metadata_csv`. Cite **medians**. Full-clip VBench n=32.
Locked bars vs caption SF **0.700 / 71.54 / 0 / 0.989**: tail ↑,
IQ ≥ 70.54, subject ≥ 0.680. RF-hosted rows also vs RF host
**0.694 / 70.22 / 0 / 0.985**.

Jobs: leftover VBench **16350479** CANCELLED by 0 at 2h17 (same
preempt pattern as 330) after writing Always + RF except
`rf_sink`. appear **16350480** COMPLETED 0:0 32m **32/32**.
Prefix VBench **16350481** COMPLETED 0:0 66m **n=32**.

## SF-hosted (the claim)

| Method | tail | vs SF | W/L | subject | IQ | Dyn | flicker | Call |
|---|---:|---:|---|---:|---:|---:|---:|---|
| notta (SF) | 0.01164 | — | — | **0.700** | **71.54** | 0 | 0.989 | baseline |
| rolling_notta | 0.01423 | +22% | 23/9 | 0.694 | **70.22** | 0 | 0.985 | host; IQ fail vs SF |
| sf_rewind | 0.01262 | +8% | 23/5/4 | 0.698 | 70.89 | 0 | 0.988 | **HOLD** small |
| sf_sick_search | 0.01164 | +0% | 19/4/9 | 0.697 | 71.54 | 0 | 0.988 | **NO** |
| sf_pseudo | 0.01492 | +28% | 23/0/9 | 0.701 | 71.66 | **0** | 0.985 | **HOLD tail.** Not Dyn |
| sf_always_search | 0.01623 | +39% | 30/2/0 | 0.687 | 71.16 | 0 | 0.984 | **HOLD** ablation. Letter holds |
| sf_sink | 0.01907 | +64% | 31/1/0 | **0.672** | 70.89 | 0 | 0.982 | **NO** subject bar |
| seed_bon | 0.00954 | **−18%** | 11/21 | **0.746** | 70.54 | 0 | 0.990 | **NO** motion. Identity damper |
| live_bon | 0.01187 | +2% | 6/5/21 | 0.723 | 71.43 | 0 | 0.989 | **NO** (21 ties) |
| appear_bon | 0.01117 | −4% | 13/19 | 0.723 | 71.23 | 0 | 0.989 | **NO** tail |

Always vs Pseudo: more tail (+39% vs +28%), subject 0.687 vs
0.701, IQ 71.16 vs 71.66. Hold-out is a mild identity/IQ brake.
Both Dyn 0. Stem Pseudo Dyn 0.50 **did not copy**.

seed_bon IQ −1.00 is on the locked line. Subject +0.046 is the
prefix-match identity bump.

## RF-hosted (vs that host)

| Method | tail vs RF | W/L vs RF | subject | IQ | Dyn | flicker | Call |
|---|---:|---|---:|---:|---:|---:|---|
| rolling_notta | — | — | 0.694 | 70.22 | 0 | 0.985 | host |
| rf_rewind | +6% | 16/9/7 | 0.692 | 70.32 | 0 | 0.984 | **HOLD** small |
| rf_sick_search | −1% | 12/11/9 | 0.695 | 70.16 | 0 | 0.985 | **NO** |
| rf_pseudo | +8% | 9/3/20 | 0.701 | 70.22 | 0 | 0.984 | **NO** (20 exact host) |
| rf_always_search | +25% | 23/9/0 | 0.695 | 70.24 | 0 | 0.983 | tail up; VBench ≈ host |
| rf_sink | +42% | 29/3/0 | — | — | — | — | **VBench missing** |

Every RF row fails the IQ−1 bar **vs caption SF** (host already
does). Cite vs RF, not vs SF, for those controllers.

## Crossed host (not ours)

| Method | tail vs SF | subject | IQ | Dyn | flicker | Call |
|---|---:|---:|---:|---:|---:|---|
| sf_roll | +44% | 0.659 | 70.04 | **1** | 0.983 | **NO** |
| rf_chunk | +121% | 0.673 | **66.84** | **1** | **0.975** | **NO** H1-like |

## Leftover

`rf_sink` official dims only. 479 died on the last dir.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
WAVE=rfsink bash wan_experiment/sbatch/submit_v2v_caption_rerun.sh
squeue -u $USER
```
