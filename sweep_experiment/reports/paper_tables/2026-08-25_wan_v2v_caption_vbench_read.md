# Caption official VBench + leftover jobs (2026-08-25 16:36)

Prompt = `metadata_csv` on every harvested sidecar. Cite **medians**.
Official quality = full-clip VBench, n=32 unless noted.
Locked bars vs caption SF (0.700 / 71.54 / 0 / 0.989): tail ↑,
IQ ≥ 70.54, subject ≥ 0.680.

## Jobs

| Job | What | State |
|---|---|---|
| **16310330** | WAVE=1 VBench | **CANCELLED by 0** 2h18 (0:15). Wrote SF family only. |
| **16328464** | seed_bon | COMPLETED 0:0 3h26 **32/32** |
| **16328465** | live_bon | COMPLETED 0:0 1h52 **32/32** |
| **16328466** | appear_bon | **CANCELLED by 0** 3h51 **27/32** |
| **16328467** | Prefix VBench | CANCELLED 0:00 (afterok 466) |
| **16328612** | sf_roll | COMPLETED 0:0 26m **32/32** |
| **16328613** | rf_chunk | COMPLETED 0:0 61m **32/32** |
| **16328614** | Cross VBench | COMPLETED 0:0 50m **n=32** |

## Official caption VBench (written)

| Method | tail vs SF | subject | IQ | Dyn | flicker | IQ bar | subject bar | Call |
|---|---|---:|---:|---:|---:|---|---|---|
| notta (SF) | — | **0.700** | **71.54** | 0 | 0.989 | — | — | baseline |
| rolling_notta | +22% | 0.694 | 70.22 | 0 | 0.985 | **fail −1.32** | hold | host, not quality-better |
| sf_rewind | +8% | 0.698 | 70.89 | 0 | 0.988 | hold −0.65 | hold | **HOLD** small |
| sf_sick_search | +0% | 0.697 | 71.54 | 0 | 0.988 | hold | hold | **NO** |
| sf_pseudo | +28% | 0.701 | 71.66 | **0** | 0.985 | hold +0.12 | hold | **HOLD tail.** Dyn 0 — stem 0.50 **did not copy** |
| sf_sink | +64% | **0.672** | 70.89 | 0 | 0.982 | hold −0.65 | **fail −0.028** | **NO** letter |
| sf_always_search | +39% | — | — | — | — | — | — | VBench leftover |
| seed_bon | **−18%** | — | — | — | — | — | — | **NO** tail (damper) |
| live_bon | +2% | — | — | — | — | — | — | **NO** (21/32 ties) |
| appear_bon | −13% (n=27) | — | — | — | — | — | — | resume 27→32 |
| sf_roll | +44% | 0.659 | 70.04 | **1** | 0.983 | fail −1.50 | fail −0.041 | **NO** |
| rf_chunk | +121% | 0.673 | **66.84** | **1** | **0.975** | fail −4.70 | fail −0.027 | **NO** (H1-like) |

## Read

- **Caption Pseudo is not a Dyn method.** Tail +28% / 23/0/9 and
  quality hold, official dynamic degree **0**. The stem 0.50 was
  filename-prompt. Do not put 0→0.50 on the caption title.
- **Caption RF fails the IQ bar** vs caption SF. Host gap is tail
  only under real captions.
- **Sink** still the large tail mover; subject now **fails** the
  −0.02 bar (stem was on the line).
- **Prefix-match** damps again under captions (seed −18%). Live is
  a skip (21 ties). Appear cancelled at 27.
- **Crossed host** still a sampler swap: Dyn 1 + identity/IQ fail.
  `rf_chunk` flicker 0.975 is the H1 neighborhood. Not a controller.

## Leftover submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
WAVE=leftover bash wan_experiment/sbatch/submit_v2v_caption_rerun.sh
squeue -u $USER
```

Scores remaining WAVE=1 VBench (always + RF family; skip-existing
on written SF rows). Resumes `appear_bon` (skip 27 mp4s) then
Prefix VBench on seed/live/appear. Do not resubmit WAVE=1 generate.
Do not scale AdaSteer.
