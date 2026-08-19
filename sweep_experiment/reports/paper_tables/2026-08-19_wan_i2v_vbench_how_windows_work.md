# How our VBench++ windows are computed (2026-08-19)

**Why this note exists:** 0–5 s vs 25–30 s looks almost flat on
subject / smoothness / flicker / dynamic median. That is the metric,
not a missing bug.

## Procedure (job 16009916)

For each method’s 32 mp4s, `score_i2v_vbench.py`:

1. Decode the 30 s file.
2. Cut frames `[round(start·fps) : round(end·fps)]` (last window
   includes the leftover frame).
3. Re-encode that slice as its **own** mp4 under `vbench_clips/wSTART_END/`.
4. Call official `vbench.VBench.evaluate(..., mode="custom_input")` on
   that directory, one quality dim at a time.
5. Average (median / mean) over the 32 independent clip scores.

There is **no** comparison of the 25–30 s slice to the 0–5 s slice
inside VBench. Each window is a new ~5 s video.

## What each dim measures (Huang et al., CVPR 2024)

| Dim | Input | What “high” means | Freeze does |
|---|---|---|---|
| subject_consistency | DINO features of frames **in that clip** | Nearby frames look like the same subject | **Up** — last 5 s is more self-similar |
| background_consistency | CLIP frame embeddings in that clip | Scene does not jump inside the clip | **Up / flat** |
| aesthetic_quality | LAION aesthetic, **mean over frames** | Frames look pretty | **Down** if oversharpened / collapsed |
| imaging_quality | MUSIQ, **mean over frames** (≈0–100) | Frames look clean | **Down** if distortion; can **up** if sharper |
| motion_smoothness | AMT: drop frames, interpolate, match | Motion is interpolable | **Up** — a still is perfectly smooth |
| dynamic_degree | RAFT flow vs a threshold | **1** if the clip is “dynamic”, else **0** | Stays **0** if already still at 0–5 s |
| temporal_flickering | High-frequency temporal noise | No sparkle / strobing | **Up** — freeze does not flicker |

Aes is typically reported on a 0–1 scale (LAION/10). IQ is MUSIQ.

## Why the drop looks small

1. **Five of seven dims reward local stability.** A frozen 25–30 s
   window is a *better* 5 s video on those axes than a moving 0–5 s
   window.
2. **Dynamic is a coin flip per clip, then a population median.**
   8/32 vs 6/32 called dynamic (means 0.250 vs 0.188) is real but
   the median is 0 both times.
3. **Aes / IQ are the appearance dims, and they do move.** Do-nothing
   median aes 0.651→0.538 (−17% relative), IQ 72.87→68.14 (−6.5%).
4. **The long-range number is the full clip, not the windows.**
   Full 30 s subject **0.848** vs every 5 s window **0.93–0.97**.
   Windows never ask “is the person at t=28 s the one at t=0?”
5. **Handpicked drift is a different question.** First 1 s vs last 1 s
   of the *same* file (sharp +167%, motion −60%). VBench windows
   cannot see that by construction.

VBench-Long (scene-split + slow/fast consistency on the full
generation) is the field’s answer to (4). We have not run it on these
I2V-32 clips. The hybrid-32 **full clip** remains the official VBench++
cite.
