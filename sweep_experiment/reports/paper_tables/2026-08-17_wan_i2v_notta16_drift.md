# Wan I2V NOTTA drift — 16v, 5 s vs 30 s (2026-08-17)

**Source:** `wan_experiment/results/i2v_notta_16v/drift_head_tail.json`
**Scorer:** `wan_experiment/scripts/score_i2v_drift.py` @ `f32cc64`
(stream head/tail 1 s, skip cond frame 0).
**Signals:** LongCat GT-free set — sharpness (Laplacian var), colorfulness
(Hasler–Süsstrunk), contrast (luma std), temporal_motion (mean |Δframe|).
**Relative change:** (tail − head) / head. Cite **medians**; means are
outlier-pulled (004 eagle 30 s sharpness +889%).

## Population (N=16, same images/seed)

| Horizon | Frames | sharp mean / median | color mean / median | contrast mean / median | motion mean / median |
|---|---|---|---|---|---|
| 5 s | 85 | +0.342 / **+0.105** | +0.063 / **+0.094** | +0.103 / **+0.085** | −0.253 / **−0.135** |
| 30 s | 481 | +2.158 / **+1.668** | +0.867 / **+0.282** | −0.073 / **−0.050** | −0.484 / **−0.601** |

## Sign counts at 30 s

| Signal | # up | # down | Read |
|---|---|---|---|
| sharpness | 15 | 1 | HF / oversharpening. Only 003 highway is down. |
| colorfulness | 11 | 5 | Oversaturation, heavy tail (009 +361%, 011 +328%). |
| contrast | 5 | 11 | Mild fade; LongCat-like, small. |
| temporal_motion | 1 | 15 | Motion death. Only 002 flower is up. |

## What this unlocks

5 s ≈ LongCat short-horizon saturation (small median moves). 30 s is
real drift: sharpness roughly **2.7×**, motion about **half**. That is
controller headroom. Wan's long-horizon signature is **sharpen + freeze**,
not LongCat's sharpen + motion inflation.

Do not run clip-level gated-BoN/TTC at t=0 (incoming context is the cond
still → gate always skips → gated = NOTTA). Five-way must be **chunked**
on the 30 s rollout. See ANALYSIS_LOG 2026-08-17.
