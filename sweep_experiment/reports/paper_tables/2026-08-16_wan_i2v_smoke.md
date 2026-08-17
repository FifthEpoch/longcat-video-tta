# Wan I2V NOTTA smoke — 2026-08-16

**Source:** job `15880611` (`i2v_notta_smoke`, gh118, COMPLETED 0:0 in 2:55).
**Runner:** `wan_experiment/scripts/run_i2v_continuation.py` @ `65ba50c`
(autograd off + `inference_mode`).
**Not paper-grade quality metrics.** Timing + infra pass only. Regenerable
from `wan_experiment/results/i2v_notta_smoke/h5s_shard0/summary.json`.

## Generate timing (5 s requested → 85 px @ 16 fps, 480×832)

| Clip | Prompt | Frames | Generate+write (s) | mp4 bytes | s / generated second |
|---|---|---|---|---|---|
| 000 | A black and white abstract video featuring mesmerizing bubbles | 85 | 11.99 | 5,911,952 | 2.26 |
| 001 | A boiling pot cooking vegetables | 85 | 8.01 | 3,923,497 | 1.51 |

Job wall 175 s includes pipeline load (~77 s on the earlier 15879723 log).
Warm clip is ~8 s for ~5.3 s of video.

## Contrast with LongCat (do not mix into quality tables)

| Stack | Setting | Wall per video |
|---|---|---|
| Wan 1.3B + Self-Forcing DMD | NOTTA I2V, 5 s | ~8–12 s generate (this smoke) |
| LongCat 13.6B | BoN k=4 × 12 chunks (~60 s) | ~26,650 s (~7.4 h) |

Wan is the method stack. LongCat stays the 13B saturation audit.

## First-frame fidelity vs cond jpg (uint8 MAE)

| Clip | mae_vs_cond | Verdict |
|---|---|---|
| 000 bubbles | 5.56 | I2V (VAE-roundtrip). Noise would be ~80–120. |
| 001 pot | 3.71 | I2V |

## Pass / remaining gate

- `n_ok == 2`, multi-MB mp4s, 85 frames: **pass**
- First-frame fidelity vs cond image: **pass**
- Next series: `i2v_notta_16v` via `wan_experiment/sbatch/submit_i2v_notta16.sh`
