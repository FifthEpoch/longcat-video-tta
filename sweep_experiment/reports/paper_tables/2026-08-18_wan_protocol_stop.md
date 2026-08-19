# STOP — do not scale the I2V-32 setup (2026-08-18)

**Lock:** User asked to verify freeze / search / gating on a larger
**industry-standard** sample, but only if our basic setup is what
recent similar papers report. It is not. **No I2V-32 (or I2V-200)
scale-up job.** No TTC.

The 32-clip hybrid VBench result stays as a **discovery** scorecard
on our protocol. It is not a standard long-horizon result.

---

## What is common for long horizon (2025–2026)

5 s is only the short sanity check. The long-horizon table is
**30–60 s+ T2V** on Wan2.1-T2V-1.3B + a Self-Forcing-style student.

| Paper | Task | Length | Data | Metric |
|---|---|---|---|---|
| Self-Forcing (NeurIPS 2025) | T2V | 5 s main; 30 s shown as extrapolation failure | MovieGen / VBench | VBench |
| Self-Forcing++ | T2V | 50–100 s | 128 MovieGen, Qwen-refined | VBench-Long |
| Relax Forcing | T2V AR | **30 s and 60 s** | 128 MovieGen, Qwen-refined | VBench-Long |
| FreqForcing | T2V | 60 s / 120 s | first 128 MovieGen, Qwen-refined | VBench-Long |
| Self Gradient Forcing | T2V | 60 s / 240 s | VBench-Long + 128 MovieGen | VBench-Long |
| Rolling Forcing / LongLive | T2V streaming | multi-minute | same cluster | VBench-Long |
| History-Guided / DFoT | visual-prefix prediction | ~64 frames | Kinetics-600 N=1024 | FVD + VBench |

Shared recipe: text start → AR from own KV cache → **N ≈ 128**
MovieGen (Qwen-extended) → **VBench-Long** on the **full** clip.

## Where we differ

| Knob | Our I2V-32 run | Field long-horizon |
|---|---|---|
| Length | 30 s | Fine (Relax Forcing grid) |
| Model | Wan 1.3B + Self-Forcing DMD | Fine |
| Task | I2V from a still | T2V, then self-continue |
| N / suite | 32 VBench-I2V stills, `custom_input` | 128 MovieGen, VBench-Long |
| “Continuation” | animate an external still | keep generating after a text-started prefix |

Official VBench-I2V is a different protocol again: **5 s / 81 frames**
on Wan-I2V-14B, hundreds of images.

Scaling I2V stills would tighten *our* error bars. It would not
compare to Relax Forcing / Self-Forcing++ / FreqForcing.

## What is allowed next

Comparable verify (not submitted): see
[`2026-08-18_wan_t2v_vbenchlong_128_spec.md`](2026-08-18_wan_t2v_vbenchlong_128_spec.md).

Innovation directions (no weights): see
[`2026-08-18_wan_nonweight_next.md`](2026-08-18_wan_nonweight_next.md).
