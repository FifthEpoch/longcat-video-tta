# Self-Forcing lineage — what we can steal at inference (2026-08-21)

**Status:** literature map + brainstorm. Not a submit.

Our stack is Wan2.1-T2V-1.3B + Self-Forcing causal DMD. Locks: no TTC,
no I2V-32 scale-up. V2V 30 s, real prefix.

## The family (newest last)

| Paper | Venue / date | What it actually adds | Train or infer? |
|---|---|---|---|
| Diffusion Forcing | 2024 | Per-frame independent noise | train |
| CausVid | CVPR 2025 | Causal student + DMD from bidirectional Wan teacher | train |
| Self-Forcing | arXiv 2506.08009 | Train on **self-rollout + KV cache** (fixes CausVid’s train/test mismatch). Rolling KV at infer. Still degrades past ~5 s. **This is our student.** | train + infer KV |
| History Guidance | ICML 2025 | CFG on *how much history* (DFoT). Vanilla HG helps consistency, **kills dynamics**; frequency HG restores motion | train DFoT; CFG at infer |
| Self-Forcing++ | arXiv 2510.02283 | Long self-rollout, teacher DMD on short windows of the student’s own long video. Optional GRPO on optical flow. **No sink** (they contrast this with RF/LongLive). Minute-scale. | train |
| Rolling Forcing | ICLR 2026 / 2509.25161 | **Rolling-window joint denoise** (progressive noise, bidirectional inside the window) + **attention sink + dynamic RoPE** on the first frames | train; sink+rerope at infer |
| LongLive | NVIDIA, 2509.22622 | Same 1.3B Wan. **Frame sink + short window**, **KV-recache** on prompt switch, streaming long tuning. Paper: sinks **do nothing** until long-rollout collapse is trained away. Weights public. | train; sink/recache at infer |
| LongLive-2.0 | 2026 | Multi-shot sink (global + per-shot) | infer infra |
| CachedSearch | 2026 | Cheap cached explore, full-commit winner | infer only |

## What we already tested in this family

- Vanilla SF rolling KV = our `notta`.
- Replay-prefix sink **without rerope** = bit-match notta. Agrees with LongLive: sink is dead on a student that was not long-tuned.
- History dropout search ≈ HG without CFG. Small bump on seed_bon, not a policy (`tail_hist` ≈ notta).
- `cached_bon` = seed_bon quality. CachedSearch is how we would *pay* for a picker that works, not a new picker.
- CFG / shift = no pixel motion on this DMD student.

## Ideas with potential, ranked for *our* paper

### 1. Swap the student (highest EV, still no TTC)

Run the **same V2V protocol** (9-latent Panda prefix → 30 s) as `notta` on **LongLive 1.3B** and/or **Rolling Forcing**.

Question: is 30 s Dyn collapse a *Self-Forcing DMD* fact, or a *causal 1.3B* fact?

- If LongLive notta already keeps motion, the paper’s host is the wrong student and a controller will not save SF-vanilla.
- If LongLive still dies, identity-control / `live_bon` is student-agnostic.

N=8, no search. Their inference code, not our replay hack. This is the one LongLive-shaped test that can actually move pixels (sink is trained-in).

### 2. `live_bon` on the current student (our result, SF-compatible)

Search **only if the real prefix is moving** (`prefix_motion ≥ ~0.012`); else notta.

This is the SF-lineage sentence “do not teacher-force a still context” at test time. Quiet-bon inverted. N=8, same 8 videos.

### 3. Prefix as LongLive KV-recache (inference, needs their kernel)

LongLive recaches KV when the *text* prompt switches so the new condition is actually seen. Our analogue: **recache the real prefix tokens as the condition** at every later chunk, with their rerope.

Not “replay prefix at RoPE 0 and hope local attn looks.” That already failed.

Only worth it **on the LongLive checkpoint**, not on vanilla SF (`sink_size_t=0`).

### 4. Rolling-window denoise *if* we take the RF checkpoint

RF’s claim is local error is corrected *before commit* because several frames share a window at staggered noise. That is a different sampler, not a seed search. Do not fake mixed-noise windows on vanilla SF (CFG-class no-op risk).

### 5. CachedSearch as cost, not quality

If (2) or a LongLive+search combo ever wins, pay k=4 with our existing KV snapshot. Already verified bit-match.

## Do not spend GPUs on

- Training SF++ / GRPO optical-flow (new weights; optical-flow reward is smoothness, our failure is Dyn=0).
- History-CFG on this DMD student (CFG probe was dead).
- Another vanilla-SF sink replay.
- hist_drop-32 / quiet_bon VBench / 0.85–1.15 band without a live-prefix gate.

## Suggested order if we keep generating

1. LongLive (or RF) V2V notta N=8 — is the host broken?
2. `live_bon` N=8 on current SF — is the gate right?
3. Only then: LongLive + prefix recache, or LongLive + live_bon.

Two different bets: **better student** vs **better gate on this student**. The literature says long-horizon quality in this family came from training (SF++, RF, LongLive), not from test-time seed search. Our N=32 VBench agrees: seed search is identity control.
