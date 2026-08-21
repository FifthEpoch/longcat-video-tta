# Sampling-space hypotheses after the V2V N=8 bake-off

**Date:** 2026-08-20
**Status:** inference, not submitted. N=32 confirm is the quality lock.
These are the *next tricks* if that confirm holds — or cheap probes
that can ride alongside it.

## Facts we can actually stand on

1. **I2V-from-still:** four seeds did not unstick a freeze. Full-clip
   VBench was a tie. The reference window was the first second *after
   a photograph* — mild motion. Matching that reference = matching a
   near-still.

2. **V2V, real prefix:** the *same* pick score (two-sided deviation
   from the reference: sharpness, color, contrast, **motion**, plus
   seam) plus four seeds raised tail `|Δframe|` **+35%** and flipped
   VBench Dyn median 0→0.5. Subject held. IQ −0.60.

3. **Maximize `|Δframe|` lost.** Greedy per-chunk twitch made the 30 s
   tail *less* dynamic. Local flicker is not prefix motion.

4. **Backtrack-from-a-dead-tail lost** and cost IQ (−2.94). Rewinding
   *after* collapse resamples from a poisoned state.

5. **`shift` / CFG do not move pixels** on this DMD student. Those
   knobs are not a search axis until we find a real hook (or a
   different checkpoint).

6. Vanilla Self-Forcing is **`sink_size_t=0`**. NVIDIA ships a
   long-rollout preset `sink=5 + window=7 + rerope` specifically so
   later chunks do not forget the start.

## The main inference (H-match)

**seed_bon did not “maximize motion.” It matched the real prefix.**

The old composite is *lower = closer to the reference*. On V2V the
reference is 2 s of real video, which already has motion. A candidate
that freezes has a large `dev_temporal_motion` and loses. A candidate
that flickers away from prefix sharpness/color also loses. The picker
is “stay like this moving prefix.”

On I2V the reference was the first second after a still, so the same
rule said “stay like this near-still.” That is why the identical
method froze there and moved here.

`motion_bon` broke the match: it *rewards* exceeding prefix motion
(flicker, seams). The next chunk then conditions on a twitchy frame
and collapses.

If H-match is right, the paper move is **prefix-conditioned test-time
search**, not “best-of-N seeds” as a generic slogan.

## Hypotheses worth a GPU (sampling space only)

Ranked by expected gain × implement cost. No TTC / no new weights.

### 1. Prefix-match pick, one-sided on motion (cheap, tests H-match)

Keep k=4 seeds. Change the score:

- Appearance (sharp / color / contrast): two-sided vs **prefix**
- Motion: **hinge** — penalize only `motion < prefix_motion` (do not
  reward flicker above the prefix)
- Small seam penalty

If this beats seed_bon’s two-sided score at N=8, H-match is causal.
If it ties, the win was just “four seeds” and the score was a wash.
**This is the highest-leverage cheap test.** Same runner, new pick.

### 2. Late-only / failure-gated seed search (cheap, tests H-horizon)

Fact: first generated chunk on the probe still had motion 0.016;
the 30 s tail is where Dyn dies. Search only once incoming motion
falls below ~0.7× prefix motion (or only chunks 3–5).

If most of seed_bon’s gain is in the last two chunks, we get ~the
same quality at ~½ the 4× cost. Field language: Early Failure
Detection + our prefix-motion sensor.

### 3. Attention sink of the real prefix (medium, highest theory)

Pin the 9 prefix latent tokens in KV for every later chunk
(`sink=5` style). Do **not** slide them out of the local window.

Hypothesis: late freeze is “forgot the moving start,” not “wrong
seed.” A sink would move Dyn **without** 4× decode. If we cannot
do this without the NVIDIA rerope checkpoint, that checkpoint is
the wave-2 artifact — still sampling-space (KV layout), not TTC.

### 4. History-dropout search (medium, History Guidance without CFG)

CFG is dead. What HG actually does is a **weak history** vs **full
history** branch. We can search that without guidance:

- cand0: replay full committed prefix (current)
- cand1: replay only the last 3 latents (short history)
- cand2–3: seeds on full history

Pick with the prefix-match score. Tests whether *how much* visual
history we condition on is a live axis now that text-CFG is not.

Vanilla full-history CFG in the HG paper **kills** dynamics. Short
history might restore them; prefix-match should stop identity drift.

### 5. Checkpoint backtrack to the last *good* chunk (cheap-medium)

Our backtrack rewound a *dead* tail. Literature backtrack rewinds
to a **good** prefix.

Keep the last committed chunk whose motion ≥ 0.8× prefix. If the
new chunk collapses, restore that checkpoint and resample. Do not
resample from the collapsed latent.

This is Temporal Backtracking Search, not “search-while-sick.”

### 6. CachedSearch on the method that already works (efficiency)

Not a quality hypothesis. If N=32 confirms seed_bon, pay for k=4
only on the winner’s last 1–2 steps (or score a 1-step preview).
Paper-facing: we already have a quality win; this is how it ships
at 1.2× notta instead of 3×.

## What not to try next

- Another `|Δframe|` maximizer (already falsified)
- Dead-tail backtrack (already falsified)
- `shift` / CFG until a probe shows pixels move
- TTC / LoRA / activation delta (locked)
- I2V-32 scale-up (wrong condition)
- N=128 of motion_bon

## Suggested order after you say go

N=32 notta vs seed_bon is already the quality confirm (in flight or
about to be). In parallel or right after, the **innovative** probe I
would actually run is **(1) prefix-match hinge** on N=8, same 8
videos we already have as a paired control. One new method dir,
same prefixes, ~51 min. That tells us whether the win is “match the
moving prefix” or “any four seeds.”

Then (2) late-only search as an efficiency ablation of seed_bon.
Then (3) sink if (1) holds — that is the method that could beat
four seeds rather than retune them.
