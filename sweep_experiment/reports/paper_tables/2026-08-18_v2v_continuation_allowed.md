# Correction — T2V was not agreed; V2V continuation is allowed

**Date:** 2026-08-18
**Rebuts:** the 2026-08-18 line that the *only* comparable next
experiment is T2V 128 MovieGen. That was a copy-able *comparison*
protocol for Relax Forcing–style tables. It was not a task lock.

The 15 August correction still stands: stay in visual continuation.
T2V-from-scratch is independent of the Wan 1.3B switch.

---

## What we actually stopped

We stopped **I2V-from-a-still** scale-up (32 or 200 photographs, no
motion in the condition). That is not the same as stopping
**video-to-video**.

## Three different visual tasks (do not collapse them)

| Task | Condition | Then | Published home | Is it “long horizon”? |
|---|---|---|---|---|
| I2V-from-still (our 32-clip run) | One photograph | Animate, then AR | CausVid zero-shot I2V; official VBench-I2V is **5 s / 81 frames** on 14B | Length can be 30 s; the *suite* is not the 30–60 s field table |
| **V2V prefix continuation** (our claim) | A **real video prefix** (many frames, with motion) | Generate the future | DFoT / History-Guided (ICML 2025); SEINE (ICLR 2024) prediction; LongCat’s own continuation; StreamingT2V’s AR-from-last-frames ablation | Yes — this *is* long-horizon generation with visual history |
| Streaming V2V *translation* | A full input video + edit text | SDEdit each chunk | CausVid Table 5 vs StreamV2V on DAVIS | Long in duration, but it is **editing**, not “what happens next” |
| T2V self-continuation | Text only | Keep generating from own KV | CausVid / Self-Forcing / Relax / Freq / SF++ MovieGen-128 + VBench-Long | Yes — the *most copied* 30–60 s table, not required for our claim |

## Why V2V continuation is okay — and a better match than T2V

Our scientific object is **exposure bias under visual re-conditioning**.
The gate, verifier, and any later TTC anchor all read a visual history.
T2V-from-scratch deletes that history.

A still is a weak history: the first second after a photograph is mild
motion, which is why the I2V-32 run regularized toward freeze. A real
prefix has identity, camera, and dynamics. That is the DFoT / SEINE /
LongCat continuation setting.

Published backing (continuation-shaped, not T2V-from-scratch):

- **CausVid** (Yin et al., CVPR 2025): same Wan 1.3B causal student does
  streaming **I2V and V2V** zero-shot. Their V2V *table* is translation
  (DAVIS + StreamV2V). Their I2V table is VBench-I2V, 6–10 s, by
  duplicating the first image into the first segment.
- **DFoT / History-Guided** (Song et al., ICML 2025): history frames →
  ~64-frame rollout, Kinetics-600 N=1024, headline **FVD** + VBench.
- **SEINE** (Chen et al., ICLR 2024): short-to-long via prediction /
  transition from visible frames; AR video prediction is an intended use.
- **StreamingT2V** (Henschel et al., CVPR 2025): main paper is T2V, but
  the diagnosis they compare against is naive AR **conditioned on the
  last frames of the previous chunk** — i.e. V2V-style continuation —
  which they say stagnates. That is our freeze observation.

Honest limit: there is **no** Relax Forcing–sized “128 videos × 30/60 s
× VBench-Long” *continuation* leaderboard. A 30 s V2V table is
defensible as a continuation paper. It will not drop into a T2V
MovieGen cell.

## What a V2V long-horizon bench would look like (not submitted)

- Model: Wan 1.3B + Self-Forcing (already switched).
- Condition: first **1–2 s of real video** (not a still). Sources we
  already have: Panda-70M / UCF prefixes, or Kinetics if we want a
  DFoT-style FVD second table.
- Generate: 30 s (60 s optional) AR from that prefix.
- Score: VBench quality 7 on the **full generated clip** (and FVD on
  any GT overlap). Same three methods: do-nothing | always-BoN | gated-BoN.
- Cite as continuation / prefix-conditioned long video, not as
  Relax Forcing T2V.

No job until the user picks this over (or in addition to) T2V MovieGen.
No TTC. I2V-from-still scale-up stays closed.
