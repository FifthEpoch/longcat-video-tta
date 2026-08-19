# Onto the field long-horizon testbed

**Weekly briefing · 15–18 August 2026**
**Audience:** Monday recap / PI update
**Scope:** why we switched model and dataset, with citations, plus the
concepts that define long-horizon generation. Not the BoN / gating
ablations and not the next-method brainstorm.

Interactive walk-through (open beside chat):
`~/.cursor/projects/Users-macrohard-Desktop-longcat-video-tta/canvases/week-switch-briefing.canvas.tsx`

Literature memo this talk compresses:
[`2026-08-15_longhorizon_field_standard.md`](2026-08-15_longhorizon_field_standard.md)

---

## Slide 1 — Title

Last week we moved the experimental stack onto the field’s long-horizon
testbed.

| From | To |
|---|---|
| LongCat-Video 13.6B | Wan2.1-T2V-1.3B + Self-Forcing causal DMD |
| Panda-70M / UCF short-clip continuation | Prompt-suite eval (MovieGen + VBench-Long) |
| PSNR / SSIM / LPIPS as headline | VBench quality 7 on the **full** clip |

This talk is the setup. Method ideas come after.

---

## Slide 2 — Agenda

1. **Model.** Why LongCat 13.6B is the wrong workhorse now, and what
   published streaming papers already run on Wan 1.3B.
2. **Dataset.** Why Panda / UCF are not a long-horizon bench, and why
   MovieGen-128 + VBench-Long is the copy-able protocol.
3. **Concepts.** Causal AR, KV cache, exposure bias, three meanings of
   “continuation,” the freeze–identity trade, and how VBench-Long scores
   a minute-scale clip.

---

## Slide 3 — Where we started (through 14 August)

The paper claim is a drift-gated, GT-free test-time controller. Until
last week the stack was LongCat-Video 13.6B on Panda-70M / UCF-101 short
clips.

| Knob | Stack as of 14 Aug | Problem |
|---|---|---|
| Model | LongCat 13.6B, 50-step, native 13/80 AR | Too expensive for the N long-horizon needs |
| Task | 14→14 then native continuation on Panda | In-domain short clips; GT dies after 1–2 chunks |
| N | 8 videos × 12 chunks at ~60 s | Gating sample only |
| Metrics | PSNR / SSIM / LPIPS + GT-free drift | PSNR is not the field headline for open-ended long gen |

LongCat stays in the paper as a **saturated-13B audit**. No more LongCat
TTC.

---

## Slide 4 — Three facts that decided the model switch

Recorded 15 August. Inclusion rule: venue papers (CVPR / NeurIPS / ICML /
ICLR 2024–2025) plus the official suites they report. See the 15 August
memo.

**(a) The streaming testbed is 1.3B-class, not 13B.**
CausVid (Yin et al., CVPR 2025) and Self-Forcing (Huang et al., NeurIPS
2025 Spotlight) both start from **Wan2.1-T2V-1.3B**. Pyramid Flow is 2B.
The only 5B-class long-horizon venue paper (One-Minute TTT, Dalal et al.,
CVPR 2025) still calls 5B a capability bottleneck. LongCat 13.6B is an
outlier: too big to get N, and pretrained to resist the drift we need as
headroom.

**(b) Headroom exists on the small causal student.**
Self-Forcing trains at 5 s and explicitly shows quality collapse when
extrapolating to 10–30 s. That collapse is the phenomenon a test-time
controller can attack. LongCat’s short-horizon in-domain table was
already saturated (14→14).

**(c) We keep the method; we change the workhorse.**
Best-of-N / gating is backbone-agnostic. Switching the model does not
change the scientific claim. It changes whether we can run N=128 at
30–60 s and cite the same numbers other papers cite.

---

## Slide 5 — What published work uses Wan 1.3B

Wan2.1-T2V-1.3B (Wan Team, 2025) is the teacher. The long-horizon
workhorse is a causal few-step student distilled from it.

| Paper | Venue | Model | Horizon on the table |
|---|---|---|---|
| CausVid (Yin et al.) | **CVPR 2025** | Wan2.1-T2V-1.3B → 4-step causal AR | 5–10 s main; **30 s** long table |
| Self-Forcing (Huang et al.) | **NeurIPS 2025 Spotlight** | Same 1.3B causal family | 5 s main; 10–30 s as **extrapolation failure** |
| Pyramid Flow (Jin et al.) | **ICLR 2025** | Own 2B MM-DiT | 5–10 s (same eval culture, not Wan) |
| Self-Forcing++ / Relax Forcing / FreqForcing | 2026 follow-ons | Wan 1.3B + Self-Forcing-style student | **30 / 60 / 120 s**, VBench-Long |

Pyramid Flow is included because it is the other 2025 streaming venue
paper — it confirms **VBench**, not Wan. 2026 follow-ons were not in the
15 August inclusion rule; they confirm the same 1.3B + 30–60 s table we
would be compared against now.

CausVid also reports streaming **I2V and V2V** zero-shot on the same
1.3B student — that is why I2V is a legitimate *discovery* path, not
why it is the long-horizon paper table.

---

## Slide 6 — Our switch

**From:** LongCat-Video 13.6B. Native 13-cond / 80-gen window. ~110 min
per 60 s video at 50 steps. N=8 was the practical ceiling.

**To:** Wan2.1-T2V-1.3B + Self-Forcing causal DMD. Public weights.
KV-cache AR. Few-step student. Same family as CausVid / Self-Forcing.
Makes N=128 at 30 s a real experiment.

**What we did not switch to**

- CogVideoX-5B (One-Minute TTT only; 4× our size budget).
- VideoCrafter2 (FreeNoise / FIFO-Diffusion era; superseded as the
  streaming testbed).
- Bidirectional Wan-I2V-14B (official VBench-I2V leaderboard model —
  different protocol: 5 s / 81 frames).

---

## Slide 7 — Why Panda-70M / UCF are not a long-horizon bench

Those sets were the right audit for short-horizon TTA (AdaSteer vs LoRA
at 28–76 frames). They are the wrong eval for minute-scale generation.

**What Panda / UCF give us.** A unique pixel future for a few seconds.
That is why PSNR / SSIM / LPIPS / FVD were honest at 28–76 frames. After
one or two chunks there is no GT tail. Our 60 s LPIPS cell was
GT-limited by construction.

**What the field does instead.** Open-ended generation is scored against
a **prompt suite**, not against a held-out pixel future. Training-source
clips in that literature are typically 3–10 s (MixKit, WebVid, OpenVid,
Kinetics). Nobody uses Panda short-clip continuation as the long-horizon
test.

PSNR is not banned. It is the wrong *headline*. It appears when a
single GT future exists (prediction, robot, DFoT on Kinetics).

---

## Slide 8 — What published long-horizon papers evaluate on

| Suite | Who uses it | N / length | What it measures |
|---|---|---|---|
| **MovieGen first 128 prompts** (Polyak et al., Meta Movie Gen, 2024) | CausVid long table; Self-Forcing; Relax / Freq / SF++ (often Qwen-refined) | N=128 · 5–120 s | Open-ended T2V; the copy-able long-horizon set |
| **VBench / VBench-Long** (Huang et al., CVPR 2024 + official Long extension) | CausVid, Self-Forcing, Pyramid Flow, 2026 follow-ons | 946 short prompts; Long = full clip, scene-split + slow/fast | Quality 7 + semantic dims; GT-free; human-aligned |
| **Kinetics-600 64-frame rollouts** | History-Guided / DFoT (Song et al., ICML 2025) | N=1024 · ~64 frames | Visual-prefix *prediction*; headline = FVD |
| **VBench-I2V** (VBench++ official) | Official I2V leaderboard (Wan-I2V-14B) | Hundreds of stills · **5 s / 81 frames** | Conditioned short I2V — not the 30 s T2V long table |

Qwen-refined MovieGen: Self-Forcing++ / Relax Forcing / FreqForcing
extend the 128 prompts with Qwen2.5 so 30–60 s clips have enough
narrative to keep generating.

---

## Slide 9 — Honest accounting: what we actually ran this week

On 15 August we correctly switched the model and correctly refused to
couple that to T2V-from-scratch (our claim is exposure bias under
*visual* re-conditioning). We then used **VBench-I2V stills** as a
continuation / I2V discovery set. Valid stress test. Not the field
long-horizon table.

| Knob | Field long-horizon 2025–26 | Our 15–18 Aug run |
|---|---|---|
| Model | Wan 1.3B + Self-Forcing-style student | Same — this switch is correct |
| Task | T2V, then self-continue from own KV | I2V from an external still, then AR |
| Data | 128 MovieGen prompts, often Qwen-refined | 32 VBench-I2V images |
| Metric | VBench-Long on the full clip | `custom_input` VBench on the full 30 s |
| Horizon | 30 / 60 / 120 s on the paper table | 30 s (length is fine) |

**Locked 18 August.** Do not scale I2V-32. The comparable verify is T2V,
128 MovieGen, VBench-Long — spec ready, not submitted.

---

## Slide 10 — Concepts: causal AR, KV cache, exposure bias

**Bidirectional vs causal.** A bidirectional video diffuser denoises the
whole clip at once and is stuck at a fixed length. A causal / AR student
predicts the next few frames from the past and can, in principle, run
forever.

**KV cache.** Each new chunk attends to keys and values already computed
for earlier frames. That is how Self-Forcing / CausVid stream in real
time. It is also how early errors stay in the context forever.

**Exposure bias.** Train (or the teacher) sees clean history. At test
time the model conditions on its own samples. Small mistakes compound.
Self-Forcing’s 10–30 s collapse is this, made visible.

Self-Forcing is named for training the student on its own rollouts
(distribution-matching distillation) so test-time AR is less foreign. It
reduces exposure bias. It does not remove it past the 5 s train horizon.

---

## Slide 11 — Concepts: three different “continuations”

We used “continuation” for LongCat and I2V. The field uses it for a
different object. Mixing them is how the 15 August T2V recommendation
and the later I2V correction happened.

| Name | Start condition | Then | Published home |
|---|---|---|---|
| **Self-continuation** (field long-horizon) | Text prompt | Keep generating from own KV cache | CausVid, Self-Forcing, Relax / Freq / SF++ |
| **I2V-from-still** (our 32-clip run) | External photograph | Animate, then AR | CausVid demo / zero-shot I2V; not the 30 s table |
| **Visual-prefix prediction** | Real history frames | Predict ~64 frames | DFoT / History-Guided, ICML 2025, Kinetics FVD |

**The freeze–identity trade.** Long AR either drifts (identity /
background break) or stagnates (standstill). StreamingT2V (CVPR 2025)
states this diagnosis. History Guidance (DFoT): vanilla history-CFG
helps identity and kills dynamics; they add fractional / frequency
guidance to put motion back. Rolling / Relax Forcing keep first-frame KV
as an **attention sink** for the same reason.

---

## Slide 12 — Concepts: how the field scores a long video

**VBench quality 7** (Huang et al., CVPR 2024): subject consistency,
background consistency, temporal flickering, motion smoothness, dynamic
degree, aesthetic quality, imaging quality. GT-free. This is what
CausVid / Self-Forcing / Pyramid Flow report.

**VBench-Long:** scores the whole long video via scene-split plus slow
and fast clips, then aggregates. The official number is the **full
clip**. Cropping the last 5 s is a diagnostic, not the paper table.

**`dynamic_degree` is 0/1 per video.** RAFT asks “is this clip
dynamic?” Median 0 means most clips fail that test. Report the mean
(fraction dynamic) alongside the median. High motion smoothness plus
median-0 dynamic degree is the freeze signature, not a smoothness win.

Human pairwise / Elo still appears (CausVid, One-Minute TTT). FPS /
latency appears because these models are sold as streaming. Neither is
required to lock the testbed.

---

## Slide 13 — What we actually did, 15–18 August

| Day | Decision or result |
|---|---|
| 15 Aug | Switch to Wan 1.3B + Self-Forcing. Keep LongCat as 13B audit. Stay in I2V / continuation (do not force T2V-from-scratch). Start cluster setup. |
| 16 Aug | Healthcheck green. First I2V smoke (job 15880611). 16 videos at 5 s and 30 s. |
| 17 Aug | 30 s drift is real (sharp +167%, motion −60%). Chunked BoN runner. 16v then 32v hybrid three-way. |
| 18 Aug | Official VBench on the 32 stills: full-clip **tie**. Protocol check: I2V-32 is not the field table. Stop scale-up. Spec T2V 128 / VBench-Long (not submitted). |

Full-clip VBench on I2V-32 is a tie. `dynamic_degree` median is 0 for
all three methods. I2V-32 / I2V-200 scale-up is closed.

---

## Slide 14 — Where we are now

**Locked**

- Workhorse = Wan 1.3B + Self-Forcing DMD.
- Headline metrics = VBench quality 7 on the full clip.
- LongCat = saturated-13B audit only. No more LongCat TTC.
- Do not scale I2V-32. Do not add TTC yet.
- Do not claim N=32 VBench as a standard long-horizon result.

**Not yet launched**

Comparable verify: T2V, 128 MovieGen (Qwen-refined), 30 s (60 s
optional), VBench-Long, do-nothing | always-BoN | gated-BoN. New
generate series — the I2V chunked runner is not a flag flip.

**Next conversation, not this one.** Non-weight method ideas (motion
verifier, failure-gated CachedSearch, shift/CFG/sink search, prefix
backtrack) wait until this testbed exists.

---

## Citation list (speaker pocket)

Venue papers (15 August inclusion rule)

- Yin et al., **CausVid**, CVPR 2025. Wan2.1-T2V-1.3B causal student;
  VBench + MovieGen-128; 30 s long table; streaming I2V/V2V zero-shot.
- Huang et al., **Self-Forcing**, NeurIPS 2025 Spotlight. Same 1.3B
  family; 5 s train; 10–30 s extrapolation failure.
- Jin et al., **Pyramid Flow**, ICLR 2025. 2B; VBench / EvalCrafter.
- Song et al., **History-Guided Video Diffusion / DFoT**, ICML 2025.
  Kinetics-600 N=1024; 64-frame FVD; history-CFG freezes dynamics.
- Huang et al., **VBench**, CVPR 2024 + official VBench-Long / VBench++.
- Dalal et al., **One-Minute TTT**, CVPR 2025. CogVideoX-5B; human Elo;
  not our size class.
- Kim et al., **FIFO-Diffusion**, NeurIPS 2024; Qiu et al., **FreeNoise**,
  ICLR 2024. Training-free longer T2V on VideoCrafter-class models.
- Wan Team, **Wan2.1**, 2025. Teacher checkpoint.
- Polyak et al., **Movie Gen**, Meta 2024. Source of the 128-prompt bench.

2026 follow-ons (confirm the current long table; not the 15 August
inclusion rule)

- Self-Forcing++, Relax Forcing, FreqForcing, Rolling Forcing / LongLive,
  Self Gradient Forcing: Wan 1.3B + Self-Forcing-style student; 128
  MovieGen (Qwen-refined); VBench-Long; 30–120 s.

Streaming diagnosis

- StreamingT2V, CVPR 2025: long AR either breaks consistency or
  stagnates to a standstill.
