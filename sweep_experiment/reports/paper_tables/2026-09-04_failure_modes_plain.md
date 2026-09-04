# Failure modes in plain language (2026-09-04)

This is not a submit. It is a read of what we actually did and
what the videos did, without lab shorthand. Canvas:
`canvases/failure-modes-plain.canvas.tsx`.

Pseudo-future Search stays **dropped** as the paper title.
Do not remake cite-128. No new GPU from this note.

---

## How to read the scores (what you would see)

We use two kinds of numbers. They answer different questions.

**Official Visual Benchmark (VBench)** scores the **whole 30 s
clip** (not just the ending). Higher is better unless noted.

| Name we used | What it is asking | Healthy on our caption Self Forcing | What a bad number looks like |
|---|---|---|---|
| **Imaging Quality** | Does this look like a clean photo, or like compression / blur / plastic paint? | about **71–72** | **50** = obviously wrecked. **18** = the picture is broken. |
| **Subject Consistency** | Is the same person / car / room still that thing at the end? | about **0.70** (N=32) or **0.666** (N=128) | Below **0.68** the identity slips (face morphs, object changes). |
| **Dynamic Degree** | A yes/no per clip: “does this video look dynamic?” We report **how many clips of N** said yes. | Self Forcing 128: **42 / 128 (32.8%)** | Median 0 means most clips are called static. **8 / 8 yes** on a wrecked method is usually jitter, not living motion. |
| **Temporal Flickering** | Do consecutive frames stay stable? | about **0.987–0.989** | **0.97** = visible jitter. We called that **twitch**. |
| **Aesthetic Quality** | A “pretty picture” score. | about **0.50** | Can go **up** while Imaging Quality goes **down** (prettier paint, worse photo). |

**Our own “tail motion”** is not VBench. It is the average
absolute pixel change in the **last ~5 s**. Higher = more
movement in the ending. Rolling’s ending moves more than Self
Forcing’s (+33% on cite-128) but **fewer** clips are called
dynamic (28.9% vs 32.8%). Those are different questions:
“did the last seconds wiggle?” vs “does the official checker
call the clip dynamic?”

**The long-horizon problem we were trying to fix.** On
image-to-video, 16 clips, do-nothing: the last second is
about **2.7× sharper** than the first second and has about
**half** the motion. The videos freeze and over-sharpen. That
is the drift. Video-to-video (real 2 s opening, then 30 s)
is the same family of death, with a real opening instead of
a still.

---

## 1. AdaSteer — we turned knobs inside the network

**What we did.** Leave the checkpoint frozen. Fit a small
offset on the **time embedding** (or on mid/late residual
blocks) using only the real 2 s opening. Hold that offset
for the whole 30 s. Three recipes: fit once; refit every
chunk; residual-block offset.

**What we hoped.** A gentle push that keeps the scene from
freezing, without training a new model.

**What happened (8 videos, real captions).** The fit
succeeded (`|δ|` ≈ 0.84–0.95 — it found a number). The
videos are unusable.

| Recipe | Imaging Quality | What that means |
|---|---:|---|
| Healthy Self Forcing | ~71 | Fine photograph |
| Fit once (`ada_fixed`) | **42.7** | Heavy damage |
| Refit every chunk (`ada_stream`) | **51.5** | Still damaged; ending moved a bit more (+11%) |
| Residual-block offset (`ada_resid`) | **17.8** | The picture is gone. Official “dynamic” said yes on **all 8** because the frames are chaotic, not because a camera is living. |

Subject Consistency also fell (0.62–0.68). This is not a
−0.6 dip on Imaging Quality. It is a collapse. The same
story as “adapt the weights at test on a distilled
autoregressive student”: the offset is in-distribution for
the 2 s opening and out-of-distribution for the 30 s
rollout.

**Closed.** Do not retune step count or learning rate as a
paper move.

---

## 2. Prefix-match / appearance pick — we rewarded looking like the opening

**What we did.** Draw 4 random seeds for the next piece.
Keep the seed whose later frames look **most like the real
2 s opening** (pixel / appearance / seam), not the seed
with more motion.

**What we hoped.** Stay on-scene *and* keep motion.

**What happened (caption N=32).** The picker prefers the
seed that **changes least**, because that is the cheapest
way to match the opening.

- `seed_bon` (prefix-match): ending motion **−18%** vs
  do-nothing. Subject Consistency **up** to 0.746 (the
  person stays the person because almost nothing happens).
- `appear_bon`: ending motion roughly flat / slightly down.
  Subject **+0.065**. Still not a motion method.
- Official Dynamic Degree stayed **0 / 32**.

We optimized for a still that resembles the first two
seconds. That is why I said “freezes motion.” You would
watch a clip that looks *more like* the opening and *moves
less*.

**Closed** as a motion method. Useful only as “identity
damper” in an appendix.

---

## 3. Freeze-score — we asked the 1.3B student to grade its own last second

**What we did.** After the model **locks** the next 21
latents (~5 s), we add a little noise, run one forward
pass, and measure “how far is the prediction from the
clean lock?” If this lock is 1.2× worse than the previous
lock, redraw it. Always-on twin: always draw a second seed
and keep the better score.

This is **not** the big Wan-14B teacher from training. It
is the same 1.3B student looking at itself. We could not
put the 14B teacher next to the ~39 GB Rolling cache on
one GPU.

**What we hoped.** Catch a dying lock the way training’s
teacher-minus-critic signal would.

**What happened (8 videos).**

- **Gated** Rolling and Self Forcing copies: the pixels are
  **identical** to doing nothing on all 8 clips. The alarm
  never went off. That is what “never redraws” / “identity”
  meant — not a metaphor. Same file, same numbers, including
  Self Forcing’s last-chunk drift 102.09.
- **Always-on** Rolling: ending motion +11%, but Imaging
  Quality **65.9** (healthy Rolling ~70.2). Loses 5 of 8
  clips vs the Rolling host.
- **Always-on** Self Forcing: ending motion **−7%**. Worse
  ending, no quality win.

The student does not know, with this recipe, that a lock
is “sick.” Drawing a second seed and keeping the better
self-score does not recover official quality.

**Closed.** Do not swap in Wan-14B on this N=8 as a rescue.

---

## 4. Mid-chunk rewrite — we tried to edit 0.7 s inside a 5 s piece

**What we did.** Self Forcing locks about 5 s after four
denoising steps. Inside that, we tried to rewrite the last
**3 latents (~0.7 s)**: mix the last step with the previous
step; redraw if sharpness/motion looked sick; lock the
first latent of the block so the person who entered does
not change; 10% nudge; pick the seed with more latent
travel.

**What we hoped.** Fix a freeze *before* the whole 5 s
commits, without changing who the clip is.

**What happened (8 videos).** Even the gentlest version
moved **who the clip is**.

| What we tried | Subject Consistency | Imaging Quality | What you would notice |
|---|---:|---:|---|
| Healthy Self Forcing (same 8) | **0.700** | **71.54** | Same person, clean photo |
| 10% nudge | 0.642 | 69.66 | Identity slip |
| “Only rewrite if sick” intra | **0.632** | 68.19 | Same numbers as always-rewrite. The sick detector did not spare healthy blocks. |
| Pick max latent travel + lock first frame | 0.627–0.640 | ~70 | More ending wiggle, flicker 0.981 — twitch neighborhood |
| Same ideas on Rolling | 0.656–0.659 | **66–67** | Identity closer; the photo looks worse |

Four denoising steps means there is no “small touch-up.”
A 10% blend is still another drawing. That is why subject
fell 0.05–0.07. We pre-registered: keep subject ≥ 0.68 and
Imaging Quality ≥ 70.5. All 14 arms missed subject 0.68.

**Closed.** Do not loosen those bars.

---

## 5. Crossed host and mix — we married the wrong sampler to the checkpoint

**What we did.** Self Forcing’s weights + Rolling’s sliding
window, or Rolling’s weights + Self Forcing’s chunk loop.
Later: stay on the native sampler, and only **switch** after
a lock that looked frozen (mix). Always-on mix: switch every
span after the first.

**What we hoped.** Rolling’s cheap roll plus Self Forcing’s
motion, without training.

**What happened.**

On 32 videos (stem-era host-split; the twitch signature
survived caption replay): both crosses pushed ending motion
to **0.028** (Rolling native was 0.018). Official Dynamic
Degree went to **32 / 32**. Flicker fell to **0.972**.
Watch 0001 / 0004 / 0007 / 0025: it is not a living camera.
It is jitter.

On caption 8 (mix): always-on Rolling→chunk is the same
signature — Dynamic Degree **8 / 8**, flicker **0.978**,
Imaging Quality 67.7. Gated mix moved the ending (+26%)
and still failed Imaging Quality / subject vs the N=32
Rolling host.

The checkpoint and the sampler were trained as a pair.
Using the other paper’s unroll is not a free upgrade. That
is also why leftover ρ and an inference-only timestep list
kill Imaging Quality: training scored **whole videos**
produced by **this** unroll, not a new one.

**Closed.** Do not scale mix.

---

## 6. Noise-list and cache-write edits — the knob is live, the photo dies

These are different implementations of one idea: “make the
next second noisier / cleaner / more aware of the future,
without a new student.”

| Name | What we changed | Ending motion vs Rolling first-8 | Imaging Quality | What you would notice |
|---|---|---|---:|---|
| Leftover ρ (more/less noise on later blocks) | Scale the injected noise | **+61% to +97%** | **64–68** (host 70.2) | More wiggle, worse photo. Same on real captions as on stem prompts. |
| Linger-high / dump-early list | Spend the same 5 steps differently | −10% / +39% | **66.3 / 68.1** | Linger does not wake the ending. Dump wakes it and still paints. |
| Context noise = 50 | Write history into the cache a little dirty (clean is 0) | +6% Rolling, +11% Self Forcing | 67.6 / 69.2 | Self Forcing video **0004** ending motion **0.190** — an explosion, not a scene. |
| FIFO lookahead | Extra pass on the noisier half of Rolling’s window | **+21%** | **68.2** | More ending motion; still a worse photo than native Rolling. Sick-only FIFO ≈ did nothing. |

Aesthetic Quality sometimes **rose** while Imaging Quality
fell. That is “prettier paint, worse photograph.”

**Closed.** Do not start 8-GPU distillation just to retry
these lists. The inference-only change is the evidence that
a new list needs a new student — which is territory A, not
a leftover-ρ rescue.

---

## 7. Extra sink — pinning the opening harder

**What we did.** Rolling already keeps the **first 3 latents**
in the cache and re-applies rotary positions at read time.
We tried (a) replay only the opening plus a short window,
(b) set `sink_size` larger, (c) LongLive’s trained sink on
a student that was not long-tuned.

**What happened.**

- Replay without Rolling’s re-index: **same pixels** as
  do-nothing. The pin never reached attention.
- Extra sink on Self Forcing: ending motion **+72%**,
  subject **0.672** (identity on the line), flicker 0.977.
- LongLive `sink_size=9` (whole opening pinned): Imaging
  Quality failed. Their sink is dead until they train long.
- Native Rolling already has the paper sink. `apply_sink_size`
  on official Rolling does not change the kernel.

**Closed** as a test-time add-on. Do not retune sink on the
cite hosts.

---

## 8. CachedSearch — a cheaper Always-search that was not cheaper

**What we did.** Snapshot the key-value cache on the CPU so
extra seeds do not replay the whole history.

**What we hoped.** Same pixels as Always-search / Pseudo, less
wall time (the CachedSearch paper’s 37% cut).

**What happened (8 videos).** Tails **matched** the non-cached
twins to six decimals (same videos). Wall time went **up**
(389 s vs 360 s for Pseudo; 393 vs 349 for Always). Copying
a ~39 GB cache to the CPU is not a cheapen on this 4-step
student.

Re-checking the hold-out before every chunk still fired
(the gate is alive) and did not lift quality.

**Closed** as a cheapen. The gate is not the paper.

---

## 9. Our handcrafted “sick” score vs official VBench

**What we did (early image-to-video).** Score the last
second with a homemade mix: too sharp, too still, color
drift. Use that to pick among seeds or to decide whether
to search.

**What we hoped.** That number tracks official quality, so
steering with it improves the paper table.

**What happened (32 image-to-video clips).** On the last
5 seconds, official Imaging Quality was **best for
do-nothing (68.2)** and **worse for always-search (66.4)**.
Our score **punishes** a sharpness change. Official Imaging
Quality **rewards** sharpness. The correlation was about
**+0.23 to +0.33** — they barely agree, and they agree in
the direction of “sharper is better,” which is the opposite
of “don’t over-sharpen.” Full 30 s official VBench was a
**tie**. Dynamic Degree median was **0** for all three
methods.

That is what “anti-aligned” meant: we were steering with a
compass pointed at a different mountain than the official
judge. Later we locked the paper table to **full-clip
VBench** and Dynamic Degree as **percent of clips**, not
this homemade last-second score.

Prefix-match (§2) and freeze-score (§3) are the same class:
the proxy we can compute quickly does not pick the video
VBench wants.

---

## What this is not saying

It is not saying “nothing works.” **Drawing several seeds
and keeping one** (Always-search) is the one frozen-student
move that raised official Dynamic Degree on cite-128
(32.8% → **50.8%**) and **held** Imaging Quality (72.07 →
72.19). It is also a known idea (best-of-N / Video-T1 /
CachedSearch), and our skip-bit on top of it is a 13%
time cut that loses 4 dynamic clips. That is why
Pseudo-future Search is not the title.

It is saying: every time we **edited** the path (weights,
noise list, cache write, window, 0.7 s rewrite, other
paper’s sampler), the photo died or the motion became
jitter. Every time we **selected** among paths the student
already knew, official Imaging Quality survived.

---

## Do not do from this note

Remake cite-128. Scale AdaSteer, mix, FIFO, leftover ρ,
mid-chunk rewrite, extra sink, or freeze-score. Cheapen
Pseudo as the next paper. Launch 8-GPU distillation
tonight.
