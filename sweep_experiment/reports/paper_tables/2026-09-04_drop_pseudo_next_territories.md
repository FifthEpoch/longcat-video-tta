# Drop Pseudo-future Search; next territories from outcomes (2026-09-04)

User lock: Pseudo-future Search is **not** the paper title. A
prefix-hold-out gate that recovers most of Always-search at
13% mean-wall save is not a CVPR idea. Do not write “we
introduce gating.” Do not cheapen Pseudo as the next submit.

This file is the outcome atlas and the fork. No GPU tonight.
Do not remake cite-128. No I2V. No TTC. No LoRA.

Canvas: `canvases/drop-pseudo-next.canvas.tsx`.

---

## The law the data already taught

Both Self Forcing and Rolling Forcing **are** “unroll inference
with a KV cache, then score the whole self-made video”
(Distribution Matching Distillation). The student only knows
how to emit what it was distilled to emit.

**Selection among those futures is safe. Editing the trajectory,
the list, the KV write, or the host sampler is not.**

Cite-128 (n=128, `metadata_csv`, full-clip VBench; Dyn = percent
of clips):

| | Self Forcing | Rolling | Pseudo | Always-search |
|---|---:|---:|---:|---:|
| Dyn% | 32.8 (42) | 28.9 (37) | 47.7 (61) | **50.8 (65)** |
| Imaging Quality | 72.07 | 71.52 | **72.38** | 72.19 |
| Subject | 0.666 | **0.685** | 0.660 | 0.661 |
| mean s / clip | 108 | **47** | 294 | 354 |
| PSNR / FVD | **9.25 / 410** | 7.98 / 436 | 9.22 / **405** | 9.21 / 425 |

Search is the only frozen-student move that lifted Dyn% and
held Imaging Quality. It is also occupied (Video-T1, CachedSearch,
LatSearch, SDVG, Early Failure Detection) and our gate is a
13% discount that loses 4 Dyn clips.

---

## Failure clusters (do not reopen)

Grouped by *why* they died, not by method name.

### 1. New sampler on an old student → twitch or paint

Crossed host (`sf_roll` / `rf_chunk`): Dyn 59–75%, flicker
~0.975. Always-on mix: Dyn 8/8, flicker 0.978. FIFO lookahead
+21% tail, IQ 68.23. Leftover ρ moves pixels and kills IQ.
Linger / dump: IQ 66.34 / 68.14. `context_noise=50` paints;
`sf_ctx` 0004 tail 0.190.

**Read:** H-DMD already named this. Rolling-only fake videos
glue frames from different noise slots. Inference-only list
or host-switch is that mismatch at test.

### 2. Mid-chunk rewrite → a new picture

Intra / lastmix / restep / bpseudo / keep-picture (nudge,
wiggle, latmot): subject < 0.68, IQ 66–70. Four DMD steps
means “a small rewrite” is another draw. Closed.

### 3. Weight TTA → collapse

AdaSteer IQ 43 / 51 / 18. Pathwise TTC’s published failure
on distilled AR is the same exhibit. LoRA-at-test-time stays
closed.

### 4. Extra memory without their train → no-op or tax

Replay-sink without Dynamic RoPE matched do-nothing. Native
Rolling already sinks the first block. Extra `sink_size` /
prefix pin: subject or IQ tax. LongLive sink was dead until
they trained long.

### 5. A score that does not match the judge → never fires or damps

Handcrafted pretty-score fought IQ. Prefix-match froze
motion. 1.3B freeze-score (`rf_tscore` / `sf_tscore`) was
**identity** on N=8 (gate never redrew). Motion-max pick
twitches. Appearance pick damps.

### 6. Cheapen that is not cheaper

CachedSearch CPU KV snap: slower, not cheaper. Re-gate
alive, no lift.

### 7. Protocol, not method

Stem prompts (`panda 0013`) infected early tables. Official
VBench = full clip; Dyn = percent of clips. I2V-from-still
N=32 is discovery only. Do not mix stem numbers into caption
tables.

---

## What is actually still true

1. **V2V prefix-continuation is the right task** for “visual
   history → long AR.” T2V 128 MovieGen is only a comparison
   table. Submit-ready, not launched.
2. **Rolling is the cheap identity host** (47 s, subject 0.685,
   Dyn 28.9%). Self Forcing is the live-er host (32.8% Dyn,
   better pixels). Always-search is the Dyn host (50.8%) at
   7–8× Rolling.
3. **The paper cannot be “Always with a skip bit.”** Gate
   neighbors already said that.
4. **The paper cannot be another frozen-student stagger.**
   Schedule, mix, FIFO, leftover ρ, ctx are that class. All
   NO.
5. **KV / sink / window are already on** in the official
   kernels. Do not retune cite hosts.

---

## Three territories (pick one; then we spec)

### A. A new student (reopen distill)

The only way to change the sampler without twitch is to
**train the student on that sampler**, same machine: unroll
+ holistic DMD. Stream Forcing (stochastic path between
independent and monotone), H-DMD (one noise slot in the fake
video), Ms. Forcing (coarser noisy tokens), Reward Forcing
(EMA-sink + motion-weighted DMD) are the published members.

Running their recipe is citing them. A paper here needs **our**
distill idea, for example: unroll on **real V2V prefixes**
(neither paper trained on a 2 s leftover), or a video-level
objective that jointly holds Dyn% and Imaging Quality (their
DMD is teacher-score, not VBench). Cost is ~1 day, 8 GPUs.
This was locked while we were hunting a TTA gadget. The
gadget hunt is over.

### B. Analysis as the contribution

Claim: a few-step forcing student cannot be rewritten at
test; the only safe TTA is selection among its own futures;
that selection is already the neighbor literature. Evidence:
cite-128 full metric suite + the closed-door atlas above.
High-risk as a CVPR *method* paper. Honest as the first half
of A, or as a findings paper if the writing is the result.

### C. A new control, not a new sampler

Still frozen weights. Must not edit the list, the KV write,
or the pixels inside a chunk. Untried in our lock:

- **Text as the live condition.** Recaption the prefix / last
  lock and recache T5 (LongLive does this at prompt switch;
  we never did it for drift). Occupied if we only copy them.
- **A tiny trained verifier** that predicts official Dyn+IQ
  from cheap signals, then selects among already-legal
  futures. Freeze-score never fired; handcrafted pretty-score
  lied. LatSearch already owns “learned latent reward +
  prune” on Wan 1.3B. We would need a different label or
  unit.
- **Do not** reopen seed-search cheapen as the title.

C is the remaining TTA-shaped fork. It is narrower than A
and easier to get scooped.

---

## Do not do

Remake cite-128. Scale I2V. TTC. LoRA / AdaSteer. Mid-chunk
rewrite. Leftover ρ. Inference-only list. Mix / ctx / FIFO /
lock-score scale-up. Extra sink. Crossed host. WAVE=3.
Cheapen Pseudo. Write a gate paper.

No submit until the user picks A, B, or C (or a named
hybrid: B as the motivation half of A).
