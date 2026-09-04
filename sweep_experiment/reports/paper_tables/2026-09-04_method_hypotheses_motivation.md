# Why these method hypotheses (2026-09-04)

The failed runs are **appendix evidence**, not the paper. They
tell us which class of idea is dead and which class is still
alive. This note restates the live hypotheses and the chain
of observations that put each one on the table.

Pseudo-future Search stays dropped as the title. No GPU
tonight. Do not remake cite-128. Canvas:
`canvases/method-hypotheses-motivation.canvas.tsx`.

---

## What the appendix is allowed to claim

Two sentences, both earned:

1. **If you edit the path** (weights at test, noise list,
   dirty cache write, other paper’s sampler, 0.7 s rewrite),
   the photograph dies or the motion becomes jitter.
2. **If you only choose among paths the student already
   knows how to emit**, official Imaging Quality survives
   and official Dynamic Degree can rise (Always-search:
   42/128 → 65/128 dynamic clips, Imaging Quality 72.07 →
   72.19).

That pair is the motivation engine. It does not name a
method. It rules out another frozen-student stagger and
rules in either (a) teach the student a new path, or (b)
choose among old paths with a judge that is not a liar.

---

## Hypothesis 1 — Distill on real video openings

**The idea.** Train the next student the same way Self
Forcing and Rolling Forcing do (unroll with a key-value
cache, score the whole self-made video), but start each
unroll from a **real 2 s leftover**, not from noise or a
still. Test stays our video-to-video protocol.

**What put this on the table.**

- Neither official paper trained on a real leftover. Their
  unroll is self-generated history after a text prompt.
  Our test is “here are two real seconds, continue.” That
  mismatch is the exposure-bias seam we actually run.
- Filename stems (`panda 0013`) made T5 ignore the scene.
  Real captions changed the pictures. Same-method deltas
  survived, but the *task* is “continue this bathroom /
  truck / kitchen,” not “continue a filename.”
- Dirtying the cache write on that real opening
  (`context_noise=50`) did not fix the seam. It painted
  (Rolling Imaging Quality 67.6; Self Forcing clip 0004
  ending motion 0.190). So the bug is not “the opening is
  too clean in the cache.” It is “the student never
  practiced *starting from a real opening*.”
- Every inference-only sampler change on the old student
  twitched or painted. The appendix says: if the path is
  new, the student has to see it in distillation.

**Ours vs citing them.** Longer unroll, Stream Forcing’s
schedule, H-DMD’s single-slot fake video, Reward Forcing’s
motion-weighted loss — those are their papers. **Putting
the real leftover in the training unroll** is the part
neither of them did, and it is exactly our test.

**What would kill it.** A student trained this way that
still freezes at 30 s the way Self Forcing does, or that
paints the way leftover ρ did. Then the seam was not the
bottleneck.

---

## Hypothesis 2 — Distill with the judge we actually use

**The idea.** Keep the unroll machine. Change what “a good
whole video” means in the loss, so it tracks **official
Dynamic Degree and Imaging Quality together**, not only
the teacher-minus-critic score from distillation.

**What put this on the table.**

- Training’s signal is “does this noised clip look like
  the big teacher minus the critic.” Our paper table is
  “how many clips are called dynamic, and does the photo
  still look clean.” Those are different questions. We
  learned that the hard way.
- The homemade “too sharp / too still” score pointed the
  wrong way: official Imaging Quality *rewards* sharpness;
  we were punishing it. Prefix-match rewarded looking like
  the opening and produced stills. The 1.3B self-score
  never redrew. So any loss we can compute cheaply at test
  has been a bad stand-in for the official table.
- Rolling’s own sink protects identity (subject 0.685)
  and *lowers* official dynamic clips (28.9% vs Self
  Forcing 32.8%). The student can be taught to hold a
  face and damp motion. The reverse (Always-search) can
  raise dynamic clips and *hold* Imaging Quality. So the
  two official numbers are not doomed to trade off — but
  only when we **select** legal futures. Distillation has
  never been asked to want both at once on *our* labels.
- Reward Forcing already reweights distillation toward
  high-motion teacher samples. Rerunning that is citing
  them. Ours would have to be a **different label** (for
  example official Dynamic Degree + Imaging Quality on
  self-rolled clips, or a V2V-specific identity+motion
  pair), not their EMA-sink recipe.

**What would kill it.** A new loss that lifts official
dynamic clips and drops Imaging Quality below the host by
a point or more — the leftover-ρ signature — or that
matches Reward Forcing’s table without a new label.

---

## Hypothesis 3 — Refresh the text, not the pixels

**The idea.** Leave the checkpoint and the sampler frozen.
After each lock (or after the real opening), **write a new
caption of what is actually on screen**, re-encode it with
T5, and continue. The next 5 s is conditioned on the scene
that exists, not on the sentence from t=0.

**What put this on the table.**

- The stem-prompt bug was a text bug. T5 heard `panda 0013`
  and the tail became a panda. The real 0013 caption is a
  bathroom stain. We already know the **language condition
  can hijack** the next seconds.
- After a real 2 s opening plus 10 s of generation, the
  original caption is stale. The model is still being
  asked for “a truck with its hood open” while the hood
  may already be a different picture. We never updated
  that sentence. We dirtied the *cache* instead
  (context noise) and the photo died. Wrong lever, same
  seam: the condition does not match the history.
- Selecting among seeds works because those seeds are
  still “the student answering the same sentence.” If the
  sentence is the lie, more seeds will not fix it.
- LongLive recaches the key-value store when a *user*
  switches prompts. That is occupied if we only copy
  their interactive switch. What we have not done is
  **recaption our own lock as a drift fix** on this
  video-to-video protocol, with official Dynamic Degree
  and Imaging Quality as the judge.

**What would kill it.** Recaption that turns every clip
into a new scene (subject collapse) or that matches
do-nothing because the new sentence is ignored. Also
killed if it is indistinguishable from LongLive’s
prompt-switch recache on their protocol.

---

## Hypothesis 4 — A tiny judge that selects, not edits

**The idea.** Keep drawing several legal futures (the one
move that did not wreck the photo). Replace the liar
proxies with a **small trained head** whose labels are
official Dynamic Degree and Imaging Quality (or a human
pair that tracks them). Use it only to pick or to skip a
draw. Do not rewrite pixels, lists, or the cache.

**What put this on the table.**

- Always-search is the existence proof: selection among
  the student’s own seeds raises official dynamic clips
  and holds Imaging Quality. The method hole is “best-of-N
  on seeds” is already the literature.
- Every *cheap* judge we tried lied. Homemade sharpness
  fought official Imaging Quality. Prefix-match froze
  motion. The 1.3B self-score never fired. So the
  hypothesis is not “selection works” (we know that). It
  is “selection only becomes *ours* if the judge is the
  same object as the paper table.”
- CachedSearch tried to make the same selection cheaper
  by snapshotting the cache. Same pixels, more time. The
  remaining cheapen is **skip a draw** when the judge
  says this lock is already good — but only if that judge
  is not the skip-bit we already dropped as a title.
- LatSearch already published a learned latent reward
  plus prune on Wan 1.3B. SDVG uses ImageReward on a 1.3B
  draft. We would need a **different label or unit** (for
  example official Dynamic Degree on the *full 30 s
  video-to-video clip*, not a mid-trajectory VLM on T2V)
  or this is citing them.

**What would kill it.** A head that predicts official
scores and still picks the same seed as motion-max
(twitch) or prefix-match (stills), or a cost cut at the
13% Pseudo scale that loses dynamic clips.

---

## How these four sit together

| Hypothesis | Appendix sentence that forced it | Frozen weights? |
|---|---|---|
| 1. Distill on real openings | New paths twitch unless the student saw them; the real leftover is a path they never saw | No |
| 2. Distill with our judge | Training’s score is not VBench; every cheap test-time score lied | No |
| 3. Refresh the text | T5 can hijack the tail; we edited the cache instead of the sentence | Yes |
| 4. Tiny official judge | Selection is safe; our selectors were not the official table | Yes |

1 and 2 are **one student paper** if we do both (real
openings in the unroll *and* a loss that wants Dynamic
Degree plus Imaging Quality). 3 and 4 are **test-time
papers** that stay on the frozen checkpoint. 3 and 4 can
sit on a new student later; they do not require one to
start.

The analysis atlas is the **appendix / related-work
motivation**, not a fifth title.

---

## Do not

Launch 8-GPU distillation tonight. Remake cite-128.
Reopen mix, FIFO, leftover ρ, AdaSteer, mid-chunk rewrite,
or Pseudo-as-title. Spec only after the user picks 1, 2,
3, 4, or the 1+2 student pair.
