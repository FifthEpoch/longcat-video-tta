# Pseudo-future Search — method note + related work (2026-08-25)

**Status:** write-up only. Do not rename `sf_pseudo` in code.
**Cite caption official N=32** (`metadata_csv`), not stem-prompt rows.
**Not a submit spec.** No WAVE=2. No intra-chunk implementation in this
window unless a dated spec says otherwise.

---

## 1. Name

**Paper name: Pseudo-future Search.**
**Code / talk short name: Pseudo.** Always-search is the no-gate
ablation. Self Forcing is the host.

It is called Pseudo because we do **not** have ground truth for the
invented 30 s. We manufacture a stand-in task: hide the last 0.7 s of
the **real** opening, ask another seed to write those frames, and treat
that held-out slice as a **pseudo-future** — a real continuation we
already possess, used as a label for “does this seed understand the
recent past?” The name is from the 2026-08-21 sampling-ideas memo
(“Pseudo-future validation”), not from a fake model or a fake caption.

Do not call it Prefix-match. Prefix-match *picks* the future that looks
like the opening and killed motion. Pseudo uses the opening only as a
**gate**, then picks the tail by motion+trust.

A fine subtitle if a venue wants jargon: **prefix hold-out search**.
Keep Pseudo in tables so they match `sf_pseudo` sidecars.

---

## 2. What we actually do

Piece 0 is a real Panda prefix: **9 latents ≈ 2.1 s**. Never searched.
Then **6 × 21 latents ≈ 30 s** of Self Forcing causal DMD (4 denoising
steps per 3-latent block). Default seed = do-nothing Self Forcing.

**Once, before any invented chunk:**

1. Split the prefix into **A** = first 6 latents (~1.4 s) and **B** =
   last 3 latents (~0.7 s).
2. Condition on A. Generate B with the default seed and with k−1 extra
   seeds (k=4). Decode. Score pixel MAE against the **real** B.
3. **Fire** if some extra seed beats the default by more than γ
   (`pseudo_gamma`; pre-registered, do not retune).
4. Restore the real B. The 2 s opening that the viewer sees is always
   the real video.

**Then, on each of the six invented chunks:**

- If the gate fired: run k=4, pick max motion among candidates that
  stay ≥ 0.8× the default seed’s motion (trust reject). Same pick as
  Always-search.
- If the gate did not fire: write the default seed (exact Self Forcing).

The gate is computed **once on the prefix**, not re-estimated every
chunk. That is why 9/32 caption clips are byte-identical to Self
Forcing: the hold-out never preferred another seed, so we never
searched. Always-search has no such skip.

We never look at the true 30 s future. B is the only GT we use, and it
is already on screen in the opening.

---

## 3. Caption official numbers (the claim)

Host = Self Forcing. Full-clip VBench. Dyn = **percent of clips**
RAFT labels dynamic, not the median.

| | Self Forcing | Pseudo | Always-search |
|---|---:|---:|---:|
| subject | 0.700 | **0.701** | 0.687 |
| IQ | 71.54 | **71.66** | 71.16 |
| Dyn% | 21.9% (7/32) | **40.6%** (13/32) | **43.8%** (14/32) |
| flicker | 0.989 | 0.985 | 0.984 |
| tail vs SF | — | **+28%** (23/0/9) | **+39%** (30/2/0) |
| mean wall | 196 s | **304 s** | **348 s** |
| median wall | 113 s | 357 s | 348 s |

Honest line: Pseudo is an **efficiency controller** that keeps most of
the search gain. It is cheaper than Always in the **mean** (~13%)
because of the 9 skips. A fired clip does the hold-out probe **plus**
k=4, so the **median** fired clip is slightly *more* expensive than
Always. When the gate fires, Pseudo is not a cheaper Always — it *is*
Always plus a probe.

On Rolling Forcing the same gate is **dead** (20/32 exact host). Cite
that as a negative: the prefix posterior is usable on the SF student,
not on the RF overlapping window.

Local examples (`~/Desktop/caption_examples`): Pseudo equals
Always-search on 5/6 files. Only panda_0004 (photo-critique book) is a
skip — Pseudo copies Self Forcing; Always leaves the book and dies
into text/noise. That is the gate doing its job.

---

## 4. Related work — whose problem is this?

Three nearby papers. None is Pseudo-future Search.

### 4.1 When to spend — Early Failure Detection

[Early Failure Detection and Intervention in Video Diffusion Models](https://arxiv.org/abs/2603.14320)
(KAIST, 2026). Wan 1.3B / 14B and CogVideoX. They convert mid-denoise
latents to an RGB preview (~39 ms), score text–video alignment, and
only then regenerate. Failures show up around step 10 of a **50-step**
sampler. Up to ~2.6× less overhead than post-hoc retry.

**Shared idea:** do not pay full search / regen on every video.
**Difference:** they inspect a T2V denoising trajectory with an
alignment scorer. We hide 0.7 s of an already-observed V2V prefix and
use pixel MAE. No VLM. Our student is **4-step DMD**, so their “look
at step 10 of 50” does not copy.

### 4.2 How to pay for search — CachedSearch

[CachedSearch](https://arxiv.org/abs/2607.23159) (2026). Same Wan 1.3B
family. Explore every best-of-N candidate under aggressive caching,
then re-generate only the winner at full compute. N=8 keeps 94.7% of
BoN gain at 63% of the cost. Mid-trajectory pruning: 3.11× exploration
saving at 88.6% capture.

**Shared idea:** keep search quality, spend less.
**Difference:** they **always** search; they cheapen each try. We
decide **whether** to search. Orthogonal. Stack later. Do not write
the paper as “we beat CachedSearch.”

### 4.3 Hold-out / verify — speculative decoding and LatSearch

Teacher-forcing hold-out and speculative decoding (Leviathan et al.;
[Hu & Zhang 2026](https://arxiv.org/abs/2601.17397) for AR video):
draft tokens, verify against a target, accept or reject.
[LatSearch](https://arxiv.org/abs/2603.14526): score / prune in latent
space before a full decode, on Wan 1.3B.

**Shared idea:** use a cheaper proxy before committing the expensive
write.
**Difference:** they verify or prune **inside one generation**. We use
a held-out **past** to gate **later chunks**. We do not currently
abort a chunk mid-denoise (see §5).

### 4.4 What we already ran and is not this method

| Method | Relation to Pseudo |
|---|---|
| Always-search | Same pick, no hold-out gate. The ablation. |
| Prefix-match (`seed_bon`) | Uses the opening as a *pick*, not a gate. Identity damper. **NO.** |
| Sick-search | Gate = “did the last chunk freeze?” After-the-fact. **NO** on caption SF. |
| Rewind | Resample a *finished* sick chunk. After-the-fact. Small HOLD. |
| Sink / LongLive | Attention pin, no search. Not ours. |
| AdaSteer | Weights / activation residual. **NO.** Do not scale. |
| TTC / LoRA-at-test-time | Locked out. |
| History Guidance | Identity vs dynamics trade. Not a seed gate. |
| StreamingT2V | Documents the freeze attractor. Not a controller. |
| Relax / Rolling Forcing | Other host. RF-hosted Pseudo is a dead gate. |

---

## 5. Intra-chunk intervention — the open hole

The user is right: **once a seed is chosen, the 21-latent chunk (~5 s)
is a sealed write.**

What the runner does today (`_run_one_chunk` → `_denoise_chunk`):

1. Sample noise for the whole chunk from that seed.
2. For each 3-latent block, run the full 4-step DMD list.
3. Decode **once** at the end of the chunk.
4. Score / commit. Only then can Rewind or the next chunk’s search
   see freeze or garbage.

So frames 1–21 of a chunk can collapse, sharpen, or morph, and we
have no hook until the chunk is finished. The first-step residual
`U_t` is logged (`noise_probe`) and is not used as an abort.

**Can we intervene inside the chunk?** Yes in principle. Three
published axes, plus one that is already almost in our loop:

| Lever | Grain | Closest paper | Fit on 4-step SF |
|---|---|---|---|
| Mid-step RGB preview + restart | denoising step | Early Failure Detection | Weak. Only 4 steps; their signal is step ~10/50. |
| Latent reward / prune | mid-trajectory latent | LatSearch; CachedSearch prune | Possible. No extra VAE if we stay in latent. |
| Block abort (every 3 latents ≈ 0.7 s) | inside the chunk, after a block | our own block loop | **Closest cheap hook.** Decode or latent-diff that block, drop the seed, restart the rest. |
| After-chunk rewind | whole ~5 s | our Rewind | Already exists; too late for intra-chunk garbage. |

Rolling Forcing is already finer (overlapping windows). That is a
host change, not an intra-chunk controller on SF. Crossed host
(`sf_roll` / `rf_chunk`) twitched. Do not “fix Pseudo” by swapping
samplers.

**2-month paper:** do **not** make intra-chunk the method. The
publishable claim is the prefix hold-out gate. Intra-chunk is the
honest limitation paragraph and the obvious next controller if the
trio (SF / Always / Pseudo) scales. A first probe, if we ever open
it: abort after block 1 of a chunk when latent/pixel motion falls
below 0.8× the prefix (same 0.8 we already pre-registered). No new
score. No ImageReward.

---

## 6. What to write in the paper

**Claim.** On long causal V2V, a prefix hold-out (pseudo-future)
decides whether this video needs best-of-k. When it does, the pick
is ordinary motion+trust search. When it does not, the video is
Self Forcing.

**Contributions, if N grows past 32.**

1. A GT-free gate that uses the observed opening as a pseudo-label
   for the unseen tail.
2. Always-search ablation: most of the motion / Dyn% gain is the
   pick; the gate is the cost cut (mean, not median).
3. Host negative: the same gate is dead on Rolling Forcing.

**Do not claim.** A new backbone. A new reward model. Intra-chunk
repair. Quality above Always on tail (Always is hotter; Pseudo skips
the 9 still-enough openings). CachedSearch-beating cost at matched
width.

**Next experiment (not this note).** Scale only Self Forcing /
Always-search / Pseudo on caption V2V. N=32 is the discovery set.
Intra-chunk stays a limitation until a dated spec.
