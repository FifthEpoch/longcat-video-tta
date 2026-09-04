# Self Forcing and Rolling Forcing: the shared experiment machine (2026-09-04)

Not a submit. Read of Huang et al. [2506.08009](https://arxiv.org/abs/2506.08009)
(Self Forcing) and Liu et al. [2509.25161](https://arxiv.org/abs/2509.25161)
(Rolling Forcing), plus their Algorithm 1 / Appendix recipes. Canvas:
`canvases/sf-rf-common-impl.canvas.tsx`.

Do not start 8-GPU Distribution Matching Distillation (DMD). Do not remake
cite-128. Mid-chunk rewrite, leftover ρ, linger/dump, extra sink, and
crossed host stay closed.

---

## The common idea (implementation, not slogan)

Both papers’ **experiments** are the same post-training machine:

1. Start from **Wan2.1-T2V-1.3B** (flow matching, shift \(k=5\)).
2. **Causal Ordinary Differential Equation (ODE) init** on 16k teacher
   pairs (CausVid recipe).
3. **Unroll the inference sampler during training** with a key-value
   (KV) cache. History is the model’s own frames, not ground truth.
4. Apply a **holistic video-level DMD** (reverse KL via teacher score
   minus critic) on that self-rolled clip. Data-free: prompts only
   (VidProM filtered + Qwen rewrite).
5. **Truncate gradients** so the sequential unroll fits: Self Forcing
   backprops only the last denoise step of each chunk (and samples
   which step \(s\) is “last”); Rolling Forcing backprops only a
   non-overlapping subset of windows.
6. Same emission unit: **chunk = 3 latent frames**, 16 fps,
   \(832\times480\). Official table is Visual Benchmark (VBench).

Rolling Forcing Algorithm 1 **is** Self Forcing Algorithm 1 with a
wider window and non-overlapping gradient windows. They even mix
**50% Self Forcing loss** because Rolling-only DMD concatenates
clean predictions from **different noise slots** and the fake video
looks like bad camera motion (the H-DMD diagnosis).

“Forcing” in the code is not the stagger. It is:
**train = infer, then score the whole self-generated video.**

That is why `sf_roll` / `rf_chunk` twitched, leftover ρ and
linger/dump killed Imaging Quality, and Pseudo-future Search
survived: search is the test-time analog of a holistic video score.
Changing the sampler the student never trained with is not.

---

## Where the two papers diverge (on top of that machine)

| | Self Forcing | Rolling Forcing |
|---|---|---|
| Few-step list (paper) | 4-step `[1000,750,500,250]` | 5-step `[1000,800,600,400,200]` |
| Attention in the live unit | Causal; one chunk at one \(t\) | Bidirectional **inside** a window of \(T\) chunks; monotone diagonal |
| When a chunk locks | After its 4 steps | When it **exits** the window |
| Memory | Rolling KV of recent \(L\); hide first chunk at train so eviction works | Recent temporal KV **plus** frozen first-chunk sink with RoPE re-index |
| Train mix | Self-rollout only | 50% Self Forcing + 50% Rolling windows |
| Long-horizon claim | Rolling KV; quality still dies past train length | 30 s MovieGen + \(\Delta\) Imaging Quality (first 5 s vs last 5 s) |
| Our live student | `self_forcing_dmd.pt` | `rolling_forcing_dmd.pt` — live list floor **556**, not paper 200 |

Our cite-128: Self Forcing Dyn **32.8%** / subject 0.666 / IQ 72.07.
Rolling Dyn **28.9%** / subject 0.685 / IQ 71.52 / tail +33%.
Rolling’s revise+sink protects identity and damps Dynamic Degree.

---

## New ideas that sit on the shared machine

Paper lock: test-time. No new student unless you say go.

### 1. Mixed inference at the lock (RF’s own Appendix E)

Rolling Forcing writes: switch frame-by-frame (Self Forcing) when
you need interaction, rolling otherwise. We have both hosts. Gate
at **window-exit**: if the just-locked block froze, emit the next
block with the Self Forcing sampler (more Dyn); if it is living,
stay Rolling (keep subject). Not leftover ρ. Not a new list.
Risk: one host-switch can twitch if we swap mid-window. Switch
only at a clean lock.

### 2. Context noise on the KV write (still first cheap)

`context_noise` is **0** today. Both papers cache **clean**
history. Self Forcing detaches KV so the student never learns to
revise the past. At test, a little noise on the write is
“do not copy a too-clean still.” Apply it to (a) self-generated
locks and (b) the **real V2V prefix** — neither paper trained on
a real 2 s leftover. That is our exposure-bias seam.

### 3. FIFO lookahead on the Rolling host

One extra forward on the **noisier half** of the window before
the head emits (Kim et al., FIFO-Diffusion). Same student, same
list. ~2× Rolling, still << Always-search.

### 4. Teacher-score reject at lock

DMD’s actual signal is \(s_{\text{real}} - s_{\text{fake}}\) on
the whole clip. We cannot train the critic. We can run the frozen
Wan score once on the about-to-lock block and refuse the lock
(redraw that block only). Holistic like their loss. Costly; N=8
first.

### 5. Do not do

New timestep list, leftover ρ, extra sink, crossed host,
mid-chunk rewrite, 8-GPU DMD. Schedule8 already showed
inference-only list change dies on Imaging Quality.

---

## If we ever train (not tonight)

The shared recipe says the cheap student move is **longer
self-rollout + the same DMD**, not a new stagger. Rolling already
admits mid-sequence memory is discarded. Stream Forcing /
H-DMD / Ms. Forcing are that class (~1 day, 8 GPUs).
