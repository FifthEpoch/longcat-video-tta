# Rolling Forcing descendants and noise-schedule cousins (2026-09-01)

Not a submit. Literature after the user asked for methods
similar to Rolling Forcing. Canvas:
`canvases/rf-schedule-neighbors.canvas.tsx`.

Do **not** mix their MovieGen T2V VBench-Long into cite-128.
Ours is V2V prefix-continuation, Panda `metadata.csv`, Dyn as
percent of clips. Mid-chunk rewrite, CFG/shift, weight TTA,
and I2V scale-up stay closed.

---

## What Rolling Forcing is (four levers)

Liu et al., ICLR 2026 / [arXiv 2509.25161](https://arxiv.org/abs/2509.25161).
Not a from-scratch Wan. Causal ODE init + ~3k-step DMD on the
same 1.3B. Official stagger `[1000, 800, 600, 400, 200]`.

| Lever | RF default | What it buys |
|---|---|---|
| **1. Diagonal** | Window of T blocks, future noisier | Next second is still being drawn |
| **2. Joint revise** | Bidirectional attn inside the window | Noisy tail can correct almost-clean head |
| **3. Lock** | Commit only at window-exit | One clean head per forward pass |
| **4. Memory** | Frozen first-frame sink + RoPE freeze | Identity. Dyn% tax (our 28.9% vs SF 32.8%) |

Our leftover ρ touched (1) globally. Extra sink touched (4).
Crossed host (`sf_roll`) broke the student–sampler pair.

---

## Papers that build on Rolling Forcing

### Training-free (almost all memory)

| Paper | Method | Lever |
|---|---|---|
| **Deep Forcing** (Yi et al., [2512.05081](https://arxiv.org/abs/2512.05081)) | Deep Sink (~half window) + temporal RoPE re-align + Participative Compression (keep KV that recent queries attend to). Host = Self Forcing. | 4 |
| **Relax Forcing** (Zhao et al., [2603.21366](https://arxiv.org/abs/2603.21366)) | KV roles: Sink / History / Tail. Pick mid-history by `sim(sink) − λ·sim(tail)`. Hybrid RoPE for a non-contiguous set. Host = SF. Their VBench-Long 30 s: Dyn **65.67** vs RF **32.71**. | 4 |
| **Forcing-KV** ([2605.09681](https://arxiv.org/abs/2605.09681)) | Hybrid KV prune (static vs dynamic heads). Speed. | 4 |

### Retrain a student

| Paper | Method | Lever |
|---|---|---|
| **Ms. Forcing** (Li et al., [2607.20940](https://arxiv.org/abs/2607.20940)) | Same RF window. Coarser patches on noisier slots (45% fewer tokens). **H-DMD**: fake video assembled from one shared source noise level, not mixed slots. Drops the auxiliary SF loss. | 1 as compute; 3 as distill |
| **Stream Forcing** (Zhu et al., [2608.10439](https://arxiv.org/abs/2608.10439)) | Per-frame noise as a logit-normal stochastic process. Curriculum from Diffusion Forcing (independent levels) to a Rolling-style monotone diagonal. | **1. The real schedule paper.** |
| **Reward Forcing** (Lu et al., CVPR 2026, [2512.04678](https://arxiv.org/abs/2512.04678)) | **EMA-Sink**: frozen frame-0 becomes a running average of evicted KV. Re-DMD weights DMD toward high-motion teacher samples. | 4 + objective |
| **LongLive** (Yang et al., [2509.22622](https://arxiv.org/abs/2509.22622)) | Parallel host. KV-recache at prompt switch, frame sink, train-long. | 4 |

### H-DMD diagnosis (keep)

RF's cheap DMD concatenates clean predictions from
**different window slots**, so adjacent frames in the fake
video came from different noise levels. Inference only emits
the head. That is why they mix in a Self Forcing loss, and
why `sf_roll` / `rf_chunk` twitched. We cannot run H-DMD at
test time.

---

## Schedule ancestors (RF cites these)

| Paper | Schedule | Train? |
|---|---|---|
| **Rolling Diffusion** (Ruhe et al., ICML 2024) | Local time per frame; later = noisier; sliding window unrolls forever | Yes. RF = this + few-step DMD + sink |
| **FIFO-Diffusion** (Kim et al., NeurIPS 2024) | Diagonal queue on a bidirectional model. Latent partitioning (narrower noise range). Lookahead: update the noisier half twice before emit | **No.** Closest training-free diagonal |
| **Diffusion Forcing** (Chen et al., 2024) | Independent noise per frame | Yes. Stream Forcing's other endpoint |
| MAGI-1 / PAVD | Chunk-wise progressive denoise | Yes |

---

## Core approaches, collapsed

Every method answers four questions.

| Question | Self Forcing | Rolling / FIFO | Diffusion Forcing |
|---|---|---|---|
| Noise vs time | One level for the chunk | Monotone diagonal | Independent per frame |
| When lock | After that chunk's 4 steps | Window / queue exit | When that frame hits t=0 |
| What can still revise | Nothing | Noisier future in-window | Noisy neighbors, no structure |
| Memory | Recent FIFO | + frozen sink | Deep / Relax / EMA rewrite this |

Stream Forcing says the interesting training space is the
**path between** independent and monotone, not either endpoint.

---

## Inspirations we can run (no new student)

Paper lock: no TTC, no I2V scale-up, no 3k-step DMD. A new
student is citing their method.

### Still open, Rolling-shaped, cheap

| Idea | From | Why it is not leftover ρ |
|---|---|---|
| Context noise on the KV write (today 0) | Self-Forcing++ / `2026-08-30_wan_rf_intervene.md` | Does not change the diagonal |
| Bump only the **next** injected block if the just-locked one froze | Stream Forcing `μ_t` | Local, then native. Global ρ taxed stills |
| **FIFO lookahead** on the RF host | FIFO-Diffusion | Extra pass on the noisy tail before the head emits. ~2× Rolling, still << Always |
| Shallower diagonal (3 of 4 slots) | FIFO latent partitioning | Narrower train–test gap inside one window |

### Open, but memory, not schedule

| Idea | From | Known risk |
|---|---|---|
| EMA-Sink at inference | Reward Forcing | They trained with it. Pure EMA might still unstick Dyn |
| Sink / History / Tail pick | Relax Forcing | Training-free on SF. Do not cite their 65% Dyn |
| Soften the native sink | Deep + Relax + 30 Aug note | Extra sink already hurt subject / flicker |

### Closed or needs a student

| Idea | Why not |
|---|---|
| Stream Forcing curriculum / H-DMD / MSP / Re-DMD | New student. Cheap vs Wan pretrain; **~1 day on 8 GPUs**, memory-heavy DMD. See `2026-09-01_rf_nonlinear_schedule.md`. |
| `sf_roll` / `rf_chunk` | Twitch. Student and sampler are a pair |
| One ρ for the whole video | Caption leftover **NO**. Imaging Quality 64–68. Not a timestep list. |
| Extra frozen sink / VAE recache / mid-chunk rewrite | Identity tax. Closed |

### Non-linear *timestep list* (not ρ)

Official Rolling Forcing list is **linear in t**. Linger-high
and dump-early on the **existing** student are the cheap
smoke. Spec (no submit):
`2026-09-01_rf_nonlinear_schedule.md`.

If the goal is **another noise schedule like Rolling**, the
test-time cousins are FIFO lookahead, a shallower diagonal,
or a non-linear list on the current checkpoint. A new
student is the Stream Forcing class — do that only if the
smoke dies on Imaging Quality and we decide to leave
test-time adaptation.
