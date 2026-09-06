# Should we run Go-with-the-Flow? And holes in SAVi-DNO (2026-09-06)

Canvas: `canvases/gwf-savi-noise.canvas.tsx`.
Not a submit. No GPU from this note. pwarp 17058386–393 still
the in-flight kill test. **No 8-GPU DMD. No remake cite-128.**

---

## Go-with-the-Flow — is there benefit in running it ourselves?

**Retrain their recipe: no.** Rank-2048 LoRA on CogVideoX-5B,
4M videos, ~40 GPU-days. That is their CVPR 2025 Oral. The
idea “pair warped \(x_T\) with matching flow and fine-tune”
is occupied. Territory A is a *new student* only if the
label is not “we did GwF on Wan.”

**Run their released checkpoint at inference: maybe, as an
appendix neighbor, not a title.** Use: a few leftover V2V
clips, their warp + their LoRA, official Imaging Quality /
Dynamic Degree. Ask: what does *correct* warped-\(x_T\)
plus a student that saw that prior look like? Their
appendix already says frozen CogVideoX + warped noise
follows the flow and **paints**. We already painted with
a cheaper, worse warp (locked extra stencil, IQ 49).
Seeing their real warp on our leftover protocol is
calibration, not a method.

**What we would actually learn**

| Run | Cost | What it answers |
|---|---|---|
| Their LoRA + their \(x_T\) warp, N=8 leftover | Their weights; no 8-GPU train | Upper bound of “noise is the motion condition” on *short* control. Not 30 s causal freeze. |
| Frozen SF + their \(x_T\) warp (no LoRA) | Cheap | Repeat of their appendix + our nwarp: prior mismatch. Likely IQ death. |
| Retrain SF/RF with warped leftover noise | 8-GPU DMD | Occupied as GwF unless the pair is “real 2 s leftover + official Dyn+IQ.” That is Hypothesis 1/2 again. |

Noise manipulation is still live. The open forks are **not**
“reimplement Burgert”:

1. **pwarp** (queued): slide the guessed *picture*, ordinary extras.
2. **Extras without the 30 s stencil**: HIWYN each extra, do
   not carry one field. Isolates “temporal lock” vs “warp.”
3. **Leftover-only noise opt** (SAVi’s fair protocol): fit
   \(\epsilon\) so the last leftover reconstructs, apply to
   the unseen future. Judge official Dyn + IQ, not PSNR.
   Our LongCat port of this was broken; do not cite those
   numbers. Wan/SF is a new port.
4. **Do not** retune nwarp \(\gamma\) or start their LoRA
   tonight.

GwF is a *control* paper: you bring a driving flow. It does
not invent living motion after a leftover freezes. That is
why running it does not replace a long-horizon idea.

---

## SAVi-DNO — holes, not a conspiracy

[arXiv:2511.18255](https://arxiv.org/abs/2511.18255), posted
23 Nov 2025. Bonn / Timofte / Gall. Still listed as arXiv on
the first author’s page; the same group has CVPR 2025
(SyncVP) and CVPR 2026 (EgoControl). **Unpublished at a
major venue is not evidence of fraud.** Ten months and one
reject cycle is normal. The paper has enough ordinary holes
that a CVPR/NeurIPS reject is the boring explanation.

### What they actually claim

Video *prediction* (next clip given past clip) on a
continuous stream. Freeze \(\theta\), Adam-update the
initial noise \(\epsilon\) when the next GT clip arrives,
carry that noise forward, mix with a little white
(\(p\)). PVDM and Vista. Report PSNR / SSIM / FVD.

DNO (Karunratanakul et al.) already optimizes noise at
test. Their increment is **carry \(\epsilon\) along a
stream**.

### Presentation / protocol holes

1. **Time indices fight.** Eq. 2 fits \(\epsilon\) on
   \((z_{s-1}\rightarrow x_s)\). Algorithm 1 predicts
   \(\hat{x}_{s+1}\) from \(z_s\), then takes
   \(\nabla\mathcal{L}(x_{s+1},\hat{x}_{s+1})\). If you
   *score* the pre-update \(\hat{x}_{s+1}\), that is
   leakage-free (loss uses the clip only after it is
   observed, to help the *next* predict). If you
   re-sample after the update and report that, you
   fit \(\epsilon\) to the scored future. We implemented
   the second by accident; the paper’s default in our
   first LongCat port was the **oracle leak**. A
   reviewer can read it either way. That is a writing
   failure.

2. **“Continuous stream” is still 16+16 (or 3+22) with
   GT next clip in the loss.** Not open-loop 30 s. Not
   “the future is unavailable.”

3. **Baseline \(\eta\) mismatch.** PVDM is reported at
   \(\eta=1\); SAVi at \(\eta=0\). They show the
   \(\eta=0\) pair in Table 7 (0.435 → 0.485 SSIM), so
   the gain is not only the mismatch — but the main
   tables hide it.

4. **Table 2 caption / alignment is broken** (autoencoder
   row vs \(k\)). Sloppy, not fatal.

5. **Code “upon acceptance.”** Still no venue, so no
   official code as of this note.

### Method holes

1. **The loss is reconstruction of a clip you will soon
   have (or just got).** Pixel L1 + Kinetics ResNet3D
   feature. That is “make last clip invertible in
   \(\epsilon\), hope the next clip likes the same
   \(\epsilon\).” PSNR +0.7–1 dB on Ego4D. Best-of-10
   PVDM oracle is SSIM 0.495; they get 0.485. Close to
   “carried seed search,” not a new generative idea.

2. **Pixel-only makes FVD worse** (535 vs 500). The
   feature-loss \(\lambda\) is a slider that buys FVD
   back and gives PSNR away (Table 8). They pick a
   point and report all three. Classic.

3. **Vista is a weak transfer.** +0.011 SSIM, +0.3 dB,
   FVD 974 → 945. UCF FVD at 50 steps is 545.1 → 543.4
   (noise). The “works on a foundation model” table
   barely moves.

4. **PVDM Inverse is a strawman.** Inversion noise is
   known-OOD; they confirm it and move on.

5. **Privacy sentence is rhetorical.** Not writing
   weights ≠ not using private frames. The optimized
   \(\epsilon\) can still carry the last clip.

6. **No VBench, no human study, no official Dynamic
   Degree.** Different task from a long-horizon freeze
   paper. Do not mix their PSNR into our caption tables.

7. **Our LongCat port.** Differentiable Euler did not
   match LongCat’s sampler; optimize vs no-optimize was
   +0.01 dB and both unusable. Dropped as a LongCat
   baseline (2026-07-20). Do not cite those numbers.

### Why a venue might say no (ordinary, not suspicious)

Increment on DNO; reconstruction-era metrics; PVDM not
a 2025 SOTA student; Vista delta small; algorithm easy
to read as leaking the future; no released code. Same
authors got other papers in. This one looks like a
workshop / TMLR / reject-and-resubmit, not a hidden
breakthrough.

### For us

SAVi-DNO is a **neighbor**, not a title. If we touch
noise-opt again: leftover-only fit, unseen future,
Wan Self Forcing, official Dyn + IQ, leakage-free.
That is the protocol we already wrote down. Fix the
Wan sampler first. Do not resurrect the LongCat port.
Do not start 8-GPU DMD.
