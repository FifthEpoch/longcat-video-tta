# Go-with-the-Flow — what they did and why they fine-tuned (2026-09-05)

Burgert et al., CVPR 2025 Oral, [2501.08331](https://arxiv.org/abs/2501.08331).
Read after the user asked whether a mid-denoise circular
noise shift is the same idea. Canvas:
`canvases/go-with-the-flow.canvas.tsx`.

Not a submit. No GPU.

---

## The idea (not a mid-step wrap)

Put **motion into the initial noise volume**, then denoise
as usual.

1. Take a 2D Gaussian slice (frame 0).
2. Transport that noise along optical flow to later frames
   (RAFT on a real video, or user polygons, or a depth
   warp). Handle expansion / contraction / holes with a
   density field and conditional white-noise fill. **Not**
   a torus wrap: pixels that leave the frame are dropped;
   empty cells are resampled, not taken from the opposite
   edge.
3. Keep each frame **spatially i.i.d. Gaussian**
   (Proposition 1). Only the *temporal* correlation is
   non-Gaussian.
4. Use that 3D tensor as \(x_T\) for DDIM.

Cousins: **HIWYN** (Chang et al., ICLR 2024) did the same
transport for *image* models, training-free, but too slow
to train on. GwF’s Algorithm 1 is the linear-time rewrite
(\(26\times\) vs HIWYN) so they can warp on the fly.

A `torch.roll` wrap is **not** their algorithm. Naive
bilinear / bicubic / nearest warp **breaks spatial
Gaussianity** (Table 1: Moran’s I 0.24–0.30, p=0). That
is why they spent the paper on a careful warp.

---

## Why image models are training-free and video models are not

They *did* ship an inference-only method. It is §3.2 / §4.3.

**Image diffusion, frame by frame (no FT).** DeepFloyd IF
super-res on DAVIS 43; DifFRelight portrait relight. Each
frame is an independent image-model call. The only thing
that ties frames is the warped noise. The image weights
were trained on spatial Gaussian; they preserve that, so
the model is in-distribution *per frame*. Temporal
structure is free extra correlation. This is HIWYN’s
setting. It works.

**Video diffusion (CogVideoX, AnimateDiff) needs FT.**
Those weights were trained to map a **spacetime i.i.d.
Gaussian volume** to a video. They already have temporal
attention and a 3D VAE. Feeding a temporally correlated
volume at test, without ever pairing that volume with the
matching flow at train, is a noise-prior mismatch. The
paper never prints “we tried frozen CogVideoX + warped
\(x_T\) and it failed.” What they do print:

- Motivation: motion control (drag, camera, transfer) is
  an under-explored *condition*. Other papers add extra
  modules. They want the condition to be the noise itself.
  A condition the student never saw has to be trained.
- Line they call surprising: “removing temporal
  Gaussianity does not deteriorate fine-tuning; it can be
  quickly adapted.” The surprise is that FT *works*, not
  that inference-only was enough.
- Table 2 ablation, DAVIS I2V: frozen
  **Original CogVideoX-5B** (text + first frame, ordinary
  noise) CoTracker mIoU **0.52** / flow err **0.67**.
  After warped-noise FT, mIoU **0.74** / flow err
  **0.36**. Text + still does not determine future motion.
- Degradation \(\gamma\): they mix warped noise with
  fresh Gaussian during FT so the model still sees
  ordinary noise. Full warp is a strong condition; they
  train a continuum so inference can back off
  (\(\gamma=0.9\) almost erases control: mIoU 0.50, back
  to the frozen number).

So they fine-tuned because (1) the *task* is “read this
warp as a command,” which is a new train pair, and (2) a
video score network trained on white spacetime noise is
off-distribution if you change that prior at test. They
did not need FT for image models, where there is no
temporal score to violate.

---

## Experiments they actually ran

**Warp algorithm (Table 1).** Moran’s I / K-S vs fixed
Gaussian, random Gaussian, bilinear / bicubic / nearest,
PYoCo, Control-a-Video, HIWYN, InfRes. Naive interp
fails Gaussianity. Their warp matches HIWYN’s Gaussianity
at 2.14 ms / 1024² vs HIWYN 55.2 ms.

**Training-free image V2V.** Super-res and relight:
quality ≈ other Gaussian warps; **best warping error**
(temporal) on IF (152 vs HIWYN 164). Naive interp:
quality collapse + sometimes fake-low warping error from
blur streaks.

**Video FT.** CogVideoX-5B T2V + I2V, rank-2048 LoRA,
4M captioned videos, 8×A100, 30k steps, ~40 GPU-days.
Same MSE as the CogVideoX recipe; only the noise is
warped. Also AnimateDiff / WebVid (qualitative).

**Downstream (Table 2 + user study, 40 people).**

| Task | Their claim |
|---|---|
| Local object drag (VIPSeg + 40 clips) | Beat SG-I2V, MotionClone, DragAnything. User win 82% / 90% |
| Motion transfer T2V (DAVIS 43) | vs DMT, MotionClone, MotionCtrl |
| Motion transfer I2V (DAVIS) | vs MotionClone, ImageConductor, frozen CogVideoX |
| Camera I2V (DL3DV 100, WonderJourney 19) | Beat MotionClone / ImageConductor on FID, flow, FVD |
| First-frame edit propagate | vs MotionClone, AnyV2V (qual) |
| Depth-warp fly-through | Crude depth video → warp noise → I2V |

VBench columns they report: subject, background, motion
smoothness, temporal flickering. **Not** Dynamic Degree.
This is a *control* paper, not a long-horizon freeze paper.

**Ablations.** \(\gamma\) sweep (0.2–0.9): tighter warp
→ better flow follow, \(\gamma\approx 0.5\) default.
33% / 12.5% data worse. CogVideoX-2B weaker than 5B.

---

## What this is not

They do not mid-denoise-shift a half-clean latent. They
do not wrap the grid. They do not try to raise official
Dynamic Degree on a 30 s causal student. They do not
claim inference-only warp on a frozen *video* model.

If we only `torch.roll` at step 2 of 4 on Self Forcing,
we inherit the two failures they already measured:
(1) naive transport breaks the noise the student knows,
(2) a video student trained on white noise will not
treat a spatial wrap as “move.” Their fix was paired FT
on a careful, Gaussianity-preserving, *temporal* warp.
That is occupied (CVPR 2025 Oral) unless the leftover
V2V protocol or the official Dyn+IQ table is the new
part — and that is Hypothesis 1/2 again.

---

## Do not

Launch 8-GPU DMD. Remake cite-128. Treat a circular
mid-step roll as “GwF without the LoRA.”
