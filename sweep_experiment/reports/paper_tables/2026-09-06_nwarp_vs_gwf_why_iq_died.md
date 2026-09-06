# Why leftover-flow extras looked like Go-with-the-Flow and died (2026-09-06)

Read after caption nwarp harvest IQ **49 / 54**. Canvas:
`canvases/nwarp-vs-gwf-why.canvas.tsx`.
Not a title. No remake of cite-128. No 8-GPU DMD.

---

## The feeling

The extra-only recipe *sounds* like Burgert et al. (Go-with-the-Flow,
CVPR 2025 Oral): transport Gaussian snow along optical flow, fill
holes, mix with a little plain snow (their γ). Imaging Quality then
fell from **71.54** to **49.18** (always-on) / **54.42** (live).
That is AdaSteer-class paint, not a −0.6 dip.

The story was close. The object we edited was not.

---

## Three different objects

| | Go-with-the-Flow | What we ran (`sf_nwarp`) | Your idea (`sf_pwarp`) |
|---|---|---|---|
| What moves | Initial noise volume \(x_T\) | Mid-step **extras** (the snow blended back) | The guessed **picture** after pass 1 (`pred`) |
| When | Once, before denoising | Every extra after pass 1, 30 s carry | After pass 1 of each 3-latent strip |
| Weights | Image models: frozen. Video models: **LoRA** | Frozen Self Forcing | Frozen Self Forcing |
| Spatial Gaussianity | Kept per frame (HIWYN warp) | Kept per frame if the field is white | Not a noise method. `pred` is a picture |
| Hole fill | Resample Gaussian | Resample Gaussian | Edge-repeat (no wrap, no noise in the photo) |
| KV | No causal leftover cache | Leftover unmoved; extras only | Leftover unmoved; **picture slides against it** |

Go-with-the-Flow never slides a half-clean latent against a
prefix sitting in memory. They never printed frozen CogVideoX +
warped \(x_T\). Video needed a paired fine-tune because the
spacetime noise prior changed.

---

## Smoking gun: the snow never moved

Truck-hood leftover (panda_0000), Farneback:

- `vy_px = 0.0079`, `vx_px = −0.0013` (pixels per pixel-frame)
- latent velocity ≈ **0.004** cells per latent frame
- After 21 extras: `y_acc = 0.245`, **`dy = 0`, `dx = 0`**

Integer transport did not fire. Always-on still changed the
video (tail 0.01139 ≠ host 0.01422) and smoke Imaging Quality
was **44.97**. Live *skip* on the same clip matched the host
tail exactly (0.01422). So the damage is the **enabled extra
recipe**, not the leftover caption, not a bad gate.

What the enabled recipe actually did, with zero shift:

1. Draw one spatial snow field.
2. **Reuse that same field** on every latent frame and every
   later extra (the 30 s carry).
3. Mix 50/50 with fresh white and renormalize
   \(\sqrt{(1-\gamma)^2+\gamma^2}\).

Each extra is then

\[
\text{extra} = \frac{\text{same frozen field} + \text{new white}}{\sqrt{2}}
\]

Half of every later snowflake is a **locked stencil**. That is
a new spacetime prior. Self Forcing was distilled on i.i.d.
white extras. Go-with-the-Flow said a video student will not
accept that without a paired train. We confirmed it on a
frozen student: the photograph dies, official Dynamic Degree
does not rise (0/8 always-on).

The last-5 s wiggle going **up** (+22%) is the stencil / mix
fighting the picture, not a living camera. Last-chunk motion
went the other way (−3.05 vs −0.94).

---

## What this is not

- Not “γ was wrong.” γ=0.5 is their default. Retuning γ on
  the same locked stencil is not a paper move.
- Not “the leftover flow was a bad teacher.” The flow was
  too *small* to shift a cell. A larger leftover would still
  change the prior; it would also move the stencil.
- Not Go-with-the-Flow without the LoRA. They warp \(x_T\)
  once. We correlated mid-step extras for 30 s.

---

## Your idea is a different kill test

After pass 1, **slide the guessed picture**, then let the
remaining passes finish a shifted scene. Ordinary white
extras (do not reuse the dead stencil). Direction = leftover
mean. Magnitude cannot be the leftover speed: 0.004 cells /
frame never crosses one pixel, so a faithful leftover slide
is a no-op on the truck clip (the extra-only lesson).

The force floor: **1 latent pixel per strip** (~8 pixels
every ~0.75 s) along the leftover’s dominant axis. Hole fill
= edge repeat. Same-wave live twin. Mixctx letters.

This reopens hole 4 on purpose (picture vs leftover in the
KV). lastmix / restep already edited a mid-strip guess and
did not save quality. If Imaging Quality dies again, stop.
Do not combine with extra-nwarp on the first wave.

---

## Do not

Call extra-only “almost GwF.” Retune γ. Remake cite-128.
Start 8-GPU DMD / their CogVideoX LoRA. Wrap the grid.
