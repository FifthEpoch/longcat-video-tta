# Mid-step noise warp — pipeline holes (2026-09-05)

Working note after the user asked how viable
“denoise as usual, then warp the remaining noise
closer to clean, keep Gaussianity.” Canvas:
`canvases/midstep-warp-holes.canvas.tsx`.

Not a submit. No GPU. Frozen Self Forcing 4-step
block in `run_v2v_chunked.py` `_denoise_one_block_guided`.

---

## Viability (honest)

**Low as stated** (“intervene closer to clean” +
spatial wrap). The timing is inverted, and a spatial
wrap of something that is still Gaussian does nothing.

**Repairable as a different object:** warp the
*remaining* `extra` **early**, **across the 3 latent
frames** with a GwF-style transport (spatial Gaussian
kept, temporal correlation added), do **not** wrap the
grid, do **not** roll the predicted clean clip against
the KV. That is a cheap N=8 kill test. It is still
“edit the path.” Go-with-the-Flow already owns the
fine-tuned version.

---

## What the loop actually is

One block = 3 latents, shape `[1, 3, 16, 60, 104]`.
Steps ≈ `[1000, 750, 500, 250]` (RF live floor 556).

```
noisy = ε0                          # i.i.d. Gaussian at t=1000
for t in steps:
    pred = G(noisy, t, KV)          # structured “clean” estimate
    if not last:
        extra = randn_like(pred)    # NEW i.i.d. Gaussian
        noisy = add_noise(pred, extra, t_next)
# then KV write of pred (context_noise=0)
```

`add_noise` is the flow-match mix: the next `noisy` is
a **weighted sum of structured `pred` and white
`extra`**. It is not Gaussian. Only `extra` (and the
step-0 `ε0`) is Gaussian.

So at the “middle” (after step 2, before the last two
steps) three tensors exist:

| Tensor | Gaussian? | Has motion structure? |
|---|---|---|
| `pred` | No. Half-clean video | Yes |
| `extra` | Yes. i.i.d. | No |
| `noisy` after `add_noise` | No. Mix | Yes, from `pred` |

You cannot warp one object that is both Gaussian and
motion-carrying. That is hole 1.

---

## Holes, in pipeline order

### H1 — The Gaussianity / effect paradox

A spatial wrap of i.i.d. `extra` is still i.i.d.
`extra`. Distribution unchanged, model sees a
different seed, **no pan**.

A spatial wrap of `pred` or of `noisy` moves the
picture. Those tensors are **not** Gaussian. Keeping
“Gaussianity of the noise” does not protect them.
GwF’s trick was: **spatial** i.i.d., **temporal**
correlation. That is the only way both sentences can
be true.

**Address:** Do not roll `pred`. If anything, GwF-transport
`extra` along time (frame 0 → 1 → 2 of the block)
and leave each frame spatially white.

### H2 — “Closer to clean” is when noise has no energy

User instinct: intervene late, once the picture
exists. For a *noise* method the opposite is true.

At t=1000, `noisy` is ~all `extra`. A structured
`extra` can steer. At t=250, `add_noise` is almost
`pred`. Warping `extra` changes grain. Warping `pred`
is an image edit (lastmix/restep class).

GwF warps \(x_T\), not \(x_{t\approx 0}\).

**Address:** If the hook is noise, put it on `ε0` and
on every `extra` **from step 0**. Late-only is a
detail twitch, not motion.

### H3 — `extra` is redrawn every step

A warp of `ε0` dies at the first `add_noise`. The
next `extra` is fresh white. Mid-step warp of one
`extra` dies at the next redraw.

**Address:** Persist the same warped-noise *recipe*
on every `extra` (and on `ε0`): same flow, same
GwF transport, new spatial Gaussian only in the holes.
Or stop redrawing and reuse a transported field.

### H4 — KV and RoPE still see the unshifted world

The prefix and every locked chunk sit in the KV at
their original spatial layout. `current_start` is
temporal RoPE, not a 2D scroll.

If you roll `pred` (or `noisy`) by one latent pixel,
the current tokens are a translated bathroom; the
cache is the untranslated leftover. Attention is
now a spatial seam. That is a new path the student
never trained. Wrap-around adds a second seam (left
edge glued to right).

GwF never does this: they warp only \(x_T\), then
the whole clip denoises **together** with no causal
cache of an unwarped prefix.

**Address:** Do not translate `pred` relative to KV.
Only change `extra`. If you must translate content,
you would have to translate the prefix tokens too —
that is a different, worse idea (paints the leftover).

### H5 — Torus wrap is not Gaussianity-preserving transport

GwF / HIWYN drop particles that leave the frame and
**resample** holes so each frame stays \(\mathcal{N}(0,1)\)
with no spatial autocorrelation. A wrap creates a
hard edge: two unrelated columns become neighbors.
Locally that is the bilinear failure in their Table 1.

**Address:** Hole-fill with fresh Gaussian, density
renormalize. No `torch.roll` of the picture.

### H6 — Where the flow comes from (chicken / stale)

Mid-step `pred` is noisy. RAFT on it is a bad
teacher and a feedback loop (use estimated motion to
create motion). Prefix flow is real but **stale**
after 10 s (the caption-staleness seam). We only
store prefix **magnitude** today, not direction.

**Address:** First kill test: **constant** velocity
\((v_x, v_y)\) from the leftover, frozen for the
whole 30 s. No mid-step RAFT. If that does not move
official Dyn without flicker, a live detector will
not save it.

### H7 — Three latents is a very short flow

A block is 3 latent frames (~0.75 s). GwF warps a
full clip’s \(x_T\). Temporal transport across 3
slots can encode a small velocity. It cannot encode
a 30 s living scene. Across chunks the recipe must
be repeated or the motion stutters at every lock.

**Address:** Same flow recipe on every block’s
`ε0`/`extra`, locked to the leftover. Expect
chunk-boundary stutter as a measured failure mode.

### H8 — Four steps, not fifty

lastmix / restep already hooked “after step 2, change
something, finish the last two steps.” Identity or
NO. There is almost no remaining denoise to hide an
edit.

**Address:** The honest noise hook is step 0 + every
`extra`, not “middle.” If we still want a mid hook,
it is on `extra` only, and we already know the
budget is two steps.

### H9 — Rolling Forcing’s window is several noise ages

RF’s live unit has chunks at different \(t\). “The”
noise in the window is not one field. A wrap that is
well-defined on an SF block is ill-defined on an RF
span.

**Address:** SF host only for any kill test.

### H10 — Frozen student never trained on structured `extra`

Same prior hole GwF used to justify LoRA on
CogVideoX. Image models accepted warped noise
training-free; video models did not. Ours is a
causal video student.

**Address:** N=8 can still kill. If it twitch-paints,
do not scale; that is their FT paper, occupied,
unless the pair is “real leftover + official Dyn.”

### H11 — Official Dynamic Degree can flip from the seam

A wrap seam or a late grain warp is high-frequency
flow in the top 5% of pixels. That is the RAFT bit
(11.25 px on 480p). mixctx already: Dyn 8/8, flicker
0.978. A “win” on Dyn with flicker in the twitch
band is DOLLAR, not motion.

**Address:** Hold only if Dyn↑ **and** IQ holds
**and** flicker stays off the twitch band **and**
subject does not drop. Same letters as mixctx.

### H12 — V2V leftover already is motion

The 2 s prefix is real optical flow sitting in the
KV. The student *can* continue it (IQ ~72). The
failure is freeze later, not “there was never a
flow.” Injecting a second, synthetic flow on
`extra` can fight the leftover (two motions) or be
ignored (H10).

**Address:** Constant velocity should be the
leftover’s own mean flow, not an arbitrary +x.
Fighting the leftover is a different bug.

---

## What a repaired hypothesis would say

Denoise as usual. From **step 0**, replace every
white `extra` (and `ε0`) with a GwF-style field:
spatially i.i.d. Gaussian, temporally transported
by a **frozen leftover velocity**, holes resampled
not wrapped. Do not touch `pred`. Do not roll
against the KV. Judge on caption N=8 with the
mixctx letters.

That is viable as a **kill test**, not as a title.
If it works, GwF already named the train-time
version. If it fails, H2/H4/H10 were the reason.

---

## Do not

Launch tonight. Remake cite-128. Warp `pred`.
`torch.roll` the picture. Put the hook only at the
last step. Start from the RF host.
