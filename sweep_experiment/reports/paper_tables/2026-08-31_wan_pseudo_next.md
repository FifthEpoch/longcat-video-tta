# Building on Pseudo-future Search (2026-08-31)

The 128-video official row is good news vs the paper baseline.
Do not retune γ or k on this 128. Do not remake videos. Always
official (**16674378**) still decides how much of the Dyn% is
the gate vs the pick.

## What 128 already says

| | Self Forcing | Rolling | Pseudo |
|---|---:|---:|---:|
| tail | 0.0119 | 0.0158 | **0.0157** |
| Dyn% | 32.8% (42) | 28.9% (37) | **47.7% (61)** |
| subject | 0.666 | **0.685** | 0.660 |
| IQ | 72.07 | 71.52 | **72.38** |
| wall | ~2 min | **45 s** | ~5 min (90/128 fired) |

Gate: **90 fire / 38 skip** (same ~70% as N=32). The 38 skips
are the mean-cost cut vs Always.

vs Self Forcing: tail +32%, Dyn% +15 points, subject holds, IQ
up. That is a quality method, not only an efficiency controller.
vs Rolling: better living-motion and IQ, tied tail, worse
subject, much more expensive.

## What to try next (builds on this gate, not a new rewrite)

Wait for Always 128 official first. Then one of:

1. **If Always Dyn% ≈ Pseudo (~48%).** The gate is almost free
   on quality. Next paper move is **cheapen the fired path**
   (CachedSearch / prune / search only early chunks). Same pick,
   less than k=4 every chunk. That is the only way toward the
   cost bar.

2. **If Always Dyn% is clearly higher.** The once-on-opening
   gate is skipping live videos. Do **not** loosen γ on this
   128. Next probe (N=8, new spec): **re-gate each chunk** —
   hold out the last 0.7 s of *committed* history as a new B,
   same MAE rule, decide whether the *next* chunk searches.
   Same method family. Not a pixel rewrite.

3. **Do not do.** Mid-chunk mix / nudge / intra (closed).
   Prefix-match pick (froze motion). Pseudo on Rolling (dead
   gate). Weight updates. CFG / shift. Retune γ or k on the
   cite set.

CachedSearch is already in the runner (`cached_bon`). It is how
we *pay* for a fired clip, not a new picker. Video-T1-style
prune is the same sentence.

## PSNR / SSIM / LPIPS / FVD

These are the LongCat short-horizon metrics. They need a
**reference**. This protocol invents 30 s after a ~2 s real
opening (`1+4×8 = 33` prefix frames).

| Metric | Valid here? | Why |
|---|---|---|
| PSNR / SSIM / LPIPS vs the **true 30 s future** | Only if that Panda clip still has ~480 frames after the opening | Panda-70M clips are usually a few seconds. Most 128 will have **no GT tail**. |
| PSNR vs the **opening** | No | That is Prefix-match. It rewards freeze. We already know it kills Dyn%. |
| FVD / FID vs the Panda **pool** (unpaired) | Yes, as a distributional check | Existing `sweep_experiment/scripts/eval_fvd.py`. N=128 is below their 256 default; use `--force`. Not a paired “this clip matches its own future.” |

Do not put PSNR in the headline table unless a duration audit
shows a real leftover window, and then score **only that
overlap** (probably a few seconds, not 30 s). LongCat already
taught us PSNR and VBench can disagree. Official claim stays
full-clip VBench + Dyn%.

First CPU check: how long are the 128 source mp4s, and how
often did Pseudo fire? Both are on disk. No new generate.
