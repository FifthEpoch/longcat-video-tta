# Success bar + neighbor papers (2026-08-30)

Locked from the user. Mid-chunk rewrite is **closed** unless the
keep-picture 8-video run passes the old quality letter. Guidance /
shift **closed**. Weight updates **closed**, but we owe a written
failure-mode for why sampling-space is the claim. Intra-chunk is an
**experiment paragraph**, not a method.

Do not launch a new GPU family tonight. Cite-128 and keep-picture
are already running.

## What “success” actually means

**Wanted:** cheaper than Rolling Forcing, and about as good as
Rolling Forcing.

**Cost we already measured** (caption, 32 videos, seconds per clip):

| | Median wall | Mean wall |
|---|---:|---:|
| Rolling Forcing (do nothing) | **45 s** | 45 s |
| Self Forcing (do nothing) | 113 s | 196 s |
| Always-search on Self Forcing | 348 s | 348 s |
| Pseudo-future Search | 357 s | 304 s |

Rolling is already the cheap host. Pseudo and Always-search are
**seven to eight times** slower than Rolling. A method that is
literally cheaper than Rolling *and* matches Rolling has to do
**less work than one Rolling pass**. That is not another k=4 search.

Honest restatement (say if this is wrong):

1. **Quality bar** = Rolling’s official 128-video row (subject,
   image quality, percent of clips that look dynamic, tail motion).
2. **Cost bar** = much cheaper than Always-search (348 s). Stretch
   goal = near Rolling (45 s).
3. **Ours** = a controller on a frozen Self Forcing student, not
   “we switched the host to Rolling.”

If 128-video Pseudo matches Rolling on quality, the remaining paper
is **cheapen it** (search less often, prune losers early), not invent
a new mid-chunk edit.

## Neighbor papers — what we take, what we do not copy

| Paper | Their move | For us |
|---|---|---|
| **SAVi-DNO** | Optimize the *noise* at test time on a frozen backbone | Same class. We already have a leakage-free port. Use as a **noise-opt baseline** in the parameter-space section, not as the method. Old LongCat port was broken; do not cite those numbers. |
| **TANGO** (ECCV 2026) | Predicted noise should look like Gaussian; if not, the trajectory is dying. Then they adapt a small LoRA. | Keep the **critic** (does the next-step noise look healthy?). Drop the LoRA (weights closed). We already ran a first-step residual probe on Wan; it matched do-nothing. So TANGO’s *trigger* may be dead on 4-step DMD. Write that. |
| **Pathwise TTC** | Do not train. Re-anchor appearance to the **first frame** at low-noise steps. They also show weight test-time opt fails on distilled AR models. | The failure of weight TTO is **our AdaSteer exhibit**. Their fix (pull toward frame 0) is the cousin of Prefix-match, which **froze motion** for us. Do not launch more TTC on Wan. Cite as the other sampling-state hypothesis we already falsified as a motion method. |
| **Latent beam search** (NeurIPS 2025) | Search in diffusion latent space at inference | This is Always-search with a fancier tree. Too expensive for the success bar unless we *prune*. |
| **Video-T1** (ICCV 2025) | Best-of-N is O(length × k). **Tree-of-Frames** expands and prunes so you do not pay full k on every frame. | This is the paper we sit next to for **cheapening** Pseudo / Always. Next method, if 128 quality holds: prune or search-once, not rewrite pixels. |
| **VISTA** (CVPR 2026) | Agent loop: generate, verify, try again | Useful as **negative** for our bar. Iterative verify is slower than Rolling. Do not build an agent. |
| **Diffusion Tree Sampling** (NeurIPS 2025) | General inference-time search / alignment for diffusion | Theory neighbor for Video-T1-style prune. Not a submit. |
| **Reward Forcing** (CVPR 2026) | Not TTA. Which *rewards* actually help streaming video. | Explains our old lesson: the handcrafted “pretty” score can fight image quality. Official VBench (full clip, Dyn = percent of clips) stays the judge. |

## What we write down (no new jobs)

**Parameter-space failure (needed for the claim).**  
AdaSteer on this Wan V2V setup: image quality 43 / 51 / 18 vs Self
Forcing ~71. Pathwise TTC’s toy result is the same story: optimizing
weights at test time on a distilled AR student is unstable. That
section is how we say “sampling-space is the way” without handwaving.

**Intra-chunk failure (experiment paragraph).**  
50/50 last-step mix, redraw last 0.7 s, redo two of four steps:
subject fell to ~0.63 and image quality to 66–70. Gate ≈ always-on
(sharpness was a bad sick signal). Self Forcing intra / restep also
ran out of GPU memory until we stopped rewriting. **Reason:** four
denoising steps means “a small rewrite” is a new picture. Closed
unless keep-picture (10% nudge / residual / first-latent lock)
passes subject ≥ 0.68 and image quality ≥ 70.5.

## What we do after the running jobs land

1. Score **128-video Pseudo vs Always vs Self Forcing vs Rolling**.
   That decides whether we have Rolling-like quality on our host.
2. If yes: **cheapen** (Video-T1 prune / search only early chunks /
   CachedSearch). That is the only path that can approach the cost
   bar.
3. If no: say so. Do not fill the gap with another mid-chunk mix.
4. Keep-picture 8-video: harvest, then close or keep. No scale-up
   on a letter fail.

No TTC submit. No LoRA. No VISTA loop. No guidance/shift.
