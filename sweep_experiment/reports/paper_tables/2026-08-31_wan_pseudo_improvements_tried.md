# Why pixels are blank, and what we tried on Pseudo (2026-08-31)

## Why PSNR / SSIM / LPIPS / FVD are missing

These are LongCat reconstruction metrics. They need a **real 30 s
future**. We confirmed every cite clip has one. They are **not**
the official headline (that is full-clip VBench + Dyn%).

| Metric | Why blank | Finish |
|---|---|---|
| PSNR / SSIM | Job **16678705** scored methods in order. Preempted at 2h20 after Self Forcing only (n=128, PSNR 9.25 / SSIM 0.279). Rolling / Pseudo / Always never started. | `bash wan_experiment/sbatch/submit_v2v_pixel128.sh` (skip-existing). |
| LPIPS | Scorer `import lpips` fails in `self_forcing`. SF row is `None`. | Install `lpips` or drop the dim. |
| FVD | **Never launched.** I3D must see **aligned tails**. A full mp4 includes the real 2 s prefix. n=128 needs `--force`. | After pixel tails exist. Do not score the generated 32 s files. |

Do not cite Self Forcing PSNR 9.25 as a bake-off.

## Pseudo-future Search (the method we are improving)

Once on the opening: hide last 3 prefix latents (~0.7 s), fire
k=4 if an extra seed beats do-nothing MAE on that real B, then
motion+trust pick on every invented chunk. γ=0, k=4. Code
`sf_pseudo`. Gate **90 fire / 38 skip** on 128. Cite: tail
0.0157, Dyn **47.7% (61/128)**, subject 0.660, IQ **72.38**,
mean wall **304 s**.

## Improvements we ran

| Attempt | Question | N | Call |
|---|---|---|---|
| Always-search (no gate) | Does the opening gate skip live videos? | 32, 128 | **Ablation, keep.** Always Dyn 50.8% (65) vs 47.7% (61). Subject/IQ match. Gate almost free. |
| Pseudo on Rolling | Same hold-out on the cheap host? | 32 | **NO.** Gate dead. |
| Prefix-match / appear pick | Pick the tail that looks like the opening? | 32 | **NO.** Freezes motion. |
| AdaSteer | Fit weights on the prefix? | 8 | **NO.** IQ 43 / 51 / 18. |
| lastmix / bpseudo / restep | Rewrite inside the 4-step block? | 8 (restep 5) | **NO.** Identity or subject 0.575. |
| Intra-chunk resample | Resample if motion/appear sick? | 8 | **NO.** Gated ≡ always. Subject 0.632. |
| Keep-picture (nudge / next-seed / wiggle / latmot) | Smaller rewrite so subject holds? | 8 × 14 | **NO.** Family closed. Subject misses 0.68. RF IQ 66–67. |
| CachedSearch | Same pick, cheaper KV? | 8 | **NO.** Same tail, **slower** (389 vs 360). |
| Re-gate each chunk | New hold-out on committed history? | 8 | **NO.** Alive (6/5/6/7/8/6), no lift, +53% wall. |

## Not tried (on purpose)

CFG / shift (dead on this DMD). TTC / LoRA (locked out). Retune
γ or k on cite 128. I2V scale-up. Video-T1 prune and
search-only-early-chunks are the leftover cheapen ideas after
CachedSearch failed. Rolling window-exit is a different path.
