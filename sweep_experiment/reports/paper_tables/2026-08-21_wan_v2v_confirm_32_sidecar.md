# V2V N=32 confirm — sidecar-paired (complete)

**Date:** 2026-08-21
**Series:** `v2v_panda_confirm_32v`
**N:** 32/32 sidecars both methods. Cite **medians**.
**Supersedes:** unpaired `summary.json` −26% (`2026-08-21_wan_v2v_confirm_32_and_tricks.md`)
and the n=12-only pair (`2026-08-21_wan_v2v_pervideo_retract.md`).
**VBench:** jobs **16122823** (confirm notta/seed_bon) and **16122824**
(tricks hist_drop/hinge_bon), both PD `QOSGrpGRES` at submit.

## Headline

| Split | N | notta | seed_bon | vs notta | seed>notta |
|---|---:|---:|---:|---:|---|
| **all 32** | 32 | 0.01353 | **0.01235** | **−8.8%** | **12/32** |
| first 8 (bake-off prefix) | 8 | 0.01675 | 0.02250 | **+34%** | 4/8 |
| last 24 | 24 | 0.01251 | 0.01017 | **−19%** | 8/24 |

Locked rule: promote past N=8 only if median tail motion **beats** notta.
**FAIL.** Do not write the paper around seed-BoN.

The N=8 +35% is real and bit-matches the bake-off (same 8 files, same
medians). It does not survive the next 24. Mean Δ over all 32 is ~0
(−0.00008): a few large lifts, more small cuts.

## By prefix motion (notta tail)

| Band | N | notta med | seed med | vs notta | W/L |
|---|---:|---:|---:|---:|---|
| quiet (<0.012) | 14 | 0.00925 | 0.00828 | −11% | 6/8 |
| mid (0.012–0.020) | 11 | 0.01486 | 0.01544 | +4% | 6/5 |
| **hot (≥0.020)** | 7 | 0.02637 | 0.02449 | −7% | **0/7** |

seed_bon **never** raises tail motion on an already-hot prefix. The
two-sided score damps them (same family as I2V matching a near-still,
now matching “don’t exceed this motion”). The N=8 win was two large
lifts on 0000 / 0007 plus a first-8 list that over-weighted mid clips.

## Per-video (sidecars)

| video | notta | seed_bon | Δ | |
|---|---:|---:|---:|---|
| panda_0000 | 0.01486 | 0.02761 | +0.01275 | first8 win |
| panda_0001 | 0.00641 | 0.00933 | +0.00292 | first8 win |
| panda_0002 | 0.02350 | 0.01787 | −0.00563 | first8 hot |
| panda_0003 | 0.02236 | 0.01917 | −0.00319 | first8 hot |
| panda_0004 | 0.03062 | 0.03061 | −0.00001 | first8 hot ~tie |
| panda_0005 | 0.00686 | 0.00671 | −0.00014 | first8 ~tie |
| panda_0006 | 0.01864 | 0.02584 | +0.00720 | first8 win |
| panda_0007 | 0.01210 | 0.02617 | +0.01407 | first8 win |
| panda_0008 | 0.01162 | 0.00722 | −0.00440 | |
| panda_0009 | 0.00970 | 0.00702 | −0.00268 | |
| panda_0010 | 0.01086 | 0.00852 | −0.00234 | |
| panda_0011 | 0.00881 | 0.01016 | +0.00135 | |
| panda_0012 | 0.01074 | 0.00690 | −0.00384 | |
| panda_0013 | 0.00613 | 0.00680 | +0.00067 | |
| panda_0014 | 0.01822 | 0.01544 | −0.00278 | |
| panda_0015 | 0.00880 | 0.00803 | −0.00077 | |
| panda_0016 | 0.01750 | 0.01470 | −0.00280 | |
| panda_0017 | 0.01571 | 0.01018 | −0.00553 | |
| panda_0018 | 0.01339 | 0.03052 | +0.01714 | largest later win |
| panda_0019 | 0.01470 | 0.00997 | −0.00473 | |
| panda_0020 | 0.00646 | 0.00656 | +0.00010 | |
| panda_0021 | 0.00981 | 0.00856 | −0.00125 | |
| panda_0022 | 0.03579 | 0.02678 | −0.00901 | hot |
| panda_0023 | 0.01368 | 0.01406 | +0.00038 | |
| panda_0024 | 0.01093 | 0.01442 | +0.00350 | |
| panda_0025 | 0.02637 | 0.02449 | −0.00188 | hot |
| panda_0026 | 0.01392 | 0.02287 | +0.00895 | |
| panda_0027 | 0.03519 | 0.02720 | −0.00799 | hot |
| panda_0028 | 0.02026 | 0.01472 | −0.00554 | hot |
| panda_0029 | 0.00877 | 0.01064 | +0.00187 | |
| panda_0030 | 0.01066 | 0.00871 | −0.00196 | |
| panda_0031 | 0.01618 | 0.00922 | −0.00696 | |

## What this means for hist_drop

On the bake-off 8, hist_drop was a **small increment on seed_bon**
(7/8, +0.0008…+0.0043) and still lost on the two hot prefixes (0002,
0003). Same damping family. N=8 +42% vs notta is still worth VBench
(IQ / subject / Dyn). It is **not** a reason to scale hist_drop to 32
until that VBench lands — the seed picker it increments just failed
the confirm.

## VBench (in flight)

```
16122823  confirm_32v  METHODS="notta seed_bon"   CLIPS=full
16122824  tricks_8v    METHODS="hist_drop hinge_bon"  CLIPS=full
```

Both PD `QOSGrpGRES` on `h200_cour` at 02:33. No commas in `--export`.
squeue typo `wc3913` is not a user.
