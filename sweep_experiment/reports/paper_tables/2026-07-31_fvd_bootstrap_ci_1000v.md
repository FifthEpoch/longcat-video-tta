# FVD bootstrap CI — paired ΔFVD vs NO-TTA, 1000v preview (matched N=898)

**Date:** 2026-07-31
**Job:** `fvd_boot_ci` (15044183, COMPLETED, 30 min)
**GT cache:** `panda_ood_budget_1000v_preview_longcat.npz` (939 ref videos) ·
gen-only window [14 cond | 14 gen] · bootstrap B=2000 · baseline `always_notta`.
**Script:** `sweep_experiment/scripts/fvd_bootstrap_ci.py` (I3D-feature-level,
paired resample of video ids so the FVD estimator bias cancels).

## Result

| Policy | point FVD | ΔFVD vs NO-TTA [95% CI] | CI excludes 0? |
|---|--:|---|:--:|
| always_notta | 81.22 | — | — |
| fixed (S10_LR5e3) | 84.77 | **+3.15 [−5.79, +12.19]** | **no (null)** |
| oracle (best-PSNR/video) | 72.28 | **−10.37 [−21.79, −2.04]** | yes (improves) |

## Interpretation

1. **The deployable fixed-config TTA has a NULL FVD effect vs NO-TTA** (CI spans
   0). This matches the PSNR (+0.02 dB) and VBench null results — TTA does not
   shift the generated feature distribution. It definitively closes the earlier
   "TTA doubles FVD" scare (fixed=216 in the 2026-07-27 matched run): that was
   the `_index_grid_videos` symlink-duplication bug, now fixed. On de-duplicated
   data fixed-config FVD ≈ NO-TTA.
2. **The PSNR-oracle improves FVD (−10.4, ~13%), but it is NOT deployable.** It
   selects the best-PSNR config per video using ground-truth PSNR, and the
   router significance analysis (2026-07-31) showed that selection is
   max-over-noise / unroutable (a real router captures ~0 of the PSNR headroom).
   So this FVD gain is an oracle upper bound, not a method result.

## Caveat (read before citing absolute numbers)

The **absolute** bootstrap FVD means (~152 vs the 81 point estimate) are inflated
because bootstrap resamples videos WITH REPLACEMENT, duplicating clips and biasing
FVD's covariance term upward. Trust the **paired ΔFVD CI** (duplication cancels
between policies on the shared resample) and the **point FVD** levels (81/85/72),
not the absolute bootstrap FVD mean.

## Bottom line

FVD is null for the deployable policy, consistent with PSNR/VBench. There is no
deployable FVD win from AdaSteer on this pool.
