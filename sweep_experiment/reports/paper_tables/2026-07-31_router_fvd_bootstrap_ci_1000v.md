# Deployable-router FVD bootstrap CI — paired ΔFVD vs NO-TTA, 1000v preview (matched N=898)

**Date:** 2026-07-31
**Job:** `router_fvd_ci` (15076548, COMPLETED, 2:21)
**GT cache:** `panda_ood_budget_1000v_preview_longcat.npz` (939 ref) · gen-only
window [14 cond | 14 gen] · bootstrap B=2000 · baseline `always_notta` · paired
resample (FVD estimator bias cancels between policies).
**Scripts:** `scripts/build_router_fvd_dirs.py` (compose OOF router picks into
matched-N policy dirs) + `sweep_experiment/scripts/fvd_bootstrap_ci.py`.

Routers are leakage-free 5-fold **OOF** ridge (argmax predicted metric), block
A+B+C; 12-action = configs only, 13-action = configs + NO-TTA (skip). VBench
routers trained on the corrected **generated-only** VBench scores.

## Result — all deployable policies are FVD-null vs NO-TTA

| Policy | deployable? | point FVD | ΔFVD vs NO-TTA [95% CI] | CI excludes 0? |
|---|:--:|--:|---|:--:|
| always_notta | — | 81.22 | — | — |
| fixed (S10_LR5e3) | yes | 84.77 | +3.15 [−5.79, +12.19] | no (null) |
| **router_psnr_ABC_12act** | yes | 82.19 | **+0.27 [−7.70, +6.37]** | no (null) |
| **router_psnr_ABC_13act** | yes | 80.71 | **−1.14 [−6.86, +3.06]** | no (null) |
| **router_vbench_ABC_12act** | yes | 82.67 | **+0.75 [−9.56, +9.89]** | no (null) |
| **router_vbench_ABC_13act** | yes | 82.43 | **+0.78 [−4.18, +4.45]** | no (null) |
| oracle (best-PSNR/video) | **no** | 72.28 | −10.37 [−21.79, −2.04] | yes (improves) |

## Interpretation

- **Every deployable policy — the fixed config and all four trained routers —
  has a NULL FVD effect vs NO-TTA** (all CIs span 0). This is the FVD analogue
  of the PSNR/VBench null: TTA does not measurably shift the generated feature
  distribution for any policy you could actually deploy.
- **Only the non-deployable PSNR-oracle moves FVD** (−10.4, CI excludes 0), and
  the router significance analysis already showed that per-video selection is
  max-over-noise / unroutable — a real router captures ~0 of it, which is exactly
  what these router rows confirm at the distribution level.
- The 13-action PSNR router (which can choose skip=NO-TTA) sits closest to
  NO-TTA (−1.14, tightest CI), as expected since it mostly declines to adapt.

## Caveat

Absolute bootstrap FVD means (~150–155) are inflated by resample-with-replacement
duplication (FVD covariance bias); trust the **paired ΔFVD CIs** and the **point
FVD** levels, not the absolute bootstrap mean.

## Bottom line

There is no deployable FVD win from AdaSteer on this pool — not from the best
fixed config, and not from any PSNR- or VBench-motivated router. The only FVD
improvement lives in the non-deployable oracle.
