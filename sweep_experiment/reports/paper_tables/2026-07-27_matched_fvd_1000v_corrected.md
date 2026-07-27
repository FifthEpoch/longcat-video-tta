# Matched FVD — 1000v OOD preview (CORRECTED, post duplication-bug fix)

**Date:** 2026-07-27
**Series:** `sweep_experiment/results/panda_ood_budget_1000v_preview`
**Eval:** `eval_fvd.py` + shared GT cache `panda_ood_budget_1000v_preview_longcat.npz`,
generated tail `video[48:62]` (cond=14, gen=14), matched to the common set
`NOTTA ∩ manifest` (N=898), one protocol for all policies (`INTERSECT_NOTTA=1`).

> **Supersedes** the pre-fix matched FVD (fixed=198–217, oracle=169–184), which
> was inflated by a symlink-duplication bug in `_index_grid_videos`
> (898 ids collapsed onto 442 unique files). See `ANALYSIS_LOG.md`
> entries 2026-07-27. Fixed by resolving config mp4s via the
> `(psnr,ssim,lpips)` filename fingerprint + bijectivity guard (commit `2b23b8b`).

## Corrected matched FVD (N=898, same video IDs, same GT cache)

| Policy | N linked | FVD | Δ vs NO-TTA | Δ% vs NO-TTA |
|---|---:|---:|---:|---:|
| always_notta | 898 | 81.22 | — | — |
| fixed_S10_LR5e3 (deployable) | 898 | 84.77 | +3.55 | +4.4% |
| oracle_best_psnr (upper bound) | 898 | 72.28 | −8.94 | −11.0% |

## Interpretation

- **Fixed AdaSteer is FVD-neutral.** 81.2 → 84.8 is within run-to-run noise and
  consistent with the online per-run config FVDs (~67–69) and the flat PSNR
  (+0.02 dB). The earlier "TTA doubles FVD" result was entirely an artifact of
  the duplication bug — not a real distributional effect.
- **PSNR-oracle routing modestly reduces pooled FVD** (−11%): the per-video
  clips selected by best-PSNR are also slightly closer to the GT distribution.
  This is an upper bound (oracle requires GT to select the winner) and is not a
  deployable number.

## Cross-checks

- Consistent with: online sweep FVD (S10_LR5e3=68.8, S20_LR1e2=67.6, N=998);
  PSNR flat across the grid (19.44–19.49 dB); LPIPS/SSIM near-identical.
- `always_notta` unchanged vs pre-fix (81.5 → 81.2, tiny GT-cache/N re-run
  jitter) — NOTTA never used the buggy indexer, so it is the anchor.
