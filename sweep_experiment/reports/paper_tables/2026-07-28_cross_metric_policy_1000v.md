# Cross-metric per-policy comparison — 1000v OOD preview (matched N=898)

**Date:** 2026-07-28
**Series:** `sweep_experiment/results/panda_ood_budget_1000v_preview`
**Pool:** matched N=898 (intersection of all policy video sets ∩ NOTTA anchor).
**VBench:** generated-only (`VBENCH_SUBDIR=vbench_results_geneval`).
**Realized metrics** follow each policy's per-video config choice (router picks
from `router_manifest.json`, oracle from best-PSNR winner). FVD from each
policy's `fvd.json`. Built by `scripts/build_cross_metric_policy_table.py`
(commit `ab9b3de`).

> Dims native: Subj/Bg/Aes/Motion/Dyn/Temp ∈ [0,1], **Imaging ∈ [0,100]** (MUSIQ).
> `VB-mean(norm)` = mean of 7 dims with Imaging scaled to [0,1]. ↓ lower better.

| Policy | FVD↓ | PSNR↑ | Subj↑ | Bg↑ | Aes↑ | Motion↑ | Dyn↑ | Imaging↑ | Temp↑ | VB-mean(norm)↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `always_notta` | 81.22 | 19.438 | 0.9578 | 0.9626 | 0.4751 | 0.9882 | 0.7082 | 62.81 | 0.9731 | 0.8133 |
| `fixed_S10_LR5e3` | 84.77 | 19.405 | 0.9571 | 0.9620 | 0.4742 | 0.9881 | 0.7016 | 62.83 | 0.9729 | 0.8120 |
| `router_psnr_ABC_13act` | **80.71** | 19.424 | 0.9570 | 0.9615 | 0.4778 | 0.9880 | 0.7149 | 63.31 | 0.9726 | **0.8150** |
| `router_psnr_ABC_12act` | 82.19 | 19.420 | 0.9574 | 0.9619 | 0.4751 | 0.9882 | 0.7105 | 62.83 | 0.9730 | 0.8135 |
| `router_vbench_ABC_13act` | 82.43 | 19.420 | 0.9580 | 0.9625 | 0.4710 | 0.9880 | 0.7082 | 61.92 | 0.9726 | 0.8114 |
| `router_vbench_ABC_12act` | 82.67 | 19.394 | 0.9574 | 0.9621 | 0.4750 | 0.9881 | 0.7138 | 62.72 | 0.9730 | 0.8138 |
| `oracle_best_psnr` | **72.28** | **19.793** | 0.9563 | 0.9614 | 0.4741 | 0.9880 | 0.7149 | 62.85 | 0.9730 | 0.8138 |

## Best cell per column

| Metric | Best deployable | Oracle (needs GT) |
|---|---|---|
| FVD↓ | `router_psnr_13act` 80.71 (< NO-TTA 81.22) | 72.28 (−8.94) |
| PSNR↑ | **NO-TTA 19.438** | 19.793 (+0.355) |
| VB-mean(norm)↑ | `router_psnr_13act` 0.8150 | 0.8138 (≈ NO-TTA 0.8133) |

## Interpretation

1. **No deployable policy beats NO-TTA by a meaningful margin on any metric
   family.** Fixed AdaSteer and all four routers sit within noise of NO-TTA on
   PSNR (19.39–19.44 vs 19.438) and VBench (4th-decimal). On PSNR, **NO-TTA is
   the best deployable policy** — a PSNR router cannot recover NO-TTA's mean
   because feature-based selection lands on the flat part of the grid.
2. **Oracle headroom is metric-specific and does NOT transfer.** Best-PSNR
   oracle gains +0.36 dB PSNR and −8.9 FVD, but its VBench is flat (0.8138 ≈
   NO-TTA 0.8133): the PSNR-winning config per video is not the VBench-winning
   one. No single per-video selection improves all metric families at once.
3. **VBench routers don't robustly win VBench.** `router_vbench_13act` edges
   Subj/Bg but is worst on Aes/Imaging → VB-mean 0.8114 (below NO-TTA). Per-dim
   confirmation of VBench un-routability (R²(gain) ≤ 0).

## Takeaway

Real per-video oracle headroom exists on **PSNR and FVD**, is **not routable**
from cheap features, and **does not extend to VBench**. Deployable AdaSteer
(fixed or routed) ≈ NO-TTA across all three metric families. Consistent with
`2026-07-27_matched_fvd_1000v_corrected.md` and `2026-07-28_router_fvd_1000v.md`.
