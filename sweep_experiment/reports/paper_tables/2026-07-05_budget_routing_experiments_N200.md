# Budget routing experiments — pilot N=200 (VBench total objective)

**Date:** 2026-07-05  
**Source:** `sweep_experiment/reports/per_video_analysis/2026-07-05/budget_routing_experiments/`  
**Fixed comparator:** S10_LR5e3  
**Oracle headroom (mean):** +0.1402 VBench total  
**Cluster:** `panda_ood_budget_pilot`, 12 configs × 200 videos, Phase-0 features @ 2026-07-06

## Total-VBench routing (apples-to-apples)

Methods trained/evaluated on **VBench total** (headroom = 0.1402 for all rows below).

| Method | Config match % | Captured % | Δ vs fixed | Deployable? |
|---|---:|---:|---:|---|
| proxy_psnr_all | 15.0 | **11.5** | +0.0161 | No — needs all-config PSNR |
| probe_simulated | 20.5 | **9.8** | +0.0137 | Partial — needs 2–3 probe TTA runs |
| baseline_linear_total | 18.0 | **9.0** | +0.0126 | Yes (features only) |
| coarse_steps_lr | 18.0 | 8.3 | +0.0116 | Yes |
| composite_psnr_ridge | 18.0 | 7.5 | +0.0105 | No — uses all-config PSNR cols |
| mlp_shallow | 17.0 | 7.2 | +0.0102 | Yes |
| pairwise_gbm_top4 | 12.5 | −0.8 | −0.0012 | Yes |
| proxy_bestof3_psnr | 5.5 | −3.1 | −0.0044 | Yes (3 probe configs) |
| pairwise_logistic_top4 | 15.0 | −7.4 | −0.0104 | Yes |

**Ceiling reference:** VBench-total oracle vs fixed = **+0.1402** (+100% captured). Quintile-adaptive deployable gate ≈ **8%** captured (prior work).

## Per-dimension routing (different objective — not in table above)

Dim routers optimize a **single VBench dimension**; captured % and headroom are on that dim’s scale (not VBench total). Do not rank against total-VBench rows.

| Method | Captured % (on target dim) | Notes |
|---|---:|---|
| dim_imaging_quality | 98.3 | Routes for IQ; total-VBench impact **not yet measured** |
| dim_aesthetic_quality | 0.0 (nan) | Negative vs mixed baseline |
| dim_dynamic_degree | 0.0 (nan) | — |
| dim_subject_consistency | 0.0 (nan) | — |

## Decisions

1. **999v × 12 for total-VBench routing:** **NO-GO** — spread is 7–12% captured vs 100% oracle; scaling unlikely to change narrative.
2. **Probe-and-route (2–3 configs):** Marginal at N=200 (+0.001 vs linear); only worth 999v if adding real probe **inference**, not simulation.
3. **Per-dim IQ routing:** Re-eval picks on **total VBench** before any scale-up.
4. **Bootstrap baseline 18.9%** in aggregate file is **stale** (first partial run OOF); trust summary **9.0%**.

## Reproduce

```bash
bash sweep_experiment/sbatch/submit_budget_routing_experiments.sh
# Results: sweep_experiment/reports/per_video_analysis/2026-07-05/budget_routing_experiments/
```
