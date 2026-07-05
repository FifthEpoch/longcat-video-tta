# Wave-1 VBench++ predictor screen @ N=200 (2026-07-06)

**Series:** `panda_ood_budget_pilot` · **Features:** `per_video_analysis/2026-07-06`  
**Fixed baseline:** S10_LR5e3 · **Oracle headroom:** +0.140 mean VBench-total (prior)

## Routing experiments (OOF captured % of oracle headroom)

| Experiment | Captured % | 95% CI | Match % | Deployable? | Notes |
|---|---:|---|---:|---|---|
| exp14_multi_verifier_deploy | 2.8 | [-2.2, 12.9] | 6.0 | Yes | probe ΔPSNR+ΔSSIM only |
| exp14_multi_verifier_full | **17.5** | [9.2, 25.8] | 7.0 | **No (ceiling)** | + GT Aes/IQ/Dyn on probe outputs |
| exp15_tail_only_gate | 1.0 | [-0.8, 6.6] | 6.5 | Yes | tail_cap=24.1% @ 15% apply |
| exp16_knn_probe_manifold | **13.0** | [6.0, 36.0] | 18.0 | Yes | **best deployable** |
| exp17_per_dim_fuse_router | 5.8 | [2.5, 20.8] | 7.5 | Yes | per-dim ridge fuse |
| exp18_logistic_3way_gate | — | — | — | Yes | failed (logistic_fit kwarg; fixed in repo) |

**Reference:** exp7 gain-probe ridge **12.8%** · exp10 GT-probe ceiling **18.4%**

## Feature×ΔVBench screen (exp19)

Only 2 pairs with |ρ|≥0.2; top signal is optical-flow vs Δtemporal_flickering (ρ≈±0.21).

## Decision (corrected)

| Gate | Threshold | Result |
|---|---|---|
| Deployable captured | >15% | **FAIL** (best 13.0%) |
| Tail captured @ 15% apply | >30% | **FAIL** (24.1%) |
| GT-probe ceiling | — | 17.5% (confirms ~18% offline upper bound) |

**Tonight:** **NO-GO** Wave-2 GPU. Paper line: oracle headroom real; honest deployable routing ~13%.

**Cluster path:** `sweep_experiment/reports/per_video_analysis/2026-07-06/wave1_predictor_experiments/`
