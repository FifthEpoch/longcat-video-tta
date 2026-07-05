# VBench gain prediction experiments @ N=200 (pilot)

**Date:** 2026-07-05  
**Series:** `panda_ood_budget_pilot` (12 AdaSteer configs, 200 videos)  
**Cluster path:** `sweep_experiment/reports/per_video_analysis/2026-07-05/vbench_gain_prediction_experiments/`  
**Commit:** `5a67a7a` (exp9 in-sample bug fixed in follow-up commit)

## Summary table (OOF unless noted)

| Experiment | Captured % | Match % | Deployable? | Notes |
|---|---:|---:|---|---|
| exp6_knn_oracle_transfer | 1.2 | 16.5 | Yes | kNN vote on Phase-0 features |
| exp7_gain_predictor_probe | **12.8** | 21.5 | Yes | Ridge on ΔVBench, probe features |
| exp8_abstain_route_3way | −0.8 | 6.0 | Yes | Headroom gate ε=0.05 → 3-way |
| exp9_multitask_aestech | **7.6** | 17.0 | Yes | OOF proxy ridge; 98% on proxy scale, fails on total |
| exp10_dover_aestech_proxy | 18.4 | 8.5 | Upper bound | GT Aes+IQ on S2/S10 probes → route |
| exp11_tier3_probe_ridge | 12.1 | 40.5 | Yes | Phase-0 + tier3 + probe, 3-way |
| exp12_trajectory_ridge | −0.2 | 34.5 | Yes | delta_norm / final_loss / grad_norm |

## Reference bars

| Baseline | Captured % |
|---|---:|
| Fixed S10 | 0 |
| Linear Phase-0 router | ~9 |
| Exp1 ridge probe 3-way (prior best) | 12.1 |
| Oracle headroom (mean) | +0.140 VBench total |
| Success bar | >25% with bootstrap CI excluding 0 |

## Headline (honest OOF)

**Best deployable:** `exp7_gain_predictor_probe` at **12.8%** — marginal improvement over prior 12.1%, still **below 25% bar**.  
**exp9 (7.6% OOF):** proxy target (0.428·Aes+0.572·IQ) captures **98% on-proxy** but does **not** transfer to total VBench — confirms decoupling.  
**exp10 (18.4%)** uses ground-truth VBench Aes+IQ on probe configs (not DOVER-on-frames); treat as ceiling for probe+DOVER routing.

## Decision

**NO-GO** on 999v×12 routing training for total-VBench objective. Optional next step: GPU DOVER on probe mp4s to see if exp10 upper bound (~18%) is reachable in deployment.
