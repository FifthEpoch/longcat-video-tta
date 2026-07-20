# 200v pilot — matched no-TTA FVD + per-OOD-quintile gains

**Series:** `panda_ood_budget_pilot` (N=200, Panda-70M, cond=14/total=28/gen_start=48, 50 steps).
**NOTTA:** joined per canonical video ID from `panda_1000v_standard/NOTTA` (frame geometry verified identical on-cluster).
**Regenerable:** matched FVD via `run_pilot_matched_fvd_baselines.py` (eval_fvd.py + shared GT cache, --force since N=200<256); per-quintile gains via `analyze_adasteer_budget_oracle.py --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv`.

## Matched-pool FVD (lower is better; all N=200, same eval pipeline)

| Policy | FVD | N |
|---|---:|---:|
| No-TTA | 368.9 | 200 |
| Fixed AdaSteer (S10_LR5e3) | 375.9 | 200 |
| Oracle-routed (PSNR-best/video) | 383.9 | 200 |

No-TTA is the FVD floor; fixed TTA +7.0, PSNR-oracle routing +15.0 over no-TTA. The grid's in-run per-config FVDs (316–336, from merged_summary.json) are a DIFFERENT estimator (in-run fixed S10 = 331 vs matched = 376) and are NOT directly comparable to this trio.

## Per-OOD-quintile PSNR improvement vs no-TTA (paired, ~40 pilot videos/quintile)

| Quintile | Δ Fixed−NOTTA | Δ Oracle−NOTTA | Routing edge (oracle−fixed) | Modal cfg |
|---|---:|---:|---:|---|
| Q1 (low OOD) | -0.227 | +0.615 | +0.842 | S20_LR1e2 |
| Q2 | +0.062 | +0.452 | +0.390 | S10_LR1e2 |
| Q3 | +1.056 | +1.682 | +0.626 | S20_LR1e2 |
| Q4 | -0.572 | +0.241 | +0.813 | S20_LR1e2 |
| Q5 (high OOD) | +0.673 | +1.743 | +1.070 | S10_LR1e2 |

Population anchors (paired, N=200): fixed AdaSteer +0.198 dB vs no-TTA; oracle +0.947 dB vs no-TTA.

**Finding:** fixed-budget AdaSteer is net-NEGATIVE vs no-TTA in 2 of 5 OOD quintiles (Q1, Q4). Per-video oracle routing is positive in all five and largest on the high-OOD tail (Q5 +1.743), directly motivating OOD-aware routing with a skip-TTA option.

**Caveat:** the analyzer's standalone per-quintile NOTTA absolute column is computed over the 999-pool quintile membership; the Δ columns above are correctly paired over the pilot subset (~40/quintile).
