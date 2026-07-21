# VBench is un-routable at 1000v across 13 feature/model variants

**Date:** 2026-07-21
**Series:** `panda_ood_budget_1000v_preview` (N≈998), 12-config AdaSteer grid.
**Features:** feature-date `per_video_analysis/2026-07-12` (Block A video-caption + probe/PSNR blocks per experiment). 5-fold OOF.
**Command:** `scripts/run_budget_routing_experiments.py --run-all --series-root sweep_experiment/results/panda_ood_budget_1000v_preview --feature-date sweep_experiment/reports/per_video_analysis/2026-07-12`
**Output:** `per_video_analysis/2026-07-21/budget_routing_experiments_1000v/routing_experiments_summary.md`

Captured % = fraction of (oracle − fixed S10/5e-3) headroom recovered by the deployable OOF policy on the stated objective. Match % = OOF exact-config agreement with oracle.

| Experiment | Objective | Captured % | Match % | Read |
|---|---|---:|---:|---|
| `dim_imaging_quality` | imaging_quality | **98.7** | 6.3 | degenerate: IQ↑ ⇔ adapt-less ⇒ ≈ no-TTA policy |
| `proxy_psnr_all` | VBench-total | +0.8 | 12.8 | ≈ 0 |
| `dim_dynamic_degree` | dynamic_degree | 0.0 | 19.4 | fixed ≈ oracle / degenerate dim |
| `dim_aesthetic_quality` | aesthetic | 0.0 | 8.5 | no headroom |
| `dim_subject_consistency` | subject | 0.0 | 8.2 | no headroom |
| `pairwise_gbm_top4` | VBench-total | −0.2 | 6.5 | HistGBM — nonlinear no help |
| `pairwise_logistic_top4` | VBench-total | −1.1 | 6.6 | |
| `probe_simulated` | VBench-total | −2.4 | 9.3 | **causal probe deltas still fail** |
| `mlp_shallow` | VBench-total | −2.9 | 21.6 | collapses to modal config |
| `composite_psnr_ridge` | VBench-total | −3.8 | 8.1 | |
| `proxy_bestof3_psnr` | VBench-total | −4.5 | 5.6 | |
| `baseline_linear_total` | VBench-total | −7.6 | 10.2 | plain ridge |
| `coarse_steps_lr` | VBench-total | −7.6 | 10.2 | two-stage bucket→LR |

## Findings

1. **VBench-total is un-routable regardless of features or model.** Every VBench-total policy captures ≤ +0.8%; linear, composite-PSNR, pairwise-logistic, HistGBM, and shallow-MLP are all ≈ 0 or negative → **signal limitation, not model capacity**. (The script was explicitly built to test GBM/MLP "at N≈999–2400"; at N≈998 they still fail.)
2. **Causal probe features don't help** (`probe_simulated` = −2.4%): even the actual S2/S10/S20 probe PSNR/SSIM deltas do not reveal the VBench-best config ⇒ per-video cross-config VBench differences are ~noise.
3. **The only routable VBench component (imaging_quality, 98.7%) is degenerate**: MUSIQ is monotone in *how little you adapt*, so the IQ-optimal router ≈ "pick the least-adaptive config" ≈ no-TTA — trivially predictable and anti-correlated with the adaptation benefit (PSNR/FVD).
4. **High Match% without capture** (mlp 21.6%, dynamic 19.4%) = model collapsing onto the modal config on a flat landscape; frequent agreement, ~0 value recovered.

**Implication:** closes the "better features/models will route VBench" hypothesis. Deployable VBench routing is dead on this pool; honest wins remain PSNR-oracle headroom (small, OOD-tail) and fixed-AdaSteer FVD.
