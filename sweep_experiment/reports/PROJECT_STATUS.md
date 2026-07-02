# LongCat Video TTA — project tracker

**Last updated:** 2026-07-02 (post matched pilot FVD baselines)  
**Cluster:** `/scratch/wc3013/longcat-video-tta` · account `torch_pr_36_mren`  
**Primary objective (current):** VBench++ gains — deployable routers + oracle ceilings

---

## Done (analysis + cluster)

| Workstream | Status | Key artifact |
|---|---|---|
| 999v per-video VBench agreement + magnitudes | ✅ | `per_video_analysis/2026-07-02/vbench_agreement/` |
| 999v retrieval (K5/K10 × SIM/RAND) + VBench | ✅ | `results/panda_1000v_retrieval/` |
| H9 budget pilot (12 configs × 200v) mp4s | ✅ | 2400/2400 mp4s |
| Budget pilot VBench backfill (12/12) | ✅ | all chunks IQ json present |
| Budget **VBench** sliding-config oracle | ✅ | `adasteer_budget_vbench_oracle_pilot.md` (+3.5% vs NOTTA oracle) |
| Oracle + cross-metric suite | ✅ | `oracle_vbench/`, `cross_metric_corr/` |
| Predictor transfer Steps 1–3 | ✅ | `predictor_transfer/` |
| **VBench headroom router** (learned) | ✅ | `vbench_headroom_router/` — config router ~45% oracle |
| OOD skip-gate policy eval | ✅ | `2026-06-30/ood_skip_gate/` |
| Budget **PSNR** oracle FVD (N=200 pilot) | ✅ | FVD **383.93** |
| Matched pilot FVD baselines (same 200v) | ✅ | NOTTA **368.85**, fixed S10 **375.88** |
| Method PSNR oracle FVD (999v) | ✅ | **149.57** vs NOTTA ~155.9 |
| Cluster finish + router + FVD sbatch scripts | ✅ | `run_cluster_finish_pipeline.sh`, etc. |

---

## In progress / needs one command

| Task | Owner | Next action |
|---|---|---|
| Refresh budget PSNR oracle report **with FVD row** | agent/user | `python3 scripts/analyze_adasteer_budget_oracle.py --bootstrap --oracle-fvd-json sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json ...` |
| Router on full pilot N=200 | agent | Backfill Phase-0 features for ~57 pilot videos missing from `2026-06-09` CSVs (router N=143 today) |
| Fix duplicate lines in `pilot_matched_fvd_summary.md` | agent | patch `write_summary()` in `run_pilot_matched_fvd_baselines.py` |
| PI slide pack / narrative sync | user | Use tiers table below — do not mix populations |

---

## Not started (planned experiments)

| Task | Priority | Notes |
|---|---|---|
| **LoRA rank=1 pilot** (mirror R8 recipe, 1000v or 200v) | **High** | Fair test of rank=1 under validated recipe; never submitted |
| **Nonlinear VBench router** (GBM/small MLP) | Medium | If linear router holds (~45% VBench headroom) |
| **1000v budget grid** full sweep | Medium | Only incremental `panda_ood_budget_1000v` (S2/S10/S20 @ LR=1e-2) exists |
| Budget **VBench** oracle FVD | Low | PSNR-oracle FVD done; VBench-oracle FVD not run |
| AdaState reproduction | Blocked | No public code; Self-Forcing port is major effort |

---

## Do not compare across rows (population traps)

| Metric | Population | Value | Label in slides |
|---|---:|---:|---|
| NOTTA VBench total | 999v standard | ~0.772 | deployable baseline |
| Fixed AdaSteer ΔVBench | 999v | ~+0.13% | deployable |
| Budget VBench oracle Δ vs NOTTA | 200v pilot, 12 configs | ~+3.5% | **oracle ceiling** |
| Learned config router | ~143–200v pilot | ~45% of VBench oracle | deployable (OOF) |
| NOTTA FVD (headline) | 999v | ~154.7 | standard sweep |
| NOTTA FVD (matched eval) | **same 200 OOD pilot** | **~368.9** | pilot-only; ≠ 155 |
| Method PSNR oracle FVD | 999v | 149.57 | oracle ceiling |
| Budget PSNR oracle FVD | 200v pilot | 383.93 | only +15 vs NOTTA@200 |

---

## Reference commands

```bash
# Status snapshot
bash scripts/run_cluster_finish_pipeline.sh

# Full CPU analysis + router
RUN_ANALYSIS=1 DATE_TAG=2026-07-02 bash scripts/run_cluster_finish_pipeline.sh

# Matched FVD baselines (GPU)
bash sweep_experiment/sbatch/submit_pilot_matched_fvd.sh

# Router only
DATE_TAG=2026-07-02 bash scripts/run_vbench_headroom_router.sh
```

---

## Changelog

- **2026-07-02:** Pilot mp4s 2400/2400; analysis chain + router; budget FVD + matched baselines complete.
- **2026-06-30:** Oracle suite, skip-gate, predictor transfer; mp4 re-run saga started.
- **2026-06-28:** 999v VBench agreement + magnitude analysis.
