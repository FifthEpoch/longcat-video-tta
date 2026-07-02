# LongCat Video TTA — project tracker

**Last updated:** 2026-07-06 (review submission batch)  
**Cluster:** `/scratch/wc3013/longcat-video-tta` · account `torch_pr_36_mren`  
**Primary objective (current):** VBench++ gains — deployable routers + oracle ceilings  
**PI review:** ~4 days — run `submit_review_experiments.sh`

---

## Review sprint (submit now)

| Phase | Script | Jobs | Delivers for PI |
|---|---|---:|---|
| **A** LoRA R1 @ 999v | `submit_lora_r1_1000v_panda.sh` | 10 GPU | Fair rank=1 vs R8 **ΔVBench** (AdaState-style table) |
| **B** Budget configs @ 999v + VBench | `submit_adasteer_budget_1000v_vbench_review.sh` | 30 GPU | Deployable fixed S10 vs VBench-oracle S2 at **full N** |
| **C** Pilot features → router N=200 | `submit_pilot_router_features.sh` | GPU fan-out | Learned router on full 200v pilot |
| **All** | `submit_review_experiments.sh` | A+B+C | One command |

**After GPU jobs finish:** `bash scripts/run_review_analysis_when_ready.sh`

```bash
# Submit everything (dry-run first):
DRY_RUN=1 bash sweep_experiment/sbatch/submit_review_experiments.sh
bash sweep_experiment/sbatch/submit_review_experiments.sh
```

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
| **LoRA rank=1 @ 999v** | **Submit Phase A** | Script ready; mirror R8 recipe |
| **Budget 999v VBench configs** | **Submit Phase B** | S2_LR1e3, S10_LR5e3, S5_LR1e3 + inline VBench |
| **Pilot router features N=200** | **Submit Phase C** | `submit_pilot_router_features.sh` |
| **Nonlinear VBench router** (GBM/MLP) | Medium | After Phase C + linear retrain |
| **Full 20-config 1000v budget grid** | Low (post-review) | 200 jobs — defer unless queue empty |
| Budget **VBench** oracle FVD | Low | PSNR FVD done |
| AdaState reproduction | Blocked | No public code |

---

## PI narrative draft (VBench++, AdaState-aware)

1. **Deployable today:** fixed AdaSteer **~+0.13%** VBench vs NOTTA @ 999v — flat (unlike AdaState +3.4% @ 5s).
2. **Oracle ceiling exists:** budget pilot VBench router **+3.5%** vs NOTTA (200v, 12 configs) — not deployable alone.
3. **Deployable lever:** learned **step×LR router** captures **~45%** of VBench oracle (OOF, pilot).
4. **Review experiments (Phase A–C):** test **LoRA R1**, **999v budget VBench configs**, **full-N router** — closes gap between pilot oracle and paper-grade N.
5. **Do not mix:** FVD pilot ~370 vs headline ~155 (different subsets); PSNR oracle ≠ VBench oracle.

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
