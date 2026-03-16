# Experiments Not Yet Run (from summary_printout.txt)

This document lists runs or experiment directories that have **not been run yet**, are **missing**, **in progress**, or **failed with no results**, as of the latest status report.

---

## 1. Individual runs: NOT STARTED or no data (—)

| Series / Section | run_id | Notes |
|------------------|--------|--------|
| Exp 3 Training Frames (delta_a) | **TF_DA3** | 14 cond, 14 gen — no summary (not started or failed) |
| Full iter sweep (Series 2) | **F10** | Missing from table (steps=20 variant or not started) |
| Delta-A optimized (Series 23) | **DAO2** | 14 cond, 14 gen — not started |
| Ratio sweep (Series 27) | **AR4**, **AR5** | 14 cond × G=4, G=12 — not started |
| Full-Model Long-Train 100v (Series 31) | **FLT3** | One of the long-train 100v runs — not started |
| Backbone: Open-Sora v2.0 | **opensora_t** | No data (—) |

---

## 2. Directories not found (experiment never launched)

The script reports **"(directory not found)"** for these result directories. The experiments have not been run on the machine that generated the report.

| Result directory | Description |
|------------------|-------------|
| `best_methods_proper_es` | Best Methods with Proper ES |
| `adasteer_long_train_es` | AdaSteer Long-Train + ES (Series 30) |
| `lora_ultra_long_train` | Ultra-LoRA Long-Train + ES (Series 32) |
| `exp5_batch_size_delta_a` | Experiment 5: Delta-A batch-size (Series 37) |
| `exp5_batch_size_lora` | Experiment 5: LoRA batch-size (Series 38) |
| `exp5_batch_size_full` | Experiment 5: Full-model batch-size (Series 39) |

---

## 3. Runs with zero completed videos (0/N)

These runs exist but completed **0** videos (failure or not started).

| Series | run_id | Config |
|--------|--------|--------|
| Hidden-State Residual (Series 19) | **DBH2** | G=4, d_tgt=hidden |
| Hidden-State Residual (Series 19) | **DBH3** | G=12, d_tgt=hidden |
| Hidden-State Residual (Series 19) | **DBH4** | lr=0.001, G=4, d_tgt=hidden |
| Hidden-State Residual (Series 19) | **DBH5** | lr=0.005, G=4, d_tgt=hidden |
| Hidden-State Residual (Series 19) | **DBH6** | lr=0.05, G=4, d_tgt=hidden |
| Delta-C LR sweep (Series 11) | **DC4** | d_lr=0.5 |
| Delta-C LR sweep (Series 11) | **DC5** | d_lr=1.0 |

---

## 4. Explicitly “not started”

| Section | Note |
|---------|------|
| **CogVideoX-5B-I2V** | Report says "(not started)" |

---

## 5. Partial / in-progress (for awareness)

- **GH_DA4**, **GH_DB4**, **GH_DC4**: 26 or 18 videos (subset; may be intentional or in progress).
- **LB1–LB6** (Ultra-Constrained LoRA): 67–91/100 (some videos not completed).
- **FLT2**: 85/86 complete.
- Summary counts: **6 in-progress** (checkpoint present), **12 empty dirs**, **195 complete** of 213 total experiment dirs.

---

## Summary

- **Single runs to (re)run or fix:** TF_DA3, F10, DAO2, AR4, AR5, FLT3, opensora_t; DBH2–DBH6, DC4, DC5.
- **Whole experiment dirs to create/run:** best_methods_proper_es, adasteer_long_train_es, lora_ultra_long_train, exp5_batch_size_delta_a, exp5_batch_size_lora, exp5_batch_size_full.
- **Backbone:** CogVideoX-5B-I2V not started.

Use this list to prioritize re-runs, debugging (e.g. 0/100), or launching missing experiment directories.
