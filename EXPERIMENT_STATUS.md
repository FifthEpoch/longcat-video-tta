# Experiment Status Report

**Last updated:** Feb 16, 2026 (evening — all retrieval code implemented)
**Cluster:** NYU Greene HPC (H200 partition)
**Account:** `torch_pr_36_mren`

---

## Queue reason → script and output (for dual preempt / no-preempt submissions)

When the same sweep is submitted twice (preemption and no-preemption), use `squeue` reason to tell them apart:

| `squeue` reason | Script used | Results directory |
|------------------|-------------|-------------------|
| **(Priority)** | `run_sweep_no_preempt.sbatch` (no preemption) | `sweep_experiment/results_no_preempt/<series>/<run_id>/` |
| **(QOSMaxGRESPerUser)** | `run_sweep.sbatch` (with `--comment="preemption=yes;requeue=true"`) | `sweep_experiment/results/<series>/<run_id>/` |

So: **Priority** = normal queue, no preemption, saves to `results_no_preempt/`. **QOSMaxGRESPerUser** = preemption queue, saves to default `results/`.

---

## Quick Summary

| Category | Status |
|----------|--------|
| Completed experiments (have `summary.json`) | ~170+ runs |
| Queued on SLURM (PD) | ~40 jobs |
| Not yet submitted | ~20 runs (batch-size ablation, backbone experiments) |

---

## 1. COMPLETED EXPERIMENTS (Results Available)

All of the following used the **same 100 Panda-70M videos** (`panda_100_480p`, `seed=42`, `max_videos=100`) unless otherwise noted.

### Experiment 0: Baselines (No TTA)
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 36 | `panda_no_tta_continuation` | 1 | DONE |

### Experiment 1: Conditioning + Generation Length
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 19 | `exp3_train_frames_full` | 4 | DONE |
| 20 | `exp3_train_frames_lora` | 4 | DONE |
| 21 | `exp3_train_frames_delta_a` | 4 | DONE |
| 22 | `exp3_train_frames_delta_b` | 4 | DONE |
| 23 | `exp3_train_frames_delta_c` | 4 | DONE |
| 24 | `exp4_gen_horizon_full` | 4 | DONE |
| 25 | `exp4_gen_horizon_lora` | 4 | DONE |
| 26 | `exp4_gen_horizon_delta_a` | 4 | DONE |
| 27 | `exp4_gen_horizon_delta_b` | 4 | DONE |
| 28 | `exp4_gen_horizon_delta_c` | 4 | DONE |

### Experiment 2: Full-Model TTA
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 1 | `full_lr_sweep` | 5 | DONE |
| 2 | `full_iter_sweep` | 4 | DONE |
| 17 | `full_long_train` | 1 (30 vids) | DONE |

### Experiment 3: AdaSteer-1 (Delta-A)
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 7 | `delta_a_lr_sweep` | 5 | DONE |
| 8 | `delta_a_iter_sweep` | 4 | DONE |
| 18 | `delta_a_long_train` | 1 (30 vids) | DONE |
| 23 | `delta_a_optimized` | 2 | DONE |
| 23b | `delta_a_norm_combined` | 1 | DONE |
| 25 | `delta_a_equiv_verify` | 4 | DONE |

### Experiment 4: AdaSteer G>1 (Delta-B)
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 9 | `delta_b_groups_sweep` | 6 | DONE |
| 9b | `delta_b_iter_sweep` | 4 | DONE |
| 10 | `delta_b_lr_sweep` | 5 | DONE |
| 19 | `delta_b_hidden_sweep` | 6 | DONE |
| 24 | `delta_b_low_lr` | 4 | DONE |
| 26 | `adasteer_groups_5step` | 6 | DONE |
| 27 | `adasteer_ratio_sweep` | 8 | DONE |

### Experiment 5: LoRA TTA
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 3 | `lora_rank_sweep` | 5 | DONE |
| 4 | `lora_lr_sweep` | 5 | DONE |
| 5 | `lora_target_sweep` | 2 | DONE |
| 6 | `lora_iter_sweep` | 4 | DONE |
| 14 | `lora_constrained_sweep` | 9 | DONE |
| 15 | `lora_builtin_comparison` | 2 | DONE |
| 16 | `lora_ultra_constrained` | 6 | DONE |

### Experiment 6: Output Residual (Delta-C)
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 11 | `delta_c_lr_sweep` | 5 | DONE |
| 12 | `delta_c_iter_sweep` | 4 | DONE |

### Early Stopping Ablation
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 13 | `es_ablation_disable` | 1 | DONE |
| 14 | `es_ablation_check_freq` | 4 | DONE |
| 15 | `es_ablation_patience` | 5 | DONE |
| 16 | `es_ablation_sigmas` | 4 | DONE |
| 17 | `es_ablation_noise_draws` | 3 | DONE |
| 18 | `es_ablation_holdout` | 3 | DONE |

### Extended Experiments
| Series | Name | Runs | Status |
|--------|------|------|--------|
| 20 | `best_methods_proper_es` | 1 | DONE |
| 21 | `norm_tune_sweep` | 6 | DONE |
| 22 | `film_adapter_sweep` | 6 | DONE |

---

## 2. QUEUED / PENDING ON SLURM (Not Yet Running)

These jobs are submitted but waiting for resources (all show `PD` / `QOSGrpMemLimit`).

### Dataset Preparation (cs partition)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2734929 | `panda70m` | Download Panda-70M 1000-video subset |
| 2734931 | `resize_v` | Resize downloaded videos to 480p |

### Experiment 1 (remaining runs)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2673485 | `tta_exp3` | Exp 3 training frames (likely remaining method) |
| 2673487-2673490 | `tta_exp4` | Exp 4 gen horizon (3 remaining runs) |

### AdaSteer Extended (Series 28-30)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2674220-2674236 | `tta_adas` (17 jobs) | AdaSteer extended data (28), param sweep (29), long-train ES (30) |
| 2691387-2691391 | `tta_adas` (5 jobs) | AdaSteer additional runs |
| 2673483-2673484 | `tta_delt` (2 jobs) | Delta-related experiments |

### Full-Model + LoRA Long-Train (Series 31-32)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2691392-2691393 | `tta_full` (2 jobs) | Full-model long-train 100v (Series 31) |
| 2691394-2691396 | `tta_lora` (3 jobs) | LoRA ultra long-train (Series 32) |

### Open-Sora 2.0 Backbone (Cross-backbone validation)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2678470 | `setup_co` | Setup conda env for Open-Sora 2.0 |
| 2678793-2678795 | `opensora` (3 jobs) | Open-Sora 2.0 TTA experiments |

---

## 3. NOT YET SUBMITTED

These experiments have configs written but have **not** been submitted to SLURM.

### Experiment 5: Retrieval-Augmented Batch-Size Ablation (NEW)

**Prerequisite:** The 1000-video Panda-70M dataset must be downloaded and resized first (Jobs 2734929, 2734931 are pending for this).

| Series | Config | Method | Runs | Dependency |
|--------|--------|--------|------|------------|
| 37 | `exp5_batch_size_delta_a.yaml` | delta_a | 5 | 1000-video pool + retrieval code (READY in `run_delta_a.py`) |
| 38 | `exp5_batch_size_lora.yaml` | lora | 5 | 1000-video pool (code READY in `run_lora_tta.py`) |
| 39 | `exp5_batch_size_full.yaml` | full | 5 | 1000-video pool (code READY in `run_full_tta.py`) |

**How it works:**
- Evaluate on the same 100 Panda-70M videos as all other experiments
- For each eval video, retrieve K-1 nearest neighbors by text-prompt similarity from the 1000-video pool
- Train shared delta/LoRA/weights on eval video + K-1 neighbors
- Generate only for the eval video
- Sweep K = {1, 5, 10, 50, 100}

**Cost & training caveat:** Pre-encoding scales linearly with K (each neighbour needs a VAE + text encode pass). Training time is constant (`num_steps` is fixed), but steps are distributed round-robin across K videos, so each video gets only ~`num_steps/K` gradient updates. At K=100 with 20 steps, the eval video itself is trained on only ~once. If large-K results look under-trained, a follow-up sweep should scale `num_steps` proportionally with K.

**To submit (once 1000-video dataset is ready — all three methods):**
```bash
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/exp5_batch_size_delta_a.yaml \
    --account torch_pr_36_mren

python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/exp5_batch_size_lora.yaml \
    --account torch_pr_36_mren

python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/exp5_batch_size_full.yaml \
    --account torch_pr_36_mren
```

### UCF-101 Cross-Dataset Evaluation
| Series | Config | Method | Runs | Status |
|--------|--------|--------|------|--------|
| 35 | `ucf101_no_tta.yaml` | full (no TTA) | 1 | Likely queued or done |
| 32 | `ucf101_best_configs.yaml` | full | 1 | Status unknown |
| 33 | `ucf101_delta_a.yaml` | delta_a | 1 | Status unknown |
| 34 | `ucf101_lora.yaml` | lora | 1 | Status unknown |

### Open-Sora 2.0 Backbone Experiments
These require a separate conda environment and codebase (Open-Sora 2.0). Jobs are queued (2678470, 2678793-2678795) but the environment setup job hasn't run yet.

---

## 4. REMAINING CODE WORK

All retrieval-augmented batch TTA code is now **fully implemented** for all three methods:

| Script | Status | What was added |
|--------|--------|----------------|
| `delta_experiment/scripts/run_delta_a.py` | DONE | `--retrieval-pool-dir`, `--batch-videos`, `retrieve_neighbors()` per-video loop, `_optimize_delta_a_batch()` |
| `lora_experiment/scripts/run_lora_tta.py` | DONE | Same pattern: `--retrieval-pool-dir`, `--batch-videos`, `finetune_lora_batch()` |
| `lora_experiment/scripts/run_full_tta.py` | DONE | Same pattern: `--retrieval-pool-dir`, `--batch-videos`, `finetune_full_batch()` |
| `sweep_experiment/sbatch/run_sweep.sbatch` | DONE | Passes `RETRIEVAL_POOL_DIR` and `BATCH_VIDEOS` for all three method cases |
| `delta_experiment/scripts/common.py` | DONE | `build_retrieval_pool()` and `retrieve_neighbors()` using SentenceTransformers |

**All batch-size ablation experiments (Series 37/38/39) are code-ready** — the only remaining dependency is the 1000-video dataset (Jobs 2734929 + 2734931).

The one remaining code task is unrelated to the batch experiments:

| Task | Effort | Blocks |
|------|--------|--------|
| Open-Sora 2.0 environment + TTA scripts | ~1 day | Backbone cross-validation experiments |

---

## 5. MAPPING: QUEUE TO EXPERIMENT

### Dataset Preparation (CPU only, no GPU)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2734929 | `panda70m` | Download 1000 Panda-70M videos from YouTube via yt-dlp. Needed as the retrieval pool for batch-size ablation (Experiment 5). |
| 2734931 | `resize_v` | Resize the 1000 downloaded videos to 480p (832×480) using ffmpeg. Produces `panda_1000_480p/`. |

### Experiment 1: Conditioning + Generation Length (remaining runs)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2673485 | `tta_exp3` | Training frames ablation — one remaining method/config that hasn't completed. Tests how the number of conditioning frames (2/7/14/24) affects TTA quality. |
| 2673487 | `tta_exp4` | Generation horizon ablation — tests how many frames to generate (16/28/44/72) after TTA. |
| 2673488 | `tta_exp4` | Generation horizon ablation — different method or frame count from above. |
| 2673490 | `tta_exp4` | Generation horizon ablation — different method or frame count from above. |

### AdaSteer Extended Experiments (Series 28, 29, 30)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2674220–2674236 | `tta_adas` (17 jobs) | **Series 28 — Extended Data (10 runs):** Tests AdaSteer with longer input videos (5s and 8s, i.e. gen_start_frame=120 and 200) to see if more conditioning data enables more groups. Sweeps G={1,2,4,12} at both durations, plus no-TTA baselines. |
| | | **Series 29 — Parameter Dimension Sweep (7 runs):** Tests AdaSteer with reduced delta dimensions (32, 64, 128, 256, 512) and multi-group variants (G=2, G=3) to find optimal parameter count vs. quality tradeoff. |
| | | **Series 30 — Long-Train with ES (5 runs):** Tests AdaSteer with lower learning rates (1e-3, 2.5e-3, 5e-3) and more training steps (100–200) with early stopping. Uses 14 and 24 conditioning frames. Answers: does AdaSteer benefit from longer training? |
| 2673483–2673484 | `tta_delt` (2 jobs) | Delta-related experiments — likely remaining delta-B or delta-C runs from earlier series. |
| 2691387–2691391 | `tta_adas` (5 jobs) | AdaSteer additional runs — likely part of series 28/29/30 that were submitted in a later batch. |

### Full-Model + LoRA Long-Train (Series 31, 32)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2691392 | `tta_full` | **Series 31 — Full-model long-train, run 1:** Full-model TTA with lr=1e-5, 500 max steps, early stopping (patience=10, check_every=1), 100 videos. Tests whether full-model TTA improves with significantly more training steps. |
| 2691393 | `tta_full` | **Series 31 — Full-model long-train, run 2:** Same but with lr=5e-5, 200 max steps. Higher LR variant. |
| 2691394 | `tta_lora` | **Series 32 — LoRA ultra long-train, run 1:** Ultra-constrained LoRA (last 1–2 blocks, very low alpha=0.01–0.05) extended to 50 training steps with early stopping. Tests whether constrained LoRA needs more iterations to converge. |
| 2691395 | `tta_lora` | **Series 32 — LoRA ultra long-train, run 2:** Different block/alpha config from above. |
| 2691396 | `tta_lora` | **Series 32 — LoRA ultra long-train, run 3:** Different block/alpha config from above. |

### Open-Sora 2.0 Backbone (Cross-backbone validation)
| Job ID | Name | Purpose |
|--------|------|---------|
| 2678470 | `setup_co` | Set up a separate conda environment for Open-Sora 2.0 (different PyTorch version, different dependencies from LongCat). |
| 2678793 | `opensora` | Open-Sora 2.0 TTA experiment — apply AdaSteer (and possibly other methods) to a different video generation backbone to test generalization. |
| 2678794 | `opensora` | Open-Sora 2.0 TTA experiment — different method or config. |
| 2678795 | `opensora` | Open-Sora 2.0 TTA experiment — different method or config. |

### Not Yet Submitted
| Experiment | Config | Jobs | Blocked By |
|------------|--------|------|------------|
| Delta-A batch-size ablation (K=1,5,10,50,100) | `exp5_batch_size_delta_a.yaml` | 5 | 1000-video dataset (Jobs 2734929+2734931 must finish) |
| LoRA batch-size ablation (K=1,5,10,50,100) | `exp5_batch_size_lora.yaml` | 5 | 1000-video dataset (Jobs 2734929+2734931 must finish) |
| Full-model batch-size ablation (K=1,5,10,50,100) | `exp5_batch_size_full.yaml` | 5 | 1000-video dataset (Jobs 2734929+2734931 must finish) |

**Total remaining GPU-hours (estimate): ~570 hours**
