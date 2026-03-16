# Investigating Exp5 Batch-Size UCF-101 Failed Runs

The **exp5 batch-size ablation (UCF-101)** runs use a 1000-video UCF-101 pool for retrieval and 100 eval videos. The status script reports failed/empty run dirs for:

- **exp5_batch_size_delta_a_ucf101:** BS_DA2, BS_DA3, BS_DA4, BS_DA5  
- **exp5_batch_size_full_ucf101:** BS_F2, BS_F3, BS_F4, BS_F5  
- **exp5_batch_size_lora_ucf101:** BS_L2, BS_L3, BS_L4, BS_L5  

(BS_*1 = `batch_videos: 1` typically completes; BS_*2–5 = 5, 10, 50, 100 often fail or never wrote `summary.json`.)

---

## 1. Find job IDs (sacct)

On the cluster, SLURM truncates job names to 8 characters, so these jobs show as **tta_exp5**:

```bash
cd /scratch/wc3013/longcat-video-tta

# Last 7 days
sacct --name=tta_exp5 --starttime=$(date -d '7 days ago' +%Y-%m-%dT%H:%M) \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End

# Or without starttime
sacct --name=tta_exp5 --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End | head -50
```

Check **State** (FAILED, CANCELLED, COMPLETED, etc.) and **ExitCode** (0:0 = success).

---

## 2. Where the logs are

Logs live under:

```
sweep_experiment/slurm_log/<jobname>_<jobid>.err
sweep_experiment/slurm_log/<jobname>_<jobid>.out
```

Job names are truncated, so you’ll see files like:

- `tta_exp5_BS_DA2_2881234.err` (or similar pattern with run_id and job ID)

List recent exp5 logs:

```bash
ls -lt sweep_experiment/slurm_log/tta_exp5* 2>/dev/null | head -30
```

---

## 3. Inspect a specific failed job

Using a **JobID** from sacct (e.g. `2881234`):

```bash
# Show status
sacct -j 2881234 --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End

# Stderr (traceback / errors)
tail -100 sweep_experiment/slurm_log/tta_exp5*_2881234.err

# Stdout (config and progress)
tail -80 sweep_experiment/slurm_log/tta_exp5*_2881234.out
```

Or use the investigation script with one or more job IDs:

```bash
bash sweep_experiment/scripts/investigate_failed_jobs.sh 2881234 2881235
```

---

## 4. Common causes of failure

| Cause | What to check | Fix |
|--------|----------------|-----|
| **Missing dataset** | `.err`: "No such file", "ucf101_1000_480p", "ucf101_100_480p" | Create/prepare UCF-101 subsets; config uses `retrieval_pool_dir` and `--data-dir` (eval set). |
| **Retrieval / encoding error** | `.err`: SentenceTransformers, embedding, or "retrieve_neighbors" | Ensure retrieval pool dir has expected structure and metadata; check `run_*_tta.py` retrieval path. |
| **OOM (full-model)** | `.err`: CUDA out of memory, killed | For full-model with large `batch_videos` (e.g. 50, 100), request more mem or reduce batch in config. |
| **Job cancelled / timeout** | sacct State CANCELLED or TIMEOUT | Resubmit with longer time or use no-preempt queue; for preemption, requeue or resubmit failed run_ids. |
| **Conda/env** | `.err`: "No module named", "torch" | Sbatch uses `CONDA_ENV`; ensure same env on node and that `${CONDA_ENV}/bin/python` exists. |

---

## 5. Resubmit only the failed run_ids

After fixing the underlying issue, resubmit only the runs that don’t have `summary.json`:

**Delta-A (exp5_batch_size_delta_a_ucf101):**

```bash
python sweep_experiment/scripts/run_sweep.py \
  --config sweep_experiment/configs/exp5_batch_size_delta_a_ucf101.yaml \
  --account torch_pr_36_mren \
  --data-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_100_480p \
  --run-ids BS_DA2 BS_DA3 BS_DA4 BS_DA5
```

**Full-model (exp5_batch_size_full_ucf101):**

```bash
python sweep_experiment/scripts/run_sweep.py \
  --config sweep_experiment/configs/exp5_batch_size_full_ucf101.yaml \
  --account torch_pr_36_mren \
  --data-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_100_480p \
  --run-ids BS_F2 BS_F3 BS_F4 BS_F5
```

**LoRA (exp5_batch_size_lora_ucf101):**

```bash
python sweep_experiment/scripts/run_sweep.py \
  --config sweep_experiment/configs/exp5_batch_size_lora_ucf101.yaml \
  --account torch_pr_36_mren \
  --data-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_100_480p \
  --run-ids BS_L2 BS_L3 BS_L4 BS_L5
```

Use the same `--data-dir` and paths as in the config (e.g. `ucf101_100_480p` for eval, `ucf101_1000_480p` for retrieval pool).

---

## 6. Baseline “failed” dirs (false positives)

The status script now **ignores** subdirs named `generated_videos` under `baseline_experiment/results/cond*_gen*/`. Those are output dirs for baseline (no-TTA) runs, not sweep run dirs. If you still see `cond14_gen2/generated_videos` etc. in the failed list, update the script to skip that name (already done in the repo). Real baseline runs have `summary.json` at the **series** level (e.g. `cond14_gen2/summary.json`), not inside `generated_videos`.
