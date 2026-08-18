# Wan in-flight jobs — 2026-08-18 11:47

Two jobs. Do not start TTC. Cancel: `scancel 15959146 15959601`.

Status check (either job):

```bash
squeue -u wc3013
sacct -j 15959146,15959601 --format=JobID,JobName,State,Elapsed,End
```

---

## 1. Search-while-sick — job **15959146**

| | |
|---|---|
| Series | `i2v_bon_32v_sick` |
| What | gated-search only; pair vs hybrid do-nothing / always-search |
| Status at 11:47 | **R** 14:15 on `gh107` |
| Log | `wan_experiment/slurm_log/wan_i2v_chunk_15959146.out` |
| Done when | `wan_experiment/results/i2v_bon_32v_sick/gated_bon_h30s_shard0/summary.json` has `n_ok=32` and the job is not in `squeue` |

```bash
# still running?
squeue -j 15959146
tail -30 wan_experiment/slurm_log/wan_i2v_chunk_15959146.out

# finished — paste this output
python wan_experiment/scripts/analyze_i2v_bon.py \
    --series-dir wan_experiment/results/i2v_bon_32v_sick \
    --baseline-dir wan_experiment/results/i2v_bon_32v_hybrid
```

Pass/fail (handcrafted last-chunk, cite medians): 11/16 near hybrid 2.16/2.66; 03/24 near always-search; 06/07 still skipped on piece 1; 30 back toward 1.44; wall between 173 and 256 s.

---

## 2. Official VBench on hybrid 32 — job **15959601**

| | |
|---|---|
| Series | `i2v_bon_32v_hybrid` (score only; no new generate) |
| What | VBench quality dims, `last5` then `full`, three method dirs |
| Status at 11:47 | **PD** Priority (waits for 15959146 or a free H200) |
| Log | `wan_experiment/slurm_log/wan_i2v_vbench_15959601.out` |
| Done when | all three `vbench_last5/joined.json` exist (and ideally `vbench_full/joined.json`) and the job is not in `squeue` |

```bash
# still running?
squeue -j 15959601
tail -40 wan_experiment/slurm_log/wan_i2v_vbench_15959601.out
ls wan_experiment/results/i2v_bon_32v_hybrid/*_h30s_shard0/vbench_last5/joined.json

# finished — paste both outputs
python wan_experiment/scripts/analyze_i2v_vbench.py \
    --series-dir wan_experiment/results/i2v_bon_32v_hybrid \
    --clip last5 \
    --out sweep_experiment/reports/paper_tables/$(date +%F)_wan_i2v_bon32_vbench_last5.md

python wan_experiment/scripts/analyze_i2v_vbench.py \
    --series-dir wan_experiment/results/i2v_bon_32v_hybrid \
    --clip full \
    --out sweep_experiment/reports/paper_tables/$(date +%F)_wan_i2v_bon32_vbench_full.md
```

`last5` is the outcome table. `full` is diluted by the shared prefix. No PSNR on these 32 stills.

---

## After you paste

Paste the analyze output here. Do not start test-time training from either result.
