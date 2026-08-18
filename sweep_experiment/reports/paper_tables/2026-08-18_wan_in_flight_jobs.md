# Wan jobs — 2026-08-18 18:45

| Job | Series | Status | What to do |
|---|---|---|---|
| **15959146** | `i2v_bon_32v_sick` | **DONE** | Table: `2026-08-18_wan_i2v_bon32_sick.md` |
| **15959601** | hybrid VBench | **INCOMPLETE** (notta only) | Do not re-analyze this job |
| **15984561** | hybrid VBench retry | **SUBMITTED** | Wait. notta skipped; always + gated still scoring |

Cancel: `scancel 15984561`. Do not start TTC.

---

## When 15984561 is actually done

Done only when **all three** exist:

```
wan_experiment/results/i2v_bon_32v_hybrid/notta_h30s_shard0/vbench_last5/joined.json
wan_experiment/results/i2v_bon_32v_hybrid/always_bon_h30s_shard0/vbench_last5/joined.json
wan_experiment/results/i2v_bon_32v_hybrid/gated_bon_h30s_shard0/vbench_last5/joined.json
```

```bash
squeue -j 15984561
tail -40 wan_experiment/slurm_log/wan_i2v_vbench_15984561.out
ls wan_experiment/results/i2v_bon_32v_hybrid/*_h30s_shard0/vbench_last5/joined.json

python wan_experiment/scripts/analyze_i2v_vbench.py \
    --series-dir wan_experiment/results/i2v_bon_32v_hybrid \
    --clip last5 \
    --out sweep_experiment/reports/paper_tables/$(date +%F)_wan_i2v_bon32_vbench_last5.md

python wan_experiment/scripts/analyze_i2v_vbench.py \
    --series-dir wan_experiment/results/i2v_bon_32v_hybrid \
    --clip full \
    --out sweep_experiment/reports/paper_tables/$(date +%F)_wan_i2v_bon32_vbench_full.md
```

If `ls` still shows only notta, the job is not finished — do not run analyze.
`last5` is the outcome table. Paste both analyze outputs here.
