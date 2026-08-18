# Wan jobs — 2026-08-18 18:37

| Job | Series | Status | What to do |
|---|---|---|---|
| **15959146** | `i2v_bon_32v_sick` | **DONE** 14:24 EDT, n_ok=32 | Analyzed. Table: `2026-08-18_wan_i2v_bon32_sick.md` |
| **15959601** | hybrid official VBench | **DONE but incomplete** | Scored **do-nothing only**. SLURM `--export` split `VIDEO_DIRS` on commas. always-search and gated-search missing. |

Do not start TTC.

---

## Resubmit VBench (always-search + gated; notta skipped)

Comma bug is fixed. Existing notta `vbench_last5` / `vbench_full` stay.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_vbench_hybrid32.sh
```

When **that** job finishes (not 15959601):

```bash
squeue -u wc3013
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

Need **three** `joined.json` files (notta, always_bon, gated_bon) before
analyze will run. `last5` is the outcome table. No PSNR on these 32 stills.

---

## Sick is finished — do not re-run analyze unless pairing

Already pasted. Re-run only if you want the retitled auto table:

```bash
python wan_experiment/scripts/analyze_i2v_bon.py \
    --series-dir wan_experiment/results/i2v_bon_32v_sick \
    --baseline-dir wan_experiment/results/i2v_bon_32v_hybrid
```
