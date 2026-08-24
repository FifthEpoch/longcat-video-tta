# Ablation — always motion-search on SF (2026-08-24)

Splits the SF-pseudo win: **gate vs pick**. Do not submit until
GO. Same first 32 as confirm / sf-family. Host = SF chunked.
`VIDEO_WORKERS=1`. VBench `afterok` L40S. No TTC. No I2V.
Do not scale to 128. Do not retune DROP / trust.

## Why

`sf_pseudo` N=32: tail **+37%**, 25/2/5, Dyn 0.5, IQ/subject hold.
Fire **27/32**. That is a loose gate. seed_bon-32 on this same
stack used a **drift** pick, always-on, and **damped** tail −9%.

Until always-on motion-k=4 is on disk, we cannot say whether we
invented a prefix sensor or a pick.

## Arm

| Method | k | Gate | Pick |
|---|---:|---|---|
| `sf_always_search` | 4 | none (every chunk after prefix) | max temporal motion among cands with motion ≥ 0.8× cand0 |

Reuse confirm notta + family `sf_pseudo` (skip-existing). Cite vs
SF notta **and** vs `sf_pseudo`.

## Calls (pre-registered)

| Result vs pseudo | Call |
|---|---|
| tail within ~5%, quality same | Gate is fake. Invention = motion pick on SF. HOLD both. Next = cost (k=2) or rewind stay-on, not 128. |
| tail down or IQ/subject fail | Gate is doing work (the 5 skips). Keep `sf_pseudo`. No always-on paper row. |
| tail up, quality tax | Gate is a brake. Keep gated for the paper; always-on is an upper bound only. |

Promote letter still vs **SF notta**. Do not scale tonight.

## Submit

If SF always is **not** already queued (same paste as RF):

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_always_search_wave.sh
```

If SF always **is** already queued:

```bash
SF_JOB=<sf_jobid> bash wan_experiment/sbatch/submit_v2v_rf_always_search.sh
```

k=4 on both hosts. Width note:
`2026-08-24_wan_v2v_always_search_k.md`. Do not start a second SF job.
