# Bedtime pair — quiet_bon N=32 + tail_hist N=8

**Date:** 2026-08-21
**Do not wait for VBench.** These test the generate-metric findings.
**No TTC. No hist_drop N=32.**

## Why these two (and not hist_drop-32)

N=32 seed_bon went **0/7** on already-hot prefixes (notta tail ≥0.020)
and **12/32** overall (−8.8%). Searching on a living prefix damps it.
`hist_drop` only incremented that same picker on the lucky 8 and still
lost on 0002/0003. Scaling it tonight is the wrong bet.

## Jobs

| Series | Method | N | What it tests | Wall |
|---|---|---:|---|---|
| `v2v_panda_quiet_32v` | `quiet_bon` | 32 | k=4 only if **prefix** motion < 0.018; else k=1 | 8 h |
| `v2v_panda_tail_8v` | `tail_hist` | 8 | always last-3-latent history, no search | 4 h |

`quiet_bon` is the causal test of the 0/7 finding. Gate is prefix
motion (known at t=0), not notta tail. Pair against confirm notta /
seed_bon via sidecars.

`tail_hist` asks whether hist_drop’s +42% was “less history” (HG
without CFG) or just more seeds. k=1, same 8 as the bake-off. If it
beats notta without search, the axis is real. If it ties notta, hist_drop
was seed-search with a costume.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
NEXT=1 bash wan_experiment/sbatch/submit_v2v_bakeoff.sh
```

Queues behind VBench 16122823/824 (`QOSGrpGRES`). Leave them.

## Do not submit tonight

- hist_drop N=32
- more seed_bon / hinge / motion_bon / backtrack / shift
- TTC / I2V-32
