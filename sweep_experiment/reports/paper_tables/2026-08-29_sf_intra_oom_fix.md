# SF intra OOM fix + unlaunched denoise hooks (2026-08-29)

**Call:** A crash is not a NO. Fix SF intra and launch bpseudo / restep.
Do not relaunch RF intra (already scored NO). Do not scancel
caption-128 VBench **16545806**. Do not retune 1.5×.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 WAVE=sf bash wan_experiment/sbatch/submit_v2v_intra8.sh
WAVE=sf bash wan_experiment/sbatch/submit_v2v_intra8.sh
SMOKE=1 WAVE=bpseudo bash wan_experiment/sbatch/submit_v2v_denoise8.sh
WAVE=bpseudo bash wan_experiment/sbatch/submit_v2v_denoise8.sh
SMOKE=1 WAVE=restep bash wan_experiment/sbatch/submit_v2v_denoise8.sh
WAVE=restep bash wan_experiment/sbatch/submit_v2v_denoise8.sh
```

GRES will queue behind 16545806 if that job is on L40S — generate
is H200, so these can run now.

## What OOM’d

`_fill_sf_intra_chunk` kept **k=4 full KV clones** (`snap_after` on
every cand) while the 137-frame cache grew toward ~39 GB. Five
copies miss a 141 GB H200. Job 16471675 died at ~55 min allocating
0.88 GB with 0.35 GB free.

**Fix:** keep one live post-block snap. Drop the losers immediately.
`empty_cache` after each block. Same trim on restep cand snaps.

## What we are testing

| Wave | Methods | Why |
|---|---|---|
| `WAVE=sf` intra | `sf_intra`, `sf_intra_always` | First real SF intra videos |
| `WAVE=bpseudo` | `sf_bpseudo`, `sf_bpseudo_always`, `rf_bpseudo` | Never launched |
| `WAVE=restep` | `sf_restep`, `sf_restep_always`, `rf_restep`, `rf_restep_always` | Never launched |

lastmix stays **NO**. No `rf_bpseudo_always` (that is RF intra always).
