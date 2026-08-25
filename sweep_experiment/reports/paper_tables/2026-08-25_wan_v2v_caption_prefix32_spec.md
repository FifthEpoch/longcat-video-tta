# Caption Prefix-match N=32 — SUBMIT-READY (2026-08-25)

Slide VBench for Prefix-match is still **stem-prompt**. WAVE=1
already reran every other slide controller on real captions
(`v2v_panda_caption_32v`, VBench **16310330** still R). Do **not**
resubmit WAVE=1. Do **not** dump WAVE=2 (quiet_bon / `sf_roll` /
host-split are not on the slide). Do **not** scale AdaSteer (N=8
NO, IQ collapse).

## What this wave is

Series `v2v_panda_caption_prefix_32v`. Same 32 paths, `metadata.csv`.
k=4. Cite vs **caption** SF notta (WAVE=1). Same-wave twins: always
`seed_bon`, gated `live_bon`, pick `appear_bon`.

| Slide row | Method | Rule |
|---|---|---|
| Always | `seed_bon` | k=4 every chunk; min prefix deviation |
| If prefix moving | `live_bon` | k=4 only if prefix motion ≥ 0.012 |
| Appearance only | `appear_bon` | k=4; appearance/seam, no motion in the pick |

## Lock

- New series. Do not overwrite stem `confirm_32v` / `forward_32v`.
- Sidecar must be `prompt_source=metadata_csv`.
- `VIDEO_WORKERS=1`. VBench `afterok` full clip, L40S.
- No TTC. No I2V. No AdaSteer N=32. No `sf_roll`.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
WAVE=prefix bash wan_experiment/sbatch/submit_v2v_caption_rerun.sh
squeue -u $USER
```

**IN FLIGHT 2026-08-25 02:52.** Preflight `mapped=1000 bad=0`.
**16328464** seed_bon, **16328465** live_bon, **16328466**
appear_bon (PD h200_cour None). VBench **16328467** afterok.
Leave **16310330** alone. Cancel this wave only:
`scancel 16328464 16328465 16328466 16328467`.

## After first R

Need `prompt_source=metadata_csv` and a real sentence, not
`panda 0013`. If stem, scancel this wave only.

## Call

NO if tail down or IQ/subject fail the locked bars vs caption SF.
Identity-up / Dyn-0 is the stem failure mode — confirm or retract
under captions. Do not mix stem Prefix-match numbers into the
caption table.
