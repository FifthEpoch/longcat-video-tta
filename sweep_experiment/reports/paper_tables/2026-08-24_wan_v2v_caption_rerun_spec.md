# Caption-conditioned V2V replay — SUBMIT-READY (2026-08-24)

Stem prompts (`panda 0013`) fought the real prefix. 0013 is a
bathroom stain; 0001/0005 are kitchen; 0020 is a flashlight. T5
heard “panda,” so even SF do-nothing could morph or freeze instead
of continuing the scene. Same-prompt stem tables stay an **audit**.
This replay is the first caption-conditioned protocol.

## Lock

- Prompt = first list caption from `metadata.csv` (`prompt_source=metadata_csv`).
- Same first-N path order as before (sorted `rglob`).
- Prefix 9 / chunk 21 / 30 s. k=4 on search arms.
- New series names. Do **not** overwrite stem dirs.
- Do **not** mix stem numbers into caption tables.
- Do **not** scancel stem always-search 16288113–115.
- No TTC. No I2V. `VIDEO_WORKERS=1`. VBench `afterok` full clip.

## Hypothesis (baseline too)

On clips whose caption is not “panda,” the old string pulled the
tail off the prefix. Caption-conditioned SF notta should keep
identity / scene better (subject, IQ) and may also raise tail
motion if the morph-or-freeze path was text-driven. Method deltas
can shrink or grow; harvest vs **caption** notta only.

## Waves

Submit script: `wan_experiment/sbatch/submit_v2v_caption_rerun.sh`

| WAVE | Series | What | N |
|---|---|---|---:|
| **1 (now)** | `v2v_panda_caption_32v` | notta, rolling_notta, SF family 4, RF family 4, both always-search | 32 |
| 2 | `v2v_panda_caption_closed_32v` | seed/quiet/live/appear + H1/H4 host split | 32 |
| 3 | `v2v_panda_caption_8v` | remaining N=8 discovery (no shift_search / knob_probe) | 8 |
| 4 | `v2v_panda_caption_128v` | notta + rolling_notta only | 128 |

Wave 1 is 12 generate + 1 VBench. H200 extras queue. Do not dump
WAVE=all unless the queue is empty.

Skip: shift_search (dead), knob_probe (no 30 s write), analysis-only
scripts, I2V, TTC.

## Status

**GENERATE HARVEST 2026-08-24 22:45.** Most arms 32/32 in
[`outcomes`](2026-08-24_wan_v2v_caption_wave1_outcomes.md). Pseudo
27 + SF always 9 still running. VBench **16310330** afterok those.
Do not submit WAVE=2.

## Harvest

Cite vs `v2v_panda_caption_32v/notta`. Pair tails with the matching
stem run only as a **confound delta** (same method, stem vs caption),
never as the paper method table.

First sidecar on wave 1 must not be `prompt_source=stem`. If it is,
scancel that wave only.
