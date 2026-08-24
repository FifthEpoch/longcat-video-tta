# SF-hosted family wave (2026-08-24)

Same four widgets as the RF family, but the **host is Self-Forcing**
and the **sampler is SF native chunked**. That is the field-shaped
claim: method-on-SF vs SF do-nothing.

RF `rolling_notta` stays a comparison row (already on disk). Do
**not** use `sf_roll` (H1 twitch). Do not scale the RF-hosted 32.

## Why this wave

Citing SF as the baseline while implementing on RF (which already
beats SF by +31%) makes “+X vs SF” a host win. This wave puts the
widget on the baseline student.

## Arms (series `v2v_panda_sf_family_32v`)

| Family | Method | k | Sampler |
|---|---|---:|---|
| A | `sf_rewind` | 1 | SF chunked; resample chunk if motion `< 0.8×` previous |
| B | `sf_sick_search` | 4 | Search only after a sick freeze; max motion + trust 0.8 |
| D | `sf_pseudo` | 4 | Hold out last 3 prefix latents; search tail if extra seed wins B |
| C | `sf_sink` | 1 | LongLive-style `sink_size` on SF. **Not HG-f. Not sf_roll.** |

`DROP=0.8` pre-registered. `VIDEO_WORKERS=1`. VBench `afterok` L40S.
Reuse confirm notta + forward rolling (skip-existing). Same first 32.

Promote an arm if median tail beats **SF notta** and IQ not worse
by ≥1.0 and subject not worse by ≥0.02. Also print vs RF so we
do not confuse a host-gap close with a method win.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_sf_family.sh
```

**16266878 rewind FAILED** 8/32: accepted resample then scored
with `committed` already advanced (`gen_only` empty → IndexError).
Fixed: score before increment. Resume
`submit_v2v_sf_rewind_resume.sh` (skip-existing keeps 8 mp4s).
Leave 16266879–881. VBench 882 was afterok-cancelled.

No TTC. No I2V. Do not resubmit the RF family.
