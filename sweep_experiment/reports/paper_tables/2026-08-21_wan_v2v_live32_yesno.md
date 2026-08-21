# live_bon N=32 is the yes/no test (2026-08-21)

**Do not investigate more at N=8.** The gate mechanism is known.
seed_bon-8 also printed +35% and then failed at 32. Another N=8
threshold / appear / pseudo table cannot make this a paper method.

**The test method is `live_bon` with `live_min=0.012`.** One shot.
Not live_hist, not LongLive, not the ideas wave. Those can finish
in the background; they are not this decision.

**Series:** `v2v_panda_live_32v`
**Baseline:** existing `v2v_panda_confirm_32v/notta` (do not regenerate).
**Submit:** `bash wan_experiment/sbatch/submit_v2v_live32.sh`

## Locked verdict

Use sidecars. Paired N (not unpaired `summary.json`). Cite medians.

| | YES — keep as the controller | NO — close this gate |
|---|---|---|
| Tail | paired median > confirm notta | paired median ≤ notta |
| Stills | prefix `< 0.012` mostly exact-or-better vs notta (no 0002-style mass damping) | stills damped like seed_bon-32 |
| VBench | IQ not worse by ≥1.0 **and** subject not worse by ≥0.02 | IQ/subject collapse (identity damper) |

Dyn median 0/0 does **not** decide — confirm notta Dyn was already 0.

**No retune after seeing N=32.** A new `live_min` is a new method and
a new N=8, not a salvage of this shot. 0000 (prefix 0.00638) is
accepted residue.

Do not scancel lineage 16140812–816 or ideas 16145125–131.
