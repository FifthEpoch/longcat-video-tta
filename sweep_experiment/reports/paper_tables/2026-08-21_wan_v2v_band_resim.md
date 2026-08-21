# Band-resim read — invert quiet_bon, do not ship the 0.85–1.15 constraint

**Date:** 2026-08-21
**Source:** `/tmp/band_resim.py` on bake-off `seed_bon` + tricks `hist_drop`
**Supersedes** the “band constraint” recommendation in
`2026-08-21_wan_v2v_picker_insight.md`.

## What the resim actually shows

Logged `prefix_motion` vs generated chunk motion are often **not on
the same living scale**:

| video | prefix_motion | notta tail | seed tail | prefix class |
|---|---:|---:|---:|---|
| 0000 | 0.0064 | 0.0149 | 0.0276 | quiet; search *invented* extra motion |
| 0001 | 0.0216 | 0.0064 | 0.0093 | **live prefix, notta collapsed**, slight recovery |
| 0002 | **0.0008** | 0.0235 | 0.0179 | **still prefix**, model invented motion, search damped |
| 0003 | **0.0009** | 0.0224 | 0.0192 | same as 0002 |
| 0004 | 0.0058 | 0.0306 | 0.0306 | quiet prefix, model hot, search ~tie |
| 0005 | 0.0052 | 0.0069 | 0.0067 | quiet, near prefix |
| 0006 | 0.0192 | 0.0186 | 0.0258 | live prefix, search raised |
| 0007 | **0.0701** | 0.0121 | 0.0262 | **live prefix, notta collapsed**, recovery |

For 0002/0003 the band `[0.85, 1.15] × 0.0008` is `[0.00067, 0.00091]`.
Every candidate sits at 0.01–0.04. Feasible set is **empty**. Fallback
`argmin |motion − prefix|` = **the stillest seed**. That is I2V-from-still
inside V2V. The constraint as written would freeze the videos whose
prefix is a hold.

On 0007 the band is `[0.060, 0.081]`. Every candidate is *below* it
(0.018–0.033). Fallback correctly pushes toward **higher** motion.
That is the collapse-recovery case.

Per-chunk switch counts (5–6 per clip) **cannot be turned into a new
tail_motion**. Later chunks condition on earlier commits. This dump is
a sensor diagnosis, not a counterfactual 30 s score.

## quiet_bon inverted the right gate

`quiet_bon` searched when `prefix < 0.018` and skipped when prefix was
already moving.

That is **backwards**:

- Still prefix (0002/0003, 0.0008): matching the prefix **damps** the
  motion the model was willing to invent. Should **not** search.
- Live prefix (0007, 0.070): notta collapsed 0.070→0.012. Search
  recovered to 0.026. Should **search**.

quiet_bon −19% at N=32 is that inversion at scale.

The earlier “0/7 hot” table used **notta tail ≥ 0.020**, not prefix
motion. It mixed two populations: still-prefix + invented tail (0002)
and live-prefix that stayed hot. The picker only has the prefix at t=0.

## Correct policy (live prefix only)

```
if prefix_motion >= T:          # living reference
    search; match prefix motion # recover collapse
else:                           # still / hold
    notta (ignore motion)       # do not match a photograph
```

T ≈ 0.012–0.015 (between 0000/0004 quiet and 0001/0006 live).

This is **`quiet_bon` with the inequality flipped.** Call it `live_bon`.

Unified I2V / V2V sentence: **never two-sided-match motion to a still
reference; only match motion when the reference is itself moving.**

N=8 sketch if T=0.012 (search 0001, 0006, 0007; skip the rest):

- Keep 0007 recovery and 0006 lift.
- Skip 0002/0003 → keep notta’s invented motion (fixes seed damping).
- Skip 0000 → **drop the largest N=8 lift** (0.015→0.028). That lift
  was “search invented extra motion on a quiet prefix,” the same
  unstable mode that becomes damping at N=32.

Do not ship the 0.85–1.15 band without the live-prefix gate. Empty
feasible ⇒ min-motion freeze.

## Next GPU (only if we still want a motion actuator)

One N=8 job: `live_bon` k=4 iff `prefix_motion >= 0.012`, else k=1.
Same 8 as the bake-off. Pair against notta / seed_bon. Promote only if
tail motion beats notta **and** 0002/0003 are not damped.

No hist_drop-32. No quiet_bon VBench. No band-constraint generate
until this gate exists.
