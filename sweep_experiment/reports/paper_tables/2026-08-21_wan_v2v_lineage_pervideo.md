# V2V lineage N=8 per-video tails (2026-08-21 12:33)

**Status:** generate partial. live_bon / live_hist / longlive_notta
n=8/8. Host jobs 16140812–815 and VBench still queued.

**Source:** `pair_v2v_tails.py` on `summary.json` rows.
**Baseline:** bake-off notta + seed_bon.

## Population (unchanged)

| Method | median tail | vs notta | vs notta losses | Honest |
|---|---:|---:|---:|---|
| notta | 0.0167 | — | — | baseline |
| seed_bon (bake-off) | 0.0225 | +35% | 0002, 0003 | identity damper on stills |
| live_bon | 0.0229 | +37% | **0/8** | Gate works. VBench still open. **Do not scale to 32.** |
| live_hist | 0.0229 | +37% | **0/8** | Same skip set; searched clips = hist_drop |
| longlive_notta | 0.0150 | −10% | 0002, 0004, 0006 | HOLD |

## Per-video tail motion

| video | prefix | gate | notta | seed_bon | live_bon | live_hist | longlive_notta |
|---|---:|---|---:|---:|---:|---:|---:|
| 0000 | 0.00638 | skip | 0.01486 | 0.02761 | **0.01486** | **0.01486** | 0.02165 |
| 0001 | 0.02160 | search | 0.00641 | 0.00933 | **0.00933** | **0.01048** | 0.01500 |
| 0002 | 0.00079 | skip | 0.02350 | 0.01787 | **0.02350** | **0.02350** | 0.00979 |
| 0003 | 0.00094 | skip | 0.02236 | 0.01917 | **0.02236** | **0.02236** | 0.01943 |
| 0004 | 0.00583 | skip | 0.03062 | 0.03061 | **0.03062** | **0.03062** | 0.01793 |
| 0005 | 0.00525 | skip | 0.00686 | 0.00671 | **0.00686** | **0.00686** | 0.01130 |
| 0006 | 0.01922 | search | 0.01864 | 0.02584 | **0.02584** | **0.02967** | 0.01358 |
| 0007 | 0.07014 | search | 0.01210 | 0.02617 | **0.02617** | **0.02856** | 0.01499 |

Bold live_* cells: skip ⇒ exact notta; search ⇒ exact seed_bon
(live_bon) or hist_drop (live_hist). n_div on searches: 0001=4,
0006=5, 0007=3.

## What this answers

The live gate (`prefix_motion >= 0.012`) is the first search policy
that is **net-nonnegative vs notta on every clip** at N=8.

- Still prefixes 0002/0003 (0.0008): seed_bon damped; live_* **left
  notta’s invented motion alone**.
- Live prefixes 0001/0006/0007: search fired; tails bit-match the
  corresponding always-on search.
- 0004/0005 (prefix ~0.005–0.006): skip, exact notta. Harmless.

**0000 is a false negative.** prefix 0.00638 < 0.012, so we skipped
a collapse that seed_bon recovered (0.01486 → 0.02761). The +37%
median does **not** need 0000 — it comes from keeping 0002/0003
high and recovering 0006/0007.

Earlier reconstruction (“0000+0007 searched”) was wrong. 0000 was
skipped. The 0.0229 median is 0006+0007 recoveries + 0002/0003
kept.

## Threshold residue

0000 (0.00638) sits above 0002/0003 (0.0008) and next to 0004/0005
(~0.0055). A lower `live_min` (~0.006) could catch 0000; 0004 is
already hot as notta so searching it is optional. Do **not** retune
and resubmit while 812–815 / ideas are queued. N=8 note only.

## Still closed

- VBench IQ / subject / Dyn on live_* (job 16140816 after remaining
  generate).
- N=32. seed_bon-8 also looked like +35%. This gate is a different
  *sign pattern* (0 losses), but N=8 is still the lucky 8.
- LongLive as a Dyn fix. 0002/0004 collapsed vs SF notta.

Do not scancel 812–815 or 16145125–131.
