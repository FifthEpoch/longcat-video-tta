# H2/H3 host-switch verdict (2026-08-22 19:31)

Offline on existing SF notta + RF rolling mp4s. No GPU.
H2 = keep the 30 s video whose **generated** chunk-0 moved more.
H3 = RF unless RF chunk-0 `< 0.8 ×` SF chunk-0.
Not the prefix-motion gate (that already lost: +9% vs always-RF +31%).

Jobs: H1/H4 **16215197–200** + VBench **16215201** submitted;
this table does not wait on them.

## Locked bar = N=128 median tail vs always-RF

| Arm | N | med | vs RF med | mean | vs RF mean | vs best | SF picks | ρ(Δc0, Δtail) | Call |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| always-RF | 128 | 0.01771 | — | 0.01861 | — | 88/40 | 0 | — | host |
| H2 bake | 128 | 0.01676 | **−5.4%** | 0.01847 | −0.8% | 81/47 | 73 | 0.506 | **NO** |
| H3 veto | 128 | 0.01771 | **+0.0%** | 0.01923 | +3.4% | 92/36 | 50 | 0.506 | **NO** |

Bake is **worse** at picking the better 30 s video than always-RF
(81/47 vs 88/40). It over-picks Self-Forcing: SF often twitches
harder in chunk 0 and then dies. ρ = 0.51 is “related, not a
router.”

Veto median **ties** the host. The +3.4% mean / +4 extra wins are
the known RF collapses (0004, 0027, 0035, 0044, 0087, …). It also
steals RF wins (e.g. 0047: c0 says SF, tail says RF). Net median
does not move. Not a method.

## The N=8 / N=32 trap (do not cite)

| Arm | N=8 vs RF | N=32 vs RF | N=128 vs RF |
|---|---:|---:|---:|
| H2 bake med | +5.4% | +3.5% | **−5.4%** |
| H3 veto med | +5.4% | +4.7% | **+0.0%** |
| bake SF picks | 3/8 | 15/32 | 73/128 |
| ρ | 0.619 | 0.604 | 0.506 |

Same story as seed / live / look: the small-N lift dies when the
pool is honest. First 32 of the 128 pair **bit-match** the N=32
pair.

## Closed

- Do **not** GPU a chunk-0 host router.
- Do **not** retune `ROLL_TRUST_FRAC = 0.8` after seeing 128.
- H1 (`sf_roll` / `rf_chunk`) and H4 (`*_recache`) stay. Those
  jobs are already in (**16215197–201**).
- Always-RF remains the host. Do not rebrand it as our controller.
