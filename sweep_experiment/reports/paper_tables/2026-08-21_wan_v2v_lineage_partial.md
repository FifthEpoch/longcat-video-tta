# V2V lineage N=8 partial (2026-08-21 12:24)

**Status:** generate partial. `live_bon` / `live_hist` / `longlive_notta`
DONE n=8/8. Host variants 16140812–815 still queued. VBench 16140816
not started. Ideas 16145125–131 still queued.

**Series:** `wan_experiment/results/v2v_panda_lineage_8v`
**Baseline:** bake-off notta `v2v_panda_bakeoff_8v`
**Analyzer:** `analyze_v2v_bakeoff.py --allow-partial`
**Cite medians. Paired N=8.**

| Method | tail motion | vs notta | last-chunk drift ↓ | last-chunk motion ↑ | script | Honest |
|---|---:|---:|---:|---:|---|---|
| notta | 0.0167 | — | 63.5052 | −0.7870 | baseline | baseline |
| live_bon | **0.0229** | **+37%** | 63.5052 | −0.7870 | PROMOTE | **CONDITIONAL** — VBench + per-video gate still open |
| live_hist | **0.0229** | **+37%** | 63.5052 | −0.7870 | PROMOTE | **CONDITIONAL** — same |
| longlive_notta | 0.0150 | −10% | 47.4005 | −0.8817 | HOLD | HOLD — host did not beat SF notta |

VBench full-clip only exists for bake-off notta (subject 0.5951, IQ
67.98, Dyn 0). live_* have no official VBench yet. The script treats
missing IQ/subject as a pass — that is why it printed PROMOTE.

## How to read the +37%

Bake-off seed_bon-8 was +35% (0.0225) and later **failed at N=32**.
Do not treat this as “scale live_bon to 32.”

Reconstructed from known bake-off tails: replacing **only 0000 and
0007** with their seed_bon / hist_drop values and leaving the other
six as notta yields median **0.02293**. That is this table’s 0.0229.

| clip | bake-off notta | seed_bon | hist_drop | live_* prediction |
|---|---:|---:|---:|---|
| 0000 | 0.01486 | 0.02761 | 0.02761 | searched (recovery) |
| 0002 | 0.02350 | 0.01787 | 0.01958 | **should skip** (still prefix, seed damped) |
| 0003 | 0.02236 | 0.01917 | 0.01993 | **should skip** |
| 0007 | 0.01210 | 0.02617 | 0.02856 | searched (log: 0.02617 / 0.02856) |

Last-chunk drift **bit-matches notta** (63.5052). That is the still-
prefix majority: skipped videos are NOTTA twins, so the median
last-chunk is the skip.

0007 logs already bit-match: live_bon 0.02617 = seed_bon, live_hist
0.02856 = hist_drop.

## What would kill it

If 0002/0003 tails match seed_bon (damped) rather than notta, the
live gate did not skip stills and this is seed_bon-8 again.

If VBench IQ drops ≥1.0 or subject drops ≥0.02, HOLD.

## longlive_notta

0.0150 < SF notta 0.0167. 0007 was 0.01499 (still collapsed vs
seed 0.026). Swapping to LongLive-1.3B is not the 30 s Dyn fix on
this V2V protocol. Wait for sink / prefix_sink / RF before closing
the host question.

## Do not

- Scale live_bon / live_hist to N=32 today.
- Scancel 812–815 or the ideas wave.
- Cite the script’s PROMOTE as a paper decision.
