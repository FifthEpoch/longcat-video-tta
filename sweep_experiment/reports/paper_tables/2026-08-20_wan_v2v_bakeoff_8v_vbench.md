# V2V N=8 bake-off — full-clip VBench (official)

**Date:** 2026-08-20
**Series:** `v2v_panda_bakeoff_8v`
**Generate jobs:** 16092846–849. VBench `joined.json` on disk for all 4
methods (extra job 16110491 skips).
**N:** 8 paired. Cite **medians**. Official number = **full clip**.
**Analyzer:** `analyze_v2v_bakeoff.py` after `a273c41` population.median fix.

## Generate-tail motion (from mp4 json)

| Method | tail \|Δframe\| | vs notta |
|---|---:|---:|
| notta | 0.0167 | — |
| seed_bon | **0.0225** | **+35%** |
| motion_bon | 0.0148 | −11% |
| backtrack | 0.0130 | −22% |

## Full-clip VBench quality 7

| Method | Subj | BG | Aes | IQ | Smooth | Dyn med (mean) | Flicker |
|---|---:|---:|---:|---:|---:|---:|---:|
| notta | 0.5951 | 0.8070 | 0.5422 | 67.98 | 0.9902 | 0.00 (0.375) | 0.9828 |
| seed_bon | 0.5956 | 0.7924 | 0.5377 | 67.38 | 0.9882 | **0.50 (0.50)** | 0.9782 |
| motion_bon | 0.5883 | 0.8093 | 0.5453 | 68.77 | 0.9916 | 0.00 (0.375) | 0.9857 |
| backtrack | 0.5951 | 0.8283 | 0.5372 | 65.04 | 0.9921 | 0.00 (0.25) | 0.9857 |

`dynamic_degree` is 0/1 per video. Median 0.5 = 4/8 clips dynamic.
notta mean 0.375 = 3/8. N=8 is noisy on a coin-flip metric.

## Locked promote rule

Promote past N=8 only if tail motion > notta **and** IQ not worse by ≥1.0
**and** subject not worse by ≥0.02.

| Method | Motion | IQ Δ vs notta | Subj Δ vs notta | Verdict |
|---|---|---:|---:|---|
| seed_bon | +35% | **−0.60** (under the 1.0 bar) | **+0.0005** | **PROMOTE** |
| motion_bon | −11% | +0.79 | −0.0068 | HOLD — no motion gain |
| backtrack | −22% | **−2.94** (fails IQ) | 0 | HOLD — worse motion and IQ |

## Read

On a **real Panda prefix**, k=4 seed search (old deviation pick) raises
tail `|Δframe|` and flips official `dynamic_degree` median 0→0.5 without
breaking subject or blowing the IQ slack. That is the opposite of
I2V-from-still, where seed-BoN stayed in the freeze attractor.

Greedy per-chunk `|Δframe|` (`motion_bon`) and prefix backtrack do **not**
win. Drop them. Shift/CFG stay dead (probe).

Do not cite last-chunk drift 53–95 as quality. Subject ~0.59 is a V2V
continuation number, not comparable to I2V-32 full-clip subject ~0.85.

## Next (allowed, not submitted)

`notta` vs `seed_bon` only, N=32, same V2V protocol. Confirm the +35%
tail motion and Dyn 0.5 hold. No TTC. No I2V-32 scale-up. No motion_bon
/ backtrack / shift_search.
