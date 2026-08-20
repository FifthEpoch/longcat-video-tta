# V2V N=8 sampling-space bake-off (generate only)

**Date:** 2026-08-20
**Series:** `v2v_panda_bakeoff_8v`
**Jobs:** **16092846** notta 18m, **16092847** seed_bon 51m,
**16092848** motion_bon 51m, **16092849** backtrack 23m. All COMPLETED 0:0.
**N:** 8 paired Panda prefixes. Cite **medians**.
**Analyzer:** `analyze_v2v_bakeoff.py` (no VBench yet).

| Method | tail motion | vs notta | last-chunk drift ↓ | last-chunk motion pick ↑ | decision |
|---|---:|---:|---:|---:|---|
| notta | 0.0167 | — | 63.51 | −0.787 | baseline |
| seed_bon | **0.0225** | **+35%** | 53.03 | −2.122 | **CONDITIONAL** — motion win; VBench IQ/subject required |
| motion_bon | 0.0148 | −11% | 82.20 | −0.876 | HOLD — greedy `|Δframe|` did not raise the 30 s tail |
| backtrack | 0.0130 | −22% | 95.25 | −0.786 | HOLD — worse than notta |

## Read

- **seed_bon** (the I2V-32 control) is the only method that raised
  generated-tail `|Δframe|`. On a **real prefix**, four seeds are not
  the same freeze attractor we saw on I2V-from-still. Do **not** promote
  to N>8 until full-clip VBench IQ / subject hold.
- **motion_bon** lost. Picking the chunk with more `|Δframe|` made the
  *full tail* less motion. Local flicker / seam-y jumps do not compose
  into a more dynamic 30 s clip.
- **backtrack** lost and was cheap (23 m vs 51 m search). After the
  6336-drift guard it barely intervened, or it intervened on the wrong
  chunks.
- Last-chunk drift is still on a weird scale (50–95). Use tail motion
  and VBench, not that composite, as the paper number.

## Locked next

Score **full-clip** VBench quality 7 on all four method dirs. Then
re-run `analyze_v2v_bakeoff.py`. Promote seed_bon only if IQ is not
worse by ≥1.0 and subject is not worse by ≥0.02. Do not scale. No TTC.
No `shift_search` (probe dead).
