# V2V knob probe — shift and CFG are dead on Self-Forcing DMD

**Date:** 2026-08-20
**Jobs:** smoke **16069897** (7m08s), probe **16069898** (5m38s). Both COMPLETED 0:0.
**Series:** `v2v_panda_smoke` / `v2v_panda_probe`
**N:** 2 Panda prefixes

## Probe (first generated chunk only)

Every `(shift, cfg)` cell has the **same** mean `|Δframe|` **0.01626**.
`shift_live=0/2`, `cfg_live=0/2`.

| shift | cfg | mean motion |
|---:|---:|---:|
| 5 / 8 / 12 | 1 / 3 / 5 | 0.01626 (all nine) |

`apply_shift` and `apply_guidance` do not change pixels on this checkpoint.
DMD default is CFG-free (`guidance_scale=1.0`). Flow `shift` is not wired
through the chunked `generator()` loop (or the scheduler ignores it after
init). Wave-2 sink / a real scheduler hook would be required to make
`shift_search` a method.

**Decision:** drop `shift_search`. Drop CFG. N=8 submit is
`SKIP_SHIFT=1 bash wan_experiment/sbatch/submit_v2v_bakeoff.sh`.

## Smoke (NOTTA, 30 s tail)

| Method | tail motion | last-chunk drift | last-chunk motion pick |
|---|---:|---:|---:|
| notta | 0.0106 | 6336.4 | −1.713 |

Tail motion is already low vs the probe’s first-chunk 0.016. The 6336
composite is prefix-vs-tail scale clash (do not use as a quality number).
Backtrack after this commit ignores drift > 100 and keys off motion
collapse.

VBench not scored on the smoke (N=2 discovery only).
