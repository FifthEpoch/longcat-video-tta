# V2V bake-off scorecard — pending cluster generate

**Date:** 2026-08-20
**Series:** `v2v_panda_bakeoff_8v` (after `v2v_panda_smoke` + `v2v_panda_probe`)
**Analyzer:** `wan_experiment/scripts/analyze_v2v_bakeoff.py`
**Spec:** [`2026-08-20_wan_v2v_sampling_bakeoff_spec.md`](2026-08-20_wan_v2v_sampling_bakeoff_spec.md)

Do **not** fill this table by hand from chat. After the N=8 jobs finish:

```
python wan_experiment/scripts/analyze_v2v_bakeoff.py \
  --series-dir wan_experiment/results/v2v_panda_bakeoff_8v \
  --out sweep_experiment/reports/paper_tables/2026-08-20_wan_v2v_bakeoff.md
```

Then score each method dir with `score_i2v_vbench.py --clip full` and re-run
the analyzer. Write a **new** dated file if numbers change.

## Methods (wave 1)

| Method | What it searches | Pick rule |
|---|---|---|
| `notta` | nothing | default shift=8, seed 0 |
| `seed_bon` | k=4 seeds | lowest I2V-32 deviation composite (control) |
| `motion_bon` | k=4 seeds | highest one-sided `|Δframe|` |
| `shift_search` | shift ∈ {8,5,12} | highest motion; drop if probe `shift_live=false` |
| `backtrack` | rewind + 1 resample | fire if outgoing drift>2.0 or motion<0.4×prefix |

## Decision rule (locked)

Cite **medians**. Promote past N=8 only if:

1. Median generated-tail `|Δframe|` **> notta**, and
2. Full-clip VBench `imaging_quality` is not worse by ≥ 1.0, and
3. Full-clip `subject_consistency` is not worse by ≥ 0.02.

A smoothness / flicker “win” that is a freeze does not count. No PSNR.
No TTC. No I2V-32 scale-up.

## Status

Generate **not yet run on cluster**. This file is the decision lock, not a
result. Supercede it with `2026-08-20_wan_v2v_bakeoff.md` (or a later date)
once `summary.json` files exist.
