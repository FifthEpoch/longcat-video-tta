# Search-while-sick spec (2026-08-18)

**Status:** scored 2026-08-18, job 15959146. Checklist pass.
See `2026-08-18_wan_i2v_bon32_sick.md`.
**Series:** `i2v_bon_32v_sick` (gated-search only)
**Baseline:** do-nothing and always-search from `i2v_bon_32v_hybrid`
**Submit:** `wan_experiment/sbatch/submit_i2v_bon32_sick.sh`

## Rule

Same three hybrid alarms. Stay-on after the first alarm, **unless**
the last search recovered the video:

- Turn memory off if `incoming − outgoing > 0.5` (`--gate-recovery 0.5`).
- Or turn memory off if outgoing last-second `< 1.0` (`--gate-sick-min 1.0`).
- Alarms can still wake the video later. Forever stay-on (`--gate-sticky`
  with both knobs at 0) is unchanged, so `i2v_bon_32v_sticky` stays
  reproducible.

Log `recovery`, `gate_off_reason` (`recovered` / `healthy`), and
`already_on_after`.

## Predicted on the existing traces

| Video | After first search | Predicted |
|---|---|---|
| 11 smoke | 2.38→1.11, recovered 1.27 | off → skip piece 2. Piece 3 may alarm again (incoming 2.15). Piece 4 may stay on (recovery only 0.41). Prefix through piece 2 matches hybrid, not always-search. |
| 16 book on fire | outgoing 0.88 `< 1.0` | off → hybrid path |
| 30 church | 1.41→0.69 | off → hybrid / do-nothing |
| 03 highway | 1.27→1.32, still high | stay on. Piece 3 outgoing 1.02 recovers 0.66 → may turn off before piece 4 |
| 24 busy street | 1.05→1.05 | stay on |
| 06 / 07 | never wake early | still skipped on piece 1 |

## Pass / fail

1. 11 and 16 last-piece back near hybrid (2.16 / 2.66), not 4.32 / 5.05.
2. 03 and 24 still near always-search (1.57 / 2.32).
3. 06 / 07 still skipped on piece 1.
4. 30 back to 1.44, not 1.69.
5. Wall-clock between hybrid (173 s) and forever-sticky (256 s).

No test-time training.
