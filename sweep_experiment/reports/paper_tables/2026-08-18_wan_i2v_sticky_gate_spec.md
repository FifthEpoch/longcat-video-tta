# Sticky gated-search spec (2026-08-18)

**Status:** implemented, not yet scored. This is the lock, not a result.
**Series:** `i2v_bon_32v_sticky` (gated-search only)
**Baseline:** reuse do-nothing and always-search from `i2v_bon_32v_hybrid`
**Submit:** `wan_experiment/sbatch/submit_i2v_bon32_sticky.sh`

## Rule

Same three alarms as the hybrid gate. New memory:

- Off at the start of each video.
- The first time any alarm fires, search four candidates.
- After that, search every later piece of that same video, even if
  incoming no longer trips an alarm. Log reason `already_on` when the
  search is only because of memory.

Alarms (unchanged):

1. Piece 1 and incoming > 0.8
2. incoming > 2.0
3. incoming jumped more than 0.5 and the previous incoming was already
   above 0.5

`--gate-sticky` is off by default, so `i2v_bon_32v_hybrid` stays
reproducible.

## What this run is for

On the hybrid 32-video set:

- Videos 03 and 24 woke up on piece 1, then went back to sleep.
  Always-search kept working and won.
- Videos 06, 07, 28, 30 must stay untouched on early pieces.
- Video 17 never wakes up. This run will not fix it.
- Video 26 already explodes when we search. Sticky can make that worse.

## Pass / fail after the job

1. Did 03 and 24 move toward always-search?
2. Did 06 / 07 / 28 / 30 stay skipped on piece 1?
3. Did any new video explode the way 26 did?

Do not write a quality win until those three are checked. Do not start
test-time training.
