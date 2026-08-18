# Why videos 11 and 16 got worse under stay-on (2026-08-18)

**Source:** paired traces from `i2v_bon_32v_hybrid` and `i2v_bon_32v_sticky`
(same images, same seeds). Last-piece scores: lower is better.
This is a diagnosis + intervention brainstorm, not a new experiment.

## What “worse” means

| Video | Do-nothing | Always-search | Hybrid gated | Sticky gated |
|---|---|---|---|---|
| 11 smoke | 11.192 | 4.319 | **2.157** | 4.319 (exact always-search) |
| 16 book on fire | 4.776 | 5.047 (always-search *hurt*) | **2.656** | 5.047 (exact always-search) |

Hybrid had unique wins always-search did not have. Stay-on erased
both and landed on always-search’s ending. That is not noise: after
piece 1 the two methods still share a prefix, then stay-on copies
always-search’s later picks.

## Piece-by-piece (the same story twice)

`incoming` = drift of the last second already committed, before the
new piece. `outgoing` = drift of the last second after we commit this
piece. `pick vs default` = chosen candidate minus the default-seed
candidate (negative = search thinks it won).

### Video 11 — smoke

| Piece | Hybrid | Sticky | Incoming | What happened |
|---|---|---|---|---|
| 1 | search (early+late alarm), pick −1.098 | same | 2.384 → out 1.112 | Shared recovery. Drift **falls by 1.27**. |
| 2 | **skip** | search (`already_on`), pick −0.222 | 1.112 | Hybrid sees a healthy-ish video and sleeps. Sticky takes a small “win.” Outgoing 1.499. Prefix splits. |
| 3 | search, pick −0.392 | search, pick 0 | hybrid in 2.153 / sticky in 1.499 | Different prefixes, different picks. |
| 4 | **skip** | search, pick **−4.108** | hybrid in 1.746 / sticky in 2.303 | Sticky’s local score says a huge win. Outgoing **4.912**. Last-piece 4.319. |

Hybrid’s win is the two skips after the recovery. Stay-on forbids those skips.

### Video 16 — book on fire

| Piece | Hybrid | Sticky | Incoming | What happened |
|---|---|---|---|---|
| 1 | search (early alarm), pick −0.384 | same | 0.844 → out 0.879 | Shared. Incoming stays **healthy** (~0.88). |
| 2 | **skip** | search (`already_on`), pick −0.406 | 0.879 | Hybrid sleeps. Sticky “wins” locally. Outgoing 1.608 (already worse). |
| 3 | search (rising), pick 0 | search, pick −1.051 | hybrid in 1.756 / sticky in 1.608 | Sticky local win; outgoing **3.507**. |
| 4 | search, pick −1.339 | search, pick **−3.640** | hybrid in 2.554 / sticky in 3.507 | Sticky local score keeps “winning.” Outgoing **7.447**. Last-piece 5.047. |

Always-search also ends at 5.047 (worse than do-nothing). Hybrid beat
both by sleeping on piece 2 while the video still looked fine, then
searching later on a better prefix.

## The mechanism (one sentence)

After a useful first search, incoming looks better or still healthy.
Hybrid goes back to sleep and keeps that prefix. Stay-on keeps
searching, the pick-score says each new candidate is better than the
default seed, and the **tail of the piece** (last-second outgoing)
gets worse. Because piece 1 still matches always-search, later picks
rebuild always-search’s path.

Two separate bugs are stacked:

1. **The on/off switch has no off.** Stay-on cannot notice a recovery
   (11: 2.38→1.11) or a still-healthy video (16: 0.88).
2. **The pick-score is not the ending.** Full-piece score (whole ~6 s
   plus the seam) can improve while the last second of that piece is
   dying. Video 16 piece 4: pick-score −3.640, last-second outgoing
   7.447. Video 11 piece 4: pick-score −4.108, outgoing 4.912.

Same pattern, smaller: video 01 recovered 1.27→0.80; hybrid slept and
beat always-search (1.936 vs 2.104); stay-on copied always-search.
Video 30 recovered 1.41→0.69; hybrid slept and matched do-nothing;
stay-on copied always-search’s harm.

## Same data, opposite of 03 / 24

| Video | After piece 1 | What stay-on should do | What we want |
|---|---|---|---|
| 11 | incoming **dropped** 2.38→1.11 | stayed on | **sleep** (keep the recovery) |
| 16 | incoming still **low** (0.88) | stayed on | **sleep** |
| 30 | incoming **dropped** 1.41→0.69 | stayed on | **sleep** |
| 03 | incoming stayed high/flat 1.27→1.32 | stayed on | **keep searching** (the first pick was a coin flip; the win is later) |
| 24 | incoming flat 1.05→1.05 | stayed on | **keep searching** |

A large piece-1 pick-score is **not** the signal to stay on. Video 11
had −1.098 and should sleep. Video 03 had −0.01 and should stay on.
That is backwards from “stay on if the first search helped a lot.”

The useful signal is **incoming after the search**: did the video
recover or is it still sick?

## Intervention ideas (predicted on these traces)

Do not implement all of these. They are alternatives. Predicted
effect assumes we keep the three hybrid alarms and only change
memory and/or how we accept a candidate.

### A. Search while sick (recovery off-switch)

Stay on after an alarm **only if incoming is still high and did not
just fall**. After a search, if outgoing dropped by more than ~0.5,
or incoming is now below ~1.0, turn the switch off.

| Video | Predicted |
|---|---|
| 11 | piece 1 recovers 1.27 → off → hybrid path. **Keep the unique win.** |
| 16 | incoming still 0.88 → off → hybrid path. **Keep the unique win.** |
| 30 | recovers 1.41→0.69 → off. **Re-save 30.** |
| 03 | 1.27→1.32, still high → stay on. **Keep the 03 catch.** |
| 24 | 1.05→1.05, still ~1 → stay on. **Keep the 24 catch** (threshold-sensitive). |
| 06 / 07 | never wake early. Safe. |
| 17 | still never wakes. Unfixed. |

This is the cheapest next experiment: one extra off condition, same
32 videos, gated-search only.

### B. Refuse a candidate whose last second got worse

After scoring the four candidates, do not commit one that ends with
last-second outgoing worse than the incoming we started with, unless
it is the default seed. Gate and stay-on can stay as they are.

| Video | Predicted |
|---|---|
| 11 piece 2 | incoming 1.11, sticky outgoing 1.50 → refuse, keep default. |
| 11 piece 4 | incoming 2.30, outgoing 4.91 → refuse the −4.108 “win.” |
| 16 piece 2–4 | outgoing 1.61 / 3.51 / 7.45, each worse than incoming → refuse. |
| 03 piece 3–4 | outgoing 1.02 then 0.96, **better** than incoming → keep. Still catch 03. |

This attacks bug 2 (the lying pick-score). It can sit under hybrid or
under stay-on. Risk: 24’s middle pieces get worse outgoing before
the last piece improves; a hard refuse might block the 24 catch.

### C. Pick on the last second of the new piece, not the whole piece

Same candidates, different score: use last-second outgoing (no
full-piece average, no seam-on-the-whole-window) to choose. The
full-piece score is what let 16 piece 4 look like −3.640 while the
tail was 7.45.

This is the cleanest fix of the pick-score if we believe the ending
of the piece is what matters for the next piece. Needs a 32-video
gated re-run (or always-search too, because the pick itself changes).
More expensive than A. Do not combine with a new gate in the same run.

### D. Stay on only when incoming is rising

Stay on only if the trend alarm is live (incoming jumped > 0.5).
Would save 11 (incoming fell) and maybe 16 (flat). **Would lose 03
and 24** (flat after piece 1). That undoes the whole stay-on run.
Reject unless we no longer care about 03/24.

### E. Weaker wake-up for video 17

Incoming 0.76–1.20, max jump 0.40, never crosses 2.0. A weaker
trend (jump > 0.3) or a freeze/sharpen-only alarm might wake it.
This does **not** follow from 11/16 and will have false positives
(06/07 sit nearby). Separate experiment, after A or B.

### F. Test-time training

Still no. 11 and 16 show that “the video needs help” is not the same
as “keep intervening.” Training on a recovered prefix, or on a
candidate the pick-score likes, would likely replay 11/16 at higher
cost.

## What I would run next

**A first** (search while sick), gated-search only, same 32 videos.
It is the only idea that, on these traces, keeps 11 and 16 **and**
03 and 24 **and** re-saves 30. Success is not “beat always-search
on the mean.” Success is:

1. 11 and 16 last-piece back near hybrid (2.16 and 2.66), not 4.32 / 5.05.
2. 03 and 24 still match always-search.
3. 06 / 07 still skipped on piece 1.
4. 30 back to do-nothing (1.444), not 1.688.
5. Wall-clock between hybrid (173 s) and stay-on (256 s), not glued to 256.

If A holds, we have a controller that searches while the last second
looks sick and stops when it recovers. That is a better paper sentence
than “stay on forever” or “one cutoff for every piece.”

If A fails because 24’s incoming sits right on 1.0, tighten the
definition of “sick” using the same traces before touching the
pick-score (B or C).
