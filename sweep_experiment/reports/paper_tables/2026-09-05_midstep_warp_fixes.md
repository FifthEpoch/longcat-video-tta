# Mid-step warp — remaining holes and idea fixes (2026-09-05)

After the glossary: `pred` = the model’s guess of
the finished strip; `extra` = the fresh white snow
blended back in before the next pass. Closed already:
HIWYN / GwF on `extra` only (Gaussianity), after
pass 1 not at t=250, do not move `pred`, no torus,
Self Forcing host only.

This note lists what is still open and the
modification to the idea for each. Canvas:
`canvases/midstep-warp-fixes.canvas.tsx`.
Not a submit. No GPU.

---

## Remaining holes → change the idea

### 1. Later passes draw a new white `extra`

If we HIWYN only the snow for the 750 blend, the
500 and 250 blends inject plain snow again and the
velocity dies.

**Modification:** The same HIWYN recipe on **every**
`extra` after pass 1 (750, 500, and 250 blends).
Same leftover velocity. New white specks only in
holes. Pass 1 still sees ordinary opening static
so the first `pred` stays the student’s usual guess.

### 2. Each strip is only 3 frames, then we reset

HIWYN on one strip can encode a short drift. The
next strip would start a new snow field. Motion
can hitch at every lock (~0.75 s).

**Modification:** Keep the HIWYN **particle field
across strips**. Strip \(n+1\)’s first `extra`
continues from strip \(n\)’s last `extra`, same
velocity. One 30 s snow trajectory, not 40
unrelated ones.

### 3. Where the direction comes from

RAFT on a half-clean `pred` is a feedback loop.
An arbitrary “always shift right” can fight the
leftover. Prefix flow goes stale, but it is the
only real motion we have at t = 0.

**Modification:** Measure mean leftover flow
**once** (direction + speed). Freeze it for the
whole 30 s. No mid-pass RAFT. No invented +x.

### 4. Two motions at once

The leftover in memory already is optical flow.
A second, different snow velocity fights it.

**Modification:** The frozen velocity **is** that
leftover mean. We are continuing the opening’s
motion in the snow, not adding a new one.

### 5. The student never trained on drifting snow

Go-with-the-Flow LoRA’d CogVideoX for this reason.
A frozen Self Forcing student may ignore the drift
or twitch. We cannot delete this hole without a
new student (occupied as their paper).

**Modification (soften, not delete):** Mix each
transported `extra` with a little plain snow, their
\(\gamma\). Default \(\gamma \approx 0.5\): half
drift, half ordinary snow, so the input is not a
fully foreign prior. If N=8 still paints or
no-ops, stop; do not start their LoRA.

### 6. Official Dynamic Degree can still flip from junk

If anything in the snow makes a hard edge, RAFT’s
top-5% bit fires and flicker dies. HIWYN hole-fill
is the prevention; the hold is the judge.

**Modification:** No wrap, no `pred` slide (already).
Harvest letters stay mixctx: Dyn up **and** Imaging
Quality holds **and** flicker off the twitch band
**and** subject holds. A Dyn-only lift is a fail.

### 7. Rolling Forcing’s window is several noise ages

“The” `extra` is not one field there.

**Modification:** Self Forcing host only. Not an
idea change; a lock.

---

## Revised recipe (one paragraph)

Pass 1 as today (ordinary opening static → first
`pred`). Do not touch that `pred`. For every later
blend, build `extra` with HIWYN: spatially white
per frame, transported along the **frozen leftover
mean flow**, holes resampled, mixed with plain snow
at \(\gamma \approx 0.5\). Carry that particle field
into the next strip. Lock the last `pred` as today.
Caption N=8, always-on + leftover-live gated twin,
mixctx letters. No GPU until the user says go.

---

## Do not

Warp `pred`. Wrap the grid. Hook only the last
pass. RAFT on `pred`. Start on Rolling Forcing.
Start 8-GPU DMD / CogVideoX LoRA.
