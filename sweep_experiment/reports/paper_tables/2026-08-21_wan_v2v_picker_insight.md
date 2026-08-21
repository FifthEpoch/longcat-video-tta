# Insight — why prefix-match search damps at N=32, and what would actually change it

**Date:** 2026-08-21
**Status:** analysis, not a new job. Licensed by sidecar N=32 + N=8 VBench +
quiet/tail generate.

## What the picker *is*

`score = |Δsharp| + |Δcolor| + |Δcontrast| + |Δmotion| + 1.0·seam`

Four relative deviations plus seam, **one sum, lower wins**. Motion is
not a constraint. It is one vote in a five-vote committee, and
appearance+seam can outvote it.

That is why the same code:

- **unfroze** N=8 when notta had already fallen *below* the moving
  prefix — `|Δmotion|` on cand0 was large, a closer-to-prefix seed won,
  tail `|Δframe|` went up (0000 +0.013, 0007 +0.014, 0018 +0.017);
- **damped** N=32 hot prefixes 0/7 — notta was already near the prefix,
  `|Δmotion|` on cand0 was small, a seed with extra twitch lost on
  sharpness/seam, the picker kept the quieter match;
- **raised VBench subject +0.039** at N=32 — quieter, prefix-like tails
  are exactly what `subject_consistency` likes;
- **froze I2V** — the reference was a near-still, so the same sum
  matched a near-still.

H-match was right about the *objective* (match the prefix) and wrong
that this objective is a *dynamics* method. It is **identity control**.
Dynamics only move when notta undershoots the prefix.

Mean Δ over 32 is ~0. A few collapse-recoveries, more small cuts.
Median −8.8%. Dyn median 0. The N=8 Dyn 0→0.5 was two recoveries
(0000, 0007) in a list that over-weighted mid clips.

## Why the follow-ups behaved that way

| Method | What we thought | What the data say |
|---|---|---|
| hinge | stop rewarding extra twitch | Too loose. 0004 −0.013 vs notta. Extra motion unconstrained, appearance still in the sum. Lost to two-sided on the lucky 8. |
| late_bon | search only after collapse | Missed the recoveries. 0000/0007 lift in **early** chunks, before `incoming` exists. N=8 −10%. |
| quiet_bon | skip search if prefix already moving | Right on 0022/0027/0028 (bit-match notta). Wrong that “prefix quiet” ⇒ “will collapse.” Still searched (and damped) the rest. −19%. |
| hist_drop | History Guidance | 7/8 a **+0.001…+0.004 bump on seed_bon**. Tail is an extra candidate, not a policy. Still lost 0002/0003 (hot). |
| tail_hist | maybe hist_drop *is* short history | ≈ notta (+0.8%). Unsticks **0002** (+0.015) and **erases** 0000/0006/0007. Different mode. |
| motion_bon | max \|Δframe\| | Flicker, then collapse. Already falsified. |

## The actual actuator

```
if notta_chunk.motion << prefix.motion:
    four seeds can recover   # 0000, 0007, 0018, 0026
else:
    four seeds can only match or damp   # all 7 hots, most quiets
```

Search is **collapse recovery**, not a motion booster. A gate on
*prefix* quiet/hot (quiet_bon) is the wrong sensor. The sensor has to
be **this chunk vs the prefix**, and it has to fire on chunk 0 (where
the N=8 wins lived).

## What would change the picker (highest leverage)

**Motion as a band constraint, appearance as the objective.**

```
feasible = {c | 0.85·prefix ≤ motion(c) ≤ 1.15·prefix}
if feasible:
    pick min(appearance + seam) among feasible
else:
    pick argmin |motion − prefix|     # recover; ignore appearance
```

- Hot + notta already in-band → cannot pick a damper below 0.85×.
  Fixes 0/7.
- Collapse, nobody in-band → recover toward prefix. Keeps 0000/0007.
- Hinge failed because there was no ceiling. Two-sided failed because
  appearance could buy a motion cut.

This can be **resimulated offline** on bake-off `seed_bon` / `hist_drop`
sidecars (`candidates[].temporal_motion` + prefix_motion). Zero GPU. If
the resimulated pick beats two-sided on the 8 *and* stops damping 0002
/ 0003 / 0004, then it earns an N=8 generate. If it does not, the
constraint is wrong and we do not spend a GPU.

## What would change the *policy* (second)

Two cheap compositions, only after the score resim:

1. **Always search chunk 0, then gate.** late_bon’s bug was skipping
   the chunk where recoveries happen. Fix: k=4 on ci=0; after that,
   k=4 only if outgoing < 0.8× prefix.
2. **`hot_tail`:** if prefix ≥ 0.018, k=1 last-3-latent history
   (tail_hist’s 0002 lift); else notta. No seed search. Tests whether
   short history is a hot-prefix specialist.

Do **not** scale hist_drop to 32. It is seed_bon plus a tail candidate.
seed_bon already failed that confirm.

## Paper residue (honest)

Prefix-match test-time search on Wan 1.3B V2V:

- **is** identity control (subject +0.039 at 32, IQ −0.77);
- **is not** a dynamics method (Dyn 0, tail −8.8%);
- **does** recover notta collapse when it happens (fat right tail of
  Δ, 0000/0007/0018);
- **does** damp videos that were already alive (0/7 hot).

That is a mechanism paragraph, not a leaderboard row. If the paper
needs a motion actuator, the next *score* has to make motion a
constraint. If the paper needs an identity story, we already have the
N=32 VBench number — do not keep searching seeds to chase Dyn.
