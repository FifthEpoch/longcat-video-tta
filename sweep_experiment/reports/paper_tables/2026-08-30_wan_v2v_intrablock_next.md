# What the mid-chunk failures teach (2026-08-30)

This is a thinking note, not a submit. Do not launch these tonight.
The GPU paste is the crashed N=8 plus Pseudo / Always on 128.

## What we already learned

Each 5-second chunk is written in **four** denoising steps, in tiny
3-frame blocks. We tried to “unstick” a block while it was still
being written.

| What we did | What happened |
|---|---|
| Average the last two steps 50/50 | The face / scene changed. Image quality dropped. |
| Throw the last tiny block away and draw a new seed | Same: it is a different 0.7 s, so identity moves. |
| Redo the last **two** of the four steps | Half the generation is new. Same identity hit. |
| Fire when the picture looks too sharp / colorful | The “fix” ran almost as often as always-on. Sharpness is a bad sick signal on this model. |
| Watch every block and keep four full GPU copies of memory | Crashed. Memory issue, not a quality result. |

The shared mistake: **we replaced the picture**. On a 4-step model,
“a little rewrite” is not little.

CFG and shift do **not** change pixels on this student. Do not use
those as the next intra hook.

## Four next ideas (keep the current picture)

**1. Nudge, do not replace.**  
Same last-step mix as before, but 10% new / 90% original, not 50/50.
Question: was the identity hit just the mix being too strong?

**2. Motion-only trigger.**  
Only intervene if this block has less motion than the previous block
(0.8×, already pre-registered). Never fire on sharpness or color.
Question: was the trigger the problem, not the rewrite?

**3. Do not touch this block. Steer the next one.**  
If a block looks frozen, leave its pixels alone. Use a different seed
only for the **following** block. Question: can we add motion without
editing something the viewer already “has”?

**4. Keep the look, borrow only the wiggle.**  
Write the default block. Write one extra seed. Keep the default
picture, add a small slice of (extra − default) — the part that
moved, not the part that changed who is in the frame.
Question: can motion come from a residual instead of a new image?

Idea 1 is the cheapest test of “we over-edited.” Idea 3 is the
safest. Idea 4 is the real method candidate if 1–3 still fail.
Idea 2 can wrap any of them.

Always-on twins stay required. Do not retune 1.5×. Do not grow
any of these past 8 videos until they beat Self Forcing on motion
**and** keep subject ≥ 0.68 and image quality ≥ 70.5.
