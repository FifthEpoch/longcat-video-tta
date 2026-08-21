# What the V2V bake-off actually said (plain language)

**Date:** 2026-08-20
**N:** 8 Panda videos. First ~2 s real, then ~30 s generated.
**Table:** [`2026-08-20_wan_v2v_bakeoff_8v_vbench.md`](2026-08-20_wan_v2v_bakeoff_8v_vbench.md)

## What we were asking

I2V-from-a-still: four random seeds did **not** unstick a freeze. Gating
when to spend those seeds was a VBench **tie**. So we asked: on a **real
video prefix** (motion already in the condition), can we change the
*sampling trajectory* — not the weights, not the gate?

## What each method did

| Method | In English | Outcome |
|---|---|---|
| notta | Generate once, default seed. Baseline. | Tail already fairly still. Dyn median 0 (most clips fail VBench’s “is this video moving?” bit). |
| seed_bon | Each 5 s chunk: try 4 seeds, keep the one the old drift score likes. | **Only win.** +35% tail motion. VBench Dyn median 0 → 0.5 (4/8 clips now “dynamic”). Subject held. IQ down 0.60 (allowed; bar was 1.0). |
| motion_bon | Same 4 seeds, but keep the chunk that *looks* most like motion (`\|Δframe\|`). | **Lost.** The 30 s tail moved *less*. Picking a twitchy 5 s piece does not make a more dynamic minute. |
| backtrack | If the last second dies, rewind and try again. | **Lost.** Less motion, IQ −2.94. |
| shift / CFG | Change Wan’s flow-shift or guidance. | **Dead.** All 9 settings produced the same pixels. The DMD student ignores those knobs in our loop. |

## Why seed_bon winning is surprising

The pick score is the same composite that was **anti-aligned with IQ** on
I2V-32 and blew up to 6336 on the V2V smoke. We did **not** invent a
clever motion verifier. We just gave the model four rolls of the dice
per chunk, on a prefix that already had motion.

So the honest finding is: **on V2V, seed diversity is a real actuator;
on I2V-from-still, it was not.** We do not yet know *which* seed
property is helping (more motion vs less sharpening vs less of a bad
seam). N=8 Dyn 4-vs-0 is a coin-flip sample.

## What I would do next (one bet)

**Confirm, don’t invent.** Run `notta` vs `seed_bon` only, **N=32**, same
V2V protocol. Drop motion_bon, backtrack, shift, CFG.

- If +35% tail motion and Dyn median stay up, and IQ/subject still hold:
  that *is* the sampling-space result. Then wave 2 is **make seed search
  cheaper** (CachedSearch) or **keep the prefix in KV** (attention sink),
  not another pick-score.
- If the motion win vanishes at N=32: N=8 was noise. Do not write the
  paper around seed-BoN.

Do **not**: scale I2V-32, retune the I2V gate, add TTC, or spend a week
on motion_bon. Optional T2V 128 is still only a compare table, not this
claim.

No job submitted until you say go.
