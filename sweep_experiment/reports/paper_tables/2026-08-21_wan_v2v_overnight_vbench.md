# V2V overnight read — N=32 VBench + quiet_bon + tail_hist

**Date:** 2026-08-21
**Jobs:** VBench 16122823 (53 min) / 16122824 (13 min); quiet_bon 16124386
(2h29); tail_hist 16124387 (15 min). All COMPLETED 0:0.
**Cite medians. Official quality = full-clip VBench.**

Confirm `summary.json` still skip-stubs notta n=12 / seed n=23. Generate
motion for those two methods stays the **sidecar pair**
(`2026-08-21_wan_v2v_confirm_32_sidecar.md`). quiet_bon and tail_hist
summaries are complete (32/8).

## 1. N=32 VBench — seed_bon is an identity damper, not a Dyn method

| Method | Subj | BG | Aes | IQ | Smooth | Dyn med | Flicker |
|---|---:|---:|---:|---:|---:|---:|---:|
| notta | 0.6652 | 0.8248 | 0.5068 | 69.65 | 0.9923 | **0.00** | 0.9863 |
| seed_bon | **0.7045** | 0.8188 | 0.5021 | 68.88 | 0.9928 | **0.00** | 0.9879 |
| Δ | **+0.039** | −0.006 | −0.005 | −0.77 | +0.001 | 0 | +0.002 |

N=8 Dyn 0→0.5 **died**. Both medians 0 at N=32.

IQ −0.77 is under the ≥1.0 fail bar. Subject **rises**. Combined with
sidecar tail motion **−8.8% (12/32)**: four-seed prefix-match at this
scale **preserves identity by damping motion**. Same failure mode as
I2V-from-still, now measured on a moving prefix.

Locked promote rule: motion must beat notta. **FAIL.** Do not write the
paper around seed-BoN / Dyn.

## 2. N=8 tricks VBench — hist_drop and hinge pass the N=8 bars

vs bake-off notta (already scored):

| Method | tail motion | Subj | IQ | Dyn | vs notta IQ | vs notta Subj | N=8 rule |
|---|---:|---:|---:|---:|---:|---:|---|
| notta | 0.01675 | 0.5951 | 67.98 | 0.00 | — | — | baseline |
| seed_bon | +34% | 0.5956 | 67.38 | 0.50 | −0.60 | +0.0005 | passed; **killed at N=32** |
| **hist_drop** | **+42%** | 0.5961 | 67.83 | 0.50 | **−0.15** | +0.001 | **PASS** |
| **hinge_bon** | +11% | 0.6010 | **68.40** | 0.50 | **+0.42** | +0.006 | **PASS** |

hist_drop is a slightly cleaner N=8 seed_bon (better IQ, same Dyn).
hinge is even cleaner on IQ. **Neither is a new dynamics story** after
§1: N=8 Dyn 0.5 is the same coin-flip that went to 0 at N=32.

**Do not scale hist_drop or hinge to 32.** The sibling method already
failed that confirm.

## 3. tail_hist — short history is not why hist_drop won

| | tail motion | vs notta | vs hist_drop |
|---|---:|---:|---:|
| tail_hist | 0.01689 | **+0.8%**, 6/8 | −29%, 2/8 |

Always last-3 latents, no search, ≈ notta. hist_drop’s +42% was **search
with an extra tail candidate**, not “drop history.”

Per-video: tail_hist **unsticks 0002** (hot prefix hist_drop lost:
0.02350→0.03827) and **erases** the big seed lifts on 0000/0006/0007.
Different mode. Do not VBench it unless we chase 0002-style twitch
(likely flicker).

## 4. quiet_bon — gate works, method still loses

quiet_bon summary **n=32 complete**, tail median **0.01089** vs notta
sidecar **0.01353 (−19%)**. FAIL vs notta. No VBench.

Gate is live: on 0022/0027/0028 quiet **bit-matches notta** (skipped
search on hot prefixes). 0025 still matched seed (prefix was probably
under 0.018 even though the notta *tail* was hot). vs seed, 16/23
exact ties (searched the same). Recovering three hots to notta does not
offset damping on the rest.

## 5. What is closed / what is not

Closed:
- seed_bon as a Dyn / motion method (N=32 generate + VBench)
- quiet_bon as a salvage of seed_bon
- tail_hist as the hist_drop mechanism
- shift/CFG, motion_bon, dead-tail / good backtrack, replay-sink (earlier)

Still true, discovery-only:
- N=8 hist_drop / hinge pass IQ+subject and raise Dyn median on *that*
  lucky 8. Same 8 where seed_bon was +34%. Not a field claim.

Paper-facing residue: **prefix-match search at 32 s raises subject
(+0.039) and kills extra motion.** That is a real measurement. It is
not a method we ship.

## 6. Next (do not submit tonight unless you want one cheap shot)

No hist_drop-32. No quiet_bon VBench. No TTC.

Optional later, only if we still want a sampling-space actuator:
**`hot_tail`** — k=1 short history **only** when prefix_motion ≥ 0.018,
else notta. Tests whether tail_hist’s 0002 lift generalizes to the 0/7
hot set without re-running search on the other 25. N=32, ~notta wall.
Not licensed until we decide the paper still wants a motion method.

Default: **stop V2V generate.** Write the N=32 VBench table as the
confirm. Revisit after the Monday recap.
