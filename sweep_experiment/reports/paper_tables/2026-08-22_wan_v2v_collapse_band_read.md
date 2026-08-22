# Collapse+band resim read (2026-08-22 00:47)

**No GPU.** Locked earn rule: 0002/0003 stay cand0, 0007-class
recover, **0022 / 0027 / 0028 stay cand0**.

| Clip | Rule | Result |
|---|---|---|
| 0002 / 0003 | stay cand0 | **Pass.** `still_prefix`. seed_bon ≠orig 4–6 (we undo those damps). |
| 0007 / 0001 / 0026 / 0010 / 0014 | recover | **Pass.** `nearest_prefix`, rec>0. |
| 0027 ch0 | stay cand0 | **Pass.** `no_collapse` (c0 0.053 > 0.7×prefix 0.048). Orig seed picked 3. |
| 0028 ch0 | stay cand0 | **Pass.** `no_collapse`. Orig already 0. |
| **0022 ch0** | stay cand0 | **Fail.** `band_appear` cand1, m=0.061 vs c0 0.036, prefix 0.068. Same pick as seed_bon. |

`damp=0` on all 48+192+192 chunks: when the policy fires it never
picks below cand0. Stills never match a hold. 0006 ch0 is
`no_collapse` (prefix ≈ notta) — we miss that N=8 lift.

## 0022

This is a **real chunk-0 collapse** (0.036 vs prefix 0.068), not a
hot-notta. The band `[0.058, 0.078]` contains cand1 at 0.061, so the
policy is supposed to take it. The 30 s seed tail still lost
(0.0268 vs notta 0.0358) after that commit. Resim cannot say whether
a full regenerate would hold 0.061.

The locked bar said stay cand0. It did not. **Do not generate.**
Do not retcon the bar because the band pick “looks right.”

## Call

Collapse+band does **not** earn a GPU. Rolling **16179112** and
appear **16179113** stay the only in-flight tests. This picker
family is closed unless a later generate is pre-registered on a
*new* rule, not this one.
