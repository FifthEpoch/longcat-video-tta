# Mixed N=8 diagnose (2026-08-21 23:12)

**Ignore** `pearson(d_tail, d_flicker)=-1.000` on every method. VBench
`temporal_flickering` is tracking our tail `|Δframe|` (d_flick ≈ −d_tail).
That line is tautological. It does **not** mean every gain is junk.

Read **IQ / Dyn / subject / win-loss** instead.

Jobs submitted after this paste: **16179112** `rolling_notta` N=32,
**16179113** `appear_bon` N=32, VBench **16179114**.

## Closed — do not try to “fix the fail dim”

### `longlive_prefix_sink`

7/8 tail wins, Dyn 0→1 on three clips. **IQ drop >1.0 on 5/8**
(0002 −5.4, 0003 −6.3, 0004 −3.7, 0006 −2.0, 0007 −1.6). 0007
+0.039 tail with Dyn still 0 — motion without dynamic degree.
The IQ fail is the **majority**, not one bad clip. **Do not
investigate an IQ fix.** Close.

### `longlive_notta`

4/4 tail, median Δ ≈ 0. IQ up on some clips, but Dyn **1→0** on
0003/0004/0006 (every clip that had Dyn). Subject −0.12 / −0.15
on 0005/0007. Quality is identity, not a Dyn fix. **Close.**

### `live_hist` Dyn 0.5

3 tail wins, 5 exact notta (skips). The N=8 Dyn 0.5 is **0007
alone** (0→1) at IQ **−7.7** and subject **−0.25**. Not a
meaningful Dyn gain. **Close.** Same family as live_bon-32 NO.

## Already in flight at N=32 — do not scancel

### `rolling_notta` (16179112)

5/3 tail, median Δ +0.0066. IQ drop>1 on **1/8** (0002 −1.55).
Large IQ *gains* on 0000/0001/0005/0007. Dyn flips both ways
(three 0→1, two 1→0). 0004 is a real tail loss (−0.020) with
subject +0.09. This is still the only host leftover. N=32 is
the test. Same bars as live_bon-32.

### `appear_bon` (16179113)

4/4 tail, **median Δ −0.001** (delta-of-median was +7%; do not
cite that). **4/8 bit-match seed_bon** (incl. 0000 +0.01275).
0004 −0.013 tail and IQ −3.1. Dyn 0→1 is 0000 — the lucky clip.
Looks like a weaker seed_bon. Leave the N=32 job; it is the
kill test. Optional: `scancel 16179113` if you want the 2-way
cap for rolling only.

## Not a flicker-junk close

Because flickering is not an independent sensor here, we do
**not** close rolling or appear from that pearson. IQ and
bit-match-to-seed are the independent reads.
