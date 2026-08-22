# Forward N=32 verdict (2026-08-22 11:40)

Jobs: **16179112** rolling_notta 27m 0:0, 32 mp4;
**16179113** appear_bon 3h09 0:0, 32 mp4;
**16179114** VBench 49m 0:0.

Paired **N=32** vs confirm notta (sidecars, not summary stubs).
Cite medians. Same bars as live_bon-32.

| Method | tail median | vs notta | mean Δ | win/loss | Subj | IQ | Dyn | Call |
|---|---:|---:|---:|---|---:|---:|---:|---|
| notta | 0.0135 | — | — | — | 0.6652 | 69.65 | 0 | baseline |
| **rolling_notta** | **0.0178** | **+31%** | **+13%** | **21/11** | 0.7018 | 70.44 | 0 | **YES on locked bars** |
| appear_bon | 0.0140 | +3% | **−2%** | 15/17 | 0.7298 | 69.06 | 0 | **NO** |

## appear_bon — NO

12/32 bit-match seed_bon. Mean tail down. More losses than wins.
Subject **+0.065** (stronger identity bump than seed_bon-32’s +0.039).
IQ −0.59 holds the −1.0 bar. Dyn 0. This is seed_bon with a
different pick, not a motion method. **Close.**

## rolling_notta — YES on the bars we locked

Tail median **and** mean up. 21/11. Quality **up** (IQ +0.79,
subject +0.037, Aes +0.024). Stills are not mass-damped (15/6
wins on prefix `< 0.012`). 0/32 bit-match seed — different object
(the host).

Dyn is still **0/0**. This is not a VBench dynamic-degree method.
It is the first honest N=32 **tail-motion** win on this V2V
protocol.

Real losses to not chase away: **0004** 0.031→0.010 (still prefix,
notta had invented motion); **0027** 0.035→0.018 (live-hot damp).
Recoveries include 0007 0.012→0.022 and several stills where the
RF student keeps more motion than SF notta (0001, 0002, 0005).

Paper read: **the student/host matters**; prefix-match search did
not. Do not rename this as our controller. It is Rolling Forcing
notta on the locked V2V protocol.

## Locked bars (pre-registered)

| Clause | rolling | appear |
|---|---|---|
| Paired tail > notta | **Yes** (median +31%, mean +13%) | No (mean −2%, 15/17) |
| Stills not mass-damped | **Yes** (15/6) | No (10/11) |
| IQ ≥ notta−1, subject ≥ notta−0.02 | **Yes** (both up) | IQ hold; subject +0.065 is damper, not a fail of the inequality |

Dyn 0/0 does not decide (notta already 0).
