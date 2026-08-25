# Slide method × caption coverage (2026-08-25 02:55)

Every named method on `wan-methods-since-switch`. Caption =
`metadata_csv`. Stem = filename prompt audit. Do not mix.

| Slide method | Caption metrics on slide? | Caption N=32 job? | Action |
|---|---|---|---|
| SF do-nothing | Yes (VBench 0.700/71.54 provisional) | **16310318** done; VBench **16310330** R | Wait 330 |
| RF do-nothing | Yes (0.694/70.22 provisional) | **16310319** done; 330 R | Wait 330 |
| Rewind SF | Yes (0.698/70.89 provisional) | **16310320** done; 330 R | Wait 330 |
| Rewind RF | Tails yes; VBench in 330 | **16310326** done | Wait 330 |
| Sick SF / RF | Tails yes; VBench in 330 | **16310321 / 327** done | Wait 330 |
| Pseudo SF / RF | Tails yes; VBench in 330 | **16310322 / 328** done | Wait 330 |
| Sink SF / RF | Tails yes; VBench in 330 | **16310323 / 329** done | Wait 330 |
| Always SF / RF | Tails yes (+39% / +25%) | **16310324 / 325** done; 330 R | Wait 330 |
| Prefix Always (`seed_bon`) | Stem only on card | **16328464** PD | Overnight |
| Prefix live (`live_bon`) | Stem only on card | **16328465** PD | Overnight |
| Prefix appear (`appear_bon`) | Stem only on card | **16328466** PD | Overnight |
| Prefix VBench | — | **16328467** afterok | Overnight |
| AdaSteer fixed/stream/resid | **Yes** N=8 caption (NO) | **16326033–036** done | Do not scale |
| Crossed host (`sf_roll` / `rf_chunk`) | No | **not yet** | Submit `WAVE=cross` |
| LongCat AdaSteer | LongCat 13.6B (other stack) | — | Not Wan caption |
| I2V sharp / official tie | I2V-from-still | — | Closed. No relaunch |

Not on the slide as a method to score: CachedSearch, History
Guidance, TTC, quiet_bon, WAVE=3 discovery, N=128 hosts.

## Do not

Resubmit WAVE=1. Scale AdaSteer. Dump full WAVE=2 (quiet +
recache). Scancel 16310330 or 16288113–115.
