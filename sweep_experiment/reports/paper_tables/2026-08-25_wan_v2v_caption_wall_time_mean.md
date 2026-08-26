# Caption N=32 wall time — cite the mean (2026-08-25)

Expected cost per video = **mean** sidecar `seconds`. Median is
only useful to see that Self Forcing’s mean is two hung clips
(panda_0002 870 s, panda_0019 989 s), not the typical method cost.

| Method | n | mean s | median s |
|---|---:|---:|---:|
| Self Forcing | 32 | **196.1** | 113.1 |
| Always-search (SF) | 32 | **348.1** | 348.1 |
| Pseudo (SF) | 32 | **303.6** | 357.0 |
| Rolling Forcing | 32 | 44.7 | 44.6 |
| Rewind (SF) | 32 | 130.5 | 127.3 |
| Sick-search (SF) | 32 | 177.0 | 186.4 |
| Sink (SF) | 32 | 100.3 | 100.4 |
| rf_always | 32 | 81.6 | 81.5 |
| rf_sink | 32 | 62.9 | 62.9 |
| seed_bon | 32 | 382.5 | 361.1 |
| live_bon | 32 | 205.7 | 125.1 |
| appear_bon | 32 | 474.4 | 430.5 |

Pseudo is cheaper than Always in the **mean** (304 vs 348). The
median of Pseudo is a gate-fired clip (probe + k=4 ≈ 357 s), so
median makes Pseudo look more expensive than Always. That is why
mean is the cost number.
