# Caption official VBench outcomes (2026-08-25)

Full clip, n=32, `metadata_csv`. Cite medians. Self Forcing and
Self Forcing always-search sit first so the other rows compare to
those two. Do **not** mix with stem-prompt 0.665 / 69.65 / Dyn 0.50.

| Method | Subject consistency | Imaging quality | Dynamic degree | Temporal flickering |
|---|---:|---:|---:|---:|
| Self Forcing | **0.700** | **71.54** | 0 | 0.989 |
| Self Forcing always-search | 0.687 | 71.16 | 0 | 0.984 |
| Rolling Forcing | 0.694 | 70.22 | 0 | 0.985 |
| Prefix-match | 0.746 | 70.54 | 0 | 0.990 |
| Rewind | 0.698 | 70.89 | 0 | 0.988 |
| Sick-search | 0.697 | 71.54 | 0 | 0.988 |
| Pseudo | 0.701 | 71.66 | **0** | 0.985 |
| Sink | 0.672 | 70.89 | 0 | 0.982 |

Caption Pseudo Dyn is **0**. Stem 0.50 did not copy.
