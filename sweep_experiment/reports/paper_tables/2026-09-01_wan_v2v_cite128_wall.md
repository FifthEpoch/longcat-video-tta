# Caption 128 mean seconds / clip (2026-09-01)

First 32 of each arm were hardlinked from caption-32. The
generate job only made the remaining **96**. Mean seconds /
clip on the paper-size row is **job elapsed / 96**, not
sidecar `seconds` on n=32 and not job-wall / 128.

| Method | Job | Elapsed | / 96 | n=32 sidecar mean |
|---|---|---:|---:|---:|
| Self Forcing | **16506077** | 02:52:40 | **108** | 196 |
| Rolling Forcing | **16506078** | 01:15:02 | **47** | 45 |
| Pseudo-future Search | **16615748** | 7h51 | **294** | 304 |
| Always-search | **16615749** | 9h26 | **354** | 348 |

Cite **108 / 47 / 294 / 354** on the caption-128 table.
Pseudo vs Always is **17%** cheaper (60 s), same sign as the
n=32 sidecar 13% (304 vs 348). Still **~6.3× Rolling**.

n=32 Self Forcing mean 196 is the two outliers (0002 / 0019).
Those clips are in the hardlink set, so they do not enter the
96. Do not mix the two estimators in one row.
