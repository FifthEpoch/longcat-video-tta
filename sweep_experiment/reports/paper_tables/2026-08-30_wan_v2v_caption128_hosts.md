# Caption V2V N=128 hosts (2026-08-30)

**Series:** `v2v_panda_caption_128v`
**Jobs:** generate 16506077 / 078; VBench **16545806** COMPLETED 0:0 33 min.
**256 mp4.** First 32 hardlinked from caption-32. Prompts `metadata_csv`.
**Cite this 128-row Self Forcing**, not the N=32 subject 0.700.

## Official full-clip (medians)

| Method | tail | subject | IQ | Aes | Dyn med | flicker |
|---|---:|---:|---:|---:|---:|---:|
| Self Forcing | 0.0119 | 0.666 | **72.07** | 0.499 | 0 | 0.988 |
| Rolling Forcing | **0.0158** (**+33%**) | **0.685** | 71.52 | 0.529 | 0 | 0.983 |

Analyzer **PROMOTE** rolling vs this 128 SF: tail up, subject up,
IQ −0.55 (inside the 1.0 band). Flicker −0.005. Host, not ours.

N=32 SF subject 0.700 does **not** copy to N=128 (0.666). Do not
mix those tables. Dyn official is **percent of clips**, not this
median 0 — backfill Dyn% before the paper host row.

## Not launched

`WAVE=cite`: `sf_pseudo` + `sf_always_search` on the same 128
(reuse first 32). That is the paper-size Pseudo table.
