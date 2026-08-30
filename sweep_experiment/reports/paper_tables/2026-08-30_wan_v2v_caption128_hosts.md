# Caption V2V N=128 hosts (2026-08-30)

**Series:** `v2v_panda_caption_128v`
**Jobs:** generate 16506077 / 078; VBench **16545806** COMPLETED 0:0 33 min.
**256 mp4.** First 32 hardlinked from caption-32. Prompts `metadata_csv`.
**Cite this 128-row Self Forcing**, not the N=32 subject 0.700.

## Official full-clip

| Method | tail | subject | IQ | Aes | Dyn% | flicker |
|---|---:|---:|---:|---:|---:|---:|
| Self Forcing | 0.0119 | 0.666 | **72.07** | 0.499 | **32.8% (42/128)** | 0.988 |
| Rolling Forcing | **0.0158** (**+33%**) | **0.685** | 71.52 | 0.529 | 28.9% (37/128) | 0.983 |

Dyn = `population.dynamic_degree` mean (official). Median is 0
on both — do not cite the median. Analyzer **PROMOTE** was tail
+ subject + IQ band. Official Dyn% **loses** (37 vs 42). Same
sign as N=32 (18.8% vs 21.9%). Host, not ours.

N=32 SF subject 0.700 / Dyn 21.9% does **not** copy to N=128
(0.666 / 32.8%). Do not mix those tables.

## Not launched

`WAVE=cite`: `sf_pseudo` + `sf_always_search` on the same 128
(reuse first 32). That is the paper-size Pseudo table.
