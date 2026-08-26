# Caption official Dynamic Degree — percent of clips (2026-08-25)

VBench authors: each clip is 0/1 (RAFT). The official model score
is the **percent of clips labeled dynamic** (`population.mean`),
not the median. Every caption N=32 row below had **median 0**
except crossed host (median 1). Cite the percent.

Source: `vbench_full/joined.json` `population.dynamic_degree`.
Prompt = `metadata_csv`.

| Method | n dynamic | Dyn% | median |
|---|---:|---:|---:|
| Self Forcing | 7/32 | **21.9%** | 0 |
| Self Forcing always-search | 14/32 | **43.8%** | 0 |
| Rolling Forcing | 6/32 | 18.8% | 0 |
| Prefix-match (seed) | 7/32 | 21.9% | 0 |
| live_bon | 10/32 | 31.2% | 0 |
| appear_bon | 8/32 | 25.0% | 0 |
| Rewind (SF) | 9/32 | 28.1% | 0 |
| Sick-search (SF) | 9/32 | 28.1% | 0 |
| Pseudo (SF) | 13/32 | **40.6%** | 0 |
| Sink (SF) | 8/32 | 25.0% | 0 |
| rf_always_search | 7/32 | 21.9% | 0 |
| rf_rewind / sick / pseudo | 8/32 | 25.0% | 0 |
| rf_sink | 5/32 | **15.6%** | 0 |
| sf_roll | 19/32 | 59.4% | 1 |
| rf_chunk | 24/32 | 75.0% | 1 |

Median 0 hid Always-search 14/32 and Pseudo 13/32 vs Self Forcing
7/32. Do not cite caption Pseudo as Dyn 0. RF sink is *below* both
hosts. Crossed host is the only family with median 1.
