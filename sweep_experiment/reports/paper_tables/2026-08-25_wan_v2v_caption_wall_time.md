# Caption N=32 wall time (2026-08-25)

Sidecar `seconds` per video. Cite **medians**. notta mean 196 s is
two outliers (panda_0002 870 s, panda_0019 989 s).

| Method | n | median s | mean s | vs SF median |
|---|---:|---:|---:|---|
| Self Forcing | 32 | **113.1** | 196.1 | — |
| Rolling Forcing | 32 | 44.6 | 44.7 | 0.39× |
| Rewind (SF) | 32 | 127.3 | 130.5 | 1.1× |
| Sick-search (SF) | 32 | 186.4 | 177.0 | 1.6× |
| Pseudo (SF) | 32 | **357.0** | 303.6 | 3.2× |
| Always-search (SF) | 32 | **348.1** | 348.1 | 3.1× |
| Sink (SF) | 32 | 100.4 | 100.3 | 0.89× |
| rf_always / rewind | 32 | 81.5 / 78.5 | 81.6 | ~0.7× |
| rf_sick / rf_pseudo | 32 | 67.5 / 68.7 | 68–74 | ~0.6× |
| rf_sink | 32 | 62.9 | 62.9 | 0.56× |
| Prefix seed_bon | 32 | 361.1 | 382.5 | 3.2× |
| live_bon | 32 | 125.1 | 205.7 | 1.1× |
| appear_bon | 32 | 430.5 | 474.4 | 3.8× |
| sf_roll / rf_chunk | 32 | 42.7 / 109.6 | 43 / 110 | — |

Always-search is flat ~348 s on every clip (four tries every
chunk). Pseudo’s median is similar because the gate fires on most
videos; the **9 exact-SF tail ties** sit at ~113 s, same as
do-nothing. The hold-out is the only cheap path.

Examples (notta / always / pseudo):  
`wan_experiment/results/v2v_panda_caption_examples/`
`panda_{0000,0001,0002,0003,0004,0006}__{self_forcing,always_search,pseudo}.mp4`
