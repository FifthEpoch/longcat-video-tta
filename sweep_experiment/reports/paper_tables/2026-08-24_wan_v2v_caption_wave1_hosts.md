# Caption WAVE=1 hosts — generate peek (2026-08-24)

Series `v2v_panda_caption_32v`. **Handcrafted tail only.**
Official quality is still pending (`16310330` afterok remaining
generate). Do **not** cite this as VBench++. Do **not** mix these
numbers into stem-prompt tables.

Protocol check **PASS:** every sidecar `prompt_source=metadata_csv`,
`bad_stemish=0`. First clip caption is the truck-hood sentence, not
`panda 0000`.

## Jobs

| Job | Method | State | Wall | n |
|---|---|---|---:|---:|
| **16310318** | notta | COMPLETED 0:0 | 1h47 | 32/32 |
| **16310319** | rolling_notta | COMPLETED 0:0 | 27m | 32/32 |
| **16310320** | sf_rewind | COMPLETED 0:0 | 1h13 | 32/32 |
| **16310321** | sf_sick_search | RUNNING | — | — |

## Caption tails (median, N=32)

| Method | tail | vs caption SF | vs stem same-method |
|---|---:|---:|---:|
| notta (SF, caption) | **0.01164** | — | stem was 0.0135 (−14%) |
| rolling_notta | 0.01423 | **+22%** | stem was 0.0178 |
| sf_rewind | 0.01262 | **+8%** | stem was 0.0143 |

Stem deltas were rolling +31% / rewind +6% vs **stem** notta.
On captions, rolling still wins the host gap; the gap shrank.
Rewind is still a small plus. No win/loss counts in this paste.

## What this does / does not say

- **Does:** WAVE=1 is on the right text. Caption SF notta exists
  for AdaSteer N=8 (first 8 of this dir). Leave 321–329 running.
- **Does not:** IQ / subject / Dyn / flicker. Method HOLD/NO.
  Paper table. WAVE=2 go.

Regenerable: sidecars under
`wan_experiment/results/v2v_panda_caption_32v/{notta,rolling_notta,sf_rewind}_h30s_shard0`.
