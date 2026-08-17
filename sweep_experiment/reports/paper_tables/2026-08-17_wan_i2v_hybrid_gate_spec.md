# Hybrid gate spec (2026-08-17)

**Status:** implemented in `run_i2v_chunked.py`; 32v three-way not yet
scored. This file is the lock, not a result table.

**Series:** `i2v_bon_32v_hybrid` (NOTTA | always-BoN k=4 | gated-BoN)
**Submit:** `wan_experiment/sbatch/submit_i2v_bon32_hybrid.sh`
**Analyzer:** `wan_experiment/scripts/analyze_i2v_bon.py`

## Decision rule (`gated_bon`)

Search k candidates (cand0 = NOTTA seed) iff any clause is true:

| Clause | Condition | Default | Why |
|---|---|---|---|
| `ch1` | `chunk == 1` and `incoming > T_ch1` | 0.8 | Catch 05/02/09 (inc 0.87–0.90) while skipping 06 (0.20) and 07 (0.68) |
| `level` | `incoming > T_late` | 2.0 | Same late-drift rule as the 16v T=2.0 run |
| `trend` | `Δincoming > 0.5` and `incoming_prev > 0.5` | 0.5 / 0.5 | Catch 12 (0.53→1.09); do not fire 06 off prev=0.20 |

Otherwise `skip` (k=1). Chunk 0 is always seed 0 (`forced_prefix`).
`always_bon` still searches every searchable chunk (`reason=always`).
`incoming` = last-1s GT-free composite vs first-1s-after-cond, **no seam**.

## Per-step fields (every method, every chunk)

Written to each video `*.json`, `summary.json`, and `gate_trace.jsonl`:

- `incoming_drift`, `incoming_prev`, `incoming_delta`
- `incoming_signals` / `incoming_devs` (sharp, color, contrast, motion, score)
- `outgoing_drift` / `outgoing_devs` (last-1s of the committed prefix after the pick)
- `gate_reason`, `gated_fired`, `search_k`
- `cand0_score`, `chosen_score`, `chosen_minus_cand0`, `chosen_breakdown`
- per-candidate `signals` + `devs` (the verifier loss terms)

Video-level rollups: `incoming_series`, `outgoing_series`, `gate_reasons`,
`last_chunk_score`, `last_chunk_breakdown`.

## Locked read after the 32v lands

- Quality win vs always-on → controller paper.
- Tie + cheaper → efficiency paper (16v T=2.0 already sits here).
- Quality loss → drop the gating claim; do not lower T blindly.
- Do **not** write “gated beats always-on” from the 16v T=2.0 numbers.
- Do **not** add TTC until this BoN path is solid.

## 16v T=2.0 baseline this run is trying to beat

Last-chunk composite, N=16: NOTTA 4.429 / always 3.226 / gated 3.378.
gated−always **+0.152 mean**, −0.131 median, 6/16 better-or-tie.
Hypothesis: hybrid flips that mean to about **−0.2** if 05/02/09/12
fire and 06/07 stay skipped early.
