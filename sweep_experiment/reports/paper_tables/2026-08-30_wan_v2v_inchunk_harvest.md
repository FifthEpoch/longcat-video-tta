# In-chunk harvest (2026-08-30)

**Jobs:** intra 16546045–050; bpseudo 051–058; restep 059–068.
**Cite vs caption SF first 8** of `v2v_panda_caption_32v/notta`
(subject 0.700 / IQ 71.54 / tail 0.0129).
Bars: IQ ≥ 70.54, subject ≥ 0.680. Do not retune 1.5×.

## Call

| Method | Generate | subject | IQ | tail vs SF | Call |
|---|---|---:|---:|---:|---|
| `sf_intra` / always | **FAILED 2:0**, 0 mp4 (OOM-fix resubmit) | — | — | — | **DEAD.** Still no video |
| `rf_intra` / always | old 8/8 | 0.645 | 66.33 | +31% | **NO** (prior) |
| `sf_lastmix` / always | 8/8 | **0.629** | **69.63** | +5% | **NO** |
| `sf_bpseudo` / always | 8/8 | **0.628** | **66.60** | +8% | **NO.** Gated ≡ always |
| `rf_bpseudo` | **FAILED 2:0** ~5 min, 0 mp4 | — | — | — | **DEAD** |
| `sf_restep` / always | **FAILED 2:0**, 0 mp4 | — | — | — | **DEAD** |
| `rf_restep` / always | 8/8 | **0.654** | **66.69** | +22% | **NO.** Gated ≡ always |

Every in-chunk hook that wrote a video **failed the letter**.
Every SF intra / SF restep / RF bpseudo job **crashed**.

Punch-or-rewrite inside the 4-step DMD is an identity tax, not
a motion method. Do not scale any of these. Do not loosen 1.5×.

## OOM tails (2026-08-30 14:15)

All four crashes are H200 allocator OOM (140 GB card, <2 GB free).
Not a Python TypeError. One-live-snap still held three GPU KV
copies. CPU-offload fix: `2026-08-30_wan_v2v_oom_cpu_snap.md`.
Smoke only. Paper method stays Pseudo-future Search.
