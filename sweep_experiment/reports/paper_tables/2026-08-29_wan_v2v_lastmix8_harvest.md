# lastmix N=8 harvest (2026-08-29)

**Series:** `v2v_panda_caption_denoise_8v` (+ smoke)
**Jobs:** smoke 16505827–832; N=8 16505833–837. All generate **0:0**.
**Prompt:** `metadata_csv`. Cite vs caption SF first 8 of
`v2v_panda_caption_32v/notta` (subject 0.700 / IQ 71.54 / tail 0.0129).
**Do not retune** 1.5×. **No TTC. No I2V.**

Caption-128 VBench resubmit **16545806** (score-only). Do not
regenerate 128.

## Call

| Method | tail vs SF | subject | IQ | Dyn med | flicker | Call |
|---|---:|---:|---:|---:|---:|---|
| Self Forcing (paired 8) | 0.0129 | 0.700 | 71.54 | 0 | 0.989 | baseline |
| `sf_lastmix` | 0.0136 (**+5%**) | **0.629** | **69.63** | 0.5 | 0.987 | **NO** |
| `sf_lastmix_always` | 0.0136 (**+5%**) | **0.629** | **69.63** | 0.5 | 0.987 | **NO.** ≈ gated |
| `rf_lastmix` | 0.0138 | **0.657** | **65.53** | 0 | 0.986 | **NO** |
| `rf_lastmix_always` | 0.0138 | **0.657** | **65.53** | 0 | 0.986 | **NO.** bit-match RF gated |

Locked bars: IQ ≥ 70.54, subject ≥ 0.680. All four fail both.
Analyzer `FAIL (motion win, quality collapse)` is the right letter.

Per-video SF: gated ≡ always on 0000–0006. Only **0007** splits
(0.0264 vs 0.0204). Tail up on 3/8 is not a motion method.

## What this means

Appear punch **does fire** (clip 007 last blocks `mix=True`). Mixing
the last DMD step with step-3 **hurts identity** more than it helps
motion. Always-on does not save it. RF span mix is the same collapse,
worse IQ.

Do **not** scale lastmix. Do **not** loosen 1.5× on this 8.
`WAVE=bpseudo` / `WAVE=restep` are different hooks, not implied NO —
do not paste them on top of 16545806. Harvest that VBench first.

Intra SF stays **OOM**. Do not relaunch.
