# Lineage + ideas N=8, all generate+VBench done (2026-08-21 21:42)

Not the yes/no. That is live_bon-32 = **NO**.
These N=8 rows are discovery. Do not scale from this table.

Jobs: lineage 16140808–816 all COMPLETED 0:0. Ideas 16145125–131
all COMPLETED 0:0. Every method dir: 8 mp4 + summary + vbench_full.

Baseline: bake-off notta (tail 0.0167, subject 0.5951, IQ 67.98,
Dyn 0).

## Lineage (`v2v_panda_lineage_8v`)

| Method | tail | vs notta | Subj | IQ | Dyn | Honest |
|---|---:|---:|---:|---:|---:|---|
| live_bon | 0.0229 | +37% | 0.5944 | 67.98 | 0 | N=8 only. Killed at 32. |
| live_hist | 0.0229 | +37% | 0.5951 | 67.94 | **0.50** | Same skip set. Do not scale. |
| longlive_notta | 0.0150 | −10% | 0.6306 | **69.93** | 0 | HOLD motion. Quality ↑. |
| longlive_sink | 0.0150 | −10% | 0.6306 | 69.93 | 0 | **Bit-match notta host.** Sink no-op. |
| longlive_live_bon | 0.0166 | −1% | 0.6291 | 70.09 | 0 | HOLD |
| longlive_prefix_sink | 0.0307 | +84% | 0.5974 | **66.01** | **1.00** | FAIL IQ −2.0. Flicker/motion junk. |
| rolling_notta | 0.0215 | +29% | 0.6300 | 68.68 | **0.50** | Best *host* N=8. Not our controller. Do not scale tonight. |

## Ideas (`v2v_panda_ideas_8v`)

| Method | tail | vs notta | Subj | IQ | Dyn | Honest |
|---|---:|---:|---:|---:|---:|---|
| appear_bon | 0.0179 | +7% | 0.6010 | 68.40 | 0.50 | Lucky 8. Same VBench as noise_bon / pseudo_appear. |
| live_appear | 0.0186 | +11% | 0.5951 | 67.98 | 0 | Weaker than live_bon-8. Last-chunk = notta. |
| pseudo_gate | 0.0185 | +11% | 0.5956 | 67.38 | 0.50 | Last-chunk = seed_bon (53.03). Gate fired. |
| pseudo_appear | 0.0179 | +7% | 0.6010 | 68.40 | 0.50 | = appear_bon on VBench. |
| noise_probe | 0.0167 | 0 | 0.5951 | 67.98 | 0 | Exact notta. Probe only. |
| noise_bon | 0.0179 | +7% | 0.6010 | 68.40 | 0.50 | = appear_bon. τ fired enough to become appear. |

N=8 idea “PROMOTE”s are the same lucky-8 trap as seed_bon-8.
**Do not submit any of these at N=32.** live_bon-32 already answered.
