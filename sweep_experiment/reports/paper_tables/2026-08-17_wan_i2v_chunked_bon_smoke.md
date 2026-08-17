# Wan I2V chunked BoN smoke — 2026-08-17

**Source:** jobs 15883525 (NOTTA), 15883526 (always-BoN k=4)
`wan_experiment/results/i2v_chunked_smoke/{notta,always_bon}_h30s_shard0/`
**N=2, 30 s, 5 × 24 latents.** Chunk 0 forced to cand0. Search from chunk 1.
**Gating sample, not paper-grade.**

## Infra

| Method | n_ok | Frames | Mean wall (s) | mp4 MB |
|---|---|---|---|---|
| chunked NOTTA | 2/2 | 481 | 83.7 | 30.0 / 18.6 |
| always-BoN k=4 | 2/2 | 481 | 265.7 | 30.2 / 17.6 |

~3.2× wall for 17 vs 5 chunk-generates (expected ~3.4×).

## Search activity (always-BoN)

| Clip | Picks (chunks 0–4) | n_divergent (of 4 searchable) |
|---|---|---|
| 000 bubbles | 0, **3**, 0, **3**, **1** | 3 |
| 001 pot | 0, **1**, **2**, 0, 0 | 2 |

5/8 searchable chunks left cand0. Search is alive.

## NOTTA verifier scores (lower = closer to first-1s ref)

| Clip | ch0 | ch1 | ch2 | ch3 | ch4 |
|---|---|---|---|---|---|
| bubbles | 3.31 | 1.75 | 3.47 | 5.26 | 4.24 |
| pot | 1.77 | 1.20 | 1.84 | 3.18 | 4.81 |

Later chunks get worse. Matches the 16v 30 s drift (sharpen + freeze).

## Do not cite as paired BoN vs NOTTA

Chunk 0 cand0 scores already differ across jobs (bubbles 3.305 vs 2.992).
Cause: `scheduler.add_noise` used unseeded `torch.randn_like`. cand0 is
not a NOTTA twin. RNG is seeded per (cand, chunk) in the follow-up
commit. Resubmit 16v only after that lands. Do not add TTC yet.
