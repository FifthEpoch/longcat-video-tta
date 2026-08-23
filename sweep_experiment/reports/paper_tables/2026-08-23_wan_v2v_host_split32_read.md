# Host-split N=32 + 128 VBench (2026-08-23 03:59)

Jobs: **16215197** sf_roll 35m 0:0 32/32;
**16215198** rf_chunk **FAILED** 3m41 2:0, 0 mp4;
**16215199** sf_recache 1h02 0:0 32/32;
**16215200** rf_recache 37m 0:0 32/32;
**16215201** host-split VBench **FAILED** 1h12 (exit 2) — scored
sf_roll / sf_recache / rf_recache; rf_chunk had no videos.
**16209128** 128 VBench **CANCELLED** at 2h06 after notta
`joined.json`; rolling subject_consistency died at 91/128.

Cite medians. Pair is vs confirm notta **and** forward
`rolling_notta` (same 32). Analyzer PROMOTE is vs SF notta only.

## H1 — weights vs sampler

| Method | θ | unroll | tail med | vs SF | vs RF | W/L vs SF | W/L vs RF | Subj | IQ | Dyn | Call |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| notta | SF | chunks | 0.0135 | — | −24% | — | 11/21 | 0.665 | 69.65 | 0 | SF host |
| rolling_notta | RF | window | 0.0178 | +31% | — | 21/11 | — | 0.702 | 70.44 | 0 | RF host |
| **sf_roll** | SF | window | **0.0281** | **+108%** | **+58%** | **28/4** | **27/5** | 0.666 | 70.09 | **1.0** | sampler **live** |
| rf_chunk | RF | chunks | — | — | — | — | — | — | — | — | **FAILED** |

`sf_roll` is **not** a bit-match of notta (0/32). The rolling
window unroll moves Self-Forcing pixels. Locked bars vs **SF
notta** pass (IQ +0.44, subject +0.001). vs **RF host** the tail
wins but subject **0.666 vs 0.702 (−0.036)** fails the −0.02 bar.
Flicker 0.972 / background 0.796 vs notta 0.986 / 0.825.

This is not “RF quality for free.” It is a twitchier SF. Dyn 1.0
at N=32 is new (seed_bon only flipped Dyn at lucky-8). **Do not
scale to 128** until (1) `rf_chunk` lands so we know whether the
ckpt still wins without the window, and (2) a few mp4s are watched
(living motion vs grain).

## H4 — VAE recache

| Method | vs own host tail | W/L | IQ vs host | Subj vs host | Dyn | Call |
|---|---:|---|---:|---:|---:|---|
| sf_recache | **−1.1%** vs notta | 15/17 | +0.34 | −0.001 | 0 | **NO** (no-op) |
| rf_recache | **+6.6%** vs rolling | **30/2** | −0.52 | −0.003 | 0 | **HOLD** |

`sf_recache` does not move the SF tail. `rf_recache` lifts almost
every clip a little and keeps Dyn 0 — that is VAE-roundtrip grain,
not a living-motion method. Do not scale. IQ/subject hold vs host
(not prefix_sink).

H2/H3 stay **NO** (yesterday, N=128).

## 128 quality (incomplete)

| Method | N | tail med | Subj | IQ | Dyn |
|---|---:|---:|---:|---:|---:|
| notta | 128 | 0.0136 | **0.648** | **70.20** | 0 |
| rolling_notta | 128 | 0.0177 (+31%) | **missing** | **missing** | — |

Do **not** call 128 YES. Resubmit VBench on `rolling_notta` only
(notta `joined.json` already exists; score script skips it).

## Next (cluster)

1. Dump `rf_chunk` traceback (32 error sidecars, 3m41).
2. Resubmit 128 VBench for `rolling_notta` only, 12 h wall.
3. Do not resubmit host-split VBench until `rf_chunk` has mp4s.
4. Do not scale `sf_roll` or `rf_recache`. No TTC. No I2V scale-up.
