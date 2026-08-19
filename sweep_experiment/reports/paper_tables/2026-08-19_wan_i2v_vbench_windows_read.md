# Read — VBench++ windows and 5 s vs 30 s first-16 (2026-08-19)

**Jobs:** 16009916 (hybrid 32 windows), 16010032 (16v head/tail)
**Tables:** [`2026-08-19_wan_i2v_bon32_vbench_trend.md`](2026-08-19_wan_i2v_bon32_vbench_trend.md),
[`2026-08-19_wan_i2v_notta16_vbench_headtail.md`](2026-08-19_wan_i2v_notta16_vbench_headtail.md)
**Cite medians.** Full-clip hybrid 32 remains the official VBench++ number.
These windows and 16-frame clips are diagnostics.

## What the 5 s windows show (N=32, paired)

Piece 0 is shared: 0–5 s is identical across do-nothing / always / gated
(subject 0.934, IQ 72.87, aes 0.650–0.651, dynamic mean 0.250).

The VBench time trend is **aesthetic and imaging decay**, not a
subject-collapse and not a RAFT unfreeze:

| Dim (do-nothing median) | 0–5 s | 25–30 s |
|---|---:|---:|
| aesthetic_quality | 0.651 | **0.538** |
| imaging_quality | 72.87 | **68.14** |
| subject_consistency | 0.934 | 0.969 |
| dynamic_degree | 0 / 0.250 | 0 / 0.188 |

Search does **not** stop the decay. At 25–30 s, do-nothing wins imaging
(68.14 vs always 66.41 vs gated 66.07). Always is slightly ahead on
aesthetic at the tail (0.545 vs 0.538). Gated is behind on both.
`dynamic_degree` **median is 0 in every window** — most clips already
fail RAFT’s dynamic test in the first 5 s. Means wobble 0.19–0.34; there
is no clean freeze-onset after piece 0.

Motion smoothness medians stay ~0.99. Always/gated **means** drop at
20–30 s (0.977 vs 0.991) — a few clips got worse, the median did not.

## What the 16-frame 5 s vs 30 s table shows (N=16, NOTTA)

Same windows as the handpicked drift table (skip cond frame 0).

**VBench does not copy the handpicked sharp +167% / motion −60% story.**
On 16-frame medians:

| Dim | 5 s Δrel | 30 s Δrel |
|---|---:|---:|
| aesthetic_quality | +0.018 | **−0.115** |
| imaging_quality | +0.015 | −0.023 |
| subject_consistency | +0.043 | +0.038 |
| motion_smoothness | +0.008 | +0.007 |
| temporal_flickering | +0.003 | +0.013 |
| dynamic_degree | — (median 0) | — (median 0) |

The only dim that clearly says “30 s tail is worse” is **aesthetic**
(−11.5% vs +1.8% at 5 s). Imaging is a mild drop. Subject / smoothness /
flicker go **up** at the last second — consistent with a freeze (the
last 16 frames are more self-similar), the opposite sign of handpicked
motion death. Handpicked sharpness *up* is oversharpening; VBench IQ at
the 30 s last-16 is slightly *down*. Do not equate the two.

**Same-duration check:** 5 s full ≈ 30 s first5 (subject −0.001, aes
−0.021, IQ −1.43). The 30 s generate is not a different movie at t=0.

**Unequal-length full clips:** 30 s full subject 0.842 vs 5 s 0.932 is a
length-averaged diagnostic, not a 5 s-matched pair. Do not cite it as
the horizon effect.

## Locked claims

1. Official hybrid-32 VBench++ is still the **full-clip tie**.
2. The window trend is the honest VBench view of long rollout:
   aes/IQ fall after piece 0; search does not reverse it.
3. 16-frame VBench is **not** a replacement for the handpicked drift
   table. Aesthetic is the overlap; freeze shows up in VBench as
   smoothness/flicker/subject *up* and dynamic median 0 from the start.
4. No PSNR. No TTC. I2V-32 scale-up stays closed.
