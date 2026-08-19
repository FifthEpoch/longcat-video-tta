# Wan I2V official VBench — `i2v_bon_32v_hybrid` / full

**Source:** job 15984561 (always + gated) + 15959601 (notta). User paste
of `analyze_i2v_vbench.py --clip full` on 2026-08-18 19:51.
Paired videos: **32**. VBench higher is better. Full 30 s is diluted by
the shared piece-0 prefix. Cite `last5` for outcomes.
Locked read: `2026-08-18_wan_i2v_bon32_vbench_read.md`.

## Population (median / mean)

| Dimension | do-nothing | always-search | gated-search | gated−always (med) |
|---|---:|---:|---:|---:|
| subject_consistency | 0.848 / 0.851 | 0.855 / 0.851 | 0.851 / 0.845 | -0.004 |
| background_consistency | 0.894 / 0.897 | 0.893 / 0.898 | 0.890 / 0.896 | -0.003 |
| aesthetic_quality | 0.587 / 0.594 | 0.593 / 0.598 | 0.591 / 0.589 | -0.003 |
| imaging_quality | 71.242 / 70.000 | 71.283 / 70.039 | 71.194 / 69.549 | -0.090 |
| motion_smoothness | 0.992 / 0.991 | 0.991 / 0.987 | 0.992 / 0.987 | 0.001 |
| dynamic_degree | 0.000 / 0.281 | 0.000 / 0.281 | 0.000 / 0.250 | 0.000 |
| temporal_flickering | 0.985 / 0.983 | 0.984 / 0.979 | 0.986 / 0.980 | 0.002 |

## Gated vs always (per-video, higher VBench wins)

| Dimension | gated>always | tie | gated<always |
|---|---:|---:|---:|
| subject_consistency | 10 | 10 | 12 |
| background_consistency | 12 | 10 | 10 |
| aesthetic_quality | 6 | 10 | 16 |
| imaging_quality | 11 | 10 | 11 |
| motion_smoothness | 11 | 10 | 11 |
| dynamic_degree | 0 | 31 | 1 |
| temporal_flickering | 9 | 10 | 13 |

## Does the handcrafted last-chunk score track VBench?

Spearman rho(last-chunk, VBench dim). Expected sign if the verifier is a useful quality proxy: **negative**.

| Dimension | do-nothing ρ | always-search ρ | gated-search ρ |
|---|---:|---:|---:|
| subject_consistency | -0.095 | -0.212 | 0.166 |
| background_consistency | 0.283 | 0.257 | 0.360 |
| aesthetic_quality | -0.094 | -0.057 | -0.041 |
| imaging_quality | 0.036 | 0.060 | 0.304 |
| motion_smoothness | 0.122 | -0.041 | -0.064 |
| dynamic_degree | 0.034 | -0.109 | -0.219 |
| temporal_flickering | -0.000 | -0.037 | -0.012 |
