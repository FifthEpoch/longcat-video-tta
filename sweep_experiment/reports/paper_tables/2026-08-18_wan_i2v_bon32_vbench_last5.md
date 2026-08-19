# Wan I2V official VBench — `i2v_bon_32v_hybrid` / last5

**Source:** job 15984561 (always + gated) + 15959601 (notta). User paste
of `analyze_i2v_vbench.py --clip last5` on 2026-08-18 19:51.
Paired videos: **32**. VBench higher is better. Handcrafted last-chunk
lower is better. Locked read: `2026-08-18_wan_i2v_bon32_vbench_read.md`.

## Population (median / mean)

| Dimension | do-nothing | always-search | gated-search | gated−always (med) |
|---|---:|---:|---:|---:|
| subject_consistency | 0.969 / 0.965 | 0.967 / 0.942 | 0.969 / 0.948 | 0.002 |
| background_consistency | 0.957 / 0.952 | 0.952 / 0.944 | 0.952 / 0.942 | -0.001 |
| aesthetic_quality | 0.535 / 0.536 | 0.548 / 0.546 | 0.522 / 0.528 | -0.026 |
| imaging_quality | 68.169 / 65.741 | 66.426 / 65.149 | 66.108 / 64.894 | -0.317 |
| motion_smoothness | 0.992 / 0.990 | 0.991 / 0.969 | 0.991 / 0.969 | 0.000 |
| dynamic_degree | 0.000 / 0.188 | 0.000 / 0.250 | 0.000 / 0.188 | 0.000 |
| temporal_flickering | 0.989 / 0.986 | 0.988 / 0.972 | 0.988 / 0.972 | -0.000 |

## Gated vs always (per-video, higher VBench wins)

| Dimension | gated>always | tie | gated<always |
|---|---:|---:|---:|
| subject_consistency | 14 | 10 | 8 |
| background_consistency | 10 | 10 | 12 |
| aesthetic_quality | 8 | 10 | 14 |
| imaging_quality | 9 | 10 | 13 |
| motion_smoothness | 12 | 8 | 12 |
| dynamic_degree | 0 | 30 | 2 |
| temporal_flickering | 11 | 10 | 11 |

## Does the handcrafted last-chunk score track VBench?

Spearman rho(last-chunk, VBench dim). Expected sign if the verifier is a useful quality proxy: **negative**.

| Dimension | do-nothing ρ | always-search ρ | gated-search ρ |
|---|---:|---:|---:|
| subject_consistency | 0.177 | 0.137 | -0.049 |
| background_consistency | 0.100 | 0.172 | -0.173 |
| aesthetic_quality | -0.072 | 0.082 | 0.027 |
| imaging_quality | 0.229 | 0.243 | 0.327 |
| motion_smoothness | -0.037 | -0.263 | -0.295 |
| dynamic_degree | 0.017 | -0.289 | 0.069 |
| temporal_flickering | 0.021 | -0.131 | -0.201 |
