# Official VBench++ — entire clip, 5 s vs 30 s (`i2v_notta_16v`)

**This is the typical VBench++ number:** score the **full generated
video**, not a 5 s slice of a longer file and not first/last 16 frames.

**Source:** job 16010032, `vbench_full/joined.json` on
`h5s_shard0` and `h30s_shard0`. Same 16 images, seed 0, NOTTA.
Cells are **median / mean**. Higher is better.

Two different generates (5 s one-shot vs 30 s one-shot), not a prefix
of the same file.

| Dimension | Entire 5 s clip (~85 fr) | Entire 30 s clip (~481 fr) | 30 s − 5 s (med) |
|---|---:|---:|---:|
| subject_consistency | 0.932 / 0.928 | 0.842 / 0.849 | **−0.090** |
| background_consistency | 0.948 / 0.943 | 0.903 / 0.900 | **−0.045** |
| aesthetic_quality | 0.626 / 0.624 | 0.583 / 0.572 | **−0.043** |
| imaging_quality | 73.362 / 72.002 | 72.299 / 72.680 | −1.063 |
| motion_smoothness | 0.991 / 0.991 | 0.992 / 0.991 | +0.001 |
| dynamic_degree | 0.000 / 0.250 | 0.000 / 0.438 | 0.000 / +0.188 |
| temporal_flickering | 0.979 / 0.980 | 0.985 / 0.983 | +0.006 |

## How to read

- **5 s full** matches the usual VBench / VBench-I2V length (~5 s,
  ~81–85 frames). Cite this as the short-horizon VBench++ number on
  these 16.
- **30 s full** is “run VBench++ on the whole generation,” which is
  what we locked for the hybrid-32 paper table and what long-horizon
  papers do (they often use **VBench-Long** for 30–60 s). It is *not*
  duration-matched to the 5 s column. Subject −0.090 is the
  long-range identity drop the 5 s *windows* could not see
  (those windows were 0.93–0.97).
- Smoothness / flicker go slightly **up** at 30 s because a freeze
  looks locally stable when averaged over 481 frames.
- Dynamic **median** stays 0. The **mean** 0.250 → 0.438 is the
  fraction of clips RAFT ever calls dynamic; a longer clip has more
  chance of one dynamic burst, so this can rise even as the tail
  freezes.
- Do not replace this with `w25_30` or first-16. Those are
  diagnostics. Hybrid-32 three-way full 30 s is
  [`2026-08-18_wan_i2v_bon32_vbench_full.md`](2026-08-18_wan_i2v_bon32_vbench_full.md)
  (N=32, chunked, do-nothing subject 0.848 — same ballpark as this
  16v 30 s 0.842).
- No PSNR. These stills have no paired 30 s GT.
