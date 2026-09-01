# Caption 128 paired 30 s pixels — four-way DONE (2026-09-01)

**Series:** `v2v_panda_caption_128v`. Prompt = `metadata_csv`.
n=128. Medians. Paired invented tail vs real leftover
(skip 33 source frames; 16 fps; n=504 frames / clip).
Jobs: **16678705** (SF), **16694796** (RF + Pseudo; Always
50/128 then preempted), **16702323** (Always remainder,
COMPLETED 0:0 1h14). Do not remake videos.

| Method | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Self Forcing | 128 | **9.25** | **0.279** | — |
| Rolling Forcing | 128 | 7.98 | 0.250 | — |
| Pseudo-future Search | 128 | 9.22 | 0.268 | — |
| Always-search | 128 | 9.21 | 0.266 | — |

Raw: SF 9.254345 / 0.279004. RF 7.976055 / 0.250440.
Pseudo 9.224638 / 0.267739. Always 9.212234 / 0.265640.
`lpips=None` in all four (`import lpips` fails in
`self_forcing` env). Always `pixel_full` has 129 files
(128 clips + `summary.json` at 01:20).

## Read

Search does **not** pay a reconstruction tax. Pseudo and
Always sit 0.03–0.04 dB under Self Forcing and 0.01–0.013
SSIM under. That is a tie for this check. The gate (Always
vs Pseudo) is also a pixel tie.

Rolling is the outlier: **−1.28 dB / −0.029 SSIM** vs Self
Forcing, while winning official subject (0.685) and matching
the tail (0.0158). Same LongCat lesson: more invented motion
leaves the literal leftover. Do not cite PSNR as the bake-off.

Headline stays full-clip VBench + Dyn%. VBench grid:
[`2026-08-31_wan_v2v_cite128_all_metrics.md`](2026-08-31_wan_v2v_cite128_all_metrics.md).

## Still open

- **LPIPS:** env gap, not a missing job.
- **FVD:** not launched. I3D on **aligned 30 s tails** only,
  `--force` (n=128 < 256). Do not score the full mp4
  (includes the real 2 s prefix).
