# Budget grid — Panda OOD-preview 1000v (12 AdaSteer configs)

**Date:** 2026-07-19
**Series:** `panda_ood_budget_1000v_preview` (N=1000 OOD-stratified segment pool)
**Dataset:** `datasets/panda_ood_budget_1000v_preview_480p`
**Geometry:** 14 cond + 14 gen, gen_start=48, 50 inference steps, guidance 4.0
**Method:** AdaSteer (`delta_a`), step×LR grid: S{2,5,10,20} × LR{1e-3,5e-3,1e-2}
**Source:** per-arm `merged_summary.json` (10 chunks × 100 = 1000 videos each),
merged via `scripts/run_preview_1000v_pipeline.sh merge`. FVD/FID are global over
998 videos (2 undecodable). Regenerable from the merged_summary.json files.
**NOTTA baseline:** submitted 2026-07-19 (jobs 14319937–946) on the SAME pool;
row to be added after it merges.

## Population-level metrics (all 1000 videos, per config)

| Config | steps | LR | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | FID↓ | train s | total s |
|---|---|---|---|---|---|---|---|---|---|
| S2_LR1e3  | 2  | 1e-3 | 19.481 | 0.6886 | 0.2536 | 66.1 | 12.4 | 15.4  | 95.5  |
| S2_LR5e3  | 2  | 5e-3 | 19.476 | 0.6888 | 0.2536 | 66.6 | 12.5 | 15.3  | 95.3  |
| S2_LR1e2  | 2  | 1e-2 | 19.479 | 0.6889 | 0.2537 | 67.2 | 12.4 | 15.3  | 94.8  |
| S5_LR1e3  | 5  | 1e-3 | **19.486** | 0.6888 | 0.2533 | 66.5 | 12.4 | 34.0  | 113.6 |
| S5_LR5e3  | 5  | 5e-3 | 19.469 | 0.6886 | 0.2535 | 66.4 | 12.4 | 34.0  | 113.4 |
| S5_LR1e2  | 5  | 1e-2 | 19.452 | 0.6879 | 0.2538 | 65.7 | 12.3 | 34.0  | 113.5 |
| S10_LR1e3 | 10 | 1e-3 | **19.486** | 0.6888 | **0.2531** | 65.7 | 12.3 | 65.3  | 145.2 |
| S10_LR5e3 | 10 | 5e-3 | 19.462 | 0.6885 | 0.2538 | 68.8 | 12.5 | 65.4  | 145.2 |
| S10_LR1e2 | 10 | 1e-2 | 19.445 | 0.6876 | 0.2545 | 67.8 | 12.5 | 65.1  | 144.5 |
| S20_LR1e3 | 20 | 1e-3 | 19.481 | 0.6886 | 0.2534 | **65.2** | 12.3 | 128.0 | 207.7 |
| S20_LR5e3 | 20 | 5e-3 | 19.441 | 0.6877 | 0.2544 | 66.2 | 12.4 | 127.9 | 207.5 |
| S20_LR1e2 | 20 | 1e-2 | 19.372 | 0.6851 | 0.2570 | 67.6 | 12.5 | 128.6 | 208.6 |
| **spread** | | | **0.114** | 0.0038 | 0.0039 | 3.6 | 0.2 | 8.4× | 2.2× |

## Reading

- **Population metrics are flat.** Across all 12 configs PSNR spans only
  **0.11 dB** (19.37–19.49), SSIM 0.0038, LPIPS 0.0039, FVD 3.6, FID 0.2. The
  step×LR budget essentially does not move the *mean* — the same in-domain
  short-horizon saturation seen in `panda_1000v_standard`.
- **Only cost scales.** train time goes 15s → 34s → 65s → 128s with steps
  (8.4×), while quality is unchanged → more steps buy nothing at the population
  level.
- **Over-adaptation is the only visible trend, and it's negative.** The most
  aggressive config **S20_LR1e2** is worst on PSNR (19.372), SSIM, and LPIPS;
  the conservative/low-LR configs are marginally best. Aggressive TTA slightly
  *hurts* in-domain.
- **Implication for the paper narrative:** a flat mean is exactly the setup where
  a **per-video router** must carry the story — the value is in per-clip config
  selection (cf. the N=200 pilot: oracle PSNR routing +0.95 dB vs no-TTA, +0.75
  vs best fixed config), not in any single fixed budget. It also motivates the
  **13th "skip-TTA" candidate**: if TTA doesn't help on average, many videos are
  better off untouched.

## Next

1. Merge NOTTA (in flight) → add the no-TTA row here; confirm AdaSteer ≈ NoTTA at
   the population level on this exact pool (apples-to-apples).
2. Per-video oracle + learned-router analysis
   (`analyze_adasteer_budget_oracle.py`) across the 5 OOD quintiles — the actual
   headline for scale-up.
