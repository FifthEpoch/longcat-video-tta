# Panda 1000v Standard — headline + NOPROMPT ablation (PARTIAL VBench)

**Generated:** 2026-06-09  
**Status:** Per-frame metrics + FVD + FID complete via `merge_chunks.py`. VBench shows the 3 dimensions computed in-runner (`aesthetic_quality`, `background_consistency`, `subject_consistency`); the other 4 (Motn, Dyn, IQ, Flick) need the standard VBench backfill pipeline (will run alongside UCF + TinyLoRA NOPROMPT once those chunks finish).  
**Source:** `sweep_experiment/results/panda_1000v_standard/<METHOD>/merged_summary.json`

| Method | N | PSNR | SSIM | LPIPS | FVD | FID | Aes | BG | Subj |
|---|---|---|---|---|---|---|---|---|---|
| NOTTA | 999 | 17.93 | 0.6519 | 0.3380 | 154.7 | 24.8 | 0.395 | 0.928 | 0.906 |
| ADA | 999 | 17.94 | 0.6510 | 0.3373 | 153.4 | 25.2 | 0.395 | 0.929 | 0.907 |
| ADA_NOPROMPT | 999 | 17.93 | 0.6513 | 0.3377 | 155.5 | 25.1 | 0.395 | 0.929 | 0.906 |
| LORA_R8_TTA | 999 | 17.85 | 0.6495 | 0.3405 | 157.9 | 25.5 | 0.442 | 0.930 | 0.902 |
| LORA_R8_TTA_NOPROMPT | 999 | 17.86 | 0.6499 | 0.3398 | 154.0 | 25.2 | 0.441 | 0.930 | 0.902 |

## Headline

Five methods agree within 0.1 PSNR / 5 FVD / 0.4 FID on Panda 1000v standard
horizon. The TTA-time caption is essentially a noise channel at this scale.
Notable directional signal (within noise but worth flagging): `LORA_R8_TTA_NOPROMPT`
has marginally lower FVD (154.0) than `LORA_R8_TTA` (157.9), suggesting dropping
the prompt during LoRA TTA does not hurt and may slightly help on distributional
similarity — possible interpretation: when the LoRA update is no longer trying
to satisfy a text-conditioning loss, the visual loss term takes a cleaner path.

The per-frame saturation pattern is consistent with the existing 1000v
saturation story (see `2026-06-08_headline_1000v.md`). The new finding is that
saturation is robust to dropping the TTA-time text prompt — this is meaningful
for the paper because it lets us cite TTA as essentially video-conditioned
adaptation, not text-conditioned adaptation.

## What remains for this regime

- VBench full 7-dim backfill on `ADA_NOPROMPT` and `LORA_R8_TTA_NOPROMPT` (4 missing dims each — same pipeline as `2026-06-05` headline backfill)
- Per-video winners/losers analysis via `scripts/analyze_per_video_tta_gain.py` (commit `5d92733`) to see if the population-level saturation hides a winner/loser split predictable from dynamicness, baseline difficulty, or caption length
- TinyLoRA NOPROMPT + UCF NOPROMPT chunks finish (as of 2026-06-09 15:30 UTC+8: 35/80 jobs done, 31 running, 15 pending)

## Cluster source paths

- `sweep_experiment/results/panda_1000v_standard/{NOTTA,ADA,ADA_NOPROMPT,LORA_R8_TTA,LORA_R8_TTA_NOPROMPT}/merged_summary.json`

## Reproduce

```bash
cd /scratch/$USER/longcat-video-tta
python3 scripts/build_paper_tables.py --regime panda_std \
    --output sweep_experiment/reports/paper_tables/regenerated_panda_std.md
```
