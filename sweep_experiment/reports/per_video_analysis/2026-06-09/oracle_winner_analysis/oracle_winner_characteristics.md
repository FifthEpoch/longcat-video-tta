# Oracle winner characteristics (Panda 1000v, N=999)

## Oracle definition (confirmed)

Per video, oracle routing picks the method with the **highest absolute PSNR** among NOTTA, AdaSteer (`ADA`), and LoRA_R8_TTA. Population oracle PSNR is the mean of those per-video best PSNR values (not the mean of per-video ΔPSNR).

| Policy | Mean PSNR | Δ vs always-NOTTA |
|---|---:|---:|
| Always NOTTA | 17.930 dB | 0.000 dB |
| Always AdaSteer | 17.938 dB | +0.008 dB |
| Always LoRA | 17.855 dB | -0.076 dB |
| **Oracle (best PSNR)** | **18.156 dB** | **+0.226 dB** |

**Oracle picks:** NOTTA 345 (34.5%) · AdaSteer 446 (44.6%) · LoRA 208 (20.8%)

Oracle ΔPSNR vs NOTTA: mean 0.226 dB, median 0.028 dB.

## Win magnitude: AdaSteer vs LoRA (head-to-head on ΔPSNR)

When a method *wins* oracle (absolute PSNR), it also tends to win on ΔPSNR. Head-to-head ΔPSNR comparison quantifies margin sizes when one TTA method beats the other.

| Comparison | N | Mean gain (dB) | Median gain (dB) |
|---|---:|---:|---:|
| AdaSteer beats LoRA (ΔPSNR) | 553 | 0.462 | 0.132 |
| LoRA beats AdaSteer (ΔPSNR) | 446 | 0.386 | 0.111 |
| AdaSteer oracle wins → Ada ΔPSNR | 446 | 0.410 | 0.134 |
| LoRA oracle wins → LoRA ΔPSNR | 208 | 0.204 | 0.038 |

**Takeaway:** AdaSteer wins are larger in magnitude — when AdaSteer beats LoRA on ΔPSNR, mean margin 0.462 dB (median 0.132) vs LoRA wins mean 0.386 dB (median 0.111). Oracle-win ΔPSNR: AdaSteer mean 0.410 dB vs LoRA 0.204 dB.

## Feature means by oracle winner bucket

_Note: `video_features.csv` not loaded — cluster path: `sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv`_

_Note: `diffusion_ood_scores.csv` not loaded — cluster path: `sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv`_

_Note: `tier3_probe_features.csv` not loaded — cluster path: `sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features.csv`_

| Feature | NOTTA mean | AdaSteer mean | LoRA mean | η² |
|---|---:|---:|---:|---:|
| `NOTTA_psnr` | 18.6 | 17.9 | 16.8 | 0.008 |
| `mean_flow` | 1.96 | 1.82 | 2.73 | 0.008 |
| `caption_len_words` | 38.3 | 38.4 | 36.9 | 0.005 |
| `caption_len_chars` | 202 | 203 | 195 | 0.005 |

## OOD hypothesis (exploratory)

Illustrative hypotheses (not forced): LoRA on moderately OOD, NOTTA on extremely OOD, AdaSteer on in-distribution. Test via diffusion OOD quintiles when `diffusion_ood_scores.csv` is available.

_OOD quintile stratification skipped — `mean_diffusion_loss_caption` not available. Re-run with `--ood-csv` after cluster Stage 1b._

## Per-bucket characterization (from available features)

### NOTTA wins (345 videos)
- NOTTA_psnr: mean 18.6, median 18.1
- mean_flow: mean 1.96, median 0.709
- caption_len_words: mean 38.3, median 37

### AdaSteer wins (446 videos)
- NOTTA_psnr: mean 17.9, median 17.4
- mean_flow: mean 1.82, median 0.66
- caption_len_words: mean 38.4, median 38

### LoRA wins (208 videos)
- NOTTA_psnr: mean 16.8, median 15.7
- mean_flow: mean 2.73, median 0.928
- caption_len_words: mean 36.9, median 35

## Top features differing across winner buckets (ANOVA η²)

| Feature | F | η² | N |
|---|---:|---:|---:|
| `NOTTA_psnr` | 4.16 | 0.008 | 999 |
| `mean_flow` | 4.12 | 0.008 | 999 |
| `caption_len_words` | 2.35 | 0.005 | 999 |
| `caption_len_chars` | 2.28 | 0.005 | 999 |

## Why oracle (+0.226 dB) >> always-AdaSteer (+0.008 dB)

- Oracle picks NOTTA on 345 videos (34.5%) where TTA hurts; always-AdaSteer forces TTA on all of them.
- Oracle picks LoRA on 208 videos where AdaSteer is suboptimal.
- Skip-Ada-if-Δ≤0 policy recovers most uplift (~+0.213 dB) because AdaSteer ΔPSNR is ≤0 on roughly half of videos.
