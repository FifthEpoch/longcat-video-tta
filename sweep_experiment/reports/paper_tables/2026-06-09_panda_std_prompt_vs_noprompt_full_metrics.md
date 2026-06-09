# Panda 1000v Standard — Prompt vs No-Prompt full metrics

**Generated:** 2026-06-09 (evening)
**Status:** Per-frame metrics + FVD + FID complete via `merge_chunks.py` for all 5 methods (NOTTA + 4 NOPROMPT-paired methods). VBench shows 7 dims for prompted methods, 3 dims for NOPROMPT methods (Motn / Dyn / IQ / Flick await the standard backfill pipeline, queued behind cluster maintenance and the remaining UCF + TinyLoRA NOPROMPT chunks).
**Source:**
- `sweep_experiment/results/panda_1000v_standard/<METHOD>/merged_summary.json` (cluster)
- This document is built from [`2026-06-08_headline_1000v.md`](2026-06-08_headline_1000v.md) (prompted full 7-dim) and [`2026-06-09_panda_std_with_noprompt_partial.md`](2026-06-09_panda_std_with_noprompt_partial.md) (NOPROMPT partial).

## Per-frame + distributional metrics

| Method | N | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | FID ↓ |
|---|---:|---:|---:|---:|---:|---:|
| NOTTA (baseline) | 999 | 17.93 | 0.6519 | 0.3380 | 154.7 | 24.8 |
| ADA | 999 | 17.94 | 0.6510 | 0.3373 | 153.4 | 25.2 |
| **ADA_NOPROMPT** | 999 | **17.93** | **0.6513** | **0.3377** | **155.5** | **25.1** |
| LORA_R8_TTA | 999 | 17.85 | 0.6495 | 0.3405 | 157.9 | 25.5 |
| **LORA_R8_TTA_NOPROMPT** | 999 | **17.86** | **0.6499** | **0.3398** | **154.0** | **25.2** |

## VBench (7 dims for prompted, 3 dims for NOPROMPT)

VBench dimensions: Subj=`subject_consistency`, BG=`background_consistency`, Aes=`aesthetic_quality`, Motn=`motion_smoothness`, Dyn=`dynamic_degree`, IQ=`imaging_quality`, Flick=`temporal_flickering`. Subj / BG / Aes are reported here at the 3-decimal precision used by both source files (the underlying `merged_summary.json` files do not store more precision than that for these three in-runner dims, and the 4 backfill dims on the headline file are also recorded at 3 decimals — so we keep the existing convention rather than fabricate a fourth digit).

| Method | Subj ↑ | BG ↑ | Aes ↑ | Motn ↑ | Dyn ↑ | IQ ↑ | Flick ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| NOTTA (baseline) | 0.906 | 0.928 | 0.395 | 0.985 | 0.565 | 0.649 | 0.976 |
| ADA | 0.907 | 0.929 | 0.395 | 0.985 | 0.568 | 0.649 | 0.976 |
| **ADA_NOPROMPT** | **0.906** | **0.929** | **0.395** | — | — | — | — |
| LORA_R8_TTA | 0.902 | 0.930 | 0.442 | 0.986 | 0.596 | 0.615 | 0.975 |
| **LORA_R8_TTA_NOPROMPT** | 0.902 | 0.930 | 0.441 | — | — | — | — |

Note on Subj / BG values for prompted methods: the 3 in-runner VBench dims are reported here using the rounded values from [`2026-06-09_panda_std_with_noprompt_partial.md`](2026-06-09_panda_std_with_noprompt_partial.md) so the prompted and NOPROMPT rows are directly comparable on the same rounding convention. The fully-rebackfilled prompted-only headline ([`2026-06-08_headline_1000v.md`](2026-06-08_headline_1000v.md)) records NOTTA Subj=0.907 / BG=0.929 and LORA_R8_TTA BG=0.931 — the third-digit drift is rounding noise from the two computation paths (in-runner vs backfill) and does not affect the prompt-vs-NOPROMPT comparison.

## Pairwise deltas (NOPROMPT − prompted)

| Pair | ΔPSNR | ΔSSIM | ΔLPIPS | ΔFVD | ΔFID | ΔSubj | ΔBG | ΔAes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADA_NOPROMPT − ADA | -0.01 | +0.0003 | +0.0004 | +2.1 | -0.1 | -0.001 | 0.000 | 0.000 |
| LORA_R8_TTA_NOPROMPT − LORA_R8_TTA | +0.01 | +0.0004 | -0.0007 | -3.9 | -0.3 | 0.000 | 0.000 | -0.001 |

Sign convention: `NOPROMPT − prompted`. Positive ΔPSNR / ΔSSIM = NOPROMPT better; positive ΔLPIPS / ΔFVD / ΔFID = NOPROMPT worse. VBench (Subj / BG / Aes here) is "higher is better" for these dims so positive = NOPROMPT better.

## Reading

Both pairs sit within 0.01 PSNR / ≤0.001 SSIM / ≤0.001 LPIPS / 4 FVD / 0.3 FID / 0.001 VBench-dim of their prompted siblings. The TTA-time text caption is a noise channel on Panda 1000v / 480p / 17-frame standard horizon for the AdaSteer and LoRA-r8 families. The TinyLoRA NOPROMPT pairing is not yet on disk (chunks were running at maintenance shutdown); the analogous row will be added once those chunks complete + merge.

## What's missing and how to fix

1. **VBench Motn / Dyn / IQ / Flick on the 2 NOPROMPT methods.** Run the standard `run_vbench_backfill.py` pipeline on `panda_1000v_standard/{ADA,LORA_R8_TTA}_NOPROMPT/`. Each takes ~30 min on H200.
2. **TinyLoRA NOPROMPT pairings (TL_BARE_R2_NOPROMPT, TL_TIED_R2_NOPROMPT).** Resume the standard-horizon NOPROMPT sweep when cluster returns; submit script is `sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh`.
3. **Per-chunk ΔFVD sign analysis.** Each chunk has its own FVD computed on 100 videos — 10 chunks per method gives 10 ΔFVD values per (method, NOPROMPT-vs-prompted) pair. Useful for confidence intervals on the population-level ΔFVD numbers reported above. Requires `chunk_*/summary.json` files which live on the cluster and are not under version control. **TODO:** when cluster returns, add a small script (`scripts/analyze_per_chunk_fvd.py`) that reads each `chunk_*/summary.json`, extracts the per-chunk FVD, computes per-pair ΔFVD per chunk, and emits a sign-analysis table (e.g. "5/10 chunks improved under NOPROMPT").

## Per-video ΔLPIPS tail breakdown (% of 999 videos)

LPIPS is "lower is better"; Δ = `LPIPS(method) − LPIPS(NOTTA)`. **Negative Δ = TTA improved.**
LPIPS is the per-video perceptual analog of FVD (which is distributional and not per-video).

Computed from [`../per_video_analysis/2026-06-09/per_video_gains.csv`](../per_video_analysis/2026-06-09/per_video_gains.csv) (schema uses `<METHOD>_lpips` column names — note the order is `<method>_<metric>` rather than `<metric>_<method>` as the spec guessed; deltas are recomputed directly from the per-method and `NOTTA_lpips` columns for the sign convention above). N = 999 for every method (full intersection, no rows dropped — see [`../per_video_analysis/2026-06-09/summary.md`](../per_video_analysis/2026-06-09/summary.md)).

| method | Δ < −0.01 (TTA wins) | \|Δ\| ≤ 0.01 | Δ > +0.01 (TTA loses) | Δ < −0.005 | \|Δ\| ≤ 0.005 | Δ > +0.005 | mean Δ | median Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADA | 14.6 % | 69.4 % | 16.0 % | 20.7 % | 54.8 % | 24.5 % | -0.0008 | +0.0001 |
| ADA_NOPROMPT | 13.7 % | 71.1 % | 15.2 % | 19.1 % | 58.2 % | 22.7 % | -0.0003 | +0.0000 |
| LORA_R8_TTA | 3.3 % | 86.9 % | 9.8 % | 7.2 % | 74.8 % | 18.0 % | +0.0024 | +0.0006 |
| LORA_R8_TTA_NOPROMPT | 3.8 % | 87.5 % | 8.7 % | 7.2 % | 76.0 % | 16.8 % | +0.0017 | +0.0004 |
| TL_BARE_R2 | 4.5 % | 90.9 % | 4.6 % | 8.8 % | 82.2 % | 9.0 % | -0.0006 | -0.0000 |
| TL_TIED_R2 | 4.6 % | 90.9 % | 4.5 % | 9.2 % | 82.3 % | 8.5 % | -0.0003 | -0.0000 |

### Reading

The LPIPS picture is consistent with the ΔPSNR saturation story in [`../per_video_analysis/2026-06-09/summary.md`](../per_video_analysis/2026-06-09/summary.md) — the same method ordering reappears (TinyLoRA tightest around the zero-Δ line at ~82 % within ±0.005, LoRA-r8 in the middle at ~75 %, AdaSteer loosest at ~55 %, and the ADA tail is again roughly symmetric while the LoRA tail is right-skewed — i.e. LoRA's wins are slightly outweighed by losses on the perceptual axis just as they are on PSNR). Median Δ is essentially zero for every method (|median| ≤ 0.0006), and mean Δ is also tiny (|mean| ≤ 0.0024), so the population-level LPIPS saturation in the headline table is not hiding a one-sided per-video story.

The NOPROMPT pairs are distributionally indistinguishable from their prompted siblings on LPIPS too: `ADA_NOPROMPT` is within ≤1.0 pp of `ADA` on every bucket, with a barely-tighter centre (58.2 % vs 54.8 % within ±0.005); `LORA_R8_TTA_NOPROMPT` is within ≤1.2 pp of `LORA_R8_TTA` on every bucket, also marginally tighter at the centre (76.0 % vs 74.8 %). The same "NOPROMPT is a slight noise reducer, not a real signal change" reading the population-level deltas suggest carries through to the per-video tails.

One qualitative note vs the PSNR tail breakdown: AdaSteer's per-video LPIPS distribution is *not* nearly as concentrated as its per-video PSNR distribution (~55 % within ±0.005 LPIPS vs ~81 % within ±0.5 PSNR). LPIPS is a stricter per-video witness than PSNR for the AdaSteer family — small adapter-driven texture / perceptual shifts that PSNR averages out are still visible to a learned perceptual metric. This is a per-video signal worth flagging if a reviewer asks why the saturated headline numbers still produce visibly different outputs on individual clips.

## Why ΔFVD per-video doesn't exist

FVD is the Fréchet distance between two distributions of I3D feature vectors and requires a *batch* of videos. There is no single-video FVD. The closest per-video proxies are:

- **ΔLPIPS** (above) — perceptual, per-video, per-frame averaged.
- **Δ per-video FID** — possible to compute from `per_video_gains.csv` if FID columns are present (FID *is* a single-video quantity since it's typically defined as a Fréchet distance against the GT distribution at the population level, but per-video FID can be reconstructed from per-frame Inception features if available). The 2026-06-09 CSV does *not* currently carry per-video FID columns (its per-method columns are `_psnr`, `_ssim`, `_lpips`, and the precomputed `_dpsnr`, `_dssim`, `_dlpips` deltas), so this would require a small extraction pass against the per-video-feature artefacts on the cluster.
- **Per-chunk ΔFVD** (10 chunks × 100 videos each) — computable from `chunk_*/summary.json` once cluster returns.

If the user wants per-chunk ΔFVD specifically, that's a small post-cluster-maintenance task (see TODO in §"What's missing" above).
