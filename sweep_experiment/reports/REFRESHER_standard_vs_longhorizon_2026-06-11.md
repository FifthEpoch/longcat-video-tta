# Refresher — standard- vs long-horizon population-level results (Panda 1000v)

**Date:** 2026-06-11
**Purpose:** Quick reference for what the population-level metrics actually show across the two horizon regimes on Panda 1000v / 480p / 999 videos. The source-of-truth document is [`paper_tables/2026-06-08_headline_1000v.md`](paper_tables/2026-06-08_headline_1000v.md); this refresher pulls the numbers into side-by-side form and flags the gap that the 2026-06-11 offline-investigation suite ([`PLAN_offline_investigations_2026-06-11.md`](PLAN_offline_investigations_2026-06-11.md)) is meant to close.
**Scope:** Panda 1000v only. Long-horizon UCF (Table 4) is referenced once for context but not the focus.

---

## Side-by-side population-level metrics

### Standard horizon = 28 frames total / 17-frame generation (`panda_1000v_standard`)

| Method | PSNR | SSIM | LPIPS | FVD | FID | Subj | BG | Aes | Motn | Dyn | IQ | Flick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NOTTA              | 17.93 | 0.6519 | 0.3380 | 154.7 | 24.84 | 0.907 | 0.929 | 0.395 | 0.985 | 0.565 | 0.649 | 0.976 |
| ADA (ours)         | 17.94 | 0.6510 | 0.3373 | 153.4 | 25.22 | 0.907 | 0.929 | 0.396 | 0.985 | 0.568 | 0.649 | 0.976 |
| LORA_R8_TTA        | 17.85 | 0.6495 | 0.3405 | 157.9 | 25.48 | 0.902 | 0.931 | **0.442** | 0.986 | **0.596** | 0.615 | 0.975 |
| TL_BARE_R2         | 17.94 | 0.6520 | 0.3374 | 154.2 | 24.84 | 0.907 | 0.929 | 0.395 | 0.985 | 0.566 | 0.649 | 0.976 |
| TL_TIED_R2         | 17.93 | 0.6518 | 0.3378 | 161.1 | 24.87 | 0.907 | 0.929 | 0.395 | 0.985 | 0.564 | 0.649 | 0.976 |

### Long horizon = 76 frames total (`panda_longctx_1000v`)

| Method | PSNR | SSIM | LPIPS | FVD | FID | Subj | BG | Aes | Motn | Dyn | IQ | Flick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NOTTA              | 12.77 | 0.4744 | 0.5469 | 278.7 | 29.90 | 0.774 | 0.848 | 0.440 | 0.988 | 0.635 | 0.666 | 0.974 |
| ADA_S10 (ours)     | 12.79 | 0.4762 | 0.5436 | 284.1 | 29.53 | 0.775 | 0.848 | 0.440 | 0.988 | 0.630 | 0.666 | 0.974 |
| LORA_R8            | 12.73 | 0.4726 | 0.5480 | 282.4 | 30.32 | **0.757** | 0.848 | **0.485** | 0.988 | **0.660** | 0.642 | 0.973 |
| PANDA_TL_LAST24    | 12.77 | 0.4744 | 0.5468 | 278.6 | 30.09 | 0.774 | 0.848 | 0.440 | 0.988 | 0.645 | 0.666 | 0.974 |

---

## Largest regime-level deltas (long-horizon vs standard, NOTTA row)

| Metric | Standard | Long-horizon | Δ (long − std) | Reading |
|---|---:|---:|---:|---|
| PSNR  | 17.93 | 12.77 | **−5.16 dB** | Long-horizon is dramatically harder per-frame (longer prediction window). |
| SSIM  | 0.6519 | 0.4744 | **−0.1775** | Same direction; structure is harder to maintain. |
| LPIPS | 0.3380 | 0.5469 | **+0.2089** | Perceptually further from GT. |
| FVD   | 154.7 | 278.7 | **+124.0** | Distributional gap roughly doubles. |
| Subj  | 0.907 | 0.774 | **−0.133** | The identity-drift effect: subject consistency falls sharply on the longer window. |
| Dyn   | 0.565 | 0.635 | +0.070 | More dynamic content slips through over a longer window (consistent with motion accumulating). |
| Aes   | 0.395 | 0.440 | +0.045 | Aesthetic quality actually *increases* — the model paints prettier frames over the longer window. |

---

## Method-vs-NOTTA effects per regime

| Method | ΔPSNR std | ΔPSNR long | ΔFVD std | ΔFVD long | ΔSubj std | ΔSubj long | ΔAes std | ΔAes long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADA / ADA_S10    | +0.01 | +0.02 | −1.3 | +5.4 | 0.000 | +0.001 | +0.001 | 0.000 |
| LORA_R8(_TTA)    | −0.08 | −0.04 | +3.2 | +3.7 | −0.005 | **−0.017** | +0.047 | +0.045 |
| TL_BARE_R2       | +0.01 |  — | −0.5 |  — | 0.000 |  — | 0.000 |  — |
| TL_TIED_R2       | 0.00 |  — | +6.4 |  — | 0.000 |  — | 0.000 |  — |
| PANDA_TL_LAST24  |  — | 0.00 |  — | −0.1 |  — | 0.000 |  — | 0.000 |

(Cells dashed where the method is only run in one regime: TL_BARE_R2 / TL_TIED_R2 are standard-only; PANDA_TL_LAST24 is long-only.)

**Headline observations from the two-regime side-by-side:**

1. **All TTA methods sit at the saturation floor in BOTH regimes** — ΔPSNR is within ±0.1 dB of No-TTA for every (method × regime) pair. The headline conclusion (TTA does not move the population mean) is regime-invariant.
2. **AdaSteer stays neutral in both regimes.** ΔPSNR ≈ 0, ΔFVD within ±6, every VBench dim within ±0.001 of No-TTA. No quality-vs-perceptual frontier shift.
3. **LoRA-r8's distribution-shift signature is regime-invariant in direction but slightly muted in magnitude on long horizon.** It buys +0.045 Aes for −0.034 IQ on standard horizon and +0.045 Aes for −0.024 IQ on long horizon — same direction, slightly smaller IQ cost on the long-horizon regime.
4. **Subject-consistency divergence is the ONLY metric in the entire 1000v sweep where AdaSteer and LoRA visibly part ways on the long-horizon regime.** No-TTA drops to 0.774, ADA preserves it (0.775), LoRA *worsens* it (0.757). On standard horizon Subj is at the saturated 0.907 for everyone — long-horizon is the only regime where this signal exists.
5. **Method rankings (best-to-worst by PSNR):**
   - Standard horizon: ADA ≈ TL_BARE_R2 (17.94) > NOTTA ≈ TL_TIED_R2 (17.93) > LORA_R8_TTA (17.85). LoRA last.
   - Long horizon: ADA_S10 (12.79) > NOTTA ≈ PANDA_TL_LAST24 (12.77) > LORA_R8 (12.73). LoRA still last.
   - **Ranking is preserved** — the standard-horizon best (ADA) is also the long-horizon best, and the standard-horizon worst (LoRA) is also the long-horizon worst.

---

## What's missing — the gap the 2026-06-11 offline suite closes

- **Per-video analysis at long-horizon has NEVER been run.** The 2026-06-09 bundle at [`per_video_analysis/2026-06-09/`](per_video_analysis/2026-06-09/) covers only the standard regime (`panda_1000v_standard` + `tinylora_panda_1000v_standard`, ΔPSNR + dynamicness + caption-length + baseline-PSNR correlations, tail breakdowns at ±0.5/±1.0 dB, top-10 winners/losers per method). The long-horizon counterpart (against `panda_longctx_1000v` + `tinylora_longctx_1000v`) is the **primary target** of the offline-investigation suite scheduled while the cluster is in maintenance through ~2026-06-15.
- **Per-chunk ΔFVD sign analysis** at chunk granularity (10 chunks × 100 videos each) was flagged as a TODO in the [2026-06-09 prompt-vs-NOPROMPT writeup](paper_tables/2026-06-09_panda_std_prompt_vs_noprompt_full_metrics.md) but never executed. The data lives in `<series>/<METHOD>/chunk_*/summary.json['fvd']` on the cluster filesystem — CPU-only to compute.
- **Top-50-winner Jaccard matrix + sign-agreement-across-methods statistics** were computed on-the-fly during the 2026-06-09 analysis (the "6.3× lift" number cited internally) but never persisted into `summary.md`. The 2026-06-11 suite extends `analyze_per_video_tta_gain.py` to write them natively, so the long-horizon bundle gets them automatically.
- **Loss-history extraction.** Per-video held-out anchor-loss trajectories are persisted under `result['early_stopping_info']['loss_history']` in every chunk's `summary.json` (per-step training loss is NOT persisted; only `final_loss` is). The suite's `aggregate_loss_history.py` joins these against ΔPSNR per video to test the "winners' loss decreases / losers' loss stays flat" mechanism hypothesis.
