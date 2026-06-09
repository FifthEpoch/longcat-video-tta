# What makes a video suitable for TTA? — Review as of 2026-06-09

**Status:** Cluster is in maintenance for several days; this document is a stocktake before the next experimental wave.  
**Companion:** [HYPOTHESES_per_video_tta_suitability_2026-06-09.md](HYPOTHESES_per_video_tta_suitability_2026-06-09.md) (literature pass on new ideas, parallel workstream)

## TL;DR

At Panda 1000v / 480p / 17-frame standard horizon, per-video ΔPSNR is dominated by sampling noise: 81–95% of clips fall within ±0.5 dB of NOTTA for every TTA method we ship, and the three structural features we have measured (RAFT mean optical flow, baseline PSNR, caption word count) explain none of the residual variance (|Spearman ρ| ≤ 0.09 across all six method × feature cells we trust). The NOPROMPT ablation rules out TTA-time caption as a useful signal channel (Δ within ±0.01 PSNR / ±4 FVD vs the headline). The richer per-video feature battery (CLIP/DINO/PySceneDetect on the TTA-visible window) and a long-horizon per-video bundle are *implemented* but never executed; both are top of the next wave alongside the in-flight diffusion-OOD hypothesis from the parallel literature pass.

## 1. Question

What per-video features predict whether a TTA recipe (AdaSteer / LoRA r=8 / TinyLoRA r=2) will improve, leave unchanged, or hurt the model on that video?

## 2. Findings from completed experiments

### 2.1 Saturation at Panda 1000v / 480p / 17-frame standard horizon

Population-level result already in `paper_tables/2026-06-08_headline_1000v.md` (Table 1): five methods agree within 0.1 PSNR / 8 FVD on N=999.

Per-video distribution (from `per_video_analysis/2026-06-09/summary.md`):

| Method | N | mean Δ | median Δ | \|Δ\|≤0.5 dB | \|Δ\|≤1.0 dB |
|---|---:|---:|---:|---:|---:|
| `ADA` | 999 | +0.0080 | −0.0003 | 808 (80.9%) | 899 (90.0%) |
| `ADA_NOPROMPT` | 999 | +0.0020 | +0.0040 | 816 (81.7%) | 902 (90.3%) |
| `LORA_R8_TTA` | 999 | −0.0756 | −0.0109 | 943 (94.4%) | 971 (97.2%) |
| `LORA_R8_TTA_NOPROMPT` | 999 | −0.0650 | −0.0078 | 931 (93.2%) | 967 (96.8%) |
| `TL_BARE_R2` | 999 | +0.0108 | −0.0016 | 950 (95.1%) | 982 (98.3%) |
| `TL_TIED_R2` | 999 | +0.0027 | +0.0009 | 951 (95.2%) | 980 (98.1%) |

TinyLoRA is the tightest family (~95% within ±0.5 dB, ~98% within ±1.0 dB); AdaSteer is the loosest (~81% / 90%). The whole table is consistent with a noise floor on the order of ±0.5 dB per video — see the operational lesson in §8.

Per-video ΔPSNR correlations against the three features we measured (Pearson r, Spearman ρ; from the same summary):

| method | r(Δ, mean_flow) ρ | r(Δ, baseline PSNR) ρ | r(Δ, caption words) ρ |
|---|---|---|---|
| `ADA` | −0.097 (−0.069) | −0.004 (+0.013) | −0.010 (+0.012) |
| `ADA_NOPROMPT` | −0.070 (−0.085) | −0.045 (+0.006) | −0.018 (−0.025) |
| `LORA_R8_TTA` | +0.021 (+0.073) | −0.143 (−0.088) | +0.001 (−0.024) |
| `LORA_R8_TTA_NOPROMPT` | −0.005 (+0.036) | −0.112 (−0.045) | −0.008 (−0.017) |
| `TL_BARE_R2` | −0.046 (−0.010) | −0.017 (+0.011) | −0.005 (−0.018) |
| `TL_TIED_R2` | −0.058 (+0.014) | −0.062 (−0.074) | +0.053 (+0.062) |

Max |Spearman ρ| across all 18 cells is 0.088 (`LORA_R8_TTA` vs baseline PSNR). Source plots are `delta_psnr_vs_{dynamicness,baseline_psnr,caption_length}.png` and `delta_psnr_histogram.png` in the same bundle directory.

**Conclusion (this regime, this scale):** the population is saturated, the tails are small, and none of the three features we measured predicts which videos sit in the tails. The cross-method top-N winner overlap and an explicit "sign-agreement across all methods" lift are not reported in `summary.md` — they would have to come from a re-analysis pass on `per_video_gains.csv`.

### 2.2 NOPROMPT confound check

From `paper_tables/2026-06-09_panda_std_with_noprompt_partial.md`:

| | PSNR | FVD |
|---|---:|---:|
| ADA → ADA_NOPROMPT | 17.94 → 17.93 (Δ −0.01) | 153.4 → 155.5 (Δ +2.1) |
| LORA_R8_TTA → LORA_R8_TTA_NOPROMPT | 17.85 → 17.86 (Δ +0.01) | 157.9 → 154.0 (Δ −3.9) |

Per-video ρ(Δ, caption words) for the NOPROMPT variants: ADA_NOPROMPT −0.025, LORA_R8_TTA_NOPROMPT −0.017 — both essentially zero, as expected when the TTA loss never sees the caption. The TTA-time prompt is a noise channel at this scale; adaptation is video-conditioned.

VBench-Aes / VBench-Subj on the NOPROMPT rows match their parent method to three decimals (Aes 0.395 / Subj 0.906 for both ADA variants; Aes 0.441–0.442 / Subj 0.902 for both LoRA variants). Full 7-dim VBench backfill on the NOPROMPT rows is pending (§5.5).

### 2.3 "Beneficiary" cohort (text-on-screen / cartoon, low baseline PSNR)

The 999-video winner lists in `summary.md` show that the same handful of low-baseline text-heavy or cartoon clips appear in the top-10 winners for multiple methods:

| video | baseline PSNR | mean_flow | caption (truncated) | top-10 winner under |
|---|---:|---:|---|---|
| `panda_0461` | 14.04 | 0.071 | "An iphone, a cup of coffee, a yellow sticky note, and a computer are on a desk…" | ADA (#9, +3.50), ADA_NOPROMPT (#1, +8.92), LORA_R8_TTA (#1, +8.25), LORA_R8_TTA_NOPROMPT (#1, +7.94) |
| `panda_0555` | 7.82 | 0.366 | "A cartoon girl looking at her phone with a speech bubble that says good morning…" | LORA_R8_TTA (#3, +3.13), LORA_R8_TTA_NOPROMPT (#3, +3.16), TL_BARE_R2 (#2, +3.14), TL_TIED_R2 (#1, +3.15) |
| `panda_0862` | 10.28 | 1.258 | "A group of cartoon people with their arms up in the air. A dragon ball z…" | LORA_R8_TTA (#2, +7.41), LORA_R8_TTA_NOPROMPT (#2, +7.55), TL_BARE_R2 (#1, +7.53) |
| `panda_0431` | 31.13 | 0.593 | "A black background with red text on it…" | LORA_R8_TTA_NOPROMPT (#4, +2.83), TL_BARE_R2 (#3, +2.89), TL_TIED_R2 (#2, +2.89) |

Caveat: `panda_0461` is simultaneously a top-10 *loser* under TL_BARE_R2 (#7, −1.23). The cohort is therefore "beneficiary under most methods, not universal" — footnote-worthy, not subset-claim-worthy. Order-of-magnitude size on visual inspection of the winner lists: ~5–10 clips.

### 2.4 Single LoRA-r8 catastrophe (`panda_0098`)

`panda_0098` ("home workshop makeover tour" — text-on-white-background) sits at the bottom of LoRA's loser list with ΔPSNR = −22.396 (44.55 → 22.16 dB) under LORA_R8_TTA and −23.516 (44.55 → 21.03 dB) under LORA_R8_TTA_NOPROMPT. The aggregate LORA_R8_TTA mean Δ = −0.0756 over 999 videos = −75.5 dB total; this single video contributes −22.4 / −75.5 ≈ 30% of the aggregate negative bias. It also explains the bulk of the gap between Pearson r (−0.143) and Spearman ρ (−0.088) for ΔPSNR vs baseline PSNR under LORA_R8_TTA — a high-baseline outlier with a huge negative Δ pulls the linear fit but not the rank correlation.

## 3. Hypotheses ruled out (this regime, this scale)

For each, the |Spearman ρ| evidence and the plot file it comes from. All values are at Panda 1000v / 480p / 17-frame standard horizon; nothing here generalises beyond that.

- **H1: Motion magnitude predicts ΔPSNR.** Ruled out — |Spearman ρ| ≤ 0.085 across all six methods; signs disagree across method families (ADA negative, LoRA positive, TinyLoRA mixed). Source: `per_video_analysis/2026-06-09/delta_psnr_vs_dynamicness.png` and the correlation table in `summary.md`.
- **H2: Hard-baseline videos benefit more (regression to the mean).** Ruled out for AdaSteer (Spearman ρ = +0.013) and TinyLoRA (TL_BARE_R2 +0.011, TL_TIED_R2 −0.074). Visible for LoRA r=8 (Pearson r = −0.143) but the rank correlation is only −0.088 and §2.4 attributes most of the linear-fit signal to `panda_0098`. Not paper-claim grade. Source: `delta_psnr_vs_baseline_psnr.png`.
- **H3: Caption length / verbosity drives gains.** Ruled out — |Spearman ρ| ≤ 0.062 across all methods (max at TL_TIED_R2, +0.062). Source: `delta_psnr_vs_caption_length.png`.
- **H4: TTA-time caption availability matters.** Ruled out — population-level Δ within ±0.01 PSNR / ±4 FVD vs the prompted variants on both ADA and LoRA-r8, and per-video ρ(Δ, caption words) for NOPROMPT variants is ≈ −0.02. Source: `paper_tables/2026-06-09_panda_std_with_noprompt_partial.md`.

## 4. Hypotheses still open

- **Long-horizon (76-frame) regime may show structure the 17-frame regime doesn't.** No per-video bundle has been generated for the `panda_longctx_1000v` series; the script supports it (`scripts/analyze_per_video_tta_gain.py` is series-agnostic).
- **PSNR may be the wrong axis.** TTA visibly changes Aes (LoRA: +0.047) and Dyn (+0.031) on the population (Table 1 of `2026-06-08_headline_1000v.md`); PSNR is insensitive to those shifts. Per-video ΔLPIPS / per-batch ΔFVD have not been factored out of the bundle's CSV.
- **VBench `dynamic_degree` (semantic motion) may capture cohort that RAFT mean-flow misses.** Backfill on NOPROMPT methods is queued (§5.5) and would provide the per-method `dynamic_degree` column.
- **Richer per-video features unmeasured.** New extraction pipeline (CLIP image↔text alignment mean/var/min, DINOv2 temporal-L2, PySceneDetect cuts + RGB-histogram backup, Laplacian-variance sharpness, RGB-histogram entropy) implemented but never run on cluster (see §5.1).
- **Diffusion-loss OOD relative to LongCat-Video itself.** New hypothesis from the parallel literature pass (commit `03d1a03`); implementation in flight, not yet wired to a runner.

## 5. Implemented but not run

Per-row format: file path(s), commit, what it would tell us, current status.

### 5.1 Per-video feature-extraction battery + correlation pipeline

- Files: `scripts/extract_video_features_for_tta.py` (introduced `187751c`, Tier-3 gen-target hotfix `28a57fc`), `scripts/correlate_tta_gain_with_features.py` (`187751c`), `scripts/sbatch/run_extract_video_features.sbatch`, `scripts/sbatch/run_correlate_tta_gain.sbatch`, `scripts/sbatch/submit_per_video_feature_pipeline.sh` (all `e090d20`).
- Tells us: whether any of the Tier-1 features (CLIP image↔text mean/var/min, DINO temporal-L2, scene cuts, Laplacian variance, RGB-hist entropy) clears the |ρ| ≥ 0.2 for ≥ 2 methods bar that would license a deployment-time selection rule.
- Status: gated on cluster environment availability (default `(base)` conda lacks torch; activation note in ANALYSIS_LOG 2026-06-09 hotfix entry). No outputs in `per_video_analysis/2026-06-09/` other than the gain bundle.

### 5.2 Long-horizon per-video analysis

- Files: `scripts/analyze_per_video_tta_gain.py` (`5d92733`) — same script that produced the 2026-06-09 bundle for `panda_1000v_standard`. Series-agnostic via `--series-path` / `--tinylora-series-path`.
- Tells us: whether the 17-frame saturation generalises to 76-frame, or whether long-horizon opens a structured winner/loser split. Long-horizon is the only 1000v regime where AdaSteer and LoRA visibly diverge at the population level (Subj 0.775 vs 0.757; Table 3 of `2026-06-08_headline_1000v.md`).
- Status: never invoked against `sweep_experiment/results/panda_longctx_1000v/` (no bundle under `per_video_analysis/`). Analysis-only — no new cluster compute.

### 5.3 NOPROMPT sweep close-out (UCF + TinyLoRA portion)

- Files: `sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh` (commit `16c1532`).
- Tells us: whether the "TTA caption is a noise channel" finding (§2.2) holds on UCF (where captions are class-y) and on the TinyLoRA family.
- Status: per ANALYSIS_LOG 2026-06-09 mid-day entry, 35/80 jobs done, 31 running, 15 pending at the time of the last status update — only the Panda × {ADA, LoRA_R8} portion has merged.

### 5.4 NOPROMPT × retrieval sweep, Panda 25K pool

- Files: `sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh` (commit `b67e8cb`).
- Tells us: whether the K-fold neighbour-caption diversity in batch-retrieval TTA carries any signal beyond the neighbour-video visual diversity.
- Status: hard-gated on the Panda 25K segment-pool build (INDEX.md row 2, job `10619044` running at last status) and the subsequent caption-embedding precompute (row 3). Not eligible to launch until both complete.

### 5.5 VBench backfill on NOPROMPT methods

- Files: `scripts/run_vbench_backfill.py`, `sweep_experiment/sbatch/run_vbench_backfill.sbatch`, `scripts/submit_vbench_backfill_all.sh` (pipeline introduced `85d0d70` + fixes `4cf8b57` / `4aba71f` / `514237f`).
- Tells us: the remaining 4 VBench dimensions (Motn, Dyn, IQ, Flick) for the NOPROMPT rows so the partial table at `paper_tables/2026-06-09_panda_std_with_noprompt_partial.md` becomes a drop-in replacement for Table 1 of the headline paper.
- Status: per `2026-06-09_panda_std_with_noprompt_partial.md` "What remains for this regime" — pending alongside the UCF + TinyLoRA NOPROMPT chunks (§5.3).

### 5.6 Diffusion-OOD hypothesis (parallel workstream)

- Files: covered in `HYPOTHESES_per_video_tta_suitability_2026-06-09.md` (commit `03d1a03`); no runner / sbatch file in the inventory yet.
- Tells us: whether per-video diffusion loss vs the base LongCat-Video model predicts TTA suitability — a model-relative OOD signal rather than a content-feature signal.
- Status: implementation in flight as of 2026-06-09 evening; not run.

### 5.7 Standard-horizon Panda retrieval (paper-blocking, prerequisite for §5.4)

- Files: `sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh` (cancelled 2026-06-08 against the 2K pool, INDEX.md row 4; will rerun against the 25K pool).
- Tells us: whether batch retrieval changes population-level metrics on Panda (UCF retrieval was uninformative due to class-block layout — Table 2 of `2026-06-08_headline_1000v.md`).
- Status: gated on the same 25K-pool build as §5.4.

## 6. Statements the paper can make today

- **Population-level saturation at Panda 1000v / 480p / 17-frame standard horizon.** AdaSteer, LoRA r=8, and TinyLoRA r=2 all land within 0.1 PSNR / 8 FVD of NOTTA. *Cite:* `paper_tables/2026-06-08_headline_1000v.md` Table 1.
- **TTA-time caption is a noise channel for Panda standard horizon.** Δ within ±0.01 PSNR / ±4 FVD for both ADA and LoRA r=8 when the TTA caption is dropped; per-video ρ(Δ, caption length) ≈ −0.02 for NOPROMPT variants. *Cite:* `paper_tables/2026-06-09_panda_std_with_noprompt_partial.md` + `per_video_analysis/2026-06-09/summary.md`.
- **Per-video gains are uncorrelated with motion / baseline PSNR / caption length at 17-frame.** |Spearman ρ| ≤ 0.09 across 18 method × feature cells. *Cite:* `per_video_analysis/2026-06-09/summary.md`.
- **TinyLoRA is the most stable TTA family on this regime.** ~95% of clips within ±0.5 dB ΔPSNR (vs ~81% for ADA, ~94% for LoRA r=8). *Cite:* same per-video summary.
- **LoRA r=8 has one catastrophic-failure video (`panda_0098`, 44.55 → 22.16 dB) responsible for ≈30% of its aggregate negative ΔPSNR bias.** Worth a footnote, not a section. *Cite:* same per-video summary, `LORA_R8_TTA` loser table.

## 7. Recommended next experimental wave

Priority order when cluster returns:

1. **Long-horizon per-video analysis** (§5.2). Analysis-only, no chunk-level compute. Cheapest experiment that could change the per-video narrative.
2. **Feature-extraction pipeline + diffusion-OOD experiment in parallel** (§5.1 + §5.6). `submit_per_video_feature_pipeline.sh` is the existing wrapper; diffusion-OOD needs to be slotted in once its runner exists.
3. **NOPROMPT sweep close-out** (§5.3). UCF + TinyLoRA standard horizon.
4. **VBench backfill on NOPROMPT methods** (§5.5). Lets `2026-06-09_panda_std_with_noprompt_partial.md` graduate from "partial" to a paper row.
5. **Standard-horizon Panda retrieval, then retrieval × NOPROMPT on the 25K pool** (§5.7 → §5.4). Only worth firing if (1)–(2) produce a usable per-video subsection — otherwise it duplicates the saturation story at higher cost.

## 8. Operational lessons recorded above (do not relitigate)

- At Panda 1000v / 480p / 17-frame, the per-video noise floor for ΔPSNR is ~±0.5 dB (81–95% of 999 clips land in this band depending on the method). Future agents debating "is a +0.3 dB chunk-level effect significant?" — no. The per-chunk noise floor on 100-video subsamples is ≈ 0.5 dB; the ADA_NOPROMPT chunk-0 "+0.68 dB" episode in ANALYSIS_LOG is the worked example.
- NOPROMPT ablation is genuinely caption-blind on Panda standard horizon (caption-length ρ ≈ −0.02 for both NOPROMPT variants). The flag wiring is verified end-to-end (ANALYSIS_LOG 2026-06-09 retrieval-NOPROMPT entry, "audit of `tta_caption_for` coverage").
- PSNR alone is insufficient at 1000v scale. The population-level effects we *can* see (LoRA +0.047 Aes, +0.031 Dyn, −0.034 IQ — Table 1) live in VBench / LPIPS / FVD, not in PSNR. Future per-video analyses should report ΔLPIPS and per-batch ΔFVD alongside ΔPSNR.
