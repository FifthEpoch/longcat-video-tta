# Plan: optimal per-video TTA gating strategy

**Date:** 2026-06-11
**Status:** PLAN — AUTHORISED 2026-06-11 (Phases 0–3 green-lit; Phase 4 requires explicit authorisation after RECOMMENDATION.md)
**Companion documents:**
- [REVIEW_per_video_tta_suitability_2026-06-09.md](REVIEW_per_video_tta_suitability_2026-06-09.md) — what we know
- [HYPOTHESES_per_video_tta_suitability_2026-06-09.md](HYPOTHESES_per_video_tta_suitability_2026-06-09.md) — literature-grounded hypotheses (12 hypotheses across themes A/B/C/D/E/G; landed in commit `03d1a03`)
- Diffusion-OOD implementation — `scripts/compute_diffusion_ood_score.py` (commit `dc115e7`)
- Tier-1 feature extractor — `scripts/extract_video_features_for_tta.py` + correlation pipeline (`scripts/correlate_tta_gain_with_features.py`, sbatch wrapper `scripts/sbatch/submit_per_video_feature_pipeline.sh`, commits `187751c` / `28a57fc` / `e090d20`)
- Per-video gain bundle — `sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv` (N = 999; produced by `scripts/analyze_per_video_tta_gain.py`)

## TL;DR

**Goal.** Find a function `g: F(v) → decision` that, given a per-video feature vector `F(v)`, decides whether to apply TTA (binary), which TTA method to use (routing), or how much to trust the predicted gain (continuous), and maximises population-level ΔPSNR / ΔLPIPS subject to a compute budget. **Approach.** A five-phase protocol: (0) collect the already-scaffolded Tier-1 / Tier-2 feature CSVs on cluster return, (1) score every single feature with both Spearman ρ and a threshold sweep, (2) train multivariate gates with leave-one-chunk-out cross-validation, (3) plot the cost-aware Pareto frontier and pick the knee, (4) re-test the winner on long-horizon. **Decision criterion.** A gate ships only if held-out gain exceeds the per-video noise floor (≥ 0.05 PSNR / ≥ 0.005 LPIPS), coverage ≥ 50 %, and feature compute ≤ 30 min per 999 videos. If no strategy clears all three, the honest paper claim is "no per-video feature gates TTA usefully at this scale; gating awaits the long-horizon regime" — fully consistent with Story A in the existing REVIEW.

## 1. Problem statement

### 1.1 Definition of "gating"

A **gating strategy** is a function `g : F(v) → decision(v)` that takes a per-video feature vector `F(v) ∈ R^d` and returns either a binary apply/skip flag, a TTA-method label, or a continuous predicted-gain score. We evaluate `g` against the existing per-video gain bundle at `sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv` (N = 999, baseline = NOTTA, six TTA methods: `ADA`, `ADA_NOPROMPT`, `LORA_R8_TTA`, `LORA_R8_TTA_NOPROMPT`, `TL_BARE_R2`, `TL_TIED_R2`).

### 1.2 Three families of gating decisions

- **G1 — Binary apply/skip.** Apply method `m` only when `g_m(v) > 0`. Saves compute when `g_m` predicts net-negative or null ΔPSNR.
- **G2 — Method routing.** Among the six available TTA methods (which separate into three between-family clusters: ADA family, LoRA-r8 family, TinyLoRA-r2 family), pick the per-video best by `argmax_m g_m(v)`. Note: the existing within-family overlap is high (e.g. `panda_0461` is a top-10 winner for both `ADA` and `ADA_NOPROMPT`; `panda_0098` is the catastrophic loser for both LoRA-r8 variants), so meaningful G2 routing is **between** the three families, not within them.
- **G3 — Continuous gain prediction.** Predict `ΔPSNR_m(v) ∈ R` directly; use as a confidence score or as the weight in a mixture-of-methods ensemble rather than as a hard gate.

### 1.3 Evaluation metrics for any gate `g`

For ΔPSNR and ΔLPIPS independently (LPIPS is "lower is better"; we report `−ΔLPIPS` so all axes have the "higher = better" convention):

1. **Gated mean Δ** = mean ΔPSNR over videos with `g(v) = ON`.
2. **Coverage** = `|{v : g(v) = ON}| / |V|`. A perfect gate that keeps only one video is useless; coverage matters.
3. **Population gain delivered** = gated mean Δ × coverage. This is what changes if we deploy `g` over the full population.
4. **Cost-aware Pareto frontier.** Plot all candidate gates on `x = (feature compute + coverage × TTA compute)` vs `y = population gain delivered`. The Pareto frontier is the set of gates where you cannot simultaneously increase `y` and decrease `x`. The recommended strategy is the **knee** of that frontier, subject to the criteria in §3.4.

### 1.4 Why this matters

REVIEW §2 establishes population-level saturation at Panda 1000v / 480p / 17-frame standard horizon: AdaSteer / LoRA-r8 / TinyLoRA-r2 all land within 0.1 PSNR / 8 FVD of NOTTA. Per-video ΔPSNR tails *are* non-degenerate (LoRA-r8 has 7 winners > +1.0 dB, 21 losers < −1.0 dB, and one catastrophic outlier `panda_0098` at −22.4 dB), but the three features already correlated against ΔPSNR (RAFT mean-flow, baseline PSNR, caption words) are all null at |Spearman ρ| ≤ 0.09. A useful per-video gate would (i) license a deployment-time selection rule that converts the saturated headline into a real positive net effect, and (ii) flag the catastrophic-failure mechanism so it can be predicted and avoided at inference time.

## 2. Master feature menu

20 candidate features, organised by compute tier. The "Existing" column reflects what is already implemented (column name, source script, hypothesis ID where applicable). Cost estimates are per 999-video run on H200, sourced from the HYPOTHESES doc §3 wherever a hypothesis is cited.

### 2.1 Already-extracted baselines (sanity-check; mean-flow / baseline-PSNR / caption-words known null)

| # | Feature | Hypothesis (1-line) | Tier | Status | Expected ρ sign (vs ΔPSNR) | Cost / 999 vids |
|---:|---|---|---|---|---|---|
| 1 | `mean_flow` (RAFT) | H1 — motion magnitude predicts ΔPSNR. **Known null** (max \|ρ\| = 0.088, REVIEW §3). | T1 | ready (`datasets/panda_1000_480p/dynamic_degree.json`) | ≈ 0 | 0 (extracted) |
| 2 | `baseline_psnr` (NOTTA PSNR) | H2 — hard-baseline videos benefit more. **Known null on rank** (LoRA Pearson −0.143 dominated by `panda_0098`). | T1 | ready (`per_video_gains.csv` `NOTTA_psnr`) | ≈ 0 (LoRA-r8 weakly −) | 0 |
| 3 | `caption_words` | H3 — verbosity drives gains. **Known null** (max \|ρ\| = 0.062). | T1 | ready (`metadata.csv`) | ≈ 0 | 0 |

### 2.2 Tier-1, cheap, runnable on a single H200 in < 5 s/video

| # | Feature | Hypothesis (1-line) | Tier | Status | Expected ρ sign | Cost / 999 vids |
|---:|---|---|---|---|---|---|
| 4 | `cut_count_pyscenedetect` + `cut_count_histogram` + `cut_density_per_frame` | H-T1-6 — PySceneDetect cuts predict the LoRA-r8 catastrophic tail (Fisher OR ≥ 3 conditional on \|Δ\| > 1 dB). | T1 | ready (`extract_video_features_for_tta.py`, not yet executed on cluster) | Fisher OR ≥ 3 vs LORA_R8 tail | bundled in ~25 min of the existing feature pipeline |
| 5 | `clip_text_image_sim_mean` + `clip_text_image_sim_var` + `clip_text_image_sim_min` | H-T1-5 — *min* per-frame CLIP score (weakest-link caption fidelity) gates caption-using methods; \|ρ\| > 0.15 with prompt-vs-NOPROMPT gap > 0.05. | T1 | ready (feature extracted; correlation pass pending) | + on caption-using methods, ≈ 0 on NOPROMPT | bundled in same ~25 min |
| 6 | `dino_temporal_l2_mean` | Theme D — DINOv2 temporal change as a semantic-motion proxy where RAFT mean-flow is null. | T1 | ready (extractor) | uncertain (probe) | bundled |
| 7 | `laplacian_variance_mean` | Theme D — frame sharpness; complement to FFT-high-freq probe (#9). | T1 | ready (extractor) | uncertain (probe) | bundled |
| 8 | `rgb_histogram_entropy_mean` | Theme D — colour-entropy as a complexity proxy. | T1 | ready (extractor) | uncertain (probe) | bundled |
| 9 | `hf_energy_ratio` (3D-spatiotemporal FFT high-freq ratio) | H-T1-3 — diffusion is spectral autoregression; LoRA has spare capacity for the under-fit high-freq band; AdaSteer does not. | T1 | **not implemented** (CPU FFT; design fully specified in HYPOTHESES §H-T1-3) | + LoRA-class, ≈ 0 AdaSteer | ~10 min |
| 10 | `flow_max`, `flow_entropy`, `flow_max_over_mean` | H-T1-4 — flow *distribution shape* (concentration) gates ADA where mean-flow is null. | T1 | scaffolded (RAFT pipeline deployed; max/entropy/ratio not stored) | + ADA family, ≈ 0 LoRA | ~30 min (RAFT pipeline rerun) |
| 11 | `bpp_h264`, `bpp_png_avg` | H-T1-2 — lossless-compression bpp confounds pixel-/latent-space generative likelihoods (Serrà et al. ICLR 2020). Required as a partial-out covariate for the Tier-2 OOD scores. | T1 | **not implemented** (one `ffprobe` + per-frame `cv2.imencode`; < 0.5 s/video) | ρ ≥ 0.4 with diffusion-OOD; \|ρ\| ≤ 0.15 with ΔPSNR | ~5 min |
| 12 | `rec_err_l1`, `rec_err_lpips` (LongCat-VAE round-trip) | H-T1-1 — latent-space typicality; predicts \|ΔPSNR\| asymmetrically on all six methods and signed ΔPSNR on LoRA-class. | T1 | **not implemented** (one VAE encode + decode per video; VAE wrappers exist in `delta_experiment/scripts/common.py`) | + \|ΔPSNR\| all methods; + signed ΔPSNR LoRA-class | ~25 min |

### 2.3 Tier-2, single LongCat-Video forward pass (~30 s/video on H200), already partially scaffolded by `compute_diffusion_ood_score.py`

| # | Feature | Hypothesis (1-line) | Tier | Status | Expected ρ sign | Cost / 999 vids |
|---:|---|---|---|---|---|---|
| 13 | `diffusion_loss_caption_t{100,500,900}` + `mean_diffusion_loss_caption` | H-T2-1 — caption-conditioned flow-matching MSE as an OOD score; positive ρ with ΔPSNR on LoRA-class methods (after partialling out `bpp` per #11). | T2 | scaffolded — script committed at `dc115e7`, not yet run on cluster | + LoRA-class | ~2-3 h total (entire OOD bundle) |
| 14 | `diffusion_loss_uncond_t{100,500,900}` + `mean_diffusion_loss_uncond` + `delta_caption_minus_uncond` | H-T2-1 unconditional + H-T2-3 lite — unconditional OOD baseline; the `delta` column is a coarse CFG-gap proxy. | T2 | scaffolded — same OOD CSV (commit `dc115e7`) | comparator for #13; partial CFG-gap signal | bundled in #13 cost |
| 15 | `latent_norm_mean`, `latent_norm_std`, `latent_kurtosis` | Theme B latent moments — already emitted by the OOD scorer for free. | T2 | scaffolded — same OOD CSV | uncertain (probe) | bundled |
| 16 | `loss_var_t` (across-t variance of flow-matching MSE) | H-T2-5 — diffusion-loss variance across noise levels is the generative analogue of EATA's reliable-sample entropy signal; + ρ on LoRA-class. | T2 | derivable from the same CSV — needs an analysis pass, not a new forward pass | + LoRA-class | ~0 (post-process) |
| 17 | `score_norm_t*` (squared norm of predicted velocity) | H-T2-2 — SCOPED-style score-norm geometry at fixed mid-noise level; distinguishes score-magnitude from loss-mismatch. | T2 | **not implemented** but bolts onto the existing OOD forward pass at near-zero marginal cost (record `‖v‖²` alongside the MSE) | + LoRA-class, ≈ 0 ADA | ~free if scheduled alongside #13 |
| 18 | `cfg_gap` (full per-(t, ε) classifier-free-guidance gap) | H-T2-3 — dynamic-CFG papers show optimal CFG scale is strongly per-sample; the gap directly measures per-video caption informativeness. | T2 | **not implemented** — needs an extra unconditional `ε` call per noise sample; ~+50% over #13's cost | + caption-using ≥ +0.15; ≈ 0 NOPROMPT | ~1 h additional |
| 19 | `lid_flipd` (FLIPD local intrinsic dimension via Hutchinson Hessian-trace) | H-T2-4 — high-LID neighbourhoods are where LoRA's extra rank capacity is best spent. | T2 | **not implemented** — +30% over #13's cost; closed-form per Kamkari et al. NeurIPS 2024 eq. 7 | + LoRA-class, ≈ 0 ADA | ~45 min additional |

### 2.4 Tier-3, mini-TTA probes (> 1 min/video on H200)

| # | Feature | Hypothesis (1-line) | Tier | Status | Expected ρ sign | Cost / 999 vids |
|---:|---|---|---|---|---|---|
| 20 | `grad_norm_θ0` (one forward + one backward at unadapted weights) | H-T3-1 — SAR-style large-gradient predictor of asymmetric LoRA tails; + \|ΔPSNR\|, − signed ΔPSNR on LoRA-r8. | T3 | **not implemented** — reuses LoRA / TinyLoRA / ADA optimiser scaffolds (`lora_experiment/scripts/run_lora_tta.py`, `delta_experiment/scripts/run_delta_a.py`, `delta_experiment/scripts/run_tinylora.py`) | + \|ΔPSNR\| LoRA-class; − signed ΔPSNR LoRA-r8 | ~30 min (or ~3 min on stratified 100-video sample per HYPOTHESES open question §4) |
| 21 | `single_step_loss_drop` (one Adam step on LoRA-r8 adapter, then re-eval loss) | H-T3-2 — DreamBooth-style overfit detector for the LoRA-r8 catastrophic tail; specifically predicts `panda_0098`-class failures. | T3 | **not implemented** — reuses `run_lora_tta.py` optimiser setup | strongly − with signed `LORA_R8_TTA` ΔPSNR (target ρ ≤ −0.20) | ~30 min full / ~3 min stratified |

### 2.5 Implementation decision for Phase 0

**Implement now (high paper-leverage and cheap):** #11 (bpp confound — required to interpret #13–#19), #12 (VAE round-trip — orthogonal to existing features), #9 (HF FFT ratio — clean LoRA-vs-ADA differential prediction), #17 (score-norm — free if bolted onto #13), #16 (loss-variance — free post-processing of #13).

**Tier-3 mini-TTA probes — scheduled for Phase 0 per user authorisation 2026-06-11.** #20 (`grad_norm_θ0`) and #21 (`single_step_loss_drop`) are now part of the Phase-0 deliverables per Decision 4 in §8 (user's explicit "test all hypotheses" instruction). These two features are the closest to the actual TTA loss surface and have the highest expected signal of any feature in the menu. Cost: ~2 extra GPU hours per 999-video run. **Status: scheduled for Phase 0 per user authorisation 2026-06-11.** Implementation note: this requires either extending `submit_per_video_feature_pipeline.sh` to schedule a third sbatch (`run_compute_tier3_probes.sbatch`) or implementing those probes inline within an existing extractor; that wrapper is a small follow-up implementation task that must land before Phase 0 can run end-to-end (flagged here; to be implemented in a separate commit).

**Defer to a follow-up wave (gated on Phase 1 results):** #18 (CFG-gap full — moderate cost; only worth it if #14's `delta` proxy shows a non-null signal), #19 (FLIPD — only if #13/#17 don't already explain LoRA tails).

## 3. Experimental protocol

All outputs land under `sweep_experiment/reports/gating_experiment/`. The directory does not yet exist and will be created by the first script that writes into it.

### 3.1 Phase 0 — data collection (≤ 1 wallclock day post-cluster-maintenance, GPU)

**Inputs:** Panda-70M 999-video intersection at `datasets/panda_1000_480p/` (480p / 17-frame standard horizon — defer long-horizon to Phase 4). Per-video gain bundle at `per_video_analysis/2026-06-09/per_video_gains.csv` (already on disk; 999 rows; columns `<METHOD>_{psnr,ssim,lpips,dpsnr,dssim,dlpips}` for the six TTA methods plus `NOTTA_*`). Tier-1 features auto-extracted from the TTA-visible window `[0, 48)` per the audit in `extract_video_features_for_tta.py` lines 1–80.

**Outputs (cluster paths under `sweep_experiment/reports/per_video_analysis/2026-06-09/`):**

| File | Schema (columns) | Producer | Status |
|---|---|---|---|
| `video_features.csv` | `video_id, n_frames_used, tta_visible_range, gen_target_range, caption, cut_count_pyscenedetect, cut_count_histogram, cut_density_per_frame, clip_text_image_sim_{mean,var,min}, dino_temporal_l2_mean, laplacian_variance_mean, rgb_histogram_entropy_mean, dino_tta_vs_genregion_sim, clip_text_genregion_sim_mean, clip_model, dino_model, hist_bins_per_channel, hist_bhattacharyya_thresh` | `extract_video_features_for_tta.py` via `submit_per_video_feature_pipeline.sh` (existing) | scaffolded, ready to run |
| `diffusion_ood_scores.csv` | `video_id, diffusion_loss_caption_t{100,500,900}, diffusion_loss_uncond_t{100,500,900}, mean_diffusion_loss_caption, mean_diffusion_loss_uncond, delta_caption_minus_uncond, latent_norm_mean, latent_norm_std, latent_kurtosis, n_visible_frames, n_gen_target_frames, seed` (16 columns) | `compute_diffusion_ood_score.py` (commit `dc115e7`) via the same submit wrapper | scaffolded, ready to run |
| `bpp_features.csv` (NEW; #11) | `video_id, bpp_h264, bpp_png_avg, n_frames` | NEW script `scripts/extract_bpp_features.py` (≤ 80 LOC; one `ffprobe` + per-frame `cv2.imencode`) | needs writing |
| `vae_recerr_features.csv` (NEW; #12) | `video_id, rec_err_l1, rec_err_lpips, n_visible_frames` | NEW script `scripts/extract_vae_recerr_features.py` (reuses `load_longcat_components` from `delta_experiment/scripts/common.py`; one VAE forward + decode per video) | needs writing |
| `fft_features.csv` (NEW; #9) | `video_id, hf_energy_ratio_3d, hf_energy_ratio_spatial_only, n_visible_frames` | NEW script `scripts/extract_fft_features.py` (CPU 3D real FFT on luma channel) | needs writing |
| `score_norm_features.csv` (NEW; #17 — DERIVED, free) | `video_id, score_norm_t{100,500,900}, mean_score_norm` | Optional ≤ 30-line patch to `compute_diffusion_ood_score.py`: record `‖pred_v‖²` alongside the MSE | needs writing (single PR; bolts onto the existing OOD scorer) |
| `loss_var_features.csv` (NEW; #16 — DERIVED, free) | `video_id, loss_var_caption, loss_var_uncond` | NEW script `scripts/derive_loss_variance.py` (reads `diffusion_ood_scores.csv`; pure pandas) | needs writing |
| `tier3_probe_features.csv` (NEW; #20/#21 — Phase-0 per Decision 4 in §8) | `video_id, grad_norm_theta0, single_step_loss_drop, n_visible_frames, seed` | NEW `scripts/compute_tier3_probes.py` (reuses LoRA-r8 optimiser scaffolds from `lora_experiment/scripts/run_lora_tta.py` / `delta_experiment/scripts/run_delta_a.py`); requires a `run_compute_tier3_probes.sbatch` wrapper (or inline integration with an existing extractor) before Phase 0 can run end-to-end — small follow-up implementation task per Decision 4 in §8 | needs writing |

**Estimated wallclock:** Stage 1a (`extract_video_features_for_tta.py`) ~25 min on one H200; stage 1b (`compute_diffusion_ood_score.py` + score-norm patch) ~2–3 h on one H200 (per the script's own docstring on noise-sample schedule); new T1 scripts (#11/#12/#9) combined ~40 min on one CPU + one H200; loss-variance derivation ~5 s; Tier-3 probes (#20/#21) ~2 GPU hours per 999-video run (per Decision 4 in §8). Total ≤ 1 wallclock day even with cluster queue.

**Concurrency:** `submit_per_video_feature_pipeline.sh` already fans 1a and 1b out in parallel and chains the correlation pass behind both with `afterok:1a:1b`. The new T1 scripts in #11/#12/#9 can be added as siblings to 1a in a single PR — Phase-0 ask: "add three new sbatch wrappers + extend the submit script". The Tier-3 probes (#20/#21) authorised under Decision 4 in §8 require a further `run_compute_tier3_probes.sbatch` wrapper (or inline integration with an existing extractor); the submit script must be extended to schedule it alongside the other Phase-0 jobs. Do not modify the existing sbatch wrappers per the "no code under scripts/ or sbatch/" constraint of this *planning* PR (the Tier-3 wrapper lands in a separate implementation commit).

### 3.2 Phase 1 — univariate gating (≤ 1 day analysis, CPU)

**Inputs:** All Phase-0 CSVs joined on `video_id` with `per_video_gains.csv`. Treat each scalar column as a candidate feature (the Tier-1 / Tier-2 family columns expand to ~30 individual scalars; we will report on a per-column basis and group by family in the leaderboard).

**For each (feature, method, metric) cell:**

1. Compute **Spearman ρ** and **Pearson r** (Pearson catches linear relationships ρ misses; ρ is robust to monotonic non-linearities and to outliers like `panda_0098`).
2. Sweep threshold `τ` over the feature's empirical quantiles `{5 %, 10 %, 25 %, 50 %, 75 %, 90 %, 95 %}` in both directions (`g = ON if feature ≥ τ` and `g = ON if feature ≤ τ`).
3. For each `(τ, direction)` pair, compute `(gated mean Δ, coverage)`.
4. Record the **best τ** per `(feature, method, metric)` cell, subject to coverage ≥ 25 % (anything below is over-selective).

**Multiple-comparison correction.** Per Decision 3 in §8, **both Bonferroni α/192 (primary) and Benjamini–Hochberg FDR at q = 0.1 (secondary) thresholds are reported in every Spearman ρ output table.** With ~20 features expanding to ~30 scalar columns × 6 methods × 2 metrics ≈ 360 cells (the user-stated 192 was a smaller upper bound on the feature column count; we use the same Bonferroni convention but recompute the critical value against the actual column count we end up with). For n = 999, two-tailed α = 0.05 / 360 = 1.4 × 10⁻⁴, the critical |ρ| ≈ 0.121. Per the user-stated 192-cell convention, anything with **|ρ| < 0.13 is not paper-worthy as a single-feature gate** — we adopt that bar verbatim. Bonferroni is the conservative headline that licenses any "feature X is a real predictor" paper claim; BH-FDR shows the texture and surfaces additional candidates worth follow-up.

**Outputs (`sweep_experiment/reports/gating_experiment/phase1/`):**

| File | Schema | Producer | Status |
|---|---|---|---|
| `gating_univariate_panda_std.csv` | `feature, method, metric, n, spearman_rho, spearman_pvalue, pearson_r, pearson_pvalue, best_tau, best_direction, best_coverage, best_gated_mean_delta, bonferroni_significant, bh_fdr_q_0p1_significant, permutation_rho_p99, permutation_rho_p95` | NEW `scripts/analyze_gating_univariate.py` (does not exist) | needs writing |
| `gating_univariate_leaderboard.md` | top-10 `(feature, method)` cells by gated mean Δ at coverage ≥ 50 % (one table per metric) + plots of the top-3 per metric | same script | needs writing |
| `gating_univariate_panda_std_leaderboard.png` | scatter of gated mean Δ vs coverage; one point per `(feature, method, metric)` cell | same script | needs writing |

### 3.3 Phase 2 — multivariate gating (≤ 1 day analysis, CPU)

**Cross-validation scheme.** The 999 videos partition cleanly into ten chunks of ~100 by their `chunk_idx` in the underlying `chunk_*/summary.json` files. Use **leave-one-chunk-out** CV (10 folds) for all multivariate models — this is genuinely held-out evaluation (no chunk's video contributes to both train and test).

**For each method in {ADA, ADA_NOPROMPT, LORA_R8_TTA, LORA_R8_TTA_NOPROMPT, TL_BARE_R2, TL_TIED_R2}:**

1. **Linear logistic regression** (sklearn `LogisticRegressionCV(Cs=5, scoring='roc_auc', cv=10)`): predict `sign(ΔPSNR > 0)`. Held-out AUC + held-out gated mean Δ at the top-K coverage choices `K ∈ {50 %, 75 %, 90 %}`. Report feature coefficients normalised by feature std.
2. **Shallow gradient-boosted tree** (`HistGradientBoostingClassifier(max_depth=3, max_iter=50, learning_rate=0.1)`): same target, same CV. Report permutation feature importances (sklearn `permutation_importance` on the held-out fold).
3. **Per-method linear regressor** (`RidgeCV`): predict ΔPSNR magnitude directly. Report held-out R² and held-out gated mean Δ when used as the gate score.

All three model families are re-run separately for ΔLPIPS as the target. **Both Bonferroni α/192 and BH-FDR q=0.1 thresholds (per Decision 3 in §8) are reported alongside any Spearman ρ / Pearson r feature-importance statistics emitted by this phase**, mirroring the §3.2 convention.

**Outputs (`sweep_experiment/reports/gating_experiment/phase2/`):**

| File | Schema | Producer | Status |
|---|---|---|---|
| `gating_multivariate_panda_std.csv` | `method, metric, model_family, fold, train_auc, heldout_auc, heldout_r2, heldout_gated_mean_delta_k50, heldout_gated_mean_delta_k75, heldout_gated_mean_delta_k90, top_feature_1, top_feature_1_importance, top_feature_2, top_feature_2_importance, top_feature_3, top_feature_3_importance` | NEW `scripts/analyze_gating_multivariate.py` (does not exist) | needs writing |
| `gating_multivariate_feature_importance_<method>.png` | per-method bar chart of permutation importances (mean across folds, error bars across folds) | same script | needs writing |
| `gating_multivariate_summary.md` | per-method table: best model family, held-out AUC / R², coverage / gated-Δ trade-off curve, top-3 features | same script | needs writing |

### 3.4 Phase 3 — cost-aware Pareto + recommendation (≤ 0.5 day, CPU)

**For every gating strategy `g` from Phase 1 (univariate τ-threshold) and Phase 2 (multivariate score with top-K cutoff):**

- **x-axis:** total compute = feature compute (extraction cost from §2's cost column, per video) + coverage × TTA compute (TTA cost per video sourced from `chunk_*/summary.json` runtime field; if absent, the LoRA-r8 TTA wallclock from `submit_standard_1000v_chunked.sh` headers, ~5 min/video on H200).
- **y-axis:** population gain delivered = gated mean Δ × coverage.

Plot all strategies on `(x, y)`. Compute the Pareto frontier (the set of strategies with no other strategy that simultaneously has lower x and higher y). Identify the **knee** by the standard distance-to-the-corner heuristic.

**Recommended strategy criteria** (all three required):

1. **|held-out gain| > per-video noise floor:** ≥ 0.05 PSNR *or* ≥ 0.005 LPIPS at the strategy's chosen coverage. (Per REVIEW §8 operational lesson, the per-video noise floor for ΔPSNR is ~±0.5 dB; 0.05 PSNR is 10× below that, i.e. requires the *aggregated* population effect to be 10× the per-video noise — the standard `√n`-scaling argument for n = 500-ish gated videos.)
2. **Coverage ≥ 50 %:** so the gate is genuinely useful and not pathologically selective.
3. **Feature compute ≤ 30 min per 999 videos:** so the gate is cheaper than running TTA on a single chunk's worth of videos (~10 min/chunk × 1 chunk = ~10 min; we allow 3× headroom).

**Three outcome cases for §5:** "clean win" (all three criteria met by ≥ 1 strategy), "partial win" (criteria met for one method family or one metric only), "no win" (no strategy clears all three).

**Compute-saved interpretation (per Decision 2 in §8):** "compute saved" is computed **strictly against the immediate 999-video run with no speculative extrapolation to unrun benchmarks**. The measured 999-video savings is itself the headline transferable claim — future researchers running similar-sized benchmarks would see similar relative savings — but the plan does not quote scaled numbers for specific benchmarks we have not actually executed. Reviewer-defensible.

**Outputs (`sweep_experiment/reports/gating_experiment/phase3/`):**

| File | Schema | Producer | Status |
|---|---|---|---|
| `gating_pareto_panda_std.csv` | `strategy_id, family (G1/G2/G3), description, method, metric, coverage, gated_mean_delta, feature_compute_min, total_compute_min, is_pareto_optimal, is_knee, meets_gain_floor, meets_coverage_floor, meets_feature_compute_floor, meets_all_three` | NEW `scripts/build_gating_pareto.py` (does not exist) | needs writing |
| `gating_pareto_panda_std.png` | x = total compute (min), y = population gain delivered (PSNR or LPIPS); markers coloured by strategy family; Pareto frontier overlaid | same script | needs writing |
| `RECOMMENDATION.md` | the three outcome cases written out explicitly; the strategy(ies) that clear all three criteria (if any); the licensed paper claim under each case | same script | needs writing |

### 3.5 Phase 4 — long-horizon validation (post-Phase-3, conditional)

**Only fires if Phase 3 produces a `recommended` strategy** (case "clean win" or "partial win"). The plan ships even if Phase 3 produces "no win" — that result is the paper claim per §5 case 3.

**Authorisation gate (per Decision 1 in §8):** the 2026-06-11 user authorisation green-lights Phases 0–3 only. Phase 4 requires **explicit separate user authorisation after the Phase-3 `RECOMMENDATION.md` is reviewed** — that document is the human-in-the-loop gate. Long-horizon already shows a method-asymmetry signal at population level that standard horizon does not (Subj diverges 0.018 between AdaSteer and LoRA r=8 at 76-frame vs 0.005 at 28-frame; ref `paper_tables/2026-06-08_headline_1000v.md` Table 3), so Phase 4 is non-trivial and merits human review before firing.

**Inputs:** `sweep_experiment/results/panda_longctx_1000v/` (76-frame regime, all four methods done per INDEX.md row 5). Re-run `scripts/analyze_per_video_tta_gain.py` with `--series-path sweep_experiment/results/panda_longctx_1000v --output-dir sweep_experiment/reports/per_video_analysis/<DATE>_longctx` to produce the long-horizon per-video gain bundle (this is REVIEW §5.2's already-pending analysis-only task and incurs no new cluster compute).

**Analysis:** apply the Phase-3 recommended gate `g` (with its frozen τ / model weights) to the long-horizon feature CSVs (which need to be re-extracted by re-running the Phase-0 pipeline against the long-horizon `datasets/panda_longctx_*` videos — same scripts, different `--videos-dir`). Report held-out gated mean Δ on long-horizon ΔPSNR / ΔLPIPS.

**Hypothesis under test:** "long-horizon makes the gates cleaner because the model has more room to differentiate per-video TTA outcomes" (per the user prompt). Falsified if held-out gated mean Δ at long-horizon ≤ short-horizon held-out gated mean Δ; confirmed if long-horizon ≥ 2× short-horizon held-out gain.

**Output:** `sweep_experiment/reports/gating_experiment/phase4/RECOMMENDATION_longhorizon.md` — accept / reject the Phase-3 gate as cross-regime usable.

## 4. Sanity / falsification controls

Built into every phase; the scripts under §3 emit these as standard columns.

- **Permutation null.** For each `(feature, method, metric)` ρ in Phase 1, recompute ρ on 1000 random `video_id → feature` shufflings (paired permutation). The real ρ must exceed the 99th percentile of the shuffle distribution to be flagged `permutation_significant`. This is orthogonal to the Bonferroni / BH correction and catches subtle data-leakage / off-by-one errors that distributional tests miss.
- **Holdout discipline.** All Phase-2 multivariate gates are evaluated leave-one-chunk-out (10 folds × 100 videos held out per fold). Phase-3 Pareto plots use the held-out gated mean Δ — **never** the in-sample one. The Phase-1 univariate τ-sweep is in-sample (you cannot leave-one-chunk-out a scalar threshold without re-sweeping per fold, which we do separately and store as `best_tau_loco_mean`, `best_tau_loco_std` columns to document the variance).
- **Known-failure check.** Report `g(panda_0098)` for every gate. `panda_0098` is the LoRA-r8 catastrophic outlier (44.55 → 22.16 dB under `LORA_R8_TTA`; 30 % of the aggregate negative bias). For any gate intended to gate the LoRA-r8 family, `g(panda_0098) = OFF` is mandatory; a gate that fires ON for `panda_0098` is automatically rejected.
- **Known-winner check.** Report `g(v)` for `v ∈ {panda_0461, panda_0555, panda_0862, panda_0431}` — the cohort that appears in the top-10 winners across ≥ 2 method families (REVIEW §2.3). For a useful gate, all four should be `g = ON` for the method under which they are top winners. A gate that misses the universal-beneficiary cohort is suspect even if it has good aggregate metrics.

**These checks land as columns** `g_panda_0098`, `g_panda_0461`, `g_panda_0555`, `g_panda_0862`, `g_panda_0431` in `gating_pareto_panda_std.csv` from §3.4 (each is a per-strategy boolean).

## 5. What the recommendation looks like

Phase 3 produces one of three outcomes; the paper claim each licenses:

### Case 1 — clean win

A single strategy (or a small Pareto frontier) clears all three criteria of §3.4 with held-out gain ≥ 0.05 PSNR or ≥ 0.005 LPIPS at coverage ≥ 50 % and feature compute ≤ 30 min / 999 videos. Paper claim: "**At Panda 1000v / 480p / 17-frame standard horizon, per-video gating recovers a +X PSNR / −Y LPIPS effect under method M that the population-saturated headline (REVIEW §2.1) hides. The gate `g` uses features F and is cheaper than a single TTA chunk.**" This is the strongest possible per-video story for the AdaSteer paper.

### Case 2 — partial win

A strategy clears criteria (1) and (2) but fails (3) — i.e., the gate works but is too expensive. **Or** a strategy clears all three criteria but only for one method family (e.g. only for `LORA_R8_TTA` because the catastrophic-tail screener H-T3-2 turns out to dominate the Pareto frontier). Paper claim: "**A per-video gate exists for method family X at standard horizon, conditional on the Tier-2/T3 forward-pass infrastructure. The simpler Tier-1-only gate does not reach the noise floor.**" Footnote-worthy, not a section. Conditionally licenses Phase 4 on the partial-win method family.

### Case 3 — no win

No strategy clears all three criteria. Paper claim: "**No per-video feature provides a useful gate at this scale (N = 999, standard horizon); gating awaits the long-horizon regime where REVIEW §4 already identifies population-level method divergence (Subj 0.775 vs 0.757) that the standard horizon does not have.**" This is fully consistent with Story A in REVIEW and is the same negative-but-honest result the existing per-video summary points at. **Phase 4 does not fire in case 3 by default** — see Decision 1 in §8 (separate authorisation required after `RECOMMENDATION.md` review).

## 6. What this plan asks the user to authorise

1. **Phase 0 cluster jobs when cluster returns:**
   - Existing `submit_per_video_feature_pipeline.sh` (no edits required to the existing sbatch wrappers).
   - New T1 feature scripts: `scripts/extract_bpp_features.py` (#11), `scripts/extract_vae_recerr_features.py` (#12), `scripts/extract_fft_features.py` (#9). Each ≤ 100 LOC.
   - New ≤ 30-line patch to `compute_diffusion_ood_score.py` to record `‖pred_v‖²` alongside the loss MSE (#17 score-norm).
   - New `scripts/derive_loss_variance.py` (#16 — pure pandas, runs on CPU in seconds).
2. **Phase 1 / 2 / 3 analysis scripts** (all CPU, all local-laptop-runnable; no cluster jobs):
   - `scripts/analyze_gating_univariate.py`
   - `scripts/analyze_gating_multivariate.py`
   - `scripts/build_gating_pareto.py`
3. **Estimated cost:** ~3 wallclock days post-cluster-maintenance for a complete answer through Phase 3. Phase 4 (long-horizon validation) is an additional ≤ 1 day if it fires.

## 7. What this plan does NOT do

- **No long-horizon experiments in scope.** Phase 4 only fires conditionally on Phase 3 producing a "clean win" or "partial win" (per §5).
- **No new model training.** All gates are evaluated against the existing per-video gain bundle for the six TTA methods that already shipped. The only Tier-3 probes in §2.4 use a single Adam step on the LoRA adapter — they are not full TTA runs but they do touch the optimiser; per Decision 4 in §8 (user authorisation 2026-06-11), they are now scheduled for Phase 0 (see §2.5 / §3.1).
- **No new headline evaluation.** No new VBench / FVD computation, no new chunk-level metrics. The gain bundle is frozen at `per_video_gains.csv`.
- **No retrieval / NOPROMPT-retrieval / VBench-backfill work** — those are separate workstreams in INDEX.md rows 4 / 5 / 6 and are unaffected by this plan.
- **No edits to `scripts/` or `sbatch/`** during this *planning* PR. Phase-0 sbatch additions are a separate authorisation that this plan asks for in §6.

## 8. Resolved decisions

All four open questions in the original plan draft were resolved by the user on 2026-06-11. The plan is now AUTHORISED for Phases 0–3; Phase 4 remains gated on a separate authorisation after `RECOMMENDATION.md` review (Decision 1 below).

**Decision 1 — Phase 4 (long-horizon validation) auto-fire:** RESOLVED → **Separate authorisation required.**
Rationale: Phase 4's relevance depends on Phase 3's outcome (a clean win, partial win, and no-win each license different long-horizon experiments). Long-horizon already shows a method-asymmetry signal at population level that standard horizon does not (Subj diverges 0.018 between AdaSteer and LoRA r=8 at 76-frame vs 0.005 at 28-frame; ref `paper_tables/2026-06-08_headline_1000v.md` Table 3), so Phase 4 is non-trivial and merits human-in-the-loop review of `RECOMMENDATION.md` before authorisation.

**Decision 2 — Cost-aware Pareto compute-saved interpretation:** RESOLVED → **Immediate 999-video run only.**
Rationale: The savings measured on the 999-video benchmark is itself a transferable claim — future researchers running similar benchmarks would see similar relative savings. We cite the measured 999-video number as the headline result; we do not speculate about specific scaled benchmarks we have not run. Reviewer-defensible.

**Decision 3 — Multiple-comparison correction:** RESOLVED → **Bonferroni α/192 primary, BH-FDR q=0.1 secondary.**
Rationale: Standard practice for the paper's audience. Bonferroni is the conservative headline that licenses any "feature X is a real predictor" claim; BH-FDR shows the texture and surfaces additional candidates worth follow-up implementation. Both reported in §3.2 and §3.3 deliverables.

**Decision 4 — Tier-3 mini-TTA probes (H-T3-1 `grad_norm_θ0` and H-T3-2 `single_step_loss_drop`):** RESOLVED → **Both included in Phase 0.**
Rationale: User explicit instruction to "test all hypotheses". These two features are the closest to the actual TTA loss surface and have the highest expected signal of any feature in the menu. Cost: ~2 extra GPU hours per 999-video run. Note: this requires the Phase-0 sbatch wrapper to also schedule these probes; §2.5 deferred-followup language must be removed.
