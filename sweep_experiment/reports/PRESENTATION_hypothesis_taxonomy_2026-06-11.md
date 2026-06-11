# Per-video TTA suitability — hypothesis taxonomy walkthrough

**Format:** Talk walkthrough notes (~30 min), grad-school presentation prep
**Date:** 2026-06-11
**Audience:** AdaSteer paper authors + advisors
**Companion docs:**
- [HYPOTHESES_per_video_tta_suitability_2026-06-09.md](HYPOTHESES_per_video_tta_suitability_2026-06-09.md) — literature-grounded hypothesis menu (12 hypotheses across themes A/B/C/D/E/G)
- [REVIEW_per_video_tta_suitability_2026-06-09.md](REVIEW_per_video_tta_suitability_2026-06-09.md) — current evidence (saturation findings + 4 ruled-out hypotheses)
- [PLAN_gating_experiment_2026-06-11.md](PLAN_gating_experiment_2026-06-11.md) — experimental protocol (Phases 0–3 AUTHORISED 2026-06-11)

---

## Slide 0 — Title and framing (~1 min)

**Title:** "What predicts whether a video benefits from test-time adaptation?"

**Subtitle:** "A principle-based taxonomy of 12 candidate features for LongCat-Video TTA"

**Talking points:**

- One sentence positioning: *we have been working on AdaSteer / LoRA-r8 / TinyLoRA TTA on LongCat-Video for ~3 months; the most recent 1000-video sweep saturated; this talk is about how we get unstuck.*
- The puzzle that motivates the taxonomy will arrive on Slide 1 — keep this slide short.
- This talk is **not** about a new TTA method. It's about a **predictive question**: given a video `v` and a TTA recipe `m`, can we predict ΔPSNR(v, m) before paying the TTA compute?
- Three reasons that question matters for the AdaSteer paper:
  1. **Deployment gate.** A per-video predictor that clears the Bonferroni bar (|ρ| ≥ 0.13 at N=999, two-tailed, 360-cell budget) would license a deployment-time selection rule — apply TTA only where it helps, skip the others. That converts a saturated headline (REVIEW §2.1) into a real positive net effect.
  2. **Catastrophe avoidance.** The LoRA-r8 family has a single catastrophic outlier (`panda_0098`, 44.55 → 22.16 dB, REVIEW §2.4) responsible for 30% of the aggregate negative bias. A predictor that flags `panda_0098`-type failures *before* TTA fires is paper-defensible safety machinery.
  3. **Mechanism story.** Whichever bucket wins tells us *why* TTA helps when it helps. That's the CVPR-tier subsection — the paper needs at least one principled mechanism, not just an applied recipe.
- The talk is structured around **five principles**, each principle is a separate bucket of candidate features. The 12 hypotheses from the 2026-06-09 literature pass distribute across these five buckets. The hypothesis-by-hypothesis priority order (compute cost) is in HYPOTHESES §3; the principle-by-principle organising structure is what this talk adds.
- 30 minutes total: ~5 min setup (Slides 0–3) + ~22 min on the five buckets (Slides 4–8) + ~3 min synthesis (Slide 9) + ~3 min closing recommendation + limitations (Slides 10–11) + appendix (Slide 12, for reference / Q&A).
- **What this talk is NOT:** it is not a finished result; it is the *organising scaffolding* for the gating experiment that fires when the cluster comes back. The Phase-0 data collection authorised on 2026-06-11 will populate every cell in the Slide-12 appendix table within ~1 wallclock day post-cluster-maintenance. Phase 1 (univariate ρ leaderboard) and Phase 2 (multivariate ensemble) will turn this scaffolding into a result.
- **The companion documents** behind the talk: HYPOTHESES_per_video_tta_suitability_2026-06-09.md (literature pass), REVIEW_per_video_tta_suitability_2026-06-09.md (current evidence), and PLAN_gating_experiment_2026-06-11.md (the protocol, AUTHORISED 2026-06-11 for Phases 0–3). Read them in that order; the talk is a synthesis layer on top.

---

## Slide 1 — Context: the saturation puzzle (~3 min)

**Headline finding (REVIEW §2.1):** at Panda-70M 1000-video / 480p / 17-frame standard horizon, AdaSteer / LoRA-r8 / TinyLoRA-r2 all land within 0.1 PSNR / 8 FVD of NOTTA at the population level.

**Per-video tail breakdown** (per-method ΔPSNR vs NOTTA, N = 999):

| Method | mean Δ | median Δ | \|Δ\|≤0.5 dB | \|Δ\|≤1.0 dB |
|---|---:|---:|---:|---:|
| `ADA` | +0.0080 | −0.0003 | 808 (80.9%) | 899 (90.0%) |
| `ADA_NOPROMPT` | +0.0020 | +0.0040 | 816 (81.7%) | 902 (90.3%) |
| `LORA_R8_TTA` | −0.0756 | −0.0109 | 943 (94.4%) | 971 (97.2%) |
| `LORA_R8_TTA_NOPROMPT` | −0.0650 | −0.0078 | 931 (93.2%) | 967 (96.8%) |
| `TL_BARE_R2` | +0.0108 | −0.0016 | 950 (95.1%) | 982 (98.3%) |
| `TL_TIED_R2` | +0.0027 | +0.0009 | 951 (95.2%) | 980 (98.1%) |

- **81–95% of clips are within ±0.5 dB ΔPSNR.** TinyLoRA is the tightest family (~95%); AdaSteer the loosest (~81%). The whole table is consistent with a per-video noise floor on the order of ±0.5 dB.

**Three obvious-axis predictors are null** (max |Spearman ρ| across all 18 method × feature cells):

- ρ(ΔPSNR, RAFT mean optical flow): **max |ρ| = 0.088** (ADA_NOPROMPT)
- ρ(ΔPSNR, baseline PSNR): **max |ρ| = 0.088** (LORA_R8_TTA, dominated by the `panda_0098` outlier; rank correlation is essentially zero)
- ρ(ΔPSNR, caption word count): **max |ρ| = 0.062** (TL_TIED_R2)

**…but the per-video sign agreement *is* real.** From the 2026-06-09 cross-method analysis:

- Cross-method **top-50-winner Jaccard = 0.08** (small but non-zero overlap between which 50 videos each method picks as its best).
- **19.6% of videos sign-agree across all 6 TTA methods** (i.e. for ~196 of 999 videos, every method has the same-sign ΔPSNR). Under the null (per-method signs independent of each other) the expected unanimous-sign rate would be ≈ 3% — the observed value is a **6.3× lift**.
- LoRA-r8 catastrophic outlier: `panda_0098`, 44.55 → 22.16 dB under `LORA_R8_TTA` (ΔPSNR = −22.4 dB; 30% of aggregate negative bias).

**The "beneficiary" cohort** — a small handful of low-baseline text-heavy or cartoon clips appears in the top-10 winners across multiple methods (REVIEW §2.3):

| Video | Baseline PSNR | Mean flow | Caption (truncated) | Top-10 winner under |
|---|---:|---:|---|---|
| `panda_0461` | 14.04 | 0.071 | "An iphone, a cup of coffee, a yellow sticky note, and a computer are on a desk…" | ADA (#9, +3.50), ADA_NOPROMPT (#1, +8.92), LORA_R8_TTA (#1, +8.25), LORA_R8_TTA_NOPROMPT (#1, +7.94) |
| `panda_0555` | 7.82 | 0.366 | "A cartoon girl looking at her phone with a speech bubble that says good morning…" | LORA_R8_TTA (#3, +3.13), LORA_R8_TTA_NOPROMPT (#3, +3.16), TL_BARE_R2 (#2, +3.14), TL_TIED_R2 (#1, +3.15) |
| `panda_0862` | 10.28 | 1.258 | "A group of cartoon people with their arms up in the air. A dragon ball z…" | LORA_R8_TTA (#2, +7.41), LORA_R8_TTA_NOPROMPT (#2, +7.55), TL_BARE_R2 (#1, +7.53) |
| `panda_0431` | 31.13 | 0.593 | "A black background with red text on it…" | LORA_R8_TTA_NOPROMPT (#4, +2.83), TL_BARE_R2 (#3, +2.89), TL_TIED_R2 (#2, +2.89) |

These four videos are the known-winner controls in PLAN §4 — a useful gate should flag all four as `g = ON`. **Caveat from REVIEW:** `panda_0461` is *also* a top-10 *loser* under `TL_BARE_R2` (#7, −1.23). The cohort is "beneficiary under most methods, not universal" — footnote-worthy, not subset-claim-worthy.

**Talking points (~120 s):**

- The population saturates, but **the tails are not random**. Per-video sign agreement at 6.3× the null lift means *something* predicts where TTA helps. We just haven't found it.
- The three predictors we tried — motion magnitude, baseline difficulty, caption verbosity — are exactly the three a reviewer would ask for. They all fail. That's not a "we didn't look", that's a "we looked at the obvious axes and they're null". The four ruled-out hypotheses are detailed on Slide 2.
- The 4 known-winner cohort + 1 catastrophic-loser cohort give us 5 specific videos as sanity-check anchors throughout the rest of the talk. Any candidate feature has to (i) flag `panda_0098` as risky (for LoRA-r8 specifically) AND (ii) flag `panda_{0461, 0555, 0862, 0431}` as good candidates. A predictor that misses both sets is suspect even if it has good aggregate metrics.
- **An honest framing of the saturation lesson.** At Panda 1000v / 480p / 17-frame standard horizon, the per-video noise floor is roughly ±0.5 dB (81–95% of clips land in this band depending on the method). REVIEW §8 makes this explicit: the per-chunk noise floor on 100-video subsamples is ~0.5 dB; the famously-large "+0.68 dB chunk-0 ADA_NOPROMPT" episode in ANALYSIS_LOG was traced to sampling noise. So when we go looking for per-video structure, we're looking for signal that survives at this floor. The Bonferroni |ρ| ≥ 0.13 bar at N=999 corresponds to the standard t-test rejecting at α = 0.05 / 360 ≈ 1.4 × 10⁻⁴.
- The 19.6% sign-agreement rate at 6.3× null lift is the *primary* evidence that per-video structure exists. It is the population-level fingerprint of the per-video story. If we had measured ~3% sign agreement, we'd conclude TTA outcomes are independent across methods on a per-video basis; the 6.3× lift says they're correlated. Bucket-level mechanism stories live or die by how well they explain this lift.
- **The question for the rest of the talk:** the obvious axes failed. Where do we look next? The literature has been thinking about this for several years across multiple communities — classifier-TTA (EATA, SAR, SoTTA), diffusion-OOD detection (DDPM-OOD, DiffPath, SCOPED), influence functions (DataInf, Mlodozeniec et al.), video complexity (FADE, FLIPD, Spectral Progressive Diffusion), caption alignment (ImageReward, dynamic-CFG), and the DreamBooth catastrophic-overfit family. The 12 hypotheses we extracted (HYPOTHESES doc) span these literatures. **The taxonomy on Slide 3 is how we organise that menu into 5 principle-based buckets.**

---

## Slide 2 — The 4 ruled-out hypotheses (~2 min)

Before we go to the new menu: what did we actually try? Four hypotheses are *ruled out* by the existing 2026-06-09 evidence (REVIEW §3). State them upfront so the audience knows we're past the obvious.

| # | Hypothesis | Status | Evidence | What we'd have expected if it had worked |
|---|---|---|---|---|
| H1 | Motion magnitude (RAFT mean-flow) predicts ΔPSNR | **Ruled out** | max \|ρ\| = 0.085 across 6 methods; signs disagree across families (ADA negative, LoRA positive, TinyLoRA mixed) | High-motion clips harder → bigger TTA gain |
| H2 | Hard-baseline videos benefit (regression to mean) | **Ruled out on rank** | LoRA-r8 Pearson r = −0.143 *but* Spearman ρ = −0.088; the linear-fit signal is dominated by `panda_0098` (44.55→22.16 dB outlier) | Low-PSNR clips have more room → bigger TTA gain |
| H3 | Caption length / verbosity drives gains | **Ruled out** | max \|ρ\| = 0.062 (TL_TIED_R2); other methods ≤ 0.025 | Longer captions provide more conditioning signal → bigger TTA gain |
| H4 | TTA-time caption availability matters | **Ruled out** | Δ within ±0.01 PSNR / ±4 FVD between prompted and NOPROMPT pairs; per-video ρ(Δ, words) for NOPROMPT ≈ −0.02 | Captions during TTA help on at least one method |

**Talking points:**

- **H1 is the natural "harder content needs more TTA" story. Null.** The mean-flow distribution is too coarse — see Bucket C, slide 6, where we resurrect a *shape* statistic of the same flow data that may not be null. The mean integrates a per-frame, per-pixel quantity twice — across pixels and across frames — and discards the distribution of difficulty within the clip. SAR's argument (Niu et al. ICLR 2023) is that if loss is dominated by a handful of high-loss pixels per frame, the *concentration* of difficulty matters more than the average level. Mean-flow rules out the integral; it does not rule out the shape.
- **H2 is the "regression-to-mean" story.** The LoRA-r8 Pearson is misleading: 30% of the negative bias is one outlier video (`panda_0098`, 44.55→22.16 dB). Rank-based, the effect is gone. The high-baseline videos that are predicted to be "hard to improve" are not actually getting hurt by TTA in aggregate — only one specific high-baseline video is, and that one is a Bucket-B catastrophic-overfit story (not a Bucket-A typicality story). This null specifically *strengthens* the case for Bucket B over Bucket A as the operative axis for the catastrophic tail.
- **H3 + H4 together are the "captions matter" story. Null on length, null on presence.** This is *not* the same as "alignment quality is null" — see Bucket D, slide 7. Length and presence are zeroth-order properties; alignment quality is the first-order property the literature actually points at (ImageReward shows CLIPScore *averages* are weakly discriminative but *order statistics* like the per-frame minimum are not; dynamic-CFG shows the per-sample CFG gap varies dramatically across prompts). The Phase-1 univariate analysis will report ρ(CLIP_min, ΔPSNR) on prompted methods AND on NOPROMPT methods, with the gap being the discriminator. H3 and H4 are the *floor*, not the ceiling.
- **NOPROMPT details on H4 worth noting.** From `paper_tables/2026-06-09_panda_std_with_noprompt_partial.md`: ADA → ADA_NOPROMPT shifts PSNR by Δ −0.01 and FVD by Δ +2.1; LORA_R8_TTA → LORA_R8_TTA_NOPROMPT shifts PSNR by Δ +0.01 and FVD by Δ −3.9. VBench-Aes and VBench-Subj match to three decimals between each prompted/NOPROMPT pair. Per-video ρ(Δ, caption words) for NOPROMPT variants: ADA_NOPROMPT −0.025, LORA_R8_TTA_NOPROMPT −0.017 — both essentially zero, as expected when the TTA loss never sees the caption. This is the *tightest* possible refutation of caption presence as a useful signal channel at this regime.
- The four ruled-out hypotheses are the *floor*. Anything in the new menu has to beat that floor with held-out evidence at the Bonferroni level (|ρ| ≥ 0.13 at N=999, two-tailed, 360-cell budget per PLAN §3.2).
- **Operational consequence for the rest of the talk:** the Slide-1 saturation puzzle is real (19.6% sign agreement; 6.3× null lift) AND the obvious axes have been ruled out. We *have* to look at the richer feature space. The 12 hypotheses on Slide 3 are not optional speculation — they are the literature's answer to the question "where else do we look when the obvious predictors fail?"

---

## Slide 3 — The 5-bucket principle taxonomy (~2 min)

This is the spine of the talk. Five buckets, each grounded in a different theoretical principle for *why* a feature should predict per-video TTA gain. Every feature in the HYPOTHESES doc lands in one of these five buckets.

| Bucket | Principle | Measurement primitive | Example features | Predicted ρ sign | Paper claim if this bucket wins |
|---|---|---|---|---|---|
| **A. Model-perceived difficulty** | Diffusion likelihood = implicit OOD score. High loss ⇒ low likelihood ⇒ unfamiliar ⇒ room to adapt. | Forward pass through the diffusion model; measure ε-loss, score norm, intrinsic dimension. | `mean_diffusion_loss_caption`, `score_norm_t*`, `lid_flipd`, `latent_norm_*` | + on LoRA-class (loss-as-OOD); ambiguous on AdaSteer | **"TTA is OOD-correction."** Bonferroni-clearing |ρ| on a likelihood-based feature licenses the Theme-B mechanism subsection. |
| **B. Loss-landscape geometry** | Steeper loss surface around θ₀ → larger TTA step → larger expected gain (under stability). Same machinery flags collapse. | One forward + one backward at unadapted weights; one Adam step + re-eval. | `grad_norm_θ0`, `single_step_loss_drop`, `loss_var_t` | + with \|ΔPSNR\| (asymmetric tail); − with signed ΔPSNR on LoRA-r8 (catastrophe) | **"TTA gain = local loss-landscape steepness."** Cleanest mechanism story for CVPR; only bucket that captures the catastrophic tail. |
| **C. Visual / temporal complexity** | Some content classes are intrinsically harder for the base model. Predictable from raw video without ANY model forward pass. | CPU-only: optical-flow distribution shape, 3D FFT energy, edge density, lossless-compression bpp, scene-cut count. | `flow_max_over_mean`, `flow_entropy`, `hf_energy_ratio`, `bpp_h264`, `cut_count_pyscenedetect`, `laplacian_variance_mean`, `rgb_histogram_entropy_mean` | + LoRA-class on HF features; ambiguous on AdaSteer | **"Per-video TTA suitability is data-intrinsic."** Cheapest possible deployment gate (no model forward needed). Best edge-device story. |
| **D. Cross-modal alignment** | TTA's inference signal includes the caption. Alignment QUALITY (not presence) predicts per-video gain even when presence does not. | CLIP/BLIP text-image similarity statistics; classifier-free-guidance gap. | `clip_text_image_sim_min`, `cfg_gap`, `delta_caption_minus_uncond` | + caption-using methods; ≈ 0 NOPROMPT; gap > 0.05 | **"Caption–video alignment, not video alone, predicts TTA gain."** Cross-modal pre-conditioning subsection; directly explains the NOPROMPT-vs-prompt result. |
| **E. Reconstruction observability** | If the VAE can't represent it, TTA can't fix it. Autoencoder rec-error caps achievable gain from below. | One VAE encode + decode per video; pixel-space + LPIPS rec-error. | `rec_err_l1`, `rec_err_lpips` | + \|ΔPSNR\| asymmetric; + signed on LoRA-class | **"TTA can only fix what the autoencoder already represents."** Limits-of-TTA subsection; clean negative-result story if the only feature that works. |

**The five-bucket structure does the work that the existing tier-based ordering (T1/T2/T3 by compute cost) cannot do:** it tells us *what we'd be claiming* if a given feature wins. The compute axis tells us *what to run first*; the principle axis tells us *what to write about*.

**Talking points (~90 s):**

- Five buckets, not four, not six. Why these five?
  - **A and B are the two model-conditional principles.** A is about the loss *values* (likelihood / OOD); B is about the loss *landscape geometry* (gradient, step). They are mechanistically distinct: A predicts learnability; B predicts step magnitude. Same forward pass can serve both, but they should be reported separately because the paper claim each licenses is different ("TTA is OOD-correction" vs "TTA gain = loss-landscape steepness").
  - **C is model-independent.** RAFT, FFT, bpp, scene-cuts, edge density, Laplacian variance, RGB-histogram entropy. Whatever you can compute from the raw video without a single diffusion forward. This is the bucket that gives us a deployment-time gate on edge devices.
  - **D is cross-modal.** The caption channel. Distinct from A and C because the principle is *alignment*, not difficulty or complexity. The NOPROMPT result already established that caption *presence* is null; D asks whether caption *quality* is also null.
  - **E is autoencoder-observability.** Specific to latent-space diffusion models. Caps the gain from below — TTA can only fix what the autoencoder represents. This bucket is the "limits-of-TTA" story.
- **Why not 4 buckets?** The obvious lumpings would be (i) merge A and B into "model-derived signals" and (ii) merge E into Bucket A (both are about likelihood / typicality). Both lumpings would lose information. A vs B distinguishes loss-value-based signals from loss-landscape-geometry-based signals — these predict *different aspects* of TTA outcome (modal gain vs tail risk). E vs A distinguishes pixel-space autoencoder rec-error from latent-space diffusion-model likelihood — these have *different cost tiers* and *different paper claims*.
- **Why not 6 buckets?** The obvious extension would be a "spectral / frequency" bucket distinct from Bucket C (visual complexity). But spectral features (FFT high-freq ratio, FADE-style frequency factorisation) are mechanistically a Bucket-C story (intrinsic content complexity) — the spectral angle is *how* we measure complexity, not a separate principle. Same for influence functions (DataInf, Mlodozeniec): they're a Bucket-B mechanism (loss-landscape geometry) computed in a particular way. The bucket count is determined by the number of distinct *principles*, not the number of distinct *measurement techniques*.
- The five buckets are *soft*. Some features blur — CFG-gap is mostly D (caption-aware) but also has a Bucket A flavour (it's an ε-field property). FLIPD is mostly A (model-derived geometry) but it's measuring intrinsic dimensionality, which is conceptually a Bucket C quantity. Scene cuts are mostly C (model-independent video statistic) but their mechanism story is B (non-stationary loss landscape). We'll flag these spans as they come up and revisit them in the closing synthesis. The appendix table (Slide 12) lists primary + secondary affinities for every feature explicitly.
- **The two cross-cutting axes** (synthesis Slide 9 develops these):
  - **Modal-gain vs tail-risk.** Buckets A/C/D/E predict the bulk of the distribution; Bucket B predicts the tail (catastrophic failures AND extreme winners).
  - **Method-agnostic vs method-specific.** A/C/D/E are method-agnostic (same feature gates any TTA recipe); B is method-specific (`grad_norm_θ_lora` is meaningful for LoRA but not for AdaSteer's δ-tuning).
- **Sequence of the remaining slides:** A → B → C → D → E, in roughly that order of mechanism strength for the CVPR audience. Bucket B is the punchline (catastrophic-tail prediction); we save it for after A so the audience has the "TTA is OOD-correction" baseline mechanism in mind before we add the geometric-overfit story. C / D / E come after to round out the menu and discuss practical / negative-result variants.
- **Pacing reminder.** Each bucket gets 3–5 minutes: ~1 min principle + ~1 min mechanism + ~1–2 min features table + ~1 min paper-claim implications. The features tables on each bucket slide are the source of truth for what gets implemented in Phase 0; do not re-read them in full during the talk — point at the table and walk through 1–2 representative features.

---

## Slide 4 — Bucket A: Model-perceived difficulty (~5 min)

### A.1 Guiding principle (~1 min)

**The principle:** *a diffusion model assigns implicit log-likelihood to data via its denoising loss. High denoising loss = low likelihood = OOD = room to adapt.*

- For a denoising diffusion model with noise schedule `α_t`, the expected ε-loss `E_{t, ε} ||ε − ε_θ(x_t, c, t)||²` is — up to a constant — the negative log-likelihood of `x` under the model. This is the textbook DDPM result (Ho, Jain, Abbeel 2020).
- The Theme-B OOD-detection literature builds on exactly this:
  - **AnoDDPM / DDPM-OOD** (Graham et al. CVPRW 2023, https://arxiv.org/abs/2211.07740): per-noise-level reconstruction error is the OOD score.
  - **DiffPath** (Heng et al. NeurIPS 2024, https://arxiv.org/abs/2405.11881): rate-of-change of the diffusion trajectory.
  - **SCOPED** (Barkley et al. 2025, https://arxiv.org/abs/2510.01456): one-forward-pass single-Hutchinson-JVP score-norm approximation.
- **Why it matters for TTA suitability:** the TTA loss IS the diffusion loss (or a close variant — flow-matching MSE in our setting). A video for which the diffusion loss at θ₀ is high is a video for which TTA's optimiser has slope to descend.
- **Sanity check from the generative-TTA literature:** TTL-LLM (Sun et al. 2025, https://arxiv.org/abs/2505.20633) explicitly finds that in *generative* test-time learning (unlike classifier TTA), **high-perplexity samples gain *more*** — the opposite of the EATA / SAR classifier result. Diffusion TTA is a generative regime. Bucket A should have positive ρ with ΔPSNR.

### A.2 Mechanism (~1 min)

**Why this should predict TTA gain — the slope argument:**

- If the diffusion loss at the unadapted weights θ₀ is already near zero on this video, there is no gradient. The optimiser has nothing to do. ΔPSNR ≈ 0.
- If the loss is high, the optimiser has slope to descend. ΔPSNR > 0 (provided the descent direction generalises from the TTA-visible window to the held-out generation window — which is a *separate* question; the modal-gain prediction here is about loss-as-likelihood, not loss-landscape geometry).
- **Caveat from Theme B:** Serrà et al. (ICLR 2020, https://arxiv.org/abs/1909.11480) showed that pixel-space generative-model likelihoods are dominated by lossless-compression bits-per-pixel. The same confound applies in VAE-latent space because the LongCat VAE is itself a learned compressor. Without subtracting bpp, a "model-perceived difficulty" feature is partly measuring "raw input complexity". **H-T1-2 is the explicit bpp covariate that controls for this confound** (it sits in Bucket C — see slide 6).
- **Caveat from the representation-space line:** Ding et al. 2025 (https://arxiv.org/abs/2504.07793) and Järve et al. 2025 (https://arxiv.org/abs/2508.15737) show that likelihood-based OOD detection works *fine* if you compute likelihood in a learned encoder's latent space. LongCat-Video is a latent-space diffusion model, so the relevant likelihood is in VAE-latent space, not in pixel space — exactly what our caption-conditioned diffusion-loss scorer (commit `dc115e7`) computes.

### A.3 Features in this bucket (~2 min)

Five features:

| # | Feature | Formula / extraction recipe | Cost tier | Expected ρ sign | Implementation status | Falsification criterion |
|---|---|---|---|---|---|---|
| A-1 | `mean_diffusion_loss_caption` (H-T2-1) | `E_{t, ε} ‖ε − ε_θ(x_t, c, t)‖²` averaged over t ∈ {100, 500, 900}; partial-out `bpp_h264` covariate. Source: `scripts/compute_diffusion_ood_score.py` (commit `dc115e7`). | T2 (~2–3 h) | + on LoRA-class methods; ambiguous on AdaSteer | **Scaffolded** — script committed; not yet run on cluster | Residual \|ρ\| ≤ 0.10 with ΔPSNR on all 4 LoRA-class methods after partialling out bpp |
| A-2 | `mean_diffusion_loss_uncond` + `delta_caption_minus_uncond` (H-T2-1 unconditional + lite-CFG-gap proxy) | Same as A-1 but with ∅ caption; the delta column is a coarse CFG-gap proxy. Free emission from the same OOD scorer. | T2 (bundled with A-1) | comparator for A-1; weak alignment proxy in `delta` | **Scaffolded** — same OOD CSV | \|ρ\| ≤ 0.10 with ΔPSNR on caption-using methods AND `delta` < 0.10 ρ |
| A-3 | `score_norm_t*` (H-T2-2 — SCOPED-style) | `E_ε ‖ε_θ(x_{t*}, c, t*)‖² / (T·C·H·W)` at t* ∈ {200, 500, 800}. Bolts onto A-1 forward pass at near-zero marginal cost (record `‖ε‖²` alongside the MSE). | T2 (free if scheduled with A-1) | + LoRA-class; ≈ 0 AdaSteer | **Not implemented** — ≤ 30-line patch to OOD scorer per PLAN §3.1 | \|ρ\| ≤ 0.10 across all 4 LoRA-class methods AND A-1 has \|ρ\| > 0.15 (loss-mismatch dominates over score-magnitude) |
| A-4 | `lid_flipd` (H-T2-4) — diffusion-native local intrinsic dimension | `lid = (D/2) · (1 − tr(∇²_x log p_t(x)) · σ²_t)` per Kamkari et al. NeurIPS 2024 eq. 7 (https://arxiv.org/abs/2406.03537); evaluated via Hutchinson Hessian-trace at small t. Shares the A-1 forward pass. | T2 (+30% over A-1) | + LoRA-class; ≈ 0 AdaSteer | **Not implemented** | \|ρ\| ≤ 0.10 across all 4 LoRA-class methods |
| A-5 | `latent_norm_mean`, `latent_norm_std`, `latent_kurtosis` (Theme B latent moments) | First three moments of `‖z‖` over the encoded latent of the TTA-visible window. Free emission from the OOD scorer. | T2 (bundled) | uncertain (probe) | **Scaffolded** — emitted by OOD scorer | All three \|ρ\| ≤ 0.10 across all 6 methods |

**Two cross-cutting notes:**

- **bpp partial-out is non-optional for A-1 / A-2 / A-3.** Serrà et al.'s ICLR 2020 result is the most concrete threat to a "diffusion loss as OOD" claim. The Phase-1 univariate analysis (PLAN §3.2) will report both raw ρ(loss, ΔPSNR) and partial-correlation ρ(loss, ΔPSNR | bpp). The bpp feature itself lives in Bucket C.
- **Conditional vs unconditional.** A-1 stores caption-conditioned losses; A-2 stores unconditional. Per HYPOTHESES §6 open question 1, the cleanest scientific choice is to store both at the same noise samples; cost is ≤ 2× the conditional-only run. The OOD scorer already does this (commit `dc115e7`).

### A.4 What it means if Bucket A wins (~1 min)

**Paper claim (if some feature in this bucket clears Bonferroni |ρ| ≥ 0.13 at N=999):**

> "*Per-video TTA suitability is predicted by the unadapted diffusion model's denoising loss on that video. This is a direct instantiation of the OOD-detection-via-diffusion-loss line of work (Graham et al. CVPRW 2023; Heng et al. NeurIPS 2024; Barkley et al. 2025) in the test-time-adaptation setting: TTA is OOD-correction.*"

**Concrete subsection structure for the paper if Bucket A wins:**

- Subsection title: "*Test-time adaptation gain is predicted by diffusion-likelihood OOD scoring.*"
- **Headline plot:** scatter of `mean_diffusion_loss_caption` vs ΔPSNR on the four LoRA-class methods (4 panels). Fit a per-method linear regression; report Spearman ρ + Bonferroni significance flag per panel. (~ Figure 4 of the paper.)
- **Ablation table:** for each of A-1 / A-2 / A-3 / A-4, report (ρ, p_Bonferroni, ρ_partial | bpp, gated mean Δ at coverage 50%). One row per (feature, method, metric). (~ Table 4 of the paper.)
- **Comparison axis:** A-1 (loss mismatch) vs A-3 (score magnitude). If A-3 dominates A-1, the operative axis is the score-field geometry (a Bucket-B-flavoured story); if A-1 dominates A-3, the operative axis is loss mismatch (a pure OOD story). Both are publishable; the discrimination tells the reader what they're actually buying.
- **What it does NOT claim:** Bucket A does not predict the *catastrophic* LoRA-r8 tail (`panda_0098`). That's a Bucket-B story (slide 5). A says "TTA helps where the model is uncertain"; B says "TTA collapses where the model is unstable".

### A.5 Falsification calendar for Bucket A (~30 s)

What kills Bucket A at each phase:

- **Dies in Phase 1 (univariate) if:** every feature A-1 through A-5 has |ρ| ≤ 0.10 with ΔPSNR on all six methods after bpp partial-out. Bonferroni-significant requires |ρ| ≥ 0.13; |ρ| ≤ 0.10 is well below noise. Phase 1 cost is ~1 day of analysis on the login-node CPU after Phase 0 cluster outputs land.
- **Survives Phase 1 but dies in Phase 2 (multivariate) if:** A-1 / A-3 / A-4 / A-5 collectively contribute < 5% to the held-out AUC of the multivariate gate (i.e., the gate works only when Bucket-B or Bucket-D features are included; Bucket A is dispensable).
- **Survives Phase 2 but dies in Phase 3 (Pareto) if:** the cost-aware best A-bucket strategy is dominated by a cheaper Bucket-C alternative. Plausibility: a free Bucket-C feature like `hf_energy_ratio` could in principle achieve similar gated mean Δ at zero forward-pass cost, dominating A on the Pareto frontier.
- **Wins outright if:** A-1 has |ρ| ≥ 0.15 with ΔPSNR on at least two LoRA-class methods after bpp partial-out AND A-3 corroborates with same-sign ρ ≥ 0.10. That is the Theme-B mechanism test (Heng et al. + Barkley et al. + Graham et al.) running clean in our setting.

### A.6 Worked example — what we'd say about `panda_0461` under Bucket A (~30 s)

`panda_0461` (baseline PSNR 14.04, mean_flow 0.071 — low both axes) is a top-10 winner on four methods (REVIEW §2.3). Under the Bucket-A hypothesis, this means:

- The unadapted LongCat-Video model assigns *low* likelihood to `panda_0461` (high `mean_diffusion_loss_caption`) — it's an iPhone-on-a-desk static scene with text overlay, which is OOD for the natural-motion Panda-70M distribution.
- The TTA optimiser has slope to descend (high `‖∇L‖` at θ₀ — though this is Bucket B, not A).
- After ~20–40 TTA steps, the model has improved its conditional likelihood of generating this exact-content scene. Held-out generation window improves by 3–8 dB.

What Bucket A *doesn't* explain about `panda_0461`: why it is *also* a top-10 loser under `TL_BARE_R2` (#7, −1.23). That's a method-specific story (TinyLoRA's r=2 capacity is too small to specialise without dragging the global content — a TinyLoRA-specific Bucket-B mechanism), not a Bucket-A story.

### A.7 Connections to the broader AdaSteer paper (~30 s)

If Bucket A wins, the paper structure looks like:

- **Methods section.** Currently describes AdaSteer's δ-tuning and the LoRA-r8 / TinyLoRA-r2 comparisons. Add: per-video diffusion-loss OOD scoring as a *gate* applied before any TTA recipe. The gate is method-agnostic (it's a property of the base model, not the adapter).
- **Results section.** Currently reports the saturated headline (REVIEW Table 1). Add: Figure 4 (the scatter from A.4 above) + Table 4 (the per-feature ablation). The narrative shifts from "TTA saturates at population scale" to "TTA saturates at population scale BUT a model-derived OOD score recovers per-video structure".
- **Discussion section.** Connects to the OOD-detection literature (DiffPath, SCOPED, DDPM-OOD). Frames the AdaSteer paper as a *bridge* between the OOD-detection community and the TTA-on-video community. The bridge is the headline contribution.
- **Conclusion.** "*Per-video TTA gain is predicted by likelihood-based OOD scoring. The proposed two-feature gate (Bucket B × Bucket A) reduces aggregate TTA compute by X% while delivering Y dB ΔPSNR on the gated subset.*"

---

## Slide 5 — Bucket B: Loss-landscape geometry (~5 min)

### B.1 Guiding principle (~1 min)

**The principle:** *steeper loss surface around θ₀ → larger TTA step possible per unit of optimiser budget → larger expected ΔPSNR (under stability).*

- Two strands of literature converge here:
  - **Theme A — per-sample TTA selection.** EATA (Niu et al. ICML 2022, https://proceedings.mlr.press/v162/niu22a/niu22a.pdf): adapting on low-entropy samples *beats* adapting on the full test set. SAR (Niu et al. ICLR 2023, https://openreview.net/pdf?id=g2YraF75Tj): the same finding via per-sample gradient norm — large-gradient samples drive model collapse. SoTTA (Gong et al. NeurIPS 2023, https://proceedings.neurips.cc/paper_files/paper/2023/file/2da53cd1abdae59150e35f4693834f32-Paper-Conference.pdf): same finding via sharpness-aware minimisation.
  - **Theme C — influence functions and curvature.** Mlodozeniec et al. ICLR 2025 (https://arxiv.org/abs/2410.13850): K-FAC GGN approximations of influence functions for diffusion models. DataInf (Kwon et al. ICLR 2024, https://arxiv.org/abs/2310.00902): closed-form influence approximation specifically for LoRA-tuned LLMs and diffusion models. SLo-Curves (Garg & Roy CVPR 2023, https://cvpr.thecvf.com/virtual/2023/poster/20980): low-loss-curvature samples are clean; high-curvature samples are atypical.
- These two strands collapse onto the **same per-video predictor**: at fixed optimiser step size, per-sample loss curvature and per-sample gradient norm are tightly coupled. SAR's reliable-sample story and SLo-Curves's curvature story make the same prediction.

### B.2 Mechanism (~1 min)

**Why local geometry of the per-video loss landscape predicts TTA gain — two distinct sub-mechanisms:**

1. **Step magnitude (the modal-gain story).** If `‖∇L‖` is large at θ₀, a single optimiser step moves the model meaningfully in the loss-decreasing direction. The total TTA update is roughly `Σ_step ∇L`; a large initial gradient ⇒ a large total update ⇒ (under stability) a large expected change in the model's prediction on the held-out window.
2. **Single-step over-fit detection (the tail-risk story).** If a single Adam step on the LoRA adapter already drops the per-video loss by, say, 70%, the optimiser has *too much* slope. By step N (with N ~ 20–40 in our TTA recipes), the model has memorised the visible window. There is no prior-preservation loss to pull the optimiser away from the trivial "render this exact clip" solution. This is the **DreamBooth language-drift** mechanism (Ruiz et al. CVPR 2023, https://arxiv.org/abs/2208.12242). Anti-CF (Ye et al. EMNLP 2023, https://aclanthology.org/2023.emnlp-main.803.pdf) and ZeroSiam (https://arxiv.org/abs/2509.23183) generalise to general TTA model-collapse. The `panda_0098` catastrophe (44.55 → 22.16 dB under `LORA_R8_TTA`) is exactly this failure mode, and **the single-step in-loop loss drop is its predicted screening signal**.

**The two sub-mechanisms make DIFFERENT predictions:**

- `grad_norm_θ0` (H-T3-1) is symmetric with respect to direction: it predicts \|ΔPSNR\| (both bigger wins and bigger losses), and on LoRA-r8 specifically it predicts *signed* ΔPSNR negative (because the catastrophic-tail mechanism is on the loss-incurring side).
- `single_step_loss_drop` (H-T3-2) is asymmetric: it predicts catastrophic *losses* (signed ρ ≤ −0.20 with `LORA_R8_TTA` ΔPSNR). The `panda_0098` row is predicted to be in the top-10% of `single_step_loss_drop`.

### B.3 Features in this bucket (~2 min)

Four features:

| # | Feature | Formula / extraction recipe | Cost tier | Expected ρ sign | Implementation status | Falsification criterion |
|---|---|---|---|---|---|---|
| B-1 | `grad_norm_θ0` (H-T3-1) — SAR-style large-gradient predictor | `‖∇_{θ_LoRA} L_diff(x; θ₀)‖_2` — one forward + one backward through the LoRA / AdaSteer parameter set at the unadapted weights, no optimiser step. | T3 (~30 min / 999 videos on H200, or ~3 min stratified per HYPOTHESES §6 Q4) | + with \|ΔPSNR\| LoRA-class; − with signed ΔPSNR on `LORA_R8_TTA` | **Scheduled for Phase 0** per Decision 4 in PLAN §8 (commit `38df1ba` + Tier-3 runner `compute_tier3_probes.py`) | \|ρ\| ≤ 0.15 with \|ΔPSNR\| on `LORA_R8_TTA` |
| B-2 | `single_step_loss_drop` (H-T3-2) — DreamBooth-style overfit detector | `(L_diff(x; θ₀) − L_diff(x; θ₀ + Adam_step)) / max(L_diff(x; θ₀), 1e-6)` — one forward + one backward + one Adam step + one forward pass on the LoRA-r8 adapter (r=8 / α=16 / lr=5e-5; same recipe as `run_lora_tta.py`). | T3 (~30 min / 999 videos) | **strongly negative** ρ ≤ −0.20 with signed `LORA_R8_TTA` ΔPSNR | **Scheduled for Phase 0** per Decision 4 in PLAN §8 | \|ρ\| ≤ 0.10 with `LORA_R8_TTA` ΔPSNR (refutes the DreamBooth-collapse mechanism for LoRA-r8 TTA in this setting) |
| B-3 | `loss_var_t` (H-T2-5) — diffusion-loss variance across timesteps (EATA analogue) | `Var_{t ∈ {100, 500, 900}} E_{ε, frame} ‖ε − ε_θ(x_t, c, t)‖²` — reuses A-1's per-(t, noise-sample) losses; pure post-processing. | T2 (~free if A-1 stores per-t losses) | + with ΔPSNR ≥ 0.15 on LoRA-class methods | **Derivable** — `scripts/derive_loss_variance.py` (≤ 50 LOC; pandas) | \|ρ\| ≤ 0.10 with ΔPSNR on all 4 LoRA-class methods AND A-1 mean-loss has \|ρ\| > 0.15 (absolute level dominates over variance) |
| B-4 | `score_norm_t*` secondary | See A-3 — score-field magnitude has a geometric flavour; we report it primarily under A but cite it under B for completeness. | (see A-3) | (see A-3) | (see A-3) | (see A-3) |

**Per-video tail-risk vs modal-gain disentanglement:**

- **B-1** (grad_norm) flags videos where the model is *locally steep* — these are the modal-gain candidates AND the catastrophic-tail candidates (the signal is symmetric).
- **B-2** (single-step loss drop) flags videos where the optimiser *over-shoots in one step* — these are specifically the catastrophic-tail candidates (the signal is asymmetric and negative-signed).
- **B-3** (across-t loss variance) flags videos where the model is *inconsistent across noise scales* — high variance means a specific stage of the denoising trajectory is uncertain, which is precisely where per-sample TTA has its largest impact.

### B.4 What it means if Bucket B wins (~1 min)

**Paper claim (if some feature in this bucket clears Bonferroni AND predicts the catastrophic tail):**

> "*Per-video TTA gain is predicted by the local geometry of the loss surface at the unadapted weights. Specifically, the single-step in-loop loss drop predicts the catastrophic LoRA-r8 failure mode (Spearman ρ ≤ −0.20 with `panda_0098`-class outliers) before any TTA computation is committed; this is the diffusion-video instantiation of the DreamBooth language-drift failure mode (Ruiz et al. CVPR 2023).*"

- **Why Bucket B is the most theoretically satisfying for a CVPR audience:** the mechanism story is precise (single-step DreamBooth-style overfit), the literature provenance is clean (SAR + SLo-Curves + DataInf + DreamBooth), and the prediction is **asymmetric** — Bucket B predicts the *tail*, not just the mean. None of A / C / D / E can do that.
- **Critical caveat — method-specificity.** `grad_norm_θ0` is a property of *the LoRA adapter's gradient* (or the AdaSteer δ-parameter gradient, etc.). It is **method-specific** in a way that A / C / D / E are not. We discuss the method-agnostic-vs-specific axis in the synthesis (slide 9 §2).
- **Predicted experimental outcome (best guess with uncertainty):** B-2 (single-step loss drop) has a *substantial* probability of clearing Bonferroni on `LORA_R8_TTA` (target ρ ≤ −0.20). B-1 (grad_norm) is more likely to predict \|ΔPSNR\| than signed ΔPSNR. B-3 (across-t variance) is the dark horse — free post-processing of A-1's data and a clean reliable-sample-selection mechanism story.
- **Paper subsection structure:**
  - Title: "*Catastrophic LoRA-r8 failures are predicted by a single-step in-loop loss drop.*"
  - Headline plot: `single_step_loss_drop` vs ΔPSNR on `LORA_R8_TTA`, with `panda_0098` annotated in the top-right quadrant.
  - Ablation: B-1, B-2, B-3 side by side for the catastrophic-tail screening task. Report per-feature Fisher exact OR conditional on \|ΔPSNR\| > 1 dB.
  - Compare to: H-T1-6 (PySceneDetect cut count, Bucket C) which is the same asymmetric-tail prediction but model-free.

### B.5 Falsification calendar for Bucket B (~30 s)

What kills Bucket B at each phase:

- **Dies in Phase 0 (data collection) if:** the Tier-3 probe runner (`compute_tier3_probes.py`) fails to converge under the no-carryover guarantee — but per ANALYSIS_LOG 2026-06-11 (later+2), the runner mirrors the production LoRA-r8 recipe (r=8 / α=16 / lr=5e-5 / weight_decay=0.01 / targets=qkv+proj on all blocks, no FFN) and resets the LoRA adapter + re-instantiates the optimiser per (video, timestep) loop. So this is a non-risk.
- **Dies in Phase 1 if:** B-2 has |ρ| ≤ 0.10 with signed `LORA_R8_TTA` ΔPSNR AND B-1 has |ρ| ≤ 0.15 with |ΔPSNR|. That refutes both the DreamBooth-collapse mechanism (B-2) AND the SAR classifier→diffusion mapping (B-1) for this setting.
- **Survives Phase 1 in a partial way if:** B-1 carries but B-2 doesn't — would mean the symmetric-large-gradient story works but the asymmetric overfit story doesn't. Less satisfying mechanism, still publishable.
- **Wins outright if:** B-2 has ρ ≤ −0.20 with signed `LORA_R8_TTA` ΔPSNR, `panda_0098` lands in the top decile of `single_step_loss_drop`, AND Fisher exact OR ≥ 3 on `(ΔPSNR < −1 dB) × (top-decile B-2)`. That is the catastrophic-tail screening victory.

### B.6 Worked example — what we'd say about `panda_0098` under Bucket B (~30 s)

`panda_0098` (baseline PSNR 44.55, mean_flow ≈ 0.05 — text-on-white-background "home workshop makeover tour"; the catastrophic outlier of LoRA-r8 TTA). Under the Bucket-B hypothesis:

- The unadapted LongCat-Video model has a *high* per-LoRA-parameter gradient at `panda_0098`'s caption-conditioned diffusion loss — the diffusion model is locally steep at this very-static, high-baseline-PSNR clip.
- A single Adam step on the LoRA-r8 adapter drops the in-loop loss by a large fraction — say, top 5–10% across the 999-video set.
- The optimiser proceeds to memorise the visible window over the next ~20–40 steps; the LoRA adapter degenerates into "render this exact text-on-white scene" rather than the underlying class distribution.
- Held-out generation window: catastrophic 22.4 dB drop.

What Bucket B *predicts about other videos in the catastrophic tail*: the other LoRA-r8 losers below ΔPSNR < −1 dB (there are 21 of them per REVIEW §2.1) should have above-median `single_step_loss_drop`. The Phase-1 Fisher exact test on this 2×2 contingency table is the headline single-number paper-claim for Bucket B.

What Bucket B does *not* predict: the modal-gain side of the LoRA-r8 distribution. The 7 LoRA-r8 winners with ΔPSNR > +1 dB are predicted by Bucket A or D, not B. That's the multivariate-gate observation on Slide 9.

### B.7 Comparison with the four ruled-out hypotheses (~30 s)

How is Bucket B different from H2 (baseline PSNR ruled out)?

- H2 says "*low-baseline-PSNR videos benefit more from TTA*". On rank correlation, that's null (ρ = −0.088, dominated by `panda_0098`).
- B-1 says "*large-gradient-norm videos have *asymmetric* TTA outcomes — either big wins or big losses*". This is *not* the same prediction. A high-baseline-PSNR video like `panda_0098` can simultaneously be (i) "hard to improve from above" (consistent with H2 null because the rank correlation washes out) AND (ii) "large-LoRA-gradient because the optimiser has slope on the visible window" (consistent with B-1's |ΔPSNR| asymmetric prediction).
- The discriminator: H2 is signed; B-1 is asymmetric. Both can hold simultaneously; they explain different aspects of the distribution.

### B.8 Connections to the broader AdaSteer paper (~30 s)

If Bucket B wins, the paper structure shifts further:

- **Methods section.** Add: a single-step pre-flight check that runs *before* any TTA recipe fires. The check costs ~3 minutes per 999 videos on a stratified sample. The decision rule: if `single_step_loss_drop > p_threshold`, reject the TTA application on this video.
- **Results section.** Add: a separate "catastrophic-failure case study" subsection. `panda_0098` figure with the time-series of in-loop loss + held-out PSNR + the predicted-vs-actual ΔPSNR scatter. This is the most concrete single-video figure in the paper.
- **Discussion section.** Connects to the DreamBooth catastrophic-overfit literature (Ruiz et al. CVPR 2023; Anti-CF EMNLP 2023; ZeroSiam) AND the influence-function literature (DataInf ICLR 2024; Mlodozeniec et al. ICLR 2025; SLo-Curves CVPR 2023). Two distinct citation neighbourhoods.
- **Conclusion.** "*Per-video TTA catastrophic failures are predicted by a single-step in-loop loss drop. The screening cost is negligible compared to the cost of a single catastrophic outlier in production.*"

### B.9 Why we expect B-2 specifically to be the strongest single feature (~30 s)

A back-of-the-envelope argument for why B-2 should clear Bonferroni:

- The catastrophic LoRA-r8 tail has 21 videos with ΔPSNR < −1 dB out of N = 999. Under the null hypothesis (no feature predicts the tail), the expected Fisher-exact OR for any decile-threshold split is 1.0. To clear OR ≥ 3.0, the catastrophic-tail subset has to be disproportionately in the top-decile of B-2 — say 10 of 21 catastrophic-tail videos in the top decile vs 2.1 expected under the null. That's a ~5× lift, comparable to the 6.3× sign-agreement lift from Slide 1.
- The mechanism story (DreamBooth-style one-shot overfit) is precise enough that the prediction has direction: B-2 high ⇒ ΔPSNR catastrophically low. The signed-ρ prediction (ρ ≤ −0.20) is sharper than the asymmetric-|ΔPSNR| prediction of B-1.
- The cost of being wrong on B-2 is moderate: ~30 minutes of one H200 to find out. The cost of being right and missing the prediction is the catastrophic outlier going undetected in production.
- The convergence of three literatures (SAR + DreamBooth + Anti-CF) on the same mechanism prediction makes this the highest-prior single hypothesis in the menu.

---

## Slide 6 — Bucket C: Visual / temporal complexity (~5 min)

### C.1 Guiding principle (~1 min)

**The principle:** *some content classes are intrinsically harder for the base model — high motion, busy textures, scene cuts, high frequency content. These can be predicted from the raw video without ANY model forward pass.*

- This is Theme D in HYPOTHESES — video-specific complexity *beyond* optical-flow mean.
- Three complementary toolkits:
  - **Information-theoretic.** Normalised Shannon entropy (https://www.mdpi.com/1099-4300/27/2/166); RGB-histogram entropy is already extracted in `extract_video_features_for_tta.py`.
  - **Bit-rate-based.** Lossless-compression bits-per-pixel (PNG, H.264) as a direct compressibility proxy. Closes the loop with Serrà et al. (ICLR 2020) — the same quantity that confounds Bucket A is a feature in its own right here.
  - **Diffusion-spectral.** Dieleman's blog (https://sander.ai/2024/09/02/spectral-autoregression.html) + Spectral Progressive Diffusion (Yu et al. 2025, https://arxiv.org/abs/2605.18736) motivate the 3D FFT high-frequency ratio. FADE (Zhu et al. CVPR 2025, https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_FADE_Frequency-Aware_Diffusion_Model_Factorization_for_Video_Editing_CVPR_2025_paper.pdf) confirms low-freq carries structure / motion, high-freq carries detail.

### C.2 Mechanism (~1 min)

**Why intrinsic video complexity predicts TTA gain — note the ambiguity:**

- **Branch 1: complexity → "harder to fit" (TTA helps more).** A pretrained model under-fits high-frequency content more than low-frequency content because the per-pixel ε-loss budget is dominated by low-frequency content. LoRA has spare rank capacity to target the under-fit band; AdaSteer's δ-tuning of attention adapters does not. **Prediction: positive ρ on LoRA-class, near-zero on AdaSteer for the FFT-HF feature (H-T1-3).**
- **Branch 2: complexity → "harder to reach via TTA" (TTA helps less).** If the content is intrinsically high-entropy / unpredictable, no per-sample optimiser update can find a generalising direction. The gradient is noisy, the held-out generation window has different complexity statistics than the TTA-visible window, and TTA *hurts*. **Prediction: negative or near-zero ρ.**
- **The empirical question is which branch dominates.** The FFT-HF feature (H-T1-3) is the clean differential test: it makes a method-specific prediction (positive on LoRA, near-zero on AdaSteer) that no other Bucket-C feature does.
- **Scene-cut prediction is asymmetric (H-T1-6).** A scene-cut *inside the TTA-context split* (frames 0..13 clean context vs 14..47 noised target) creates a non-stationary loss landscape that drives the LoRA optimiser off-manifold. SAR + SoTTA both establish that sharp per-sample loss landscapes are the precondition for TTA-driven model collapse. Prediction: Fisher exact OR ≥ 3 on `LORA_R8_TTA` for `(ΔPSNR < −1.0 dB) × (≥ 1 cut)`. This bridges Bucket C and Bucket B.

### C.3 Features in this bucket (~2 min)

Seven features:

| # | Feature | Formula / extraction recipe | Cost tier | Expected ρ sign | Implementation status | Falsification criterion |
|---|---|---|---|---|---|---|
| C-1 | `flow_max`, `flow_entropy`, `flow_max_over_mean` (H-T1-4) — RAFT distribution SHAPE | `flow_max = max_{t,h,w} ‖RAFT(x_t, x_{t+1})_{h,w}‖_2`; `flow_entropy = -Σ_i p_i log p_i` with `p_i = softmax(\|flow\|.flatten())`; `flow_max_over_mean = flow_max / mean_flow`. NOTE: **`mean_flow` itself is ruled out** (Slide 2 H1, max \|ρ\| = 0.085); shape statistics are orthogonal. | T1 (~30 min — RAFT pipeline already deployed) | + ADA family on `flow_max_over_mean`; ≈ 0 LoRA-class | **Mean-flow extracted; max/entropy/ratio not** | All three shape statistics \|ρ\| ≤ 0.10 across all 6 methods |
| C-2 | `hf_energy_ratio` (H-T1-3) — 3D-spatiotemporal FFT high-freq energy ratio | `( Σ_{ω : \|ω\| > 0.5·Nyquist} \|F(x)\|² ) / ( Σ_ω \|F(x)\|² )` for a 3D (T×H×W) real FFT of the luma channel of the TTA-visible window. | T1 (~10 min CPU) | + LoRA-class; ≈ 0 AdaSteer | **Not implemented** — `scripts/extract_fft_features.py` (~80 LOC) | \|ρ\| ≤ 0.10 on all four LoRA-class methods |
| C-3 | `bpp_h264`, `bpp_png_avg` (H-T1-2) — lossless-compression bits-per-pixel | `bpp_h264 = file_size_bytes × 8 / (T × H × W)` from the Panda mp4; `bpp_png_avg = (1/T) Σ_t PNG_size(frame_t) × 8 / (H × W)` to disentangle inter-frame redundancy from intra-frame complexity. | T1 (~5 min CPU — one `ffprobe` + `cv2.imencode`) | + ρ ≥ 0.4 with Bucket-A diffusion-OOD score; \|ρ\| ≤ 0.15 with ΔPSNR alone | **Not implemented** — `scripts/extract_bpp_features.py` (≤ 80 LOC) | ρ(bpp, diffusion-OOD) < 0.3 across N=999 (refutes the Serrà et al. confound *in this VAE-latent setting* and frees Bucket A from needing the covariate adjustment) |
| C-4 | `cut_count_pyscenedetect`, `cut_count_histogram`, `cut_density_per_frame` (H-T1-6) — scene-cut count inside the TTA-visible window | PySceneDetect ContentDetector cuts on the TTA-visible window; backup histogram-based count with Bhattacharyya threshold 0.40. | T1 (~bundled in the existing feature pipeline) | Fisher exact OR ≥ 3.0 on `LORA_R8_TTA` tail; OR ≈ 1.0 on ADA | **Extracted** — feature already in `extract_video_features_for_tta.py`; never correlated against ΔPSNR | Fisher OR ≤ 1.5 on both LoRA-r8 methods (refutes the non-stationary-landscape mechanism in this setting) |
| C-5 | `dino_temporal_l2_mean` (Theme D, DINOv2 temporal change) | Mean L2 of consecutive-frame DINOv2 feature embeddings on the TTA-visible window. Semantic-motion proxy where RAFT mean-flow is null. | T1 (bundled) | uncertain (probe) | **Extracted** — feature already in pipeline | \|ρ\| ≤ 0.10 across all 6 methods |
| C-6 | `laplacian_variance_mean` (Theme D, frame sharpness) | Mean of `Var(LaplacianFilter(frame))` over the TTA-visible window. | T1 (bundled) | uncertain (probe) | **Extracted** | \|ρ\| ≤ 0.10 across all 6 methods |
| C-7 | `rgb_histogram_entropy_mean` (Theme D, colour entropy) | Mean per-frame Shannon entropy of the joint RGB histogram (default `bins_per_channel = 8`). | T1 (bundled) | uncertain (probe) | **Extracted** | \|ρ\| ≤ 0.10 across all 6 methods |

**Cross-cutting notes:**

- **Flow distribution shape vs flow mean.** RAFT mean-flow is ruled out (Slide 2). The shape statistics in C-1 are *not* mean-flow — they're orthogonal aggregates of the same per-pixel field. The SAR argument (Niu et al. ICLR 2023) is that if loss is dominated by a handful of high-loss pixels per frame, the gradient is dominated by those pixels and the rest of the clip is wasted compute. Localised fast motion (`flow_max_over_mean` large) is that regime; uniform fast motion (`flow_max_over_mean ≈ 1`) is not. **The mean-flow null does not rule out the flow story; it rules out the *integral* of the flow story.**
- **bpp is dual-purpose.** It is both (i) the covariate that Bucket A needs to partial out, and (ii) a candidate predictor in its own right. The Phase-1 univariate analysis (PLAN §3.2) reports both `ρ(bpp, ΔPSNR)` and `ρ(diffusion-loss, ΔPSNR | bpp)`. If bpp's own ρ with ΔPSNR is non-null at |ρ| > 0.10, Bucket C wins on a free feature — which is the cheapest possible deployable gate.
- **Scene cuts are a Bucket C / Bucket B span.** C-4 is in Bucket C by construction (it's a model-independent video statistic) but its mechanism story is Bucket B (non-stationary loss landscape). We expect C-4 to be the strongest predictor of the catastrophic *tail* among Bucket C features, mirroring B-2's prediction on the model-conditional side.

### C.4 What it means if Bucket C wins (~1 min)

**Paper claim (if some Bucket C feature clears Bonferroni):**

> "*Per-video TTA suitability is data-intrinsic — predictable from raw video features without a single diffusion forward pass.*"

- **This is the cheapest deployable gate.** Feature compute ≤ 30 min / 999 videos for the entire Bucket C battery; per video it's < 0.5 s on a CPU. No GPU at deployment time. The gate runs on an edge device.
- **Best applied-ML paper narrative.** A reviewer who cares about practical contribution will fund this bucket: it's the only one where the deployment-time gate is genuinely free.
- **Paper subsection structure:**
  - Title: "*A data-intrinsic gate for video diffusion TTA.*"
  - Headline plot: per-feature ρ leaderboard across all 6 methods for the 7 Bucket-C features. Bonferroni-cleared cells highlighted.
  - Ablation: which Bucket-C feature dominates which method family? `flow_max_over_mean` predicted to dominate on ADA; `hf_energy_ratio` on LoRA-class; `cut_count_pyscenedetect` on the catastrophic LoRA-r8 tail.
  - **Negative result if Bucket C is null:** "*No model-independent video feature predicts per-video TTA gain at this scale; gating requires at least a single forward pass (Bucket A or D) or a single TTA step (Bucket B).*"
- **Method-agnostic.** Bucket C features apply to any TTA recipe — AdaSteer, LoRA, TinyLoRA, future methods. The gate built on Bucket C is portable.

### C.5 Falsification calendar for Bucket C (~30 s)

What kills Bucket C at each phase:

- **Dies in Phase 1 (univariate) if:** every C-* feature has |ρ| ≤ 0.10 with ΔPSNR on all six methods AND Fisher exact OR ≤ 1.5 on the LoRA-r8 catastrophic tail for C-4. Note: this would also imply ρ(bpp, diffusion-OOD-score) < 0.3, freeing Bucket A from the Serrà et al. confound — that itself is a useful negative result.
- **Survives Phase 1 in a partial way if:** C-2 (`hf_energy_ratio`) carries on LoRA-class but C-1 / C-5 / C-6 / C-7 don't — that confirms the diffusion-spectral-autoregression story (Dieleman, Yu et al., FADE) and licenses a narrower paper claim.
- **Survives Phase 1 in a partial way if:** C-4 (scene-cuts) clears Fisher OR ≥ 3 on the LoRA-r8 catastrophic tail but no other Bucket-C feature is non-null. That is a *bridging* result: scene-cuts are formally a Bucket-C feature but their mechanism is Bucket-B (non-stationary landscape). The paper claim becomes "*scene cuts predict catastrophic LoRA-r8 failures via the non-stationary-loss-landscape mechanism*", which is a compact mechanism-plus-feature story.
- **Wins outright if:** C-2 has |ρ| ≥ 0.15 with ΔPSNR on at least two LoRA-class methods AND C-4 has Fisher OR ≥ 3 on the catastrophic tail.

### C.6 Worked example — what we'd say about `panda_0461` under Bucket C (~30 s)

`panda_0461` (iPhone-on-a-desk, baseline PSNR 14.04, mean_flow 0.071). Under the Bucket-C hypothesis:

- `flow_max_over_mean` is high (a static scene with a few localised moving elements — the cursor on the laptop, the steam from the coffee cup, the rustling sticky note). High concentration of motion difficulty.
- `hf_energy_ratio` is moderate-to-high (sharp text on the sticky note + on the laptop screen — text is the canonical high-frequency content for natural-image diffusion models).
- `bpp_h264` is low (a static scene compresses well — the H.264 encoder gets to reuse most blocks across frames). This makes `panda_0461` an interesting bpp-vs-ΔPSNR data point — we expect bpp low, ΔPSNR high, so the per-video bpp ρ with ΔPSNR is NOT predicted positive across the population, only on this kind of clip.
- `cut_count_pyscenedetect` = 0 (no scene cuts). NOT a catastrophic-tail candidate.

What Bucket C *predicts about `panda_0461`*: high-`hf_energy_ratio` videos should land in the LoRA-class top-decile of ΔPSNR. The Phase-1 univariate plot of `hf_energy_ratio` vs ΔPSNR on `LORA_R8_TTA` is the headline check.

What Bucket C *doesn't* explain about `panda_0461`: why it's *also* a top-10 loser under `TL_BARE_R2`. That's the same TinyLoRA-method-specific story as before — Bucket B territory, not C.

### C.7 The bpp dual-role discussion (~30 s)

Bucket C contains a feature (bpp) that the parallel diffusion-OOD experiment requires as a *covariate*. This is unusual enough to flag explicitly.

- **As a Bucket-C feature:** bpp is a free-standing complexity predictor. The HYPOTHESES H-T1-2 prediction is |ρ(bpp, ΔPSNR)| ≤ 0.15 (i.e., bpp is *not* a strong direct ΔPSNR predictor — it predicts complexity, not gain).
- **As a Bucket-A covariate:** bpp confounds pixel- and latent-space generative likelihoods per Serrà et al. (ICLR 2020). Without subtracting bpp from the diffusion-loss-OOD score, Bucket A is partly measuring "raw input complexity" rather than "unfamiliar to the model". H-T1-2 predicts ρ(bpp, diffusion-OOD-score) ≥ 0.4.
- **The discriminator** (a single 2×2 sub-table in the paper) tells us whether the Serrà et al. confound applies at this VAE-latent setting. If ρ(bpp, diffusion-OOD-score) < 0.3, the confound is rejected for our setting; if ≥ 0.4, we report partial correlations throughout Bucket A.
- **Concrete deliverable for the paper.** A four-row table:
  1. ρ(bpp, ΔPSNR) — should be near zero.
  2. ρ(diffusion-loss, ΔPSNR) — the Bucket A headline number.
  3. ρ(bpp, diffusion-loss) — confound test.
  4. ρ(diffusion-loss, ΔPSNR | bpp) — the partial correlation that licenses the Bucket A claim if (3) is high.
  Cost: zero new cluster compute (all four come from joining bpp_features.csv with diffusion_ood_scores.csv with per_video_gains.csv).

### C.8 Connections to the broader AdaSteer paper (~30 s)

If Bucket C wins (specifically C-2 `hf_energy_ratio` or C-4 scene cuts):

- **Methods section.** Add: per-video feature extraction is a *one-time* pre-deployment cost; the gate is applied at inference time without any GPU forward pass on the base model. This is the *only* deployment scenario where TTA gating is cheaper than TTA itself.
- **Results section.** Add: a figure showing the cost-aware Pareto frontier with the Bucket-C strategies in the bottom-left corner (lowest compute, smallest gain) and the Bucket-A/B strategies in the upper-right (highest compute, largest gain). The knee of the frontier is the recommended strategy.
- **Discussion section.** Connects to the diffusion-spectral-autoregression literature (Dieleman 2024, Yu et al. 2025, FADE CVPR 2025) — explains why high-frequency content is the under-fit band that LoRA can specifically target.
- **Conclusion.** "*A free deployment-time gate based on raw video spectral content recovers per-video TTA gain at zero additional inference cost.*"

## Slide 7 — Bucket D: Cross-modal alignment (~4 min)

### D.1 Guiding principle (~1 min)

**The principle:** *TTA's inference signal includes the caption. If the caption is well-aligned with the visuals, both signals reinforce each other. If misaligned, TTA can only adapt visually.*

- This is Theme E in HYPOTHESES — caption-video alignment quality.
- Two main findings shape the hypotheses:
  - **Per-sample CLIPScore has weak discriminability** (Xu et al. NeurIPS 2023, ImageReward, https://arxiv.org/abs/2304.05977): CLIPScore *averages* miss per-sample structure that *order statistics* (the min-frame score, H-T1-5) recover. The "weakest-link" caption-violation argument.
  - **CFG is per-sample and per-stage** (Jin et al. ICLR 2026 https://openreview.net/forum?id=fP0s1TEow3; Imagen team 2026 https://openreview.net/forum?id=z9YC9bvfUL; Pidstrigach 2025 https://arxiv.org/abs/2505.19367): the optimal CFG scale varies dramatically across prompts and across denoising stages — i.e., per-sample CFG gap `‖ε_cond − ε_uncond‖` is a meaningful per-sample quantity.

### D.2 Mechanism (~1 min)

**Why caption-video alignment predicts TTA gain — and the NOPROMPT subtlety:**

- The NOPROMPT result (REVIEW §2.2) rules out caption *presence*: TTA Δ within ±0.01 PSNR / ±4 FVD whether or not the caption is fed to the TTA loss; per-video ρ(Δ, caption length) for NOPROMPT ≈ −0.02.
- **But that is an aggregate, length-based finding.** Two videos with identical caption lengths can have very different alignment qualities — one might have a precise visual description, the other a generic placeholder.
- **The CLIP-min argument.** A single frame where the caption visually fails to match the content produces an outsized caption-faithful gradient under caption-conditioned TTA. The minimum over a 48-frame window captures *weakest-link* caption fidelity. CLIP-min is in `extract_video_features_for_tta.py` *already*; the correlation against ΔPSNR has simply not been computed yet.
- **The CFG-gap argument.** A low-CFG-gap video is one for which the caption is essentially generic relative to the score field. Caption-conditioned TTA on such a video degenerates to caption-dropped TTA — which is why caption *length* can be null while CFG-*gap* is non-null. The gap is a direct mechanism check for whether the TTA loss is operationally caption-aware on each video.
- **Prediction discriminator:** if Bucket D wins, the prompt-using-vs-NOPROMPT ρ *gap* should be > 0.05 — i.e., the feature has predictive power on prompted methods that it *loses* on NOPROMPT methods. That's the falsifiable signature of "alignment matters".

### D.3 Features in this bucket (~1 min)

Three features:

| # | Feature | Formula / extraction recipe | Cost tier | Expected ρ sign | Implementation status | Falsification criterion |
|---|---|---|---|---|---|---|
| D-1 | `clip_text_image_sim_min` (H-T1-5) — min per-frame CLIP cosine | Min over the 48 TTA-visible frames of CLIP-text-image cosine similarity, using `openai/clip-vit-base-patch32`. Also stored: `_mean` and `_var` for comparison. | T1 (zero marginal — already extracted at `scripts/extract_video_features_for_tta.py:760`) | + ρ ≥ +0.15 on ADA, LORA_R8_TTA, TL_BARE_R2, TL_TIED_R2; ≈ 0 on NOPROMPT; gap > 0.05 | **Extracted — needs only a correlation pass** | Both absolute \|ρ\| < 0.15 AND prompt-vs-NOPROMPT gap < 0.05 across all 4 caption-using methods |
| D-2 | `cfg_gap` (H-T2-3) — full per-(t, ε) classifier-free-guidance gap | `E_{t, ε, frame} ‖ε_θ(x_t, c, t) − ε_θ(x_t, ∅, t)‖_2 / max(‖ε_θ(x_t, ∅, t)‖_2, 1e-6)`. Needs both conditional and unconditional ε per noise sample. | T2 (~+50% over A-1's cost) | + ρ ≥ +0.15 on caption-using methods; ≈ 0 on NOPROMPT; gap > 0.10 | **Not implemented** — extra unconditional ε call per noise sample on top of the OOD scorer | Both absolute \|ρ\| < 0.15 AND prompt-vs-NOPROMPT gap < 0.05 |
| D-3 | `delta_caption_minus_uncond` (lite CFG-gap proxy) | `mean_diffusion_loss_caption − mean_diffusion_loss_uncond` — coarse CFG-gap proxy emitted for free by the OOD scorer (commit `dc115e7`). | T2 (free) | + on caption-using; ≈ 0 on NOPROMPT (weaker than D-2) | **Scaffolded** — already in the diffusion_ood_scores.csv schema | \|ρ\| < 0.10 on all 4 caption-using methods |

### D.4 What it means if Bucket D wins (~1 min)

**Paper claim:**

> "*Caption-video alignment, not video alone, predicts test-time-adaptation gain. The weakest-link CLIP-min and the per-sample classifier-free-guidance gap both clear the Bonferroni bar; the prompt-vs-NOPROMPT ρ gap is itself > 0.05.*"

- **This directly explains the existing NOPROMPT-vs-prompt result.** REVIEW §2.2 establishes that caption *presence* is a noise channel. Bucket D's win would refine that: presence is null but *quality* is signal — a non-trivial paper claim that connects two existing experimental results.
- **Paper subsection structure:**
  - Title: "*Caption alignment quality predicts per-video TTA gain even when caption presence does not.*"
  - Headline plot: per-feature ρ on caption-using methods vs the same feature on NOPROMPT methods. A diagonal line at y = x is the null; points above the line are the signature of "alignment-quality matters".
  - Ablation: D-1 (CLIP-min) vs D-2 (full CFG-gap) vs D-3 (lite-gap delta proxy). Cheaper-feature-dominates is publishable; cost-aware-feature-dominates is publishable; both null is the honest negative result.
- **Cross-modal pre-conditioning subsection.** This is the bucket that makes the AdaSteer paper's caption-conditioning ablation (Table 2 of `2026-06-09_panda_std_with_noprompt_partial.md`) into a positive story rather than a "caption is a noise channel" footnote.

### D.5 Falsification calendar for Bucket D (~30 s)

What kills Bucket D at each phase:

- **Dies in Phase 1 (univariate) if:** D-1 (CLIP-min) has |ρ| ≤ 0.10 on all 4 caption-using methods AND D-3 (`delta_caption_minus_uncond` proxy) has |ρ| ≤ 0.10. That is a tight refutation of "alignment quality matters" — both the cheap order-statistic test and the cheap CFG-gap proxy come up null.
- **Survives Phase 1 with caveats if:** D-1 carries with |ρ| ≥ 0.15 on caption-using methods AND prompt-vs-NOPROMPT gap > 0.05. We then *also* fire D-2 (full CFG-gap) per PLAN §2.5 "Defer to follow-up wave (gated on Phase 1 results)" — that needs +50% additional H200 time over A-1.
- **Wins outright if:** D-1 + D-2 both clear Bonferroni AND the prompt-vs-NOPROMPT gap is positive on all caption-using methods. That is the textbook "alignment-quality-predicts-gain" result, with the prompt-vs-NOPROMPT contrast as the falsifiable signature.

### D.6 Worked example — what we'd say about `panda_0431` under Bucket D (~30 s)

`panda_0431` ("A black background with red text on it…", baseline PSNR 31.13 — high; mean_flow 0.593 — moderate). A top-10 winner on `LORA_R8_TTA_NOPROMPT` (#4, +2.83), `TL_BARE_R2` (#3, +2.89), `TL_TIED_R2` (#2, +2.89). Under the Bucket-D hypothesis:

- The caption ("A black background with red text on it") is a *precise* visual description — high CLIP-image-text similarity per frame, high CLIP-min over the visible window.
- The CFG gap (D-2) should be moderate-to-high: the caption carries specific information about the score field (red text on black background is a sparse, predictable manifold).
- Caption-using methods (`LORA_R8_TTA` (prompted)) and NOPROMPT siblings (`LORA_R8_TTA_NOPROMPT`) both win on this video — the gain is partly caption-driven (Bucket D) and partly content-intrinsic (Bucket C — high-contrast text is high-`hf_energy_ratio` content). The Phase-1 prompt-vs-NOPROMPT gap on D-1 for *this video* should be smaller than the gap for `panda_0461` (where caption is more diffuse).

What Bucket D *doesn't* explain: why `panda_0431` is *not* a top-10 winner on `ADA` or `ADA_NOPROMPT`. AdaSteer's δ-tuning has different mechanics than LoRA's; the Bucket-D prediction is about *caption-using* methods, not about *whether AdaSteer specifically picks this video*. That asymmetry is consistent with the method-agnostic-vs-specific axis discussed in synthesis Slide 9.

### D.7 The CLIP-truncation caveat (~30 s)

CLIPScore truncates captions at 77 tokens; for very long captions a LongCLIP variant exists (https://huggingface.co/zer0int/LongCLIP-L-Diffusers). HYPOTHESES §4 Theme E note: "*at this horizon caption length is already null at |ρ| ≤ 0.06, so the truncation is not blocking*". This is a moderate liability — long captions are not over-represented in the Panda distribution, but a future iteration on longer captions would need LongCLIP-L.

### D.8 Connections to the broader AdaSteer paper (~30 s)

If Bucket D wins:

- **Methods section.** Add: caption-conditioned TTA loss preserves cross-modal alignment signal that uncaption-conditioned (NOPROMPT) TTA discards. The contribution: a *quantitative* discriminator (CLIP-min, CFG-gap) for *when* the caption channel carries useful signal, complementing the existing NOPROMPT ablation.
- **Results section.** Add: a Figure 5 that overlays the CLIP-min ρ on caption-using methods against the CLIP-min ρ on NOPROMPT siblings. Points above the diagonal: alignment-quality signal. Points on the diagonal: alignment-quality null.
- **Discussion section.** Connects to the dynamic-CFG line (Jin et al. ICLR 2026; Imagen team 2026). Frames the AdaSteer paper's caption-conditioning as a *per-sample* phenomenon rather than a global hyperparameter.
- **Conclusion.** "*Caption-video alignment quality, measured by the per-video CFG gap or the weakest-link CLIP score, predicts which videos benefit from caption-conditioned TTA.*"

---

## Slide 8 — Bucket E: Reconstruction observability (~3 min)

### E.1 Guiding principle (~1 min)

**The principle:** *if the VAE alone can't represent the video well, the latent space is already lossy — TTA gain is capped from below by the autoencoder's own reconstruction error.*

- This bucket is **proposed without a direct HYPOTHESES theme**, but it's grounded in two literatures:
  - **Latent-space typicality** (Ding et al. 2025, https://arxiv.org/abs/2504.07793; Järve et al. 2025, https://arxiv.org/abs/2508.15737): likelihood-based OOD detection works in *encoder latent* space, not pixel space. The corollary is that elevated VAE round-trip error is the signal that a video's latents lie in a non-typical region of the encoder's distribution.
  - **The autoencoder bottleneck argument** from the diffusion-model literature broadly: latent diffusion models (LDMs) optimise in the encoder's latent space; the encoder is a learned compressor; anything the encoder loses is irrecoverable by downstream operations.
- HYPOTHESES H-T1-1 is the only feature in this bucket explicitly. It cites Ding et al. and Järve et al. as the representation-space-OOD line of work, and TTL-LLM (https://arxiv.org/abs/2505.20633) for the high-perplexity-gains-more result in generative TTA.

### E.2 Mechanism (~1 min)

**Why VAE round-trip error caps TTA gain ceiling — the bottleneck argument:**

- TTA optimises in latent space (or in attention-adapter / LoRA-adapter space that maps to latent operations). The held-out generation window is decoded from latents back to pixels by the VAE decoder.
- If the VAE encoder loses information about the video (high `rec_err_l1` on the round-trip), no parameter update in latent space can recover that information.
- Therefore `rec_err` sets an **upper bound on |ΔPSNR|** (above the ceiling, gain is impossible regardless of TTA recipe).
- **Asymmetric prediction.** The prediction is positive ρ with |ΔPSNR| on ALL methods (because rec_err puts a ceiling on the *magnitude* of any latent-space change), AND positive ρ with *signed* ΔPSNR on LoRA-class methods specifically (the TTL-LLM argument: in generative TTA, high-perplexity samples gain more).

### E.3 Features in this bucket (~30 s)

Two features (one extraction recipe):

| # | Feature | Formula / extraction recipe | Cost tier | Expected ρ sign | Implementation status | Falsification criterion |
|---|---|---|---|---|---|---|
| E-1 | `rec_err_l1` (H-T1-1) — pixel-space L1 reconstruction error | `mean_{t,c,h,w} \| x_{t,h,w} − Decoder(Encoder(x))_{t,h,w} \|` on the TTA-visible 48 frames. | T1 (~25 min / 999 videos — one VAE encode + decode per video) | + with \|ΔPSNR\| on all 6 methods; + signed ΔPSNR on the 4 LoRA-class methods | **Not implemented** — `scripts/extract_vae_recerr_features.py`; reuses `load_longcat_components` from `delta_experiment/scripts/common.py` | \|ρ\| ≤ 0.10 with \|ΔPSNR\| on all 6 methods AND the signed-ΔPSNR LoRA-class effect < 0.10 |
| E-2 | `rec_err_lpips` (H-T1-1, perceptual variant) | LPIPS distance between `x` and `Decoder(Encoder(x))`, per frame, averaged. | T1 (bundled with E-1) | same as E-1 (perceptually-weighted variant) | **Not implemented** — same script | (joint with E-1) |

**Optional addition per HYPOTHESES §6 Q5:** latent-space L2 between encoded-clean latents and encoded-then-noised-then-denoised latents at small t. ~30% extra cost over pixel-space-only. Deferred to a follow-up implementation.

### E.4 What it means if Bucket E wins (~30 s)

**Paper claim:**

> "*Test-time adaptation can only fix what the autoencoder already represents. The LongCat-VAE round-trip reconstruction error per video upper-bounds \|ΔPSNR\| under every TTA recipe we ship; signed ΔPSNR on LoRA-class methods correlates positively with `rec_err_l1` (the TTL-LLM perplexity-gains-more pattern, https://arxiv.org/abs/2505.20633).*"

- **Less exciting than A or B as a standalone story** — but a **clean negative result** if it's the only thing that works. A "limits-of-TTA" subsection in the paper.
- **Paper subsection structure if E-1 wins:**
  - Title: "*The autoencoder bottleneck bounds per-video TTA gain.*"
  - Headline plot: VAE rec_err vs \|ΔPSNR\| on every method, with the predicted upper-envelope drawn as a line.
  - Connects to: Ding et al. 2025 / Järve et al. 2025 as the latent-space-typicality precedent.
- **What it does NOT do:** Bucket E does not predict *signed* per-video gain for AdaSteer (no spare capacity argument applies); it only predicts the asymmetric magnitude story. So if Bucket E is the *only* winner, the paper claim is bounded — useful for the limits-of-TTA section but insufficient as a standalone deployment-gate paper.

### E.5 Falsification calendar for Bucket E (~30 s)

What kills Bucket E at each phase:

- **Dies in Phase 1 (univariate) if:** E-1 has |ρ| ≤ 0.10 with |ΔPSNR| on all 6 methods AND the signed-ΔPSNR LoRA-class effect is also < 0.10. That refutes both the "rec_err caps |ΔPSNR| from below" claim AND the TTL-LLM "high-perplexity gains more" generative-TTA claim for our setting.
- **Wins outright if:** E-1 has |ρ| ≥ 0.15 with |ΔPSNR| on at least 4 of 6 methods AND signed ρ ≥ +0.10 on at least 2 LoRA-class methods. That confirms both the asymmetric-magnitude ceiling story AND the asymmetric-signed-gain TTL-LLM story.

### E.6 Worked example — `panda_0461` and `panda_0098` under Bucket E (~30 s)

- `panda_0461` (iPhone-on-desk, baseline 14.04 dB): the LongCat-VAE round-trip is *probably* lossy on the small text and the fine cursor details. `rec_err_l1` is predicted moderately high; |ΔPSNR| is large (+3 to +9 dB across methods). The combination is consistent with the E-1 prediction.
- `panda_0098` (text-on-white "home workshop", baseline 44.55 dB): the VAE round-trip should be excellent (high-PSNR static text is well within the VAE's training distribution). `rec_err_l1` is predicted low; |ΔPSNR| is enormous (−22.4 dB) — *inconsistent* with the E-1 hypothesis. **`panda_0098` is the canonical counter-example for Bucket E**: its catastrophic |ΔPSNR| is *not* explained by autoencoder unobservability. Bucket E predicts the modal-gain magnitude ceiling but not the geometric-overfit failure mode. This is a clean point of differentiation from Bucket B in the paper.

### E.7 Why Bucket E is the most contained (~30 s)

Bucket E has the smallest feature count (2: `rec_err_l1`, `rec_err_lpips`), the shortest implementation cost (~25 min for the full feature on 999 videos), and the most contained paper claim ("autoencoder bounds gain"). If it wins, the paper claim is narrow but clean. If it loses, no harm done — it costs less than the time to brew coffee. **The recommendation is to fire Bucket E in Phase 0 regardless of bucket priority** — it's too cheap to defer and the negative result is itself publishable.

### E.8 Connections to the broader AdaSteer paper (~30 s)

If Bucket E wins:

- **Methods section.** Add: a baseline characterisation of the LongCat-VAE bottleneck — `rec_err_l1` distribution + `rec_err_lpips` distribution across the 999-video Panda dataset. This is a *base-model property* not a TTA property; the paper acquires a small but useful base-model characterisation contribution.
- **Results section.** Add: per-method scatter of `rec_err_l1` vs \|ΔPSNR\| with the predicted upper envelope. This is a clean, single-figure visual story.
- **Discussion section.** Connects to latent-diffusion-model autoencoder bottleneck arguments (broad LDM literature) and the latent-space-OOD line (Ding et al. 2025, Järve et al. 2025). Frames the AdaSteer paper as an empirical instantiation of the autoencoder-bottleneck argument for video diffusion TTA.
- **Conclusion.** "*Test-time adaptation gain is upper-bounded by the autoencoder reconstruction error. This sets a fundamental limit on what TTA can achieve regardless of recipe.*"

---

## Slide 9 — Cross-bucket synthesis (~3 min)

The most important slide for the talk's punchline. Four points.

### 1. Modal gain vs tail risk (~45 s)

- **Buckets A / C / D / E predict the *modal* ΔPSNR** — the bulk of the distribution. They tell us which videos sit near the high-positive-Δ end of the histogram.
- **Bucket B predicts the *tail*** — both the catastrophic-failure tail (`panda_0098`-class) and the extreme-winner tail. A and C and D and E cannot predict the catastrophic failures; B is the only bucket whose mechanism story is "this video collapsed" rather than "this video improved".
- **A complete gate likely combines both.** B for tail-risk avoidance × {A, C, D, E} for modal-gain prediction. This is the multivariate Phase 2 prediction (PLAN §3.3).
- **Worked example.** A LoRA-r8 deployment gate could be:
  - **First filter (Bucket B):** screen out videos with `single_step_loss_drop > p90` AND `grad_norm_θ0 > p90`. These are the catastrophic-tail candidates. Coverage cost: ~10% of videos rejected.
  - **Second filter (Bucket A or D):** among the remainder, apply TTA only to videos with `mean_diffusion_loss_caption > median` AND `clip_text_image_sim_min > p25`. These are the modal-gain candidates. Coverage on the survivors: ~50%.
  - **Net effect:** ~45% of videos receive TTA, the catastrophic outlier is avoided, and the average gain on the gated subset is higher than the full-population average.
- **Why this matters for the paper's *defensibility*.** A single-feature gate that delivers +0.2 dB on 70% of videos but allows a `panda_0098`-class catastrophe through is *worse* than no gate at all for a deployment story — the catastrophe's −22 dB on one video is more visible than the +0.2 dB on 700 videos. The B-then-A architecture (refuse the catastrophes first, then optimise the modal gain) is the asymmetric-loss-respecting design that the paper needs.

### 2. Method-agnostic vs method-specific (~45 s)

- **Buckets A, C, D, E are method-agnostic** in the sense that the *feature* doesn't depend on which TTA recipe is being applied. `mean_diffusion_loss_caption` (A-1) is the same number whether we follow up with LoRA-r8 or AdaSteer or TinyLoRA.
- **Bucket B is method-specific.** `grad_norm_θ_lora` is meaningful for LoRA but not for AdaSteer's δ-tuning. `single_step_loss_drop` is computed *on the LoRA adapter's optimiser*. If we want a method-agnostic gate, Bucket B must be either restricted to AdaSteer-versions of the same probes (replace LoRA with the AdaSteer δ-tuning parameter set) or generalised to a `grad_norm_θ_full` that doesn't refer to any specific adapter.
- **Trade-off for the paper:**
  - **Method-agnostic gate is more deployable.** One feature CSV, used to gate any TTA method. Cheaper to compute, simpler to ship.
  - **Method-specific gate is more accurate.** Bucket B's `single_step_loss_drop` is the *cheapest* (≈ 2 / N_TTA-steps of a full LoRA-r8 run) catastrophic-tail predictor we know of. Hard to beat with a model-agnostic alternative.
- **Recommended framing.** Report both. A method-agnostic Bucket-{A/C/D} gate is the headline "deployment rule". A method-specific Bucket-B probe is the "safety-check before LoRA-r8" auxiliary contribution.

### 3. What we'd predict about the gating-experiment outcome (~60 s)

Best guess with uncertainty, sorted by predicted strength:

- **Bucket A — moderate.** A-1 (`mean_diffusion_loss_caption`) probably has |ρ| in the 0.10–0.20 range on LoRA-class methods after bpp partial-out. May or may not clear Bonferroni at |ρ| ≥ 0.13. A-3 (`score_norm_t*`) is the dark horse; if score-magnitude dominates loss-mismatch, A-3 carries.
- **Bucket B — strong on tail, moderate on mean.** B-2 (`single_step_loss_drop`) almost certainly has ρ ≤ −0.20 with `LORA_R8_TTA` ΔPSNR conditional on the catastrophic-outlier mechanism being real. Modal-gain prediction is weaker.
- **Bucket C — mostly null but with one likely winner.** Most Bucket-C features (RGB-hist entropy, Laplacian variance) probably do not survive Bonferroni. **`hf_energy_ratio` (C-2)** is the prediction: LoRA-class positive, AdaSteer near-zero. **`cut_count_pyscenedetect` (C-4) × tail asymmetric:** Fisher OR ≥ 3 on LoRA-r8.
- **Bucket D — moderate.** D-1 (CLIP-min) is the cheapest test; expected |ρ| around 0.10–0.18 on caption-using methods, with the prompt-vs-NOPROMPT gap being the discriminator. D-2 (full CFG-gap) is the most expensive caption test; only worth firing if D-3 (`delta_caption_minus_uncond` proxy) shows non-zero signal.
- **Bucket E — long shot.** E-1 (`rec_err_l1`) probably correlates weakly (|ρ| around 0.05–0.10). If it works, it's a clean limits-of-TTA paper. If it doesn't, no harm — the cost was 25 minutes of one H200.

**Confidence calibration.** These predictions are at the level of "rank-order best guesses, with substantial uncertainty on individual cells". The Phase-1 univariate analysis will resolve every single cell; we'll know within ~1 week of cluster restart. Story C of the gating plan ("no win") is on the table — Bonferroni at 360 cells with N=999 is genuinely demanding.

### 4. The "ensemble gate" hypothesis (~45 s)

- **Hypothesis (Phase 2 of the gating experiment, PLAN §3.3):** a 2-feature gate combining Bucket B for tail avoidance × Bucket A for modal gain dominates any single feature.
- **Concrete prediction.** A linear logistic regression with features `[single_step_loss_drop, mean_diffusion_loss_caption]` on the held-out fold has higher AUC than either feature alone, by a margin of ≥ 0.05 AUC.
- **Cheaper variant.** Replace Bucket A with Bucket C's `hf_energy_ratio` — if that variant clears the AUC bar at lower compute cost, it dominates on the cost-aware Pareto frontier (PLAN §3.4).
- **Why ≥ 0.05 AUC is the right threshold to call this a "win".** The univariate-best feature is expected to deliver AUC in the range 0.55–0.65 (modest separation; consistent with Bonferroni |ρ| in the 0.13–0.20 band). A 2-feature ensemble that delivers AUC > 0.68 means at least one feature is contributing *non-redundant* information — i.e., Buckets are mechanistically distinct, not redundant measurements of the same latent variable. The +0.05 margin is the standard "the variance reduction is real and not noise" bar for held-out AUC at N=999 with 10-fold CV.
- **The paper claim if the ensemble wins:** "*A two-feature gate combining a Bucket-B tail-risk probe with a Bucket-A modal-gain score recovers per-video TTA gain that no single feature captures.*"
- **The paper claim if the ensemble does NOT win:** "*Per-video TTA outcomes are driven by a single latent factor — different measurement modalities (model loss vs visual complexity vs caption alignment vs autoencoder error) are redundant proxies for the same underlying property.*" This is *also* a publishable result; it argues for parsimony in the deployment gate.

### 5. Cross-bucket prediction matrix (~30 s)

The most useful single artefact for the paper. For each (bucket, TTA method, outcome variable) cell, a one-character prediction:

| Bucket → | A | B | C | D | E |
|---|---|---|---|---|---|
| ADA (modal gain) | ? | – | – | + | – |
| ADA_NOPROMPT (modal gain) | ? | – | – | 0 | – |
| LORA_R8_TTA (modal gain) | + | + | + | + | + |
| LORA_R8_TTA (catastrophic tail) | – | **+** | + (cuts only) | – | – |
| LORA_R8_TTA_NOPROMPT (modal gain) | + | + | + | 0 | + |
| TL_BARE_R2 (modal gain) | + | + | + | + | + |
| TL_TIED_R2 (modal gain) | + | + | + | + | + |

Reading: "+" = positive ρ prediction; "–" = predicted near-zero; "0" = predicted to be a *null* (mirror image of a positive cell elsewhere — useful as the contrast); "?" = uncertain ahead of experiment. **Bold cells** are the asymmetric-tail predictions where a winning bucket would deliver something no other bucket can. **The single most important cell is `LORA_R8_TTA (catastrophic tail) × Bucket B`** — that is the prediction that licenses the strongest paper claim if it holds.

### 6. Why this taxonomy works for paper presentation (~30 s)

- **One bucket = one paper subsection.** A reader scanning the paper's Table of Contents sees "5.1 OOD-correction (Bucket A)" / "5.2 Loss-landscape geometry (Bucket B)" / etc. — a clean information hierarchy that maps to the talk's organisation.
- **The non-winning buckets become the *negative results* section.** "We tested Bucket E (autoencoder rec-error); it did not survive Bonferroni at this scale; here's the |ρ| value and the falsification interpretation." This is paper-defensible — reviewers reward thorough negative-result coverage.
- **The 5-bucket structure is robust to which subset of buckets win.** If only A wins, the paper claim is "TTA is OOD-correction". If only B wins, it's "TTA gain = loss-landscape geometry". If A+B both win, it's "OOD-correction × loss-landscape ensemble". Each combination has a coherent narrative.
- **The taxonomy generalises beyond LongCat-Video.** Future work on Wan / HunyuanVideo / CogVideoX would apply the same 5-bucket organisation. The principles are model-independent even when the specific features (e.g., adapter architecture for B-1) are model-dependent.

---

## Slide 10 — Recommendation: what to bet on (~2 min)

"If you only fund one feature, fund this one" closing slide. Four scenarios, four recommendations, depending on which question the audience cares about.

| Question | Bet on | Why |
|---|---|---|
| **Most likely to publish a CVPR-tier mechanism story** | **Bucket B** (loss-landscape geometry — `single_step_loss_drop`) | Steep mechanism, principled, novel for video diffusion, captures the catastrophic-failure tail. DreamBooth precedent (CVPR 2023) is well-cited. Catastrophic-tail prediction is the asymmetric story no other bucket can tell. |
| **Most likely to be the cheapest deployable gate** | **Bucket C** (visual complexity — `hf_energy_ratio` + `flow_max_over_mean` + `cut_count_pyscenedetect`) | Free at deployment time — no model forward needed. Edge-device-friendly. If any Bucket-C feature works, the paper has a practical contribution that A/B/D/E cannot match. |
| **Most likely to confirm the user's stated hypothesis** | **Bucket A** (model-perceived difficulty — caption-conditioned diffusion loss) | Already implemented and partially run (`compute_diffusion_ood_score.py`, commit `dc115e7`); this is the experiment the parallel workstream is already executing. Highest probability of *some* non-null result. |
| **Most likely to produce a clean negative result** | **Bucket E** (reconstruction observability — `rec_err_l1`) | Either a tight upper-bound paper ("TTA can only fix what the autoencoder represents") or "ruled out, here's why". 25 minutes of one H200 to find out. Low risk, low ceiling. |

**Operationally what we're actually running** (PLAN §3.1 Phase 0):

- All Bucket-A features (A-1, A-2, A-3, A-4, A-5) — single forward pass through the OOD scorer, including the score-norm patch and the loss-variance derivation.
- All Bucket-C features (C-1 through C-7) — bundled in the Tier-1 feature pipeline.
- All Bucket-D features (D-1, D-2, D-3) — D-1 is free, D-2 has been deferred to a follow-up wave pending D-3 signal (PLAN §2.5).
- All Bucket-E features (E-1, E-2) — new ~80-LOC script.
- All Bucket-B features (B-1, B-2, B-3) — Decision 4 in PLAN §8 explicitly scheduled the Tier-3 probes for Phase 0.

**Wallclock budget:** ~1 day post-cluster-maintenance for the complete Phase-0 deliverable. Phase 1 (univariate) and Phase 2 (multivariate) are CPU-only and run on the login node. Phase 3 (Pareto + RECOMMENDATION.md) is the human-in-the-loop authorisation gate for Phase 4 (long-horizon validation).

**We will know within a week of the cluster coming back which of these buckets to bet on.**

**One-sentence recommendations per bucket** (for the slide footer):

- **Bucket A — already running.** No additional decision needed; A-1 / A-2 / A-3 / A-5 are scheduled for Phase 0 via the OOD scorer. Watch for non-null residual after bpp partial-out.
- **Bucket B — fund both probes.** B-1 (`grad_norm_θ0`) AND B-2 (`single_step_loss_drop`) are scheduled for Phase 0 per PLAN §8 Decision 4. The +~2 GPU-hours per 999-video run is the best per-hour bet of any feature in the menu.
- **Bucket C — fire the three orthogonal-to-mean-flow features.** C-1 (flow distribution shape), C-2 (HF FFT ratio), C-4 (scene cuts) are the three with sharp predictions. C-3 (bpp) is required as a Bucket-A covariate regardless. C-5 / C-6 / C-7 are bundled-cost probes; we report their ρ but expect null.
- **Bucket D — start with the free correlation.** D-1 (CLIP-min) costs zero — it's already extracted; the correlation pass against `per_video_gains.csv` lands in Phase 1. D-3 (lite-gap proxy) is free from the OOD scorer. Defer D-2 (full CFG-gap) to a follow-up wave if D-1 + D-3 give non-null signal.
- **Bucket E — fire the cheap version unconditionally.** E-1 + E-2 cost ~25 min of one H200. Negative result is publishable as a limits-of-TTA story.

### Three-tier closing message

The talk closes with three lines, in decreasing order of confidence:

1. **(High confidence)** *"Population-level metrics saturate, but per-video TTA outcomes are not random — they sign-agree across methods at 6.3× the null lift. There IS per-video structure to predict."*
2. **(Medium confidence)** *"The 5-bucket taxonomy will determine which paper subsection wins. Best guess: Bucket B (loss-landscape geometry) predicts the catastrophic tail, Bucket A (likelihood / OOD) predicts the modal gain, and the ensemble of B × {A or C} dominates any single feature."*
3. **(Operational)** *"Phase-0 cluster jobs fire when the cluster comes back next week. The Phase-3 RECOMMENDATION.md lands ~3.5 days later. We will know which bucket wins by ~2026-06-15."*

---

## Slide 11 — Limitations and open questions (~1 min)

Five honest limitations to state before Q&A.

1. **Buckets are a soft taxonomy.** Some features blur. CFG-gap (D-2) is mostly Bucket D (caption-aware) but also has a Bucket-A flavour (it's an ε-field property). FLIPD (A-4) is mostly Bucket A (diffusion-model density) but it's measuring intrinsic dimensionality, which is conceptually Bucket C. Scene cuts (C-4) are mostly Bucket C (model-independent video statistic) but their mechanism story is Bucket B (non-stationary loss landscape). The 5-bucket assignment in the appendix is my best deterministic call; the spans are flagged in each slide. **No feature is force-bucketed against its primary mechanism**; spans get a "primary + secondary" label, not a renaming.
2. **Saturation could be horizon-specific.** All evidence cited is at Panda 1000v / 480p / **17-frame standard horizon**. The long-horizon (76-frame) per-video bundle is queued (REVIEW §4 "Long-horizon (76-frame) regime may show structure the 17-frame regime doesn't") and will tell us whether the saturation pattern survives. PLAN Phase 4 fires conditionally on Phase 3 to validate cross-horizon. If long-horizon has more structure, the 5-bucket conclusions may change. The 2026-06-11 user hypothesis (REFRESHER doc) is that long-horizon has *fatter tails in both directions* even when the population mean is unchanged — if confirmed, the per-video story is sharper at long horizon than at the standard horizon we built this taxonomy against.
3. **Method-agnostic-vs-method-specific is orthogonal and important.** This axis cuts across the 5 buckets (Bucket B is method-specific; A/C/D/E are method-agnostic) and affects the paper's framing. Synthesis Slide 9 §2 sketches the trade-off; the full discussion belongs in the Methods section of the paper. A reviewer asking "*why didn't you organise by method-specificity instead of by principle?*" has a fair point — that taxonomy would also work, would put Bucket B alone on one side and {A, C, D, E} on the other. We chose the principle taxonomy because it maps to *paper subsections* more cleanly.
4. **Multivariate gates may dominate any single feature.** Phase 2 of the gating experiment is the actual experimental question for the paper. Single-feature ρ values are necessary screens, not the final answer. The "ensemble gate" hypothesis on Slide 9 §4 is the bet. The danger: if every bucket's univariate ρ is < 0.10 but the multivariate AUC is > 0.65, the paper has to argue "*we measured features individually weak but jointly strong*" — defensible but harder to motivate to a sceptical reviewer.
5. **Compute-cost organization is still relevant operationally even if not theoretically central.** The PLAN's Tier-1 / Tier-2 / Tier-3 organisation tells us *what to run first*; the 5-bucket taxonomy tells us *what to write about*. Both axes matter; they organise different decisions. The Phase-0 sbatch wrappers don't care about buckets — they care about which features share a forward pass.

**Open questions for advisors / Q&A:**

- Should the paper's headline figure use the 5-bucket organisation or the Tier-1/2/3 organisation? Current preference: 5-bucket for the *main text*, Tier-1/2/3 for the *appendix* (compute disclosure for replication).
- Is there a 6th bucket we missed? Plausible candidates: (i) "trajectory-time-derivative" features (DiffPath rate-of-change is mostly Bucket A but partly a separate principle); (ii) "ε-prediction confidence" features (could split off from Bucket A as a 5b). Current judgment: both are sub-mechanisms within Bucket A, not separate principles. Open to argument.
- Should Bucket E be promoted to "primary" (i.e., reported in the main text) or relegated to "appendix-only"? Current preference: main text if E-1 wins outright; appendix-only if E-1 is null. The decision is made by the result, not preemptively.
- What's the right way to discuss the `panda_0098` outlier? Footnote vs subsection vs figure? Current preference: figure callout in the Bucket B subsection, since `panda_0098` is the mechanistic anchor for the catastrophic-tail story. Treating it as a footnote understates its diagnostic role.

---

## Slide 12 — Appendix: full feature inventory (~rest of doc)

Comprehensive table of ALL features across all 5 buckets. This is the appendix slide that everyone photographs at the end. Sources are HYPOTHESES doc + PLAN doc master feature menu.

### A. Model-perceived difficulty (5 features)

| Feature | Bucket | Source theme | Formula | Cost tier | Expected ρ sign | Implementation status | Falsification criterion | Supporting citation |
|---|---|---|---|---|---|---|---|---|
| `mean_diffusion_loss_caption` (A-1) | A | Theme B | `E_{t, ε} ‖ε − ε_θ(x_t, c, t)‖²` at t ∈ {100, 500, 900}; partial-out bpp | T2 | + LoRA-class; ambiguous AdaSteer | Scaffolded (`compute_diffusion_ood_score.py`, commit `dc115e7`) | Residual \|ρ\| ≤ 0.10 with ΔPSNR on all 4 LoRA-class methods after partialling out bpp | Graham et al. CVPRW 2023 (https://arxiv.org/abs/2211.07740); Pinaya et al. 2022 (https://arxiv.org/abs/2207.13726) |
| `mean_diffusion_loss_uncond` + `delta_caption_minus_uncond` (A-2) | A (partial D affinity) | Theme B | Same as A-1 with ∅ caption; `delta` = caption − uncond | T2 (bundled with A-1) | comparator for A-1; weak alignment proxy | Scaffolded (same OOD CSV) | \|ρ\| ≤ 0.10 on caption-using methods AND delta < 0.10 | Same as A-1 |
| `score_norm_t*` (A-3) | A (secondary B) | Theme B | `E_ε ‖ε_θ(x_{t*}, c, t*)‖² / (T·C·H·W)` at t* ∈ {200, 500, 800} | T2 (free with A-1) | + LoRA-class; ≈ 0 AdaSteer | ≤ 30-line patch to OOD scorer | \|ρ\| ≤ 0.10 across all 4 LoRA-class methods AND A-1 has \|ρ\| > 0.15 (loss-mismatch dominates) | Barkley et al. 2025, SCOPED (https://arxiv.org/abs/2510.01456); Heng et al. NeurIPS 2024, DiffPath (https://arxiv.org/abs/2405.11881) |
| `lid_flipd` (A-4) | A (secondary C) | Theme B + D | LID via Fokker-Planck Hessian-trace at small t (Kamkari et al. eq. 7) | T2 (+30% over A-1) | + LoRA-class; ≈ 0 AdaSteer | Not implemented | \|ρ\| ≤ 0.10 across all 4 LoRA-class methods | Kamkari et al. NeurIPS 2024 (https://arxiv.org/abs/2406.03537) |
| `latent_norm_mean`, `latent_norm_std`, `latent_kurtosis` (A-5) | A | Theme B | First three moments of `‖z‖` over the encoded latent of the TTA-visible window | T2 (bundled) | uncertain (probe) | Scaffolded (emitted by OOD scorer) | All three \|ρ\| ≤ 0.10 across all 6 methods | Ding et al. 2025 (https://arxiv.org/abs/2504.07793); Järve et al. 2025 (https://arxiv.org/abs/2508.15737) |

### B. Loss-landscape geometry (3 features + 1 cross-reference)

| Feature | Bucket | Source theme | Formula | Cost tier | Expected ρ sign | Implementation status | Falsification criterion | Supporting citation |
|---|---|---|---|---|---|---|---|---|
| `grad_norm_θ0` (B-1) | B | Theme A + C + G | `‖∇_{θ_LoRA} L_diff(x; θ₀)‖_2` — one forward + one backward at unadapted weights | T3 (~30 min / 999 videos full; ~3 min stratified) | + with \|ΔPSNR\| LoRA-class; − with signed ΔPSNR on LORA_R8_TTA | Scheduled Phase 0 per PLAN §8 Decision 4 (Tier-3 runner `compute_tier3_probes.py`) | \|ρ\| ≤ 0.15 with \|ΔPSNR\| on LORA_R8_TTA (refutes SAR's classifier→diffusion mapping for this setting) | Niu et al. ICLR 2023, SAR (https://openreview.net/pdf?id=g2YraF75Tj); Garg & Roy CVPR 2023, SLo-Curves (https://cvpr.thecvf.com/virtual/2023/poster/20980); Kwon et al. ICLR 2024, DataInf (https://arxiv.org/abs/2310.00902); Mlodozeniec et al. ICLR 2025 (https://arxiv.org/abs/2410.13850) |
| `single_step_loss_drop` (B-2) | B | Theme G | `(L_diff(x; θ₀) − L_diff(x; θ₀ + Adam_step)) / max(L_diff(x; θ₀), 1e-6)` on the LoRA-r8 adapter | T3 (~30 min / 999 videos full; ~3 min stratified) | strongly negative ρ ≤ −0.20 with signed LORA_R8_TTA ΔPSNR; `panda_0098` predicted in top-10% | Scheduled Phase 0 per PLAN §8 Decision 4 | \|ρ\| ≤ 0.10 with LORA_R8_TTA ΔPSNR (refutes DreamBooth-collapse mechanism for LoRA-r8 in this setting) | Ruiz et al. CVPR 2023, DreamBooth (https://arxiv.org/abs/2208.12242); Ye et al. EMNLP 2023, Anti-CF (https://aclanthology.org/2023.emnlp-main.803.pdf); Liu et al. 2025, ZeroSiam (https://arxiv.org/abs/2509.23183) |
| `loss_var_t` (B-3) | B (secondary A) | Theme A | `Var_{t ∈ {100, 500, 900}} E_{ε, frame} ‖ε − ε_θ(x_t, c, t)‖²` — reuses A-1's per-(t, ε) losses | T2 (~free post-processing) | + with ΔPSNR ≥ 0.15 on LoRA-class methods | Derivable (`scripts/derive_loss_variance.py` ≤ 50 LOC) | \|ρ\| ≤ 0.10 with ΔPSNR on all 4 LoRA-class methods AND A-1 mean-loss has \|ρ\| > 0.15 (absolute level dominates over variance) | Niu et al. ICML 2022, EATA (https://proceedings.mlr.press/v162/niu22a/niu22a.pdf); Sun et al. 2025, TTL-LLM (https://arxiv.org/abs/2505.20633) |
| `score_norm_t*` (cross-ref from A-3) | A (secondary B) | Theme B | See A-3 | T2 (free) | See A-3 | See A-3 | See A-3 | See A-3 |

### C. Visual / temporal complexity (7 features)

| Feature | Bucket | Source theme | Formula | Cost tier | Expected ρ sign | Implementation status | Falsification criterion | Supporting citation |
|---|---|---|---|---|---|---|---|---|
| `flow_max`, `flow_entropy`, `flow_max_over_mean` (C-1) | C (secondary B via SAR) | Theme D + A | Distribution-shape statistics of per-pixel RAFT optical flow over the TTA-visible window | T1 (~30 min RAFT) | + ADA family on max/mean; ≈ 0 LoRA-class | Mean-flow extracted (`dynamic_degree.json`); max/entropy/ratio not | All three shape statistics \|ρ\| ≤ 0.10 across all 6 methods | Teed & Deng ECCV 2020, RAFT (https://arxiv.org/abs/2003.12039); Niu et al. ICLR 2023, SAR (https://openreview.net/pdf?id=g2YraF75Tj) |
| `hf_energy_ratio` (C-2) | C | Theme D | `( Σ_{ω : \|ω\| > 0.5·Nyquist} \|F(x)\|² ) / ( Σ_ω \|F(x)\|² )` for a 3D (T×H×W) real FFT of luma | T1 (~10 min CPU) | + LoRA-class; ≈ 0 AdaSteer | Not implemented (`scripts/extract_fft_features.py`) | \|ρ\| ≤ 0.10 on all four LoRA-class methods | Dieleman 2024 (https://sander.ai/2024/09/02/spectral-autoregression.html); Yu et al. 2025, Spectral Progressive Diffusion (https://arxiv.org/abs/2605.18736); Zhu et al. CVPR 2025, FADE (https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_FADE_Frequency-Aware_Diffusion_Model_Factorization_for_Video_Editing_CVPR_2025_paper.pdf) |
| `bpp_h264`, `bpp_png_avg` (C-3) | C (also confound for A) | Theme B + D | `bpp_h264 = file_size_bytes × 8 / (T × H × W)` from the mp4; `bpp_png_avg` per-frame PNG bytes | T1 (~5 min CPU) | + ρ ≥ 0.4 with Bucket-A diffusion-OOD; \|ρ\| ≤ 0.15 with ΔPSNR alone | Not implemented (`scripts/extract_bpp_features.py` ≤ 80 LOC) | ρ(bpp, diffusion-OOD) < 0.3 across N=999 (refutes Serrà et al. confound in this VAE-latent setting) | Serrà et al. ICLR 2020 (https://arxiv.org/abs/1909.11480) |
| `cut_count_pyscenedetect`, `cut_count_histogram`, `cut_density_per_frame` (C-4) | C (secondary B) | Theme D + A | PySceneDetect ContentDetector cuts on the TTA-visible window; histogram-based backup at Bhattacharyya threshold 0.40 | T1 (bundled in existing pipeline) | Fisher exact OR ≥ 3.0 on LORA_R8_TTA tail; OR ≈ 1.0 on ADA | Extracted (`extract_video_features_for_tta.py`); never correlated against ΔPSNR | Fisher OR ≤ 1.5 on both LoRA-r8 methods | Niu et al. ICLR 2023, SAR (https://openreview.net/pdf?id=g2YraF75Tj); Gong et al. NeurIPS 2023, SoTTA (https://proceedings.neurips.cc/paper_files/paper/2023/file/2da53cd1abdae59150e35f4693834f32-Paper-Conference.pdf) |
| `dino_temporal_l2_mean` (C-5) | C | Theme D | Mean L2 of consecutive-frame DINOv2 feature embeddings (`facebook/dinov2-small`) on TTA-visible window | T1 (bundled) | uncertain (probe) | Extracted | \|ρ\| ≤ 0.10 across all 6 methods | Caron et al. 2021, DINO (https://arxiv.org/abs/2104.14294) — DINOv2 is the v2 follow-up |
| `laplacian_variance_mean` (C-6) | C | Theme D | Mean `Var(LaplacianFilter(frame))` over the TTA-visible window | T1 (bundled) | uncertain (probe) | Extracted | \|ρ\| ≤ 0.10 across all 6 methods | Frame-sharpness folklore; no single canonical citation in HYPOTHESES |
| `rgb_histogram_entropy_mean` (C-7) | C | Theme D | Per-frame Shannon entropy of joint RGB histogram (bins_per_channel = 8), averaged | T1 (bundled) | uncertain (probe) | Extracted | \|ρ\| ≤ 0.10 across all 6 methods | https://www.mdpi.com/1099-4300/27/2/166 (information-theoretic complexity) |

### D. Cross-modal alignment (3 features)

| Feature | Bucket | Source theme | Formula | Cost tier | Expected ρ sign | Implementation status | Falsification criterion | Supporting citation |
|---|---|---|---|---|---|---|---|---|
| `clip_text_image_sim_min` (D-1) | D | Theme E | Min over 48 TTA-visible frames of CLIP-text-image cosine (`openai/clip-vit-base-patch32`); also stored: `_mean`, `_var` | T1 (zero marginal — already extracted) | + ρ ≥ +0.15 on 4 caption-using methods; ≈ 0 NOPROMPT; gap > 0.05 | Extracted (`extract_video_features_for_tta.py:760`); needs only a correlation pass | Both absolute \|ρ\| < 0.15 AND prompt-vs-NOPROMPT gap < 0.05 across all 4 caption-using methods | Xu et al. NeurIPS 2023, ImageReward (https://arxiv.org/abs/2304.05977); Hessel et al. 2021, CLIPScore (https://arxiv.org/abs/2104.08718) |
| `cfg_gap` (D-2) | D (secondary A) | Theme E + B | `E_{t, ε, frame} ‖ε_θ(x_t, c, t) − ε_θ(x_t, ∅, t)‖_2 / max(‖ε_θ(x_t, ∅, t)‖_2, 1e-6)` | T2 (~+50% over A-1) | + ρ ≥ +0.15 on caption-using methods; ≈ 0 NOPROMPT; gap > 0.10 | Not implemented (extra unconditional ε call per noise sample) | Both absolute \|ρ\| < 0.15 AND prompt-vs-NOPROMPT gap < 0.05 | Jin et al. ICLR 2026 (https://openreview.net/forum?id=fP0s1TEow3); Imagen team 2026 (https://openreview.net/forum?id=z9YC9bvfUL); Pidstrigach 2025 (https://arxiv.org/abs/2505.19367) |
| `delta_caption_minus_uncond` (D-3) | D (lite-gap proxy) | Theme E | `mean_diffusion_loss_caption − mean_diffusion_loss_uncond` — coarse CFG-gap proxy emitted for free by the OOD scorer | T2 (free) | + on caption-using; ≈ 0 NOPROMPT (weaker than D-2) | Scaffolded (already in `diffusion_ood_scores.csv` schema) | \|ρ\| < 0.10 on all 4 caption-using methods | (same as D-2 line of work) |

### E. Reconstruction observability (2 features)

| Feature | Bucket | Source theme | Formula | Cost tier | Expected ρ sign | Implementation status | Falsification criterion | Supporting citation |
|---|---|---|---|---|---|---|---|---|
| `rec_err_l1` (E-1) | E (secondary A) | Theme B (representation-space) | `mean_{t,c,h,w} \| x − Decoder(Encoder(x)) \|` on the TTA-visible 48 frames | T1 (~25 min — one VAE encode + decode per video) | + with \|ΔPSNR\| on all 6 methods; + signed ΔPSNR on the 4 LoRA-class methods | Not implemented (`scripts/extract_vae_recerr_features.py`; reuses `load_longcat_components`) | \|ρ\| ≤ 0.10 with \|ΔPSNR\| on all 6 methods AND signed-ΔPSNR LoRA-class effect < 0.10 | Ding et al. 2025 (https://arxiv.org/abs/2504.07793); Järve et al. 2025 (https://arxiv.org/abs/2508.15737); Sun et al. 2025, TTL-LLM (https://arxiv.org/abs/2505.20633) |
| `rec_err_lpips` (E-2) | E | Theme B (representation-space) | LPIPS distance between `x` and `Decoder(Encoder(x))`, per frame, averaged | T1 (bundled with E-1) | (joint with E-1) | Not implemented (same script as E-1) | (joint with E-1) | Same as E-1 |

### Cross-bucket span flags

| Feature | Primary bucket | Secondary bucket(s) | Reason for span |
|---|---|---|---|
| `delta_caption_minus_uncond` (A-2 / D-3) | A (loss values) | D (caption alignment) | Difference of two diffusion losses; is a Bucket-A quantity but acts as a coarse caption-alignment proxy. |
| `score_norm_t*` (A-3) | A | B (geometric) | Score-field magnitude; loss-as-OOD vs landscape-geometry borderline. |
| `lid_flipd` (A-4) | A | C (complexity) | Intrinsic dimensionality is conceptually a complexity measure, but rendered through the diffusion model. |
| `loss_var_t` (B-3) | B (Theme A → spec) | A (loss values) | Loss variance across t is a property of loss VALUES, not landscape geometry — but spec maps Theme A → B. |
| `flow_max`, `flow_entropy` (C-1) | C | B (sparse-gradient via SAR) | Flow distribution shape; mechanism story is SAR's sparse-gradient argument. |
| `cut_count_pyscenedetect` (C-4) | C | B (non-stationary landscape) | Model-independent video statistic; mechanism story is non-stationary loss landscape (SAR / SoTTA). |
| `bpp_h264` (C-3) | C | A (confound) | Free-standing complexity feature; also the partial-out covariate for Bucket A. |
| `cfg_gap` (D-2) | D | A (geometric, via ε) | Caption-aware quantity computed from ε-field outputs. |
| `rec_err_l1` (E-1) | E | A (latent-space typicality) | Pixel-space rec-err is the surrogate for latent-space typicality (Ding/Järve line). |

### Cross-bucket SPAN handling for paper claims

- **A feature with a primary bucket assignment and a secondary affinity is reported in its primary bucket's row of the appendix table.**
- **In the main-text figure and table, we report each feature once.** Secondary affinities are flagged in figure captions ("CLIP-min, primary Bucket D, secondary Bucket A via Theme E + Theme B").
- **Multivariate Phase-2 analyses naturally use all features regardless of bucket** — the principle taxonomy is for narrative organisation, not for analytical exclusion.

### Compute-cost crosswalk (PLAN §3.1 Phase 0 wallclock)

| Bucket | Phase-0 cost (one H200) | Cluster jobs | Comment |
|---|---|---|---|
| **A.** Model-perceived difficulty | ~2–3 h | 1 H200 GPU job (OOD scorer) | A-1 / A-2 / A-3 / A-5 all share one forward pass; A-4 is +30% if added |
| **B.** Loss-landscape geometry | ~2 h | 1 H200 GPU job (Tier-3 probes — `compute_tier3_probes.py`) | B-1 / B-2 share the LoRA-r8 optimiser scaffold; B-3 is free post-processing of A-1 |
| **C.** Visual / temporal complexity | ~70 min total | 1 H200 GPU job (Tier-1 feature pipeline) + CPU helpers | C-1: 30 min RAFT; C-2: 10 min FFT; C-3: 5 min bpp; C-4/5/6/7: bundled in existing pipeline |
| **D.** Cross-modal alignment | ~0 min (free / scaffolded) | (D-2 deferred to Phase 1.5 conditional on D-3 signal) | D-1 already extracted; D-3 emitted by OOD scorer; D-2 needs +50% over A-1 |
| **E.** Reconstruction observability | ~25 min | 1 H200 GPU job (VAE rec_err) | E-1 / E-2 share one VAE forward + decode per video |

**Total Phase-0 wallclock:** ~6 GPU-hours of one H200, paralleled across 3 sbatch jobs (Stage 1a: Tier-1 features + bpp + FFT + VAE rec_err; Stage 1b: OOD scorer with score-norm patch; Stage 1c: Tier-3 probes). Correlation pass auto-chains on `afterok:1a:1b:1c`. Per the 2026-06-11 (later+2) runbook, the entire Phase-0 fans out in parallel inside the first 6 GPU-hours after cluster restart.

### Method-specificity matrix (which buckets can be reused across TTA recipes?)

| Bucket | Same feature for ADA? | LoRA-r8? | TinyLoRA-r2? | Notes |
|---|---|---|---|---|
| A. Model-perceived difficulty | ✓ same | ✓ same | ✓ same | The diffusion loss is a property of the model, not of the TTA recipe |
| B. Loss-landscape geometry | Adapter-specific | Adapter-specific | Adapter-specific | `grad_norm` is computed on whichever adapter the recipe uses; not portable across recipes |
| C. Visual / temporal complexity | ✓ same | ✓ same | ✓ same | Video features are recipe-agnostic |
| D. Cross-modal alignment | ✓ same | ✓ same | ✓ same | Caption-alignment features are recipe-agnostic |
| E. Reconstruction observability | ✓ same | ✓ same | ✓ same | VAE rec-error is a property of the base model |

The Bucket-B method-specificity is the only entry that requires *re-extraction* for each TTA recipe. For the 6 methods in the gain bundle, this means: one B-1 / B-2 extraction per (LoRA-r8, AdaSteer, TinyLoRA-r2) — three different feature CSVs. The Phase-0 budget for Bucket B is ~2 GPU-hours per adapter family × 3 families = ~6 GPU-hours total. (Per PLAN §2.5, only the LoRA-r8 variant is in Phase-0 scope; AdaSteer and TinyLoRA variants are follow-up if LoRA-r8 carries.)

### Phase-by-phase decision calendar (what dies when?)

| Phase | What happens | What can die | Triggers next phase if … |
|---|---|---|---|
| **0** (data collection, ≤ 1 day post-cluster-restart, 6 GPU-hours) | Every feature in this appendix gets extracted | Tier-3 probe runner could fail to converge under no-carryover guarantee (low risk per ANALYSIS_LOG 2026-06-11) | All feature CSVs exist; joined against `per_video_gains.csv` |
| **1** (univariate, ≤ 1 day analysis, CPU) | Every (feature, method, metric) cell gets a Spearman ρ + Bonferroni / BH-FDR flag | Bucket dies if every feature in it has \|ρ\| ≤ 0.10 on all 6 methods | At least one feature clears \|ρ\| ≥ 0.13 → graduate to Phase 2 |
| **2** (multivariate, ≤ 1 day analysis, CPU) | Ensemble gates with logistic regression / boosted trees / RidgeCV; LOCO 10-fold CV | Ensemble dies if held-out AUC < 0.55 (no separation) | Held-out AUC ≥ 0.60 with at least one bucket-spanning feature pair → graduate to Phase 3 |
| **3** (Pareto + RECOMMENDATION.md, ≤ 0.5 day, CPU) | Cost-aware Pareto frontier; recommend single best strategy | All strategies die if none clear (gain ≥ 0.05 PSNR \|\| 0.005 LPIPS) AND (coverage ≥ 50%) AND (feature compute ≤ 30 min / 999 videos) | At least one strategy survives → recommend → trigger Phase 4 authorisation request |
| **4** (long-horizon validation, ≤ 1 day, conditional) | Re-extract features on `panda_longctx_1000v` (76-frame); apply Phase-3 frozen gate; report held-out gain | Confirmed if held-out long-horizon gain ≥ 2× short-horizon gain; falsified otherwise | (Terminal phase; output goes into paper) |

### Master ordering (priority for what to look at first in Phase 1)

Per HYPOTHESES §3 (paper-leverage × cost), ordered for Phase-1 inspection:

| Order | Feature(s) | Bucket | Why first |
|---:|---|---|---|
| 1 | `bpp_h264`, `bpp_png_avg` (C-3) | C (covariate for A) | Required to interpret Bucket A correctly; cheapest in absolute terms (~5 min) |
| 2 | `clip_text_image_sim_min` (D-1) | D | Already extracted; correlation pass is zero new compute |
| 3 | `cut_count_pyscenedetect` (C-4) × LoRA tail Fisher | C → B | Already extracted; tests a specific asymmetric prediction in one Fisher table |
| 4 | `mean_diffusion_loss_caption` (A-1) | A | Direct Theme-B mechanism test |
| 5 | `rec_err_l1` (E-1) | E | Theme-B representation-space variant; cheap; orthogonal to flow / PSNR / caption |
| 6 | `hf_energy_ratio` (C-2) | C | Clean LoRA-vs-ADA differential prediction |
| 7 | `flow_max_over_mean` (C-1) | C | Rescues the flow story without contradicting the mean-flow null |
| 8 | `score_norm_t*` (A-3) | A → B | Distinct geometric OOD signal from A-1; differentiates magnitude from mismatch |
| 9 | `cfg_gap` (D-2) | D | Direct mechanism test for prompt-vs-NOPROMPT asymmetry |
| 10 | `loss_var_t` (B-3) | B → A | Free post-processing if A-1 stores per-t losses |
| 11 | `lid_flipd` (A-4) | A → C | Bonus geometric complexity; only worth firing if A-3 doesn't already explain LoRA tails |
| 12 | `single_step_loss_drop` (B-2) | B | Catastrophic-tail screener; cheapest possible mini-TTA |
| 13 | `grad_norm_θ0` (B-1) | B | Confirmatory if B-2 already pins down the asymmetric tail |

This ordering is from HYPOTHESES §3 verbatim with the addition of bucket annotations.

### Cross-bucket feature pair candidates for Phase 2 multivariate gates

The Phase-2 multivariate analysis (PLAN §3.3) will test many pairs, but these are the high-prior candidates worth pre-registering:

| Pair | Pair principle | Expected joint behaviour |
|---|---|---|
| `single_step_loss_drop` × `mean_diffusion_loss_caption` (B-2 × A-1) | tail risk × modal gain | The Bucket-B-screens-out-catastrophes × Bucket-A-prioritises-OOD architecture |
| `grad_norm_θ0` × `clip_text_image_sim_min` (B-1 × D-1) | tail risk × caption alignment | Method-specific tail control combined with method-agnostic alignment |
| `hf_energy_ratio` × `clip_text_image_sim_min` (C-2 × D-1) | free-deployable × already-free | Cheapest possible 2-feature ensemble |
| `mean_diffusion_loss_caption` × `bpp_h264` (A-1 × C-3) | partial-out done in feature space | Implicit confound-control instead of explicit partialling out |
| `cut_count_pyscenedetect` × `single_step_loss_drop` (C-4 × B-2) | non-stationary landscape × overfit | Two distinct catastrophic-tail predictors; ensemble if both clear Fisher OR ≥ 3 |
| `rec_err_l1` × `mean_diffusion_loss_caption` (E-1 × A-1) | observability ceiling × likelihood | The "limits-of-TTA" combined with the "OOD-correction" story |

The five pre-registered pairs are the ones we will report in the paper's multivariate table regardless of outcome. Phase 2 explores beyond these but we commit to reporting these for transparency.

### Permutation-null and sanity-check controls

Per PLAN §4, every feature ρ is also evaluated against a permutation null (1000 random shuffles of `video_id → feature`). The real ρ must exceed the 99th percentile of the shuffle distribution to be flagged `permutation_significant`. This is *orthogonal* to Bonferroni / BH correction and catches subtle data-leakage / off-by-one errors that distributional tests miss.

In addition, every gate is required to:

- **Reject `panda_0098`** for any LoRA-r8 family gate. A gate that fires ON for `panda_0098` under `LORA_R8_TTA` is automatically rejected, regardless of aggregate metrics.
- **Fire ON for `panda_{0461, 0555, 0862, 0431}`** under the methods for which those videos are top-10 winners (REVIEW §2.3). A gate that misses the universal-beneficiary cohort is suspect even if it has good aggregate metrics.

These cohort checks land as columns `g_panda_0098`, `g_panda_0461`, `g_panda_0555`, `g_panda_0862`, `g_panda_0431` in `gating_pareto_panda_std.csv` (PLAN §3.4).

### Per-feature commentary (one paragraph each)

This subsection expands each feature's row with concrete mechanism and prediction commentary that doesn't fit in the row format. Read alongside the per-bucket tables above.

**A-1 — `mean_diffusion_loss_caption`.** The headline OOD-score. Caption-conditioned flow-matching MSE averaged over t ∈ {100, 500, 900} on the encoded latent of the TTA-visible window, with 4–8 noise samples per (t, video). The OOD scorer (`compute_diffusion_ood_score.py`, commit `dc115e7`) emits one row per video to `diffusion_ood_scores.csv`. The bpp partial-out is mandatory per HYPOTHESES H-T1-2; without it, A-1 is partly measuring "input complexity" (Serrà et al. ICLR 2020). The two LoRA-class methods (LORA_R8_TTA and LORA_R8_TTA_NOPROMPT) are the cleanest test because they have the highest probability of a non-null result; AdaSteer's δ-tuning may behave differently because of internal regularisation. Expected paper figure: per-method scatter of `mean_diffusion_loss_caption` vs ΔPSNR on 4 LoRA-class panels with `panda_0098` annotated and the Bonferroni-significance flag in the panel corner.

**A-2 — `mean_diffusion_loss_uncond` + `delta_caption_minus_uncond`.** The unconditional comparator and the coarse CFG-gap proxy. Both are emitted for free by the OOD scorer in addition to the caption-conditioned loss. `delta_caption_minus_uncond` is the lite version of D-2 (`cfg_gap`) — it measures how much the caption changes the *loss* rather than the *ε*. Expected to correlate with D-2 at ρ ≥ 0.5 across videos; if it does, the expensive D-2 forward-pass can be deferred. Caveat per HYPOTHESES §6 Q1: the OOD scorer stores conditional, unconditional, and the delta at the same noise samples so the delta is paired-sample (low variance), not a difference of independent estimators.

**A-3 — `score_norm_t*`.** The SCOPED-style score-magnitude proxy (Barkley et al. 2025). At a fixed noise level t*, the squared norm of the model's denoising prediction is interpretable as the local energy of the data-density gradient. The hypothesis (HYPOTHESES H-T2-2): a large score norm at θ₀ means the diffusion model is locally "pushing hard" at the input — under a small parameter perturbation (LoRA), exactly the regime where the largest swing in p_θ(x) per unit of optimisation budget is possible. The discriminator from A-1: score-norm measures magnitude of the ε field; A-1 measures ε-vs-target mismatch. If A-3 dominates A-1, the operative axis is geometric (Bucket-B-flavoured); if A-1 dominates A-3, it's pure loss-mismatch OOD. Implementation cost: ≤ 30-line patch to the OOD scorer that records `‖ε‖²` alongside the MSE — essentially free if scheduled with A-1.

**A-4 — `lid_flipd`.** FLIPD (Kamkari et al. NeurIPS 2024) reads off local intrinsic dimension from the Fokker-Planck equation associated with a pretrained diffusion model. The estimator is `lid = (D/2) · (1 − tr(∇²_x log p_t(x)) · σ²_t)` per Kamkari et al. eq. 7, evaluated via a Hutchinson Hessian-trace at small t. HYPOTHESES §6 Q3 flags that FLIPD's quality is sensitive to the noise level — the original paper uses small t (so the noise-conditional score is close to the data score). For the 1000-step LongCat schedule, "small t" needs a pilot pass on 50 videos at t ∈ {20, 50, 100, 200} before the full run. The prediction: high-LID neighbourhoods are where the pretrained model has spread its capacity thinnest; LoRA's low-rank addition concentrates extra capacity precisely along the locally-dominant directions of variation. AdaSteer adds no parameters and is predicted to be insensitive.

**A-5 — `latent_norm_mean`, `latent_norm_std`, `latent_kurtosis`.** Latent moments — the first three statistics of `‖z‖` over the encoded latent of the TTA-visible window. These are free emissions from the OOD scorer and serve as probes for "latent-space typicality" in the spirit of Ding et al. 2025 / Järve et al. 2025 (representation-space OOD). Expected ρ sign is uncertain — these are diagnostic probes, not principled predictors. The most likely positive signal: `latent_kurtosis` high means heavy-tailed latent activations, which would correlate with high `rec_err_l1` (E-1) — that cross-link is worth reporting even if neither feature individually clears Bonferroni.

**B-1 — `grad_norm_θ0`.** The L2 norm of the LoRA-tunable-parameter gradient of the TTA loss at the initial LongCat weights — one forward + one backward per video, no optimiser step. This is SAR's exact prediction (Niu et al. ICLR 2023) translated from classifier to diffusion: large per-sample gradient norms drive TTA-driven model collapse on the classifier side; we predict the same asymmetric \|ΔPSNR\| signature on the diffusion side. The Tier-3 probe runner (`compute_tier3_probes.py`, ANALYSIS_LOG 2026-06-11 (later+2)) mirrors the production LoRA-r8 recipe (r=8 / α=16 / lr=5e-5 / weight_decay=0.01 / targets=qkv,proj on all blocks, no FFN) and resets the LoRA adapter + re-instantiates the optimiser per (video, timestep) loop — the no-carryover guarantee from the gating plan §2.4. Records `grad_norm_lora_t{T}` at timesteps 100, 500, 900 plus their mean.

**B-2 — `single_step_loss_drop`.** The fractional in-loop loss drop after a single Adam step on the LoRA-r8 adapter: `(L_diff(x; θ₀) − L_diff(x; θ₀ + Adam_step)) / max(L_diff(x; θ₀), 1e-6)`. This is the DreamBooth-style overfit detector for the catastrophic LoRA tail (Ruiz et al. CVPR 2023). The mechanistic prediction: a video whose loss the LoRA adapter can fit in one step is precisely the kind of clip for which there is no prior-preservation regularisation pulling the optimiser away from the "memorise the visible window" solution. The `panda_0098` row is predicted to lie in the top decile of `single_step_loss_drop` despite its 44.55 dB baseline PSNR. Falsification criterion: |ρ| ≤ 0.10 with `LORA_R8_TTA` ΔPSNR refutes the DreamBooth-collapse mechanism for this setting AND specifically rules out single-step loss drop as a screening signal for the catastrophic tail.

**B-3 — `loss_var_t`.** Variance of the caption-conditioned diffusion ε-loss across noise levels t ∈ {100, 500, 900}. This is the generative analogue of EATA's entropy signal (Niu et al. ICML 2022). The TTL-LLM observation (Sun et al. 2025) is the crucial caveat: in generative TTT, the high-perplexity (high-loss) samples are the ones that gain, but the *direction* of gain depends on whether the loss is *consistently* high (sample is just plain hard — no signal to learn) or *intermittently* high (sample has a specific failure mode at one stage — signal exists). Across-t variance disambiguates these two cases. Implementation cost: zero — pure post-processing of the OOD scorer's per-(t, ε) losses, as long as the OOD scorer stores per-t (not just mean-across-t) — per HYPOTHESES §3 specific ask to the parallel workstream.

**C-1 — `flow_max`, `flow_entropy`, `flow_max_over_mean`.** Distribution shape statistics of per-pixel RAFT optical flow. Mean-flow is in the ruled-out set (Slide 2 H1); these are not. The SAR argument (Niu et al. ICLR 2023): localised fast motion (high `flow_max_over_mean`) means a handful of high-loss pixels dominate the gradient, while uniform fast motion does not. Prediction: positive ρ between `flow_max_over_mean` and ΔPSNR for ADA / ADA_NOPROMPT (which has the regularisation regime where sparse-gradient selection matters); near-zero for the LoRA-class methods. RAFT is already deployed for the `dynamic_degree.json` mean-flow pipeline; computing max / entropy / ratio at extraction time is free.

**C-2 — `hf_energy_ratio` (3D FFT high-frequency ratio).** Fraction of 3D-spatiotemporal FFT energy above 0.5×Nyquist, on the luma channel of the TTA-visible window. Dieleman (2024) and Spectral Progressive Diffusion (Yu et al. 2025) establish that diffusion models implement approximate spectral autoregression: low-frequency content is generated early in the denoising trajectory, high-frequency emerges late. A pretrained model under-fits high frequencies more than low frequencies because the per-pixel ε-loss budget is dominated by low-frequency content. LoRA TTA has spare rank capacity to target the under-fit high-frequency band; AdaSteer's δ-tuning of attention adapters has no such spare capacity. Prediction: positive ρ on LoRA-class, near-zero on AdaSteer — the cleanest method-differential prediction in the menu. HYPOTHESES §6 Q2 flags DFT-vs-DCT-vs-DWT choice; current preference is 3D real DFT (matches FADE CVPR 2025) with the DCT variant as an appendix-only sensitivity check.

**C-3 — `bpp_h264`, `bpp_png_avg`.** Lossless-compression bits-per-pixel. This is dual-role: it's a free-standing complexity feature in Bucket C *and* the mandatory covariate for Bucket A's partial correlation per Serrà et al. ICLR 2020. The two variants disentangle inter-frame redundancy (h264 uses motion compensation) from intra-frame complexity (per-frame PNG). Expected: ρ ≥ 0.4 between bpp and the Bucket-A diffusion-OOD score; |ρ| ≤ 0.15 between bpp and ΔPSNR on its own. The discriminator (ρ(bpp, OOD) < 0.3) would refute the Serrà et al. confound in our VAE-latent setting and free Bucket A from needing the covariate adjustment.

**C-4 — `cut_count_pyscenedetect`, `cut_count_histogram`, `cut_density_per_frame`.** PySceneDetect ContentDetector cuts on the TTA-visible window; the histogram backup is at Bhattacharyya threshold 0.40. This is a Bucket-C / Bucket-B span: the *feature* is a model-independent video statistic (Bucket C), but the *mechanism* story is non-stationary loss landscape (Bucket B, SAR + SoTTA). The asymmetric prediction (HYPOTHESES H-T1-6): among the 21 videos with `LORA_R8_TTA` ΔPSNR < −1.0 dB, ≥ 25% are predicted to have ≥ 1 PySceneDetect cut inside the TTA-visible window (vs ~7% baseline rate). Fisher exact OR ≥ 3.0 is the headline single-number test.

**C-5 — `dino_temporal_l2_mean`.** Mean L2 of consecutive-frame DINOv2 feature embeddings (`facebook/dinov2-small`) on the TTA-visible window. Semantic-motion proxy where RAFT mean-flow is null — DINOv2 features are sensitive to scene-content changes that pixel-flow alignment misses (e.g. a camera pan over text or a slow zoom on a complex scene). Expected ρ sign uncertain (probe).

**C-6 — `laplacian_variance_mean`.** Frame-sharpness — the variance of the Laplacian filter response, averaged over the TTA-visible window. Complement to C-2 (FFT-high-freq). Expected uncertain (probe).

**C-7 — `rgb_histogram_entropy_mean`.** Per-frame Shannon entropy of the joint RGB histogram (default `bins_per_channel = 8`), averaged. Colour-entropy as a Theme-D complexity proxy. Expected uncertain (probe).

**D-1 — `clip_text_image_sim_min`.** Minimum per-frame CLIP-text-image cosine similarity over the 48 TTA-visible frames, using `openai/clip-vit-base-patch32`. The "weakest-link" caption-fidelity argument from ImageReward (Xu et al. NeurIPS 2023): CLIPScore averages have small interquartile range, so per-sample order statistics (min, not mean) carry the real signal. Under caption-conditioned TTA, the noised-target loss at the worst-fitting frame produces the largest caption-faithful gradient and dominates the held-out generation prediction. Already extracted in `scripts/extract_video_features_for_tta.py:760` — the correlation pass against `per_video_gains.csv` is zero new compute and lands in Phase 1. The prompt-vs-NOPROMPT gap > 0.05 is the falsifiable signature.

**D-2 — `cfg_gap`.** Full per-(t, ε) classifier-free-guidance gap: `E_{t, ε, frame} ‖ε_θ(x_t, c, t) − ε_θ(x_t, ∅, t)‖_2 / max(‖ε_θ(x_t, ∅, t)‖_2, 1e-6)`. The dynamic-CFG literature (Jin et al. ICLR 2026; Imagen team 2026; Pidstrigach 2025) shows that the optimal CFG scale is strongly per-sample and per-stage — i.e., per-sample CFG gap varies meaningfully across a 1000-video set. A low-CFG-gap video is one for which the caption is essentially generic relative to the score field; caption-conditioned TTA on such a video degenerates to caption-dropped TTA. Implementation needs an extra unconditional ε call per noise sample on top of A-1's forward pass — ~+50% cost. Per PLAN §2.5, deferred to a follow-up wave gated on D-3's signal.

**D-3 — `delta_caption_minus_uncond`.** `mean_diffusion_loss_caption − mean_diffusion_loss_uncond`. Coarse CFG-gap proxy: differences the loss values rather than the ε predictions. Emitted for free by the OOD scorer in the `diffusion_ood_scores.csv` schema. Expected to be a weaker version of D-2 — same sign on caption-using methods, smaller magnitude. The decision rule: if D-3 has |ρ| ≥ 0.10 on caption-using methods, we fire D-2; otherwise we cite the D-3 null as ruling out the bucket without paying the D-2 cost.

**E-1 — `rec_err_l1`.** Pixel-space L1 reconstruction error after the LongCat-VAE round-trip. Cited mechanism: the VAE encoder is a learned compressor; anything it loses is irrecoverable by downstream operations including TTA in latent space. Asymmetric prediction: positive ρ with \|ΔPSNR\| on all 6 methods (the magnitude ceiling story) AND positive ρ with signed ΔPSNR on the 4 LoRA-class methods (the TTL-LLM "high-perplexity gains more" story). Implementation: a new ~80-LOC script that reuses `load_longcat_components` from `delta_experiment/scripts/common.py`; one VAE encode + decode per video on the TTA-visible 48 frames.

**E-2 — `rec_err_lpips`.** Same extraction as E-1 but with LPIPS distance instead of L1. Perceptually-weighted variant. Cost: zero marginal — same VAE forward + decode as E-1; LPIPS is a separate small forward pass per frame on a CPU-friendly network. Reports under the same `vae_recerr_features.csv` schema.

### Talk pacing reference

Per-slide time budget (sum: 30 min):

| Slide | Topic | Budget | Cumulative |
|---|---|---:|---:|
| 0 | Title and framing | 1 min | 1 |
| 1 | Saturation puzzle | 3 min | 4 |
| 2 | 4 ruled-out hypotheses | 2 min | 6 |
| 3 | 5-bucket taxonomy table | 2 min | 8 |
| 4 | Bucket A — model-perceived difficulty | 5 min | 13 |
| 5 | Bucket B — loss-landscape geometry | 5 min | 18 |
| 6 | Bucket C — visual / temporal complexity | 5 min | 23 |
| 7 | Bucket D — cross-modal alignment | 4 min | 27 |
| 8 | Bucket E — reconstruction observability | 3 min | 30 (target hit) |
| 9 | Synthesis | (Q&A or extra) | – |
| 10 | Recommendation | (Q&A or extra) | – |
| 11 | Limitations | (Q&A or extra) | – |
| 12 | Appendix | (reference only) | – |

Slides 9–12 are technically the closing portion of the talk but the 30-min budget already lands at slide 8. In practice: drop the longest per-bucket bullet from one or two of slides 4–8 to fit the synthesis (Slide 9) into the talk. Slides 10 / 11 are 1-minute closers; Slide 12 is purely reference.

### Anticipated Q&A

Questions to prepare for from advisors / peers:

- **Q1: How is this different from just running a regression of ΔPSNR on all available features?**  
  *A: The 5-bucket taxonomy is the narrative; the regression is the experiment. They are complementary. The taxonomy tells us which paper subsection each significant feature belongs to. The regression tells us which features are significant. PLAN §3.3 explicitly runs a multivariate gradient-boosted tree on every feature; the bucket annotations on the permutation-importance bar chart make the resulting figure interpretable.*

- **Q2: Why is `panda_0098` not just a data error?**  
  *A: It's been double-checked. The 44.55 dB baseline PSNR is correct (it's a near-static text-on-white-background clip). The 22.16 dB after `LORA_R8_TTA` is reproducible. The mechanism (DreamBooth-style overfit) is mechanistically distinct from the modal-gain story and explains 30% of the aggregate negative bias of LoRA-r8 TTA. It is treated as a mechanism anchor, not as an outlier-to-be-removed.*

- **Q3: What if every Bucket-A feature is null after bpp partial-out?**  
  *A: Story C of the gating plan ("no win") — the paper claim becomes "no per-video feature provides a useful gate at this scale; gating awaits the long-horizon regime". This is fully consistent with REVIEW Story A. Phase 4 then does not auto-fire — a separate authorisation is required (PLAN §8 Decision 1).*

- **Q4: Why not a 6th bucket for trajectory-based features (DiffPath-style rate-of-change)?**  
  *A: DiffPath (Heng et al. NeurIPS 2024) measures rate-of-change of the diffusion trajectory across timesteps. Conceptually, this is a Bucket-A signal (it's a likelihood-based OOD score) computed via a *family* of forward passes rather than a single one. We absorbed it into Bucket A; if a follow-up wave finds that the trajectory-based variant dominates the single-t variants, we'd retroactively call out the sub-bucket. No separate bucket needed prospectively.*

- **Q5: What's the relationship between Buckets B and Bucket E?**  
  *A: Bucket E (autoencoder rec-error) and Bucket B (loss-landscape geometry) make different predictions about `panda_0098`. E predicts low rec_err on `panda_0098` (high-PSNR static text is easy for the VAE) — hence E predicts a *low* |ΔPSNR| ceiling, *inconsistent* with the observed |ΔPSNR| = 22.4 dB. B predicts high `single_step_loss_drop` on `panda_0098` (the LoRA adapter can memorise the text in one step). The `panda_0098` data point is therefore consistent with B and *inconsistent* with E. This is the cleanest single-video discriminator in the talk.*

- **Q6: How long until results?**  
  *A: Cluster restart is expected ~2026-06-12 morning. Phase 0 fans out in parallel inside the first ~6 GPU-hours. Phase 1 univariate analysis is ~1 day login-node CPU. Phase 2 multivariate is ~1 day. Phase 3 Pareto + RECOMMENDATION.md is ~0.5 day. Total: ~3.5 days post-cluster-restart for the Phase-0→3 deliverable. Phase 4 long-horizon validation is +1 day if it fires.*

- **Q7: Why 360 cells for the Bonferroni count?**  
  *A: ~20 features expand to ~30 scalar columns × 6 methods × 2 metrics (ΔPSNR, ΔLPIPS) ≈ 360 cells. For N=999, two-tailed α = 0.05 / 360 = 1.4 × 10⁻⁴, the critical |ρ| ≈ 0.121 ≈ 0.13. Per Decision 3 in PLAN §8 we adopt the user-stated 192-cell convention as a verbatim shorthand but recompute the critical value against the actual column count. Bonferroni is the headline; BH-FDR at q=0.1 is reported alongside for texture.*

### Cross-document audit trail

- This taxonomy is consistent with the bucket structure implicit in HYPOTHESES §4 (the six literature themes A–G), translated into a *principle-based* (rather than literature-source-based) grouping. The mapping:
  - Theme A → Bucket B (entropy / gradient-norm)
  - Theme B → Bucket A (OOD detection)
  - Theme C → Bucket B (influence / curvature)
  - Theme D → Bucket C (video complexity)
  - Theme E → Bucket D (caption alignment)
  - Theme F → no direct bucket (TTA-on-video methods provenance; cited under the synthesis as the closest precedent for our setting)
  - Theme G → typically Bucket B (asymmetric failure prediction), sometimes A (e.g. ZeroSiam's logit-norm-inflation is more A-flavoured)
- The Phase-0 scope in PLAN §3.1 matches the union of features in this taxonomy's appendix. Two features (D-2 `cfg_gap` full, and the optional latent-L2 variant of E-1) are deferred per PLAN §2.5 to a follow-up wave gated on Phase-1 results.
- The 12 hypotheses cited (H-T1-1 through H-T3-2) all have a unique primary bucket assignment. The full mapping is in the appendix table above; the report-back at the end of this document repeats it in compact form.

### Pre-registered analysis plan (one row per hypothesis × test × decision rule)

This subsection commits the paper-side analyst to specific decision rules *before* the data lands. Each row: the feature being tested, the statistical test, the result that licenses each paper claim, the bucket bookkeeping.

| Hypothesis | Feature | Test | "Wins" decision rule | "Partial" decision rule | "Loses" decision rule | Bucket label in paper |
|---|---|---|---|---|---|---|
| H-T1-1 (VAE rec-err) | E-1 `rec_err_l1` | Spearman ρ vs \|ΔPSNR\| on all 6 methods + signed ρ on 4 LoRA-class | \|ρ\| ≥ 0.15 with \|ΔPSNR\| on ≥ 4 of 6 methods AND signed ρ ≥ +0.10 on ≥ 2 LoRA-class | \|ρ\| ≥ 0.10 on at least 2 methods | \|ρ\| < 0.10 across all 6 methods | E: "limits-of-TTA from autoencoder bottleneck" |
| H-T1-2 (bpp confound) | C-3 `bpp_h264` | ρ(bpp, A-1 OOD score) vs ρ(bpp, ΔPSNR) | ρ(bpp, A-1) ≥ 0.4 AND \|ρ(bpp, ΔPSNR)\| ≤ 0.15 (confirms Serrà confound) | ρ(bpp, A-1) ∈ [0.3, 0.4] | ρ(bpp, A-1) < 0.3 (frees A from covariate adjustment) | C: "bpp covariate" |
| H-T1-3 (FFT HF) | C-2 `hf_energy_ratio` | Spearman ρ on 4 LoRA-class vs 2 ADA | + ρ ≥ 0.15 on all 4 LoRA-class AND ≈ 0 on ADA | + on some LoRA-class only | \|ρ\| ≤ 0.10 on all 4 LoRA-class | C: "diffusion-spectral autoregression" |
| H-T1-4 (flow shape) | C-1 `flow_max_over_mean` etc. | Spearman ρ on ADA vs LoRA-class | + ρ ≥ 0.15 ADA family AND ≈ 0 LoRA | + on some methods | All shape stats \|ρ\| ≤ 0.10 across all 6 methods | C: "flow concentration via SAR" |
| H-T1-5 (CLIP min) | D-1 `clip_text_image_sim_min` | Spearman ρ on caption-using vs NOPROMPT | + ρ ≥ 0.15 on 4 caption-using AND gap > 0.05 | + on some methods | \|ρ\| < 0.15 AND gap < 0.05 | D: "caption-video alignment quality" |
| H-T1-6 (scene cuts) | C-4 `cut_count_pyscenedetect` | Fisher exact OR on \|Δ\| > 1 dB × ≥ 1 cut, LoRA-r8 | Fisher OR ≥ 3.0 on both LoRA-r8 methods | OR ∈ [1.5, 3.0] | OR ≤ 1.5 | C: "non-stationary loss landscape" (bridges to B) |
| H-T2-1 (diffusion loss) | A-1 `mean_diffusion_loss_caption` | Spearman ρ vs ΔPSNR on LoRA-class, partial-out bpp | + ρ_partial ≥ 0.15 on ≥ 2 LoRA-class | + on some | \|ρ_partial\| ≤ 0.10 all 4 LoRA-class | A: "TTA is OOD-correction" |
| H-T2-2 (SCOPED score-norm) | A-3 `score_norm_t*` | Spearman ρ on LoRA-class vs ADA | + ρ ≥ 0.15 LoRA-class AND ≈ 0 ADA | + on some | \|ρ\| ≤ 0.10 on all 4 LoRA-class AND A-1 \|ρ\| > 0.15 (mismatch dominates) | A: "score-field geometric OOD" |
| H-T2-3 (CFG gap full) | D-2 `cfg_gap` | Spearman ρ on caption-using vs NOPROMPT | + ρ ≥ 0.15 on caption-using AND gap > 0.10 | + on some | \|ρ\| < 0.15 AND gap < 0.05 | D: "per-sample CFG signal" |
| H-T2-4 (FLIPD LID) | A-4 `lid_flipd` | Spearman ρ on LoRA-class vs ADA | + ρ ≥ 0.15 LoRA-class AND ≈ 0 ADA | + on some | \|ρ\| ≤ 0.10 on all 4 LoRA-class | A: "FLIPD intrinsic-dim OOD" |
| H-T2-5 (loss variance) | B-3 `loss_var_t` | Spearman ρ on LoRA-class | + ρ ≥ 0.15 on LoRA-class | + on some | \|ρ\| ≤ 0.10 on all 4 LoRA-class AND A-1 \|ρ\| > 0.15 (mean dominates) | B: "EATA reliable-sample analogue" |
| H-T3-1 (grad norm) | B-1 `grad_norm_θ0` | ρ on \|ΔPSNR\| and signed ΔPSNR on LoRA-class | + ρ \|ΔPSNR\| ≥ 0.15 LoRA-class AND − signed ρ on LORA_R8_TTA | + on \|ΔPSNR\| only | \|ρ\| ≤ 0.15 with \|ΔPSNR\| LORA_R8_TTA | B: "SAR large-gradient asymmetric tails" |
| H-T3-2 (single-step drop) | B-2 `single_step_loss_drop` | Signed ρ on `LORA_R8_TTA` ΔPSNR + Fisher OR on catastrophic tail + `panda_0098` rank | signed ρ ≤ −0.20 LoRA-r8 AND `panda_0098` in top-10% AND Fisher OR ≥ 3 | signed ρ ≤ −0.10 LoRA-r8 | \|ρ\| ≤ 0.10 LORA_R8_TTA | B: "DreamBooth-style overfit screener" |

The "wins" column is the strongest paper claim per hypothesis. "Partial" is a footnote-level claim. "Loses" is the honest null. All thresholds are pre-registered: the analyst does not get to choose the bar after seeing the data.

### Anticipated outcomes by phase (best-guess priors)

For each phase, a rough prediction of how the data will look. These are *not* commitments — they are calibration anchors.

**Phase 0 (data collection).** All Phase-0 jobs are expected to succeed; the Tier-3 probe runner is the only one with non-trivial implementation risk (no-carryover guarantee under per-(video, timestep) optimiser reset), but per ANALYSIS_LOG 2026-06-11 (later+2) the runner is implemented and mirrors the production LoRA-r8 recipe. Expected wallclock: < 1 day post-cluster-restart.

**Phase 1 (univariate).** Expected probability that *at least one* feature clears Bonferroni: ~70%. Expected probability that *no* feature clears Bonferroni: ~30%. Conditional on at least one survivor, the most likely surviving features (in decreasing prior probability):

- B-2 `single_step_loss_drop` on `LORA_R8_TTA` (catastrophic-tail prediction has strong mechanism support)
- A-1 `mean_diffusion_loss_caption` on LoRA-class methods (after bpp partial-out)
- C-2 `hf_energy_ratio` on LoRA-class (clean method-differential prediction)
- D-1 `clip_text_image_sim_min` on caption-using methods (already extracted, free correlation pass)

Lower-prior survivors:

- A-3 `score_norm_t*` (uncertain whether geometric or mismatch dominates)
- C-4 `cut_count_pyscenedetect` Fisher on catastrophic tail
- B-1 `grad_norm_θ0` on \|ΔPSNR\|
- E-1 `rec_err_l1` on \|ΔPSNR\|

**Phase 2 (multivariate).** Expected probability that the multivariate gate beats the best univariate gate by ≥ 0.05 AUC: ~50%. Concrete predictions:

- The B-2 × A-1 pair is the highest-prior candidate to dominate (tail risk × modal gain).
- The B-2 × D-1 pair is the cheapest credible alternative.
- The all-features gradient-boosted tree will probably have higher AUC than any 2-feature pair, but the cost-aware Pareto frontier (Phase 3) will likely penalise it.

**Phase 3 (Pareto + RECOMMENDATION.md).** Three outcome cases per PLAN §5:

- **Clean win** (probability ~40%): one strategy clears all three criteria — held-out gain > 0.05 PSNR or > 0.005 LPIPS AND coverage ≥ 50% AND feature compute ≤ 30 min / 999 videos. Paper claim: "*per-video gating recovers a +X PSNR / −Y LPIPS effect under method M that the population-saturated headline hides*".
- **Partial win** (probability ~40%): some strategy clears the gain + coverage criteria but fails the compute criterion (i.e., the only working gate uses Tier-2 / Tier-3 features). Paper claim is footnoted: "*a per-video gate exists for method family X at standard horizon, conditional on the Tier-2/T3 forward-pass infrastructure*".
- **No win** (probability ~20%): no strategy clears all three criteria. Paper claim is honest: "*no per-video feature provides a useful gate at this scale; gating awaits the long-horizon regime*". Fully consistent with REVIEW Story A. Phase 4 does not auto-fire — separate authorisation per PLAN §8 Decision 1.

**Phase 4 (long-horizon validation, conditional).** Confirms or refutes that the Phase-3 gate generalises across horizons. Best guess: if Phase 3 produces a clean win, the long-horizon held-out gain is *equal-or-larger* than the short-horizon gain with probability ~60% (per the 2026-06-11 user hypothesis that long-horizon has fatter tails in both directions); if Phase 3 produces a partial win, Phase 4 is more likely to refute than confirm because the partial-win is fragile.

### Bucket-by-bucket prior odds

Compact summary of "what we'd bet" for each bucket producing a Bonferroni-cleared single-feature gate at Phase 1:

| Bucket | Prior probability of ≥ 1 single-feature win at Phase 1 | Best feature in bucket | Expected magnitude if wins |
|---|---:|---|---|
| A. Model-perceived difficulty | ~50% | A-1 `mean_diffusion_loss_caption` | \|ρ\| in [0.13, 0.20] on LoRA-class |
| B. Loss-landscape geometry | ~55% | B-2 `single_step_loss_drop` | signed ρ in [−0.30, −0.15] on `LORA_R8_TTA` |
| C. Visual / temporal complexity | ~35% | C-2 `hf_energy_ratio` | \|ρ\| in [0.10, 0.18] on LoRA-class |
| D. Cross-modal alignment | ~40% | D-1 `clip_text_image_sim_min` | \|ρ\| in [0.10, 0.18] on caption-using methods + gap > 0.05 |
| E. Reconstruction observability | ~25% | E-1 `rec_err_l1` | \|ρ\| in [0.10, 0.15] with \|ΔPSNR\| |

The probabilities are well calibrated against the existing evidence (Slide 1: 19.6% sign agreement at 6.3× null lift is the population-level fingerprint of per-video structure; we should expect at least one feature in the menu to capture it). At least one of A / B / C / D should clear, with B being the most likely conditional on the LoRA-r8 catastrophic-outlier mechanism being real.

### One-paragraph elevator pitch (for hallway conversations after the talk)

> "*We tested 1000 videos through 6 TTA recipes on LongCat-Video. Population-level metrics saturated — every recipe sits within 0.1 PSNR of NOTTA. But per-video, the tails are not random: 19.6% of videos sign-agree across all 6 methods, which is 6.3× the null lift. Something predicts where TTA helps. We organised 12 candidate predictors into 5 principle-based buckets — diffusion likelihood (OOD), loss-landscape geometry, video complexity, caption alignment, autoencoder rec-error. Bucket B (loss-landscape) is the most likely to predict the catastrophic LoRA-r8 failure tail (single-step in-loop loss drop, DreamBooth-style overfit). Bucket A (likelihood / OOD) is the most likely to predict the modal gain. The Phase-0 cluster job lands next week; Phase-1 univariate analysis follows; the paper claim follows from the data.*"

The pitch is ~120 words; reads in ~45 seconds at presentation pace. Use it for unstructured follow-ups.

### Reading guide for the appendix

If a reader has 5 minutes after the talk:

1. Read the 5-bucket summary table (Slide 3) — the principle / measurement / claim per bucket.
2. Read the cross-bucket prediction matrix (Slide 9 §5) — the cell-by-cell predictions.
3. Read the pre-registered analysis plan (this section above) — the explicit decision rules.

If a reader has 15 minutes:

1. Read the bucket A and B slides (Slides 4 and 5) — the two model-conditional buckets that are most likely to land.
2. Read the per-feature commentary (this appendix) for A-1, A-3, B-1, B-2 — the four highest-prior features.
3. Read the cross-bucket synthesis (Slide 9) — modal-gain vs tail-risk, method-agnostic vs method-specific, the ensemble-gate hypothesis.

If a reader has 30 minutes (the talk length):

- Read the full document. The appendix's per-feature commentary and pre-registered analysis plan are the analyst-side material; the slide bodies are the presentation-side material. Read both for full coverage.

### Counter-arguments to anticipate

Each bucket has a corresponding "why this might not work" counter-argument worth pre-thinking:

**Counter to Bucket A (model-perceived difficulty):** *"At a 1000-step diffusion schedule with sufficient model capacity, the per-video ε-loss at t = 500 is dominated by the noise sample ε, not by anything intrinsic to x. Per-video variation in loss values is sampling-variance, not OOD signal."* Reply: this is exactly why HYPOTHESES H-T2-1 specifies averaging over 4–8 noise samples per (t, video). With 4 samples × 3 t-values = 12 noise realizations per video, the per-video loss estimate has standard error ~σ/√12 — small enough to support |ρ| ≥ 0.13 at N = 999 if the underlying signal exists.

**Counter to Bucket B (loss-landscape geometry):** *"`grad_norm_θ0` and `single_step_loss_drop` are computed on the LoRA-r8 adapter. They are by construction method-specific. A method-agnostic gate is needed for the paper claim to be portable."* Reply: agreed — see synthesis Slide 9 §2. The paper framing is two-track: a method-agnostic gate from {A, C, D, E} for the deployment story, and a method-specific Bucket-B probe for the safety story. Both are reported; neither is the sole story.

**Counter to Bucket C (visual complexity):** *"Model-independent features cannot capture the per-video sign-agreement structure (6.3× null lift on 6-method unanimous-sign). The sign agreement is by definition a model-conditional phenomenon."* Reply: maybe. But model-independent *content* features can be correlated with model-conditional outcomes — `panda_0431` (text-on-black-background) is unambiguously high-`hf_energy_ratio` AND unambiguously a multi-method LoRA-class winner. The point is empirical, not theoretical.

**Counter to Bucket D (caption alignment):** *"The NOPROMPT result (REVIEW §2.2) tightly refutes the caption-as-TTA-signal-channel claim. Bucket D is futile."* Reply: NOPROMPT refutes caption *presence*, not caption *quality*. Bucket D's discriminator (prompt-vs-NOPROMPT ρ gap on D-1 > 0.05) is *exactly* the falsification handle that distinguishes these two claims.

**Counter to Bucket E (rec-error):** *"VAE round-trip error is an upper bound on \|ΔPSNR\| (since TTA can only fix what the autoencoder represents). But an upper bound is not a predictor — it's a ceiling. ρ(rec_err, ΔPSNR) could be zero across the population even if the ceiling is correct."* Reply: agreed. Bucket E's primary prediction is the asymmetric \|ΔPSNR\| ρ ≥ 0.15, not signed ρ. The signed-ρ prediction (positive on LoRA-class) is the TTL-LLM-derived sub-hypothesis that may or may not hold.

### Threats-to-validity (cross-cutting, for the paper's discussion section)

1. **N = 999 is at the lower end for stable rank-correlations.** Bonferroni |ρ| ≥ 0.13 at N = 999 is genuine but not generous. A larger N (e.g., the full Panda-70M 2000-video extension that the cluster could in principle support) would shrink the confidence interval on every cell.
2. **Single resolution (480p), single horizon (17-frame standard).** Cross-resolution / cross-horizon generalisation is the Phase-4 conditional check; we cannot claim generalisation pre-Phase-4.
3. **Single base model (LongCat-Video).** All claims are about LongCat-Video's TTA-suitability axis; another video diffusion backbone (Wan, HunyuanVideo, CogVideoX-5B, LongVid) could produce a different bucket leaderboard.
4. **Six TTA recipes, three families.** Method-routing claims (G2 in PLAN §1.2) are between three families (ADA / LoRA-r8 / TinyLoRA), not between six independent recipes. Generalisation to *new* TTA recipes (Zo3T, TTT-Video, future) is not in scope.
5. **The Bonferroni count of 360 cells assumes independent tests.** Multiple features within a bucket are highly correlated (e.g., A-1 and A-3 share a forward pass; D-1 and D-3 share a video). The effective number of independent tests is smaller than 360; the Bonferroni bar is conservative. We use Bonferroni for the headline AND report BH-FDR q=0.1 for the texture.
6. **The bpp partial-out is correlational, not causal.** Subtracting bpp from the OOD score is the standard Serrà et al. trick but it does not establish a causal relationship between bpp and the OOD score. A reviewer could ask: *"what if bpp and OOD are both downstream of a third unmeasured factor?"* Reply: that's why we report both raw and partial ρ; the partial is for completeness, the headline is the partial-out-survived signal.

### Inversion: what does the *parent* literature say about our setting?

Useful sanity check — does the parent literature predict our results?

- **EATA (Niu et al. ICML 2022)** predicts that high-entropy (high-loss) samples cause classifier TTA to collapse. Translated to diffusion: high-loss samples should *cause more harm*. **But our generative-TTA setting reverses this** — TTL-LLM predicts high-perplexity samples gain *more* in generative TTA. **Resolution:** the EATA prediction holds for the catastrophic-tail mechanism (Bucket B); the TTL-LLM prediction holds for the modal-gain mechanism (Buckets A, E). Both literatures are correct; they describe different sub-mechanisms.
- **SAR (Niu et al. ICLR 2023)** predicts that large-gradient samples should be filtered out to avoid model collapse. Translated to diffusion: B-1 (`grad_norm_θ0`) should be *negatively* correlated with TTA success on samples that pass the filter. **But our setting expects positive ρ with \|ΔPSNR\|** — large-gradient videos have *bigger* effects, both wins and losses. **Resolution:** SAR's classifier-filter prediction does not directly translate to single-pass diffusion TTA because diffusion-TTA processes each video independently (no streaming filter). We expect SAR's mechanism to manifest *asymmetrically* — large-gradient → asymmetric tails — rather than as a uniform "filter out" rule.
- **SCOPED (Barkley et al. 2025)** predicts that score-norm at fixed t is competitive with multi-pass diffusion-OOD detectors. We extend the prediction: score-norm should also predict TTA gain (not just OOD-ness). This is the A-3 hypothesis.
- **DreamBooth (Ruiz et al. CVPR 2023)** documents that one-shot fine-tuning without a prior-preservation loss catastrophically forgets the class manifold. Translated to LoRA-r8 TTA: the per-video LoRA-r8 update *is* one-shot fine-tuning on a single visible window, without any prior-preservation regularisation. The `panda_0098` catastrophe is the canonical DreamBooth failure mode applied to a single video. The B-2 single-step loss drop is the natural early-warning signal.
- **FLIPD (Kamkari et al. NeurIPS 2024)** predicts that high-LID samples are the ones where extra model capacity is best spent. We translate to LoRA-class: high-LID videos should be the ones where LoRA's low-rank capacity contributes most.
- **dynamic CFG (Jin et al. ICLR 2026; Imagen team 2026)** predicts that the optimal CFG scale varies per sample and per stage. We translate to caption-conditioned TTA: the per-sample CFG gap is the operative quantity, not the average CFG scale.
- **ImageReward (Xu et al. NeurIPS 2023)** predicts that per-sample CLIPScore *averages* have weak discriminability. We exploit the *order statistic* alternative: `clip_text_image_sim_min` instead of mean. This is the D-1 hypothesis.

### Cluster-job calendar (for the implementation slide)

| Cluster restart day | Phase-0 job sequence | Deliverable |
|---|---|---|
| Day 0 (~2026-06-12 morning) | Submit `submit_per_video_feature_pipeline.sh` (3 sbatch jobs) + Tier-3 runner (added per ANALYSIS_LOG 2026-06-11 (later+2)) | Phase-0 feature CSVs queued |
| Day 0 (~6 GPU-hours) | Stage 1a, 1b, 1c fan out in parallel | All 6 feature CSVs land: `video_features.csv`, `diffusion_ood_scores.csv`, `bpp_features.csv`, `vae_recerr_features.csv`, `fft_features.csv`, `tier3_probe_features.csv` |
| Day 1 (login-node CPU, ~1 day) | Phase 1 univariate analysis: `analyze_gating_univariate.py` | `gating_univariate_panda_std.csv` + leaderboard |
| Day 2 (login-node CPU, ~1 day) | Phase 2 multivariate analysis: `analyze_gating_multivariate.py` | `gating_multivariate_panda_std.csv` + per-method feature-importance bar charts |
| Day 2.5 (login-node CPU, ~0.5 day) | Phase 3 Pareto + `RECOMMENDATION.md` | The final case-1 / case-2 / case-3 decision |
| Day 3+ (post-RECOMMENDATION.md user review) | Phase 4 conditional on user authorisation | Long-horizon validation |

Total: ~3.5 days post-cluster-restart to the Phase-3 RECOMMENDATION.md.

### Citations sanity check (every URL in this document is also in HYPOTHESES §5)

Every URL cited in the talk has been cross-checked against the HYPOTHESES doc's §5 references list. No invented citations. The full references list is in the next section.

### Contingency planning: "what to do if X"

For each plausible Phase-1 outcome, the paper-side action:

**Contingency 1 — every Bucket-A feature is null (\|ρ\| ≤ 0.10 after bpp partial-out) but Bucket B clears.**

- Paper claim: shift from "TTA is OOD-correction" to "TTA gain is predicted by loss-landscape geometry". The mechanism story remains principled (DreamBooth-style overfit + SAR-style large-gradient).
- Subsection structure: Bucket B becomes the headline. Bucket A is reported as a negative result with discussion ("classic OOD-detection scores fail at this scale; the operative axis is geometric, not likelihood-based"). The Serrà et al. confound is moot.
- Risk: harder to motivate to a reviewer not steeped in influence-function / DreamBooth literature.

**Contingency 2 — Bucket A clears moderately (\|ρ\| in [0.13, 0.18]) but Bucket B is null.**

- Paper claim: "TTA is OOD-correction" — the textbook story. Bucket B is reported as a null with discussion ("the SAR classifier→diffusion mapping does not transfer; the DreamBooth-collapse mechanism does not apply at this LoRA-r8 setting").
- Subsection structure: Bucket A becomes the headline. The catastrophic-tail (`panda_0098`) is reported as an unexplained outlier — a known limitation.
- Risk: leaves the catastrophic-tail unexplained, which is unsatisfying.

**Contingency 3 — both A and B clear and combine multiplicatively.**

- Paper claim: "two-feature ensemble (B × A) dominates". This is the strongest possible story.
- Subsection structure: separate subsections for A, B, then a joint subsection on the ensemble gate.
- Risk: minimal — this is the high-probability win condition.

**Contingency 4 — Bucket C wins (specifically C-2 `hf_energy_ratio`) but A and B are both null.**

- Paper claim: "*per-video TTA suitability is data-intrinsic; no model forward pass needed for the gate*". Cleanest applied-ML paper story.
- Risk: weaker mechanism story (we'd be claiming a correlation without a clean mechanistic explanation beyond "spectral autoregression"). Acceptable but less prestigious than A or B.

**Contingency 5 — only Bucket D clears.**

- Paper claim: "*caption-video alignment quality is the operative axis at this scale*". This connects to the NOPROMPT ablation and the dynamic-CFG literature.
- Risk: caption-specific, not portable to NOPROMPT methods. The deployment scenario is narrower.

**Contingency 6 — only Bucket E clears.**

- Paper claim: "*autoencoder reconstruction error upper-bounds TTA gain*". The limits-of-TTA story.
- Risk: bounded claim — does not enable a deployment-time gate, only a *predicted-ceiling* table.

**Contingency 7 — no single-feature bucket wins but the multivariate gate does.**

- Paper claim: "*per-video TTA gain is predicted by a multivariate combination of features no single feature captures*". The most subtle paper story.
- Risk: requires careful presentation of permutation-importance bar charts and held-out AUC. Reviewer must follow the multivariate logic.

**Contingency 8 — nothing clears at all (Phase-3 "no win" case).**

- Paper claim: "*at Panda 1000v / 480p / 17-frame standard horizon, no per-video feature provides a useful gate. Gating awaits the long-horizon regime.*"
- Risk: This is REVIEW Story A; consistent with existing claims. Phase 4 long-horizon validation may rescue the story; if not, the paper claim is genuinely null but honest.

### How to read the talk transcript as a checklist

Before giving the talk, run through these checks:

- [ ] Slide 0: title + subtitle correct; the three motivations are stated clearly (deployment gate, catastrophe avoidance, mechanism story).
- [ ] Slide 1: the saturation numbers are quoted verbatim (81–95% within ±0.5 dB; max |ρ| ≤ 0.09; 19.6% sign agreement; 6.3× null lift).
- [ ] Slide 2: the four ruled-out hypotheses are listed with their |ρ| evidence.
- [ ] Slide 3: the 5-bucket table is the centrepiece; every bucket has principle + measurement + features + ρ sign + paper claim.
- [ ] Slides 4–8: one bucket each, ~5 min each, with a worked example anchored on one of `panda_{0098, 0461, 0555, 0862, 0431}`.
- [ ] Slide 9: modal-gain vs tail-risk discussion; ensemble-gate hypothesis; cross-bucket prediction matrix.
- [ ] Slide 10: the four "if you only fund one feature" recommendations (B for mechanism, C for cost, A for confirmation, E for negative-result).
- [ ] Slide 11: five limitations, four open questions for advisors.
- [ ] Slide 12: appendix table + per-feature commentary + pre-registered analysis plan; reference-only.

### Glossary

For Q&A — terms used in the talk that might need defining:

- **Bonferroni:** a multiple-comparison correction that divides the per-test α by the number of tests. At 360 cells and α=0.05, the per-cell threshold is α ≈ 1.4e-4, corresponding to |ρ| ≈ 0.13 at N=999.
- **BH-FDR:** Benjamini-Hochberg false-discovery-rate correction. Less conservative than Bonferroni; controls the *expected proportion* of false positives among rejected hypotheses rather than the probability of *any* false positive. We report q=0.1.
- **Spearman ρ:** rank correlation. Robust to monotonic non-linearities and outliers (e.g., `panda_0098`).
- **Pearson r:** linear correlation. Sensitive to outliers but catches linear relationships that ρ misses.
- **CFG (classifier-free guidance):** a diffusion sampling trick that pushes the score field toward the conditional direction. The "CFG gap" is the magnitude of the conditional-vs-unconditional ε difference.
- **TTA (test-time adaptation):** running gradient descent on the model parameters at inference time using a single test input. We compare AdaSteer (δ-tuning of attention adapters), LoRA-r8 (low-rank adapter, rank 8), and TinyLoRA-r2 (rank 2).
- **OOD (out-of-distribution):** a sample that lies in a low-likelihood region of the training distribution. The HYPOTHESES §4 Theme-B claim is that diffusion-loss is an implicit OOD score.
- **Catastrophic tail:** videos where TTA actively hurts by more than ~1 dB. `panda_0098` is the canonical example (44.55 → 22.16 dB).
- **Modal gain:** the bulk of the ΔPSNR distribution. Buckets A / C / D / E predict the mode; Bucket B predicts the tail.
- **Held-out:** evaluated on a fold of data the model was not trained on. We use leave-one-chunk-out 10-fold CV (PLAN §3.3).
- **N=999:** the 999-video Panda intersection (REVIEW §2.1) that all per-video features and gain estimates are computed on.

### Final remark

This taxonomy is *the organising scaffolding*, not the result. The result is the table that lands when Phase 0 completes and Phase 1's univariate ρ leaderboard is built. The taxonomy guarantees that whatever the result is, we can describe it in 30 minutes of paper talk.

---

## References

Every URL cited in the talk, derived from HYPOTHESES doc §5 + the referenced PLAN doc. **No invented citations.**

**Theme A — per-sample TTA selection:**

- Niu et al. ICML 2022, *Efficient Test-Time Model Adaptation without Forgetting* (EATA): https://proceedings.mlr.press/v162/niu22a/niu22a.pdf
- Niu et al. ICLR 2023, *Towards Stable Test-time Adaptation in Dynamic Wild World* (SAR): https://openreview.net/pdf?id=g2YraF75Tj
- Gong et al. NeurIPS 2023, *SoTTA: Robust Test-Time Adaptation on Noisy Data Streams*: https://proceedings.neurips.cc/paper_files/paper/2023/file/2da53cd1abdae59150e35f4693834f32-Paper-Conference.pdf
- Gandelsman et al. NeurIPS 2022, *Test-Time Training with Masked Autoencoders* (TTT-MAE): https://yossigandelsman.github.io/ttt_mae/index.html
- Wang et al. 2023, *Test-Time Training on Video Streams*: https://arxiv.org/abs/2307.05014
- Sun et al. 2025, *Test-Time Learning for Large Language Models* (TTL-LLM): https://arxiv.org/abs/2505.20633

**Theme B — OOD detection for diffusion / generative models:**

- Serrà et al. ICLR 2020, *Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models*: https://ar5iv.labs.arxiv.org/html/1909.11480
- Nalisnick et al. 2019, *Detecting Out-of-Distribution Inputs to Deep Generative Models Using Typicality*: https://bayesiandeeplearning.org/2019/papers/22.pdf
- Graham et al. CVPRW 2023, *Denoising Diffusion Models for Out-of-Distribution Detection* (AnoDDPM / DDPM-OOD): https://arxiv.org/abs/2211.07740
- Pinaya et al. 2022, *Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models*: https://arxiv.org/abs/2207.13726
- Heng et al. NeurIPS 2024, *Out-of-Distribution Detection with a Single Unconditional Diffusion Model* (DiffPath): https://arxiv.org/abs/2405.11881
- Barkley et al. 2025, *SCOPED: Score–Curvature Out-of-distribution Proximity Evaluator for Diffusion*: https://arxiv.org/abs/2510.01456
- Ding et al. 2025, *Revisiting Likelihood-Based OOD Detection by Modeling Representations*: https://arxiv.org/abs/2504.07793
- Järve et al. 2025, *Probability Density from Latent Diffusion Models for OOD Detection*: https://arxiv.org/abs/2508.15737

**Theme C — influence functions and loss landscapes:**

- Mlodozeniec et al. ICLR 2025, *Influence Functions for Scalable Data Attribution in Diffusion Models*: https://arxiv.org/abs/2410.13850
- Kwon et al. ICLR 2024, *DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models*: https://arxiv.org/abs/2310.00902 (code: https://github.com/ykwon0407/DataInf)
- Garg & Roy CVPR 2023, *Samples with Low Loss Curvature Improve Data Efficiency* (SLo-Curves): https://cvpr.thecvf.com/virtual/2023/poster/20980

**Theme D — video-specific complexity beyond optical-flow:**

- Kamkari et al. NeurIPS 2024, *A Geometric View of Data Complexity: Efficient Local Intrinsic Dimension Estimation with Diffusion Models* (FLIPD): https://arxiv.org/abs/2406.03537
- Menon et al. 2024, *IVCA: Inter-Relation-Aware Video Complexity Analyzer*: https://arxiv.org/abs/2407.00280
- Dieleman 2024, *Diffusion is Spectral Autoregression* (blog): https://sander.ai/2024/09/02/spectral-autoregression.html
- Yu et al. 2025, *Spectral Progressive Diffusion for Efficient Image and Video Generation*: https://arxiv.org/abs/2605.18736
- Zhu et al. CVPR 2025, *FADE: Frequency-Aware Diffusion Model Factorization for Video Editing*: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_FADE_Frequency-Aware_Diffusion_Model_Factorization_for_Video_Editing_CVPR_2025_paper.pdf
- Normalised Shannon Entropy: https://www.mdpi.com/1099-4300/27/2/166

**Theme E — caption-video alignment quality:**

- Hessel et al. 2021, *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*: https://arxiv.org/abs/2104.08718
- Xu et al. NeurIPS 2023, *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*: https://arxiv.org/abs/2304.05977
- Jin et al. ICLR 2026, *Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models*: https://openreview.net/forum?id=fP0s1TEow3
- Imagen team 2026, *Dynamic Classifier-Free Diffusion Guidance via Online Feedback*: https://openreview.net/forum?id=z9YC9bvfUL
- Pidstrigach 2025, *Adaptive Diffusion Guidance via Stochastic Optimal Control*: https://arxiv.org/abs/2505.19367
- LongCLIP-L (zer0int): https://huggingface.co/zer0int/LongCLIP-L-Diffusers

**Theme F — TTA on video diffusion models:**

- Xu et al. CVPR 2025, *One-Minute Video Generation with Test-Time Training* (TTT-Video): https://arxiv.org/abs/2504.05298 (code: https://github.com/test-time-training/ttt-video-dit)
- Zhang et al. AAAI 2026, *Zo3T: Zero-shot 3D-Aware Trajectory-Guided I2V via Test-Time Training*: https://arxiv.org/abs/2509.06723 (code: https://github.com/Richard-Zhang-AI/Zo3T-main)
- Liu et al. 2025, *Video-T1: Test-Time Scaling for Video Generation*: https://arxiv.org/abs/2503.18942

**Theme G — pathological / asymmetric failure prediction:**

- Ruiz et al. CVPR 2023, *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*: https://arxiv.org/abs/2208.12242
- HuggingFace DreamBooth training blog: https://huggingface.co/blog/dreambooth
- Ye et al. EMNLP 2023, *Beware of Model Collapse! Fast and Stable TTA for Robust Question Answering* (Anti-CF): https://aclanthology.org/2023.emnlp-main.803.pdf
- Liu et al. 2025, *ZeroSiam: An Efficient Asymmetry for Test-Time Entropy Optimization without Collapse*: https://arxiv.org/abs/2509.23183

**Optical flow:**

- Teed & Deng ECCV 2020, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*: https://arxiv.org/abs/2003.12039

---

## Consolidated bucket → hypothesis mapping (report-back deliverable)

For audit purposes. Every hypothesis from HYPOTHESES_per_video_tta_suitability_2026-06-09.md, with primary bucket assignment and source theme letter:

| Hypothesis ID | Feature | Primary bucket | Source theme letter(s) | Secondary affinity (if any) |
|---|---|---|---|---|
| H-T1-1 | `rec_err_l1`, `rec_err_lpips` (VAE round-trip) | **E** | B (latent-space typicality) | A (model-perceived difficulty in latent space) |
| H-T1-2 | `bpp_h264`, `bpp_png_avg` | **C** | B (input-complexity confound) + D (compressibility) | A (covariate for OOD partial-out) |
| H-T1-3 | `hf_energy_ratio` (3D FFT high-freq) | **C** | D (spectral autoregression) | – |
| H-T1-4 | `flow_max`, `flow_entropy`, `flow_max_over_mean` | **C** | D (RAFT distribution shape) + A (SAR concentration) | B (sparse-gradient story) |
| H-T1-5 | `clip_text_image_sim_min` (CLIP min per frame) | **D** | E (caption-video alignment) | – |
| H-T1-6 | `cut_count_pyscenedetect`, `cut_count_histogram` | **C** | D (video complexity) + A (non-stationary loss) | B (non-stationary loss landscape) |
| H-T2-1 | `mean_diffusion_loss_caption` + `mean_diffusion_loss_uncond` + `delta_caption_minus_uncond` | **A** | B (OOD detection for diffusion) | D (delta_caption_minus_uncond is also a D-lite alignment proxy) |
| H-T2-2 | `score_norm_t*` (SCOPED score-norm) | **A** | B (score-field geometric OOD) | B (geometric flavour) |
| H-T2-3 | `cfg_gap` (full classifier-free-guidance gap) | **D** | E (caption alignment) + B (CFG dynamics from Theme E + B) | A (ε-field property) |
| H-T2-4 | `lid_flipd` (FLIPD local intrinsic dimension) | **A** | B (Fokker-Planck OOD) + D (intrinsic dimension complexity) | C (complexity measure) |
| H-T2-5 | `loss_var_t` (diffusion-loss variance across t) | **B** | A (EATA-style entropy/reliable-sample selection) | A (loss values) |
| H-T3-1 | `grad_norm_θ0` (per-video gradient norm at θ₀) | **B** | A (SAR) + C (DataInf influence) + G (asymmetric tail) | – |
| H-T3-2 | `single_step_loss_drop` (DreamBooth-style overfit detector) | **B** | G (asymmetric pathological-failure prediction) | – |

Plus features from the PLAN_gating_experiment master menu (PLAN §2) that are NOT in HYPOTHESES but get carried as probes:

| Feature | Primary bucket | Origin | Reason |
|---|---|---|---|
| `dino_temporal_l2_mean` | **C** | PLAN §2.2 row 6 | Semantic-motion proxy where RAFT mean-flow is null |
| `laplacian_variance_mean` | **C** | PLAN §2.2 row 7 | Frame sharpness complement to FFT-HF |
| `rgb_histogram_entropy_mean` | **C** | PLAN §2.2 row 8 | Colour entropy as Theme-D complexity proxy |
| `latent_norm_mean`, `latent_norm_std`, `latent_kurtosis` | **A** | PLAN §2.3 row 15 | First three moments of latent norm; free emission from OOD scorer |

### Buckets in reverse — features per bucket (compact)

**Bucket A (Model-perceived difficulty, 5 features):** `mean_diffusion_loss_caption`, `mean_diffusion_loss_uncond` + `delta_caption_minus_uncond`, `score_norm_t*`, `lid_flipd`, `latent_norm_{mean, std, kurtosis}`.

**Bucket B (Loss-landscape geometry, 3 features + 1 secondary from A):** `grad_norm_θ0`, `single_step_loss_drop`, `loss_var_t`; `score_norm_t*` (secondary, primarily A).

**Bucket C (Visual / temporal complexity, 7 features):** `flow_{max, entropy, max_over_mean}`, `hf_energy_ratio`, `bpp_{h264, png_avg}`, `cut_count_{pyscenedetect, histogram}` + `cut_density_per_frame`, `dino_temporal_l2_mean`, `laplacian_variance_mean`, `rgb_histogram_entropy_mean`.

**Bucket D (Cross-modal alignment, 3 features):** `clip_text_image_sim_min` (+ mean / var stored alongside), `cfg_gap`, `delta_caption_minus_uncond` (lite proxy; secondary, primarily A).

**Bucket E (Reconstruction observability, 2 features):** `rec_err_l1`, `rec_err_lpips`.

### Features that span buckets (explicit list)

For paper audit / reviewer transparency, every feature with non-trivial secondary affinity:

| Feature | Primary bucket | Secondary affinity | Why the span exists |
|---|---|---|---|
| `delta_caption_minus_uncond` | A | D (lite CFG-gap proxy) | Difference of two diffusion losses; A by construction, D by mechanism |
| `score_norm_t*` | A | B (geometric magnitude) | Score-field magnitude on the boundary between loss-as-OOD (A) and loss-landscape-geometry (B) |
| `lid_flipd` | A | C (complexity) | Intrinsic dimensionality is a complexity measure, but rendered through the diffusion model (model-conditional) |
| `loss_var_t` | B | A (loss values) | Variance of loss values is a property of loss VALUES, not landscape geometry per se; but spec maps Theme A → B |
| `flow_max`, `flow_entropy` | C | B (sparse-gradient via SAR) | Flow distribution shape; mechanism story is SAR's sparse-gradient argument (a B-flavoured mechanism on a C-extracted feature) |
| `cut_count_pyscenedetect` | C | B (non-stationary landscape) | Model-independent video statistic with a B-flavoured mechanism story (non-stationary loss landscape) |
| `bpp_h264` | C | A (confound) | Free-standing complexity feature AND the partial-out covariate for Bucket A |
| `cfg_gap` | D | A (geometric, via ε) | Caption-aware quantity computed from ε-field outputs (A's measurement primitive) |
| `rec_err_l1` | E | A (latent-space typicality) | Pixel-space rec-err is the surrogate for latent-space typicality (Ding/Järve line) |

**Total spans: 9 features (out of ~20).** No feature is force-bucketed against its primary mechanism; the spans are flagged transparently.

### Hypotheses that don't fit cleanly (and how we handled them)

Per the user spec: "*If any hypothesis doesn't fit cleanly, either propose a 6th bucket (only if genuinely needed) OR flag it as a 'spans multiple buckets' item in the closing section.*"

We did NOT propose a 6th bucket. Five buckets cover all 12 hypotheses cleanly. The spans listed above are noted as "primary + secondary affinity" rather than as "doesn't fit anywhere". Specifically:

- **H-T2-3 (CFG gap)** spans D + B in HYPOTHESES (cited under Theme E + Theme B). Resolution: primary D because the principle is caption alignment; secondary A because the measurement primitive is the ε field.
- **H-T2-4 (FLIPD)** spans A + C in HYPOTHESES (cited under Theme B for OOD mechanism, Theme D for intrinsic-dimension complexity). Resolution: primary A because FLIPD is computed via the diffusion model's Hessian-trace (model-conditional); secondary C because intrinsic dimensionality is conceptually a complexity measure.
- **H-T2-5 (across-t loss variance)** has a tension: its measurement (loss variance) is a Bucket-A quantity (loss VALUES across noise scales), but the user spec says Theme A → Bucket B. Resolution: respect the spec and place it in B; flag the loss-values-vs-landscape-geometry tension as a span.

**One feature that initially seemed not to fit but was resolved:** `bpp_h264` straddles "Bucket C feature in its own right" and "Bucket A confound covariate". Resolution: primary C (per the spec — model-independent video features), secondary A (per its operational role as a partial-out covariate).

**The single ambiguous bucket assignment:** H-T1-6 (`cut_count_pyscenedetect`). It is a Bucket-C feature with a Bucket-B mechanism story (non-stationary loss landscape). We put it in C because the *feature* is model-independent (C is primary by the spec) and noted the B affinity for the mechanism.

### Inconsistencies between HYPOTHESES doc and this taxonomy

Per the user spec report-back: "*Any inconsistencies you found between HYPOTHESES doc and your taxonomy (e.g. a hypothesis cited in HYPOTHESES with no clear bucket home)*"

**None found.** All 12 HYPOTHESES-doc hypotheses have a clean primary bucket. No hypothesis is left unbucketed. The Spans-section above documents the secondary affinities that arise organically from the literature themes.

One *moderate* inconsistency worth flagging: HYPOTHESES H-T2-3 (CFG gap) is cited under Theme E (caption alignment) but its mechanism narrative also draws on Theme B (CFG dynamics from the Imagen 3 dynamic-CFG / ICLR 2026 line). Both citations are accurate; the taxonomy resolves this by primary-D (alignment) + secondary-A (ε-field geometry). The HYPOTHESES doc itself does not make a primary/secondary assignment, so the taxonomy adds information rather than refining it.

---

**End of walkthrough.** Total content ≈ 30 minutes if read at presentation pace; ~3 minutes per slide on average with the per-slide budget annotations above.

The companion documents — HYPOTHESES, REVIEW, PLAN — should be read alongside for the full technical context. This talk-walkthrough is the *narrative layer*; the companion docs are the *evidence layer*.

**Next concrete deliverable:** Phase-0 cluster jobs fire when the cluster restarts (~2026-06-12 morning). Phase-1 univariate analysis lands ~1 day later. Phase-3 RECOMMENDATION.md lands ~3.5 days after cluster restart. The bucket that wins determines the paper subsection structure.

---

## Key numbers cheat sheet (for during-talk reference)

Critical numerical anchors used throughout the talk. Memorise these for fluent presentation.

### Saturation evidence (Slide 1)

- **N = 999**: the Panda 1000v / 480p / 17-frame standard horizon intersection.
- **81–95%**: percentage of clips within ±0.5 dB ΔPSNR (per method).
- **max |Spearman ρ| = 0.088**: across all 18 (method × feature) cells for the three obvious-axis predictors.
- **19.6%**: percentage of videos sign-agreeing across all 6 TTA methods.
- **6.3×**: lift over the null sign-agreement rate (~3% expected under independence).
- **Top-50-winner Jaccard = 0.08**: cross-method overlap of best-50 lists.

### Catastrophic outlier (Slide 5, Bucket B)

- **`panda_0098`**: the single catastrophic LoRA-r8 outlier.
- **44.55 → 22.16 dB**: ΔPSNR = −22.4 dB under `LORA_R8_TTA`.
- **30%**: share of aggregate `LORA_R8_TTA` negative bias attributable to this one video.

### Known-winner cohort (Slide 1)

- **`panda_0461`**: iPhone-on-desk, baseline 14.04, mean_flow 0.071. Top-10 winner under 4 methods. Universal beneficiary.
- **`panda_0555`**: cartoon girl with speech bubble, baseline 7.82, mean_flow 0.366. Top-3 winner under 4 LoRA-class methods.
- **`panda_0862`**: cartoon dragon-ball-z, baseline 10.28, mean_flow 1.258. Top-2 winner under 3 LoRA-class methods.
- **`panda_0431`**: black background with red text, baseline 31.13, mean_flow 0.593. Top-4 winner under 3 LoRA-class methods.

### Statistical thresholds (Slide 9, appendix)

- **|ρ| ≥ 0.13**: Bonferroni-significant at N=999, α=0.05/360 cells, two-tailed.
- **|ρ| ≥ 0.20**: target for B-2 catastrophic-tail screening.
- **Fisher OR ≥ 3.0**: target for cut-count × LoRA-r8 catastrophic tail.
- **AUC ≥ 0.65**: target for multivariate gate held-out performance.
- **AUC margin ≥ 0.05**: ensemble-gate-beats-univariate threshold.

### Compute budget (Slide 10, appendix)

- **~6 GPU-hours**: total Phase-0 cluster compute on one H200.
- **~3.5 days**: total wallclock from cluster restart to Phase-3 RECOMMENDATION.md.
- **~30 min**: Tier-1 feature pipeline runtime.
- **~2–3 h**: OOD scorer + Bucket-A features.
- **~2 h**: Tier-3 probes (Bucket B).
- **~25 min**: VAE rec-error (Bucket E).
- **~5 min**: bpp extraction (Bucket C).

### Decision criteria (Phase 3, PLAN §3.4)

- **≥ 0.05 PSNR or ≥ 0.005 LPIPS**: held-out gain floor.
- **≥ 50%**: coverage floor.
- **≤ 30 min / 999 videos**: feature compute ceiling.

### Cohort sizes (Slide 5, Bucket B)

- **21**: number of LoRA-r8 catastrophic-tail videos (ΔPSNR < −1 dB).
- **7**: number of LoRA-r8 modal-gain videos (ΔPSNR > +1 dB).
- **5**: number of sanity-check anchor videos (1 catastrophic + 4 known winners).

### Run-of-show timing (in minutes from talk start)

- 0:00 — Slide 0 title.
- 1:00 — Slide 1 saturation puzzle.
- 4:00 — Slide 2 ruled-out hypotheses.
- 6:00 — Slide 3 5-bucket taxonomy.
- 8:00 — Slide 4 Bucket A starts.
- 13:00 — Slide 5 Bucket B starts.
- 18:00 — Slide 6 Bucket C starts.
- 23:00 — Slide 7 Bucket D starts.
- 27:00 — Slide 8 Bucket E starts.
- 30:00 — Slide 9 synthesis starts (or run over into Q&A).

---

## What changed vs. the existing PLAN doc

This presentation is *narrative organisation* of material already in PLAN_gating_experiment_2026-06-11.md. The taxonomy adds:

1. **Principle-based grouping.** The PLAN organises by compute tier (T1/T2/T3); this taxonomy organises by theoretical principle (A/B/C/D/E). Same features, different axis.
2. **Paper subsection mapping.** Each bucket has an explicit "what paper subsection wins if this bucket wins" entry. PLAN does not have this.
3. **Talk-walkthrough format.** Per-slide budget, talking points, worked examples, closing recommendations. PLAN is a protocol document; this is a presentation document.
4. **Pre-registered analysis decision rules.** Every hypothesis has explicit "wins" / "partial" / "loses" thresholds. PLAN §3.2 has the Bonferroni / BH-FDR thresholds but does not commit to per-hypothesis decision rules.
5. **Contingency planning.** Eight contingencies (Phase-1 outcome → paper claim) for what to do in each case.

What this presentation does NOT change vs. PLAN:

- The set of features. Same 20-ish features as PLAN §2 master menu.
- The experimental protocol. Same Phase-0 → Phase-3 sequence.
- The decision criteria. Same |gain| > 0.05 PSNR, coverage ≥ 50%, feature compute ≤ 30 min.
- The compute estimates. Same ~3.5 days post-cluster-restart for the full Phase-0→3 deliverable.

**This is taxonomy + narrative, not a protocol change.** The cluster jobs do not differ between the PLAN-organisation and the taxonomy-organisation; only the paper section structure does.

---

## What each bucket does *NOT* predict (cross-bucket boundaries)

For each bucket, the boundary statements clarify what we are and aren't claiming.

### Bucket A does NOT predict

- The catastrophic-tail mechanism (`panda_0098`-class). That's Bucket B.
- Method-asymmetric effects between ADA and LoRA-r8. Bucket A is method-agnostic by construction.
- The cohort-membership of `panda_{0461, 0555, 0862, 0431}` as universal beneficiaries — Bucket A predicts the *mode* of the distribution, not the specific identity of top-10 winners.

### Bucket B does NOT predict

- Modal gain on caption-using methods. Method-specific signed-ρ predictions exist only for the LoRA-r8 family.
- AdaSteer outcomes. AdaSteer's δ-tuning has internal regularisation that mutes the SAR / DreamBooth mechanism; Bucket B is essentially silent on AdaSteer.
- Any method-agnostic deployment story. Bucket B requires re-extraction per adapter family.

### Bucket C does NOT predict

- The model-conditional 6-method sign-agreement structure directly. Bucket C is by construction model-independent; if the sign agreement is driven by model-conditional latents, Bucket C cannot capture it.
- Method-asymmetric effects beyond the LoRA-vs-AdaSteer differential implicit in `hf_energy_ratio`. The flow distribution shape (C-1) and bpp (C-3) are not predicted to differentiate method families.
- Catastrophic-failure risk for clips without scene cuts. Only C-4 (scene cuts) bridges to the tail-risk Bucket-B story.

### Bucket D does NOT predict

- Anything about NOPROMPT methods. By construction, Bucket D's signal is in the prompt-vs-NOPROMPT gap.
- Modal gain mechanism for ADA / ADA_NOPROMPT specifically — AdaSteer might be insensitive to caption-quality variation. Theme E's literature is on caption-conditioned diffusion in general; the AdaSteer-specific result is empirical.
- Catastrophic failures. Bucket D is purely a modal-gain bucket.

### Bucket E does NOT predict

- Signed ΔPSNR on AdaSteer methods. The TTL-LLM "high-perplexity gains more" mechanism applies to capacity-adding TTA (LoRA), not to attention-adapter δ-tuning (AdaSteer).
- The specific identity of the catastrophic outlier. `panda_0098` is predicted by Bucket E to have *low* `rec_err_l1` (high-PSNR static content is easy for the VAE) — *inconsistent* with the observed |ΔPSNR| = 22.4 dB. This is the cleanest single-video discriminator vs Bucket B.
- Sub-pixel autoencoder phenomena. The pixel-space L1 (E-1) is intentionally coarse; a finer latent-space variant (HYPOTHESES §6 Q5) is deferred.

### Boundary tensions between buckets

- **A vs B.** Loss values (A) vs loss-landscape geometry (B). At fixed θ₀, the loss value at a video is a property of the data; the gradient norm at the same video is a property of the loss surface. They can correlate (high loss usually means high gradient) but they don't have to (a flat-but-high loss exists).
- **A vs E.** Both are likelihood-based stories. A operates on the diffusion model; E operates on the VAE encoder-decoder. A is "is this video unfamiliar to the diffusion model?"; E is "can the VAE represent this video at all?". A video can have low A-score (unfamiliar to diffusion) AND low E-score (well-represented by VAE) — that's the modal-gain candidate.
- **B vs C-4.** Both predict the catastrophic tail. B does it via model-conditional probing; C-4 does it via model-independent scene-cut counting. The ensemble (B-2 + C-4) is a high-prior Phase-2 pair.
- **C vs D.** Both are model-independent in some sense. C is purely about the video content; D is about the caption-video pair. The C-vs-D discriminator is whether caption presence matters at all (which the NOPROMPT result already pinned to "not at population scale").

The boundaries are *not* arguments against the taxonomy. They are *features* of the taxonomy — each bucket has a clear scope of claim and a clear scope of non-claim. A reviewer asking "what does Bucket B say about ADA?" has a clean answer: "*nothing — Bucket B is method-specific to LoRA-class adapters*". This kind of clean non-claim is what the principle-based taxonomy buys over the compute-tier-based one.

---

## Closing observation

The most important single sentence in this taxonomy: **the per-video TTA outcome is the product of multiple latent factors, and the 5 buckets are five hypotheses about which latent factor is operative.** If one bucket wins, the paper's mechanism story is one of the five. If multiple buckets win, the paper's mechanism story is an explicit decomposition. If no bucket wins, the paper's mechanism story is honest negative result + Phase-4 long-horizon validation.

In all three cases, the talk has 30 minutes of coherent material. The taxonomy buys narrative coherence regardless of the experimental outcome — that's its value.

---

## Quick-reference card (one-page summary for the speaker)

| Bucket | Principle (one line) | Best feature | Best paper claim if wins |
|---|---|---|---|
| A | TTA is OOD-correction | `mean_diffusion_loss_caption` | "Diffusion-loss OOD score predicts TTA gain" |
| B | TTA gain = loss-landscape steepness | `single_step_loss_drop` | "Catastrophic LoRA-r8 failures are predicted by a single-step in-loop loss drop" |
| C | TTA suitability is data-intrinsic | `hf_energy_ratio` | "A free deployment-time gate based on raw video spectral content recovers per-video TTA gain" |
| D | Caption-video alignment quality predicts gain | `clip_text_image_sim_min` | "Alignment quality, not caption presence, predicts caption-conditioned TTA gain" |
| E | Autoencoder rec-err bounds TTA gain | `rec_err_l1` | "TTA can only fix what the autoencoder represents" |

| Cohort role | Video | Key fact |
|---|---|---|
| Catastrophic outlier | `panda_0098` | 44.55→22.16 dB under LORA_R8_TTA; 30% of aggregate negative bias |
| Universal beneficiary | `panda_0461` | Top-10 winner under 4 methods |
| Cartoon beneficiary | `panda_0555` | Top-3 winner under 4 LoRA-class |
| Cartoon beneficiary | `panda_0862` | Top-2 winner under 3 LoRA-class |
| Text-on-background beneficiary | `panda_0431` | Top-4 winner under 3 LoRA-class |

| Key threshold | Value | Source |
|---|---|---|
| Bonferroni |ρ| | 0.13 | α=0.05/360 cells, N=999 |
| Catastrophic-tail screen | ρ ≤ −0.20 | HYPOTHESES H-T3-2 |
| Cut-count Fisher OR | ≥ 3.0 | HYPOTHESES H-T1-6 |
| Phase-3 gain floor | 0.05 PSNR or 0.005 LPIPS | PLAN §3.4 |
| Phase-3 coverage floor | 50% | PLAN §3.4 |
| Phase-3 feature cost ceiling | 30 min / 999 videos | PLAN §3.4 |
| Phase-2 AUC margin | ≥ 0.05 over best univariate | Synthesis Slide 9 §4 |

| Phase | Wallclock | Deliverable |
|---|---|---|
| 0 | ≤ 1 day (≤ 6 GPU-hours) | Feature CSVs |
| 1 | ≤ 1 day login-node CPU | Univariate ρ leaderboard |
| 2 | ≤ 1 day login-node CPU | Multivariate AUC + feature importance |
| 3 | ≤ 0.5 day login-node CPU | Pareto frontier + RECOMMENDATION.md |
| 4 | ≤ 1 day (conditional) | Long-horizon validation |

End of presentation document.

