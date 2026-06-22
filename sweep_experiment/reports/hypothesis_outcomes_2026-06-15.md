# TTA Gating Hypotheses — Experimental Outcomes (Research Updates 06/15)

**Eval set:** Panda `panda_1000v_standard`, **N = 999** (1 corrupt video excluded: `panda_0473`)  
**Primary metrics:** Spearman ρ(ΔPSNR, feature) per method; population mean ΔPSNR vs NOTTA  
**Verdict bar:** A feature **passes** only if **|ρ| ≥ 0.2 on ≥ 2 methods** (pre-registered in `correlate_tta_gain_with_features.py`)

**Data note:** H1–H4 numbers verified locally from `per_video_gains.csv`. **H1 motion extended** and **H5–H8** bootstrap CIs are from cluster job **11135260** (`criteria_correlation_full`: all feature CSVs joined, OOD + bpp + fft + vae + tier3 + motion). Prior point estimates without bootstrap (jobs **10810217** / **10847206**) superseded for H5–H8 headline features.

---

## H1 — Video dynamicity (RAFT mean-flow) predicts TTA gain

**Predicted:** Higher motion / dynamicity systematically predicts TTA benefit (or harm).  
**Verdict:** **Fail**

| Metric | AdaSteer (ADA) | LoRA R8 TTA | Threshold | Pass? |
|---|---:|---:|---|---|
| Spearman ρ(ΔPSNR, `mean_flow`) | **−0.069** | **+0.073** | \|ρ\| ≥ 0.2 | No |
| N | 999 | 999 | | |
| Mean ΔPSNR vs NOTTA | +0.008 dB | −0.076 dB | | |
| \|Δ\| > 0.5 dB (gain / loss share) | 9.0% / 10.1% | 1.7% / 3.9% | ~10–15% (slide) | Descriptive only |

### H1 extended — motion battery with bootstrap CIs (job 11135260; motion subset also in 11132325)

| Feature | ADA ρ [95% CI] | LoRA ρ [95% CI] | Pass? |
|---|---|---|---|
| `mean_flow` (RAFT) | −0.061 [−0.124, +0.001] | +0.086 [+0.024, +0.148] | No — sign flip, \|ρ\| < 0.2 |
| `dino_temporal_l2_mean` | −0.033 [−0.096, +0.030] | +0.109 [+0.044, +0.173] | No — LoRA CI excludes 0 but \|ρ\| < 0.2 |
| `latent_temporal_l2_mean` | −0.042 [−0.107, +0.020] | +0.114 [+0.053, +0.175] | No — strongest latent motion signal, still < 0.2 |
| `pixel_mse_temporal_mean` | ~0 [−0.067, +0.064] | **+0.163 [+0.100, +0.226]** | No for 2-method bar; LoRA-only signal |

- Motion is not a usable gate: correlations are near zero and **sign-flip across methods** (ADA slightly negative, LoRA slightly positive).
- Four motion representations (RAFT, DINO, VAE latent, pixel MSE) **do not clear |ρ| ≥ 0.2 on both methods**; bootstrap CIs confirm null for ADA and only weak LoRA-only hints for pixel MSE / latent motion.

---

## H2 — Lower baseline (NOTTA) PSNR → more TTA benefit

**Predicted:** Worse initial reconstruction → more headroom for TTA to help.  
**Verdict:** **Fail**

| Metric | AdaSteer (ADA) | LoRA R8 TTA | Threshold | Pass? |
|---|---:|---:|---|---|
| Spearman ρ(ΔPSNR, NOTTA PSNR) | **+0.013** | **−0.088** | \|ρ\| ≥ 0.2 | No |
| N | 999 | 999 | | |
| Mean NOTTA PSNR | 17.930 dB | 17.930 dB | | |
| Oracle picks NOTTA (descriptive) | 345 / 999 (34.5%) | — | | |

- No consistent “harder video → more gain” signal; ADA ρ ≈ 0, LoRA ρ weakly negative (opposite direction, still below threshold).
- Oracle routing shows **large per-video spread** (win margins ~0.37–0.39 dB) but baseline PSNR does not explain it.

---

## H3 — Caption length affects TTA performance

**Predicted:** Longer captions systematically help or hurt TTA.  
**Verdict:** **Fail**

| Metric | AdaSteer (ADA) | LoRA R8 TTA | Threshold | Pass? |
|---|---:|---:|---|---|
| ρ(ΔPSNR, caption words) | **+0.013** | **−0.023** | \|ρ\| ≥ 0.2 | No |
| ρ(ΔPSNR, caption chars) | **+0.020** | **−0.046** | \|ρ\| ≥ 0.2 | No |
| N | 999 | 999 | | |

- Caption length is **uncorrelated** with ΔPSNR at N=999; matches slide claim of negligible correlation.
- Strongest Tier-1 signal in the partial battery is `rgb_histogram_entropy_mean` (ADA ρ = +0.160), still **below the |ρ| ≥ 0.2 bar**.

---

## H4 — Excluding caption during TTA (video-only fit) is beneficial

**Predicted:** Fitting without text supervision improves TTA.  
**Verdict:** **Inconclusive**

| Method | Mean ΔPSNR (w/ caption) | Mean ΔPSNR (no prompt) | Mean PSNR (w/ vs no) | No-prompt wins (per-video) |
|---|---:|---:|---:|---:|
| AdaSteer | **+0.008** | +0.002 | 17.938 vs 17.932 | 512 / 999 |
| LoRA R8 | −0.076 | **−0.065** | 17.855 vs **17.865** | 553 / 999 |

- Population means are **mixed**: caption helps AdaSteer slightly (+0.006 dB), no-prompt helps LoRA slightly (+0.011 dB less harm).
- Per-video no-prompt wins a **bare majority**, but effect sizes are tiny — does not support a strong “drop caption” policy.

---

## H5 — Model-perceived difficulty (diffusion OOD loss) → more TTA benefit

**Predicted:** Higher denoising loss (OOD) → low likelihood → more room to adapt → higher ΔPSNR.  
**Verdict:** **Fail / Falsified** (wrong sign on headline OOD feature)

> **Source:** Full-battery bootstrap correlation (job **11135260**, N=999).

| Feature | ADA ρ [95% CI] | LoRA ρ [95% CI] | Threshold | Pass? |
|---|---|---|---|---|
| `mean_diffusion_loss_uncond` | **−0.162 [−0.223, −0.101]** | **−0.130 [−0.191, −0.067]** | \|ρ\| ≥ 0.2, predicted **+** | **Falsified** |
| `mean_diffusion_loss_caption` | **−0.162** | **−0.130** | \|ρ\| ≥ 0.2, predicted **+** | **Falsified** |
| `latent_norm_mean` (cheap OOD proxy) | **−0.180 [−0.241, −0.118]** | **−0.122 [−0.184, −0.056]** | \|ρ\| ≥ 0.2 | Fail — strongest proxy, mean \|ρ\| ≈ **0.151** |
| N | 999 | 999 | | |

**6-method OOD headline (`mean_diffusion_loss_uncond`, point ρ from job 10847206):**

| Method | ρ | N | Verdict |
|---|---:|---:|---|
| ADA | **−0.162** | 999 | Wrong sign |
| ADA_NOPROMPT | −0.140 | 999 | Wrong sign |
| LORA_R8_TTA | **−0.130** | 999 | Wrong sign |
| LORA_R8_TTA_NOPROMPT | −0.137 | 999 | Wrong sign |
| TL_BARE_R2 | −0.048 | 999 | Null |
| TL_TIED_R2 | +0.004 | 999 | Null |

- Diffusion OOD was the clearest mechanistic signal, but **inverted**: higher loss → **less** ΔPSNR (ADA ρ=−0.162, LoRA ρ=−0.130), not more — directly falsifies the slide hypothesis.
- Best cheap proxy `latent_norm_mean` (mean |ρ| ≈ 0.11–0.15) still **0/6 methods ≥ 0.2** — ranks videos weakly, not deployably. Quintile check: ADA mean ΔPSNR drops from **+0.11 dB** (lowest OOD) to **−0.12 dB** (highest OOD).

---

## H6 — Loss norm / steep loss surface → larger TTA step → larger ΔPSNR

**Predicted:** Large gradient / score-norm at θ₀ → steeper surface → bigger stable TTA update → higher ΔPSNR.  
**Verdict:** **Fail**

> **Source:** Full-battery bootstrap correlation (job **11135260**, N=999).

| Feature | ADA ρ [95% CI] | LoRA ρ [95% CI] | Threshold | Pass? |
|---|---|---|---|---|
| `mean_grad_norm_lora` | **−0.136 [−0.197, −0.073]** | **−0.141 [−0.202, −0.076]** | \|ρ\| ≥ 0.2 | No — CIs exclude 0 but wrong sign vs prediction |

| Feature | Tier | Mean \|ρ\| (6 meth.) | # meth. ≥ 0.2 | Threshold | Pass? |
|---|---|---:|---:|---|---|
| `mean_grad_norm_lora` | T3P | **0.093** | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `grad_norm_lora_t100` | T3P | 0.093 | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `loss_var_caption` / `loss_var_uncond` | OOD | **< 0.045** | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `mean_loss_drop_pct` (1-step probe) | T3P | 0.045 | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `delta_caption_minus_uncond` (CFG-gap proxy) | OOD | 0.087 | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| N | | 999 | | | |

- Pre-step LoRA grad-norm probe (SAR-style) mean |ρ| = **0.093**; loss-variance and one-step loss drop **< 0.045** — steep-surface story does not predict gain.
- One forward+backward pass is cheap, but **doesn't buy a gate** at this resolution.

---

## H7 — Visual / temporal complexity (bpp / FFT) improves TTA prediction

**Predicted:** Raw complexity (compression bits, high-frequency energy) explains who benefits from TTA.  
**Verdict:** **Fail**

> **Source:** Full-battery bootstrap correlation (job **11135260**, N=999).

| Feature | ADA ρ [95% CI] | LoRA ρ [95% CI] | Threshold | Pass? |
|---|---|---|---|---|
| `bpp_png_avg` | **+0.178 [+0.115, +0.240]** | **+0.094 [+0.031, +0.158]** | \|ρ\| ≥ 0.2 | No — ADA clears bar alone, not both methods |

| Feature | Tier | Mean \|ρ\| (6 meth.) | # meth. ≥ 0.2 | Threshold | Pass? |
|---|---|---:|---:|---|---|
| `bpp_png_avg` | T1 | **0.099** | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `bpp_h264` | T1 | **0.024** | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `hf_energy_ratio_spatial_only` | T1 | **0.039** | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `hf_energy_ratio_3d` | T1 | (cluster) | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `mean_flow` (baseline motion) | T1 | 0.051 | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `flow_max` (flow shape) | T1 | 0.034 | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| N | | 999 | | | |

- Strongest complexity proxy PNG bpp (|ρ| ≈ **0.10**) and weakest H.264 bpp (|ρ| ≈ **0.02**); FFT HF ratio |ρ| ≈ **0.04**.
- Motion, cuts, bpp, and FFT all **null** — complexity does not explain who wins TTA.

---

## H8 — Reconstruction observability (VAE encode–decode error caps TTA gain)

**Predicted:** High VAE round-trip error → lossy latents → TTA gain capped by autoencoder floor.  
**Verdict:** **Fail**

> **Source:** Full-battery bootstrap correlation (job **11135260**, N=999).

| Feature | ADA ρ | LoRA ρ | Threshold | Pass? |
|---|---:|---:|---|---|
| `rec_err_l1` | **+0.142** | (below bar) | \|ρ\| ≥ 0.2 on ≥ 2 methods | No |
| `rec_err_lpips` | (below bar) | **+0.143** | \|ρ\| ≥ 0.2 on ≥ 2 methods | No |

| Feature | Tier | Mean \|ρ\| (6 meth.) | # meth. ≥ 0.2 | Threshold | Pass? |
|---|---|---:|---:|---|---|
| `rec_err_l1` | T1 | **0.087** | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| `rec_err_lpips` | T1 | **0.090** | 0 / 6 | \|ρ\| ≥ 0.2 | No |
| N | | 999 | | | |

- VAE round-trip L1 (|ρ| ≈ **0.09**) and LPIPS (|ρ| ≈ **0.09**) — poor recon does not predict larger gain.
- Latent-norm OOD (H5) beats pixel rec-error as a weak correlate, but neither clears the bar.

---

## H9 — OOD-adaptive TTA budget (steps + learning rate)

**Predicted:** High OOD → more TTA steps + lower LR; low OOD → fewer steps + higher LR.  
**Verdict:** **Pilot complete — population pattern does not support H9; quintile oracle pending**

| Experiment | Status | Result |
|---|---|---|
| OOD-stratified step/LR pilot (12 configs × 200v) | **Done** | Merge complete (12/12 configs, 200 videos each) |
| Population best PSNR | Done | **`S2_LR1e2` 18.126 dB** (2 steps, LR 1e-2) — fewer steps + higher LR than fixed S10/LR5e-3 |
| Population best FVD | Done | **`S10_LR1e3` 316.5**; worst **`S2_LR1e2` 335.6** (PSNR winner) |
| OOD-quintile oracle / adaptive policy | **Pending** | Re-run `analyze_adasteer_budget_oracle.py` after `--baseline-series-root` fix |
| Population baseline (fixed budget, 10 steps) | Done | ADA +0.008 dB, LoRA −0.076 dB (1000v standard) |

**Pilot population highlights (merged summaries, N=200):**

| Config | Mean PSNR | Mean FVD |
|---|---:|---:|
| S2_LR1e2 (best PSNR) | **18.126** | 335.6 (worst) |
| S2_LR5e3 | 18.113 | — |
| S5_LR1e3 | 18.105 | — |
| S10_LR1e3 (best FVD) | — | **316.5** |
| S5_LR5e3 | — | 316.7 |
| S20_LR1e3 | — | 318.6 |

- At **population** level the PSNR winner is **2 steps / LR 1e-2** — the opposite of the H9 directional prediction (more steps + lower LR for high OOD). Best FVD favours **10 steps / LR 1e-3** (closer to “more steps, lower LR” but still not quintile-specific).
- **Q5 rescue:** cannot confirm from population aggregates alone; need quintile-stratified oracle table (fixed S10/LR5e-3 vs oracle-best per quintile). Cluster command after `git pull`:

```bash
python scripts/analyze_adasteer_budget_oracle.py --bootstrap \
  --series-root sweep_experiment/results/panda_ood_budget_pilot \
  --baseline-series-root sweep_experiment/results/panda_1000v_standard \
  --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \
  --output sweep_experiment/reports/per_video_analysis/2026-06-20/adasteer_budget_oracle_pilot.md
```

- Slide 10’s OOD→more-benefit gating rule was tested as H5 and **falsified**; H9 is specifically the **adaptive budget** experiment.

---

## Supporting context — Oracle routing (Slide 4)

Per-video routing is worth pursuing even though population TTA ≈ 0.

| Policy | Mean PSNR | Δ vs always-NOTTA |
|---|---:|---:|
| Always NOTTA | 17.930 dB | 0.000 dB |
| Always AdaSteer | 17.938 dB | +0.008 dB |
| Always LoRA | 17.855 dB | −0.076 dB |
| 2-way oracle (NOTTA vs ADA) | 18.124 dB | **+0.193 dB** |
| 3-way oracle (NOTTA / ADA / LoRA) | 18.156 dB | **+0.226 dB** |

**Bootstrap oracle uplift** (per-video, B=5000, seed=42): mean Δ = **+0.226 dB**, 95% CI **[+0.186, +0.271] dB**, CI excludes 0: **yes**.

| Oracle winner | N | Mean win margin | Median margin |
|---|---:|---:|---:|
| AdaSteer | 446 | 0.389 dB | 0.111 dB |
| NOTTA | 345 | 0.369 dB | 0.123 dB |
| LoRA | 208 | — | — |

| Oracle FVD (job 11061632, N≈998–999, 14 cond + 14 gen frames) | NOTTA | ADA | LoRA R8 | Oracle best PSNR |
|---|---:|---:|---:|---:|
| FVD | **155.94** | **156.22** | **158.85** | **149.57** |
| Δ vs NOTTA | — | +0.28 | +2.91 | **−6.37** |

- ~**50/50** AdaSteer win/loss on ΔPSNR (497 gain / 502 loss); oracle PSNR uplift **+0.19–0.23 dB** is **statistically real** (bootstrap CI excludes 0), not noise from N=999.
- Oracle FVD **confirmed** (job **11061632**): oracle_best_psnr **149.57** vs always-NOTTA **155.94** (−6.37). Headline online FVD (154.7 / 153.4 / 157.9) is ~1–2 points lower — normal for disk mp4 vs in-memory eval; use **one protocol per table**.

---

## Summary table (for deck closing slide)

| # | Hypothesis (slide) | Verdict |
|---|---|---|
| H1 | RAFT mean-flow → TTA gain | **Fail** (\|ρ\| < 0.09; extended motion battery also fails) |
| H2 | Low baseline PSNR → more gain | **Fail** |
| H3 | Caption length → TTA effect | **Fail** |
| H4 | No-caption TTA better | **Inconclusive** |
| H5 | OOD / diffusion loss → more benefit | **Fail / Falsified** |
| H6 | Loss norm → larger ΔPSNR | **Fail** |
| H7 | Visual/temporal complexity | **Fail** |
| H8 | VAE rec error caps gain | **Fail** |
| H9 | OOD-adaptive steps/LR | **Pilot done — population fails H9 direction; quintile oracle pending** |

**Bottom line:** Phase 0 gating is **complete for H1–H8** (full-battery bootstrap job **11135260**). **No feature clears |ρ| ≥ 0.2 on both ADA and LoRA**; strongest signal is `latent_norm_mean` (mean |ρ| ≈ **0.151**). OOD is the one “interesting” result and it **refutes** the original prediction (bootstrap CIs exclude 0, wrong sign). Per-video oracle PSNR routing shows **+0.226 dB** headroom with bootstrap CI **[+0.186, +0.271]**, and oracle FVD drops to **149.57** vs **155.94** always-NOTTA — but we still lack a deployable offline gate. **H9 pilot** (200v × 12-config grid) is merged; population PSNR peaks at **S2_LR1e2** (opposite of H9 direction); quintile oracle report still needed to adjudicate Q5 rescue.
