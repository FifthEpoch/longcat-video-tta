# Binary TTA/no-TTA gate + initial-loss probe (PSNR, 1000v preview)

**Date:** 2026-08-04
**Series:** `sweep_experiment/results/panda_ood_budget_1000v_preview` (12 AdaSteer configs + NOTTA, seed-matched)
**N:** 900 common videos (898 with finite PSNR gain)
**Script:** `scripts/analyze_initial_loss_prediction.py`
**Raw:** `sweep_experiment/reports/per_video_analysis/initial_loss_prediction_1000v.json`
**Regenerate:**
```
python3 scripts/analyze_initial_loss_prediction.py \
  --series-root sweep_experiment/results/panda_ood_budget_1000v_preview --notta-run NOTTA \
  --out sweep_experiment/reports/per_video_analysis/initial_loss_prediction_1000v.json
```

## Q1 — does the cheap 2-step (S2) loss predict per-video PSNR gain?

| Probe feature | Spearman [95% CI] | Deployable? |
|---|---|---|
| final_loss (= loss after 2 TTA steps) | −0.083 [−0.148, −0.019] | CI excludes 0 but <1% variance |
| base_loss (≡ final_loss here) | −0.083 [−0.148, −0.019] | redundant |
| loss_reduction / rel_reduction | n/a (base-loss ≡ total-loss ⇒ ≡0) | — |
| delta_norm | +0.042 [−0.023, +0.109] | null |
| grad_norm_mean / grad_norm_first | ~−0.04 [CI spans 0] | null |
| **OOF ridge (all probe features)** | corr(pred,actual) **+0.059 [−0.000, +0.117]** | **no** (touches 0) |

## Q2 — binary gate (route TTA vs no-TTA, then apply best fixed config)

Best fixed config = `S2_LR1e2` (mean gain **−0.003 dB**; all 12 configs ≤ 0 mean gain).

| Quantity | Value [95% CI] | Read |
|---|---|---|
| Fraction videos where fixed > no-TTA | 52.7% | ~coin flip |
| Always-fixed vs no-TTA (population effect) | **−0.0028 [−0.0252, +0.0187] dB** | **null** |
| PERFECT-gate headroom vs always-no-TTA | +0.0666 [+0.0542, +0.0807] dB | = noise floor |
| PERFECT-gate headroom vs always-fixed | +0.0694 [+0.0542, +0.0872] dB | **= noise floor** |
| **noise floor E\|g\|/2 (pure max-over-noise)** | **≈ +0.069 dB** | ceiling ≡ noise |
| [ref] 12-config oracle (more noisy draws) | +0.3547 [+0.3113, +0.4011] dB | same artifact, ×more draws |
| Probe → binary-help predictability | AUC ≈ 0.50 (all features); OOF 0.508 | **chance** |
| OOF gate vs always-no-TTA | +0.0030 [−0.0149, +0.0214] dB | **null** |
| OOF gate vs always-fixed | +0.0058 [−0.0053, +0.0190] dB | **null** |

## Why the ceiling is a noise artifact (no simulation needed)

For per-video gain \(g=\mathrm{PSNR}_\text{fixed}-\mathrm{PSNR}_\text{notta}\):
\[
\mathbb{E}[\mathrm{relu}(-g)] = \tfrac{1}{2}\big(\mathbb{E}|g| - \mathbb{E}[g]\big).
\]
The perfect-gate-vs-fixed ceiling **is** \(\mathbb{E}[\mathrm{relu}(-g)]\). Since \(\mathbb{E}[g]\approx0\)
(CI includes 0), it collapses to \(\tfrac12\mathbb{E}|g|\) = pure measurement noise. The +0.069 dB
"headroom" is manufactured by maxing over noisy per-video draws, not signal.

## Verdict

- **Q1:** no deployable regression from initial loss (only a <1%-variance final_loss link).
- **Q2:** even a *perfect* TTA/no-TTA oracle only recovers the noise floor, and the cheap probe
  predicts the gate at chance → **binary gate ruled out for PSNR on in-domain Panda.**
- **Recommendation:** deploy a single fixed config (≡ no-TTA within noise); no router.
- **Still open:** OOD/long-horizon regimes (where \(\mathbb{E}[g]\neq0\) could make a gate real)
  and seed-space best-of-k (headroom from genuinely different videos, not noise).
