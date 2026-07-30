# Router significance / randomness analysis — 1000v preview (N=898)

**Date:** 2026-07-31
**Series:** `sweep_experiment/results/panda_ood_budget_1000v_preview`
**Features:** `2026-07-12` · block **A+B+C** (video/caption + diffusion-OOD + VAE-profile)
**Method:** 5-fold leakage-free OOF ridge (seed 42), identical picks to the router
matrix. Bootstrap B=10000, sign-flip/null draws=10000, noise-floor sims=5000.
**Regenerate:** `python3 scripts/router_significance_analysis.py --series-root
sweep_experiment/results/panda_ood_budget_1000v_preview --feature-date
sweep_experiment/reports/per_video_analysis/2026-07-12 --output-dir <out>
--metrics psnr vbench --blocks A+B+C --actions 12 13`

## Verdict (all four variants)

**The per-video config router is not winning from randomness — it is not winning
at all.** On every test the router is at or below the noise floor, and the
config-oracle "headroom" is fully explained by measurement noise (max-over-noise).

## (1) Realized gain — bootstrap 95% CI + sign-flip test

| Metric | Actions | N | Δ vs fixed [95% CI] | sign-flip p | Δ vs NO-TTA [95% CI] | sign-flip p |
|---|---:|---:|---|---:|---|---:|
| PSNR | 12 | 898 | −0.0148 [−0.0324, +0.0022] dB | 0.094 | −0.0176 [−0.0358, −0.0003] dB | 0.048 |
| PSNR | 13 | 898 | −0.0110 [−0.0288, +0.0064] dB | 0.224 | −0.0138 [−0.0293, +0.0008] dB | 0.072 |
| VBench | 12 | 898 | −0.0069 [−0.0151, +0.0008] raw | 0.085 | −0.0066 [−0.1758, +0.1639] raw | 0.937 |
| VBench | 13 | 898 | −0.1276 [−0.2551, −0.0004] raw | 0.048 | −0.1273 [−0.2395, −0.0182] raw | 0.024 |

No Δ-vs-fixed CI is above 0. The one nominally significant cell (VBench 13-action,
p=0.048) is significant in the **wrong direction** — the router is reliably *worse*
than the best fixed config because the added NO-TTA action degrades it.

## (2) Randomness nulls — beats random targeting?

`random_pick` = uniform random valid config/video. `shuffle_picks` = the router's
OWN picks permuted across videos (identical config-usage marginal, random
targeting); gain above this = genuine per-video adaptivity. `match%` vs 1/12≈0.083.

| Metric | Actions | Δ vs fixed | random null (mean / hi95) | p_rand | shuffle null (mean / hi95) | p_shuf | match% / chance |
|---|---:|---:|---:|---:|---:|---:|---:|
| PSNR | 12 | −0.0148 | −0.0272 / +0.0021 | 0.217 | −0.0029 / +0.0101 | **0.955** | 0.050 / 0.083 |
| PSNR | 13 | −0.0110 | −0.0246 / +0.0037 | 0.185 | −0.0001 / +0.0133 | **0.942** | 0.019 / 0.083 |
| VBench | 12 | −0.0069 | −0.0034 / +0.0024 | 0.886 | −0.0009 / +0.0050 | **0.976** | 0.085 / 0.083 |
| VBench | 13 | −0.1276 | −0.0034 / +0.0416 | 1.000 | +0.0001 / +0.0829 | **0.999** | 0.020 / 0.083 |

`p_shuf` = 0.94–0.999: random reassignment of the router's own picks does as well
or better ~94–99.9% of the time → the per-video targeting carries no signal.
match% is at or below the 1/12 chance floor everywhere.

## (3) Config-oracle headroom vs the max-over-noise floor

`noise floor` = simulated headroom if all 12 configs were equal per video and
differed only by Gaussian noise (σ from the observed across-config spread).
Observed ≤ floor ⇒ oracle is max-over-noise (unroutable). σ estimated from the
same values → floor is conservative (over-states real signal).

| Metric | oracle headroom [95% CI] | noise floor [95% CI] | σ | captured by router |
|---|---|---|---:|---:|
| PSNR | +0.3575 [+0.3160, +0.4014] dB | +0.4281 [+0.3935, +0.4627] dB | 0.2627 | −0.04 |
| VBench | +0.0978 [+0.0883, +0.1079] raw | +0.0985 [+0.0921, +0.1047] raw | 0.0604 | −0.07 |

Observed headroom sits at or below the noise floor for both metrics; the
deployable router captures a *negative* fraction of it.

## Implications

1. Do **not** present a per-video AdaSteer config router as a positive result on
   this grid. If used at all, it is a clean negative result: per-video config
   selection is not learnable here, and the oracle gain is a noise artifact.
2. The upcoming TTA-method comparison (DNO CVPR'24, Direct Noise Opt ICML'25 vs
   AdaSteer) must be framed at the **population level** — does any method move
   PSNR/FVD/VBench vs NO-TTA more than AdaSteer's fixed config (~+0.02 dB, ≈0)?
   Not "our router beats X."
3. Consistent with the earlier routability diagnostic (OOF R²(gain) ≤ 0,
   corr(NO-TTA, config) ≈ 0).
