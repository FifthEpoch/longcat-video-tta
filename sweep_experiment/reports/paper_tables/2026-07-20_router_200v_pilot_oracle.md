# AdaSteer budget-grid oracle analysis (H9)

**Series:** `sweep_experiment/results/panda_ood_budget_pilot`
**Fixed headline AdaSteer:** `S10_LR5e3` (S10/LR=5e-3).
**Pilot grid oracle N = 200** videos with per-video PSNR across the budget grid (denominator for pick frequencies).
**NOTTA baseline:** `NOTTA` from `/scratch/wc3013/longcat-video-tta/sweep_experiment/results/panda_1000v_standard` (union N = 999 when joined).




## Full grid population metrics (merged summaries)

| run_id | steps | lr | N | PSNR (dB) | SSIM | LPIPS | FVD | FID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2_LR1e3 | 2 | 1e-03 | 200 | 18.052 | 0.6369 | 0.3461 | 325.9 | 62.7 |
| S2_LR5e3 | 2 | 5e-03 | 200 | 18.113 | 0.6372 | 0.3452 | 320.9 | 61.8 |
| S2_LR1e2 | 2 | 1e-02 | 200 | 18.126 | 0.6390 | 0.3448 | 335.6 | 61.6 |
| S5_LR1e3 | 5 | 1e-03 | 200 | 18.105 | 0.6370 | 0.3454 | 317.5 | 62.3 |
| S5_LR5e3 | 5 | 5e-03 | 200 | 18.053 | 0.6370 | 0.3466 | 316.7 | 63.1 |
| S5_LR1e2 | 5 | 1e-02 | 200 | 17.900 | 0.6330 | 0.3513 | 319.7 | 63.6 |
| S10_LR1e3 | 10 | 1e-03 | 200 | 18.086 | 0.6370 | 0.3454 | 316.5 | 61.5 |
| S10_LR5e3 | 10 | 5e-03 | 200 | 17.929 | 0.6328 | 0.3506 | 331.2 | 63.4 |
| S10_LR1e2 | 10 | 1e-02 | 200 | 17.991 | 0.6332 | 0.3477 | 331.4 | 63.8 |
| S20_LR1e3 | 20 | 1e-03 | 200 | 18.022 | 0.6366 | 0.3460 | 318.6 | 62.0 |
| S20_LR5e3 | 20 | 5e-03 | 200 | 17.908 | 0.6330 | 0.3506 | 318.9 | 64.0 |
| S20_LR1e2 | 20 | 1e-02 | 200 | 17.877 | 0.6262 | 0.3538 | 334.3 | 65.3 |
| ORACLE (best PSNR/video) | — | — | — | 18.744 | 0.6503 | 0.3297 | 383.9 | — |

> Oracle FVD/FID from ``/scratch/wc3013/longcat-video-tta/sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json``: FVD=383.9, FID=— (N=200 videos).

## Population routing uplift

| Policy | Mean PSNR | Δ vs always-NOTTA | Δ vs fixed AdaSteer |
|---|---:|---:|---:|
| Always NOTTA | 17.798 dB | 0.000 dB | — |
| Fixed AdaSteer (`S10_LR5e3`) | 17.996 dB | +0.198 dB | 0.000 dB |
| **Oracle (best grid PSNR)** | **18.744 dB** | **+0.947 dB** | **+0.748 dB** |

**Bootstrap oracle uplift vs fixed AdaSteer** (per-video, B=5000, seed=42): mean Δ=+0.748 dB, 95% CI [+0.524, +1.025] dB, CI excludes 0: yes.

**Oracle config picks (top):** `S20_LR1e2` 51 (25.5%) · `S10_LR1e2` 40 (20.0%) · `S20_LR5e3` 17 (8.5%) · `S5_LR1e2` 15 (7.5%) · `S2_LR5e3` 15 (7.5%) · `S5_LR5e3` 14 (7.0%) · `S2_LR1e2` 12 (6.0%) · `S2_LR1e3` 10 (5.0%)

## Table 2 — Oracle pick frequency + routing uplift

Denominator: **N = 200** pilot videos with a grid oracle winner (not the NOTTA union size when `--baseline-series-root` is used).

*Mean ΔPSNR when picked = winner PSNR − fixed `S10_LR5e3` PSNR on videos where that config wins.*

| Config | Oracle picks | Freq | Mean ΔPSNR vs `S10_LR5e3` |
| --- | --- | --- | --- |
| `S2_LR1e3` | 10 | 5.0% | +0.627 dB |
| `S2_LR5e3` | 15 | 7.5% | +0.267 dB |
| `S2_LR1e2` | 12 | 6.0% | +0.687 dB |
| `S5_LR1e3` | 7 | 3.5% | +2.189 dB |
| `S5_LR5e3` | 14 | 7.0% | +0.249 dB |
| `S5_LR1e2` | 15 | 7.5% | +0.910 dB |
| `S10_LR1e3` | 7 | 3.5% | +0.524 dB |
| `S10_LR5e3` | 7 | 3.5% | +0.000 dB |
| `S10_LR1e2` | 40 | 20.0% | +0.644 dB |
| `S20_LR1e3` | 5 | 2.5% | +0.280 dB |
| `S20_LR5e3` | 17 | 8.5% | +0.216 dB |
| `S20_LR1e2` | 51 | 25.5% | +1.258 dB |
| **Overall oracle vs fixed `S10_LR5e3`** | **200** | **100.0%** | **+0.748 dB [+0.524, +1.025]** |

| Metric | N | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| Oracle ΔPSNR vs fixed AdaSteer | 200 | 0.748 dB | 0.197 dB | 0.068 dB | 0.672 dB |
| Oracle ΔPSNR vs NOTTA | 200 | 0.947 dB | 0.641 dB | -0.597 dB | 2.844 dB |

## OOD quintile stratification

OOD column: `mean_diffusion_loss_caption` (low=Q1, high=Q5).

### Mean PSNR by OOD quintile and config

| quintile | N | fixed AdaSteer | oracle-best | best grid run |
|---|---:|---:|---:|---|
| Q1 | 200 | 18.318 dB | 19.161 dB | `S20_LR1e2` |
| Q2 | 200 | 19.029 dB | 19.419 dB | `S10_LR1e2` |
| Q3 | 199 | 19.546 dB | 20.171 dB | `S20_LR1e2` |
| Q4 | 200 | 17.936 dB | 18.749 dB | `S20_LR1e2` |
| Q5 | 200 | 15.151 dB | 16.221 dB | `S10_LR1e2` |

### H9 pattern check (high OOD → more steps, lower LR?)

| quintile | modal oracle run | steps | LR |
|---|---|---:|---:|
| Q1 | `S20_LR1e2` | 20 | 1e-02 |
| Q2 | `S10_LR1e2` | 10 | 1e-02 |
| Q3 | `S20_LR1e2` | 20 | 1e-02 |
| Q4 | `S20_LR1e2` | 20 | 1e-02 |
| Q5 | `S10_LR1e2` | 10 | 1e-02 |

### Quintile-adaptive policy (pick modal-best run per OOD quintile)

- Mean PSNR: **18.036 dB** vs fixed AdaSteer 17.996 dB (**+0.040 dB**).

## Interpretation notes

- Positive oracle uplift vs fixed S10/LR5e-3 means per-video budget routing has headroom even if population fixed-budget TTA ≈ 0.
- H9 predicts high-OOD quintiles favour *more* steps and *lower* LR; check the pattern table above (opposite sign would extend the H5 falsification).
