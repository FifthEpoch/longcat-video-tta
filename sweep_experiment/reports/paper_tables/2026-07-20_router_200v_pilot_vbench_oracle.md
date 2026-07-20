# AdaSteer budget-grid VBench++ oracle analysis

**Series:** `sweep_experiment/results/panda_ood_budget_pilot`
**Fixed AdaSteer:** `S10_LR5e3` (S10/LR=5e-3).
**VBench grid configs:** 12 of 12 (`S2_LR1e3`, `S2_LR5e3`, `S2_LR1e2`, `S5_LR1e3`, `S5_LR5e3`, `S5_LR1e2`, `S10_LR1e3`, `S10_LR5e3`, `S10_LR1e2`, `S20_LR1e3`, `S20_LR5e3`, `S20_LR1e2`)
**Active VBench dims:** aesthetic_quality, background_consistency, dynamic_degree, imaging_quality, motion_smoothness, subject_consistency, temporal_flickering (7/7)
**Union N:** 999 videos with PSNR; oracle denominators vary by VBench coverage.

## VBench coverage (per grid config)

| Config | Videos (any dim) | All 7 dims |
|---|---:|---:|
| `S2_LR1e3` | 200 | 200 |
| `S2_LR5e3` | 200 | 200 |
| `S2_LR1e2` | 200 | 200 |
| `S5_LR1e3` | 200 | 200 |
| `S5_LR5e3` | 200 | 200 |
| `S5_LR1e2` | 200 | 200 |
| `S10_LR1e3` | 200 | 200 |
| `S10_LR5e3` | 200 | 200 |
| `S10_LR1e2` | 200 | 200 |
| `S20_LR1e3` | 200 | 200 |
| `S20_LR5e3` | 200 | 200 |
| `S20_LR1e2` | 200 | 200 |

## Population routing uplift (VBench-driven oracle)

Oracle picks the grid config with **max VBench** per video (not PSNR). Δ columns are mean(oracle − baseline) on paired videos.

| Oracle target | N | Oracle mean | Fixed mean | NOTTA mean | Δ vs fixed | Δ vs NOTTA |
|---|---:|---:|---:|---:|---:|---:|
| VBench total | 200 | 9.538 | 9.398 | 9.993 | +0.140 | -0.455 |
| Subj | 200 | 0.908 | 0.901 | — | +0.007 | — |
| BG | 200 | 0.933 | 0.927 | — | +0.007 | — |
| Aes | 200 | 0.450 | 0.436 | — | +0.014 | — |
| Motn | 200 | 0.986 | 0.984 | — | +0.002 | — |
| Dyn | 200 | 0.640 | 0.595 | — | +0.045 | — |
| IQ | 200 | 61.943 | 60.970 | — | +0.973 | — |
| Flick | 200 | 0.974 | 0.973 | — | +0.002 | — |

### Bootstrap 95% CI — VBench-total oracle Δ vs fixed AdaSteer

| Stat | Value |
|---|---:|
| Mean Δ | +0.140 |
| 95% CI | [+0.112, +0.171] |
| CI excludes 0 | yes |

## PSNR oracle vs VBench-total oracle (config agreement)

- Videos with both oracles: **200**
- Same config picked: **30** (15.0%)

- PSNR-oracle uplift vs fixed (from PSNR script): **+0.748 dB**
- VBench-total oracle Δ vs fixed: **+0.140**

## Oracle pick frequency — VBench total

| Config | Picks | % |
|---|---:|---:|
| `S20_LR1e2` | 57 | 28.5% |
| `S10_LR1e2` | 26 | 13.0% |
| `S20_LR5e3` | 21 | 10.5% |
| `S5_LR1e2` | 17 | 8.5% |
| `S2_LR1e2` | 13 | 6.5% |
| `S10_LR1e3` | 13 | 6.5% |
| `S10_LR5e3` | 13 | 6.5% |
| `S5_LR5e3` | 13 | 6.5% |
| `S2_LR5e3` | 13 | 6.5% |
| `S5_LR1e3` | 6 | 3.0% |
| `S20_LR1e3` | 4 | 2.0% |
| `S2_LR1e3` | 4 | 2.0% |

## Oracle pick frequency — Subj

| Config | Picks | % |
|---|---:|---:|
| `S20_LR1e2` | 46 | 23.0% |
| `S20_LR5e3` | 21 | 10.5% |
| `S10_LR1e2` | 19 | 9.5% |
| `S2_LR1e2` | 17 | 8.5% |
| `S5_LR1e2` | 16 | 8.0% |
| `S10_LR5e3` | 15 | 7.5% |
| `S5_LR5e3` | 13 | 6.5% |
| `S5_LR1e3` | 13 | 6.5% |
| `S2_LR1e3` | 12 | 6.0% |
| `S20_LR1e3` | 11 | 5.5% |
| `S2_LR5e3` | 11 | 5.5% |
| `S10_LR1e3` | 6 | 3.0% |

## Oracle pick frequency — BG

| Config | Picks | % |
|---|---:|---:|
| `S20_LR1e2` | 34 | 17.0% |
| `S10_LR1e2` | 28 | 14.0% |
| `S20_LR5e3` | 21 | 10.5% |
| `S5_LR1e2` | 18 | 9.0% |
| `S10_LR1e3` | 16 | 8.0% |
| `S2_LR1e2` | 15 | 7.5% |
| `S5_LR1e3` | 14 | 7.0% |
| `S5_LR5e3` | 13 | 6.5% |
| `S2_LR1e3` | 13 | 6.5% |
| `S10_LR5e3` | 12 | 6.0% |
| `S2_LR5e3` | 9 | 4.5% |
| `S20_LR1e3` | 7 | 3.5% |

## Oracle pick frequency — Aes

| Config | Picks | % |
|---|---:|---:|
| `S20_LR1e2` | 45 | 22.5% |
| `S10_LR1e2` | 24 | 12.0% |
| `S10_LR5e3` | 20 | 10.0% |
| `S5_LR5e3` | 17 | 8.5% |
| `S20_LR5e3` | 15 | 7.5% |
| `S2_LR1e2` | 15 | 7.5% |
| `S5_LR1e3` | 15 | 7.5% |
| `S10_LR1e3` | 11 | 5.5% |
| `S20_LR1e3` | 11 | 5.5% |
| `S2_LR5e3` | 10 | 5.0% |
| `S5_LR1e2` | 9 | 4.5% |
| `S2_LR1e3` | 8 | 4.0% |

## Oracle pick frequency — Motn

| Config | Picks | % |
|---|---:|---:|
| `S20_LR1e2` | 47 | 23.5% |
| `S10_LR1e2` | 28 | 14.0% |
| `S5_LR1e2` | 18 | 9.0% |
| `S2_LR1e3` | 16 | 8.0% |
| `S2_LR5e3` | 15 | 7.5% |
| `S20_LR5e3` | 14 | 7.0% |
| `S5_LR1e3` | 12 | 6.0% |
| `S5_LR5e3` | 12 | 6.0% |
| `S2_LR1e2` | 11 | 5.5% |
| `S10_LR1e3` | 11 | 5.5% |
| `S20_LR1e3` | 9 | 4.5% |
| `S10_LR5e3` | 7 | 3.5% |

## Oracle pick frequency — Dyn

| Config | Picks | % |
|---|---:|---:|
| `S2_LR1e3` | 193 | 96.5% |
| `S2_LR5e3` | 2 | 1.0% |
| `S20_LR1e3` | 1 | 0.5% |
| `S20_LR5e3` | 1 | 0.5% |
| `S5_LR5e3` | 1 | 0.5% |
| `S2_LR1e2` | 1 | 0.5% |
| `S20_LR1e2` | 1 | 0.5% |

## Oracle pick frequency — IQ

| Config | Picks | % |
|---|---:|---:|
| `S20_LR1e2` | 57 | 28.5% |
| `S10_LR1e2` | 28 | 14.0% |
| `S20_LR5e3` | 21 | 10.5% |
| `S5_LR1e2` | 18 | 9.0% |
| `S10_LR5e3` | 14 | 7.0% |
| `S2_LR1e2` | 13 | 6.5% |
| `S10_LR1e3` | 13 | 6.5% |
| `S2_LR5e3` | 12 | 6.0% |
| `S5_LR5e3` | 11 | 5.5% |
| `S5_LR1e3` | 6 | 3.0% |
| `S2_LR1e3` | 4 | 2.0% |
| `S20_LR1e3` | 3 | 1.5% |

## Oracle pick frequency — Flick

| Config | Picks | % |
|---|---:|---:|
| `S20_LR1e2` | 58 | 29.0% |
| `S10_LR1e2` | 31 | 15.5% |
| `S20_LR5e3` | 17 | 8.5% |
| `S2_LR1e3` | 14 | 7.0% |
| `S5_LR1e3` | 14 | 7.0% |
| `S20_LR1e3` | 13 | 6.5% |
| `S5_LR1e2` | 12 | 6.0% |
| `S10_LR5e3` | 10 | 5.0% |
| `S5_LR5e3` | 8 | 4.0% |
| `S10_LR1e3` | 8 | 4.0% |
| `S2_LR5e3` | 8 | 4.0% |
| `S2_LR1e2` | 7 | 3.5% |

## OOD quintile stratification (VBench-total oracle)

OOD column: `mean_diffusion_loss_caption` (low=Q1, high=Q5).

| Quintile | N | Fixed VBench | Oracle VBench | Modal config | Steps | LR |
|---|---:|---:|---:|---|---:|---:|
| Q1 | 200 | 9.526 | 9.631 | `S20_LR1e2` | 20 | 1e-02 |
| Q2 | 200 | 8.864 | 9.019 | `S20_LR1e2` | 20 | 1e-02 |
| Q3 | 199 | 9.438 | 9.560 | `S20_LR1e2` | 20 | 1e-02 |
| Q4 | 200 | 9.447 | 9.607 | `S20_LR1e2` | 20 | 1e-02 |
| Q5 | 200 | 9.715 | 9.874 | `S10_LR1e2` | 10 | 1e-02 |

### Quintile-adaptive VBench policy (modal oracle config per quintile)

| Target | Adaptive mean | Fixed mean | Δ vs fixed |
|---|---:|---:|---:|
| VBench total | 9.409 | 9.398 | +0.011 |
| Subj | 0.900 | 0.901 | -0.001 |
| BG | 0.928 | 0.927 | +0.001 |
| Aes | 0.439 | 0.436 | +0.003 |
| Motn | 0.984 | 0.984 | +0.000 |
| Dyn | 0.605 | 0.595 | +0.010 |
| IQ | 61.059 | 60.970 | +0.089 |
| Flick | 0.972 | 0.973 | -0.001 |

### Per-dimension modal oracle config by OOD quintile

**Subj** (`subject_consistency`):

| Quintile | Modal config | Steps | LR |
|---|---|---:|---:|
| Q1 | `S20_LR1e2` | 20 | 1e-02 |
| Q2 | `S20_LR1e2` | 20 | 1e-02 |
| Q3 | `S20_LR5e3` | 20 | 5e-03 |
| Q4 | `S20_LR1e2` | 20 | 1e-02 |
| Q5 | `S20_LR1e2` | 20 | 1e-02 |

**BG** (`background_consistency`):

| Quintile | Modal config | Steps | LR |
|---|---|---:|---:|
| Q1 | `S20_LR1e2` | 20 | 1e-02 |
| Q2 | `S10_LR1e2` | 10 | 1e-02 |
| Q3 | `S10_LR1e2` | 10 | 1e-02 |
| Q4 | `S10_LR1e2` | 10 | 1e-02 |
| Q5 | `S10_LR1e2` | 10 | 1e-02 |

**Aes** (`aesthetic_quality`):

| Quintile | Modal config | Steps | LR |
|---|---|---:|---:|
| Q1 | `S20_LR1e2` | 20 | 1e-02 |
| Q2 | `S20_LR1e2` | 20 | 1e-02 |
| Q3 | `S20_LR1e2` | 20 | 1e-02 |
| Q4 | `S20_LR1e2` | 20 | 1e-02 |
| Q5 | `S10_LR1e2` | 10 | 1e-02 |

**Motn** (`motion_smoothness`):

| Quintile | Modal config | Steps | LR |
|---|---|---:|---:|
| Q1 | `S20_LR1e2` | 20 | 1e-02 |
| Q2 | `S20_LR1e2` | 20 | 1e-02 |
| Q3 | `S20_LR1e2` | 20 | 1e-02 |
| Q4 | `S10_LR1e2` | 10 | 1e-02 |
| Q5 | `S20_LR1e2` | 20 | 1e-02 |

**Dyn** (`dynamic_degree`):

| Quintile | Modal config | Steps | LR |
|---|---|---:|---:|
| Q1 | `S2_LR1e3` | 2 | 1e-03 |
| Q2 | `S2_LR1e3` | 2 | 1e-03 |
| Q3 | `S2_LR1e3` | 2 | 1e-03 |
| Q4 | `S2_LR1e3` | 2 | 1e-03 |
| Q5 | `S2_LR1e3` | 2 | 1e-03 |

**IQ** (`imaging_quality`):

| Quintile | Modal config | Steps | LR |
|---|---|---:|---:|
| Q1 | `S20_LR1e2` | 20 | 1e-02 |
| Q2 | `S20_LR1e2` | 20 | 1e-02 |
| Q3 | `S20_LR1e2` | 20 | 1e-02 |
| Q4 | `S20_LR1e2` | 20 | 1e-02 |
| Q5 | `S10_LR1e2` | 10 | 1e-02 |

**Flick** (`temporal_flickering`):

| Quintile | Modal config | Steps | LR |
|---|---|---:|---:|
| Q1 | `S20_LR1e2` | 20 | 1e-02 |
| Q2 | `S20_LR1e2` | 20 | 1e-02 |
| Q3 | `S20_LR1e2` | 20 | 1e-02 |
| Q4 | `S20_LR1e2` | 20 | 1e-02 |
| Q5 | `S20_LR1e2` | 20 | 1e-02 |

## Interpretation notes

- If VBench-oracle Δ ≪ PSNR-oracle Δ (in comparable units), perceptual routing needs different features than pixel routing (cf. method-level PSNR vs VBench-total oracle gap on 999v).
- Compare quintile modal configs here to the PSNR H9 table: high-OOD may prefer different steps/LR for VBench than for PSNR.
- Quintile-adaptive policy captures only a fraction of oracle headroom when modal configs differ across quintiles but within-quintile variance is large.
