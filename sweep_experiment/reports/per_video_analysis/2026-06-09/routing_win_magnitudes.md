# Routing win magnitudes

**N = 999** videos (exclude corrupt/missing clips from denominator).

## Oracle routing uplift

| Policy | Mean PSNR | Δ vs always-NOTTA |
|---|---:|---:|
| Always NOTTA | 17.930 dB | 0.000 dB |
| Always AdaSteer | 17.938 dB | +0.008 dB |
| Always LoRA | 17.855 dB | -0.076 dB |
| **Oracle (best PSNR)** | **18.156 dB** | **+0.226 dB** |
| Skip AdaSteer if ΔPSNR ≤ 0 | 18.143 dB | +0.213 dB |
| Skip both TTA if ΔPSNR ≤ 0 | 18.156 dB | +0.226 dB |

**Oracle picks:** NOTTA 345 (34.5%) · AdaSteer 446 (44.6%) · LoRA 208 (20.8%)

| Metric | N | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| Oracle ΔPSNR vs NOTTA | 999 | 0.226 dB | 0.028 dB | 0.000 dB | 0.149 dB |

654 / 999 videos (65.5%) have oracle gain > 0.

## Head-to-head

| LoRA beats AdaSteer (ΔPSNR) | 446 | 44.6% |
| AdaSteer beats LoRA | 553 | 55.4% |

## When LoRA beats AdaSteer on ΔPSNR

| Metric | N | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| LoRA ΔPSNR vs NOTTA | 446 | 0.021 dB | -0.003 dB | -0.045 dB | 0.033 dB |
| Margin: LoRA Δ − Ada Δ | 446 | 0.386 dB | 0.111 dB | 0.041 dB | 0.380 dB |

## When AdaSteer beats LoRA on ΔPSNR

| Metric | N | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| AdaSteer ΔPSNR vs NOTTA | 553 | 0.309 dB | 0.083 dB | 0.013 dB | 0.290 dB |
| Margin: Ada Δ − LoRA Δ | 553 | 0.462 dB | 0.132 dB | 0.038 dB | 0.361 dB |

## When NOTTA wins oracle (best absolute PSNR)

| Metric | N | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| NOTTA absolute PSNR | 345 | 18.638 dB | 18.052 dB | 12.448 dB | 23.438 dB |
| Margin over AdaSteer PSNR | 345 | 0.391 dB | 0.126 dB | 0.049 dB | 0.366 dB |
| Margin over LoRA PSNR | 345 | 0.257 dB | 0.059 dB | 0.021 dB | 0.174 dB |
| Margin over best alternative | 345 | 0.132 dB | 0.045 dB | 0.016 dB | 0.124 dB |

## AdaSteer oracle wins

| Metric | N | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| Ada ΔPSNR vs NOTTA | 446 | 0.410 dB | 0.134 dB | 0.045 dB | 0.383 dB |
| Margin over LoRA PSNR | 446 | 0.476 dB | 0.150 dB | 0.052 dB | 0.458 dB |

## LoRA oracle wins

| Metric | N | Mean | Median | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| LoRA ΔPSNR vs NOTTA | 208 | 0.204 dB | 0.038 dB | 0.013 dB | 0.122 dB |
| Margin over AdaSteer PSNR | 208 | 0.397 dB | 0.123 dB | 0.040 dB | 0.395 dB |
