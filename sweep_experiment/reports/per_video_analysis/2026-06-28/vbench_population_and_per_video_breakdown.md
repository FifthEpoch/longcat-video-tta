# VBench++ population & per-video breakdown (Panda 1000v)

- **Eval set:** Panda 1000v, N=999 videos (28 frames)
- **Baseline:** NOTTA (`panda_1000v_standard/NOTTA`)
- **Per-video stats:** Δ vs NOTTA, win/tie/loss threshold ±0.01
- **Sources:** `merged_summary.json` (population); `vbench_agreement` run 2026-06-28 (per-video)
- **TTA types:** AdaSteer = `delta_a` shared δ; LoRA = rank-8 adapters; retrieval = AdaSteer + K neighbors (SIM or RAND)

---

## A. Population-level VBench++ (999-video mean)

| Method | TTA | Subj | BG | Aes | Motn | Dyn | IQ | Flick | **Total** | **ΔTotal** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **NOTTA** | — | 0.907 | 0.929 | 0.395 | 0.985 | 0.565 | 0.649 | 0.976 | **0.772** | — |
| **AdaSteer** | δ only | 0.907 | 0.929 | 0.396 | 0.985 | 0.568 | 0.649 | 0.976 | **0.773** | +0.001 |
| **LoRA-R8** | LoRA | 0.902 | 0.931 | **0.442** | 0.986 | **0.596** | **0.615** | 0.975 | **0.778** | +0.006 |
| **K5 SIM** | δ+K5 SIM | 0.903 | 0.931 | **0.442** | 0.986 | **0.601** | **0.615** | 0.975 | **0.779** | +0.007 |
| **K5 RAND** | δ+K5 RAND | 0.903 | 0.931 | **0.442** | 0.986 | 0.594 | **0.615** | 0.975 | **0.778** | +0.006 |
| **K10 SIM** | δ+K10 SIM | 0.903 | 0.931 | 0.441 | 0.986 | **0.611** | **0.615** | 0.975 | **0.780** | +0.008 |
| **K10 RAND** | δ+K10 RAND | 0.903 | 0.930 | **0.442** | 0.986 | **0.606** | **0.615** | 0.975 | **0.780** | +0.007 |

*Total = mean of 7 VBench++ dims (AdaState convention). Population PSNR/FVD also flat for AdaSteer; retrieval FVD slightly worse (+2–7).*

### Population Δ vs NOTTA (by dim)

| Method | Subj | BG | Aes | Motn | Dyn | IQ | Flick |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdaSteer | 0.000 | 0.000 | +0.001 | 0.000 | +0.003 | 0.000 | 0.000 |
| LoRA-R8 | −0.005 | +0.002 | **+0.047** | +0.001 | **+0.031** | **−0.034** | −0.001 |
| K5 SIM | −0.004 | +0.002 | **+0.047** | +0.001 | **+0.036** | **−0.034** | −0.001 |
| K5 RAND | −0.004 | +0.002 | **+0.047** | +0.001 | +0.029 | **−0.034** | −0.001 |
| K10 SIM | −0.004 | +0.002 | +0.046 | +0.001 | **+0.046** | **−0.034** | −0.001 |
| K10 RAND | −0.004 | +0.001 | **+0.047** | +0.001 | **+0.041** | **−0.034** | −0.001 |

---

## B. Per-video win% / tie% / loss% vs NOTTA (±0.01)

*NOTTA omitted (baseline). **win% / tie% / loss%** per dimension.*

### AdaSteer (`ADA`) — mostly flat

| Dim | win% | tie% | loss% | Pattern |
|---|---:|---:|---:|---|
| Subj | 7.0 | 86.4 | 6.6 | flat |
| BG | 6.4 | 87.9 | 5.7 | flat |
| Aes | 13.6 | 73.6 | 12.7 | flat |
| Motn | 0.4 | 99.1 | 0.5 | flat |
| Dyn | 2.8 | 94.6 | 2.5 | flat |
| **IQ** | **46.5** | 5.0 | **48.4** | **≈ coin flip** |
| Flick | 1.1 | 97.8 | 1.0 | flat |

### LoRA-R8 (`LORA_R8_TTA`) — frontier shift + motion/temporal spread

| Dim | win% | tie% | loss% | Pattern |
|---|---:|---:|---:|---|
| Subj | 5.1 | 72.5 | 22.3 | modest ↓ |
| BG | 16.7 | 72.8 | 10.4 | mixed |
| **Aes** | **93.5** | 5.4 | 1.1 | **strong ↑** |
| Motn | 22.7 | 58.1 | 19.1 | spread |
| Dyn | 26.7 | 49.6 | 23.6 | spread |
| **IQ** | 43.5 | 0.1 | **56.4** | **net ↓** |
| Flick | 31.2 | 36.3 | 32.4 | spread |

### AdaSteer + retrieval — K5/K10 × SIM/RAND (all four nearly identical)

| Dim | K5 SIM | K5 RAND | K10 SIM | K10 RAND | Pattern |
|---|---:|---:|---:|---:|---|
| Subj | 8.4 / 22.0 | 8.0 / 21.5 | 7.2 / 22.5 | 6.9 / 22.4 | modest ↓ |
| BG | 18.6 / 12.1 | 20.1 / 11.3 | 18.5 / 13.3 | 18.5 / 11.8 | mixed |
| **Aes** | **92.1 / 1.7** | **92.5 / 2.0** | **92.5 / 2.0** | **92.0 / 1.4** | **~92% win** |
| Motn | 0.3 / 0.2 | 0.4 / 0.3 | 0.3 / 0.1 | 0.3 / 0.3 | flat |
| Dyn | 5.2 / 1.6 | 4.6 / 1.7 | 6.3 / 1.7 | 5.6 / 1.5 | mostly ties* |
| **IQ** | **25.0 / 74.9** | **24.8 / 74.9** | **24.0 / 76.0** | **23.8 / 76.0** | **~75% loss** |
| Flick | 0.5 / 0.3 | 0.4 / 0.8 | 0.5 / 0.8 | 0.4 / 0.6 | flat |

*Cells show win% / loss%; tie% omitted for space. *Dynamic degree is often binary → most videos tie at ±0.01 even when population Dyn shifts +0.03–0.05.*

---

## C. Compact per-video comparison (win% / loss% only)

| Dim | AdaSteer | LoRA-R8 | K5 SIM | K5 RAND | K10 SIM | K10 RAND |
|---|---:|---:|---:|---:|---:|---:|
| Subj | 7/7 | 5/22 | 8/22 | 8/22 | 7/23 | 7/22 |
| BG | 6/6 | 17/10 | 19/12 | 20/11 | 19/13 | 19/12 |
| **Aes** | **14/13** | **94/1** | **92/2** | **93/2** | **93/2** | **92/1** |
| Motn | 0/1 | 23/19 | 0/0 | 0/0 | 0/0 | 0/0 |
| Dyn | 3/3 | 27/24 | 5/2 | 5/2 | 6/2 | 6/2 |
| **IQ** | **47/48** | **44/56** | **25/75** | **25/75** | **24/76** | **24/76** |
| Flick | 1/1 | 31/32 | 1/0 | 0/1 | 1/1 | 0/1 |

---

## D. Synthesis

| Family | Population story | Per-video story | Agreement? |
|---|---|---|---|
| **No-TTA** | reference | — | — |
| **AdaSteer alone** | ΔTotal ≈ +0.001; all dims ≈0 | ~81% PSNR ties @0.5 dB; VBench mostly ties except IQ ~50/50 | Yes — flat everywhere |
| **LoRA-R8** | Aes↑ Dyn↑ IQ↓; ΔTotal +0.006 | ~94% videos win Aes; ~56% lose IQ; more motion/temporal spread | Yes — frontier visible per video |
| **AdaSteer + retrieval** | Same frontier as LoRA on Aes/IQ; Dyn↑ at population | ~92% win Aes, ~75% lose IQ; K5≈K10, SIM≈RAND | Yes — population shift = per-video redistribution |
| **Retrieval vs AdaSteer** | Retrieval moves Aes/IQ/Dyn; AdaSteer does not | Retrieval adds coherent per-video Aes↑ IQ↓; AdaSteer flat | Retrieval effect is **not** explained by aggregate AdaSteer noise |

**Regenerate on cluster:**

```bash
git pull
bash scripts/run_panda_vbench_breakdown.sh
```
