# Panda 1000v retrieval @ N=999 (2048-clip pool)

**Date:** 2026-07-05 (VBench 7-dim backfill confirmed same day)  
**Series:** `panda_1000v_retrieval` — eval `panda_1000_480p`, pool `panda_2048_480p`  
**Cluster:** `sweep_experiment/results/panda_1000v_retrieval/`  
**Merge:** 10 chunks × 4 methods, 999 videos each; 14 vbench files/chunk (7 dims × 2 formats)

## Population metrics

| Method | K | Pick | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | Subj | BG | Aes | Motn | Dyn | IQ | Flick | **VB total** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K5_RAND | 5 | seq | 17.901 | 0.6511 | 0.3374 | **155.7** | 0.903 | 0.931 | 0.442 | 0.986 | 0.594 | 0.615 | 0.975 | **0.778** |
| K5_SIM | 5 | sim | 17.891 | 0.6514 | 0.3371 | 157.0 | 0.903 | 0.931 | 0.442 | 0.986 | 0.601 | 0.615 | 0.975 | **0.779** |
| K10_RAND | 10 | seq | 17.873 | 0.6508 | 0.3383 | 162.1 | 0.903 | 0.931 | 0.442 | 0.986 | 0.606 | 0.615 | 0.975 | **0.780** |
| K10_SIM | 10 | sim | 17.887 | 0.6508 | 0.3375 | 159.7 | 0.903 | 0.931 | 0.441 | 0.986 | 0.611 | 0.615 | 0.975 | **0.780** |

*VB total = unweighted mean of 7 VBench++ dims (same convention as headline tables).*

## vs Panda 1000v standard ([`2026-06-08_headline_1000v.md`](2026-06-08_headline_1000v.md))

| Method | PSNR | FVD | Aes | IQ | Dyn | **VB total** |
|---|---:|---:|---:|---:|---:|---:|
| NOTTA | 17.93 | 154.7 | 0.395 | 0.649 | 0.565 | 0.772 |
| ADA (S10/LR5e-3) | **17.94** | **153.4** | 0.396 | 0.649 | 0.568 | 0.773 |
| LORA_R8_TTA | 17.85 | 157.9 | **0.442** | **0.615** | **0.596** | **0.778** |
| K5_RAND (retrieval) | 17.901 | 155.7 | **0.442** | **0.615** | 0.594 | **0.778** |
| K10_SIM (retrieval) | 17.887 | 159.7 | 0.441 | **0.615** | 0.611 | **0.780** |

## Headline (confirmed 7-dim)

1. **SIM ≈ RAND:** VB-total spread **0.778–0.780** (Δ=0.002); PSNR spread **≤0.03 dB**. Caption similarity does not beat sequential/random neighbours.

2. **PSNR/FVD: no win vs ADA.** Retrieval PSNR **0.04–0.07 dB below** ADA; FVD **+2 to +9** worse (K10 most costly).

3. **LoRA-like VBench tradeoff, not ADA-like:** all retrieval configs show **Aes↑ (+0.046)** and **IQ↓ (−0.034)** vs ADA, matching LORA_R8's aesthetic/quality shift. VB total ≈ LORA (0.778) not ADA (0.773) — but PSNR still tracks ADA more than LORA.

4. **K10 adds compute, not quality:** higher Dyn (+0.01 vs K5) but worse FVD; no PSNR benefit.

## Decision

**Retrieval is not a deployable headline method** on Panda 999v / 2048 pool. Population story: batch neighbour TTA replicates LoRA's **Aes↑ IQ↓** tradeoff without PSNR gains. **25K pool deprioritized.** Optional: per-video win/loss vs NOTTA for tail cases only.

## Reproduce

```bash
python sweep_experiment/scripts/merge_chunks.py \
  --results-dir sweep_experiment/results/panda_1000v_retrieval --recursive
for m in K5_RAND K10_RAND K5_SIM K10_SIM; do
  python scripts/update_merged_with_vbench.py \
    --method-dir sweep_experiment/results/panda_1000v_retrieval/$m --force
done
```
