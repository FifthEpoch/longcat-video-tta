# Panda 1000v retrieval @ N=999 (2048-clip pool)

**Date:** 2026-07-05  
**Series:** `panda_1000v_retrieval` — eval `panda_1000_480p`, pool `panda_2048_480p`  
**Cluster:** `sweep_experiment/results/panda_1000v_retrieval/`  
**Merge:** 10 chunks × 4 methods, 999 videos each

## Population metrics (from `merge_chunks.py`)

| Method | K | Pool pick | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | FID↓ | Aes | BG | Subj |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K5_RAND | 5 | sequential | 17.901 | 0.6511 | 0.3374 | **155.7** | 25.1 | 0.442 | 0.931 | 0.903 |
| K5_SIM | 5 | similarity | 17.891 | 0.6514 | 0.3371 | 157.0 | 25.1 | 0.442 | 0.931 | 0.903 |
| K10_RAND | 10 | sequential | 17.873 | 0.6508 | 0.3383 | 162.1 | 25.0 | 0.441 | 0.930 | 0.903 |
| K10_SIM | 10 | similarity | 17.887 | 0.6508 | 0.3375 | 159.7 | 25.1 | 0.441 | 0.931 | 0.903 |

*VBench columns above are the 3 dims printed at merge time; run full backfill + `update_merged_with_vbench.py` per method for all 7 dims.*

## vs Panda 1000v standard baselines ([`2026-06-08_headline_1000v.md`](2026-06-08_headline_1000v.md))

| Method | PSNR | FVD | Aes (7-dim table) |
|---|---:|---:|---:|
| NOTTA | 17.93 | 154.7 | 0.395 |
| ADA (S10/LR5e-3) | **17.94** | **153.4** | 0.396 |
| LORA_R8_TTA | 17.85 | 157.9 | **0.442** |
| K5_RAND (retrieval) | 17.901 | 155.7 | 0.442† |
| K10_SIM (retrieval) | 17.887 | 159.7 | 0.441† |

†Partial merge print only; confirm with full VBench backfill.

## Headline

1. **SIM ≈ RAND on Panda** (unlike UCF's class-block confound, this is a clean test): PSNR spread across all 4 configs is **≤0.03 dB**; FVD spread ≤6.4. Caption-similarity neighbours do not beat random/sequential neighbours at population level.

2. **Retrieval batch-TTA ≈ single-video AdaSteer on PSNR/FVD**, not a win: all retrieval variants sit **0.04–0.07 dB below ADA** on PSNR; FVD **+2 to +9** vs ADA (K10 worst).

3. **Aesthetic bump** (~0.442 vs ADA 0.396) mirrors LoRA's Aes↑ pattern — possible batch-adaptation side effect; needs full 7-dim VBench + per-video analysis before claiming tradeoff.

4. **K10 > K cost, no quality gain:** longer batch training (10× vs 5× neighbours) does not improve PSNR; FVD degrades at K=10.

## Decision

**Retrieval is not a paper headline win** on Panda 1000v with 2048 pool — confirms UCF-style SIM≈RAND null at a meaningful scale. Narrative: batch retrieval augmentation does not beat fixed-budget AdaSteer on population metrics; optional per-video / Aes tradeoff follow-up only.

## Reproduce

```bash
python sweep_experiment/scripts/merge_chunks.py \
  --results-dir sweep_experiment/results/panda_1000v_retrieval --recursive
for m in K5_RAND K10_RAND K5_SIM K10_SIM; do
  python scripts/update_merged_with_vbench.py \
    --method-dir sweep_experiment/results/panda_1000v_retrieval/$m --force
done
```
