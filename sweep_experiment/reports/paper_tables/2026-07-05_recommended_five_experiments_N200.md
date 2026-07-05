# Recommended five-experiment program @ N=200

**Date:** 2026-07-05  
**Source:** `recommended_five_experiments/recommended_five_experiments_summary.md`  
**Fixed comparator:** S10_LR5e3 | **Oracle headroom (total):** +0.140  
**Cluster path:** `sweep_experiment/reports/per_video_analysis/2026-07-05/recommended_five_experiments/`

## Results (VBench total unless noted)

| # | Experiment | Captured % | Δ vs fixed (approx) | vs 9% linear | Verdict |
|---|---|---:|---|---:|---|
| 1a | Commit probe (S2→S5, S10→S10) | 2.9 | +0.004 | worse | Fail |
| 1a | — Dyn captured | 33.3 | — | — | Partial on Dyn only |
| 1b | Ridge probe 3-way (S2+S10 → S5/S10/S20) | **12.1** | +0.017 | **best** | Fail 25% bar |
| 2 | ΔDyn router → **total VBench** | 4.9 | +0.007 | worse | Fail |
| 2 | ΔDyn router → Dyn (in-sample) | 100 | — | — | Not deployable OOF |
| 3 | Pairwise logistic top-4 | −7.4 | −0.010 | worse | Fail |
| 3 | Pairwise GBM top-4 | −0.8 | −0.001 | worse | Fail |
| 4 | Best-of-3 LPIPS NR | −3.0 | — | τ≈0 | Fail |
| 4 | Best-of-3 SSIM NR | −4.3 | — | τ≈0 | Fail |
| 4 | Best-of-3 PSNR ref | −3.1 | — | τ≈0 | Fail |
| 5 | IQ-constrained TTA | — | — | — | Not run (GPU) |

## Decisions

- **999v × 12 config routing:** NO-GO  
- **Real GPU probe-and-route:** LOW ROI (simulation ridge 12.1% ≈ PSNR proxy ceiling 11.5%)  
- **Paper line:** Oracle headroom real (+0.14); no deployable router clears 25% captured on total VBench @ N=200  

## Reproduce

```bash
bash sweep_experiment/sbatch/submit_recommended_five_experiments.sh
# or rerun failed + aggregate:
python3 scripts/run_recommended_five_experiments.py --aggregate-only \
  --output-dir sweep_experiment/reports/per_video_analysis/2026-07-05/recommended_five_experiments
```
