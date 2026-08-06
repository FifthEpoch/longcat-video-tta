# EXP2 — placement arms RELIABLE-N FVD (N=500): AdaSteer TTA degrades FVD

**Date:** 2026-08-06 · **Series:** `placement_ablation_panda_512v` (8 chunked gen jobs
15410043–15410052, all COMPLETED) · **Scorer:** `run_placement_arms_fvd.sbatch` (job
15410055) → `eval_fvd.py` + preview GT cache
`gt_caches/panda_ood_budget_1000v_preview_longcat.npz` (939 ref videos) + 14/14 window +
`--force`. **Matched-N common set = 500** (`linked=500 missing=0` for all three arms).

Δ vs NO-TTA (negative = better). FVD decomposed into mean-shift + covariance (trace) terms.

| Policy | N | FVD | Δ vs NO-TTA | mean_term | trace_term | mean_frac |
|---|---:|---:|---:|---:|---:|---:|
| **NOTTA** | 500 | **139.914** | +0.000 | 8.951 | 130.963 | 6.40% |
| ADA_ADALN | 500 | 149.825 | **+9.910** | 10.076 | 139.749 | 6.73% |
| ADA_RESID | 500 | 152.636 | **+12.722** | 10.829 | 141.808 | 7.09% |

## Interpretation (definitive)

This escapes the small-N covariance bias that made the N=80 FVD uninterpretable. **Same arms
@ N=80:** NOTTA 814.6 / ADALN 807.9 / RESID 808.9 — pure rank-deficiency noise (400-dim I3D
features, N=80). At **N=500 ≫ 400**, NOTTA = 139.9 ≈ the 157@N≈900 headline, so the scale is
trustworthy.

At reliable N, **both AdaSteer placements DEGRADE FVD**, monotonically
**NOTTA (139.9) < ADALN (149.8) < RESID (152.6)** — and the degradation shows up in **both**
the mean-shift term (8.95 → 10.08 → 10.83) and the dominant covariance/trace term
(130.96 → 139.75 → 141.81). Residual placement is the **worst**, not a rescue.

**Answer to "does residual placement help FVD": NO.** It is the worst of the three. Placement
is conclusively not an FVD lever, and AdaSteer's per-video δ *hurts* distributional realism
regardless of insertion site.

## Supersession / audit trail

Supersedes the "FVD — matched-N=80" section of
`2026-08-05_placement_ablation_exp2.md` (explicitly caveated there as no-CI / small-N noise /
rank-inconsistent). That section reported ADALN −6.7 / RESID −5.7 vs NOTTA at N=80 and called
it null; the reliable N=500 result shows the true sign is the **opposite** (both +worse) — a
textbook demonstration of why we scaled N. Do not cite the N=80 FVD numbers.

## Consolidated EXP2 verdict (all metrics, best available N)

- **PSNR/SSIM (N=80):** RESID > ADALN is real (+0.05 dB, p=0.013) but neither beats no-TTA.
- **VBench++ 7-dim (N=80, gen-only):** no dimension moves (all null).
- **FVD (N=500, RELIABLE):** NOTTA best; ADALN +9.9; RESID +12.7 → **TTA hurts FVD, residual worst.**

**EXP2 is closed as a negative across every metric.** Placement is not the unlock. The next
FVD lever is **EXP3 (TANGO distribution-level sampling guidance)**, now unblocked after fixing
a guidance-magnitude bf16 no-op bug (the raw 1/n-normalized gradient × λ rounded to zero in
bf16; rescaled to a fraction λ of the per-sample velocity norm).

## Repro

```bash
SERIES_NAME=placement_ablation_panda_512v MAX_VIDEOS=512 CHUNK=128 COMPUTE_VBENCH=0 \
  bash delta_experiment/sbatch/submit_placement_ablation.sh
# FVD auto-chains via run_placement_arms_fvd.sbatch ->
#   OUTPUT_ROOT=.../placement_arms_placement_ablation_panda_512v
cat sweep_experiment/reports/budget_oracle_fvd_1000v_preview/placement_arms_placement_ablation_panda_512v/placement_arms_fvd_summary.md
```
