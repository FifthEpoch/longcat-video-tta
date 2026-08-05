# EXP2 — AdaSteer vector-placement ablation (adaln vs mid-late residual)
**Date:** 2026-08-05 · **Series:** `placement_ablation_panda` · N=80 OOD-stratified preview
**Config:** delta_a, steps=10, lr=1e-3, cond=14/frames=28/gsf=48, seed=42; IDENTICAL
across arms except `--delta-placement`. Residual = auto ~55–80% depth band.
**Motivation:** `sweep_experiment/reports/2026-08-04_literature_v2v_tta_directions.md`

Paired per-video Δ (better = +), bootstrap 95% CI + sign-flip p
(`scripts/analyze_population_effect.py`).

| Comparison | metric | NO-TTA | TTA | Δ | 95% CI | p | effect |
|---|---|---:|---:|---:|---|---:|---|
| **RESID − ADALN** (isolates placement) | psnr | 19.2563 | 19.3047 | **+0.0485** | [+0.0097, +0.0981] | 0.013 | **REAL** |
|  | ssim | 0.6941 | 0.6953 | **+0.0012** | [+0.0001, +0.0026] | 0.041 | **REAL** |
|  | lpips | 0.2582 | 0.2560 | +0.0022 | [−0.0000, +0.0054] | 0.105 | null |
| RESID − NOTTA | psnr | 19.2963 | 19.3047 | +0.0084 | [−0.0041, +0.0216] | 0.211 | null |
|  | ssim | 0.6950 | 0.6953 | +0.0003 | [−0.0000, +0.0006] | 0.119 | null |
| ADALN − NOTTA | psnr | 19.2963 | 19.2563 | −0.0400 | [−0.0902, −0.0016] | 0.076 | null |
|  | ssim | 0.6950 | 0.6941 | −0.0010 | [−0.0023, +0.0001] | 0.102 | null |

Means: **RESID 19.305 ≈ NOTTA 19.296 > ADALN 19.256.**

**Interpretation.** Placement is a *real* lever (residual beats the global-AdaLN δ,
p=0.013), but the effect is that the AdaLN δ mildly **degrades** PSNR (−0.04 vs NOTTA)
and residual placement **removes that harm**, landing back at no-TTA (+0.008, null).
Residual placement does **not** beat no-TTA. The 2026-08-04 hypothesis (AdaLN is a bad
insertion site) is partially confirmed — AdaLN is measurably worse than the residual
band — but fixing placement recovers to neutral, it does not create a gain.

**Methodological note.** The analyzer detected a real +0.05 dB effect at N=80 (p=0.013),
so the vs-NOTTA nulls are true nulls, not underpowering.

---

## All-metric close-out (added 2026-08-05) — VBench++ (7-dim, gen-only) + FVD

Per the standing rule *evaluate on ALL metrics, not just PSNR/SSIM/LPIPS.*

### VBench++ — 7 dims, gen-only clips (`vbench_results_geneval`, N=80), RESID − ADALN

| dim | ADALN | RESID | Δ (RESID−ADALN) | 95% CI | p | effect |
|---|---:|---:|---:|---|---:|---|
| subject_consistency | 0.9551 | 0.9566 | +0.0014 | [−0.0006, +0.0039] | 0.219 | null |
| background_consistency | 0.9595 | 0.9606 | +0.0011 | [−0.0003, +0.0025] | 0.147 | null |
| aesthetic_quality | 0.4553 | 0.4566 | +0.0013 | [−0.0019, +0.0045] | 0.442 | null |
| motion_smoothness | 0.9891 | 0.9893 | +0.0002 | [−0.0001, +0.0005] | 0.197 | null |
| dynamic_degree | 0.7000 | 0.6875 | −0.0125 | [−0.0625, +0.0250] | 1.000 | null |
| imaging_quality | 61.7473 | 61.7717 | +0.0243 | [−0.2118, +0.2546] | 0.852 | null |
| temporal_flickering | 0.9746 | 0.9748 | +0.0002 | [−0.0000, +0.0004] | 0.138 | null |

**No VBench dimension moves** — every CI includes 0. (Numbers are gen-only; they
differ from the earlier online 3-dim values, e.g. subject_consistency 0.941→0.955,
confirming the online eval was conditioning-frame contaminated.)

### FVD — matched-N=80, same `eval_fvd.py` + preview GT cache + 14/14 window + `--force`

| Policy | N | FVD | Δ vs NO-TTA |
|---|---:|---:|---:|
| NOTTA | 80 | 814.604 | +0.000 |
| ADA_ADALN | 80 | 807.924 | **−6.681** |
| ADA_RESID | 80 | 808.871 | **−5.733** |

Both arms nudge FVD down ~0.7–0.8% of baseline — **within small-N FVD noise** (point
estimate only, no CI; FVD is heavily biased/noisy at N=80, cf. 157.05 @ N≈900). Crucially
the ranking is **rank-inconsistent** with pixel: on FVD **ADALN edges RESID**, the opposite
of PSNR/SSIM. So placement does not move FVD in any trustworthy direction.

## Consolidated verdict (all metrics)
- **PSNR/SSIM:** RESID > ADALN (real), but neither beats no-TTA (AdaLN mildly harms;
  residual recovers to neutral).
- **VBench++ (7 dims):** no dimension moves (all null).
- **FVD:** ~−6 for both arms (<1%, no CI, rank-inconsistent with pixel) → null/inconclusive.

**Placement is not the unlock.** Moving AdaSteer's δ from global AdaLN to the mid-late
residual band is a real-but-negligible pixel lever and touches neither VBench nor FVD in
a trustworthy way. Directional next steps stay with EXP3 (TANGO noise-gaussianity FVD
guidance) and EXP1 (better per-video gate probe), not placement.

**Caveats.** N=80; pixel effect tiny (~0.05 dB). FVD is a point estimate (no bootstrap CI);
deltas are sub-1% and rank-inconsistent, so treated as null. Residual band was a single δ
across the whole ~55–80% band — the objective-specific / appearance-vs-motion multi-vector
design (EXP2b) remains untested. VBench is now clean gen-only 7-dim (contaminated online
3-dim superseded).

**Repro.** VBench: `sweep_experiment/sbatch/run_placement_arms_vbench_geneval.sbatch` →
`VBENCH_SUBDIR=vbench_results_geneval scripts/analyze_population_effect.py`. FVD:
`sweep_experiment/sbatch/run_placement_arms_fvd.sbatch` (scorer
`scripts/score_placement_arms_fvd.py`; clip resolution fixed in commit 61becb2 —
metric-fingerprint match for renamed `ytid_segN` clips).
