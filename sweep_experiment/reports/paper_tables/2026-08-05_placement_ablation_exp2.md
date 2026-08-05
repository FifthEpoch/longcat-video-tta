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

**Caveats.** N=80; effect tiny (~0.05 dB). FVD NOT scored for the arms (COMPUTE_FVD=0;
NOTTA=157.05). Residual band was a single δ added across the whole ~55–80% band — the
objective-specific / appearance-vs-motion multi-vector design (EXP2b) is untested.
VBench rows partial (arm subdir ≠ preview NOTTA subdir) and full-clip-contaminated —
pixel metrics are the clean signal.
