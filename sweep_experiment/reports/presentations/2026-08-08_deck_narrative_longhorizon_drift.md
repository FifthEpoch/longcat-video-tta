# Presentation deck — narrative start: "LongCat drifts under long-horizon rollout"

**Purpose:** This is the OPENING NARRATIVE for the next presentation deck. It is
organized slide-by-slide. All numbers are from the completed drift run
(`diag_longhorizon_drift.py`, SLURM job **15497180**, NOTTA, N=24 videos × 8
chunks, reencode geometry cond=14 / frames=28 / gen_start=48, 50 steps, CFG=4.0,
seed=42). Figures live in `../paper_figures/2026-08-08_longhorizon_drift/` and are
regenerable via `scripts/make_drift_presentation_figs.py`. Full data + verdict
table: `../paper_tables/2026-08-08_longhorizon_drift_presentation.md`.

**Status caveat for the room:** this is a NOTTA-baseline *discovery*. Two controls
are still running (native-geometry + fixed-delta intervention) — flag them as the
immediate follow-through so we don't over-claim.

---

## Slide 1 — The hook

**"On the easy task, nothing we do matters. So we made the task hard — and the
model breaks in a way we can fix."**

- Every intervention we tried (AdaSteer delta, placement ablation, TANGO
  gaussianity guidance) was ~null on our short, in-domain, single-chunk 14→14
  continuation.
- Diagnosis (2026-08-06 problem-difficulty audit): LongCat-Video (13.6B, RLHF,
  continuation-pretrained) is *too strong* for that framing — headroom is small
  **by construction**, not because TTA can't help.
- The field finds its headroom in **long-horizon autoregressive rollout**, where
  error compounds chunk-over-chunk. So we tested that regime.

---

## Slide 2 — What we did (setup)

- Identical per-chunk geometry to all our prior runs; we simply **feed the
  model's own generated tail back** as the next chunk's conditioning and repeat
  for **8 chunks**. Nothing is trained — this is the **NOTTA baseline**.
- We measure quality per chunk with two families:
  - **GT-free (path-independent)** signals, defined for the *entire* rollout even
    after the source clip's ground truth runs out: sharpness (Laplacian
    variance), colorfulness (Hasler–Süsstrunk), contrast, temporal motion.
  - **GT-referenced** metrics (PSNR / SSIM / LPIPS) where the source clip still
    overlaps the rollout.

---

## Slide 3 — Headline result

**LongCat degrades strongly and MONOTONICALLY over the rollout.**

![Normalized GT-free drift](../paper_figures/2026-08-08_longhorizon_drift/drift_gtfree_normalized.png)

- Short-horizon saturation does **not** survive chaining.
- Over-saturation: colorfulness **+58%**. HF-artifact accumulation: sharpness
  **+258%**. Contrast **+13%**. All monotone.

---

## Slide 4 — The drift mode is specific (not generic collapse)

![Raw GT-free panels](../paper_figures/2026-08-08_longhorizon_drift/drift_gtfree_raw.png)

- Temporal motion is **flat/noisy (−12%, non-monotone)** → the failure is **NOT
  motion collapse** (this distinguishes us from motion-drift papers).
- The mode is **progressive over-saturation + rising high-frequency (non-white)
  residual** — precisely the signature a whiteness/spectral steering term or a
  re-anchoring correction is designed to flatten.

---

## Slide 5 — GT-referenced fidelity collapses too

![PSNR/LPIPS collapse](../paper_figures/2026-08-08_longhorizon_drift/drift_gt_fidelity.png)

- PSNR **19.0 → 9.8 dB (−48%, −1.12 dB/chunk)**, SSIM 0.71 → 0.31, LPIPS
  **+201%** — front-loaded (PSNR already 12.5 dB by chunk 3).
- Caveat we state up front: lead with the GT-free signals, because PSNR partly
  reflects *legitimate* divergence from a single GT path (continuation is
  multimodal), and GT coverage shrinks (n = 24 → 13) as the clip runs out.

---

## Slide 6 — Data appendix (per-chunk, mean over N=24)

| Chunk | Sharpness | Motion | Colorfulness | Contrast | PSNR | SSIM | LPIPS | GT n |
|------:|----------:|-------:|-------------:|---------:|-----:|-----:|------:|-----:|
| 1 | 0.0070 | 0.0232 | 0.1488 | 0.2358 | 19.02 | 0.710 | 0.248 | 24 |
| 2 | 0.0088 | 0.0218 | 0.1556 | 0.2402 | 14.87 | 0.587 | 0.409 | 24 |
| 3 | 0.0108 | 0.0238 | 0.1624 | 0.2465 | 12.51 | 0.505 | 0.520 | 19 |
| 4 | 0.0138 | 0.0210 | 0.1693 | 0.2515 | 11.65 | 0.432 | 0.597 | 18 |
| 5 | 0.0168 | 0.0228 | 0.1762 | 0.2563 | 11.11 | 0.390 | 0.638 | 16 |
| 6 | 0.0200 | 0.0211 | 0.1894 | 0.2583 | 10.58 | 0.351 | 0.691 | 15 |
| 7 | 0.0223 | 0.0202 | 0.2149 | 0.2629 | 10.16 | 0.321 | 0.715 | 14 |
| 8 | 0.0251 | 0.0204 | 0.2354 | 0.2674 |  9.82 | 0.311 | 0.746 | 13 |

**Verdict (chunk 1 → 8):** sharpness +258%, colorfulness +58%, contrast +13%
(all monotone); motion −12% (flat); PSNR −48%, SSIM −56%, LPIPS +201%.

---

## Slide 7 — Why this matters / what's next

- **Why it matters:** the drift regime gives our interventions room to show an
  effect for the first time, and the mode (HF/over-saturation) points at a
  concrete fix (whiteness/spectral steering = TANGO++; re-anchoring corrections).
- **In flight (controls — do not over-claim yet):**
  1. **Native-geometry control** (13-cond/93-frame window): proves the drift is
     *inherent*, not an artifact of our short-window re-conditioning.
  2. **Intervention test:** hold a fixed AdaSteer delta across the rollout — does
     it flatten the curve, or decay (→ motivates a *streaming* per-chunk delta)?
- **Ask of the deck:** frame drift as the enabling discovery; the interventions
  are the paper's contribution once the controls land.
