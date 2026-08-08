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

## Slide 7 — Follow-through control #1: a FIXED delta does NOT flatten the drift

**Result (EXP-B, `delta_reencode`, N=24, reencode geometry, paired seeds vs
NOTTA; mean `delta_norm`=0.139 so the delta really trained):** holding a single
AdaSteer delta (trained once on the chunk-0 context) fixed across the rollout
leaves the curves **essentially parallel to NOTTA** — the degradation slope is
unchanged.

![NOTTA vs fixed delta](../paper_figures/2026-08-08_longhorizon_drift/drift_intervention_notta_vs_delta.png)

| Signal (chunk 1 → 8) | NOTTA | Fixed delta | Read |
|---|--:|--:|---|
| Sharpness / HF artifacts | +258% | +276% | slightly **worse** (adds HF) |
| Colorfulness (saturation) | +58.2% | +47.5% | marginally better |
| Contrast | +13.4% | +12.5% | tied |
| Temporal motion | −12.0% | +4.4% | marginally better |
| PSNR | −48.4% (→9.82) | −47.1% (→10.06) | +0.24 dB late (noise) |
| LPIPS | +200.9% | +197.5% | tied |

- **Takeaway:** a context-0 delta goes **stale** as the rollout leaves the trained
  distribution — exactly the predicted failure. This is a clean motivation slide,
  not a loss: it sets up the fix.

---

## Slide 8 — What's next

- **Immediate:** re-run the **native-geometry control** (13-cond/93-frame). The
  first attempt (job 15504259) hit the 12 h wall before writing its summary; a
  lighter budget (fewer videos/chunks) will finish and confirm the drift is
  *inherent*, not an artifact of our short-window re-conditioning.
- **The fix (EXP4 — streaming delta):** re-fit / update the steering vector
  **per chunk** instead of once, so it tracks the drifting distribution. The
  fixed-delta null above is the direct motivation.
- **Complementary (TANGO++):** a whiteness/spectral steering term targeting the
  rising HF (non-white) residual — the dominant, monotone drift mode here.
- **Ask of the deck:** drift = the enabling discovery; the streaming delta (and
  TANGO++) are the paper's contribution.
