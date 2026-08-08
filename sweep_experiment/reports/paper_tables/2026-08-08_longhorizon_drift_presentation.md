# Long-horizon drift finding — presentation pack (2026-08-08)

**Headline:** On our short in-domain 14→14 continuation, every intervention was
null because LongCat is already near-saturated. But when we **chain the model on
its own output** (true autoregressive rollout), LongCat degrades **strongly and
monotonically** — opening real headroom for test-time correction.

**Source run:** `diag_longhorizon_drift.py`, SLURM job **15497180** — NOTTA,
N=24 videos × 8 chunks, reencode geometry (cond=14 / frames=28 / gen_start=48,
50 steps, CFG=4.0, seed=42). Figures reproducible via
`scripts/make_drift_presentation_figs.py`.

---

## Talking points (bullets)

- **Setup.** Same per-chunk geometry as all our AdaSteer/placement/TANGO runs;
  we simply feed the model's own generated tail back as the next chunk's
  conditioning and repeat for 8 chunks. Nothing is trained (NOTTA baseline).
- **Finding.** Quality degrades monotonically with chunk index — short-horizon
  saturation does **not** survive chaining. This is the headroom the field
  reports (Rolling Forcing, Pathwise TTC, Self-Forcing) and that our single-chunk
  task was hiding by construction.
- **Drift mode is specific, not generic collapse.** Using **GT-free**
  (path-independent) signals defined for the entire rollout:
  - Colorfulness **+58%** (monotone) → progressive **over-saturation**.
  - Laplacian-variance sharpness **+258%** (monotone) → **high-frequency artifact
    accumulation** (a rising, non-white residual).
  - Contrast **+13%** (monotone).
  - Temporal motion **−12%, non-monotone/flat** → motion-collapse is **NOT** the
    mode here (distinguishes us from motion-drift papers).
- **GT-referenced fidelity collapses too.** PSNR **19.0 → 9.8 dB (−48%,
  −1.12 dB/chunk)**, SSIM 0.71 → 0.31, LPIPS **+201%** — front-loaded (PSNR is
  already 12.5 dB by chunk 3). Lead with the GT-free signals since PSNR partly
  reflects legitimate divergence from a single GT path (continuation is
  multimodal), and GT coverage shrinks (n = 24 → 13) as the source clip runs out.
- **Why it matters.** The rising HF/over-saturation residual is exactly the
  signature a whiteness/spectral steering term (TANGO++) or a re-anchoring
  correction is designed to flatten — so the drift regime is where our
  interventions can finally show an effect.
- **In flight (controls, do not over-claim yet).** Two jobs running to harden the
  claim: (1) a **native-geometry control** (13-cond/93-frame window) to prove the
  drift is inherent and not a short-window re-conditioning artifact, and (2) an
  **intervention test** where a fixed AdaSteer delta is held across the rollout
  (does it flatten the curve, or decay — motivating a streaming per-chunk delta?).

---

## Per-chunk data (job 15497180, mean over N=24)

| Chunk | Sharpness (Lap-var) | Temporal motion | Colorfulness | Contrast | PSNR (dB) | SSIM | LPIPS | GT n |
|------:|--------------------:|----------------:|-------------:|---------:|----------:|-----:|------:|-----:|
| 1 | 0.0070 | 0.0232 | 0.1488 | 0.2358 | 19.02 | 0.710 | 0.248 | 24 |
| 2 | 0.0088 | 0.0218 | 0.1556 | 0.2402 | 14.87 | 0.587 | 0.409 | 24 |
| 3 | 0.0108 | 0.0238 | 0.1624 | 0.2465 | 12.51 | 0.505 | 0.520 | 19 |
| 4 | 0.0138 | 0.0210 | 0.1693 | 0.2515 | 11.65 | 0.432 | 0.597 | 18 |
| 5 | 0.0168 | 0.0228 | 0.1762 | 0.2563 | 11.11 | 0.390 | 0.638 | 16 |
| 6 | 0.0200 | 0.0211 | 0.1894 | 0.2583 | 10.58 | 0.351 | 0.691 | 15 |
| 7 | 0.0223 | 0.0202 | 0.2149 | 0.2629 | 10.16 | 0.321 | 0.715 | 14 |
| 8 | 0.0251 | 0.0204 | 0.2354 | 0.2674 |  9.82 | 0.311 | 0.746 | 13 |

## Drift verdict (chunk 1 → chunk 8)

| Signal | Type | Chunk 1 | Chunk 8 | % change | Slope/chunk | Monotone? |
|---|---|--:|--:|--:|--:|:--:|
| Sharpness (HF artifacts) | GT-free | 0.0070 | 0.0251 | **+258%** | +0.00268 | yes |
| Colorfulness (saturation) | GT-free | 0.1488 | 0.2354 | **+58%** | +0.01179 | yes |
| Contrast | GT-free | 0.2358 | 0.2674 | **+13%** | +0.00447 | yes |
| Temporal motion | GT-free | 0.0232 | 0.0204 | −12% | −0.00040 | no (flat) |
| PSNR | GT-ref | 19.02 | 9.82 | **−48%** | −1.12 dB | yes |
| SSIM | GT-ref | 0.710 | 0.311 | −56% | — | yes |
| LPIPS | GT-ref | 0.248 | 0.746 | **+201%** | +0.066 | yes |

---

## Figures (300 dpi, `paper_figures/2026-08-08_longhorizon_drift/`)

1. **Headline** — GT-free signals as % of chunk 1:
   `drift_gtfree_normalized.png`
2. **Raw units** — 2×2 GT-free panels (motion-collapse ruled out):
   `drift_gtfree_raw.png`
3. **Fidelity collapse** — PSNR + LPIPS with GT coverage:
   `drift_gt_fidelity.png`
