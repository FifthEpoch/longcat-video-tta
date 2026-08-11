# Presentation deck — "From measuring drift to a drift-gated test-time controller"

**Purpose:** This is the CURRENT full presentation narrative, slide-by-slide. It
supersedes `2026-08-08_deck_narrative_longhorizon_drift.md` (kept for audit trail)
by folding in that deck's measurement + negative-results content AND reframing it
around the **switch**: from parameter-space AdaSteer deltas (null) to a
**sampling-space, training-free, drift-gated GT-free test-time controller** with
two actuators (best-of-N search + anchored TTC correction).

**The one-sentence switch:** *We stopped trying to fix drift by nudging the
model's parameters (it can't — drift is exposure bias, an input-distribution
problem) and started fixing it in sampling space at test time, only on the chunks
that have actually drifted.*

**Provenance of numbers:**
- Drift measurement (Slides 1–6): `diag_longhorizon_drift.py`, NOTTA, reencode
  job **15497180** (N=24×8ch) and native runs (30 s N=12 / 60 s N=8, seed=42,
  50 steps, CFG=4.0). Figures: `../paper_figures/2026-08-08_longhorizon_drift/`.
- Negative catalogue (Slides 7–9): EXP-B fixed delta, EXP4 streaming delta,
  clean-anchored re-fit, routing/ramp gates — all logged in `../ANALYSIS_LOG.md`
  (2026-08-08 … 2026-08-10) and `../experiment_outputs/2026-08-{08,09,10}.md`.
- The switch + method (Slides 10–14): built 2026-08-10, commit `aa357e5`.
  best-of-N search sweep in flight (SLURM **15599852–15599859**, k=4 native 12ch
  N=8); anchored TTC actuator (`--method ttc|ttc_gated`) built + syntax-checked.

**Status (2026-08-10):** measurement is solid; the parameter-space intervention
line is a *closed, well-controlled negative*; the positive method is **built and
launched**, results pending. The deck is honest about what is proven (problem +
why the naive fix fails) vs. in-flight (does the controller beat NOTTA per-video).

---

# ACT I — THE PROBLEM IS REAL (measurement)

## Slide 1 — The hook

**"On the easy task, nothing we do matters. So we made the task hard — and the
model breaks in a way sampling-space correction is designed to fix."**

- Every intervention we tried on the short, in-domain, single-chunk 14→14
  continuation (AdaSteer delta, placement ablation, TANGO guidance) was ~null.
- Diagnosis (2026-08-06 problem-difficulty audit): LongCat-Video (13.6B, RLHF,
  continuation-pretrained) is *too strong* for that framing — headroom is small
  **by construction**, not because test-time adaptation can't help.
- The field finds its headroom in **long-horizon autoregressive rollout**, where
  error compounds chunk-over-chunk. So we moved there.

---

## Slide 2 — Setup: chained rollout, two metric families

- Identical per-chunk geometry to all prior runs; we **feed the model's own
  generated tail back** as the next chunk's conditioning and repeat. Nothing is
  trained — this is the **NOTTA baseline**.
- Quality per chunk, two families:
  - **GT-free (path-independent):** sharpness (Laplacian var), colorfulness
    (Hasler–Süsstrunk), contrast, temporal motion — defined for the *entire*
    rollout, even after the source clip's ground truth runs out.
  - **GT-referenced:** PSNR / SSIM / LPIPS, where the source clip still overlaps.
- **Why this matters later:** the GT-free family is exactly the signal our
  controller uses as a **verifier and a gate** — it needs no ground truth, so it
  is *deployable* at inference.

---

## Slide 3 — Drift is real and it COMPOUNDS with horizon (native protocol)

**The honest, corrected picture.** Naïve short-window (reencode 14/14) rollout
*overstates* drift (sharpness +258%, colorfulness +58%, PSNR −48%) — most of that
is a re-conditioning measurement artifact. Under LongCat's **native** window
(13-cond/80-gen) the model is far more robust at 30 s, **but drift grows
monotonically as we push to ~60 s** (the field-standard long horizon):

| GT-free signal (chunk 1 → last) | Native 30 s (6 ch, N=12) | **Native 60 s (12 ch, N=8)** | reading |
|---|--:|--:|---|
| Sharpness / HF artifacts | +28% | **+48%** | artifacts accumulate faster the longer you roll |
| Temporal motion | +8% | **+45%** | spurious motion / instability injected over time |
| Contrast | +3% | **−16%** | a fade sets in only at long horizon |
| Colorfulness (saturation) | +4% | +5.7% | mild — NOT the driver |

- Perceptual fidelity also decays natively (30 s): **LPIPS +96%, SSIM −40%,
  PSNR −21%**.
- **Drift mode = HF-artifact accumulation + motion instability + contrast fade.**
  It is *not* motion collapse (motion inflates) and *not* mainly over-saturation.
- **Horizon length is itself a lever:** the gap an intervention can open should
  **widen** at longer horizons → evaluate natively at ≥1 min, not 30 s.
- **Caveat:** native N=8–12 is a **gating** sample; GT metrics span a tiny window
  (source clips short), so judge long-horizon drift by the **GT-free** curves.

*(Figures: `drift_gtfree_normalized.png`, `drift_geometry_control_native_vs_reencode.png`.)*

---

# ACT II — WHY THE OBVIOUS FIX FAILED (the diagnosis, not just a catalogue)

## Slide 4 — A single AdaSteer delta cannot flatten drift — 4 axes, all null

We treated drift as something a learned steering vector (activation-space bias)
could cancel. It can't. **Four distinct delta recipes, all null/harmful under the
per-video paired test** (bootstrap CI + sign-flip permutation on |drift|):

| Delta recipe | Idea | Verdict (paired, native 60 s unless noted) |
|---|---|---|
| **Fixed** (EXP-B) | train once on chunk-0 context, hold fixed | curves parallel to NOTTA; goes stale (reencode geom) |
| **Streaming-generated** (EXP4) | re-fit each chunk on recent *generated* window | NULL; p ≥ 0.26; raises per-video volatility (mean-curve "flattening" was cancellation) |
| **Streaming-clean** | re-fit toward *clean* chunk-0 latents | NULL; p ≥ 0.53; fixes saturation but overshoots contrast fade |
| **Time-scheduled ramp** | more delta influence late | CONTRAINDICATED by chunk-interaction gate (no crossover; harm grows late) |

- **Per-video routing also ruled out:** heterogeneity looked routable (no-TTA best
  4/8; 23–39% oracle gap) but cross-signal consistency **p = 0.71** and the oracle
  gap **≤ the min-over-noise floor** → a **noise ceiling**, not signal.

---

## Slide 5 — The diagnosis: we were pulling the wrong lever

**Drift is exposure bias — an INPUT-distribution shift, not a weight defect.** The
model conditions each chunk on its own increasingly-degraded output, so the input
leaves the training manifold. A **global activation-space bias vector** shifts
population statistics but:

- has **no capacity** to reduce *per-video* drift (it trades one axis for another —
  saturation↓ but contrast-fade↑),
- **self-supervises on the model's own drifted output**, so it partly reproduces
  the drift it's meant to remove.

**Independently confirmed by the literature.** Pathwise Test-Time Correction (2026)
shows test-time *parameter* optimization collapses on this exact problem, and that
the fix lives in **sampling space / conditioning correction**. Our 4-axis null is
the same result, arrived at independently — this is a **credibility asset**, not a
loss.

> **Reframe:** the negative catalogue is not the paper's ending — it is the
> *ablation* that proves the parameter-space family is the wrong one, and points
> to the sampling-space family.

---

# ACT III — THE SWITCH

## Slide 6 — Parameter space → sampling space (what actually changed)

| | Parameter-space (what we abandoned) | Sampling-space (the switch) |
|---|---|---|
| **Where it acts** | model weights / activation bias | the latent trajectory & conditioning *during* denoising |
| **When** | before/instead of sampling | at test time, inside the sampler |
| **Training** | fits a delta (self-supervised on drifted output) | **none** — frozen model |
| **Granularity** | one global vector per video | per-chunk, per-step, per-candidate |
| **Matches the bug?** | no (bug is input-distribution) | **yes** (acts on the trajectory that carries the bad input) |

- **All sampling-space, all training-free, all test-time** — no fine-tuning, no
  extra data, deployable on the frozen released model.
- Grounded in hot 2025–26 work: **Video-T1** (ICCV'25), **MCTS-TTS** (ICLR'26),
  **Verifier Matters** (BMVC'25), **Pathwise TTC** (2026), **DFoT / History-Guided**
  (ICLR'25), **Rolling Forcing** (2025).

---

## Slide 7 — Novelty: it's the CONTROLLER, not the actuators

**Be honest (this protects us in review):**
- Plain **best-of-N** is *not* novel — TTC uses BoN (N=5) as a baseline; Video-T1
  builds on it.
- A straight **TTC re-implementation** is *not* novel either.

**Our novel contribution = a drift-GATED, GT-free test-time controller** that
decides, per-video / per-chunk:
1. **WHETHER to intervene** — a **GT-free drift gate** (does this chunk's incoming
   context deviate from the real-frame reference?), and
2. **HOW to intervene** — a pluggable **actuator**: best-of-N *search* or anchored
   *correction*.

The **GT-free drift verifier** is the piece that answers TTC's stated open problem
("reward design for error accumulation") without ground truth. Gating is the
*mechanism*, not just a diagnostic.

---

# ACT IV — THE METHOD & THE NEW FINDING

## Slide 8 — The controller

```
per chunk t:
  ctx_t         = model's own generated tail (conditioning)
  drift_t       = GT_free_deviation(ctx_t, real_frame_reference)   # no GT needed
  if drift_t <= gate_threshold:   pass through (NOTTA)             # healthy chunk
  else:                            apply actuator(ctx_t)           # drifted chunk
```

- **Gate signal = the same GT-free family from Slide 2** (sharpness / colorfulness /
  contrast / motion deviation from the initial *real* conditioning frames — the
  deployable anchor). Chunk 0 is real → deviation ≈ 0 → never corrected.
- Two actuators share this gate, so the controller is **actuator-agnostic**.

---

## Slide 9 — Actuator A: best-of-N test-time search (built + LAUNCHED)

- Per chunk, generate **k candidate continuations**; **candidate 0 reuses the NOTTA
  seed** ⇒ best-of-N is a **strict superset of NOTTA** (can only tie or win).
- A **GT-free drift verifier** scores each candidate (relative deviation of
  sharpness / colorfulness / contrast / motion from the real-frame reference +
  a seam-continuity penalty) and keeps the most stable one — so **a bad chunk never
  poisons the downstream context**.
- **Status:** `--method bestof` built; sweep **in flight** (SLURM 15599852–859,
  k=4, native 12ch, N=8, paired to `longhorizon_sweep_notta_native_12ch`). All
  candidates logged for a post-hoc oracle ceiling. Analyzer:
  `scripts/analyze_bestof_search.py`.

*(Result slide to fill when jobs land: verifier-pick vs random-pick vs oracle +
paired drift reduction vs NOTTA.)*

---

## Slide 10 — Actuator B: anchored TTC correction + the gate = the controller (built)

- **Sampling-space, frozen model:** during the **low-noise refinement band**
  (σ ≤ 0.3), re-anchor the sampled trajectory's *appearance* toward the clean
  first frame — directly counters exposure-bias appearance drift on the trajectory
  that carries the degraded input.
- **`--method ttc`** = ungated baseline; **`--method ttc_gated`** = **the
  controller**: apply the correction **only on chunks whose incoming context has
  drifted past a GT-free threshold**, else pass through. Same GT-free signal as the
  search verifier — one gate, two actuators.
- Runs on the repo's self-contained `SAViDNO_LongCat` engine + `TTC_LongCat`
  sampler (the shipped pipeline exposes no per-step handles). **Engineering fix:**
  decode the full `[cond|gen]` latent stack **jointly** so frame geometry matches
  the pipeline (13 cond + 80 gen = 93 frames); decoding gen-only would drop the
  shared VAE boundary frame and corrupt chaining.
- **Status:** built + syntax-checked (commit `aa357e5`). Clean paired baseline =
  `ttc --ttc-weight 0` on the **same engine** (both reencode-style conditioning),
  so the paired test isolates the correction effect.

---

# ACT V — RIGOR & ROADMAP

## Slide 11 — We do not fool ourselves: three statistical gates

The negatives taught us that "gains" here are often **max-over-noise artifacts**.
Every positive claim must clear:

1. **Random-pick baseline** — the verifier's selection must beat picking a random
   candidate. If verifier ≈ random, the "gain" is just best-of-noise (the same trap
   that killed the PSNR router). `analyze_bestof_search.py` reports
   verifier-pick vs random-pick vs oracle; the **oracle−random gap is the noise
   floor**.
2. **Oracle-over-candidates ceiling** — how much of the achievable (by-metric)
   improvement the GT-free verifier actually captures ⇒ tells us whether to tune
   the verifier or raise k.
3. **Paired per-video sign-flip test** (`compare_drift_paired.py`) — headline drift
   reduction vs NOTTA must survive the same bootstrap-CI + permutation test that the
   deltas *failed*. Same bar for the positive as for the negatives.

---

## Slide 12 — Status & what's running

- **In flight:** best-of-N k=4 native 12ch N=8 (jobs 15599852–859) → gate analysis.
- **Built, next to launch:** `ttc` weight sweep {0, 0.05, 0.1, 0.2} + `ttc_gated`,
  paired to `ttc-w0`.
- **Then:** whichever actuator clears the three gates → scale N, add the
  matched-horizon plot, and the deck's headline becomes a **positive method**.

---

## Slide 13 — The honest headline (how to pitch it in the room)

**"Long-horizon drift in LongCat is real and compounds with horizon. It is
exposure bias, so parameter-space adaptation can't fix it — we proved this across
four delta recipes and a routing ablation, matching an independent 2026 finding.
The fix is a training-free, GT-free, drift-gated test-time controller that
intervenes in sampling space only on the chunks that have drifted — with best-of-N
search and anchored correction as interchangeable actuators. Method is built and
launched; results are gated by the same paired test the negatives failed."**

- **If the controller wins:** positive method paper (measurement + diagnosis +
  method).
- **If it ties:** still a strong **measurement + why-the-obvious-fixes-fail** paper,
  now with sampling-space evidence too — a complete, honest story either way.

---

## Appendix A — Per-chunk NOTTA data (reencode, N=24)

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

**Verdict (chunk 1→8, reencode):** sharpness +258%, colorfulness +58%, contrast
+13% (monotone); motion −12% (flat); PSNR −48%, SSIM −56%, LPIPS +201%. *(Inflated
by the reencode protocol — see Slide 3 for the corrected native numbers.)*

## Appendix B — Slide → source-of-truth map

| Slides | Content | Source |
|---|---|---|
| 1–3, App A | drift measurement | `2026-08-08_deck_narrative_longhorizon_drift.md`, job 15497180 + native runs |
| 4–5 | delta nulls + diagnosis | `ANALYSIS_LOG.md` 2026-08-08…08-10; `compare_drift_paired.py` |
| 6–7 | the switch + novelty | `ANALYSIS_LOG.md` 2026-08-10 (pivot + controller framing) |
| 8–10 | controller + actuators | `diag_longhorizon_drift.py`, `ttc_longcat.py`, commit `aa357e5` |
| 9, 11 | search + gates | `scripts/analyze_bestof_search.py`; jobs 15599852–859 |
