# VBench++ Config Routing — Presentation Slides
**Date:** 2026-07-06  
**Audience:** PI / partner meeting (no prior chat context assumed)  
**Pilot:** 200 OOD-stratified Panda clips × 12 AdaSteer configs  
**Fixed AdaSteer baseline:** S10_LR5e-3 (headline deployable config @ 999v)

**Metric convention (used throughout):** Routing gains are reported vs **NOTTA** and vs **fixed AdaSteer**. Internal routing score = fraction of per-video config-oracle headroom recovered (oracle mean **+0.140** VBench total vs fixed S10 @ N=200). Absolute VBench deltas are always shown alongside.

**Our router (throughout this deck):** **Structured blocks A (+ optional B)** — **9-d video/caption** (`video_caption_only`, **20.8%**) or **21-d video/caption + diffusion-OOD** (`video_caption_ood`, **18.9%** when OOD pass allowed). Ridge OOF @ N=200 → **one** AdaSteer pass. No Tier-3 / probe / TTA eval metrics.

---

## Slide 1 — Title

**VBench++ Config Routing for AdaSteer**  
*Pick step×LR from the input video, then run AdaSteer once*

**Purpose (last week → this week):** N=200 OOD budget pilot (+0.140 config-oracle headroom). Screened structured deploy routers (Blocks A/B/C). **Headline:** **9-d video/caption stats → 20.8%** captured (**+0.029 vs fixed**); with **OOD allowed**, **A+B → 18.9%** (+0.027). **~2×** prior VAE-only router (9.7%). Internal >25% bar still not met.

---

## Slide 2 — OOD score (pilot stratification)

**What it is:** Per-video difficulty under the **frozen LongCat-Video base model** — how poorly the model predicts the flow-matching velocity on the video’s visible frames. **Higher score ⇒ model is more “surprised” ⇒ treated as more out-of-distribution (OOD).**

**Computation** (forward-only; no TTA adapters; same visible-window / cond–target split as AdaSteer TTA — `scripts/compute_diffusion_ood_score.py`):

1. Encode the visible pixel window to VAE latents \(x_0\); split into clean context latents + target latents (matching the TTA loop).
2. Sample noise \(\varepsilon\) and noise level \(\sigma = t/T\) with \(T{=}1000\), \(t \in \{100,500,900\}\).
3. Corrupt targets: \(x_t = (1-\sigma)\,x_0 + \sigma\,\varepsilon\).
4. Predict velocity with caption conditioning \(c\): \(\hat v = f_\theta(x_t, t, c)\).
5. Per-timestep loss (target tokens only):

\[
\mathcal{L}_t \;=\; \big\|\,\hat v - (\varepsilon - x_0)\,\big\|_2^2
\]

6. **OOD score** = mean over timesteps: \(\text{OOD} = \frac{1}{|T|}\sum_{t} \mathcal{L}_t\)  
   (column: `mean_diffusion_loss_caption`; unconditional variant also computed).

**Pilot use:** Rank all Panda clips by OOD; split into **5 quintiles**; sample **40 videos per quintile → N=200** stratified set (`scripts/sample_ood_quintile_videos.py`).

**Formulation sources:**
- **Flow-matching objective:** Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- **OOD interpretation:** Kingma et al., *Variational Diffusion Models*, NeurIPS 2021 — higher forward loss ⇒ lower estimated likelihood; used as a **video-only** difficulty feature (not a calibrated density).

---

## Slide 3 — Problem & baselines

**Question:** Can we pick AdaSteer (steps × LR) per video from the **input clip alone**, then run **one** adaptation pass, to improve VBench++ total?

We always compare against two deployable references:

| Baseline | Role |
|----------|------|
| **NOTTA** | Does test-time adaptation help at all? |
| **Fixed AdaSteer (S10/LR5e-3)** | Does *smart config choice* beat our best single recipe? |

**Population context @ 999v:** Fixed AdaSteer ≈ NOTTA on VBench total (**+0.001**, ~+0.13%).  
**Pilot @ N=200:** Per-video **config oracle** vs fixed S10 = **+0.140** mean headroom (12-config grid; Slide 4).

---

## Slide 4 — Config oracle (upper bound, not deployable)

**Setup:** For each of the 200 pilot videos we **already ran all 12 AdaSteer configs** in the budget grid:

| Steps | Learning rates |
|-------|----------------|
| S2, S5, S10, S20 | LR = 1e-3, 5e-3, 1e-2 |

**Config oracle** = per-video argmax of measured VBench total (7-dim mean):

\[
\text{oracle}(v) = \arg\max_{c \in \{12\ \text{configs}\}} \ \text{VBench}_\text{total}(v, c)
\]

**Headroom vs fixed S10:** **+0.140** mean on the pilot.

**Why not deployable:** Requires running (or perfectly predicting) **all 12 configs** at inference. Offline upper bound only.

**How we score our router:** Fraction of this gap recovered by a **single-run** policy — best **20.8%** (Block A) → **+0.029 VBench total vs fixed**; **18.9%** when OOD block is included (A+B).

---

## Slide 5 — Our router (structured blocks A + optional B)

### Deploy workflow

```
Input video (+ caption)  →  extract Block A (and optionally Block B)  →  x(v)  →  ridge  →  ONE AdaSteer
```

**Rules:** No AdaSteer before config choice. No probe runs. No Tier-3 LoRA. Block B = **one frozen base-DiT OOD pass** (Slide 2) — optional.

---

### Input — **x(v) = [A | B?]** (concatenated blocks)

| Block | # dims | Source | Required? |
|-------|--------|--------|-----------|
| **A** `video_caption` | **9** | `video_features.csv` — cuts, CLIP sim, DINO temporal, Laplacian, RGB entropy | **Yes (default)** |
| **B** `diffusion_ood` | **12** | `diffusion_ood_scores.csv` — frozen DiT flow loss @ t∈{100,500,900} + summaries | Optional (PI-approved) |

**Not used:** Tier-3, probe ΔPSNR/SSIM, NOTTA/AdaSteer eval outputs, bpp/FFT, 130-d VAE stack (tested — worse @ N=200).

---

### Results @ N=200 (5-fold OOF ridge)

| Config | Blocks | Captured % | Δ vs fixed | Match % |
|--------|--------|------------|------------|---------|
| **`video_caption_only`** | A | **20.8%** | **+0.029** | 18.5% |
| **`video_caption_ood`** | A+B | **18.9%** | **+0.027** | **21.0%** |
| `vae_inference_embedding` | C (130-d) | 9.7% | +0.014 | 16.5% |
| `diffusion_ood_only` | B | 4.9% | +0.007 | 18.0% |

**Deploy pick:** **A** if minimizing extra compute; **A+B** if OOD pass is acceptable (best oracle-config match).

---

### Training & inference

Same as before: 12 ridge models, argmax predicted VBench total, offline labels from pilot 12-config sweep only.

---

## Slide 6 — Main result (headline)

| Method | Δ vs **NOTTA** | Δ vs **fixed AdaSteer** | **% oracle headroom recovered** | 1× AdaSteer? |
|--------|----------------|-------------------------|--------------------------------|--------------|
| Fixed AdaSteer (S10) @ 999v | +0.001 (~+0.13%) | — | **0%** | Yes |
| **Our router — video/caption (Block A)** | **~+0.030** | **+0.029** | **20.8%** | **Yes** |
| **Our router — video/caption + OOD (A+B)** | **~+0.028** | **+0.027** | **18.9%** | **Yes** (+ DiT OOD pass) |
| **AdaState** (literature) | ~+0.026 (~+3.4%)† | N/A | — | Yes (different stack) |
| Config oracle (pilot) | ~+0.141 | +0.140 | **100%** | No (12 configs) |

†AdaState: vs their base generator, not our NOTTA. **% oracle headroom** = (method − fixed S10) / (config oracle − fixed S10); oracle mean gap = **+0.140** VBench total @ N=200.

**Takeaway:** **9-d video/caption stats** recover **~21%** of config-oracle headroom — **~2×** VAE-only (9.7%) and the old 51-d lab router (9%). Adding OOD (A+B) trades ~2pp captured for higher config match (21%). Still **below** AdaState and internal **>25%** bar, but the **strongest honest deploy result so far**.

---

## Slide 7 — Experimental setup

- **200 OOD-stratified videos** × **12 AdaSteer configs** (offline labels)
- **Router input:** Block A (9-d) ± Block B (12-d OOD)
- **Success bar (internal):** >25% — **not met**; best **20.8%** (A only)
- **Our router:** `video_caption_only` (best captured) · `video_caption_ood` (when OOD OK)

**Sample-size caveat:** 200 clips sufficient for honest OOF. Stacking 177-d VAE+Phase-0+probe **overfit** (4.2%). **Deploy-strict VAE-only (130-d) does not overfit** — 9.7% vs 9.0% for the heavier 51-d lab bundle.

---

## Slide 8 — Video-only routing variants (ranked)

All rows: **inference-path features only**, **one AdaSteer run** at deploy.

| Method | Oracle headroom recovered | Δ vs fixed |
|--------|---------------------------|------------|
| Fixed AdaSteer (reference) | 0% | — |
| **Our router — video/caption (A)** | **20.8%** | **+0.029** |
| **Our router — video/caption + OOD (A+B)** | **18.9%** | **+0.027** |
| VAE inference embedding (C) | 9.7% | +0.014 |
| Lab Phase-0 linear (51-d) | 9.0% | +0.013 |
| Phase-0 shallow MLP | 7–8% | +0.010–0.011 |
| Phase-0 kNN (exp6) | 1.2% | ~+0.002 |
| Pairwise top-4 classifiers | negative | hurts vs fixed |

**Non-oracle reference:** config oracle = **100%** / +0.140 vs fixed (needs all 12 configs — not our router class).

---

## Slide 9 — Complete experiment inventory

Everything screened @ N=200. Rows marked **†** required extra TTA passes or full grid — **not** our router; listed because we ran them and falsified them.

| Line | What we tried | Outcome |
|------|---------------|---------|
| **Our router — video/caption (A)** | 9-d ridge on cuts/CLIP/DINO | **Best: 20.8%, +0.029** |
| **Our router — A+B** | + diffusion-OOD | **18.9%, +0.027** (best match 21%) |
| VAE inference (C) | 130-d encode profile | 9.7% — superseded by A |
| A+B+C stacked | 151-d | **Overfit:** 10.1% |
| Phase-0 MLP / coarse grid | Nonlinear / binned | **Weak:** 7–8% |
| exp6 kNN (Phase-0 only) | Memory-based | **Fail:** 1.2% |
| Pairwise / best-of-3 NR | Proxy classifiers | **Fail:** negative |
| VAE + probe (`vae_profile_probe`) | VAE + AdaSteer probe metrics | 12.2% — **not deploy-strict** |
| VAE stacked w/ Phase-0+probe | 177-d | **Overfit:** 4.2% |
| Per-dim / tail / quintile gates | Video-only gating variants | **Fail:** ≤8% on total |
| Panda / UCF retrieval | Neighbour-batch TTA | **Null:** SIM≈RAND |
| 999v × 12 routing retrain | Scale video-only router | **NO-GO** |
| † Probe-and-route (exp7, exp16, …) | ΔPSNR/ΔSSIM from **prior probe TTA** | **13% but 3× TTA — rejected** |
| † GT / verifier on probe outputs | Oracle-ish judges on probe mp4s | **~18% ceiling — not video-only** |

---

## Slide 10 — AdaState comparison (honest)

| | Fixed AdaSteer | **Our router** | AdaState |
|--|----------------|----------------|----------|
| Mechanism | One config for all | VAE profile → pick config → **one** AdaSteer | Pathwise correction during generation |
| Δ VBench total (vs relevant base) | +0.001 vs NOTTA | **+0.029 vs fixed (A)** | **~+0.026 vs their base** |
| Inference cost | 1× AdaSteer | **1× AdaSteer** | 1× (different stack) |

**Say:** Our router adds a **real but modest** population lift over fixed AdaSteer using **only the VAE encode you already pay for** — no OOD/Tier-3/probe. Oracle (+0.140) shows headroom remains; closing it needs **better latent-side predictors**, not multi-pass probing.

**Do not say:** “We beat AdaState” or “routing matches AdaState today.”

---

## Slide 11 — Opportunity size

```
NOTTA ──► Fixed S10 ──► Our router (A or A+B) ──► Config oracle
         +0.001          +0.029 (A) / +0.027 (A+B)    +0.140 vs fixed
                         (21% / 19% of oracle gap)
```

- Our router vs NOTTA: **~+0.030 (A)** — meaningful lift over fixed S10; still **not** AdaState-scale
- Oracle vs fixed: **~5×** our best gain — routing problem real; cheap video-side features carry most signal
- Internal bar (>25% headroom): **not met**; **20.8%** is closest yet

---

## Slide 12 — Next steps (VAE inference router)

| Item | Status |
|------|--------|
| Learned head on VAE latents (small MLP / GBM, strict OOF) | Open |
| Scale routing labels 500–1K (same VAE-only constraint) | Not started |
| 999v × 12 routing retrain with VAE-only features | Not started |
| Probe-and-route / verifier-on-probe lines | **Deprioritized** — violate single-run constraint |
| Heavier Phase-0 battery (OOD/Tier-3) | **Deprioritized** — no deploy gain vs VAE-only @ N=200 |

---

## Slide 13 — Claims we can make today

1. **Fixed AdaSteer is flat vs NOTTA** (+0.13% @ 999v).
2. **Config oracle +0.140 vs fixed** — per-video budget choice matters in principle.
3. **Our video-only router: +0.013 vs fixed, ~+0.014 vs NOTTA** — one AdaSteer run, 5-fold OOF @ N=200.
4. **Below internal 25% headroom bar and below AdaState magnitude** — open problem, not a null routing hypothesis.
5. **Multi-pass probe routing screened and rejected** for the product story (inventory Slide 9).

---

## Slide 14 — Research pitch (3 sentences)

Per-video **config oracle headroom is +0.140 vs fixed**, so step×LR choice could matter. **Our router** picks the config from **video-only Phase-0 features**, then runs AdaSteer **once**, recovering **+0.013 vs fixed** today (9% of oracle gap). The next step is **stronger video-only prediction** — not running extra AdaSteer probes before adapting.

---

## Slide 15 — FAQ backup (optional)

**Q: Why two baselines?**  
NOTTA = “is TTA worth it?” Fixed S10 = “does *smart config choice* beat our best single recipe?” Our router wins on **both** (+0.014 vs NOTTA, +0.013 vs fixed).

**Q: What exactly is “our router”?**  
Phase-0 linear ridge: **51** video-only features → argmax predicted VBench over 12 configs → one AdaSteer run. **9%** oracle headroom, **5-fold OOF**.

**Q: Why N=200?**  
OOD-stratified pilot for honest OOF iteration; 999v×12 retrain explicitly **NO-GO** for video-only routing.

---

## Reference numbers (single source)

| Quantity | Value | Source |
|----------|-------|--------|
| NOTTA VBench total @ 999v | 0.772 | headline 1000v tables |
| Fixed AdaSteer @ 999v | 0.773 | +0.001 vs NOTTA |
| Oracle headroom vs fixed (pilot) | +0.140 | budget routing @ N=200 |
| **Our router — input dimension** | **51** | `join_feature_tables` @ 2026-07-06 pilot pipeline |
| **Our router — headroom recovered** | **9%** (9.0% exact) | `baseline_linear_total` |
| **Our router — Δ vs fixed** | **+0.013** | 0.09 × 0.140 |
| **Our router — Δ vs NOTTA** | **~+0.014** | +0.001 + 0.013 |
| AdaState VBench total | ~+3.4% rel (~+0.026 abs) | PROJECT_STATUS / paper |
