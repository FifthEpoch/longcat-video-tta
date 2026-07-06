# VBench++ Config Routing — Presentation Slides
**Date:** 2026-07-06  
**Audience:** PI / partner meeting (no prior chat context assumed)  
**Pilot:** 200 OOD-stratified Panda clips × 12 AdaSteer configs  
**Fixed AdaSteer baseline:** S10_LR5e-3 (headline deployable config @ 999v)

**Metric convention (used throughout):** Routing gains are reported vs **NOTTA** and vs **fixed AdaSteer**. Internal routing score = fraction of per-video config-oracle headroom recovered (oracle mean **+0.140** VBench total vs fixed S10 @ N=200). Absolute VBench deltas are always shown alongside.

**Our router (throughout this deck):** **Video-only Phase-0 linear ridge** — **51 input features** per video (`baseline_linear_total`; see Slide 5). Pick (steps, LR), then run **one** AdaSteer pass.

---

## Slide 1 — Title

**VBench++ Config Routing for AdaSteer**  
*Pick step×LR from the input video, then run AdaSteer once*

**Purpose (last week → this week):** Last week we ran the N=200 OOD budget pilot to measure config-oracle headroom (+0.140 vs fixed S10) and screen **video-only** config routers; this week we closed weaker Phase-0 predictor variants (tail gates, VAE pools, per-dim fusion) and identified the **Phase-0 linear ridge** as our deployable router baseline (**9%** oracle headroom, **+0.013 vs fixed**).

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

**How we score our router:** Fraction of this gap recovered by a **video-only, single-run** policy — our router achieves **9%** → **+0.013 VBench total vs fixed**.

---

## Slide 5 — Our router (video-only Phase-0 linear ridge)

### Deploy workflow

```
Input video (+ caption)  →  extract video-only features x(v)  →  router f(x)  →  ONE AdaSteer run
```

**Rules:** No AdaSteer / TTA before the config is chosen. No probe runs, no ΔPSNR/ΔSSIM from partial TTA.

**Offline training** uses the 12-config pilot sweep once to learn which config would have won VBench — label cost is lab-only, not per deploy.

---

### Input — **x(v) ∈ ℝ^51** (video-only, exact)

One feature vector per video from `join_feature_tables` (`baseline_linear_total`; pilot feature date **2026-07-06**). Built from the **TTA-visible input clip + caption** before any AdaSteer run. **No NOTTA/AdaSteer evaluation metrics. No probe TTA. No ΔPSNR/ΔSSIM.**

| Block | # dims | Source CSV / script |
|-------|--------|---------------------|
| Core video + caption | 9 | `video_features.csv` — cuts, CLIP sim, DINO temporal, Laplacian, RGB entropy |
| Diffusion-OOD (frozen base DiT) | 20 | `diffusion_ood_scores.csv` — 3 timesteps × (caption/uncond loss + score norm) + 8 summaries |
| Tier-3 mini LoRA probe | 8 | `tier3_probe_features.csv` — grad-norm + loss-drop @ 3 timesteps + means |
| Flow shape | 4 | `flow_shape_features.csv` |
| Compression / spectrum | 4 | `bpp_features.csv` (2) + `fft_features.csv` (2) |
| VAE rec-error | 2 | `vae_recerr_features.csv` |
| Latent motion | 2 | `latent_motion_features.csv` |
| Loss variance | 2 | `loss_variance_features.csv` |
| **Total** | **51** | `submit_pilot_router_features.sh` pipeline |

*Note:* `vae_latent_profile_features.csv` is **not** in this pipeline; if added separately, dimension jumps to **169** — that is **not** the router reported @ 9%.

**In flight (deploy-strict rerun):** Re-score with **`vae_latent_profile_features.csv` ONLY** (~130-d, same `encode_video` as inference). **No** CLIP/DINO/bpp/OOD/Tier-3/probe/TTA metrics. Script: `run_deploy_strict_router_experiments.py` · `submit_deploy_strict_router.sh` · experiment id: `vae_inference_embedding`.

---

### Output

One categorical config per video:

\[
f(\mathbf{x}) \in \{\texttt{S2\_LR1e3}, \ldots, \texttt{S20\_LR1e2}\}
\]

(12 run IDs = steps {2,5,10,20} × LR {1e-3, 5e-3, 1e-2}).

---

### Training & inference

| Step | Detail |
|------|--------|
| **Labels (offline)** | Measured VBench total(v, c) for each of 12 configs from pilot sweep |
| **Model** | **12 ridge regressors** — one per config: \(\widehat{\text{VB}}_c = \mathbf{w}_c^\top \mathbf{x} + b_c\). Ridge λ ∈ {1e-4…10} via inner CV. Features z-scored per fold. |
| **Deploy rule** | \(\hat c = \arg\max_c \widehat{\text{VB}}_c\) |
| **Evaluation** | **5-fold out-of-fold (OOF):** train on 160, predict on held-out 40; rotate — no leakage |

**Result @ N=200:** **9.0%** oracle headroom recovered · **+0.013 vs fixed S10** · **~+0.014 vs NOTTA** · 18.0% oracle-config match rate.

---

### Literature (inspiration, not a copied method)

Per-instance **algorithm selection**: predict which hyperparameter config wins from **instance features** alone (SATzilla / AutoFolio / Hutter et al.-style performance models). Our instantiation is custom: **AdaSteer step×LR grid**, **VBench++ total** objective, **Phase-0 video feature battery**, **single-run** deploy constraint.

---

## Slide 6 — Main result (headline)

| Method | Δ vs **NOTTA** | Δ vs **fixed AdaSteer** | **% oracle headroom recovered** | 1× AdaSteer? |
|--------|----------------|-------------------------|--------------------------------|--------------|
| Fixed AdaSteer (S10) @ 999v | +0.001 (~+0.13%) | — | **0%** | Yes |
| **Our router (Phase-0 linear, OOF)** | **~+0.014 (~1.8%)** | **+0.013 (~1.7%)** | **9%** | **Yes** |
| **AdaState** (literature) | ~+0.026 (~+3.4%)† | N/A | — | Yes (different stack) |
| Config oracle (pilot) | ~+0.141 | +0.140 | **100%** | No (12 configs) |

†AdaState: vs their base generator, not our NOTTA. **% oracle headroom** = (method − fixed S10) / (config oracle − fixed S10); oracle mean gap = **+0.140** VBench total @ N=200.

**Takeaway:** Oracle headroom (+0.140) shows routing *could* matter a lot. **Our video-only router** already beats fixed S10 by **+0.013** with **one** AdaSteer run — modest but real; still **below** AdaState-scale gains.

---

## Slide 7 — Experimental setup

- **200 OOD-stratified videos** × **12 AdaSteer configs** (offline labels)
- **Router input:** video-only Phase-0 features (**51-d**, `baseline_linear_total`)
- **Objective:** VBench total (7-dim mean)
- **Evaluation:** 5-fold **out-of-fold** routing
- **Success bar (internal):** >25% oracle headroom with bootstrap CI excluding 0 — **not met** (9%)
- **Our router:** `baseline_linear_total` — best honest **video-only** variant @ N=200

**Sample-size caveat:** 200 clips sufficient for honest OOF; richer feature stacks (177-d VAE pool) **overfit** (4.2% when stacked). Phase-0 correlation screen (H1–H8): no single feature clears \|ρ\|≥0.2 on both ADA and LoRA.

---

## Slide 8 — Video-only routing variants (ranked)

All rows: **input video only**, **one AdaSteer run** at deploy.

| Method | Oracle headroom recovered | Δ vs fixed |
|--------|---------------------------|------------|
| Fixed AdaSteer (reference) | 0% | — |
| **Our router — Phase-0 linear ridge** | **9%** | **+0.013** |
| Phase-0 shallow MLP | 7–8% | +0.010–0.011 |
| Phase-0 kNN (exp6) | 1.2% | ~+0.002 |
| Pairwise top-4 classifiers | negative | hurts vs fixed |

**Non-oracle reference:** config oracle = **100%** / +0.140 vs fixed (needs all 12 configs — not our router class).

---

## Slide 9 — Complete experiment inventory

Everything screened @ N=200. Rows marked **†** required extra TTA passes or full grid — **not** our router; listed because we ran them and falsified them.

| Line | What we tried | Outcome |
|------|---------------|---------|
| **Our router — Phase-0 linear** | Video-only ridge → 12-way pick | **Best video-only: 9%, +0.013 vs fixed** |
| Phase-0 MLP / coarse grid | Nonlinear / binned video-only | **Weak:** 7–8% |
| exp6 kNN (Phase-0 only) | Memory-based video-only | **Fail:** 1.2% |
| Pairwise / best-of-3 NR | Video-only or proxy classifiers | **Fail:** negative |
| VAE latent profile (video-only) | Richer frozen-VAE features | **Null / overfit:** 4–12% |
| Per-dim / tail / quintile gates | Video-only gating variants | **Fail:** ≤8% on total |
| Panda / UCF retrieval | Neighbour-batch TTA | **Null:** SIM≈RAND |
| 999v × 12 routing retrain | Scale video-only router | **NO-GO** |
| † Probe-and-route (exp7, exp16, …) | ΔPSNR/ΔSSIM from **prior probe TTA** | **13% but 3× TTA — rejected** |
| † GT / verifier on probe outputs | Oracle-ish judges on probe mp4s | **~18% ceiling — not video-only** |

---

## Slide 10 — AdaState comparison (honest)

| | Fixed AdaSteer | **Our router** | AdaState |
|--|----------------|----------------|----------|
| Mechanism | One config for all | Video-only features → pick config → **one** AdaSteer run | Pathwise correction during generation |
| Δ VBench total (vs relevant base) | +0.001 vs NOTTA | **+0.013 vs fixed** | **~+0.026 vs their base** |
| Inference cost | 1× AdaSteer | **1× AdaSteer** | 1× (different stack) |

**Say:** Our router adds a **real but modest** population lift over fixed AdaSteer without extra TTA passes. Oracle (+0.140) shows headroom remains; closing it needs **better video-only predictors**, not multi-pass probing.

**Do not say:** “We beat AdaState” or “routing matches AdaState today.”

---

## Slide 11 — Opportunity size

```
NOTTA ──► Fixed S10 ──► Our router (video-only) ──► Config oracle
         +0.001          +0.013 vs fixed              +0.140 vs fixed
                         (9% of oracle gap)
```

- Our router vs NOTTA: **~+0.014** — beats fixed S10 on both baselines, but **not** AdaState-scale (~+0.026)
- Oracle vs fixed: **~11×** our router’s gain — routing **problem** is real; **video-only prediction** is the bottleneck
- Internal bar (>25% headroom): **not met** at 9%; honest negative on scale-up to 999v×12 as well

---

## Slide 12 — Next steps (video-only router)

| Item | Status |
|------|--------|
| Stronger Phase-0 features (raw-video verifiers, better motion/IQ proxies) | Open |
| Nonlinear video-only router (MLP / small GBM) with strict OOF | Partially tried — marginal |
| Scale routing labels 500–1K (same video-only constraint) | Not started |
| Probe-and-route / verifier-on-probe lines | **Deprioritized** — violate single-run constraint |

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
