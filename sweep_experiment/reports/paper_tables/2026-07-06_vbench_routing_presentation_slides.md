# VBench++ Config Routing — Presentation Slides
**Date:** 2026-07-06  
**Audience:** PI / partner meeting (no prior chat context assumed)  
**Pilot:** 200 OOD-stratified Panda clips × 12 AdaSteer configs  
**Fixed AdaSteer baseline:** S10_LR5e-3 (headline deployable config @ 999v)

**Metric convention (used throughout):** Routing gains are reported vs **NOTTA** and vs **fixed AdaSteer**. Internal routing score = fraction of per-video config-oracle headroom recovered (oracle mean **+0.140** VBench total vs fixed S10 @ N=200). Absolute VBench deltas are always shown alongside.

**Our router (throughout this deck):** **VAE inference embedding** — **130 input features** from `vae_latent_profile_features.csv` only (`vae_inference_embedding`; see Slide 5). Reuses LongCat `encode_video` on the input clip → ridge pick config → **one** AdaSteer pass. **No** CLIP/OOD/Tier-3/probe/TTA-side metrics.

---

## Slide 1 — Title

**VBench++ Config Routing for AdaSteer**  
*Pick step×LR from the input video, then run AdaSteer once*

**Purpose (last week → this week):** Last week we ran the N=200 OOD budget pilot (+0.140 config-oracle headroom vs fixed S10) and screened video-side routers; this week we tightened the deploy bar to **VAE inference embedding only** (130-d, same `encode_video` as LongCat). **Headline result:** **9.7%** oracle headroom recovered, **+0.0136 vs fixed S10** — matches the heavier 51-d lab router without OOD/Tier-3/probe inputs.

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

**How we score our router:** Fraction of this gap recovered by a **single-run** policy from **inference-path VAE features only** — our router achieves **9.7%** → **+0.0136 VBench total vs fixed**.

---

## Slide 5 — Our router (VAE inference embedding)

### Deploy workflow

```
Input video  →  encode_video (LongCat VAE)  →  latent profile x(v)  →  router f(x)  →  ONE AdaSteer run
```

**Rules:** No AdaSteer / TTA before the config is chosen. No probe runs. No DiT OOD forwards. No Tier-3 LoRA probes. **Only** pooled statistics of the VAE latent tensor you already compute for inference.

**Offline training** uses the 12-config pilot sweep once to fit ridge weights — label cost is lab-only, not per deploy.

---

### Input — **x(v) ∈ ℝ^130** (VAE inference path only)

One feature vector per video from `vae_latent_profile_features.csv` (`extract_vae_latent_profile_features.py`; pilot date **2026-07-06**). Built from **TTA-visible pixels [0:48)** via the same `encode_video` path as AdaSteer. Pooled full / context / target latent regions (~130 scalars).

| Block | # dims | Source |
|-------|--------|--------|
| LongCat-VAE latent profile | **130** | `vae_latent_profile_features.csv` — ctx/tgt/full channel + token-norm + temporal-delta pools |

**Not used:** `video_features.csv`, OOD, Tier-3, bpp/FFT, motion, probe metrics, NOTTA/AdaSteer eval outputs.

*Lab ablation (superseded for deploy):* 51-d Phase-0 bundle (`baseline_linear_total`) reached **9.0%** / +0.013 but required OOD DiT + Tier-3 LoRA — **not** inference-only.

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

**Result @ N=200:** **9.7%** oracle headroom recovered · **+0.0136 vs fixed S10** · **~+0.014 vs NOTTA** · 16.5% oracle-config match rate.

---

### Literature (inspiration, not a copied method)

Per-instance **algorithm selection**: predict which hyperparameter config wins from **instance features** alone (SATzilla / AutoFolio / Hutter et al.-style performance models). Our instantiation: **AdaSteer step×LR grid**, **VBench++ total** objective, **inference VAE latent profile**, **single AdaSteer** deploy constraint.

---

## Slide 6 — Main result (headline)

| Method | Δ vs **NOTTA** | Δ vs **fixed AdaSteer** | **% oracle headroom recovered** | 1× AdaSteer? |
|--------|----------------|-------------------------|--------------------------------|--------------|
| Fixed AdaSteer (S10) @ 999v | +0.001 (~+0.13%) | — | **0%** | Yes |
| **Our router (VAE inference embedding, OOF)** | **~+0.014 (~1.8%)** | **+0.0136 (~1.7%)** | **9.7%** | **Yes** |
| **AdaState** (literature) | ~+0.026 (~+3.4%)† | N/A | — | Yes (different stack) |
| Config oracle (pilot) | ~+0.141 | +0.140 | **100%** | No (12 configs) |

†AdaState: vs their base generator, not our NOTTA. **% oracle headroom** = (method − fixed S10) / (config oracle − fixed S10); oracle mean gap = **+0.140** VBench total @ N=200.

**Takeaway:** Oracle headroom (+0.140) shows routing *could* matter a lot. **Our VAE-only router** beats fixed S10 by **+0.0136** with **one** AdaSteer run and **no extra inference beyond VAE encode** — modest but real; still **below** AdaState-scale gains and internal **>25%** bar.

---

## Slide 7 — Experimental setup

- **200 OOD-stratified videos** × **12 AdaSteer configs** (offline labels)
- **Router input:** LongCat-VAE latent profile only (**130-d**, `vae_inference_embedding`)
- **Objective:** VBench total (7-dim mean)
- **Evaluation:** 5-fold **out-of-fold** routing
- **Success bar (internal):** >25% oracle headroom with bootstrap CI excluding 0 — **not met** (9.7%)
- **Our router:** `vae_inference_embedding` — best **inference-path-only** variant @ N=200

**Sample-size caveat:** 200 clips sufficient for honest OOF. Stacking 177-d VAE+Phase-0+probe **overfit** (4.2%). **Deploy-strict VAE-only (130-d) does not overfit** — 9.7% vs 9.0% for the heavier 51-d lab bundle.

---

## Slide 8 — Video-only routing variants (ranked)

All rows: **inference-path features only**, **one AdaSteer run** at deploy.

| Method | Oracle headroom recovered | Δ vs fixed |
|--------|---------------------------|------------|
| Fixed AdaSteer (reference) | 0% | — |
| **Our router — VAE inference embedding** | **9.7%** | **+0.0136** |
| Lab Phase-0 linear (51-d, OOD+Tier-3) | 9.0% | +0.013 |
| Phase-0 shallow MLP | 7–8% | +0.010–0.011 |
| Phase-0 kNN (exp6) | 1.2% | ~+0.002 |
| Pairwise top-4 classifiers | negative | hurts vs fixed |

**Non-oracle reference:** config oracle = **100%** / +0.140 vs fixed (needs all 12 configs — not our router class).

---

## Slide 9 — Complete experiment inventory

Everything screened @ N=200. Rows marked **†** required extra TTA passes or full grid — **not** our router; listed because we ran them and falsified them.

| Line | What we tried | Outcome |
|------|---------------|---------|
| **Our router — VAE inference embedding** | VAE latent profile → 12-way ridge | **Best deploy: 9.7%, +0.0136 vs fixed** |
| Lab Phase-0 linear (51-d) | OOD DiT + Tier-3 + aux | 9.0%, +0.013 — **not inference-only** |
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
| Δ VBench total (vs relevant base) | +0.001 vs NOTTA | **+0.0136 vs fixed** | **~+0.026 vs their base** |
| Inference cost | 1× AdaSteer | **1× AdaSteer** | 1× (different stack) |

**Say:** Our router adds a **real but modest** population lift over fixed AdaSteer using **only the VAE encode you already pay for** — no OOD/Tier-3/probe. Oracle (+0.140) shows headroom remains; closing it needs **better latent-side predictors**, not multi-pass probing.

**Do not say:** “We beat AdaState” or “routing matches AdaState today.”

---

## Slide 11 — Opportunity size

```
NOTTA ──► Fixed S10 ──► Our router (VAE-only) ──► Config oracle
         +0.001          +0.0136 vs fixed              +0.140 vs fixed
                         (9.7% of oracle gap)
```

- Our router vs NOTTA: **~+0.014** — beats fixed S10 on both baselines, but **not** AdaState-scale (~+0.026)
- Oracle vs fixed: **~10×** our router’s gain — routing **problem** is real; **latent-side prediction** is the bottleneck
- Internal bar (>25% headroom): **not met** at 9.7%; honest negative on scale-up to 999v×12 as well

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
