# VBench++ Config Routing — Presentation Slides
**Date:** 2026-07-07  
**Audience:** PI / partner meeting (no prior chat context assumed)  
**Pilot:** 200 OOD-stratified Panda clips × 12 AdaSteer configs  
**Fixed AdaSteer baseline:** S10_LR5e-3 (headline deployable config @ 999v)

**Experiment:** `deploy_strict_router/` — structured feature blocks **A / B / C**, ridge OOF @ N=200. **No** Tier-3, probe, or TTA eval metrics as inputs.

**Metric convention:** Δ columns show **absolute (+ relative % vs that baseline)**. Denominators @ 999v Panda standard: **NOTTA = 0.772**, **fixed S10 = 0.773**. Router Δ from N=200 OOF. **% oracle headroom recovered** = (policy − fixed) / (oracle − fixed); oracle gap = **+0.140**.

**Our router (headline):** **Block A — `video_caption_only`** → **20.8%** oracle headroom (Slide 6). Same model family also tested on **Block C** (VAE pooled profile @ 9.7%; Slide 5).

---

## Slide 1 — Title

**VBench++ Config Routing for AdaSteer**  
*Pick step×LR from pre-adaptation video features, then run AdaSteer once*

**Headline @ N=200:** Block A (`video_caption_only`) → **20.8%** oracle headroom · **+0.029 (+3.8%) vs fixed S10** · **+0.030 (+3.9%) vs NOTTA**.

---

## Slide 2 — Problem & baselines

**Question:** Can we pick AdaSteer (steps × LR) per video from **pre-adaptation features**, then run **one** adaptation pass?

| Baseline | Role |
|----------|------|
| **NOTTA** | Does TTA help at all? |
| **Fixed AdaSteer (S10/LR5e-3)** | Does *smart config choice* beat our best single recipe? |

**@ 999v:** Fixed AdaSteer vs NOTTA = **+0.001 (+0.13%)**.  
**Pilot @ N=200:** Config oracle vs fixed S10 = **+0.140 (+18.1%)** (Slide 3).

---

## Slide 3 — Config oracle (upper bound, not deployable)

**Grid:** S2/S5/S10/S20 × LR {1e-3, 5e-3, 1e-2} → **12 configs** (offline labels on pilot).

\[
\text{oracle}(v) = \arg\max_{c \in \{12\}} \ \text{VBench}_\text{total}(v, c)
\]

**Headroom vs fixed S10:** **+0.140 (+18.1%)** · requires all 12 configs at inference.

---

## Slide 4 — Structured feature blocks (experiment design)

We scored **three deploy-side blocks** in isolation and combination (same ridge pipeline, 5-fold OOF):

| Block | Name | Dims | Source |
|-------|------|-----:|--------|
| **A** | `video_caption` | 9 | `video_features.csv` — cuts, CLIP, DINO, Laplacian, RGB |
| **B** | `diffusion_ood` | 12 | `diffusion_ood_scores.csv` — frozen DiT @ t∈{100,500,900} |
| **C** | `vae_inference` | 130 | `vae_latent_profile_features.csv` — LongCat `encode_video` pools |

**Full ablation results** on Slide 7. **Headline router = Block A alone** (best captured %).

---

## Slide 5 — Router model & two input variants

Both experiments use the **same router architecture** (`run_budget_config_task`); only the **feature vector x(v)** differs.

---

### A. Shared router model (both runs)

**Task:** pick one of **12 AdaSteer configs** (S2/S5/S10/S20 × 3 LRs) per video to maximize **VBench total** after a single adaptation.

| Component | Specification |
|-----------|----------------|
| **Model** | **12 independent ridge regressors** — one per config \(c\) |
| **Per-config predictor** | \(\widehat{\text{VB}}_c(v) = b_c + \mathbf{w}_c^\top \mathbf{x}(v)\) |
| **Deploy rule** | \(\hat c(v) = \arg\max_c \widehat{\text{VB}}_c(v)\) |
| **Regularization** | Ridge λ ∈ {10⁻⁴, 10⁻³, …, 10} chosen by inner CV |
| **Preprocessing** | Z-score **x** per fold (train stats → apply on val) |
| **Training labels** | Measured VBench total(v, c) from pilot sweep (**offline only** — not router inputs) |
| **Evaluation** | **5-fold out-of-fold (OOF)** @ N=200 — no leakage |

**Not a neural router:** linear model, no hidden layers, no probe/TTA features in **x**.

---

### B. Router 1 — video/caption (**20.8%** captured) · `video_caption_only`

| | |
|--|--|
| **Input x(v)** | **9 scalars** from `video_features.csv` |
| **From raw video?** | **Yes — pixels + caption**, no LongCat VAE profile |
| **Compression** | None (already low-dim hand stats) |

**How x(v) is built** (`extract_video_features_for_tta.py`, frames **[0:48)**):

| # | Feature | Computation |
|---|---------|-------------|
| 3 | Cuts | PySceneDetect + histogram cut counts, density |
| 3 | Caption↔video | CLIP text–image similarity (mean, var, min) |
| 1 | Motion | DINO temporal L2 on pixels |
| 2 | Texture | Laplacian variance, RGB histogram entropy |

**Deploy:** mp4 + caption → compute 9 stats → ridge → **one** AdaSteer.

---

### C. Router 2 — VAE pooled profile (**9.7%** captured) · `vae_inference_embedding`

| | |
|--|--|
| **Input x(v)** | **130 scalars** from `vae_latent_profile_features.csv` |
| **From raw video?** | **Yes — input pixels [0:48)** via LongCat VAE, **not** GT eval frames [48:62) |
| **Compression** | **Hand pooling** of full latent tensor (~10⁵–10⁶ values → 130 summaries) |

**How x(v) is built** (`extract_vae_latent_profile_features.py`):

1. `encode_video` on visible pixels → latent **[1, C, T, H, W]** (same path as inference).
2. Split latent into **context / target / full** regions (within [0:48), TTA split — not eval GT).
3. Pool each region: per-channel **mean & std**, **token-norm** stats, **temporal-delta** stats, ctx/tgt **energy ratios** → **130 numbers**.

**Deploy:** mp4 → VAE encode (already required) → cache 130-d profile → ridge → **one** AdaSteer.

**Note:** We do **not** feed the million-element latent to ridge; fixed summarization keeps N=200 tractable.

---

**Headline for PI comparisons (Slide 6):** Router 1 (Block A) is best; Router 2 (Block C) shown for context on Slide 6.

---

## Slide 6 — Main result: comparison with AdaState

**Presentation anchor.** Both **our routers** (same ridge model, Slide 5) vs fixed AdaSteer and AdaState. **Headline = video/caption @ 20.8%.**

| Method | Δ vs **NOTTA** (base **0.772**) | Δ vs **fixed AdaSteer** (base **0.773**) | **% oracle headroom recovered** | 1× AdaSteer? |
|--------|--------------------------------|------------------------------------------|--------------------------------|--------------|
| Fixed AdaSteer (S10) @ 999v | **+0.001 (+0.13%)** | — | **0%** | Yes |
| **Our router** (video/caption, 9-d ridge) | **+0.030 (+3.9%)** | **+0.029 (+3.8%)** | **20.8%** | **Yes** |
| Our router (VAE pooled, 130-d ridge) | **+0.015 (+1.9%)** | **+0.014 (+1.8%)** | **9.7%** | Yes |
| **AdaState** (literature)† | **+0.026 (+3.4%)** | N/A (different base) | — | Yes (different stack) |
| Config oracle (pilot) | **+0.141 (+18.3%)** | **+0.140 (+18.1%)** | **100%** | No (12 configs) |

†AdaState **+3.4%** vs **their** no-TTA base — not our NOTTA 0.772.

**Takeaway:**
- **Config routing works:** best router captures **20.8%** of oracle gap · **~30×** relative lift vs fixed-vs-NOTTA (+3.8% vs +0.13%).
- **vs AdaState (honest):** Video/caption router is similar **relative** scale (**+3.9%** vs NOTTA vs **+3.4%**) — different mechanism; do **not** claim we beat AdaState.
- **VAE-pooled router:** same ridge, weaker signal (**9.7%** / **+1.8%** vs fixed) — validates that cheap video stats beat latent pooling alone in this pilot.
- **Internal >25% bar:** not met (20.8% best).

---

## Slide 7 — Block ablation (full `deploy_strict_router` run)

**Source:** `per_video_analysis/2026-07-06/deploy_strict_router/summary.md`

| Experiment | Blocks | # feat | Captured % | Match % | Δ vs fixed |
|---|---|---:|---:|---:|---:|
| **`video_caption_only`** | **A** | 9 | **20.8** | 18.5 | **+0.029 (+3.8%)** |
| `video_caption_ood` | A+B | 21 | 18.9 | 21.0 | +0.027 (+3.5%) |
| `vae_inference_embedding` | C | 130 | 9.7 | 16.5 | +0.014 (+1.8%) |
| `video_caption_ood_vae` | A+B+C | 151 | 10.1 | 19.5 | +0.014 (+1.8%) |
| `diffusion_ood_only` | B | 12 | 4.9 | 18.0 | +0.007 (+0.9%) |

**Read:**
- **A alone wins** on captured headroom (headline router).
- **A+B** trades −1.9 pp captured for +2.5 pp match rate (optional variant).
- **C (VAE-only)** and **A+B+C** do **not** beat A @ N=200 — VAE stack adds dims without gain here.
- **B alone** is weak — needs video/caption context.

---

## Slide 8 — AdaState: mechanism & apples-to-oranges

| | Fixed AdaSteer | **Our router** (video/caption, Slide 6) | **AdaState** |
|--|----------------|----------------------------------------|--------------|
| **Mechanism** | One config for all | **12 ridge models** → argmax predicted VBench → one AdaSteer | Pathwise correction during sampling |
| **Router model** | — | Linear ridge (shared architecture; Slide 5A) | Their adaptive stack |
| **Input x(v)** | — | 9-d CLIP/cuts/DINO on pixels + caption | Their generator features |
| **Reported lift** | +0.001 (+0.13%) vs NOTTA | **+0.029 (+3.8%) vs fixed** | **+0.026 (+3.4%) vs their base** |
| **Comparable?** | Baseline | **Headline result** | Partial — different base & method |

---

## Slide 9 — Opportunity size

```
NOTTA ──► Fixed S10 ──► Our router (Block A) ──► AdaState (ref) ──► Config oracle
       +0.13%          +3.8% vs fixed            +3.4%†            +18.1% vs fixed
                       (20.8% of oracle gap)
```

- **~5×** absolute headroom remains to oracle after routing  
- **>25%** internal bar: not met (20.8%)

---

## Slide 10 — Next steps

| Item | Status |
|------|--------|
| Scale Block A router calibration 500–1K | Open |
| Nonlinear router on Block A, strict OOF | Open |
| Optional A+B variant if match rate matters more than captured % | Design choice |
| Probe-and-route / multi-pass TTA before routing | **Out of scope** |

---

## Slide 11 — Claims we can make today

1. Fixed AdaSteer ≈ NOTTA @ 999v (**+0.13%**).
2. Config oracle **+18.1% vs fixed** — per-video budget choice matters.
3. **Block A router: +3.8% vs fixed / +3.9% vs NOTTA**, **20.8%** oracle headroom, one AdaSteer, 5-fold OOF.
4. Full block ablation (Slide 7): VAE-only **9.7%** — **does not beat Block A** in this pilot.
5. **vs AdaState:** comparable **~3–4% relative** lift — different method; do not claim a win.

---

## Slide 12 — FAQ backup

**Q: What is the router model?**  
**12 ridge regressors** (linear): predict VBench total per config from **x(v)**; pick argmax; 5-fold OOF. See Slide 5A.

**Q: What are the two input variants?**  
**Router 1 (headline):** 9-d video/caption stats → **20.8%**. **Router 2 (ablation):** 130-d **pooled** LongCat-VAE profile (not raw latent) → **9.7%**. Same ridge, different **x(v)**.

**Q: Which runs are in the AdaState table?**  
**Both routers:** video/caption (**20.8%**, headline) and VAE pooled (**9.7%**). Same ridge model (Slide 5A).

**Q: How are percentages computed?**  
**% = Δ / baseline VB total** (NOTTA **0.772**, fixed **0.773** @ 999v).

---

## Reference numbers

| Quantity | Absolute Δ | Relative % |
|----------|------------|------------|
| **Block A router** (`video_caption_only`) | +0.029 vs fixed | **+3.8%** |
| Block A vs NOTTA | +0.030 | **+3.9%** |
| Block C (`vae_inference_embedding`) | +0.014 vs fixed | **+1.8%** |
| Block A+B (`video_caption_ood`) | +0.027 vs fixed | **+3.5%** |
| Oracle headroom recovered (Block A) | — | **20.8%** |
| AdaState vs their base | ~+0.026 | **+3.4%** |

**Cluster path:** `per_video_analysis/2026-07-06/deploy_strict_router/summary.md`  
**Paper table:** `2026-07-07_deploy_router_structured_blocks.md`
