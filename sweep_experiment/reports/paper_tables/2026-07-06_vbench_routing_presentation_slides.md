# VBench++ Config Routing — Presentation Slides
**Date:** 2026-07-07  
**Audience:** PI / partner meeting (no prior chat context assumed)  
**Pilot:** 200 OOD-stratified Panda clips × 12 AdaSteer configs  
**Fixed AdaSteer baseline:** S10_LR5e-3 (headline deployable config @ 999v)

**Metric convention:** Routing gains vs **NOTTA** and **fixed AdaSteer**. **% oracle headroom recovered** = (policy − fixed) / (config oracle − fixed); oracle mean gap = **+0.140** VBench total @ N=200.

**Our router (this deck):** **Block A** (9-d video/caption stats) with **optional Block B** (12-d diffusion-OOD). Linear ridge, 5-fold OOF @ N=200 → **one** AdaSteer pass. No probe runs, no Tier-3 LoRA, no prior TTA metrics as inputs.

---

## Slide 1 — Title

**VBench++ Config Routing for AdaSteer**  
*Pick step×LR from the input video (+ optional OOD), then run AdaSteer once*

**Headline @ N=200:** **Block A → 20.8%** oracle headroom recovered (**+0.029 vs fixed S10**). With **Block B (OOD) allowed → 18.9%** (+0.027), best oracle-config match (21%).

---

## Slide 2 — Problem & baselines

**Question:** Can we pick AdaSteer (steps × LR) per video from **pre-adaptation signals**, then run **one** adaptation pass?

| Baseline | Role |
|----------|------|
| **NOTTA** | Does TTA help at all? |
| **Fixed AdaSteer (S10/LR5e-3)** | Does *smart config choice* beat our best single recipe? |

**@ 999v:** Fixed AdaSteer ≈ NOTTA (**+0.001**, ~+0.13%).  
**Pilot @ N=200:** Config oracle vs fixed S10 = **+0.140** headroom (Slide 3).

---

## Slide 3 — Config oracle (upper bound, not deployable)

**Grid:** S2/S5/S10/S20 × LR {1e-3, 5e-3, 1e-2} → **12 configs** (measured offline on pilot).

\[
\text{oracle}(v) = \arg\max_{c \in \{12\}} \ \text{VBench}_\text{total}(v, c)
\]

**Headroom vs fixed S10:** **+0.140** mean · **not deployable** (needs all 12 runs at inference).

**Router score:** fraction of this gap recovered with **one** AdaSteer after routing.

---

## Slide 4 — Block B: diffusion-OOD (optional router input)

**What it is:** Frozen **base LongCat DiT** flow-matching loss on the visible window — higher loss ⇒ more “surprised” / OOD (`compute_diffusion_ood_score.py`).

**Computation (no TTA adapters):**
1. VAE-encode visible window; context / target split (same as AdaSteer).
2. Sample \(t \in \{100,500,900\}\), noise \(\varepsilon\), corrupt targets.
3. Predict velocity \(\hat v = f_\theta(x_t, t, c)\); loss \(\|\hat v - (\varepsilon - x_0)\|^2\) on targets.
4. **12 router features** in pilot CSV: per-t caption/uncond losses + score norms + summaries (e.g. `mean_diffusion_loss_caption`).

**Also used for pilot design:** OOD quintile stratification → N=200 (`sample_ood_quintile_videos.py`).

**Sources:** Lipman et al. ICLR 2023 (flow matching); Kingma et al. NeurIPS 2021 (loss as difficulty proxy).

---

## Slide 5 — Our router

### Deploy workflow

```
Input video (+ caption)  →  Block A  [→  Block B if allowed]  →  x(v)  →  ridge  →  ONE AdaSteer
```

**Rules:** No AdaSteer / probe TTA before config choice. Block B = one frozen DiT OOD pass — **optional**.

---

### Feature space **x(v) = [A | B?]**

| Block | Name | Dims | Source | Deploy |
|-------|------|-----:|--------|--------|
| **A** | `video_caption` | **9** | `video_features.csv` | **Default** |
| **B** | `diffusion_ood` | **12** | `diffusion_ood_scores.csv` | Optional |

**Block A (9):** cut counts (×3), CLIP text–image sim (×3), DINO temporal L2, Laplacian variance, RGB histogram entropy.

**Block B (12):** frozen DiT OOD features @ t∈{100,500,900} + aggregate stats (Slide 4).

**Offline labels only:** pilot VBench total for all 12 configs (lab calibration — not router inputs).

---

### Model & evaluation

| Step | Detail |
|------|--------|
| **Model** | 12 ridge regressors (one per config); \(\hat c = \arg\max_c \widehat{\text{VB}}_c\) |
| **Regularization** | Ridge λ via inner CV; features z-scored per fold |
| **Eval** | **5-fold OOF** @ N=200 (no leakage) |
| **Script** | `run_deploy_strict_router_experiments.py` |

---

## Slide 6 — Main result: comparison with AdaState

**This slide is the presentation anchor** — same columns as prior deck; numbers updated for Block A / A+B router @ N=200 OOF.

| Method | Δ vs **NOTTA** | Δ vs **fixed AdaSteer** | **% oracle headroom recovered** | 1× AdaSteer? |
|--------|----------------|-------------------------|--------------------------------|--------------|
| Fixed AdaSteer (S10) @ 999v | +0.001 (~+0.13%) | — | **0%** | Yes |
| **Our router — Block A** (`video_caption_only`) | **~+0.030 (~3.7%)** | **+0.029** | **20.8%** | **Yes** |
| **Our router — A+B** (`video_caption_ood`) | **~+0.028 (~3.5%)** | **+0.027** | **18.9%** | **Yes** (+ OOD pre-pass) |
| **AdaState** (literature) | **~+0.026 (~+3.4%)**† | N/A (different base) | — | Yes (different stack) |
| Config oracle (pilot) | ~+0.141 | **+0.140** | **100%** | No (12 configs) |

†**AdaState:** reported **~+3.4% relative** VBench total vs **their** base generator (~**+0.026** absolute on comparable scales) — **not** vs our NOTTA/fixed S10. Different model, task, and metric protocol; row is **context**, not a claimed win.

**% oracle headroom** = (method − fixed S10) / (config oracle − fixed S10); pilot oracle gap = **+0.140**.

**Takeaway for PI:**
- **Routing works:** Block A recovers **~21%** of config-oracle gap with **one** AdaSteer — **~10×** our prior flat fixed-vs-NOTTA story @ 999v.
- **vs AdaState (honest):** Our **absolute** lift vs fixed (**+0.029**) is **similar in magnitude** to AdaState’s reported **+0.026** vs their base — but **bases and stacks differ**. We do **not** claim to beat AdaState; we claim **comparable-order population lift** from a **different, cheaper mechanism** (config routing, not pathwise correction).
- **Internal bar (>25% oracle headroom):** still **not met** (20.8%); config oracle (+0.140) shows headroom remains.

**Do not say:** “We beat AdaState.” **Do say:** “Honest config routing reaches AdaState-scale **absolute** VBench deltas vs our fixed baseline, with one AdaSteer and no probe TTA.”

---

## Slide 7 — Block ablation (router feature blocks)

| Config | Blocks | Dims | Captured % | Δ vs fixed | Match % |
|--------|--------|-----:|-----------:|-----------:|--------:|
| **A only** | video/caption | 9 | **20.8** | +0.029 | 18.5 |
| **A + B** | + diffusion-OOD | 21 | **18.9** | +0.027 | **21.0** |
| B only | OOD alone | 12 | 4.9 | +0.007 | 18.0 |

**Takeaways:** Signal lives in **Block A**. OOD alone is weak; stacked with A it **improves match rate** but **slightly lowers** captured headroom (−1.9 pp). Do **not** stack unrelated high-dim blocks @ N=200 (overfits).

---

## Slide 8 — AdaState: mechanism & apples-to-oranges

| | Fixed AdaSteer | **Our router (A or A+B)** | **AdaState** |
|--|----------------|---------------------------|--------------|
| **What it optimizes** | One step×LR for all videos | Per-video step×LR before adapting | Pathwise / per-step correction during sampling |
| **Input at deploy** | Video + caption | Block A (+ optional Block B OOD) | Their generator state / features |
| **Adaptation cost** | 1× AdaSteer | **1× AdaSteer** | 1× (their stack; not LongCat AdaSteer) |
| **Reported VBench lift** | +0.001 vs NOTTA @ 999v | **+0.029 vs fixed** (Block A @ N=200 OOF) | **~+0.026 vs their base** (~+3.4% rel) |
| **Comparable to us?** | Our baseline | **Yes — primary result** | **Partial** — magnitude only; rebase differs |

**Why both slides matter:** Slide 6 = **numbers for the PI** (NOTTA / fixed / us / AdaState / oracle). Slide 8 = **why the comparison is informative but not a horse race.**

---

## Slide 9 — Opportunity size (vs oracle & AdaState)

```
NOTTA ──► Fixed S10 ──► Our router (A) ──► AdaState (ref) ──► Config oracle
         +0.001          +0.029              ~+0.026†           +0.140
                         (20.8% of oracle gap)
```
†AdaState vs their base — not on this chain.

- **Fixed → router (A):** **+0.029** — main deploy win vs status quo  
- **Router → oracle:** **~5×** headroom still on table (+0.140 − 0.029)  
- **Router vs AdaState magnitude:** similar **absolute** Δ (~0.03) — **different mechanism & base** (Slide 8)  
- **>25%** internal bar: not met (20.8%)

---

## Slide 10 — Next steps

| Item | Status |
|------|--------|
| Scale routing calibration 500–1K (same A / A+B features) | Open |
| Small nonlinear router on A (+ B), strict OOF | Open |
| 999v × 12 retrain with this feature set | Not started |
| Probe-and-route / extra AdaSteer before routing | **Out of scope** |

---

## Slide 11 — Claims we can make today

1. Fixed AdaSteer ≈ NOTTA @ 999v (+0.13%).
2. Config oracle **+0.140 vs fixed** @ N=200 — per-video budget choice matters in principle.
3. **Our router (Block A): +0.029 vs fixed**, **20.8%** of oracle gap, **one** AdaSteer, 5-fold OOF.
4. **Optional Block B (OOD): +0.027 vs fixed**, **21%** oracle-config match.
5. **vs AdaState:** similar **absolute** VBench lift (~+0.03) but **different base & method** — config routing, not pathwise correction; do not claim a win.

---

## Slide 12 — FAQ backup

**Q: What is “our router”?**  
Ordered blocks **A** (9-d video/caption) **[+ B (12-d OOD)]** → ridge argmax over 12 configs → one AdaSteer. Eval: 5-fold OOF @ N=200.

**Q: Why two deploy variants (A vs A+B)?**  
**A** maximizes recovered headroom with lowest pre-pass cost. **A+B** when a frozen DiT OOD pass is acceptable — better config match, −1.9 pp captured.

**Q: How do we compare to AdaState?**  
See **Slide 6** (numbers) and **Slide 8** (mechanism). Similar **absolute** Δ (~+0.03) but different base and method — **not** a direct win; informative benchmark.

**Q: Why N=200?**  
OOD-stratified pilot for honest OOF; same feature contract scales to larger label sets later.

---

## Reference numbers

| Quantity | Value |
|----------|-------|
| Oracle headroom vs fixed (pilot) | +0.140 |
| **Block A — headroom recovered** | **20.8%** |
| **Block A — Δ vs fixed** | **+0.029** |
| **A+B — headroom recovered** | **18.9%** |
| **A+B — Δ vs fixed** | **+0.027** |
| **A+B — oracle match rate** | **21.0%** |
| Fixed AdaSteer vs NOTTA @ 999v | +0.001 |
| AdaState VBench lift (literature) | ~+0.026 abs / ~+3.4% rel vs their base |
| Internal success bar | >25% oracle headroom (not met; best 20.8%) |

**Cluster path:** `per_video_analysis/2026-07-06/deploy_strict_router/`  
**Paper table:** `2026-07-07_deploy_router_structured_blocks.md`
