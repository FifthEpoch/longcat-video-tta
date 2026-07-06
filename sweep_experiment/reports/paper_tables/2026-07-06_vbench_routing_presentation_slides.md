# VBench++ Config Routing — Presentation Slides
**Date:** 2026-07-07  
**Audience:** PI / partner meeting (no prior chat context assumed)  
**Pilot:** 200 OOD-stratified Panda clips × 12 AdaSteer configs  
**Fixed AdaSteer baseline:** S10_LR5e-3 (headline deployable config @ 999v)

**Experiment:** `deploy_strict_router/` — structured feature blocks **A / B / C**, ridge OOF @ N=200. **No** Tier-3, probe, or TTA eval metrics as inputs.

**Metric convention:** Δ columns show **absolute (+ relative % vs that baseline)**. Denominators @ 999v Panda standard: **NOTTA = 0.772**, **fixed S10 = 0.773**. Router Δ from N=200 OOF. **% oracle headroom recovered** = (policy − fixed) / (oracle − fixed); oracle gap = **+0.140**.

**Our router (this deck):** **Block A — `video_caption_only`** (9-d video/caption stats) → ridge → **one** AdaSteer. Best result in the block ablation: **20.8%** captured.

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

## Slide 5 — Our router (Block A — `video_caption_only`)

### Deploy workflow

```
Input video (+ caption)  →  Block A features x(v)  →  ridge  →  ONE AdaSteer
```

**Rules:** No AdaSteer / probe TTA before config choice. No Tier-3 LoRA.

### Feature space **x(v)** — 9 dimensions

| Feature group | Dims | Source |
|---------------|-----:|--------|
| Cut structure | 3 | pyscenedetect + histogram cut counts, cut density |
| Caption–video alignment | 3 | CLIP text–image sim (mean, var, min) |
| Motion / texture | 3 | DINO temporal L2, Laplacian variance, RGB entropy |

**CSV:** `video_features.csv` · **Experiment ID:** `video_caption_only` · **Eval:** 5-fold OOF @ N=200.

**Offline labels only:** pilot VBench for 12 configs (not router inputs).

---

## Slide 6 — Main result: comparison with AdaState

**Presentation anchor.** **Our router row = Block A only** (20.8% run).

| Method | Δ vs **NOTTA** (base **0.772**) | Δ vs **fixed AdaSteer** (base **0.773**) | **% oracle headroom recovered** | 1× AdaSteer? |
|--------|--------------------------------|------------------------------------------|--------------------------------|--------------|
| Fixed AdaSteer (S10) @ 999v | **+0.001 (+0.13%)** | — | **0%** | Yes |
| **Our router — Block A** | **+0.030 (+3.9%)** | **+0.029 (+3.8%)** | **20.8%** | **Yes** |
| **AdaState** (literature)† | **+0.026 (+3.4%)** | N/A (different base) | — | Yes (different stack) |
| Config oracle (pilot) | **+0.141 (+18.3%)** | **+0.140 (+18.1%)** | **100%** | No (12 configs) |

†AdaState **+3.4%** vs **their** no-TTA base — not our NOTTA 0.772.

**Takeaway:**
- **Block A routing works:** **20.8%** of oracle gap · **~30×** relative lift vs fixed-vs-NOTTA (+3.8% vs +0.13%).
- **vs AdaState (honest):** Similar **relative** scale (**+3.9%** vs NOTTA vs **+3.4%**) — different mechanism; do **not** claim we beat AdaState.
- **Internal >25% bar:** not met (20.8%) — closest honest result so far.

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

| | Fixed AdaSteer | **Our router (Block A)** | **AdaState** |
|--|----------------|--------------------------|--------------|
| **Mechanism** | One config for all | Per-video step×LR from 9-d video/caption stats | Pathwise correction |
| **Pre-adapt cost** | None | Cheap CLIP/DINO/cut stats on clip | Their stack |
| **Reported lift** | +0.001 (+0.13%) vs NOTTA | **+0.029 (+3.8%) vs fixed** | **+0.026 (+3.4%) vs their base** |
| **Comparable?** | Baseline | **Primary result** | Partial — different base |

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

**Q: Which run is the 20.8%?**  
Experiment **`video_caption_only`** — Block **A** only (9 features from `video_features.csv`). Not VAE-only (that is Block C @ **9.7%**).

**Q: What about VAE / OOD blocks?**  
Same pilot ablation (Slide 7). **A wins** on captured %; A+B optional for match rate; C alone underperforms A.

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
