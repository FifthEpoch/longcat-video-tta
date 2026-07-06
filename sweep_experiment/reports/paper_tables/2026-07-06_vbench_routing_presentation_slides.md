# VBench++ Config Routing — Presentation Slides
**Date:** 2026-07-07  
**Audience:** PI / partner meeting (no prior chat context assumed)  
**Pilot:** 200 OOD-stratified Panda clips × 12 AdaSteer configs  
**Fixed AdaSteer baseline:** S10_LR5e-3 (headline deployable config @ 999v)

**Metric convention:** **Δ vs NOTTA** / **Δ vs fixed** report absolute VBench total change and **% relative to that row’s baseline** (same style as AdaState **+3.4%**). Denominators @ 999v Panda standard: **NOTTA = 0.772**, **fixed S10 = 0.773** (`panda_1000v_standard`). Router Δ from **N=200 OOF** pilot. **% oracle headroom recovered** = (policy − fixed) / (config oracle − fixed); oracle gap = **+0.140**.

**Our router (this deck):** **Block A** — 9-d video/caption stats → ridge → **one** AdaSteer. No probe, no Tier-3, no prior TTA metrics as inputs.

---

## Slide 1 — Title

**VBench++ Config Routing for AdaSteer**  
*Pick step×LR from the input video, then run AdaSteer once*

**Headline @ N=200:** Block A router → **20.8%** oracle headroom recovered · **+0.029 (+3.8%) vs fixed S10** · **~+0.030 (+3.9%) vs NOTTA**.

---

## Slide 2 — Problem & baselines

**Question:** Can we pick AdaSteer (steps × LR) per video from **pre-adaptation signals**, then run **one** adaptation pass?

| Baseline | Role |
|----------|------|
| **NOTTA** | Does TTA help at all? |
| **Fixed AdaSteer (S10/LR5e-3)** | Does *smart config choice* beat our best single recipe? |

**@ 999v:** Fixed AdaSteer vs NOTTA = **+0.001 (+0.13%)**.  
**Pilot @ N=200:** Config oracle vs fixed S10 = **+0.140 (+18.1%)** headroom (Slide 3).

---

## Slide 3 — Config oracle (upper bound, not deployable)

**Grid:** S2/S5/S10/S20 × LR {1e-3, 5e-3, 1e-2} → **12 configs** (measured offline on pilot).

\[
\text{oracle}(v) = \arg\max_{c \in \{12\}} \ \text{VBench}_\text{total}(v, c)
\]

**Headroom vs fixed S10:** **+0.140 (+18.1%)** · **not deployable** (needs all 12 runs at inference).

**Router score:** fraction of this gap recovered with **one** AdaSteer after routing.

---

## Slide 4 — Pilot design note (OOD stratification)

Pilot clips were **OOD-stratified** (40 videos × 5 quintiles by frozen DiT difficulty) so the 200-video set spans easy→hard base-model surprise — not a router input in Block A, but explains “OOD” in the pilot name.

**Source:** `compute_diffusion_ood_score.py` + `sample_ood_quintile_videos.py` (see paper methods).

---

## Slide 5 — Our router (Block A)

### Deploy workflow

```
Input video (+ caption)  →  Block A features x(v)  →  ridge  →  ONE AdaSteer
```

**Rules:** No AdaSteer / probe TTA before config choice.

### Feature space **x(v)** — 9 dimensions

| Feature group | Dims | Source |
|---------------|-----:|--------|
| Cut structure | 3 | pyscenedetect + histogram cut counts, cut density |
| Caption–video alignment | 3 | CLIP text–image sim (mean, var, min) |
| Motion / texture | 3 | DINO temporal L2, Laplacian variance, RGB entropy |

**CSV:** `video_features.csv` · **Model:** 12 ridge regressors, argmax predicted VBench · **Eval:** 5-fold OOF @ N=200.

**Offline labels only:** pilot VBench for all 12 configs (lab calibration — not router inputs).

---

## Slide 6 — Main result: comparison with AdaState

**Presentation anchor.** Every Δ shows **absolute (+ relative % vs that column’s baseline)**.

| Method | Δ vs **NOTTA** (base **0.772**) | Δ vs **fixed AdaSteer** (base **0.773**) | **% oracle headroom recovered** | 1× AdaSteer? |
|--------|--------------------------------|------------------------------------------|--------------------------------|--------------|
| Fixed AdaSteer (S10) @ 999v | **+0.001 (+0.13%)** | — | **0%** | Yes |
| **Our router — Block A** | **+0.030 (+3.9%)** | **+0.029 (+3.8%)** | **20.8%** | **Yes** |
| **AdaState** (literature)† | **+0.026 (+3.4%)** | N/A (different base) | — | Yes (different stack) |
| Config oracle (pilot) | **+0.141 (+18.3%)** | **+0.140 (+18.1%)** | **100%** | No (12 configs) |

†**AdaState:** **+3.4%** is vs **their** no-TTA / base generator (not our NOTTA 0.772). Different model and protocol — **context row**, not a claimed win.

**% oracle headroom** = (method − fixed) / (oracle − fixed); pilot gap **+0.140**.

**Takeaway for PI:**
- **Routing works:** Block A recovers **20.8%** of config-oracle gap — **~30×** the relative lift of fixed-vs-NOTTA (+3.8% vs +0.13%).
- **vs AdaState (honest):** Similar **relative** scale (**+3.9%** vs NOTTA vs AdaState **+3.4%** vs their base) and similar **absolute** Δ (~0.03) — **different mechanism** (config routing vs pathwise correction). Do **not** claim we beat AdaState.
- **Internal bar (>25% oracle headroom):** still **not met** (20.8%).

**Do not say:** “We beat AdaState.” **Do say:** “Config routing reaches **AdaState-comparable relative VBench lift** with one AdaSteer and no probe TTA.”

---

## Slide 7 — AdaState: mechanism & apples-to-oranges

| | Fixed AdaSteer | **Our router (Block A)** | **AdaState** |
|--|----------------|--------------------------|--------------|
| **What it optimizes** | One step×LR for all videos | Per-video step×LR before adapting | Pathwise correction during sampling |
| **Input at deploy** | Video + caption | 9-d video/caption stats | Their generator state / features |
| **Adaptation cost** | 1× AdaSteer | **1× AdaSteer** | 1× (their stack) |
| **Reported lift** | +0.001 (+0.13%) vs NOTTA | **+0.029 (+3.8%) vs fixed** · **+0.030 (+3.9%) vs NOTTA** | **+0.026 (+3.4%) vs their base** |
| **Comparable?** | Our baseline | **Primary result** | **Partial** — magnitude only |

---

## Slide 8 — Opportunity size

```
NOTTA ──► Fixed S10 ──► Our router (A) ──► AdaState (ref) ──► Config oracle
       +0.13%          +3.8% vs fixed       +3.4%†            +18.1% vs fixed
                       (20.8% of oracle gap)
```
†AdaState vs their base — not on this chain.

- **Fixed → router (A):** **+3.8%** relative — main deploy win  
- **Router → oracle:** **~5×** absolute headroom remains (+0.140 − 0.029)  
- **>25%** internal bar: not met (20.8%)

---

## Slide 9 — Next steps

| Item | Status |
|------|--------|
| Scale routing calibration 500–1K (Block A features) | Open |
| Small nonlinear router on Block A, strict OOF | Open |
| 999v × 12 retrain | Not started |
| Probe-and-route / extra AdaSteer before routing | **Out of scope** |

---

## Slide 10 — Claims we can make today

1. Fixed AdaSteer ≈ NOTTA @ 999v (**+0.13%**).
2. Config oracle **+18.1% vs fixed** @ N=200 — per-video budget choice matters in principle.
3. **Our router (Block A): +3.8% vs fixed / +3.9% vs NOTTA**, **20.8%** of oracle gap, one AdaSteer, 5-fold OOF.
4. **vs AdaState:** comparable **relative** lift (~**3.9%** vs NOTTA vs their **+3.4%**) — different base & method; do not claim a win.
5. Below **25%** internal oracle-headroom bar.

---

## Slide 11 — FAQ backup

**Q: What is “our router”?**  
Block **A** (9-d video/caption) → ridge over 12 configs → one AdaSteer. 5-fold OOF @ N=200.

**Q: How are the percentages computed?**  
**% = Δ / baseline VBench total.** NOTTA **0.772**, fixed S10 **0.773** @ 999v; router Δ from N=200 pilot OOF. AdaState **+3.4%** uses **their** base (Slide 6 †).

**Q: How do we compare to AdaState?**  
Slide 6 (numbers) + Slide 7 (mechanism). Similar **~3–4% relative** scale; not a direct horse race.

---

## Reference numbers

| Quantity | Absolute Δ | Relative % |
|----------|------------|------------|
| NOTTA → fixed S10 @ 999v | +0.001 | **+0.13%** |
| Fixed → **router Block A** (pilot OOF) | +0.029 | **+3.8%** |
| NOTTA → **router Block A** (pilot OOF) | +0.030 | **+3.9%** |
| Fixed → config oracle (pilot) | +0.140 | **+18.1%** |
| NOTTA → config oracle (pilot) | +0.141 | **+18.3%** |
| AdaState vs their base (literature) | ~+0.026 | **+3.4%** |
| Oracle headroom recovered (router A) | — | **20.8%** |

**Baselines for %:** NOTTA VB total **0.772**, fixed S10 **0.773** @ 999v Panda standard.

**Cluster path:** `per_video_analysis/2026-07-06/deploy_strict_router/`  
**Paper table:** `2026-07-07_deploy_router_structured_blocks.md`
