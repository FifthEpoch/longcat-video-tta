# VBench++ Config Routing — Presentation Slides
**Date:** 2026-07-07  
**Audience:** PI / partner meeting (no prior chat context assumed)  
**Pilot:** 200 OOD-stratified Panda clips × 12 AdaSteer configs  
**Fixed AdaSteer baseline:** S10_LR5e-3 (headline deployable config @ 999v)

**Metric convention:** **Δ vs NOTTA** / **Δ vs fixed** show absolute change + **% relative to that baseline** (AdaState-style). Denominators @ 999v Panda standard: **NOTTA = 0.772**, **fixed S10 = 0.773**. Router Δ from **N=200 OOF** (`vae_inference_embedding`). **% oracle headroom recovered** = (policy − fixed) / (oracle − fixed); oracle gap = **+0.140**.

**Our router (this deck):** **LongCat-VAE latent profile only** (~130-d from `encode_video` on the input video) → ridge → **one** AdaSteer. **No** CLIP/cuts/OOD/Tier-3/probe/TTA-side metrics. Reuses the VAE encode inference already requires.

---

## Slide 1 — Title

**VBench++ Config Routing for AdaSteer**  
*Pick step×LR from the VAE representation of the input video, then run AdaSteer once*

**Headline @ N=200:** VAE router → **9.7%** oracle headroom recovered · **+0.014 (+1.8%) vs fixed S10** · **+0.015 (+1.9%) vs NOTTA**.

---

## Slide 2 — Problem & baselines

**Question:** Can we pick AdaSteer (steps × LR) from the **VAE embedding of the input clip** (computed anyway for LongCat), then run **one** adaptation pass?

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

The **200-video pilot** was OOD-stratified (40 × 5 quintiles by frozen DiT difficulty) for representative easy→hard clips. That OOD score was used to **build the pilot set**, **not** as a router input in this experiment.

---

## Slide 5 — Our router (VAE inference embedding)

### Deploy workflow

```
Input video  →  encode_video (LongCat VAE)  →  latent profile x(v)  →  ridge  →  ONE AdaSteer
```

**Rules:** No AdaSteer / probe TTA before config choice. **Only** features derived from the same VAE encode path LongCat inference already runs.

---

### Feature space **x(v) ∈ ℝ^130**

**Source:** `vae_latent_profile_features.csv` · `extract_vae_latent_profile_features.py`  
**Input pixels:** TTA-visible window **[0:48)** @ 480p (same as AdaSteer).

Pooled statistics over VAE latents **[B, C, T, H, W]**:

| Pool | What is summarized |
|------|---------------------|
| **Full** window | Per-channel mean/std, token-norm stats, temporal-delta stats |
| **Context** latents | Same pools on clean-context region |
| **Target** latents | Same pools on generation-target region |
| **Ctx vs tgt** | Per-channel energy ratios |

**Not used as router inputs:** `video_features.csv` (CLIP/DINO/cuts), diffusion-OOD, Tier-3 LoRA probes, AdaSteer/NOTTA eval metrics, probe PSNR/SSIM.

---

### Model & evaluation

| Step | Detail |
|------|--------|
| **Model** | 12 ridge regressors; \(\hat c = \arg\max_c \widehat{\text{VB}}_c\) |
| **Regularization** | Ridge λ via inner CV; z-score per fold |
| **Eval** | **5-fold OOF** @ N=200 · experiment `vae_inference_embedding` |
| **Offline labels** | Pilot VBench for 12 configs (calibration only) |

---

## Slide 6 — Main result: comparison with AdaState

**Presentation anchor.** Every Δ: **absolute (+ relative % vs that column’s baseline)**.

| Method | Δ vs **NOTTA** (base **0.772**) | Δ vs **fixed AdaSteer** (base **0.773**) | **% oracle headroom recovered** | 1× AdaSteer? |
|--------|--------------------------------|------------------------------------------|--------------------------------|--------------|
| Fixed AdaSteer (S10) @ 999v | **+0.001 (+0.13%)** | — | **0%** | Yes |
| **Our router — VAE embedding** | **+0.015 (+1.9%)** | **+0.014 (+1.8%)** | **9.7%** | **Yes** |
| **AdaState** (literature)† | **+0.026 (+3.4%)** | N/A (different base) | — | Yes (different stack) |
| Config oracle (pilot) | **+0.141 (+18.3%)** | **+0.140 (+18.1%)** | **100%** | No (12 configs) |

†**AdaState:** **+3.4%** vs **their** no-TTA base — not our NOTTA 0.772. Different model/protocol.

**Takeaway for PI:**
- **Routing works at deploy bar:** VAE-only features recover **9.7%** of config-oracle gap with **one** AdaSteer — **~14×** fixed-vs-NOTTA on a relative scale (+1.8% vs +0.13%).
- **vs AdaState (honest):** Our lift is **smaller** (**+1.9%** vs NOTTA vs AdaState **+3.4%**) — similar *idea* (pre-adapt signal → better generation), **not** similar magnitude yet.
- **Internal bar (>25% oracle headroom):** **not met** (9.7%).

**Do not say:** “We match AdaState.” **Do say:** “Honest VAE-embedding routing beats fixed AdaSteer with **no extra TTA before routing**; headroom remains vs oracle (+18.1%).”

---

## Slide 7 — AdaState: mechanism & apples-to-oranges

| | Fixed AdaSteer | **Our router (VAE)** | **AdaState** |
|--|----------------|----------------------|--------------|
| **What it optimizes** | One step×LR for all | Per-video step×LR from VAE profile | Pathwise correction during sampling |
| **Input at deploy** | Video + caption | **~130-d VAE latent pools** (encode path) | Their generator state / features |
| **Extra pre-pass** | None beyond AdaSteer | **VAE encode only** (already required) | Their stack |
| **Reported lift** | +0.001 (+0.13%) vs NOTTA | **+0.014 (+1.8%) vs fixed** · **+0.015 (+1.9%) vs NOTTA** | **+0.026 (+3.4%) vs their base** |
| **Comparable?** | Our baseline | **Primary result** | **Partial** — not same base |

---

## Slide 8 — Opportunity size

```
NOTTA ──► Fixed S10 ──► Our router (VAE) ──► AdaState (ref) ──► Config oracle
       +0.13%          +1.8% vs fixed        +3.4%†            +18.1% vs fixed
                       (9.7% of oracle gap)
```
†AdaState vs their base.

- **Fixed → VAE router:** **+1.8%** relative — real but modest  
- **Router → oracle:** **~10×** absolute headroom still on table (+0.140 − 0.014)  
- **vs AdaState:** we are at **~half** their relative lift today (**1.9%** vs **3.4%**)  
- **>25%** internal bar: not met (9.7%)

---

## Slide 9 — Next steps

| Item | Status |
|------|--------|
| Richer VAE-side router (MLP / low-rank on latent pools), strict OOF | Open |
| Scale routing calibration 500–1K (same VAE-only contract) | Open |
| 999v × 12 retrain with VAE profile features | Not started |
| Probe-and-route / CLIP-cut side features | **Out of scope for this deploy story** |

---

## Slide 10 — Claims we can make today

1. Fixed AdaSteer ≈ NOTTA @ 999v (**+0.13%**).
2. Config oracle **+18.1% vs fixed** — per-video budget choice matters in principle.
3. **VAE-embedding router: +1.8% vs fixed / +1.9% vs NOTTA**, **9.7%** of oracle gap, one AdaSteer, 5-fold OOF.
4. **vs AdaState:** same *class* of idea, **~half** their relative lift — do not claim parity.
5. Below **25%** internal oracle-headroom bar.

---

## Slide 11 — FAQ backup

**Q: What is “our router”?**  
**Only** pooled LongCat-VAE latent stats (~130-d) from `encode_video` → ridge → one AdaSteer. No CLIP, OOD, or probe features.

**Q: Why VAE and not CLIP/cuts?**  
Deploy contract: reuse the **inference VAE encode**, no extra adapters or prior TTA metrics. (Lab ablations with other feature blocks were run separately; **this deck is the VAE-only experiment**.)

**Q: How are percentages computed?**  
**% = Δ / baseline VB total.** NOTTA **0.772**, fixed **0.773** @ 999v. Router Δ from N=200 OOF.

---

## Reference numbers

| Quantity | Absolute Δ | Relative % |
|----------|------------|------------|
| NOTTA → fixed S10 @ 999v | +0.001 | **+0.13%** |
| Fixed → **VAE router** (pilot OOF) | +0.014 | **+1.8%** |
| NOTTA → **VAE router** (pilot OOF) | +0.015 | **+1.9%** |
| Fixed → config oracle (pilot) | +0.140 | **+18.1%** |
| AdaState vs their base (literature) | ~+0.026 | **+3.4%** |
| Oracle headroom recovered | — | **9.7%** |
| Oracle-config match rate | — | **16.5%** |

**Cluster path:** `per_video_analysis/2026-07-06/deploy_strict_router/`  
**Paper table:** `2026-07-07_deploy_strict_router_vae_only.md`
