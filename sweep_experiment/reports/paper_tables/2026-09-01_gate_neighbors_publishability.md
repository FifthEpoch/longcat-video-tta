# Gate neighbors + publishability (2026-09-01)

Not a submit spec. Literature check after the user asked
whether a prefix-hold-out gate is publishable if it does
not beat Always-search.

## Closest papers (same *problem*, not the same gate)

| Paper | Gate / skip | What they spend | vs our hold-out |
|---|---|---|---|
| [Early Failure Detection](https://arxiv.org/abs/2603.14320) (KAIST, 2026) | Mid-denoise RGB preview + VLM alignment; intervene only if fail | Regen / hierarchical fix | Closest *whether-to-spend*. 50-step T2V, not 4-step V2V prefix MAE. They report quality *and* ≤2.6× vs post-hoc retry. |
| [Video-T1](https://arxiv.org/abs/2503.18942) ToF (ICCV 2025) | Adaptive expand / prune per frame | Always searching, variable width | Cheapen *inside* search. O(TN) → nearer O(N+T). |
| [CachedSearch](https://arxiv.org/abs/2607.23159) (2026) | Cache-skip while drift ≤ τ; always search N | Full compute only on winner | Always-on search, cheaper tries. 94.7% of BoN-8 at 63% cost. Our CPU KV snap **failed** on 4-step DMD. |
| [LatSearch](https://arxiv.org/abs/2603.14526) (2026) | Learned latent reward; resample / prune | Mid-trajectory, Wan 1.3B | Proxy before full decode. Trained Qwen-VL reward. Up to 79% runtime vs search SOTA. |
| [DSA](https://arxiv.org/abs/2606.04432) (2026) | Confidence head → fewer denoise steps | Step count, not BoN | Adaptive *depth*, not *whether to BoN*. Needs a trained head. |
| [TANGO](https://arxiv.org/abs/2607.15849) (ECCV 2026) | Noise residual ≠ Gaussian → terminal point | Rank-8 LoRA | Critic we can cite. Adaptation locked out. Our U_t probe matched do-nothing. |
| [Temporal Backtracking Search](https://arxiv.org/abs/2606.13861) (2026) | Verify, then restart from a clean prefix | Generate–verify–restart | Always searching; reallocates to failed prefixes. Reasoning videos, not freeze. |
| [SDVG](https://arxiv.org/abs/2604.17397) (Hu & Zhang, 2026) | ImageReward on a 1.3B draft block; accept or regenerate with 14B | Per-block, 1.59× at 98% quality | Closest *accept/reject* on AR video. They verify **this** block with a VLM. We verify a **held-out past** with MAE to gate later chunks. Same 1.3B / 4-step family. |

Nobody else hides the last 0.7 s of an observed V2V prefix
and uses pixel MAE vs extra seeds as a fire bit for the
unseen 30 s. That slice is ours. The *class* (decide
whether to pay search) is not.

## What we can honestly claim

Always-search is the quality method. Pseudo is Always with
a Self Forcing abort on 38/128. Cite-128: Always +4 Dyn
clips, subject/IQ tie, **13%** mean-wall save. That is an
efficiency controller, not a quality win over current
search.

Neighbors who published in this class showed **large**
cost cuts at matched quality (CachedSearch 37% off,
LatSearch up to 79%, EFD 2.6× vs retry) or a quality
win over do-nothing *and* a cheaper retry. A 13% save
that loses 4 Dyn clips will not survive a “why not
Always / why not CachedSearch / why not ToF prune”
review.

## What would make a paper

1. **Reframe.** Headline = seed search lifts Dyn% on long
   causal V2V (32.8% → 50.8%). Gate is the analysis:
   a prefix probe recovers most of it. Always stays the
   ablation, not the thing we fail to beat.
2. **A real cheapen** that keeps Dyn% and drops cost
   toward Rolling (search-early, prune k, Video-T1).
   Then the hold-out + prune is a method.
3. **A skip-set quality win.** If Always *hurts* the 38
   skips (N=32 subject 0.687 vs Pseudo 0.701), show it
   per-video on 128. Then the gate is safety, not a
   discount. Cite-128 subject is a tie — do not claim
   this until the 38-clip table exists.

Do not write “we introduce gating.” Write “we use the
observed opening as a pseudo-label for whether seed
search will help the unseen tail.” That sentence is
still unoccupied. It is not enough without (1)+(2) or
(1)+(3).
