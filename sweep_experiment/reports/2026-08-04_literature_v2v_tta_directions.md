# Literature review — per-video TTA gating, steering-vector placement, streaming long-horizon
**Date:** 2026-08-04 · **Author:** agent (deep web review) · **Status:** decisions memo

## Why this memo
AdaSteer parameter-TTA (a single learned δ on the AdaLN/timestep embedding) is
**population-null** on PSNR / FVD / VBench in *every* regime we have (short- and
long-horizon Panda, long-horizon UCF). ~40% of videos benefit per-video, but (a)
that benefit is **unroutable** from deploy features, (b) the oracle headroom is a
**max-over-noise artifact**, and (c) FVD gets **worse** under TTA in long-horizon.
Three questions follow, each answered from the current literature:

1. How do other teams decide **per-video / per-frame whether to adapt** in V2V /
   video continuation? (Our final-loss probe was unpredictive — did we probe the
   wrong signal?)
2. **Where** should a learned vector be inserted, in theory and in practice?
   (Is our global-AdaLN δ in the wrong place?)
3. Is **streaming** (evolving the bias chunk-by-chunk) the right framing for
   long-horizon?

---

## Theme 1 — per-video "when-to-adapt" probes (V2V / continuation)

The field's gates are **not** the final self-supervised loss (which we showed is
unpredictive). They are cheaper, GT-free, and read *during* generation:

| Paper (venue/year) | Signal used to decide / correct | Reported effect | Relevance to us |
|---|---|---|---|
| **Early Failure Detection & Intervention in Video Diffusion** (arXiv 2603.14320, 2026) | Decode a **Latent→RGB preview (19.7 ms, 200× faster than the full decoder)** at each denoising step; an **intermediate alignment score s_k (k≈T−10)** predicts the final score s_0*. Dynamic detector aggregates a short run of intermediate scores → success/failure by step ~10–11 of 50. | Sample-adaptive failure detection early in denoising | A **cheap intermediate-denoising preview** predicts final quality — a far better gate feature than our final loss. |
| **Forget, Anticipate and Adapt: TTT for Long Videos** (arXiv 2606.26515, 2026) | **Anticipative Head**: predict next frame from current, compare to the real next frame; adapt **only** when the pixel "surprise" exceeds a learned threshold. Adaptive Window skips TTT on temporally-similar frames. Trainable from as few as **1** long video. | Makes TTT tractable on hours-long video by adapting selectively | A concrete **surprise gate**: adapt only high-surprise videos/chunks. Directly matches "only ~40% benefit." |
| **TANGO — Test-Time Noise Guided Adaptation** (arXiv 2607.15849, 2026) | Uses the denoiser as a **critic of its own predicted noise**; measures deviation of predicted noise from an expected Gaussian to detect "terminal points" (manifold departures); uses noise guidance as a **test-time optimization target**. | **−28.3% FVD** on autoregressive video generation | The **only** method in this space with a large reported FVD gain, in *exactly* our AR-continuation setting. Predicted-noise gaussianity = GT-free gate feature **and** optimization target. |
| **Pathwise Test-Time Correction** (arXiv 2602.05871, 2026) | Training-free: re-anchor to earliest-frame context S0 at the stage where global layout stabilizes, then **re-noise + resample** so the correction is smoothly assimilated (avoids flicker). | Extends stable AR generation from a few s to **>30 s** | Correction (not parameter-TTA); a strong long-horizon **baseline** to compare against. |
| **VDS-TTT** (LLM, 2505.19475) / **SAFER** (2606.22351) / **GD-Adapt** (ACL-F 2026) | Generate N candidates → **verifier** scores → adapt/keep only above threshold; reliability via ensemble uncertainty + manifold adherence; **combine multiple verifiers**. | Robust selective adaptation | Our best-of-k is exactly a verifier problem — consensus: **multiple verifiers + reliability weighting**, not a single metric. |

**Takeaway (reframes our null probe):** we probed the wrong signal. The final
self-sup loss carries no per-video signal, but the literature's gates —
(i) an intermediate-step decoded preview + quality critic, (ii) next-frame
surprise, (iii) predicted-noise gaussianity — are cheaper *and* GT-free *and*
demonstrably predictive. Re-running our gate on these signals is the cheapest
high-value next step.

---

## Theme 2 — where to insert the vector (the big architectural finding)

**Consensus across steering + video-customization literature:**

- **Activation steering (LLMs; masked-diffusion LMs, arXiv 2512.24143):** semantic
  concepts are most linearly accessible in the **middle-to-late residual stream
  (~60–75 % of depth)**. A **single early or single late layer is ineffective**; a
  **contiguous mid-late band** works best; the **residual stream (esp. post-MLP)**
  dominates other submodules. Default heuristic when sweeping is too costly: inject
  at **~60 % depth**.
- **Video customization** (CustomTTT AAAI'25; Follow-Your-Motion 2506.05207; B-LoRA):
  **appearance and motion are controlled by distinct, localized layers/heads** —
  appearance by early-mid **spatial** layers (e.g. i=2,6), motion by **temporal**
  layers/heads (e.g. i=2,4–5); B-LoRA isolates content vs style in **2 specific
  blocks (4,5 of 11)**; motion-specialized **attention heads** can be steered with
  **no parameter updates** at all.

**What OUR AdaSteer does** (`delta_experiment/scripts/run_delta_a.py`,
`DeltaAWrapper`): it adds **one global δ to the timestep/AdaLN embedding**
(`t_embedder` output), which is **broadcast identically to every block** through
AdaLN modulation. That means our intervention is:
- **layer-agnostic** — cannot target the concept-rich mid-late band;
- a **global modulation of every block's norm**, not a residual-stream edit;
- **unable to separate appearance vs motion.**

Per the literature this is close to the **worst-case** insertion point — which is
the strongest mechanistic explanation yet for why AdaSteer is a null lever. It is
not that "TTA doesn't help"; it may be that **we injected in the one place the
literature says barely moves anything.**

**Actionable ablation (high upside):** modify `DeltaAWrapper` to inject
**per-block residual-stream biases on a mid-late band** (~60–75 % of LongCat's DiT
depth — confirm block count on cluster) via forward hooks on `dit.blocks[i]`
instead of `t_embedder`, and test an **appearance/motion split** (spatial vs
temporal submodule). Clean 3-way ablation: `global-AdaLN δ` (current) vs
`mid-late-band residual δ` vs `appearance/motion-split δ`, on a small OOD-stratified
N first.

---

## Theme 3 — streaming long-horizon (evolving bias, chunk-by-chunk)

The user's "update the bias one small chunk at a time" intuition is well supported:

| System | Mechanism | Note |
|---|---|---|
| **Self-Forcing** (NeurIPS 2025) | AR rollout w/ KV-cache **during training** to close train-test gap; rolling KV-cache for extrapolation. | Real-time; but **degrades beyond trained horizon** (their own 30 s sliding-window drifts). |
| **LongLive** (2509.22622) | Streaming long **tuning**: roll out own generation, apply DMD only to the **newest chunk**; **frame-sink** (keep first-frame-chunk tokens as a global anchor); KV-recache on prompt switch. | Local supervision + first-chunk anchor. |
| **Rolling Forcing** | Denoise a **rolling window** jointly with noise increasing toward the future; one clean frame exits per pass; **attention sink** of initial frames. | Less multi-minute drift than Self-Forcing/CausVid. |
| **Stream-T1** (2026, training-free) | On a **frozen** streaming model: (a) **noise propagation** (slerp previous chunk's noise into the new chunk's init), (b) **reward pruning** of candidate chunks (short image-reward + long video-reward over a 10-chunk window, dynamic weighting: early favors frame quality, late favors history consistency), (c) **memory sinking**. | Closest match to "evolve locally, weight later chunks toward history." |

**Takeaway:** frame an **evolving AdaSteer** as Stream-T1-style local adaptation —
re-fit the (mid-late-band) bias per chunk on the **most-recent observed frames**,
with a **decay/anchor to the first chunk** (frame sink), and use **predicted-noise
gaussianity (TANGO)** as the per-chunk trigger. This unifies Themes 1–3.

---

## Recommended next experiments (ranked by expected value / cost)

1. **Re-probe the gate with the right signal** (cheap, mostly offline): predicted-
   noise gaussianity (TANGO) / intermediate-step preview score / next-frame
   surprise — instead of final loss. If any correlates with per-video FVD/PSNR
   benefit, the ~40 % gate becomes deployable.
2. **Vector-placement ablation** (medium; highest upside): residual-stream δ on a
   mid-late band, + appearance/motion split, vs the current global-AdaLN δ. May
   convert AdaSteer from null to a real lever. Small OOD-stratified N first.
3. **TANGO-style noise guidance as an FVD lever** (medium): we already have the
   differentiable sampler (`comparison_methods/scripts/savi_dno_longcat.py`); add a
   predicted-noise-gaussianity guidance target. Only lit method with a large FVD
   gain in our exact AR-continuation setting.
4. **Streaming evolving-bias** (larger): only if (2)/(3) show life; chunkwise local
   re-fit + first-chunk sink (Stream-T1 framing).
5. **Best-of-k** (jobs 15284148–155 running) → extend `analyze_bestofk_headroom.py`
   to FVD + winnable VBench dims with multi-verifier reliability weighting.

## References (arXiv / venue)
- Early Failure Detection & Intervention in Video Diffusion — 2603.14320 (2026)
- Forget, Anticipate and Adapt: TTT for Long Videos — 2606.26515 (2026)
- TANGO: Test-Time Noise Guided Adaptation — 2607.15849 (2026)
- Pathwise Test-Time Correction — 2602.05871 (2026)
- CustomTTT — AAAI 2025 (2412.15646); Follow-Your-Motion — 2506.05207; B-LoRA
- Activation Steering for Masked Diffusion LMs — 2512.24143; VDS-TTT — 2505.19475; SAFER — 2606.22351
- Self-Forcing — NeurIPS 2025 (2506.08009); LongLive — 2509.22622; Rolling Forcing; Stream-T1 (2026)
- Video-T1 — ICCV 2025 (2503.18942)
