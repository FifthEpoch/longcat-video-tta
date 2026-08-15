# Field standard for long-horizon / streaming video generation

**Date:** 2026-08-15 · **Type:** literature memo (paper-citable) · **Decision:** switch
base model + eval off LongCat-13.6B / Panda short-continuation onto the field's
1.3B streaming testbed.

**Inclusion rule:** peer-reviewed at a reputable venue (CVPR / ICCV / ECCV / NeurIPS
/ ICML / ICLR) in 2024–2025, or the official eval suite those papers report
(VBench, CVPR 2024). Unpublished / lightly-cited arXiv follow-ons (Pathwise TTC,
Rolling Forcing, Self-Forcing++, BAgger) are **not** used as standards here.

---

## 1. What published long-horizon / streaming papers actually use

| Work (venue) | Base model (size) | Task | Generated horizon | Training / source data | How they measure vs baseline |
|---|---|---|---|---|---|
| **CausVid** (Yin et al., **CVPR 2025**) | **Wan2.1-T2V-1.3B** distilled to a 4-step causal AR student | streaming T2V / I2V / V2V, KV-cache | **5–10 s** main table; **30 s** long-video table vs FIFO / Pyramid Flow / StreamingT2V | MixKit (~6K clips) as a distillation toy; teacher is Wan2.1 | **VBench** quality+semantic on MovieGen first **128** prompts; **VBench-Long** total (84.27); human pairwise vs CogVideoX / Pyramid Flow / MovieGen; latency/FPS |
| **Self-Forcing** (Huang et al., **NeurIPS 2025** Spotlight) | **Wan2.1-T2V-1.3B** causal AR (built on CausVid) | streaming T2V, KV-cache, real-time | trained on **5 s**; they explicitly show quality **degrades when extrapolating to 10–30 s** | data-free DMD (no video needed after ODE init); MixKit/CausVid init | VBench on 5 s; qualitative long-rollout degradation; FPS on 4090/H100 vs CausVid / Wan / MAGI |
| **Pyramid Flow** (Jin et al., **ICLR 2025**) | own **2B** MM-DiT (SD3-style) | AR flow-matching T2V / I2V | **5–10 s** @ 768p 24 fps | WebVid-10M + OpenVid-1M + Open-Sora Plan (~10M shots after filter) | **VBench** (16 dim) + **EvalCrafter** + human preference |
| **FIFO-Diffusion** (Kim et al., **NeurIPS 2024**) | pretrained T2V (VideoCrafter2 in demos) | training-free infinite-length sampling | **100 frames** (~4–6 s at 16–24 fps); conceptually unbounded | none (inference-only) | vs FreeNoise / Gen-L-Video / LaVie+SEINE: temporal consistency, visual quality, motion (qual + VBench-style tables in follow-on comparisons) |
| **FreeNoise** (Qiu et al., **ICLR 2024**) | **VideoCrafter** (trained on **16 frames**) | training-free longer T2V | **64 frames** (4× train length); also 100-frame comparisons | none (inference-only) | **FVD / KVD** of long-gen *subset* vs the model's own short-gen; **CLIP-SIM** adjacent-frame consistency |
| **One-Minute TTT** (Dalal et al., **CVPR 2025**) | **CogVideoX-5B** + TTT layers → 7.2B | long-context T2V storyboards | **18 s** elimination; **63 s / ~1 min** headline | **~7 h Tom & Jerry** (81 episodes × ~5 min), human storyboards; multi-stage 3→9→18→30→63 s | **human Elo** on 100 videos/method (text following, motion, aesthetics, temporal); they do **not** lead with FVD/PSNR |
| **History-Guided / DFoT** (Song et al., **ICML 2025**) | own DFoT (DiT-XL class) | video **prediction** + sliding-window rollout | **64-frame** Kinetics rollout; “extremely long” qualitative | **Kinetics-600** (128², ~10 s sources); RE10K; Minecraft | **FVD** (headline; best 170.4); **VBench** quality/consistency/dynamics; LPIPS only on deterministic robot tasks |
| **CogVideoX** (Yang et al., **ICLR 2025**) — *base-model paper, not a long-horizon method* | 2B / **5B** | bidirectional T2V | **~10 s** (161 frames @ 16 fps, 768×1360) | in-house web video + captioner | selected **VBench** dims (action, scene, dynamic degree, objects, style); FVD/CLIP4Clip ablations on WebVid-500 |
| **VBench / VBench-Long** (Huang et al., **CVPR 2024** + official Long extension) | n/a (eval suite) | eval only | short: ~16 fr / ~2 s; **Long: 5–10 s+** Sora-like | **946** standard prompts (16 dims); Long uses a slow/fast consistency protocol | **Quality:** subject/background consistency, flicker, motion smoothness, dynamic degree, aesthetic, imaging. **Semantic:** 9 dims. Human-aligned. **This is what CausVid/Self-Forcing/Pyramid Flow report.** |
| **OURS (current)** | **LongCat-Video 13.6B** (RLHF, continuation-pretrained) | 14→14 then native 13/80 AR continuation | 30–60 s native, **N=8** | Panda-70M in-domain short clips | hand-crafted GT-free drift + PSNR/SSIM/LPIPS (GT dies after 1–2 chunks) + paired \|drift\| |

---

## 2. Three facts that decide the switch

**(a) The streaming/long-horizon testbed is Wan2.1-1.3B, not 13B-class models.**
CausVid (CVPR 2025) and Self-Forcing (NeurIPS 2025) — the two papers that *define*
the current streaming-AR setting — both start from **Wan2.1-T2V-1.3B**. Pyramid
Flow is 2B. FreeNoise/FIFO use VideoCrafter-class models trained on 16 frames.
The only 5B-class long-horizon paper (One-Minute TTT, CVPR 2025) is a
domain-specific storyboard study and still calls 5B a capability bottleneck.
LongCat 13.6B is an outlier: too big to get N, and pretrained to *not* drift.

**(b) Nobody evaluates long generation on Panda short-clip continuation.**
Eval is **prompt suites**, not GT-aligned pixel metrics:
- VBench 946 prompts (5 s) + VBench-Long (10 s+)
- MovieGen first 128 prompts (CausVid long table)
- Kinetics-600 64-frame rollouts when the task is *prediction* (DFoT)
Source videos in training sets are typically **3–10 s shots** (MixKit, WebVid,
OpenVid, Kinetics). One-Minute TTT is the exception (5-min cartoons, used as
*training* for 63 s gen, evaluated by humans — not FVD).

**(c) The field's vs-baseline metrics are VBench-Long quality dims + human, not PSNR.**
- Streaming T2V: VBench / VBench-Long (subject/background consistency, flicker,
  motion smoothness, imaging/aesthetic, dynamic degree) + human pairwise/Elo + FPS.
- Training-free longer-gen (FreeNoise): FVD/KVD of long vs own short + CLIP-SIM.
- Prediction (DFoT): FVD on Kinetics-600, VBench as auxiliary.
- PSNR/SSIM/LPIPS appear only when a single GT future exists (prediction / robot).
  They are **not** the headline for open-ended long T2V — which is why our 60 s
  LPIPS cell is GT-limited by construction.

---

## 3. Recommended new stack (smallest model that is still the field standard)

| Knob | Switch **from** | Switch **to** | Why |
|---|---|---|---|
| **Base model** | LongCat-Video 13.6B | **Wan2.1-T2V-1.3B** (official weights). For true streaming AR, prefer the **CausVid or Self-Forcing 1.3B causal checkpoint** (same backbone, KV-cache, already the published streaming baseline). | 10× smaller; every 2025 streaming paper is here; headroom exists (Self-Forcing still degrades past 5 s). |
| **Task** | in-domain 14→14 / native 13/80 continuation on Panda | **Stay in continuation / I2V** (condition on a real image or short prefix, roll out AR). Horizons: **5 s / 10 s / 30 s**. T2V-from-scratch is *not* required — it was only the default task of the 1.3B streaming papers. | Exposure bias is a *conditioning* problem; I2V/continuation is the matching task. CausVid (CVPR 2025) already reports streaming I2V; VBench-I2V is the official suite. |
| **Eval set** | Panda-70M preview clips | **VBench-I2V image suite** (subject/background/camera splits) for conditioned gen; optionally MovieGen-128 images for 10–30 s. Kinetics-600 64-frame rollouts only if we also want a DFoT-style prediction/FVD table. | Field-standard *conditioned* eval. Do not switch to T2V prompts just to match CausVid's T2V table. |
| **Headline metrics** | PSNR/SSIM/LPIPS + hand-crafted drift | **VBench-Long quality 7:** subject consistency, background consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic quality, imaging quality. Keep our GT-free drift curves as a *diagnostic* (in-loop verifier), not the paper headline. | Field-standard, GT-free, defined at any length. We already have `eval_vbench.py`. |
| **N** | 8 videos × 12 chunks | **≥100 prompts** at 5 s; **≥32–128** at 10–30 s (MovieGen-128 is the published long set). | 1.3B + 4-step CausVid/Self-Forcing makes this cheap vs LongCat 50-step 13.6B. |

**Do not switch to:** CogVideoX-5B (One-Minute TTT only; 4× our size budget),
VideoCrafter2 (FreeNoise/FIFO era; superseded as a streaming testbed), or
Kinetics-600 as the *only* eval (that is the prediction setting, not streaming T2V).
Kinetics-600 is a valid *second* table if we also want a DFoT-style FVD number.

---

## 4. What this does *not* change

- The **method** stays sampling-space test-time control (best-of-N verifier +
  gated TTC). Those actuators were designed to be backbone-agnostic.
- The **credibility gates** stay: verifier vs random-pick, paired sign-flip on
  whatever headline metric we adopt (now VBench quality dims / drift, not PSNR).
- LongCat results remain in the paper as the “saturated 13B continuation model”
  negative / difficulty-audit evidence (Slides 1a–1c).

---

## 5. Immediate next engineering step

1. Pull Wan2.1-T2V-1.3B (and, if we want streaming AR out of the box, the
   Self-Forcing or CausVid 1.3B causal weights — both public).
2. Run a **NOTTA / no-control** 5 s and 30 s **VBench-I2V** smoke on ~16
   conditioning images (not T2V-from-scratch) to confirm the backbone drifts
   past 5 s under visual re-conditioning.
3. Port `bestof` + `ttc` onto that sampler (same verifier; VBench quality dims
   as the offline eval).
4. Only then scale N. Do **not** keep spending H200 hours on LongCat TTC v2
   except to finish the already-submitted w0 smoke.
