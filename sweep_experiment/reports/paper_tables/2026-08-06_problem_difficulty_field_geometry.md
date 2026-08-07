# Problem-difficulty audit: are we setting LongCat too easy a task?

**Date:** 2026-08-06 · **Type:** literature / methodology memo (paper-citable table)
**Motivation:** Repeated null/tiny effects (AdaSteer, TANGO EXP3, placement EXP2)
across pixel / FVD / VBench. Hypothesis (raised by W.C.): the base model is *too
strong* and our task geometry is *too easy*, so headroom is small by construction.
This memo tabulates what comparable teams actually do (base model + frame geometry
+ eval), and concludes the hypothesis is well-supported.

---

## 1. What the field uses vs. what we use

| Work (venue) | Base model | Task | Cond frames | Generated horizon | Res / fps | Data | How they create difficulty / measure |
|---|---|---|---|---|---|---|---|
| **STAS** (2026, arXiv) | Wan2.1-1.3B; CogVideoX-5B; Wan2.2-TI2V-5B | **T2V from scratch** | 0 | **49–81 frames** | 480–704p | VBench prompts | VBench 16-dim × 5 seeds (4,700+ vids); +0.37 total; **explicitly notes gains dilute in video-level averages, concentrate at cross-chunk seams** |
| **History-Guided / DFoT** (ICML 2025, 2502.06764) | own DFoT (fine-tuned foundation) | prediction + rollout | flexible (variable history) | **64 frames, sliding-window rollout** | — | Kinetics-600, RE10K | "challenging setup that requires outstanding consistency to avoid blowing up"; OOD history; FVD best at small guidance ω=1.5 |
| **AID** (ICCV 2025) | I2V diffusion adapted | instruction video prediction | 1–2 | multi-frame | — | SSv2, Bridge, Epic-100 | hard action dynamics; FVD/KVD (K400 I3D) over 2,048–9,342 samples |
| Classic video prediction | MCVD / PVDM / VideoSDE / etc. | prediction | BAIR 1 · K600 5 | 11–15 | 64px | BAIR, K600, UCF-101 | random robot motion / diverse action; **FVD avg over 100 runs (noisy at small N)** |
| **Pathwise TTC** (2602.05871), **Rolling Forcing** (2509.25161), **BAgger** (2512.12080), **Self-Forcing** (2506.08009), **Meta-ARVDM** (2503.10704) | distilled AR DiTs (Self-Forcing family) | **streaming long-horizon** | rolling / autoregressive | **30 s – minutes** | — | game/world-model + T2V | **error accumulation / drift**; drift signatures = over-saturation, over-smoothing, motion-diversity loss; Meta-ARVDM: standard metrics FAIL → needle-in-haystack eval |
| **OURS** | **LongCat-Video 13.6B (RLHF, continuation-pretrained)** | continuation | 14 | **14 frames, single chunk (~0.5–1 s), in-domain Panda** | 480p | Panda (in-domain) | video-level **mean** FVD/PSNR/VBench over one short clip |

## 2. Why our setup is easy — two independent reasons

**(a) We picked the model built to make our exact task trivial.** LongCat-Video
(arXiv 2510.22200, Meituan, 13.6B, RLHF/GRPO) is **natively pretrained on
video-continuation**; its headline capability is *"minutes-long videos without
color drifting or quality degradation."* Native long-gen recipe: `num_segments=11`,
`num_frames=93`, `num_cond_frames=13` overlap, KV-cache. We handed this drift-
resistant model the **easiest slice of its home task**: a single 14→14 in-domain
continuation. It saturates → little for any TTA/steering to fix.

**(b) The field's headroom comes from difficulty knobs we removed:**
- **Long-horizon autoregressive rollout** (the dominant hard setting): TTC / Rolling
  Forcing / BAgger / Self-Forcing / Meta-ARVDM all target **error accumulation over
  30 s–minutes**. That drift is the headroom correction methods feed on. Our single
  14-frame chunk has ~0 accumulation.
- **Weaker / smaller base models:** STAS steers Wan2.1-**1.3B** and still gets only
  +0.37 VBench. On a saturated 13.6B RLHF model, expect less.
- **Harder / OOD dynamics + long generation:** DFoT = 64-frame Kinetics-600 rollout;
  AID = SSv2/Epic action; classic VP = random/ diverse motion.
- **Localized metrics:** STAS shows effects live at cross-chunk seams and are
  **diluted by video-level averaging** — which is exactly our reporting.

## 3. Our own prior data already agrees

Project state (AGENTS.md §3): AdaSteer is *"per-video net-positive in OOD
long-horizon scenarios; saturated at the population level for in-domain short
horizon."* Same story: **headroom lives in long-horizon / OOD, not the regime we've
been sweeping.**

## 4. Critical implementation note discovered during this audit

Our "long-context" path (`ttc_longcat.py`, 93 frames) generates **all 79 gen frames
in a single diffusion call** — it is NOT true autoregressive chaining. So the
cross-chunk exposure-bias accumulation that the whole long-video literature studies
**never occurs in our current pipeline**. Any "long-horizon" claim we make today is
really "one big single-shot continuation," not streaming rollout.

## 5. Recommendation (before any new build/direction switch)

Relocate evaluation to where headroom demonstrably exists and where the field
competes:
1. **True long-horizon autoregressive rollout** on LongCat (feed the model's own
   frames back as conditioning; 8–11 chained chunks, ≈minutes). Measure **per-chunk
   degradation (drift curves)**, not a single short chunk, not video-level means.
2. **Second, weaker base model** (Wan2.1-1.3B or CogVideoX-5B) as a contrast testbed.
3. **OOD / high-motion** continuation instead of in-domain Panda short clips.
4. **Localized / drift metrics** (cross-chunk seam consistency, sharpness/saturation/
   motion trends over chunk index) alongside aggregates.

**Decisive cheap first step:** a NOTTA drift-curve diagnostic — run true
autoregressive rollout (8–11 chunks) on a handful of clips and plot quality vs chunk
index. Degradation ⇒ headroom found (every intervention we've discussed gets room).
No degradation ⇒ LongCat is too strong for this framing ⇒ switch base model. Either
outcome resolves the question empirically. Diagnostic build: this same dated set
(`diag_longhorizon_drift`).

## 6. Status update — 2026-08-07: diagnostic built

The `diag_longhorizon_drift` build named in §5 now exists and is pushed:
- `delta_experiment/scripts/diag_longhorizon_drift.py` — NOTTA true-autoregressive rollout;
  per-chunk GT-free drift signals (sharpness / colorfulness / temporal_motion / seam ratio) +
  PSNR/SSIM/LPIPS where GT overlaps; per-signal slope + %-change verdict in `summary.json`.
- `delta_experiment/sbatch/run_longhorizon_drift.sbatch` + `submit_longhorizon_drift.sh` —
  H200, account `torch_pr_36_mren`, chunkable (`START_VIDEO_IDX`/`CHUNK_SIZE`).
- `scripts/plot_drift_curves.py` — `summary.json` -> per-metric + headline drift-curve PNGs.

Run: `bash delta_experiment/sbatch/submit_longhorizon_drift.sh` (defaults N=24, chunks=8,
cond=14/frames=28/gsf=48, same geometry as the AdaSteer/placement/EXP3 runs so the drift curve is
directly comparable). Read the drift verdict in `summary.json`: **degradation over chunk index =>
headroom found**; **flat => switch base model.**

## 6. Status update — 2026-08-07: diagnostic built

The `diag_longhorizon_drift` build named in §5 now exists and is pushed:
- `delta_experiment/scripts/diag_longhorizon_drift.py` — NOTTA true-autoregressive rollout;
  per-chunk GT-free drift signals (sharpness / colorfulness / temporal_motion / seam ratio) +
  PSNR/SSIM/LPIPS where GT overlaps; per-signal slope + %-change verdict in `summary.json`.
- `delta_experiment/sbatch/run_longhorizon_drift.sbatch` + `submit_longhorizon_drift.sh` —
  H200, account `torch_pr_36_mren`, chunkable (`START_VIDEO_IDX`/`CHUNK_SIZE`).
- `scripts/plot_drift_curves.py` — `summary.json` -> per-metric + headline drift-curve PNGs.

Run: `bash delta_experiment/sbatch/submit_longhorizon_drift.sh` (defaults N=24, chunks=8,
cond=14/frames=28/gsf=48, same geometry as the AdaSteer/placement/EXP3 runs so the drift curve is
directly comparable). Read the drift verdict in `summary.json`: **degradation over chunk index =>
headroom found**; **flat => switch base model.**
