# AdaSteer Analysis Log

**Purpose:** Append-only log of decisions, findings, and narrative changes
during paper preparation. Every meaningful experimental conclusion or
methodology decision goes here, dated and tagged. NEVER edit past entries
(rebut them with a new entry instead).

**Format:**
```
## YYYY-MM-DD — Short title
**Tags:** [methodology|finding|decision|negative-result|paper-narrative]
**Owner:** name
**Refs:** files / commits / cluster paths

Body...
```

---

## 2026-06-13 (later) — Implementation: VAE-Decoder-Only TTA (Modification 2; post-D1 PIVOT)
**Tags:** recipe-modification, vae-decoder-tta, implementation, post-d1-pivot
**Refs:**
- `delta_experiment/scripts/run_vae_decoder_tta.py` (new ~520 LOC) — standalone per-video TTA runner that freezes the DiT entirely and adapts only `vae.decoder.parameters()` via AdamW (default lr=1e-5, 10 steps, grad-clip=1.0, weight-decay=0) on the pure pixel-MSE round-trip loss `MSE(VAE.decode(VAE.encode(pixel_frames_train)), pixel_frames_train)`. Encoder stays frozen and is run once per video under `no_grad`. Decoder is snapshotted once at model-load time and `load_state_dict`-restored from the snapshot at the end of every video so adapter state never leaks across the stream; decoder drift `||Δw||_2` (current vs snapshot) is logged per-video as a movement sanity-check. Optional LPIPS auxiliary wired (`--vae-tta-lpips-weight`, default 0.0 = byte-identical-to-MSE-only). During TTA the DiT and text_encoder are off-loaded to CPU and restored before inference — TTA peak GPU is dominated by the decoder forward+backward (~10–15 GB at 48 frames × 480p on H200, far below the DiT-TTA peak).
- `delta_experiment/scripts/common.py` — single 1-line addition: `_METHOD_SLUG["vae_decoder_tta"] = "vae-dec"` so the post-run video-rename step picks up the new method.
- `sweep_experiment/sbatch/run_sweep.sbatch` — new `vae_decoder` dispatch case wired to the new runner via env vars `VAE_TTA_STEPS / VAE_TTA_LR / VAE_TTA_LPIPS_WEIGHT / VAE_TTA_GRAD_CLIP / VAE_TTA_WEIGHT_DECAY`; `METHOD` error message + valid-values list updated to include `vae_decoder`. Existing methods are unchanged — the new case is additive and does not touch the existing `lora` / `delta_a` / `delta_b` / `delta_c` / `norm_tune` / `film` / `full` flows.
- `sweep_experiment/sbatch/submit_smoke_vae_decoder_tta.sh` (new) — single-chunk × 100-video smoke wrapper on Panda 1000v, default `VAE_TTA_LR=1e-5`, run-id `VAE_DEC_TTA_LR1e-5`, sbatch wall 12 h (~3–5 GPU h expected). Mirrors the Mod 1 `submit_smoke_x0_loss.sh` pattern (same frame geometry: 28 frames, gen_start=48, num_cond=14, tta_total=48, tta_context=14, 50 inference steps, guidance 4.0) so the resulting per-video ΔPSNR / ΔLPIPS / ΔFVD against NOTTA and LORA_R8_TTA on the same `chunk_0` videos are directly comparable. Wrapper supports `DRY_RUN=1` for inspection without firing.
- `sweep_experiment/reports/LITERATURE_tta_recipe_modifications_2026-06-12.md §3.2` — implementation status flipped from "deferred" to "implemented" with the same template the Mod 1 entry uses (commit-hash discovery via `git log --grep`, default-knob breakdown, smoke-wrapper pointer, decision rule, LPIPS-decoded variant deferred to follow-up).
- `sweep_experiment/reports/INDEX.md` "Implemented but not yet run (recipe modifications)" — new row for Mod 2, sibling to the Mod 1 row, with explicit pointers to the runner, dispatch, and wrapper plus the §2.3 4-video beneficiary cohort as the primary scale-up signal.

**Rationale (why Mod 2 next, not Mod 3/4/5):** The D1 smoke-test for Modification 1 (anchor-frame x0 consistency loss; commit `870aea3`) returned median ΔPSNR ≈ +0.0093 dB, below the +0.05 dB threshold from LITERATURE doc §3.1's falsification criterion. The "loss formulation is the binding constraint" hypothesis is rejected. The next most-likely structural hypothesis from the literature deep-dive is the user's own (REVIEW §4.1): the DiT is too capable to need adaptation, and the VAE round-trip is the bottleneck for per-video continuation fidelity. Mod 2 is exactly the experiment that tests this: it adapts ONLY the VAE decoder, with the DiT frozen, on the cleanest possible per-video supervisory signal (round-trip reconstruction of the visible frames — clean ground truth, no diffusion noise, no flow-matching σ-weighting). The smoke-test plus its falsification rule (ΔPSNR > +1.0 dB on ≥3 of {`panda_0461`, `_0555`, `_0862`, `_0431`} → scale up) cleanly rules in or out the "VAE is the bottleneck" claim with one chunk of compute (~3–5 GPU h).

**Why pure-MSE, not MSE+LPIPS, first:** parameter parsimony. Two scalar dependencies on the result (LR + LPIPS-weight) double the smoke-test search space without doubling the diagnostic power. If pure-MSE clears the smoke threshold, layering LPIPS adds perceptual quality on top; if pure-MSE doesn't, LPIPS won't either (it acts on the same decoder output the MSE already constrains). The LPIPS auxiliary is wired (gated on `lpips` package availability with a graceful fallback) so the follow-up is a one-flag flip.

**Verification before scale-up:** the smoke-test fires when the cluster is available (`bash sweep_experiment/sbatch/submit_smoke_vae_decoder_tta.sh`). Decision rule per LITERATURE doc §3 Mod 2 falsification criterion:
- **PRIMARY scale-up:** ΔPSNR > +1.0 dB on ≥3 of {`panda_0461`, `panda_0555`, `panda_0862`, `panda_0431`} (the §2.3 beneficiary cohort). If yes, run 10-chunk × {1e-6, 1e-5, 1e-4} LR sweep ≈ 30 GPU-h (per LITERATURE doc §4 priority-2 row).
- **SECONDARY scale-up:** aggregate median |ΔPSNR| > 0.5 dB across the 100-video chunk → also scale up; optionally layer LPIPS arm (`VAE_TTA_LPIPS_WEIGHT=0.1`) as one extra wave.
- **Null outcome:** neither cohort triggers → the VAE is not the binding constraint either; pivot to Mod 3 (augmentation-consistency, MEMO/TTT-MAE) or document as a deeper negative result with a paper subsection on "where TTA fails on video DiTs and why" (the saturation reproduces across three independent recipe families: weight-target axis [DiT-LoRA vs TinyLoRA-SVD vs AdaSteer-AdaLN], loss-formulation axis [Mod 1: anchor-x0 added vs not], and now adaptation-target axis [DiT vs VAE-decoder]).

**Implementation safety:** the new runner is standalone (no edits to `run_delta_a.py` / `run_lora_tta.py` / `run_tinylora.py`), the new dispatch case is additive, and the single 1-line `common.py` edit only adds a key to a dict (not used by any existing method). The runner's per-video decoder-restore is the critical correctness invariant — if it ever fails to restore, the second video would see an off-pristine decoder and subsequent videos compound the drift. Restore is `load_state_dict(snapshot)` with the snapshot stored in the decoder's native dtype on CPU, so restoration is byte-identical to the post-model-load state. The restore is also wrapped in `try/finally`-equivalent logic in the error path so a failing video does not poison the next one. Awaiting cluster availability for empirical verification.

---

## 2026-06-13 — Wrapper landed: TTOM iteration-saturation sweep (Track D Wave D2 fire-ready)
**Tags:** runbook, track-d, d2, ttom-control
**Refs:**
- `sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh` (new) — 3 methods (ADA / LORA_R8_TTA / TL_BARE_R2) × 5 TTA-step values ({10, 20, 40, 80, 160}) × Panda 1000v chunk_0 = 15 jobs; per-method TTA-step env-var verified against the case statements in `sweep_experiment/sbatch/run_sweep.sbatch` (`delta_a` → `DELTA_STEPS`, `lora` → `NUM_STEPS`) and `delta_experiment/sbatch/run_tinylora.sbatch` (`tinylora` → `TTA_STEPS`); all other knobs frozen at the headline Panda recipes from `submit_standard_1000v_chunked.sh` so the only changing variable is the TTA-step count.
- `sweep_experiment/reports/RUNBOOK_friday_morning_2026-06-12.md` §4 D2 — status flipped from BLOCKED to READY; `**Command:**` block + new "Wrapper exists" line replace the prior "**TODO**" bullet.
- `sweep_experiment/reports/RUNBOOK_friday_morning_2026-06-12.md` §5 dependency graph — D2 row updated from "BLOCKED on submit_ttom_iteration_sweep.sh" to "READY ; fire alongside D1 + A1".
- `sweep_experiment/reports/INDEX.md` "Implemented but not yet run (recipe modifications)" — new row pointing at the wrapper.

Authorisation flow: user awoke briefly during overnight off-hours and explicitly authorised landing the D2 wrapper *without* the prior gate of "wait for D1 positive signal" — so D1, A1, and D2 can fire as a single submission burst on cluster restart. Wrapper is wrapper-only (no runner / sbatch-target changes), syntax-validated, dry-run produces 15 sbatch lines with correct run-ids (`ADA_TTA{N}`, `LORA_R8_TTA_TTA{N}`, `TL_BARE_R2_TTA{N}`) and correct method-specific TTA-step env-vars. Output dirs land alongside the headline cells (`{ADA,LORA_R8_TTA}_TTA{N}` under `panda_1000v_standard/`; `TL_BARE_R2_TTA{N}` under `tinylora_panda_1000v_standard/`) so paper-table builders pick them up naturally. Fire command: `bash sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh`.

---

## 2026-06-12 (later+3) — Runbook update: Track D (recipe-modification & TTOM control) added
**Tags:** runbook, track-d, recipe-modifications, ttom-control
**Refs:**
- [`RUNBOOK_friday_morning_2026-06-12.md`](RUNBOOK_friday_morning_2026-06-12.md) — new §4 Track D with Wave D1 (Modification 1 smoke-test) and Wave D2 (TTOM iteration-saturation sweep); §5 dependency graph updated to show Track D as a separate downstream branch ungated on Phase 0–3; sections 4→5, 5→6, 6→7, 7→8 renumbered to make room.
- Commit `870aea3` — Modification 1 implementation (anchor-frame x0 consistency loss; `sweep_experiment/sbatch/submit_smoke_x0_loss.sh` wrapper landed).
- Commit `a388b8e` — TTOM positioning paper fragment (`PAPER_FRAGMENT_ttom_positioning_2026-06-12.md`; the §"Suggested control" block is the D2 spec).

Adds a Friday-afternoon work track covering two waves that can fire any time after the cluster restart, **independent of Phase 0–3 results**. D1 (Modification 1 anchor-frame x0 consistency loss smoke-test) is ready — `submit_smoke_x0_loss.sh` exists, ~2 GPU h on H200 (single chunk × 100 videos), decision rule lifted verbatim from `LITERATURE_tta_recipe_modifications_2026-06-12.md` §1. D2 (TTOM iteration-saturation sweep, 3 methods × 5 tta-steps × ~100 videos ≈ ~1500 runs ≈ ~125 GPU h serial) is spec'd from `PAPER_FRAGMENT_ttom_positioning_2026-06-12.md` "Suggested control" but BLOCKED on a missing sbatch wrapper (`sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh`), explicitly deferred until D1 produces a positive signal OR the user separately authorises the wrapper. Pure documentation update; no code, sbatch, or script changes.

---

## 2026-06-12 (later+2) — Implementation: anchor-frame x0 consistency loss (Modification 1)
**Tags:** recipe-modification, anchor-x0-loss, implementation
**Refs:**
- `delta_experiment/scripts/common.py` — `compute_flow_matching_loss_conditioned` gains optional `anchor_x0_weight: float = 0.0` parameter (default 0.0 is byte-identical to pre-patch, verified by a regression test that hits the code path with `forward_fn` stubbed) plus a `return_components: bool = False` debug flag.
- `delta_experiment/scripts/run_tinylora.py`, `delta_experiment/scripts/run_delta_a.py`, `lora_experiment/scripts/run_lora_tta.py` — new `--anchor-x0-weight FLOAT` CLI flag (default 0.0), threaded through the `optimize_tinylora` / `optimize_delta_a` / `_optimize_delta_a_batch` / `finetune_lora_on_conditioning` / `finetune_lora_batch` call sites; banner prints the resolved value; method-label suffix `+ x0 (λ=<weight>)` is shown when active.
- `sweep_experiment/sbatch/run_sweep.sbatch`, `delta_experiment/sbatch/run_tinylora.sbatch`, `lora_experiment/sbatch/run_lora_tta.sbatch` — accept `ANCHOR_X0_WEIGHT` env-var override; the `--anchor-x0-weight` runner flag is emitted only when the weight is strictly > 0 (case statement matches `0|0.0|0.00|0.000|""`) so legacy submit wrappers that don't set the env var produce byte-identical runner invocations.
- `sweep_experiment/sbatch/submit_smoke_x0_loss.sh` (new) — single-chunk `LORA_R8_TTA` smoke-test at λ=1.0 on Panda 1000v, output to `sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA_X0_W1.0/chunk_0/`. Uses the EXACT headline LORA_R8_TTA hyperparameters (rank=8 / α=16 / lr=5e-5 / 10 steps / wd=0.01 / max-grad-norm=10) from `submit_standard_1000v_chunked.sh` so the ONLY changing variable is the x0-loss term. ~2 GPU h on H200.
- `sweep_experiment/reports/INDEX.md` — new "Implemented but not yet run (recipe modifications)" section pointing at the smoke-test wrapper as the cluster-restart verification step.
- `sweep_experiment/reports/LITERATURE_tta_recipe_modifications_2026-06-12.md` — Modification 1 implementation-status line updated.

Modification 1 of 5 from the 2026-06-12 literature pass. The patch is a ≤80-LOC bolt-on to `compute_flow_matching_loss_conditioned`: at every TTA step, when `anchor_x0_weight > 0`, compute the rectified-flow x0 recovery `pred_x0 = noisy_target − σ · pred_v` on the noised target portion and add `anchor_x0_weight · F.mse_loss(pred_x0, target_latents)` to the returned scalar loss. Sigma broadcasts as `[B,1,1,1,1]` (the same per-sample broadcast shape used to build `noisy_target` two lines above) — no per-token indexing is needed even though the DiT receives per-token timesteps (cond=0, target=σ·1000) because the noise was applied at this per-sample sigma only on the target portion, so the recovery is well-defined exactly there. Per Sangare et al. CVPR 2026, the existing v-prediction MSE implicitly down-weights global-structure gradients by `(α/σ)²` at low SNR, exactly where the conditioning-on-clean-frames signal should pay off most; the x0 term restores that signal. Zero extra forward passes.

**Verification before scale-up:** the smoke-test fires when the cluster returns from maintenance (`bash sweep_experiment/sbatch/submit_smoke_x0_loss.sh`). Decision rule per the literature doc §3.1 falsification criterion: if median |ΔPSNR| > 0.5 dB on the chunk vs the headline LORA_R8_TTA chunk_0, scale to the full 4-method × 4-λ × 10-chunk sweep (~80 GPU-h). If λ=1.0 produces NaN grads OR |ΔPSNR| < 0.05 dB, the loss formulation is not the binding constraint — pivot to Modification 2 (VAE-decoder-only TTA). The byte-identical-when-off guarantee means this commit is safe to ship to main even though the cluster is offline; existing submit wrappers that don't set `ANCHOR_X0_WEIGHT` are unchanged.

---

## 2026-06-12 (later+1) — Paper fragment: TTOM positioning paragraph for related-work
**Tags:** paper-defense, ttom, positioning, related-work
**Refs:**
- [`PAPER_FRAGMENT_ttom_positioning_2026-06-12.md`](PAPER_FRAGMENT_ttom_positioning_2026-06-12.md) (new) — full draft positioning paragraph + supporting context (TTOM claim summary, ≥3 axes of difference, suggested iteration-sweep control)
- INDEX.md (new "Paper fragments" section seeded with this row)
- Source paper: TTOM (Qu et al., ICLR 2026 — [OpenReview](https://openreview.net/pdf?id=wqCwcTZsrv); [arXiv 2510.07940](https://arxiv.org/abs/2510.07940)). PDF read end-to-end via WebFetch — no reliance on secondary sources.
- Companion to: [`LITERATURE_tta_recipe_modifications_2026-06-12.md`](LITERATURE_tta_recipe_modifications_2026-06-12.md) §1 / §5 (where TTOM was first flagged as the strongest "your saturation may be real" precedent).

Draft 200-400-word positioning paragraph for the AdaSteer paper's related-work section, distinguishing our per-video reconstructive TTA setting from TTOM's iteration-axis saturate-then-degrade observation. **Strongest axis of difference:** the *supervisory signal*: TTOM optimizes a JSD against an LLM-synthesized layout proxy (no ground-truth video), whereas we optimize a flow-matching loss against the clean visible frames of the held-out video itself — a direct self-supervised signal that materially changes what "more iterations" means mechanistically. Two further axes (optimization variable: three adapter families incl. non-LoRA AdaSteer vs. their fixed rank-32 cross-attention LoRA; test-time loop scope: per-video reset vs. their cross-prompt streaming memory) round out the defense. Honest "where comparison is close" section flags the LoRA-on-cross-attention-DiT axis as one where overclaiming difference would be tactically weak — both papers are in the same small recent literature that updates DiT-attached parameters at inference time. **Specific control TTOM suggests we add:** an explicit iteration sweep on a stratified ~100-video Panda-1000v subsample (`--tta-steps ∈ {10,20,40,80,160}` × three method families × 100 videos ≈ 125 GPU-h serial — well below the ~220 GPU-h priority-1–4 recipe-modifications budget). Either outcome of that sweep is paper-defensible: crossover seen → "our saturation has the same mechanism as TTOM's"; monotonic flat → "our saturation is at the per-video noise floor, mechanistically distinct from TTOM's over-optimization saturation". No code or sbatch changes in this commit — pure paper-fragment delivery.

---

## 2026-06-12 (later) — Literature pass: TTA recipe modifications worth trying after gating Phase 0-3
**Tags:** literature, recipe-modification, paper-narrative
**Refs:**
- [`LITERATURE_tta_recipe_modifications_2026-06-12.md`](LITERATURE_tta_recipe_modifications_2026-06-12.md) (new) — 10-theme literature pass + 5 selected modifications with mechanism / cost / falsification / priority
- INDEX.md (new "Literature passes" section pointing at this file)
- Companion to: [`REVIEW_per_video_tta_suitability_2026-06-09.md`](REVIEW_per_video_tta_suitability_2026-06-09.md) (saturation evidence) and [`PLAN_gating_experiment_2026-06-11.md`](PLAN_gating_experiment_2026-06-11.md) (the gating experiment this pass slots behind)

Targeted literature pass on TTA *recipe* modifications (not gating) worth queuing behind gating Phase 0-3 RECOMMENDATION.md. Ten themes searched (TTA-for-diffusion specifically; latent-space-only TTA; anchor-frame consistency loss; MEMO/TTT-MAE augmentation-consistency; CFG-aware TTA; prompt ensembling; curriculum/annealed-timestep TTA; meta-learning/amortized TTA; continual streaming TTA; recent CVPR/ICCV/NeurIPS 2024-2026 video-diffusion work). Five modifications selected (each with primary + secondary citation, mechanism, cost, falsification criterion):

1. **Anchor-frame x0 consistency loss** (Theme 3) — *priority 1*, small/medium cost. Add `‖pred_x0_train − train_latents‖²` (or VAE-decoded perceptual variant) auxiliary term to `compute_flow_matching_loss_conditioned`. Exploits free supervisory signal from visible frames 0-47 currently discarded. Citations: Sangare et al. CVPR 2026 x0-supervision, RaMViD 2022, MiVID 2025, CustomTTT AAAI 2025.
2. **VAE-decoder-only TTA** (Theme 2) — *priority 2*, small/medium cost recipe pivot. Freeze DiT, adapt only VAE decoder weights on round-trip reconstruction MSE + LPIPS. Tests the user's "VAE is the bottleneck" hypothesis directly. Citations: REPA-E ICCV 2025, LeanVAE ICCV 2025.
3. **Augmentation-consistency TTA (MEMO/TTT-MAE)** (Theme 4) — *priority 3*, small cost. Convert existing per-step round-robin over augmented variants into a cross-augmentation consistency loss on predicted x0. Citations: MEMO NeurIPS 2022, TTT-MAE NeurIPS 2022.
4. **Annealed-timestep curriculum + limited-interval guidance** (Themes 5+7) — *priority 4*, small cost easy add (bundle with priority-1 sbatch wave). Anneal σ-sampling from large to small across TTA steps; restrict CFG to a middle σ interval at inference. Citations: Yi et al. CVPR 2024 DTC, Kynkäänniemi et al. NeurIPS 2024 LIG.
5. **Continual streaming TTA with stochastic weight restoration** (Theme 9) — *priority 5*, medium cost recipe pivot, conditional on 1+2 results. Don't reset adapter between videos; apply CoTTA-style stochastic restoration to control drift. Citations: Wang/Sun/Gandelsman JMLR 2025, CoTTA CVPR 2022.

Total to clear priorities 1-4: ~10 days wallclock, ~220 GPU-h. Priority 5 is conditional and expensive (~400 GPU-h serial across chunks). Honest "what the literature does NOT support" section calls out Theme 6 prompt ensembling as dead for our setting (NOPROMPT already ruled it out at population level) and flags TTOM 2026's saturation-then-degradation observation as the explicit literature argument against simply increasing TTA iteration count. Five open questions for the user on λ-sweep, pure-latent vs VAE-decoded anchor variants, Modification-2 baselines, streaming serialisation strategy, and Modification 1×3 interaction.

**Honest threat-finding:** TTOM (ICLR 2026 submission, [OpenReview](https://openreview.net/pdf?id=wqCwcTZsrv)) explicitly reports saturation-then-degradation when test-time optimization iterations exceed a soft optimum on a different video-diffusion task (layout-controlled generation). Their setting is not ours but the qualitative shape matches and confirms we should not be expecting a "more iterations" recipe to work. No paper directly forecloses our research question (per-video TTA on LongCat-Video continuation at 1000v scale with conditioning frames as the only supervisory signal); the closest prior art (CustomTTT AAAI 2025) uses teacher-LoRA distillation, not self-supervised reconstruction, so the research gap remains open.

---

## 2026-06-12 — Repo cleanup: hypothesis + presentation docs removed from main
**Tags:** repo-cleanup, removal, record-keeping
**Owner:** agent (per Wenchen request)
**Refs:**
- `.gitignore` (new section: `sweep_experiment/reports/HYPOTHESES_*.md`, `sweep_experiment/reports/PRESENTATION_*.md`)
- `sweep_experiment/reports/INDEX.md` (HYPOTHESES row removed from "Standalone stocktake / review documents"; "Presentations" section removed entirely)
- Removed from main: `sweep_experiment/reports/HYPOTHESES_per_video_tta_suitability_2026-06-09.md` (still in history at commit `03d1a03`); `sweep_experiment/reports/PRESENTATION_hypothesis_taxonomy_2026-06-11.md` (still in history at commit `6e9a984`)

Both files were documentation-only artifacts not needed on the cluster (no code, no sbatch, no result-reproducibility role). User wants them local-only to keep the cluster checkout lean. Files remain accessible via `git show 03d1a03:sweep_experiment/reports/HYPOTHESES_per_video_tta_suitability_2026-06-09.md` and `git show 6e9a984:sweep_experiment/reports/PRESENTATION_hypothesis_taxonomy_2026-06-11.md` if needed. `.gitignore` updated with `HYPOTHESES_*.md` / `PRESENTATION_*.md` patterns under `sweep_experiment/reports/` so future drafts don't get re-pushed by accident. Operational documents (RUNBOOK, PLAN_offline_investigations, PLAN_gating_experiment, REFRESHER, REVIEW, paper_tables, per_video_analysis bundles) are unchanged.

---

## 2026-06-11 (later+3) — Presentation: 5-bucket principle-based hypothesis taxonomy
**Tags:** presentation, hypothesis-taxonomy, paper-narrative
**Refs:**
- [`PRESENTATION_hypothesis_taxonomy_2026-06-11.md`](PRESENTATION_hypothesis_taxonomy_2026-06-11.md) (new) — ~30-minute talk-walkthrough notes (1533 lines of structured markdown)
- Companion: [`HYPOTHESES_per_video_tta_suitability_2026-06-09.md`](HYPOTHESES_per_video_tta_suitability_2026-06-09.md), [`REVIEW_per_video_tta_suitability_2026-06-09.md`](REVIEW_per_video_tta_suitability_2026-06-09.md), [`PLAN_gating_experiment_2026-06-11.md`](PLAN_gating_experiment_2026-06-11.md)
- INDEX.md "Presentations" section (new row)

Talk-walkthrough markdown organising the 12+ per-video TTA-suitability hypotheses around theoretical principle rather than compute cost. Five buckets:

- **A. Model-perceived difficulty** (diffusion likelihood / OOD): mean_diffusion_loss_caption, score_norm_t*, lid_flipd, latent moments — Theme B in HYPOTHESES
- **B. Loss-landscape geometry** (gradient norm / single-step probes): grad_norm_θ0 (H-T3-1), single_step_loss_drop (H-T3-2), loss_var_t (H-T2-5) — Themes A + C + G in HYPOTHESES
- **C. Visual / temporal complexity** (model-independent video features): flow distribution shape (H-T1-4), hf_energy_ratio (H-T1-3), bpp (H-T1-2), scene-cuts (H-T1-6), DINO temporal-L2 / Laplacian variance / RGB-hist entropy — Theme D in HYPOTHESES
- **D. Cross-modal alignment** (caption-video matching): CLIP_min (H-T1-5), cfg_gap (H-T2-3), delta_caption_minus_uncond — Theme E in HYPOTHESES
- **E. Reconstruction observability** (VAE round-trip error): rec_err_l1 / rec_err_lpips (H-T1-1) — bucket I proposed without a direct HYPOTHESES theme, grounded in the latent-space-typicality line (Ding et al. 2025, Järve et al. 2025)

Each bucket maps to a different paper subsection structure if its top feature wins the gating experiment. Bucket B is flagged as the tail-risk predictor (catastrophic-failure detection — `panda_0098` 44.55→22.16 dB) while A/C/D/E predict modal gain; the closing recommendation is to combine B × {A or C} as a multivariate gate for the deployment story. 9 features have non-trivial secondary affinities (cross-bucket spans: bpp is C primary + A confound covariate; score_norm is A primary + B geometric flavour; FLIPD is A primary + C complexity; loss_var_t is B per spec + A loss-values flavour; flow distribution shape is C primary + B sparse-gradient mechanism; scene cuts are C primary + B non-stationary-landscape mechanism; CFG-gap is D primary + A ε-field; rec-err is E primary + A latent-space-typicality; delta_caption_minus_uncond is A primary + D lite-alignment-proxy). No feature is unbucketed; no 6th bucket proposed.

~30-minute talk content structured as 13 slides (0 title; 1 saturation puzzle; 2 ruled-out hypotheses; 3 5-bucket table; 4–8 per-bucket detail; 9 synthesis with modal-gain-vs-tail-risk + method-agnostic-vs-method-specific + ensemble-gate hypothesis + cross-bucket prediction matrix; 10 four-scenario recommendation; 11 limitations + open questions; 12 appendix with per-feature commentary + pre-registered analysis plan + contingency planning + glossary + quick-reference card). Companion to PLAN_gating_experiment_2026-06-11.md (which tests every feature in this taxonomy across Phase 0–3) and HYPOTHESES_per_video_tta_suitability_2026-06-09.md (literature-grounded hypothesis menu). No code or sbatch changes; this is taxonomy + narrative on top of the already-AUTHORISED Phase-0 protocol.

---

## 2026-06-11 (later+2) — Runbook: Friday 2026-06-12 cluster-restart launch sequence
**Tags:** runbook, cluster-restart, friday-morning
**Refs:**
- [`RUNBOOK_friday_morning_2026-06-12.md`](RUNBOOK_friday_morning_2026-06-12.md) (new) — single executable runbook
- New code (Tier-3 probes, closes the gap noted in commit `38df1ba`): `scripts/compute_tier3_probes.py`, `scripts/sbatch/run_compute_tier3_probes.sbatch`
- Extended: `scripts/sbatch/submit_per_video_feature_pipeline.sh` (now fans out Tier-3 in parallel with Stage 1a/1b; correlation depends on `afterok:1a:1b:1c`; `SKIP_TIER3=1` mirrors `SKIP_OOD=1`), `scripts/correlate_tta_gain_with_features.py` (new `--tier3-csv` flag; T3P tier appears alongside T1 / T3 / OOD in `correlation_table.md`, ranking, plots, and `summary.md`)
- INDEX.md: new "Runbooks" section pointing at the runbook

Single document consolidating every cluster-restart action authorised before the 2026-06-09 → 2026-06-12 maintenance window: (Track A1) gating Phase 0 — feature extraction + diffusion-OOD + Tier-3 probes via `submit_per_video_feature_pipeline.sh` (~3-4 h, 3 GPU jobs + 1 CPU correlation auto-chained on `afterok`); (Track A2) NOPROMPT sweep close-out via `submit_standard_1000v_noprompt.sh` (80 jobs; gated on a smoke check of `sacct -j 10618645`; ~5-7 wallclock days with the 2-way GPU cap); (Track B) the offline-investigation A1-A4 login-node CPU sequence from `PLAN_offline_investigations_2026-06-11.md` (~15 min total); (Track C) VBench backfill of the 4 NOPROMPT methods × 2 datasets after A2 merges (~1 day with 8-way parallelism), then paper-table rebuild via `build_paper_tables.py`. **Critical path:** A2 NOPROMPT sweep → C VBench backfill → paper-table rebuild (~6-7 wallclock days). Everything else completes inside the first 6 GPU h (A1) or 15 CPU min (B).

**Implementation gap closed (the one thing flagged as "follow-up implementation task" in `PLAN_gating_experiment_2026-06-11.md` §2.5 / §3.1 + the same-day "Tier-3 probes wrapper TODO" surfaced in `ANALYSIS_LOG.md` 2026-06-11 (later) entry):** Tier-3 probes now have a runner and an sbatch wrapper. `compute_tier3_probes.py` mirrors `lora_experiment/scripts/run_lora_tta.py`'s LoRA recipe (r=8 / α=16 / lr=5.0e-5 / weight_decay=0.01 / targets=qkv,proj on all blocks, no FFN — verified against `sweep_experiment/sbatch/submit_standard_1000v_chunked.sh` line for `LORA_R8_TTA`); resets the LoRA adapter + re-instantiates the optimiser per (video, timestep) loop so there is zero carry-over between videos (the no-carryover guarantee in the gating plan §2.4 / HYPOTHESES H-T3-1+H-T3-2 spec); records `grad_norm_lora_t{T}` (L2 norm of LoRA-parameter gradients, H-T3-1) + `loss_drop_pct_t{T}` (fractional drop after one Adam step, H-T3-2) at timesteps 100/500/900, plus their means + per-timestep `loss_t0/loss_t1` audit columns. Output CSV schema (joined by `video_id`): `video_id, grad_norm_lora_t{100,500,900}, mean_grad_norm_lora, loss_drop_pct_t{100,500,900}, mean_loss_drop_pct, loss_t0_t{100,500,900}, loss_t1_t{100,500,900}, n_visible_frames, n_gen_target_frames, lora_rank, lora_alpha, lora_lr, lora_targets, seed`. Sbatch wrapper mirrors `run_compute_diffusion_ood.sbatch` directives (h200 GPU, 192G mem, 4h time, account `torch_pr_36_mren`, module + conda + unset PYTHONHOME/PYTHONPATH preamble, ERR trap).

**Status:** runbook READY — executes when cluster comes back online. Cluster restart expected ~2026-06-12 morning per the user. No further plan work needed before launch.

---

## 2026-06-11 (later+1) — Plan: offline investigations during cluster maintenance
**Tags:** plan, offline-investigation, paper-narrative
**Refs:**
- [`PLAN_offline_investigations_2026-06-11.md`](PLAN_offline_investigations_2026-06-11.md) (new)
- [`REFRESHER_standard_vs_longhorizon_2026-06-11.md`](REFRESHER_standard_vs_longhorizon_2026-06-11.md) (new)
- New scripts: `scripts/compare_horizons_per_video.py`, `scripts/analyze_per_chunk_fvd.py`, `scripts/aggregate_loss_history.py`; extended `scripts/analyze_per_video_tta_gain.py`
- INDEX.md "Plans / proposals" + "Standalone stocktake / review documents" + "Analysis tools" sections (rows added)

The GPU cluster is in maintenance through ~2026-06-15, but the login node still has filesystem access to every past experiment output. We laid out and committed a 5-analysis offline suite (A1–A5) that runs on the login-node CPU in ≤ 15 min total: (A1) long-horizon per-video gain analysis against `panda_longctx_1000v` + `tinylora_longctx_1000v` — this is the primary gap (the 2026-06-09 standard-horizon bundle has no long-horizon counterpart); (A2) side-by-side standard- vs long-horizon distribution comparison to test the user's 2026-06-11 hypothesis that long-horizon has fatter tails in BOTH directions even when the population mean is unchanged; (A3) per-chunk ΔFVD sign analysis on both regimes (closes the deferred TODO from `paper_tables/2026-06-09_panda_std_prompt_vs_noprompt_full_metrics.md`); (A4) per-video held-out-anchor-loss aggregation joined against ΔPSNR (per-step TRAINING loss is not persisted by any runner; the held-out anchor loss stored under `result['early_stopping_info']['loss_history']` is — it's the right quantity for the mechanism question anyway); (A5) record-keeping refactor of `analyze_per_video_tta_gain.py` to write ΔLPIPS tails + top-50-winner Jaccard matrix + sign-agreement statistics into `summary.md` natively (these were computed on-the-fly in the 2026-06-09 analysis but never persisted; the "6.3× lift" number now lands in the document automatically).

**Loss-history availability finding (recorded for the audit trail):** the per-step training loss accumulated inside each runner's `optimize_*` function is **not** persisted to JSON (only `final_loss = losses[-1]` is) and is also **not** printed per-step to stdout, so slurm-log parsing would not recover it either. The held-out anchor-loss trajectory, by contrast, IS file-based — `early_stopping.py::AnchoredEarlyStopper.state` writes a `loss_history: List[(step, anchor_loss)]` into the per-video result dict whenever ES is enabled (the default). `aggregate_loss_history.py` therefore reads the anchor-loss path; the slurm-log fallback was deemed unnecessary.

---

## 2026-06-11 (later) — Gating plan: all 4 open decisions resolved, Phases 0–3 authorised
**Tags:** plan-resolution, gating-experiment, authorisation
**Refs:**
- [`PLAN_gating_experiment_2026-06-11.md`](PLAN_gating_experiment_2026-06-11.md) (now AUTHORISED; §8 rewritten from "Open questions" to "Resolved decisions"; §2.5 / §3.1 / §3.2 / §3.4 / §3.5 updated to reflect locked-in choices)
- Earlier same-day entry below: original 4 open questions in §8

User authorisation 2026-06-11 of all 4 open questions in the gating plan:

1. **Phase 4 (long-horizon validation) auto-fire:** RESOLVED → Separate authorisation required after Phase 3's `RECOMMENDATION.md` is reviewed. Long-horizon shows a real method-asymmetry signal at population level (Subj 0.018 between AdaSteer and LoRA r=8 at 76-frame vs 0.005 at 28-frame; ref `paper_tables/2026-06-08_headline_1000v.md` Table 3), so Phase 4 is non-trivial and merits human review.
2. **Cost-aware Pareto compute-saved interpretation:** RESOLVED → Immediate 999-video run only as the headline savings number. Transferable-to-future-research is implicit, not speculated about specific unrun benchmarks.
3. **Multiple-comparison correction:** RESOLVED → Bonferroni α/192 primary, BH-FDR q=0.1 secondary. Both reported in every Spearman ρ table in §3.2 / §3.3 deliverables.
4. **Tier-3 mini-TTA probes (H-T3-1 `grad_norm_θ0` + H-T3-2 `single_step_loss_drop`):** RESOLVED → Both included in Phase 0 per the user's explicit "test all hypotheses" instruction. Cost +~2 GPU hours per 999-video run. §2.5 deferred-follow-up framing removed; §3.1 Phase 0 scope updated.

**Follow-up implementation task surfaced by Decision 4:** the Tier-3 probes need a `run_compute_tier3_probes.sbatch` wrapper (or inline integration with an existing extractor) plus an extension of `submit_per_video_feature_pipeline.sh` to schedule it, before Phase 0 can run end-to-end. This commit only updates the plan documents; the wrapper is flagged as a small follow-up implementation task to be landed in a separate commit when we're ready to fire Phase 0.

**Plan status:** PLAN → PLAN-AUTHORISED. Phases 0–3 green-lit; Phase 4 gated on RECOMMENDATION.md review. See updated [`PLAN_gating_experiment_2026-06-11.md`](PLAN_gating_experiment_2026-06-11.md) §8 for the four resolved decisions verbatim with rationale, and §2.5 / §3.1 / §3.2 / §3.4 / §3.5 for the protocol-level language updates.

---

## 2026-06-11 — Plan: optimal per-video TTA gating-strategy experiment
**Tags:** plan, gating-experiment, paper-narrative
**Refs:**
- [`PLAN_gating_experiment_2026-06-11.md`](PLAN_gating_experiment_2026-06-11.md) (new)
- Companion docs: [`REVIEW_per_video_tta_suitability_2026-06-09.md`](REVIEW_per_video_tta_suitability_2026-06-09.md), [`HYPOTHESES_per_video_tta_suitability_2026-06-09.md`](HYPOTHESES_per_video_tta_suitability_2026-06-09.md)
- INDEX.md "Plans / proposals" section (new)

Detailed five-phase plan for finding the optimal per-video gating strategy
for TTA on LongCat-Video, integrating the diffusion-OOD scorer (commit
`dc115e7`), the existing Tier-1 feature battery, and the 12 literature-
grounded hypotheses across themes A/B/D/E/G into a 20-row master feature
menu. Three gating families considered (binary apply/skip, between-family
method routing, continuous gain prediction). Recommendation criteria are
explicit: held-out gain > per-video noise floor (≥ 0.05 PSNR / ≥ 0.005
LPIPS), coverage ≥ 50 %, feature compute ≤ 30 min per 999 videos.

**Status: PLAN — requires user authorisation before execution.** The plan
asks specifically for green-light on (a) Phase 0 cluster jobs via the
existing `submit_per_video_feature_pipeline.sh` plus three new ≤ 100-LOC
Tier-1 feature scripts (`extract_bpp_features.py`, `extract_vae_recerr_features.py`,
`extract_fft_features.py`) and a ≤ 30-LOC patch to the OOD scorer for the
score-norm feature; (b) Phase 1/2/3 CPU analysis scripts
(`analyze_gating_univariate.py`, `analyze_gating_multivariate.py`,
`build_gating_pareto.py`); (c) ~3 wallclock days total cost. Phase 4
(long-horizon validation) is conditional on Phase 3 producing a clean or
partial win; case 3 (no win) licenses the honest paper claim that gating
awaits the long-horizon regime — fully consistent with REVIEW Story A.

---

## 2026-06-09 (latest+3) — Prompt-vs-NOPROMPT full-metrics table + per-video ΔLPIPS tail breakdown
**Tags:** paper-table, noprompt, lpips-tail-breakdown
**Refs:**
- [`paper_tables/2026-06-09_panda_std_prompt_vs_noprompt_full_metrics.md`](paper_tables/2026-06-09_panda_std_prompt_vs_noprompt_full_metrics.md) (new)
- Built from [`paper_tables/2026-06-08_headline_1000v.md`](paper_tables/2026-06-08_headline_1000v.md) (prompted full 7-dim VBench) and [`paper_tables/2026-06-09_panda_std_with_noprompt_partial.md`](paper_tables/2026-06-09_panda_std_with_noprompt_partial.md) (NOPROMPT 3-dim partial)
- LPIPS tail computed from [`per_video_analysis/2026-06-09/per_video_gains.csv`](per_video_analysis/2026-06-09/per_video_gains.csv) (schema is `<METHOD>_lpips`, not `lpips_<METHOD>` as spec guessed)

Consolidated the prompt-vs-NOPROMPT picture for Panda 1000v / 480p / 17-frame
standard horizon: per-frame (PSNR / SSIM / LPIPS), distributional (FVD / FID),
and the 3 in-runner VBench dims for all 5 methods (NOTTA + ADA + ADA_NOPROMPT +
LORA_R8_TTA + LORA_R8_TTA_NOPROMPT), plus the full 7-dim VBench for the two
prompted methods. Both NOPROMPT pairs sit within 0.01 PSNR / ≤0.001 SSIM /
≤0.001 LPIPS / 4 FVD / 0.3 FID / 0.001 VBench-dim of their prompted siblings —
the TTA-time text prompt is a noise channel on this regime for both the
AdaSteer and LoRA-r8 families.

Also added a per-video ΔLPIPS tail breakdown structured analogously to the
existing ΔPSNR tail breakdown (LPIPS is the per-video perceptual analog of
FVD, which is distributional and not per-video). Headline: **82.3 % of the
999 clips are within ±0.005 LPIPS of NOTTA for TL_TIED_R2** (tightest), down
to 54.8 % for ADA (loosest); same method ordering as the ΔPSNR tail (TinyLoRA
tightest, LoRA-r8 middle, AdaSteer loosest), and median Δ is essentially 0
for every method (|median Δ| ≤ 0.0006), confirming the population-level LPIPS
saturation isn't hiding a one-sided per-video story. NOPROMPT variants are
within ≤1.2 pp on every bucket of their prompted siblings — distributionally
indistinguishable on the perceptual axis too.

TODOs recorded in the new doc: (1) VBench Motn/Dyn/IQ/Flick backfill on the
2 NOPROMPT methods, (2) TinyLoRA NOPROMPT pairings once cluster returns,
(3) per-chunk ΔFVD sign analysis via `chunk_*/summary.json` once those files
are accessible again.

---

## 2026-06-09 (latest+2) — Pre-maintenance stocktake on per-video TTA suitability
**Tags:** review, stocktake, paper-narrative
**Refs:**
- [`REVIEW_per_video_tta_suitability_2026-06-09.md`](REVIEW_per_video_tta_suitability_2026-06-09.md) (new)
- Companion: [`HYPOTHESES_per_video_tta_suitability_2026-06-09.md`](HYPOTHESES_per_video_tta_suitability_2026-06-09.md) (parallel literature pass)
- INDEX.md "Standalone stocktake / review documents" section (new)

Cluster is in maintenance for several days. Consolidated the completed-experiment
conclusions on per-video TTA suitability (Panda 1000v / 480p / 17-frame), the
hypotheses ruled out at this scale, the implemented-but-not-run experiment
inventory, and the recommended next-wave priority order into one document so
the next experimental wave starts from a known baseline. No new findings — see
the review for citations into the existing `paper_tables/` and
`per_video_analysis/` artefacts.

---

## 2026-06-09 (later x3) — Hotfix: Tier-3 gen-target auto-detection + env-activation note
**Tags:** bugfix, infra, per-video, criteria
**Refs:** [scripts/extract_video_features_for_tta.py](../../scripts/extract_video_features_for_tta.py) commit fix above

First cluster run of the criteria-correlation pipeline (commit `187751c`)
revealed two issues:

1. **Tier-3 gen-target auto-detection was wrong** — set to `[0:48]` (same
   as TTA-visible) instead of `[48:62]` for `panda_1000v_standard`. Made
   the cross-window DINO/CLIP diagnostics self-similarity. Fixed in the
   commit above by sourcing all four config constants
   (`TTA_TOTAL_FRAMES`, `GEN_START_FRAME`, `NUM_FRAMES`, `NUM_COND_FRAMES`)
   from `submit_standard_1000v_chunked.sh`.

2. **`(base)` conda env on the cluster does not have torch.** Runner
   sbatch wrappers activate `/scratch/$USER/conda-envs/longcat` before
   invoking the trainer (see e.g. `sweep_experiment/sbatch/run_sweep.sbatch`
   line 252, `delta_experiment/sbatch/run_tinylora.sbatch` line 151,
   `lora_experiment/sbatch/run_lora_tta.sbatch` line 92, and the canonical
   setup in `env_setup/01_setup_longcat_env.sbatch` line 52). Same
   activation is required before running the feature-extraction script.
   Cluster-command preamble updated below.

**Lesson recorded above (do not lose):** any new analysis script that
imports torch / transformers / open-clip MUST run in the same env as
the TTA runners. Default `(base)` conda lacks these.

---

## 2026-06-09 (latest+1) — Per-video feature battery: chasing a non-random ΔPSNR predictor
**Tags:** methodology, decision, paper-narrative
**Owner:** agent
**Refs:**
- New scripts: `scripts/extract_video_features_for_tta.py`, `scripts/correlate_tta_gain_with_features.py` (commit `187751c`)
- Builds on the per-video gains CSV from `74093eb`'s bundle at `sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv`.
- Outputs target `sweep_experiment/reports/per_video_analysis/2026-06-09/{video_features.csv, criteria_correlation/}`.

**User feedback:** we cannot ship "no per-video correlation" as a paper result —
the three features the existing analysis tested (RAFT mean-flow, baseline PSNR,
caption word count) all came in at |ρ| < 0.15 across every method. The bet is
that *what makes a video easy for TTA* is not random, we just haven't tested the
right features. Extend the per-video pipeline with a richer Tier-1 battery
(scene cuts via PySceneDetect + RGB-histogram backup, CLIP image↔text alignment
mean/var/min, DINOv2 temporal-L2, Laplacian-variance sharpness, RGB-histogram
entropy colour-diversity) computed *only* on the TTA-visible frames (frames
[0:48] for `panda_1000v_standard` per the audit block at the top of the
extractor script), plus two Tier-3 diagnostics (DINO TTA↔gen-region mean
similarity, CLIP↔gen-region caption alignment) flagged as not-online-actionable.

If no Tier-1 feature clears |ρ| ≥ 0.2 for ≥ 2 methods, the summary.md template
honestly says so and lists the next iteration's feature candidates (caption LM
perplexity, action-vs-object classification, optical-flow second moments,
CLIP-vs-DINO disagreement, base-VAE recon error on the visible window).
We do not silently widen the threshold post-hoc.

---

## 2026-06-09 (later) — Panda 1000v std NOPROMPT confirms TTA-caption is a noise channel
**Tags:** finding, paper-narrative, ablation
**Refs:**
- [`paper_tables/2026-06-09_panda_std_with_noprompt_partial.md`](paper_tables/2026-06-09_panda_std_with_noprompt_partial.md)
- merge_chunks.py output for `panda_1000v_standard` ran 2026-06-09 15:39 UTC+8

Both Panda standard-horizon NOPROMPT methods completed all 10 chunks and
merged cleanly. Population-level results agree with headline within ~0.01 PSNR /
~3 FVD / ~0.1 FID:

| | PSNR | FVD |
|---|---|---|
| ADA → ADA_NOPROMPT | 17.94 → 17.93 (Δ −0.01) | 153.4 → 155.5 (Δ +2.1) |
| LORA_R8_TTA → LORA_R8_TTA_NOPROMPT | 17.85 → 17.86 (Δ +0.01) | 157.9 → 154.0 (Δ −3.9) |

**Conclusion for the paper:** the TTA-time text prompt contributes negligibly
to the adaptation loss on Panda 1000v standard horizon. Adaptation is
essentially video-conditioned. This is robust to dropping the caption
unconditionally during TTA — the model's behaviour at inference (where the
real prompt is restored) is unchanged within sample-size noise.

**Open question:** is the population-level saturation hiding a winner/loser
split? Earlier ADA_NOPROMPT smoke chunk_0 showed +0.68 PSNR vs ADA-merged
(later attributed to chunk-0 happening to contain easier videos), so per-chunk
or per-video variance is non-zero — we just need to characterise it. The
new `scripts/analyze_per_video_tta_gain.py` (commit `5d92733`) is the right
tool.

**Lesson recorded above (do not lose):** per-chunk noise floor on this eval
set is approximately 0.5 dB PSNR (NOTTA per-video PSNR std ≈ 5 dB / √100 ≈
0.5 dB chunk-level standard error). Treat any chunk-level effect ≤ 0.5 dB
as sampling variation until the merge confirms it.

---

## 2026-06-09 (latest) — Per-video TTA-gain analysis tooling (winners/losers + feature correlations)
**Tags:** methodology, decision, paper-narrative
**Owner:** agent
**Refs:**
- New script: `scripts/analyze_per_video_tta_gain.py`
- Sibling (kept, not modified): `scripts/plot_dynamicness_correlation.py`
- INDEX.md "Analysis tools" row (new section)
- Inputs assumed on cluster:
  - Per-video metrics: `<series>/<METHOD>/chunk_*/summary.json["results"]`
    under `sweep_experiment/results/panda_1000v_standard` and
    `delta_experiment/results/tinylora_panda_1000v_standard`
  - Dynamicness: `datasets/panda_1000_480p/dynamic_degree.json`
    (RAFT-small mean optical flow, June 1-2; same JSON the headline
    `plot_dynamicness_correlation.py` figure consumes)
  - Captions: `datasets/panda_1000_480p/metadata.csv` (`filename`, `caption`)

**Motivation.** The 80-job NOPROMPT sweep's first chunk-0 smoke run of
`ADA_NOPROMPT` looked +0.68 dB PSNR better than the headline ADA, which
would have been a meaningful ablation signal. When all 10 chunks merged,
the gain washed out to ≈ 0 dB — the chunk-0 effect was pure sampling
noise on the 100-video chunk. Combined with the broader 1000v
saturation finding (ANALYSIS_LOG entry 2026-06-08 "VBench backfill
complete; saturation confirmed across all 1000v regimes"), the question
shifts from "does TTA improve on average?" (no) to "does TTA help on
SOME videos and hurt others, with the average washing out?". If the
answer is yes, the followup question is: what video-level features
predict the winners? That changes the paper narrative from "AdaSteer is
neutral at 1000v scale" to "AdaSteer is neutral at population level but
non-trivially helpful on a characterizable subset of videos."

**Hypotheses tested.** For each TTA method on `panda_1000v_standard`
(ADA, LORA_R8_TTA, ADA_NOPROMPT, LORA_R8_TTA_NOPROMPT — plus the two
TinyLoRA methods when their dirs exist), compute per-video
ΔPSNR = method_psnr − NOTTA_psnr and correlate against:
1. **Dynamicness.** RAFT mean optical flow per video. High-motion clips
   may give TTA more "headroom" because NOTTA's prediction degrades
   faster with motion. Reported as Pearson r AND Spearman ρ; visualised
   as ΔPSNR vs flow-quintile (log-x).
2. **Baseline difficulty.** The video's own NOTTA PSNR. Hypothesis: TTA
   helps when NOTTA is bad (low-PSNR videos) and is a no-op (or
   regression) when NOTTA is already strong. Tested via per-method
   scatter of (NOTTA-PSNR, ΔPSNR) with a least-squares regression line;
   negative slope = "TTA preferentially fixes hard cases".
3. **Caption complexity.** Caption word-count quintiles. Hypothesis:
   short captions (less prior) leave more room for TTA gradient signal
   to matter. (The NOPROMPT ablation also pertains to this axis from
   the opposite direction.)
4. **Distribution shape.** Histogram of per-video ΔPSNR per method. A
   symmetric ±2 dB spread around 0 means TTA has real per-video effects
   that average to zero (wins paid for by losses); a right-skewed
   distribution means TTA is net-positive on a subset. The "subset"
   then needs to be characterizable for the paper to claim it.

**Outputs of `scripts/analyze_per_video_tta_gain.py` (per --output-dir):**
- `per_video_gains.csv` — long-format table, one row per video_id, with
  every method's per-video PSNR/SSIM/LPIPS and the Δ-against-baseline
  columns. Joinable to other per-video signal CSVs
  (`per_video_difficulty_signals.py` etc.).
- `delta_psnr_vs_dynamicness.png` — quintile-binned mean ΔPSNR ± SEM
  per method, log-x on dynamicness, per-bin n annotated.
- `delta_psnr_vs_baseline_psnr.png` — per-method scatter + LS-fit line;
  panel title shows Pearson r and slope.
- `delta_psnr_histogram.png` — overlaid translucent histograms; legend
  shows per-method N, mean, and median.
- `delta_psnr_vs_caption_length.png` — same shape as the dynamicness
  plot, binned by caption-word-count quintile.
- `summary.md` — data integrity table, per-method tail counts at
  ±0.5 dB and ±1.0 dB (the "tails matter for the paper narrative"
  number), per-method Pearson + Spearman correlations against the three
  features, and top-10 winners + top-10 losers per method (video_id,
  truncated caption, mean_flow, baseline PSNR, ΔPSNR).

**Lesson — do NOT extrapolate from 1 chunk.** Each chunk in the
`panda_1000v_standard` series is exactly 100 videos (10 chunks × 100 =
999 videos, where one source video is silently skipped on filtering).
The standard deviation of mean PSNR over 100-video subsamples of this
eval set is roughly **0.4-0.7 dB** for any single method (estimable
from the per-video PSNR variance of NOTTA: σ ≈ 5 dB / √100 ≈ 0.5 dB).
A single chunk's mean PSNR differing from the population mean by
±0.5-1.0 dB is therefore ENTIRELY consistent with the null hypothesis
"this ablation does nothing." Future agents: when a smoke chunk shows
a >0.3 dB effect, ALWAYS wait for at least 3-4 additional chunks before
treating it as signal, and prefer estimating the per-chunk variance
from the existing NOTTA chunks (which are reproducible per chunk) over
arguing from a single delta. The corresponding diagnostic
(reading per-chunk noise off NOTTA) lives in the per-video CSV
emitted by this script: `NOTTA_psnr` standard deviation across rows
divided by √100 gives the per-chunk-mean noise floor.

**Smoke validation (local).**
- `python3 -m py_compile scripts/analyze_per_video_tta_gain.py` clean.
- `python3 scripts/analyze_per_video_tta_gain.py --help` prints full usage.
- End-to-end synthetic 5-method × 60-video × 3-chunk run (locally
  fabricated `summary.json` / `dynamic_degree.json` / `metadata.csv`)
  produces all 5 output artefacts without crash and correctly drops
  rows with NaN PSNR.

**Cluster command (user runs this; NO slurm submission needed).**
```bash
cd /scratch/$USER/longcat-video-tta && git pull && \
    python3 scripts/analyze_per_video_tta_gain.py \
        --series-path sweep_experiment/results/panda_1000v_standard \
        --tinylora-series-path delta_experiment/results/tinylora_panda_1000v_standard \
        --output-dir sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)
```

**Note on partial NOPROMPT methods.** Currently only `ADA_NOPROMPT` and
`LORA_R8_TTA_NOPROMPT` are 10/10-chunk complete on `panda_1000v_standard`.
The TinyLoRA `_NOPROMPT` variants are still in flight (INDEX.md row 6).
The script auto-detects every method dir with at least one chunk and
proceeds; partially-merged methods will have fewer rows in the
intersection but are still analysable. Re-run after each round of
chunk completion.

**What this script intentionally does NOT do.**
- It does NOT recompute dynamicness (uses the precomputed June 1-2
  RAFT JSON — re-running `scripts/compute_dynamic_degree.py` on the
  same eval set should be byte-stable but is unnecessary).
- It does NOT replace `scripts/plot_dynamicness_correlation.py` —
  that script shows raw per-method metric curves vs flow bins and is
  the right tool when the question is "does the underlying metric value
  change with dynamicness?"; the new script focuses on ΔPSNR
  distributions and adds caption / baseline-PSNR axes.
- It does NOT classify themes (use `scripts/diagnose_long_horizon_failures.py`
  for caption-keyword theme buckets) — the regex-based theme taxonomy
  there is purpose-built for the long-horizon regression narrative and
  is orthogonal to the dynamicness / baseline-difficulty / caption-length
  hypotheses tested here.

---

## 2026-06-09 — Retrieval × NOPROMPT TTA ablation: 40-job Panda sweep queued (pending 25K pool)
**Tags:** decision, methodology, in-flight, paper-narrative
**Owner:** agent
**Refs:**
- `sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh` (new)
- Combines two existing knobs:
  - `--tta-disable-caption` (added in commit 16c1532; helpers
    `add_tta_disable_caption_args` / `tta_caption_for` in
    `delta_experiment/scripts/common.py`)
  - Batch-level retrieval (existing in
    `sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh`;
    `--batch-videos K --batch-method similarity|sequential
    --retrieval-pool-dir ...`)
- Runner: `delta_experiment/scripts/run_delta_a.py` (METHOD=delta_a in
  `sweep_experiment/sbatch/run_sweep.sbatch`).
- Pool dependency: `datasets/panda_segment_pool` after step 2 (25K-pool
  build) + step 3 (caption-embedding precompute) — see `INDEX.md`
  "Pending merges and in-flight sweeps" rows 2-3.
- Paper-table destination: `paper_tables/<date>_panda_retrieval_noprompt.md`
  (one of the rows is the existing NOTTA from `panda_1000v_standard`,
  reused — NOTTA does not run TTA so dropping the TTA caption is a no-op).

**Hypothesis.** The headline retrieval-augmented AdaSteer sweep
(`K{5,10}_{SIM,RAND}`) trains on `[eval_video, n_1, n_2, ..., n_{K-1}]`
where each entry contributes a flow-matching loss
`MSE(model(x_t, t, encoder_hidden_states=text_i), v_i)`. Two effects
plausibly drive any retrieval gain (or loss): (a) the additional VISUAL
distribution coverage from neighbour clips, and (b) the additional TEXT
diversity from neighbour captions. The standalone NOPROMPT ablation on
the headline standard-horizon table (entry 2026-06-09 "TTA without text
prompt", `submit_standard_1000v_noprompt.sh`) tests (b) at K=1. This
sweep tests it at K∈{5,10}: if dropping captions at TTA time changes
retrieval results substantially, the K-fold caption diversity carries
real signal; if not, retrieval gains/losses come from neighbour-video
variance alone and the caption channel is dispensable.

**Configuration.** Identical to `submit_retrieval_1000v_chunked.sh`
modulo surgical changes:
1. Run IDs are suffixed with `_NOPROMPT` (`K5_SIM_NOPROMPT`,
   `K5_RAND_NOPROMPT`, `K10_SIM_NOPROMPT`, `K10_RAND_NOPROMPT`).
2. Each job is exported with `TTA_DISABLE_CAPTION=1`; `run_sweep.sbatch`
   line 367-369 translates this to `--tta-disable-caption` on the
   `delta_a` runner CLI (also wired for `lora` and `tinylora` but those
   methods are NOT in this sweep). `run_delta_a.py` line 872 wraps the
   per-entry TTA `encode_prompt(...)` with `tta_caption_for(args, caption)`,
   which returns `""` when the flag is set. Because that call is inside
   the `for entry in training_entries:` loop at line 848 — and
   `training_entries = [eval_entry] + neighbors` at line 800 — the same
   wrap covers BOTH the eval video caption AND every retrieved neighbour
   caption in the same code path. The inference `pipe.generate(...,
   prompt=eval_entry["caption"], ...)` at line 1118 is unchanged so
   PSNR / SSIM / LPIPS / FVD / FID / VBench all see the real caption.
3. Default scope is Panda ONLY (`ONLY_DATASET=panda` default); UCF
   dispatch is wired but opt-in via `ONLY_DATASET={ucf,both}`. UCF was
   already shown to be a poor retrieval testbed (class-block layout —
   see headline `ucf101_932v_retrieval` row in `INDEX.md`).
4. Default Panda pool: `datasets/panda_segment_pool` (the 25K-target
   destination; currently 3,302 segments, pending the in-flight step 2
   build + step 3 embedding precompute). Overridable via `PANDA_POOL=...`.
   Default UCF pool unchanged: `datasets/ucf101_pool_max` (26K).
5. Job-name prefix `t1krnp_` (retrieval + no-prompt; distinguishes from
   `t1kr_` headline retrieval and `t1knp_` standard-horizon no-prompt).
6. NOTTA is intentionally NOT in this sweep — NOTTA has no TTA step so
   `NOTTA_NOPROMPT` would be byte-identical to `NOTTA`. The existing
   NOTTA row from `panda_1000v_standard` is reused when building the
   `paper_tables/<date>_panda_retrieval_noprompt.md` paper table.

**Audit of `tta_caption_for` coverage (Task 1 of this sweep).**
Verified that EVERY `encode_prompt(...)` call run during TTA training
in both `delta_experiment/scripts/run_delta_a.py` and
`lora_experiment/scripts/run_lora_tta.py` already wraps its caption
argument with `tta_caption_for(args, ...)`. Findings:
- `delta_experiment/scripts/run_delta_a.py:872` — the SOLE
  `encode_prompt` call in the file; sits inside the
  `for entry in training_entries:` loop (line 848) which iterates over
  `[eval_entry] + neighbors` (line 800). Already wrapped. Inference
  uses `pipe.generate(..., prompt=eval_entry["caption"], ...)` at line
  1118 (unwrapped — correct: inference must see the real caption).
  Other `encode_prompt` matches in this file are dict-key strings
  (829, 838, 1258) or log strings (1028, 1324) — not call sites.
- `lora_experiment/scripts/run_lora_tta.py:1150` (batch-level retrieval
  path, inside `for te in training_entries:` at line 1136) — wrapped.
- `lora_experiment/scripts/run_lora_tta.py:1194` (instance-level
  fallback path, for `--batch-videos=1`) — wrapped. Inference at line
  1339 uses raw `caption` (unwrapped — correct).
No code edits were needed for Task 1; the previous no-prompt commit
(16c1532) already covered the per-entry loop, which IS the
neighbour-caption code path. The retrieval-noprompt ablation is
therefore byte-accurate at the runner level.

**Total compute.** 4 methods × 1 dataset × 10 chunks = 40 jobs.
Per-chunk wall: 14 h for K=5, 22 h for K=10 (no-prompt does not change
per-step cost). At the 2-way GPU cap, ~3 days wall.

**Workflow guard.** `--tta-disable-caption` defaults to `False`. Without
`TTA_DISABLE_CAPTION=1` in the export, behaviour is byte-identical to
the headline retrieval submitter. Verified via DRY_RUN: default Panda-
only prints exactly 40 sbatch lines (all with `TTA_DISABLE_CAPTION=1`
and `--export` strings otherwise matching the headline retrieval
submitter), `ONLY_DATASET=both` prints exactly 80, `ONLY_DATASET=ucf`
prints exactly 40, and `ONLY_METHODS="K5_SIM_NOPROMPT"` filters to 10.

**Launch sequence (gated on step 2 + 3 completion).**
1. After step 2 finishes, verify pool size:
   `ls datasets/panda_segment_pool/videos/*.mp4 | wc -l` ≈ 22-25K.
2. Run step 3 (caption embeddings):
   `sbatch --account=torch_pr_36_mren \
       --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/datasets/panda_segment_pool \
       delta_experiment/sbatch/precompute_pool_embeddings.sbatch`.
3. Verify `caption_embeddings.npy` + `.json` exist in the pool.
4. Smoke-test before firing 40 jobs:
   `DRY_RUN=0 NUM_CHUNKS=1 ONLY_METHODS="K5_SIM_NOPROMPT" \
       bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh`
   This launches exactly 1 chunk × 1 method (~14 h wall). Validates the
   pool / embeddings are wired and the `--tta-disable-caption` flag
   reaches `run_delta_a.py`.
5. Full launch:
   `bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh`.

---

## 2026-06-09 — Panda 25K segment-pool build: csv-limit + per-segment-resume fixes
**Tags:** finding, decision, methodology, in-flight
**Owner:** agent (relaunch pending)
**Refs:**
- `scripts/build_panda_segment_pool.py` (patched)
- Failed job: `10617270` (`build_panda_segment_pool.sbatch`,
  `SOURCE_METADATA=datasets/panda_metadata_full/panda70m_training_full.csv`)
  crashed at 49 s during step 2/5 metadata streaming.
- INDEX.md "Pending merges and in-flight sweeps" row 2.

**Failure mode (csv field-size-limit).** The Python stdlib `csv` reader
has a per-field hard limit of 131072 bytes by default. The full Panda-70M
training metadata (`panda70m_training_full.csv`, ~12 GB) stores
per-source captions, timestamps, and matching-score arrays as stringified
JSON-ish lists inside single CSV cells. For long-form videos those cells
routinely exceed 131072 bytes (~18.7 segments / source on average; cell
sizes scale roughly linearly with segment count). The 800K-row
`panda70m_training_2m.csv` subset that Phase 2A used capped at 2-3
segments/source so the limit was never hit. The first long-source row in
the full metadata triggered `_csv.Error: field larger than field limit
(131072)` after only 49 seconds.

**Resume-logic finding.** Independent of the csv crash, the pre-patch
script tracked resume state per-source: it built
`existing_sources: Set[str]` from `manifest.jsonl` and skipped any source
whose `source_video_id` was already present. This was fine in Phase 2A
(every source in the small subset only had ~3 segments and all were
emitted in one shot) but is wrong for the 25K-pool extension: with the
full metadata we want each of the 1614 already-processed sources to be
re-scanned so the newly-visible chunk indices (~16-17 more per source on
average, after filters) get emitted. The pre-patch behaviour would have
limited the relaunch to processing only the 2048 - 1614 = 434
not-yet-processed sources and cap the pool at roughly 3.3K + (434 ×
~10 segs/source filtered) ≈ 7.6K segments — well short of the 25K
target.

**Fixes applied (single commit).**
1. `scripts/build_panda_segment_pool.py`: after the imports, raise the
   csv field-size limit:
   ```python
   try:
       csv.field_size_limit(sys.maxsize)
   except OverflowError:
       csv.field_size_limit(2**31 - 1)
   ```
   The `try/except` guards platforms where `sys.maxsize` overflows the
   underlying C `int` (Windows / 32-bit-int builds).
2. `scripts/build_panda_segment_pool.py`: replace the per-source
   `existing_sources: Set[str]` resume index with a per-segment
   `done_chunks: Dict[str, Set[int]]` (source_video_id ->
   set(chunk_index)). Build it from `manifest.jsonl`; drop the
   source-level `if vid in existing_sources: continue` skip; inside the
   per-row segment loop, skip individual segments via
   `if seg["seg_idx"] in already_done_here: continue`. The per-source
   `max_segments_per_source` budget is initialised to
   `len(already_done_here)` so the cap acts as a TOTAL cap (existing +
   new), preserving the docstring's semantics now that sources are
   revisited. The existing per-file `dst.exists() and size > 100 KB`
   guard inside `_encode_segment` remains as a last-line defence
   against re-cuts.

**Why the fix is correct.**
- Raising the csv field limit is a no-op for the already-shipped 2m
  subset (its cells fit comfortably under 131072) and is the documented
  workaround for the full Panda-70M metadata (the Panda-70M repo's own
  loader sets `csv.field_size_limit(sys.maxsize)` for the same reason).
- The per-segment resume is strictly more permissive than the per-source
  resume AND strictly more idempotent: the set of skipped (source,
  chunk_index) pairs is exactly the set of mp4 files already present in
  `panda_segment_pool/videos/`. ffmpeg is never invoked for those pairs
  (they're filtered out before `segment_tasks` is queued), so the 3,302
  existing clips cannot be re-cut. The 3,302 rows in `metadata.csv` and
  `manifest.jsonl` are preserved verbatim by the existing manifest-read
  + rebuild logic in step 4/5.

**Expected pool size after relaunch.** Full Panda-70M averages ~18.7
segments / source across all 2048 source videos. Conservative filtered
yield (paper-grade settings: `desirable_filtering == "desirable"`,
2 ≤ duration ≤ 20 s, `matching_score ≥ 0.0`) is ~10-12 segments/source
on the long-form distribution. The 1,614 sources currently contributing
~2 segs/source (3,302 / 1,614 ≈ 2.04) will pick up ~8-10 additional
chunks each (≈13-16K new segments); the remaining 434
not-yet-processed sources contribute ~4-5K new segments. Total
projection: **~22-25K segments** after relaunch, up from the current
**3,302**. Wall: ~4-12 h on the existing 16-CPU sbatch (idempotent;
already-cut clips are zero-cost).

**Relaunch command (user runs on cluster).**
```
sbatch --account=torch_pr_36_mren \
    --export=ALL,SOURCE_METADATA=/scratch/wc3013/longcat-video-tta/datasets/panda_metadata_full/panda70m_training_full.csv \
    datasets/build_panda_segment_pool.sbatch
```
No new env-var knob is required: the per-segment-resume path is
strictly more correct than the pre-patch source-level path, so it is
unconditional.

**Verification when the job finishes.**
```
ls /scratch/wc3013/longcat-video-tta/datasets/panda_segment_pool/videos/*.mp4 | wc -l
wc -l /scratch/wc3013/longcat-video-tta/datasets/panda_segment_pool/metadata.csv
```
Expect ~22-25K mp4 files and ~22-25K + 1 (header) rows.

---

## 2026-06-09 — "TTA without text prompt" ablation: 80-job sweep queued
**Tags:** decision, methodology, in-flight, paper-narrative
**Owner:** Wenchen / agent
**Refs:**
- `sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh` (new)
- `delta_experiment/scripts/common.py` — added `add_tta_disable_caption_args` /
  `tta_caption_for` helpers
- Patched runners: `delta_experiment/scripts/run_delta_a.py`,
  `delta_experiment/scripts/run_tinylora.py`,
  `lora_experiment/scripts/run_lora_tta.py`
- Patched sbatch wrappers: `sweep_experiment/sbatch/run_sweep.sbatch`,
  `delta_experiment/sbatch/run_tinylora.sbatch` (translate
  `TTA_DISABLE_CAPTION=1` → `--tta-disable-caption`)
- Existing headline table being ablated: `paper_tables/2026-06-08_headline_1000v.md`

**Hypothesis.** AdaSteer / LoRA / TinyLoRA all train against the
flow-matching loss `MSE(model(x_t, t, encoder_hidden_states=text), v)` at
TTA time, where `text` is the eval video's caption. We do not yet know
whether the caption matters for the TTA gradient signal: the caption may
be (a) useful prior for what content to preserve in the conditioning
window, or (b) saturated noise — the visual reconstruction signal alone
might dominate. If (b), TTA gains should be unchanged when we drop the
caption, and we can claim "visual-only TTA" as a simpler primitive. If
(a), we expect a measurable gap vs the headline ADA / LORA_R8_TTA /
TL_* numbers, especially on UCF where captions are more class-y.

**Configuration.** Identical to `submit_standard_1000v_chunked.sh` modulo
two surgical changes:
1. Run IDs are suffixed with `_NOPROMPT` (e.g. `ADA_NOPROMPT`,
   `LORA_R8_TTA_NOPROMPT`, `TL_BARE_R2_NOPROMPT`, `TL_TIED_R2_NOPROMPT`).
   NOTTA is omitted because there is no TTA step to disable the caption
   for — `NOTTA_NOPROMPT` would be byte-identical to `NOTTA`.
2. Each job is exported with `TTA_DISABLE_CAPTION=1`. The sbatch wrapper
   translates this to `--tta-disable-caption` on the runner CLI; the
   runner replaces the caption with `""` (the same null-prompt convention
   used by `comparison_methods/savi_dno_longcat.py::_get_null_embeds`)
   ONLY for the call to `encode_prompt(...)` that produces the TTA-time
   `prompt_embeds`. The retrieval-augmented batch path (which we are not
   submitting here but shares the same runners) blanks neighbour captions
   too, since they all flow through the same code path. The
   `pipe.generate_vc(..., prompt=eval_entry["caption"], ...)` inference
   call is unchanged so the generated video and all downstream metrics
   (PSNR / SSIM / LPIPS / FVD / FID / VBench) see the real caption.

**Why empty string vs a special null token.** The project already uses
`prompt=""` as the unconditional / CFG-null branch (see
`comparison_methods/scripts/savi_dno_longcat.py:403`). UMT5 tokenizes
`""` to mostly-padding input ids; the resulting `last_hidden_state`
serves as the "null" conditioning. Mirroring this convention avoids a
schema drift between TTA and inference unconditional branches.

**Series dirs / merge plan.** The `_NOPROMPT` runs land in the SAME
existing series dirs as the headline standard-horizon table —
`sweep_experiment/results/panda_1000v_standard/`,
`sweep_experiment/results/ucf101_1000v_standard/`,
`delta_experiment/results/tinylora_panda_1000v_standard/`,
`delta_experiment/results/tinylora_ucf101_1000v_standard/` — so the same
`merge_chunks.py --recursive` command picks them up, and a single
`build_paper_tables.py` run rebuilds the standard-horizon table with the
ablation rows next to ADA / LORA_R8_TTA / TL_*.

**Total compute.** 4 methods × 2 datasets × 10 chunks = 80 jobs.
Per-chunk wall: 12 h for sweep methods (ADA, LORA), 16 h for tinylora;
matches headline runs.

**Workflow guard.** All four runners default `--tta-disable-caption=False`
so the existing headline runs and any future submissions through the same
sbatch wrappers without `TTA_DISABLE_CAPTION=1` are byte-identical to
their pre-patch behaviour. Verified via `DRY_RUN=1` of the new submitter:
80 sbatch lines, all with `TTA_DISABLE_CAPTION=1` in their `--export`
clauses.

---

## 2026-06-08 (latest) — Cancelled 40 t1kr_panda_* jobs that fired against 2K pool
**Tags:** in-flight, methodology
**Refs:** previous entry; user squeue paste at 12:15 AM 2026-06-09 UTC+8
showing job IDs 10615946–10616023 all on `t1kr_panda_*`.

Between the "submit now" instruction and the 25K-pool pivot, the user
fired the 40-job sweep against the 2K pool (`panda_2048_480p`). Detected
during pool-verification round-trip and cancelled before any chunk could
complete (max wall at cancel time was ~25 min; smallest chunks need ~14 h).

**Cancellation:**
```bash
scancel $(squeue -u $USER -h --format="%i %j" | awk '$2 ~ /^t1kr_panda_/ {print $1}')
rm -rf sweep_experiment/results/panda_1000v_retrieval/
```

No useful outputs are lost (no chunk completed). Next: proceed to step 1
of the 4-step pipeline (metadata download) per the previous entry.

**Workflow lesson:** when a multi-step pivot follows a launch instruction
in the same session, the cancel-cleanup commands should be paired with
the pivot recommendation to prevent racing launches. Future agents:
when you pivot, lead with `scancel` if any matching jobs are already
queued, even if you didn't think the user had submitted yet.

---

## 2026-06-08 (later) — Pivoted Panda submission to 4-step pipeline (build 25K pool first)
**Tags:** decision, methodology, paper-narrative
**Refs:**
- `sweep_experiment/reports/INDEX.md` "Pending merges and in-flight sweeps"
- Verified pool state: `panda_2048_480p` has 2048 entries embedded;
  `panda_segment_pool` has 3302 segments embedded; no 25K pool exists;
  no `panda70m_training_*.csv` metadata on disk (was cleaned up after
  the failed `build_panda_pool_10k` job in late May).

The user explicitly asked: "Can we make sure the embedding database of
25K embeddings are present for the 2 datasets?" UCF (`ucf101_pool_max`)
is at 26K. Panda is at 3.3K maximum. To match the user's stated target
and produce a paper-defensible Panda retrieval result, we need a 25K
Panda pool BEFORE submitting `panda_1000v_retrieval`.

**Pipeline pivot (replaces "submit retrieval now" plan):**

1. Re-download full Panda-70M training metadata (`datasets/panda_metadata_full/panda70m_training_full.csv`, ~2.73 GB) via `download_panda70m_full_metadata.sbatch` (gdown). Wall ~30-60 min.

2. Re-run `build_panda_segment_pool.sbatch` with `SOURCE_METADATA` pointing at the full CSV. Builder is idempotent — keeps existing 3,302 segments and adds new ones. Full metadata stores ~18.7 segs/video; matched against our 2300 source videos, projected ~25-30K segments after duration / score / desirable filters. Wall ~4-12 h on 16 CPU workers.

3. Pre-compute embeddings on the expanded pool via `precompute_pool_embeddings.sbatch`. Wall ~30 min on 1 GPU.

4. Launch the 40-job retrieval sweep with `PANDA_POOL=/scratch/$USER/longcat-video-tta/datasets/panda_segment_pool` (env-var override now supported in `submit_retrieval_1000v_chunked.sh` after today's patch). Wall ~3 days with the 2-way GPU cap.

**Net cost vs the discarded "submit now" path:** ~6-14 hours of pre-launch
work (mostly idle queueing) buys us a paper-grade 25K-pool Panda retrieval
experiment instead of a 2K-pool one that would be re-litigated.

**Why this was missed earlier:** Phase 2B job 9970342 failed in 1m52s
(probably "metadata path missing" right after `build_panda_pool_10k`'s
metadata was cleaned up to free disk). The failure was logged but the
follow-up "redownload metadata + retry" step was never queued. INDEX.md
"Pending merges and in-flight sweeps" section now exists specifically to
prevent this kind of dropped-handoff failure mode.

---

## 2026-06-08 — Panda 1000v retrieval submission queued; merge step pending
**Tags:** decision, in-flight, methodology
**Owner:** Wenchen / agent
**Refs:**
- `sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh`
- Submit command: `ONLY_DATASET=panda bash sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh`

Decision: launch the Panda 1000v batch-retrieval sweep (4 methods ×
10 chunks = 40 jobs) — this is the only paper-relevant retrieval
experiment we never ran. UCF retrieval was uninformative due to
class-block layout (see prior entry).

**Configuration as of submission:**
- Eval set: `datasets/panda_1000_480p` (1000 videos, 100 vids × 10 chunks)
- Retrieval pool: `datasets/panda_2048_480p` (2048 clips) — **NOT** the
  25K segment pool the user originally ambitioned. The 25K pool requires
  Phase 2B (full Panda-70M metadata + segment extraction) which was
  started in late May but never completed.
- AdaSteer base: `delta_steps=10`, `delta_lr=5.0e-3` (same as 1000v ADA headline)
- Methods: K5_RAND (sequential), K10_RAND (sequential), K5_SIM (similarity), K10_SIM
- Wall-time: K=5 ~14h/chunk; K=10 ~22h/chunk; with 2-way GPU cap → ~3 days

**REMINDER FOR FUTURE-ME:** When all 40 jobs finish, the merge step is:
```bash
cd /scratch/$USER/longcat-video-tta
python sweep_experiment/scripts/merge_chunks.py \
    --results-dir sweep_experiment/results/panda_1000v_retrieval \
    --recursive
python scripts/update_merged_with_vbench.py \
    --series-dir sweep_experiment/results/panda_1000v_retrieval --force
python scripts/build_paper_tables.py --regime panda_std \
    --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_panda_retrieval_followup.md
```
After merge: re-run VBench backfill if any of the 7 dims are missing,
then update `INDEX.md` row for `panda_1000v_retrieval` from `RUNNING`
to `DONE` and append a new entry to this log with the result table.

**Pool-size caveat for the paper:** if results show no gain even with
the diverse 2048-clip pool, that's still a meaningful negative result
(pool diversity was sufficient — retrieval didn't help). If results show
some gain, the followup question is whether scaling pool to 25K helps
further. We can defer the 25K build until we see the 2048-pool result.

---

## 2026-06-08 — VBench backfill complete; saturation confirmed across all 1000v regimes
**Tags:** finding, paper-narrative
**Owner:** Wenchen / agent
**Refs:**
- [`paper_tables/2026-06-08_headline_1000v.md`](paper_tables/2026-06-08_headline_1000v.md)
- VBench env: commit `4cf8b57`, sbatch convention: `4aba71f`
- 85 method dirs backfilled with 4 missing dims (motion_smoothness,
  dynamic_degree, imaging_quality, temporal_flickering)

Full 7-dim VBench is now available across all 1000v headline series. Three
findings:

1. **AdaSteer ≈ No-TTA on every metric in every regime.** PSNR / SSIM /
   LPIPS / FVD / FID / all 7 VBench dims agree to within their per-video
   noise. This is the same saturation we already saw with the binned
   per-dynamicness analysis. **The paper cannot claim AdaSteer
   distributional improvement at 1000v.**

2. **LoRA-R8 trades quality dimensions, doesn't strictly improve.**
   Consistent pattern across all 4 regimes: Aes ↑ (+0.04–0.05), Dyn ↑
   (+0.02–0.03), but IQ ↓ (−0.02 to −0.03), Subj ↓ (−0.005, Panda only).
   Worth a paragraph: "LoRA shifts the model toward perceptually-rated-as-
   prettier frames at the cost of per-frame quality and subject identity."
   Not a strict win.

3. **Long-horizon causes Subj drop (identity drift).** Subj 0.907 → 0.774
   on Panda (std → long-ctx). This is the only metric where AdaSteer and
   LoRA visibly diverge: AdaSteer preserves Subj (0.775), LoRA worsens it
   (0.757). Possible angle for the paper: AdaSteer as identity-preserving
   long-context TTA.

Combined with the per-video win/loss analysis from earlier (June 1–2),
the paper narrative becomes:
- **Population-level:** AdaSteer is net-neutral at 1000v scale.
- **Per-video:** AdaSteer wins/loses on individual videos; net-positive
  in OOD long-horizon scenarios.
- **vs LoRA:** AdaSteer has comparable distributional behaviour without
  LoRA's identity-drift cost in long context.

---

## 2026-06-08 — Batch retrieval at 1000v: UCF results uninformative; Panda not yet tested
**Tags:** negative-result, methodology, decision-needed
**Refs:**
- `ucf101_932v_retrieval/{K5_SIM,K5_RAND,K10_SIM,K10_RAND}/merged_summary.json`
- AGENTS notes from late May / early June

The 4 UCF retrieval rows in Table 2 (K5_SIM, K5_RAND, K10_SIM, K10_RAND)
are essentially indistinguishable from each other (Dyn 0.699–0.704) AND
from NOTTA (0.697). This is **not** a "retrieval doesn't work" result.
Two reasons:

1. UCF eval set and retrieval pool are both alphabetically ordered by
   class. So both `_SIM` (cosine-similarity retrieval on captions) AND
   `_RAND` (positional/sequential sampling) end up retrieving same-class
   neighbours. The K=5 batch is essentially "more samples from the same
   class", which is not what batch-retrieval is supposed to test.

2. **Panda 1000v retrieval was never submitted.** The Panda segment pool
   (`datasets/panda_segment_pool/`) was built and embedded in late May,
   but the actual retrieval-augmented TTA sweep on Panda 1000v has not
   been launched.

**Decision needed:** Submit Panda 1000v retrieval (4 methods × 10 chunks
= 40 jobs, ~70 min/dir × 4 dirs / 8 parallel = ~6 h wall) before paper
submission. This is the only experiment that could give a positive
batch-retrieval signal.

---

## 2026-06-08 — TL_TIED_R2 (Panda) and LORA_R8_TTA (UCF longhorizon) had stale partial merges
**Tags:** methodology
**Refs:** `delta_experiment/results/tinylora_panda_1000v_standard/TL_TIED_R2/`,
`sweep_experiment/results/ucf101_683v_longhorizon/LORA_R8_TTA/`

`merged_summary.json` for these two dirs had stale numbers from a
premature `merge_chunks.py` run that captured only 8/10 (TL_TIED_R2) or
2/7 (LORA_R8_TTA) chunks. Re-running merge_chunks.py + update_merged_with_vbench.py
--force fixed both. Final values now in line with peer methods (FVD 161.1
and 185.9 respectively, vs the bogus 174 and 442).

**Lesson:** Whenever the recap shows a number that doesn't match peers,
check `merged_summary.json["num_videos"]` first. Stale partial merges are
the most common source of "weird" numbers.

---

## 2026-06-05 — Eight-way concurrent backfill on courtesy partitions
**Tags:** methodology
**Refs:** sbatch commit `4aba71f`

Discovered that `--comment="preemption=yes;requeue=true"` plus
`--gres=gpu:h200:1` (no explicit `--partition`) routes jobs to courtesy
partitions (`h200_cds`, `h200_courtesy_a`) which bypass the standard
QOSMaxGRESPerUser=2 limit. Got 8 concurrent backfill jobs running in
parallel — completed 74 dirs in ~3.5 hours instead of the predicted
12–13 hours.

**Lesson for future paper-grade sweeps:** Use the courtesy-partition
sbatch convention for jobs that can tolerate preemption (anything with
`--force` idempotence or chunk-level result files).

---

## 2026-06-01 — FVD sample-size bias quantified
**Tags:** finding, paper-narrative
**Refs:** `weekly_recap_2026-06-01.md`, FVD diagnostic runs

Confirmed that 200v / 100v FVD numbers in early discovery sweeps inflate
method-level differences by ~1.2× compared to N=999. This explains why
discovery runs showed AdaSteer FVD gains of 30–50 that compress to ~1.3
at 1000v scale. **Do not cite small-N FVD differences in the paper without
the sample-size caveat.**

---

## 2026-06-01 — Eval-set drift between 200v and 1000v subsets
**Tags:** methodology, caveat
**Refs:** `weekly_recap_2026-06-01.md`

The 200v eval subsets used in early discovery work were NOT drawn from
the same population as the 1000v paper-grade subsets. PSNR differences
of ~0.5 dB between them are partly population drift, not method effects.
**For the paper, only compare methods within the same N (do not mix 200v
and 1000v rows in the same table without flagging).**

---

## 2026-05 — TinyLoRA selection (TL_BARE_R2 and TL_TIED_R2)
**Tags:** decision
**Refs:** `delta_experiment/results/tinylora_sweep/TL_*` (13 variants)

Picked TL_BARE_R2 (rank=2, n_tie=1, qkv_proj, all blocks, 20 steps,
lr=1e-3) and TL_TIED_R2 (same but n_tie=48) as the headline TinyLoRA
configs after a 13-variant discovery sweep on Panda 100v. The other 11
variants are kept in `tinylora_sweep/` as discovery rows.

---

## 2026-05 — LoRA-R8 selection as TTA baseline (LORA_R8_TTA)
**Tags:** decision
**Refs:** `submit_standard_1000v_chunked.sh` header docstring

Picked LORA_R8 (rank=8, alpha=16, all blocks, 10 steps, lr=5e-5, weight
decay 0.01, max grad norm 10) as the LoRA TTA baseline after the
`lora_rank_sweep/` discovery. Best PSNR vs the rank-1/rank-2/rank-4
variants. The previous rank-1 lr=2e-4 variant was DROPPED for catastrophic
collapse at 20 steps.
