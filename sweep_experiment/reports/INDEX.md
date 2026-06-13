# AdaSteer Experiment Index

**Purpose:** Single source of truth for "what experiments exist, where their
results live, what is paper-quality vs discovery, and what remains to be
run." Every agent / human working on this paper should read this first.

**Update rule:** Append a row whenever a new experiment series completes,
update the Status / Findings columns when re-merged. NEVER delete rows
even if results are superseded — mark them `superseded` and keep them
for audit trail.

**Owners:** Wenchen (PI) and any active agent. Last updated: 2026-06-09.

---

## Headline 1000v paper-grade experiments (the 4 we'd publish today)

| Series | Dataset | N | Frames | Methods | Status | Cluster path | Paper table | Key finding |
|---|---|---|---|---|---|---|---|---|
| `panda_1000v_standard` | Panda-70M | 999 | 28 | NOTTA, ADA, LORA_R8_TTA, TL_BARE_R2, TL_TIED_R2 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/panda_1000v_standard/`, `delta_experiment/results/tinylora_panda_1000v_standard/` | Table 1 of [`paper_tables/2026-06-08_headline_1000v.md`](paper_tables/2026-06-08_headline_1000v.md) | AdaSteer ≈ NoTTA on every metric. LoRA shifts distribution (Aes↑, IQ↓). |
| `panda_1000v_standard` + `_NOPROMPT` pairings | Panda-70M | 999 | 28 (17-frame gen) | NOTTA, ADA, ADA_NOPROMPT, LORA_R8_TTA, LORA_R8_TTA_NOPROMPT | DONE (per-frame + FVD + FID) for all 5; VBench partial (3 in-runner dims) for the 2 NOPROMPT methods; full 7-dim VBench done for prompted methods | `sweep_experiment/results/panda_1000v_standard/{NOTTA,ADA,ADA_NOPROMPT,LORA_R8_TTA,LORA_R8_TTA_NOPROMPT}/merged_summary.json` | [`paper_tables/2026-06-09_panda_std_prompt_vs_noprompt_full_metrics.md`](paper_tables/2026-06-09_panda_std_prompt_vs_noprompt_full_metrics.md) (full-metrics prompt-vs-NOPROMPT comparison + per-video ΔLPIPS tail breakdown) | TTA-time text prompt is a noise channel on this regime: both pairs sit within 0.01 PSNR / ≤0.001 SSIM/LPIPS / 4 FVD / 0.3 FID / 0.001 VBench-dim. Per-video ΔLPIPS tails: TinyLoRA tightest (~82 % within ±0.005), LoRA-r8 middle (~75 %), AdaSteer loosest (~55 %, same ordering as ΔPSNR). |
| `ucf101_932v_standard` | UCF-101 | 932 | 28 | NOTTA, ADA, LORA_R8_TTA, TL_BARE_R2, TL_TIED_R2 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/ucf101_932v_standard/`, `delta_experiment/results/tinylora_ucf101_932v_standard/` | Table 2 | Same saturation pattern. 932v not 1000v because some chunks failed. |
| `ucf101_932v_retrieval` | UCF-101 | 932 | 28 | K5_SIM, K5_RAND, K10_SIM, K10_RAND | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/ucf101_932v_retrieval/` | Table 2 | All 4 retrieval variants ≈ NOTTA. UCF class-block layout means SIM and RAND retrieve same-class neighbours. NOT a useful retrieval testbed. |
| `panda_longctx_1000v` | Panda-70M | 999 | 76 | NOTTA, ADA_S10, LORA_R8, PANDA_TL_LAST24 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/panda_longctx_1000v/`, `delta_experiment/results/tinylora_longctx_1000v/` | Table 3 | Saturated at PSNR ~12.77. Subj drops 0.907→0.774 vs std (drift effect). AdaSteer preserves Subj (0.775); LoRA worsens it (0.757). |
| `ucf101_683v_longhorizon` | UCF-101 | 683 | 76 | NOTTA, ADA, LORA_R8_TTA | DONE + VBench backfilled (2026-06-08) | `sweep_experiment/results/ucf101_683v_longhorizon/` | Table 4 | All within 0.02 PSNR. LoRA Aes↑ (0.394→0.433), IQ↓ (0.450→0.430). 683 not 1000 because original chunked submit hit class-name skip. |

---

## Missing / not-yet-run experiments (paper-blocking or paper-relevant)

| Series | Why it's needed | Cluster status | Decision |
|---|---|---|---|
| `panda_1000v_retrieval` (K5/K10 × SIM/RAND) | UCF retrieval is uninformative due to class-block layout. Panda hash-ordered pool would give a clean retrieval signal. | Pool built (`datasets/panda_segment_pool/`, ~3K segments), embeddings precomputed (commit `64f608a`). NEVER submitted. | **OPEN** — pending decision 2026-06-08. |
| 200v "gain disappears" comparison | Show research partner that small-N gains compress at scale. | Existing 26-100v discovery runs available; no actual N=200 series. | Skip or use 100v `panda_cover_candidates` as proxy. |
| Larger Panda retrieval pool (25K segments) | Original ambition: 25K segments from full Panda metadata for richer retrieval. | Phase 2A: 3K-segment pool built. Phase 2B: full-metadata download started but never completed → 25K. | Decide after Panda 1000v retrieval result. |

---

## Implemented but not yet run (recipe modifications)

Patches landed in the repo but awaiting cluster availability for verification.
Smoke-tests in this section gate scale-up to a full sweep — they fire as soon
as the cluster returns from maintenance.

| Series / wrapper | Modification | Implementation commit | Smoke-test command | Decision rule | Then |
|---|---|---|---|---|---|
| `panda_1000v_standard/LORA_R8_TTA_X0_W1.0` (single chunk × 100 videos) — wrapper [`sweep_experiment/sbatch/submit_smoke_x0_loss.sh`](../sbatch/submit_smoke_x0_loss.sh) | **Modification 1: anchor-frame x0 consistency loss** — adds `pred_x0 = noisy_target − σ·pred_v` MSE term to `compute_flow_matching_loss_conditioned`, controlled by `--anchor-x0-weight` CLI flag (default 0.0 = byte-identical to pre-patch). Per Sangare et al. CVPR 2026; rationale in [`LITERATURE_tta_recipe_modifications_2026-06-12.md §3.1`](LITERATURE_tta_recipe_modifications_2026-06-12.md). | This commit (anchor-frame x0 loss landing) | After cluster returns: `cd /scratch/wc3013/longcat-video-tta && git pull && bash sweep_experiment/sbatch/submit_smoke_x0_loss.sh` (~2 GPU h on H200) | Compare chunk_0 PSNR vs headline `LORA_R8_TTA/chunk_0`. **Scale up** if median \|ΔPSNR\| > 0.5 dB in either direction. **Move to Modification 2** if NaN grads OR \|ΔPSNR\| < 0.05 dB (loss formulation not the binding constraint). | If scale-up: 4-method × 4-λ × 10-chunk sweep (λ ∈ {0.01, 0.1, 1.0, 10.0}; ADA / LORA_R8_TTA / TL_BARE_R2 / TL_TIED_R2) ≈ 80 GPU-h per LITERATURE doc §4 priority-1 row. |
| `panda_1000v_standard/VAE_DEC_TTA_LR1e-5` (single chunk × 100 videos) — wrapper [`sweep_experiment/sbatch/submit_smoke_vae_decoder_tta.sh`](../sbatch/submit_smoke_vae_decoder_tta.sh); runner [`delta_experiment/scripts/run_vae_decoder_tta.py`](../../delta_experiment/scripts/run_vae_decoder_tta.py); dispatch `METHOD=vae_decoder` in [`run_sweep.sbatch`](../sbatch/run_sweep.sbatch) | **Modification 2: VAE-decoder-only TTA** — freezes the DiT entirely; adapts only `vae.decoder` params per video on the round-trip reconstruction loss `MSE(VAE.decode(VAE.encode(pixel_frames_train)), pixel_frames_train)` (optional LPIPS auxiliary via `--vae-tta-lpips-weight`). Decoder is snapshotted once at load time and restored from snapshot at the end of every video (no cross-video drift). Per Leng et al. ICCV 2025 (REPA-E) + Cheng et al. ICCV 2025 (LeanVAE); rationale in [`LITERATURE_tta_recipe_modifications_2026-06-12.md §3.2`](LITERATURE_tta_recipe_modifications_2026-06-12.md). Selected as the post-D1 PIVOT after the Mod 1 smoke-test returned median ΔPSNR = +0.0093 dB (below the 0.05 dB threshold; see ANALYSIS_LOG D1 verdict). | This commit (VAE-decoder TTA landing) | After cluster returns: `cd /scratch/${USER}/longcat-video-tta && git pull && bash sweep_experiment/sbatch/submit_smoke_vae_decoder_tta.sh` (~3–5 GPU h on H200; TTA step is 5–10× cheaper than a DiT-forward TTA step, so inference dominates). | **PRIMARY: scale up** if ΔPSNR > +1.0 dB on ≥3 of {`panda_0461`, `panda_0555`, `panda_0862`, `panda_0431`} (the §2.3 beneficiary cohort). **SECONDARY: scale up** if aggregate median \|ΔPSNR\| > 0.5 dB across the 100-video chunk. **Null** if neither triggers → bottleneck is not the VAE either; pivot to Mod 3 (augmentation-consistency) or document as a deep negative result with a paper subsection on "where TTA fails on video DiTs". | If scale-up: 10-chunk × {1e-6, 1e-5, 1e-4} LR sweep ≈ 30 GPU-h. Optional independent LPIPS-on arm `VAE_DEC_TTA_LR1e-5_LPIPS0.1` adds one more 10-chunk wave. |
| `panda_1000v_standard/{ADA,LORA_R8_TTA}_TTA{10,20,40,80,160}/chunk_0` + `tinylora_panda_1000v_standard/TL_BARE_R2_TTA{10,20,40,80,160}/chunk_0` (15 jobs total) — wrapper [`sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh`](../sbatch/submit_ttom_iteration_sweep.sh) | **TTOM iteration-saturation control (Track D Wave D2)** — sweeps the TTA-step count across {10, 20, 40, 80, 160} for three methods (ADA / LORA_R8_TTA / TL_BARE_R2) on Panda 1000v `chunk_0` to test whether our setting reproduces TTOM's saturate-then-degrade curve. Per `PAPER_FRAGMENT_ttom_positioning_2026-06-12.md §"Suggested control"`. | 2026-06-13 (wrapper landing under user authorization for overnight fire) | `bash sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh` (15 jobs × per-job wall 6–24 h ≈ ~125 GPU-h serial; ~24–48 wallclock h with the 2-way h200 cap) | Plot ΔPSNR / ΔLPIPS / ΔFVD vs TTA-step count per method. **Crossover at high-iter end** → shared mechanism with TTOM (over-optimization saturation). **Monotonic-flat at noise floor** → distinct mechanism (per-video noise floor). Either outcome is paper-defensible. | Update the TTOM positioning fragment with the realised curve; pre-empts the "did you just not run enough iterations?" reviewer challenge. |

---

## Active discovery / ablation experiments (not paper-grade, kept for audit)

These exist but should NOT be mixed with headline tables. They are kept to
document the methodology trail (how we picked LR / steps / target blocks).
Per-series N is small; FVD/FID values are sample-size-biased.

| Series | N | Methods | Purpose | Status |
|---|---|---|---|---|
| `panda_adasteer_ablation` | 100 | AS_CLIP_T10, AS_CLIP_T15 | CLIP threshold sweep | Discovery |
| `panda_cover_candidates` | 26 | NOTTA, DV_BARE, LORA_R8_S10 | LoRA-collapse cover | Discovery |
| `panda_longctx` | 50 | NOTTA, ADA_S10, LORA_R8 | Long-context discovery (precursor to `panda_longctx_1000v`) | Superseded by 1000v |
| `ucf_longctx` | 50 | NOTTA, ADA_S10, LORA_R8 | UCF long-ctx discovery | Superseded by `ucf101_683v_longhorizon` |
| `ucf500_lora_collapse_cover` | 30 | NOTTA, LORA_R8_S50, ADA_S10_AREG_D2 | LoRA collapse documentation on UCF | Discovery |
| `delta_a_iter_sweep`, `delta_a_lr_sweep` | 99 | DA1-DA10 | AdaSteer hyperparameter discovery | Superseded by `panda_1000v_standard/ADA` |
| `delta_b_*`, `delta_c_*` | 93-99 | DB1-DB11, DC1-DC5 | Variant family ablations | Discovery |
| `full_iter_sweep`, `full_lr_sweep` | 99 | F1-F9 | Full fine-tune ablation | Discovery |
| `lora_rank_sweep` | 99 | L1-L5 | LoRA rank sweep | Discovery |
| `tinylora_sweep` | 100 | TL_* (13 variants) | TinyLoRA discovery | Superseded by `tinylora_panda_1000v_standard/{TL_BARE_R2, TL_TIED_R2}` |

---

## Datasets and retrieval pools

### Eval sets

| Name | Cluster path | N | Notes |
|---|---|---|---|
| Panda 1000v eval | `datasets/panda_1000_480p/` | 1000 | Used for all Panda eval runs |
| Panda 100v eval | `datasets/panda_100_480p/` | 100 | Discovery |
| UCF-101 1000v eval | `datasets/ucf101_1000_480p/` | 1000 | Used for `ucf101_932v_*` runs |
| UCF-101 std eval | `datasets/ucf101_std_480p/` | (varies) | Used by `submit_retrieval_1000v_chunked.sh` for UCF retrieval |
| UCF-101 test eval | `datasets/ucf101_test_480p/` | (varies) | Older runs |

### Retrieval pools — embedding-database status

The retrieval-augmented sweeps require pre-computed `caption_embeddings.npy` +
`caption_embeddings.json` in the pool directory. Without these, `K_SIM` runs
fall back to encoding captions per-job (~30-60 s/job overhead). **Verify
embedding presence before any retrieval submission.**

| Pool name | Cluster path | Pool size (entries) | Embeddings precomputed? | Used by |
|---|---|---|---|---|
| Panda 2048-clip pool | `datasets/panda_2048_480p/` | 2048 | Yes (per submit_retrieval_1000v_chunked.sh header docstring; verify with `ls .../caption_embeddings.*`) | `panda_1000v_retrieval` (default in submit script) |
| Panda segment pool (Phase 2A) | `datasets/panda_segment_pool/` | ~3000 | Status UNCONFIRMED — verify on cluster | not yet wired into any submit script |
| Panda segment pool (Phase 2B target) | (would be `datasets/panda_segment_pool_25k/` or similar) | 25000+ | NOT BUILT — Phase 2B started late May, never completed | future: replace `panda_2048_480p` in retrieval submit script if built |
| UCF-101 max chunked pool | `datasets/ucf101_pool_max/` | ~26000 | Yes (used successfully by completed `ucf101_932v_retrieval` sweep) | `ucf101_932v_retrieval` |

**CURRENT GAP:** Panda retrieval submitted today uses the 2K-entry pool, not
25K. UCF retrieval was already on a 26K pool. If the 2K-pool Panda result
shows no gain, we still need the 25K Panda pool to fully claim "retrieval
doesn't help" — pool diversity could be the confound.

### Verify embedding-database presence (run on cluster)

```bash
cd /scratch/$USER/longcat-video-tta
for pool in datasets/panda_2048_480p \
            datasets/panda_segment_pool \
            datasets/ucf101_pool_max; do
    echo "=== $pool ==="
    if [ -d "$pool" ]; then
        ls -la "$pool"/caption_embeddings.* 2>&1 | head -5
        if [ -f "$pool/caption_embeddings.npy" ]; then
            python -c "
import numpy as np, json
e = np.load('$pool/caption_embeddings.npy')
with open('$pool/caption_embeddings.json') as f: m = json.load(f)
print(f'  shape={e.shape} dtype={e.dtype} entries={len(m) if isinstance(m, list) else len(m.get(\"captions\", m))}')"
        fi
    else
        echo "  (pool dir does not exist)"
    fi
    echo
done
```

---

## Pending merges and in-flight sweeps (UPDATE WHEN STATUS CHANGES)

| Sweep / job | Submit date | Job IDs | Expected wall | Next-step command |
|---|---|---|---|---|
| 1. Panda full metadata download (`panda_metadata_full/panda70m_training_full.csv`, 12 GB CSV / 2.6 GB ZIP) | 2026-06-08 (no-op skip) | 10616455 (COMPLETED 35s — file already on disk from June 1) | n/a | DONE — proceed to step 2. The metadata had been on disk under `datasets/panda_metadata_full/` the whole time; earlier verification looked at the wrong path. |
| 2. Panda 25K segment pool build (extends existing 3.3K pool to ~22-25K via full metadata) | 2026-06-09 (1:38 AM UTC+8 relaunch) | 10619044 (RUNNING; previous attempt 10617270 FAILED at 49s on csv field-size-limit, fixed in commit 5d565d4) | ~1-3 h on 16 CPU workers (per Phase 2A baseline); 12h hard cap; idempotent | After done: verify `ls datasets/panda_segment_pool/videos/*.mp4 \| wc -l` ≈ 22K+ and `cat datasets/panda_segment_pool/validation_report.json`, then submit step 3 |
| 3. Panda 25K embedding precompute | After step 2 | TBD | ~30 min on 1 GPU | After done: verify `caption_embeddings.npy` shape ≈ (25000+, 384), then launch step 4 |
| 4. Panda 1000v retrieval sweep (40 jobs, K5/K10 × SIM/RAND, against 25K pool) | After step 3 | TBD | ~3 days with 2-way GPU cap | Merge: `python sweep_experiment/scripts/merge_chunks.py --results-dir sweep_experiment/results/panda_1000v_retrieval --recursive`; then `python scripts/update_merged_with_vbench.py --series-dir sweep_experiment/results/panda_1000v_retrieval --force`; then `python scripts/build_paper_tables.py --regime panda_std --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_panda_retrieval_followup.md` |
| 5. Retrieval × NOPROMPT ablation, Panda only (40 jobs: 4 methods × 1 dataset × 10 chunks) | 2026-06-09 (script committed; submission GATED on rows 2+3) | TBD | ~14 h K=5 / 22 h K=10 per chunk; ~3 days wall with 2-way GPU cap | Wait for rows 2+3 to complete (25K Panda pool + caption embeddings). Verify pool: `ls datasets/panda_segment_pool/videos/*.mp4 \| wc -l` ≈ 22-25K and `ls datasets/panda_segment_pool/caption_embeddings.*`. Smoke-test: `DRY_RUN=0 NUM_CHUNKS=1 ONLY_METHODS="K5_SIM_NOPROMPT" bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh`. Full submit: `bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh`. Merge (same series dir as headline retrieval): `python sweep_experiment/scripts/merge_chunks.py --results-dir sweep_experiment/results/panda_1000v_retrieval --recursive`. Then `python scripts/update_merged_with_vbench.py --series-dir sweep_experiment/results/panda_1000v_retrieval --force`; then `python scripts/build_paper_tables.py --regime panda_std --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_panda_retrieval_noprompt.md` (reuses the NOTTA row from `panda_1000v_standard` — NOTTA has no TTA so dropping the TTA caption is a no-op). |
| 6. Standard-horizon × NOPROMPT ablation (80 jobs: 4 methods × 2 datasets × 10 chunks; methods: ADA_NOPROMPT, LORA_R8_TTA_NOPROMPT, TL_BARE_R2_NOPROMPT, TL_TIED_R2_NOPROMPT) | 2026-06-09 (smoke 1:14 AM UTC+8; full pending smoke confirmation) | Smoke: 10618645 (ADA_NOPROMPT × Panda × chunk_0); full sweep TBD | Smoke: ~8 h. Full sweep: ~5-7 days wall with 2-way GPU cap (ADA/LoRA ~8 h/chunk; TinyLoRA ~12 h/chunk) | While smoke runs, watch slurm log for `TTA no-caption : 1` (sbatch) + `TTA no-caption : True` (Python). When smoke `merged_summary.json` looks sane, full submit: `bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh`. Merge: `python sweep_experiment/scripts/merge_chunks.py --results-dir sweep_experiment/results/panda_1000v_standard --recursive` (and the UCF + tinylora series dirs — `_NOPROMPT` methods land alongside headline ADA/LORA/TL_*). Then `python scripts/update_merged_with_vbench.py` per series (`--force`); then `python scripts/build_paper_tables.py --regime panda_std --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_headline_1000v_noprompt.md` (and `--regime ucf_std`). |
| 7. Phase 0 gating feature pipeline (`scripts/sbatch/submit_per_video_feature_pipeline.sh` — Stage 1a `extract_video_features_for_tta.py`, 1b `compute_diffusion_ood_score.py`, 1c `compute_tier3_probes.py`, Stage 2 `correlate_tta_gain_with_features.py`) | 2026-06-13 (re-fire ready after bug-fix commit on `main`) | Previous overnight run errored=1000 on every video; outputs were header-only CSVs. Two independent bugs fixed in `main` on 2026-06-13 — see [ANALYSIS_LOG 2026-06-13 — Phase 0 bug-fix](ANALYSIS_LOG.md). New job IDs TBD on cluster restart. | TBD on re-fire (pre-bug-fix wall budget was ~3-6 GPU-h Stage 1a + ~6-12 GPU-h Stage 1c + ~30 min Stage 2; re-fire should match) | On the cluster, after `git pull`: **delete or rename the existing header-only CSVs first** (`rm sweep_experiment/reports/per_video_analysis/2026-06-09/{video_features,diffusion_ood_scores,tier3_probe_features}.csv` — the wrappers default to `RESUME=0` so they will overwrite cleanly anyway, but explicitly removing the broken outputs is safer because partial-row CSVs from a failed run could otherwise be picked up by the correlation Stage 2 join), then `bash scripts/sbatch/submit_per_video_feature_pipeline.sh`. |

**Pivot rationale (2026-06-08):** the original same-day plan was to submit
step 4 against the 2048-clip pool, but verification showed neither a 25K
nor any other Panda pool exists at the user's stated target size. We
pivoted to a 4-step pipeline so the actual experiment lines up with the
paper claim. Records of this pivot are in `ANALYSIS_LOG.md` (entry 2026-06-08).

**Cancellation note (2026-06-08, 12:15 AM UTC+8 next day):** the user
submitted the original 2K-pool sweep (job IDs 10615946–10616023, all
`t1kr_panda_*`) before the pivot landed. All 40 jobs were cancelled
before any chunk completed. The `sweep_experiment/results/panda_1000v_retrieval/`
directory was wiped to avoid mixing 2K-pool and 25K-pool partial outputs.

---

## Analysis tools

Stand-alone scripts that consume the per-method `chunk_*/summary.json`
or `merged_summary.json` files and emit paper-narrative artefacts. They
do NOT submit slurm jobs; the user runs them on the cluster after a
fresh `git pull`.

| Tool | Inputs | Outputs | Purpose |
|---|---|---|---|
| `scripts/plot_dynamicness_correlation.py` | `<series>/<METHOD>/chunk_*/summary.json` + `datasets/<eval>/dynamic_degree.json` | One multi-panel PNG (per-bin per-metric PSNR/SSIM/LPIPS + win-rate panel) + sidecar `.binned.json` | "Does raw per-video metric value vary with dynamicness?" Used for headline figure. |
| `scripts/analyze_per_video_tta_gain.py` (new, 2026-06-09; extended 2026-06-11) | `<series>/<METHOD>/chunk_*/summary.json` (auto-detects methods under both `sweep_experiment/results/panda_1000v_standard` and `delta_experiment/results/tinylora_panda_1000v_standard`) + `datasets/<eval>/dynamic_degree.json` + `datasets/<eval>/metadata.csv` | `per_video_gains.csv`, four PNGs (`delta_psnr_vs_{dynamicness,baseline_psnr,caption_length}.png` + `delta_psnr_histogram.png`), `summary.md` with **ΔPSNR + ΔLPIPS** tails, top-10 winners/losers, Pearson + Spearman r vs three features, **cross-method top-50-winner Jaccard matrix** (with random-overlap baseline reference line), and **sign-agreement-across-all-non-baseline-methods** tables for both ΔPSNR and ΔLPIPS (observed unanimous count vs independence expectation; the "Nx lift" number cited internally in the 2026-06-09 analysis). | "Who wins / who loses from TTA, and what video-level features predict it?" Diagnostic for the per-video subset story when population-level ΔPSNR ≈ 0. See ANALYSIS_LOG entry 2026-06-09 for the motivating "+0.68 dB chunk-0 was sampling noise" lesson; 2026-06-11 entry for the Jaccard / sign-agreement persistence refactor. |
| `scripts/compare_horizons_per_video.py` (new, 2026-06-11) | Two `per_video_gains.csv` bundles (typically `per_video_analysis/2026-06-09` for standard horizon and a long-horizon bundle produced by `analyze_per_video_tta_gain.py` against `panda_longctx_1000v`) | `summary.md` with per-method tail-breakdown side-by-side table at ±0.5/±1.0 dB (PSNR) and ±0.005/±0.01 (LPIPS) plus a one-line hypothesis verdict (`fatter both tails / shrinking both tails / asymmetric shift / inside noise band`); `side_by_side_tails.csv` long format; `overlay_dpsnr_<METHOD>.png` + `overlay_dlpips_<METHOD>.png` per shared method | "Does the long-horizon regime have fatter tails in BOTH directions vs standard horizon (the user's 2026-06-11 hypothesis), even when the population mean is identical?" |
| `scripts/analyze_per_chunk_fvd.py` (new, 2026-06-11) | `<series>/<METHOD>/chunk_*/summary.json['fvd']` (per-chunk FVD; falls back to `merged_summary.json['fvd_per_chunk']` if individual chunk files are absent). Accepts multiple series roots in one invocation. | `per_chunk_fvd.csv` (long: series, method, chunk, fvd_method, fvd_baseline, dfvd); `per_chunk_fvd_summary.csv` (wide: series, method, n_chunks, wins, losses, ties, mean/std/median/min/max ΔFVD, two-sided sign-test p); `summary.md` with per-series tables + interpretation guide ("wins ≥ 8/10 and p < 0.11 = non-trivial per-chunk FVD effect"); `boxplot_<series>.png` with per-chunk ΔFVD points overlaid on a 0-reference line | Closes the per-chunk ΔFVD sign-analysis TODO flagged in `paper_tables/2026-06-09_panda_std_prompt_vs_noprompt_full_metrics.md`. Distributional FVD needs ≥ 2 videos so per-chunk (N=100) is the right granularity for the paired sign-test, and the 10-chunk sample of ΔFVDs has structure invisible to the 1000-video global FVD. |
| `scripts/aggregate_loss_history.py` (new, 2026-06-11) | `<series>/<METHOD>/chunk_*/summary.json['results'][i]['early_stopping_info']['loss_history']` (List[(step, anchor_loss)] sampled every `check_every` steps; the per-video held-out anchor loss is the right quantity here — the per-step training loss is NOT persisted to JSON by any TTA runner, only `final_loss`). Joins per video against the same series's `NOTTA` PSNR for the ΔPSNR axis. | `per_video_loss_curves.csv` (long); `per_video_loss_summary.csv` (wide: method, video_id, n_checks, initial_loss, best_loss, final_loss, `loss_decrease_pct = (initial − best) / initial`, stopped_early, baseline_psnr, method_psnr, dpsnr, winner_band); `loss_curves_<METHOD>.png` with per-video curves coloured by winner / middle / loser ΔPSNR band; `loss_decrease_vs_dpsnr.png` per-method scatter; `summary.md` with per-method Pearson r + Spearman ρ between `loss_decrease_pct` and ΔPSNR, plus per-band group means | "Does the held-out anchor loss decrease systematically more for winning videos than for losing ones? — i.e. is the loss the runner optimises the right per-video mechanism, or is it satisfied uniformly while the resulting per-video PSNR effect comes from somewhere else?" |
| `scripts/extract_video_features_for_tta.py` (new, 2026-06-09) | `datasets/panda_1000_480p/videos/*.mp4` + `datasets/panda_1000_480p/metadata.csv` + CLIP (`openai/clip-vit-base-patch32`) + DINOv2 (`facebook/dinov2-small`) + PySceneDetect (optional) | `video_features.csv` with Tier-1 (cuts, CLIP image↔text mean/var/min, DINO temporal-L2, Laplacian variance, RGB-hist entropy) + Tier-3 diagnostic (DINO TTA↔gen-region sim, CLIP↔gen-region) per video. Idempotent on `video_id`. | Per-video feature battery scoped to TTA-visible frames (the `gen_start_frame - tta_total_frames : gen_start_frame` slice that the runners actually decode; for `panda_1000v_standard` this is `[0:48]`). Audit block at top of script documents the slice derivation across all 4 runners. Use ALONGSIDE `analyze_per_video_tta_gain.py` to feed `correlate_tta_gain_with_features.py`. |
| `scripts/correlate_tta_gain_with_features.py` (new, 2026-06-09) | `per_video_gains.csv` (from `analyze_per_video_tta_gain.py`) + `video_features.csv` (from `extract_video_features_for_tta.py`) | `correlation_table.{md,csv}` (Spearman ρ per method × feature, |ρ| highlights), `top_features_per_method.md`, `plot_<feature>.png` per Tier-1 feature, `winners_losers_by_top_feature.md`, `summary.md` with feature ranking + paper-claim recommendation | "Does any structural feature predict per-video ΔPSNR strongly enough to be a deployment-time selection rule?" Bar: |ρ| ≥ 0.2 for ≥ 2 methods. Honest fallback list of next-iteration features baked into the summary template if nothing clears the bar. |
| `scripts/per_video_difficulty_signals.py` | `datasets/<eval>/` mp4 files + optional `--gains-csv` | Per-video signals CSV (cuts, SSIM, motion, hist χ²) + correlation tables | Frame-level difficulty (cuts, motion bursts, scene changes). Complements the dynamicness axis. |
| `scripts/diagnose_long_horizon_failures.py` | NoTTA + treatment chunk dirs + dataset `metadata.csv` | Per-video deltas CSV + theme-bucket + quintile summary printed to stdout | Long-horizon AdaSteer regression diagnosis; coarse caption-keyword theme buckets. |

**Recommended invocation (Panda standard horizon):**
```bash
cd /scratch/$USER/longcat-video-tta && git pull && \
    python3 scripts/analyze_per_video_tta_gain.py \
        --series-path sweep_experiment/results/panda_1000v_standard \
        --tinylora-series-path delta_experiment/results/tinylora_panda_1000v_standard \
        --output-dir sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)
```

**Per-video feature-correlation follow-up (run after the gain analysis exists):**

> NOTE: the feature-extraction script imports `torch` / `transformers`, so
> it MUST be run inside the same conda env the TTA runners use. The
> default `(base)` conda env on the cluster does NOT have torch — activate
> `/scratch/$USER/conda-envs/longcat` first (env created by
> `env_setup/01_setup_longcat_env.sbatch`; same env activated by
> `sweep_experiment/sbatch/run_sweep.sbatch`,
> `delta_experiment/sbatch/run_tinylora.sbatch`, etc.).

```bash
cd /scratch/$USER/longcat-video-tta && git pull && \
    module load anaconda3/2025.06 && \
    source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh && \
    conda activate /scratch/$USER/conda-envs/longcat && \
    unset PYTHONHOME PYTHONPATH && \
    python3 scripts/extract_video_features_for_tta.py \
        --videos-dir datasets/panda_1000_480p \
        --captions-csv datasets/panda_1000_480p/metadata.csv \
        --tta-visible-frames auto \
        --output sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/video_features.csv \
        --device cuda && \
    python3 scripts/correlate_tta_gain_with_features.py \
        --gains-csv sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/per_video_gains.csv \
        --features-csv sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/video_features.csv \
        --output-dir sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/criteria_correlation/
```

---

## Code commits relevant to result reproducibility

| Commit | Description | Affected series |
|---|---|---|
| `64f608a` | Fix `batch_method=random` -> `sequential` in retrieval submit script | `ucf101_932v_retrieval/K*_RAND` |
| `4cf8b57` | VBench backfill env: pin opencv-python-headless==4.11.0.86, setuptools<80 | All 1000v VBench dims |
| `4aba71f` | VBench backfill sbatch: use `--gres=gpu:h200:1` + preemption comment | All 1000v VBench backfill jobs |
| `514237f` | VBench backfill submit script: propagate `PARTITION` env | (subsequent backfill submissions) |

---

## Where today's results live

- **Per-method merged summaries:** `*/results/<series>/<METHOD>/merged_summary.json` on cluster
- **Daily raw output logs:** `sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md`
- **Paper-ready tables:** `sweep_experiment/reports/paper_tables/`
- **Analysis log (decisions, findings):** [`ANALYSIS_LOG.md`](ANALYSIS_LOG.md)
- **VBench cache (compute reuse):** `/scratch/$USER/vbench-cache/` on cluster
- **Backfill targets TSVs:** `sweep_experiment/reports/vbench_backfill_targets*.tsv`

---

## Standalone stocktake / review documents

| Date | Document | Purpose |
|---|---|---|
| 2026-06-11 | [`REFRESHER_standard_vs_longhorizon_2026-06-11.md`](REFRESHER_standard_vs_longhorizon_2026-06-11.md) | Tight side-by-side refresher of standard- vs long-horizon population-level results on Panda 1000v: per-method PSNR / SSIM / LPIPS / FVD / FID / VBench in both regimes, the largest regime-to-regime deltas (e.g. NOTTA PSNR drops 17.93 → 12.77 dB and Subj drops 0.907 → 0.774 on the longer window), method-ranking preservation across regimes, and the "what's missing" pointer to the offline-investigation suite. Companion to `PLAN_offline_investigations_2026-06-11.md`. |
| 2026-06-09 | [`REVIEW_per_video_tta_suitability_2026-06-09.md`](REVIEW_per_video_tta_suitability_2026-06-09.md) | Pre-maintenance-window stocktake of where we stand on the per-video-TTA-suitability question: completed findings, hypotheses ruled out, implemented-but-not-run inventory, next-wave priority. Companion to `HYPOTHESES_per_video_tta_suitability_2026-06-09.md`. |

---

## Literature passes

| Date | Document | Purpose |
|---|---|---|
| 2026-06-12 | [`LITERATURE_tta_recipe_modifications_2026-06-12.md`](LITERATURE_tta_recipe_modifications_2026-06-12.md) | Targeted literature pass on TTA *recipe modifications* (not gating) worth queuing behind the gating experiment Phase 0–3. Ten search themes covered (TTA-for-diffusion specifically, latent-space-only TTA, anchor-frame consistency loss, MEMO/TTT-MAE augmentation-consistency, CFG-aware TTA, prompt ensembling, curriculum/annealed timesteps, meta-learning/amortized TTA, continual streaming TTA, recent CVPR/ICCV/NeurIPS 2024–2026 video-diffusion work). Selected five modifications: (1) anchor-frame x0 consistency loss — priority 1, small cost, exploits free supervisory signal from visible frames 0–47; (2) VAE-decoder-only TTA — priority 2, small/medium cost recipe pivot, tests the "VAE is the bottleneck" hypothesis; (3) augmentation-consistency MEMO/TTT-MAE — priority 3, small cost, layers on (1); (4) annealed-timestep curriculum — priority 4, small cost easy add (bundle with priority-1 sbatch wave); (5) continual streaming TTA with CoTTA-style stochastic restoration — priority 5, medium cost, conditional on 1+2 results. Each entry includes mechanism, cost, expected effect on the saturation finding, primary + secondary citations, falsification criterion, and priority. ~31 references covering 2022–2026. Honest "what the literature does NOT support" section (Theme 6 prompt ensembling is dead for our setting; more TTA iterations is contraindicated per TTOM 2026). Open questions for the user on λ-sweep, pure-latent vs VAE-decoded anchor variants, baselines for Modification 2, streaming serialisation strategy, and Modification 1 × 3 interaction. |

---

## Paper fragments

Standalone paper-defense paragraphs / footnotes / subparagraphs drafted for the AdaSteer paper's related-work section, ready for the user to edit and drop into the manuscript. One row per fragment.

| Date | Document | Purpose |
|---|---|---|
| 2026-06-12 | [`PAPER_FRAGMENT_ttom_positioning_2026-06-12.md`](PAPER_FRAGMENT_ttom_positioning_2026-06-12.md) | Paper-defense positioning paragraph distinguishing our per-video reconstructive TTA setting from TTOM (Qu et al., ICLR 2026 — [OpenReview](https://openreview.net/pdf?id=wqCwcTZsrv)), whose iteration-axis saturate-then-degrade observation is the closest "your saturation may be real" evidence in recent literature. Distinguishes on three mechanistic axes (optimization variable: three adapter families incl. non-LoRA AdaSteer vs. their fixed rank-32 cross-attention LoRA; supervisory signal: flow-matching loss on clean visible frames vs. their JSD on LLM-generated layout masks; test-time loop scope: per-video reset vs. their cross-prompt streaming memory). Honest "where comparison is close" section calls out the LoRA-on-cross-attention-DiT axis where overclaiming difference would be tactically weak. Flags one specific control (an iteration sweep on a stratified ~100-video subset, ~125 GPU-h) the paper should report rather than wave past, with the outcome-interpretation rule that either outcome (crossover seen vs. not seen) is paper-defensible. |

---

## Plans / proposals (awaiting user authorisation)

| Date | Document | Purpose | Status |
|---|---|---|---|
| 2026-06-11 | [`PLAN_offline_investigations_2026-06-11.md`](PLAN_offline_investigations_2026-06-11.md) | Login-node-only analyses that can run while the GPU cluster is in maintenance (~2026-06-12 → ~2026-06-15). Five analyses (A1–A5): long-horizon per-video gain analysis (the gap — same script as the 2026-06-09 standard-horizon bundle but pointed at `panda_longctx_1000v` / `tinylora_longctx_1000v`); side-by-side horizon comparison testing the "fatter tails in both directions" hypothesis; per-chunk ΔFVD sign analysis (the deferred TODO from the 2026-06-09 prompt-vs-NOPROMPT work) on both regimes; per-video held-out-anchor-loss-history aggregation joined against ΔPSNR; and a native-output refactor of `analyze_per_video_tta_gain.py` to write ΔLPIPS tails + top-50-winner Jaccard matrix + sign-agreement statistics into `summary.md`. CPU-only; total wall ≤ 15 min on the login node. | PLAN — login-node command sequence ready to run; companion `REFRESHER_standard_vs_longhorizon_2026-06-11.md` |
| 2026-06-11 | [`PLAN_gating_experiment_2026-06-11.md`](PLAN_gating_experiment_2026-06-11.md) | Paper-grade experimental plan for finding the optimal per-video TTA gating strategy on LongCat-Video. 20-feature master menu (Tier-1 / Tier-2 / Tier-3), five-phase protocol (data collection → univariate → multivariate → cost-aware Pareto → long-horizon validation), explicit recommendation criteria (held-out gain > 0.05 PSNR or > 0.005 LPIPS; coverage ≥ 50 %; feature compute ≤ 30 min / 999 videos), sanity controls (permutation null, leave-one-chunk-out CV, known-failure check on `panda_0098`, known-winner check on `panda_0461 / 0555 / 0862 / 0431`). | AUTHORISED — Phases 0–3 green-lit 2026-06-11 (Phase 4 gated on RECOMMENDATION.md review) |

---

## Runbooks

| Date | Document | Purpose | Status |
|---|---|---|---|
| 2026-06-12 (later) | [`RUNBOOK_friday_morning_2026-06-12.md`](RUNBOOK_friday_morning_2026-06-12.md) | Single executable Friday-morning runbook for the 2026-06-12 cluster restart. Consolidates the gating-plan Phase 0 (feature extraction + diffusion-OOD + Tier-3 probes), the offline-investigation suite A1–A5 (login-node CPU only), the NOPROMPT sweep close-out (Panda + UCF × 4 methods × 10 chunks; gated on smoke job 10618645), and the post-NOPROMPT VBench backfill into 3 parallel work tracks (Track A — GPU sbatch jobs, Track B — login-node analyses, Track C — VBench backfill after Track A's NOPROMPT jobs finish). Critical path: A2 NOPROMPT sweep → C VBench backfill → paper-table rebuild (~6–7 wallclock days). Includes pre-flight cluster-health probes, copy-pasteable commands for every step, fallback knobs (`SKIP_OOD` / `SKIP_TIER3` / `RESUME`), monitor commands, log paths, and per-track commit blocks. **Track D added 2026-06-12 (later):** Friday-afternoon recipe-modification & TTOM control with Wave D1 (Modification 1 anchor-frame x0 consistency loss smoke-test — ready, `submit_smoke_x0_loss.sh`, ~2 GPU h) and Wave D2 (TTOM iteration-saturation sweep — spec'd, ~125 GPU h, BLOCKED on missing `submit_ttom_iteration_sweep.sh` wrapper); both ungated on Phase 0–3 results. | READY — fires when cluster comes back online (expected ~2026-06-12 morning) |
