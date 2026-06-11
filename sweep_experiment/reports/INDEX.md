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
| `scripts/analyze_per_video_tta_gain.py` (new, 2026-06-09) | `<series>/<METHOD>/chunk_*/summary.json` (auto-detects methods under both `sweep_experiment/results/panda_1000v_standard` and `delta_experiment/results/tinylora_panda_1000v_standard`) + `datasets/<eval>/dynamic_degree.json` + `datasets/<eval>/metadata.csv` | `per_video_gains.csv`, four PNGs (`delta_psnr_vs_{dynamicness,baseline_psnr,caption_length}.png` + `delta_psnr_histogram.png`), `summary.md` with tails, top-10 winners/losers, Pearson + Spearman r vs three features | "Who wins / who loses from TTA, and what video-level features predict it?" Diagnostic for the per-video subset story when population-level ΔPSNR ≈ 0. See ANALYSIS_LOG entry 2026-06-09 for the motivating "+0.68 dB chunk-0 was sampling noise" lesson. |
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
| 2026-06-09 | [`REVIEW_per_video_tta_suitability_2026-06-09.md`](REVIEW_per_video_tta_suitability_2026-06-09.md) | Pre-maintenance-window stocktake of where we stand on the per-video-TTA-suitability question: completed findings, hypotheses ruled out, implemented-but-not-run inventory, next-wave priority. Companion to `HYPOTHESES_per_video_tta_suitability_2026-06-09.md`. |
| 2026-06-09 | [`HYPOTHESES_per_video_tta_suitability_2026-06-09.md`](HYPOTHESES_per_video_tta_suitability_2026-06-09.md) | Literature pass on new ideas for what predicts per-video TTA gain (parallel workstream to the review above). |

---

## Plans / proposals (awaiting user authorisation)

| Date | Document | Purpose | Status |
|---|---|---|---|
| 2026-06-11 | [`PLAN_gating_experiment_2026-06-11.md`](PLAN_gating_experiment_2026-06-11.md) | Paper-grade experimental plan for finding the optimal per-video TTA gating strategy on LongCat-Video. 20-feature master menu (Tier-1 / Tier-2 / Tier-3), five-phase protocol (data collection → univariate → multivariate → cost-aware Pareto → long-horizon validation), explicit recommendation criteria (held-out gain > 0.05 PSNR or > 0.005 LPIPS; coverage ≥ 50 %; feature compute ≤ 30 min / 999 videos), sanity controls (permutation null, leave-one-chunk-out CV, known-failure check on `panda_0098`, known-winner check on `panda_0461 / 0555 / 0862 / 0431`). | PLAN — awaiting user authorisation before execution |
