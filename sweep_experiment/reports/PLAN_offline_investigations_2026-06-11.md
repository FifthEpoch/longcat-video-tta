# Plan: offline investigations during cluster maintenance

**Date:** 2026-06-11
**Status:** PLAN — login-node-only analyses while the GPU cluster is down for maintenance through ~2026-06-12 to ~2026-06-15
**Companion:** [REFRESHER_standard_vs_longhorizon_2026-06-11.md](REFRESHER_standard_vs_longhorizon_2026-06-11.md)

## TL;DR

While the GPU cluster is in maintenance, the most valuable offline work is to **close the long-horizon per-video analysis gap** — the script (`scripts/analyze_per_video_tta_gain.py`) already exists, the inputs (`<series>/<METHOD>/chunk_*/summary.json`) are on the login node's filesystem, and only CPU + numpy + matplotlib is required. We bundle this with three companion offline analyses (horizon comparison, per-chunk ΔFVD, and per-video TTA-loss-history aggregation) for a comprehensive offline data pull. Total wall time on the login node CPU is ≤ 15 min for the whole suite; everything below this point assumes the user runs it during the maintenance window and pushes results back to `main`.

---

## Login-node data-source inventory (verified 2026-06-11 against the repo)

| Source | Path pattern | Verified by | Available on login node? |
|---|---|---|---|
| Per-chunk `summary.json` (per-video PSNR / SSIM / LPIPS, per-chunk FVD / FID / VBench) | `<series>/<METHOD>/chunk_*/summary.json` | `scripts/analyze_per_video_tta_gain.py::load_per_video_metrics` (already consumes this); `sweep_experiment/scripts/merge_chunks.py` (produces these) | **YES** — plain JSON on shared filesystem |
| Per-video records | `chunk_*/summary.json['results']` (alias `per_video_results`); one dict per video with `psnr`, `ssim`, `lpips`, `final_loss`, `early_stopping_info`, etc. | All TTA runners populate this via `save_results(experiment_summary, .../summary.json)` (e.g. `delta_experiment/scripts/run_tinylora.py` lines 727–767) | **YES** |
| Per-video held-out anchor-loss trajectory | `chunk_*/summary.json['results'][i]['early_stopping_info']['loss_history']` (List[(step, anchor_loss)] sampled every `check_every` steps, default 2) | `delta_experiment/scripts/early_stopping.py::AnchoredEarlyStopper.state` (lines 272–284) → `run_tinylora.py` line 629 stores it on `result` | **YES** (when early stopping was enabled — the default for all sweep + delta + tinylora + lora runners) |
| Merged `fvd_per_chunk` (10 per-method FVDs) | `<series>/<METHOD>/merged_summary.json['fvd_per_chunk']` | `sweep_experiment/scripts/merge_chunks.py::merge_frechet_stats` (lines 121–127) | **YES** |
| Per-step *training* loss | NOT PERSISTED — only `final_loss` (the last training-step loss) lands in the per-video result dict; the full `losses[]` list inside `optimize_*` is discarded | grep of `scripts/run_tinylora.py`, `run_delta_a.py`, `run_delta_b.py`, `run_delta_c.py`, `run_norm_tune_tta.py`, `run_film_tta.py`, `lora_experiment/scripts/run_lora_tta.py`: every runner appends `losses.append(loss.item())` but only `final_loss = losses[-1]` is saved | **NO** in JSON. Would require parsing slurm stdout (`sweep_experiment/logs/*.out` etc.), but the runners don't print per-step loss either (they only print "Train time" / mean v-norm at end), so the per-step training loss is **not recoverable** from any persisted artefact. |
| `chunk_*/fvd_fid_stats.npz` (sufficient statistics for per-chunk FVD/FID) | `<series>/<METHOD>/chunk_*/fvd_fid_stats.npz` | `delta_experiment/scripts/common.py::finalize_online_eval` (line 2937) | **YES** — already merged into `merged_summary.json['fvd_per_chunk']` so re-merging not needed; only useful if we want frame-level diagnostics beyond what's in JSON |

**Key finding on loss-history availability:** the per-video held-out anchor-loss trajectory IS file-based (in JSON), but the per-video TRAINING-loss trajectory is NOT — neither in JSON nor in slurm stdout. Path A4 below uses the anchor-loss trajectory (the held-out loss is the right quantity for the mechanism question anyway: the training loss has random-σ-per-step noise that obscures the trend).

---

## Key offline analyses

### A1 — Long-horizon per-video analysis (the gap)

- **Question:** at long-horizon (76 frames), is the population-level ΔPSNR ≈ 0 hiding a winner/loser split? And do the per-video correlations with `mean_flow` / baseline PSNR / caption length hold up at the longer window?
- **Inputs (on cluster):**
  - `sweep_experiment/results/panda_longctx_1000v/{NOTTA, ADA_S10, LORA_R8}/chunk_*/summary.json`
  - `delta_experiment/results/tinylora_longctx_1000v/PANDA_TL_LAST24/chunk_*/summary.json`
  - `datasets/panda_1000_480p/dynamic_degree.json` (shared with the standard-horizon bundle)
  - `datasets/panda_1000_480p/metadata.csv`
- **Command:** see Step "A1" in the login-node sequence below.
- **Outputs:**
  - `sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon/per_video_gains.csv`
  - `…/delta_psnr_histogram.png` (overlaid per-method ΔPSNR histograms)
  - `…/delta_psnr_vs_dynamicness.png` (ΔPSNR vs RAFT mean-flow quintile)
  - `…/delta_psnr_vs_baseline_psnr.png` (per-method scatter + regression vs baseline PSNR)
  - `…/delta_psnr_vs_caption_length.png` (ΔPSNR vs caption-words quintile)
  - `…/summary.md` (NOW includes: tails @ ±0.5/±1.0 dB, top-10 winners/losers, ΔLPIPS tails @ ±0.005/±0.01, **cross-method top-50-winner Jaccard matrix**, **sign-agreement-across-methods table for ΔPSNR + ΔLPIPS** — see A5)
- **Expected wall time:** ~5 min on a login node CPU (no model loading).

### A2 — Side-by-side standard vs long-horizon distribution comparison

- **Question (locked 2026-06-11):** does long-horizon have fatter tails in BOTH directions (more winners AND more losers) compared to standard horizon, with the |Δ|≤0.5 dB band shrinking correspondingly — even though both regimes look identical at the population mean?
- **Inputs (on cluster):**
  - `sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv` (standard horizon, exists)
  - `sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon/per_video_gains.csv` (long horizon, produced by A1)
- **Outputs:** `sweep_experiment/reports/horizon_comparison/2026-06-11/{summary.md, side_by_side_tails.csv, overlay_dpsnr_<METHOD>.png, overlay_dlpips_<METHOD>.png}`. `summary.md` includes a "fatter both tails / shrinking both tails / asymmetric shift / inside noise band" verdict per shared method at the ±0.5/±1.0 dB thresholds.
- **Expected wall time:** ~30 s on CPU.

### A3 — Per-chunk ΔFVD sign analysis (the deferred TODO)

- **Question:** does TTA improve FVD per-chunk (10 chunks × 100 videos each) more often than chance, even when the merged global FVD is statistically indistinguishable from No-TTA? FVD is distributional (it needs ≥ 2 videos) so the chunk-level paired difference is the right granularity.
- **Inputs (on cluster):**
  - `sweep_experiment/results/panda_1000v_standard/{NOTTA,ADA,LORA_R8_TTA}/chunk_*/summary.json` (+ TinyLoRA siblings under `delta_experiment/results/tinylora_panda_1000v_standard/`)
  - `sweep_experiment/results/panda_longctx_1000v/{NOTTA,ADA_S10,LORA_R8}/chunk_*/summary.json` (+ `delta_experiment/results/tinylora_longctx_1000v/PANDA_TL_LAST24/`)
- **Outputs:** `sweep_experiment/reports/horizon_comparison/2026-06-11/{per_chunk_fvd.csv, per_chunk_fvd_summary.csv, boxplot_<series>.png}` plus extra sections in this directory's `summary.md` (`analyze_per_chunk_fvd.py` writes its own `summary.md`; co-locating it with the A2 bundle prevents `summary.md` collision — see the script's `--output-dir` argument).
- **Method:** for each (series, method ≠ NOTTA), compute per-chunk ΔFVD = FVD_method(chunk_c) − FVD_NOTTA(chunk_c), report wins/10, two-sided sign-test p-value, and mean/std/median across chunks. Also produces a boxplot per series.
- **Expected wall time:** ~30 s on CPU.
- **Conflict note:** A2 and A3 both write `summary.md` into the same output dir. The plan below puts A3 under a `per_chunk_fvd/` subdirectory of the horizon-comparison bundle to avoid the collision.

### A4 — Per-video TTA loss-history aggregation (CONDITIONAL on Step-2 finding)

- **Conditional decision (resolved):** per-step training loss is **not file-based** and not in slurm stdout either, but per-video held-out anchor-loss trajectories ARE file-based (under `result['early_stopping_info']['loss_history']` in each chunk's `summary.json`). We therefore implement the **file-based** path (`scripts/aggregate_loss_history.py`) and NOT the slurm-stdout-parsing fallback.
- **Question:** for winning videos (ΔPSNR > +0.5 dB), does the held-out anchor loss systematically decrease across TTA steps? For losing videos (ΔPSNR < −0.5 dB), does it stay flat or increase? — i.e. is the held-out loss the actual mechanism, or is it satisfied on every video while the resulting checkpoint helps some and hurts others for reasons orthogonal to it?
- **Inputs (on cluster):** same as A1 (per-chunk `summary.json` files).
- **Outputs:** `sweep_experiment/reports/loss_history/2026-06-11/{summary.md, per_video_loss_curves.csv, per_video_loss_summary.csv, loss_curves_<METHOD>.png, loss_decrease_vs_dpsnr.png}`. `summary.md` includes per-method Pearson r + Spearman ρ between `loss_decrease_pct = (initial_loss − best_loss) / initial_loss` and ΔPSNR, plus per-band (winners / middle / losers) group means.
- **Expected wall time:** ~3-5 min on CPU (the long-horizon bundle has 999 × 4 methods ≈ 4 000 records, each with a ~10-checkpoint trajectory).

### A5 — Hypothesis-probing tables persisted into `summary.md` natively

This is a refactor of `scripts/analyze_per_video_tta_gain.py` (already implemented in this branch) so that its `summary.md` includes — in addition to what was there before (tail counts, top-10 winners/losers, Pearson + Spearman correlations vs `mean_flow` / baseline PSNR / caption words):

- **ΔLPIPS tail counts** at ±0.005 and ±0.01 (perceptual analog of the ΔPSNR tails; the 2026-06-09 prompt-vs-NOPROMPT analysis computed these on-the-fly and never persisted them).
- **Cross-method top-50-winner Jaccard matrix** (ΔPSNR top-50). Header shows the random-overlap baseline `k/(2N−k)` so the lift number is in the document, not just in a follow-up note.
- **Sign agreement across all non-baseline methods**, for both ΔPSNR (favourable_sign=+1) and ΔLPIPS (favourable_sign=−1). Includes per-method `p(Δ>0)`, observed unanimous count, expected unanimous count under independence, and the `Nx lift` number that the 2026-06-09 review cited internally.

A1 picks up these new sections automatically because they live inside `write_summary_md`. No separate runner needed.

---

## Login-node command sequence

Run on a single login node bash session. No GPU; no module activations needed beyond Python 3 with numpy + matplotlib (already part of the `(base)` env).

```bash
cd /scratch/$USER/longcat-video-tta && git pull origin main

# ---------- A1 — long-horizon per-video analysis  (~5 min on CPU) ----------
python3 scripts/analyze_per_video_tta_gain.py \
    --series-path sweep_experiment/results/panda_longctx_1000v \
    --tinylora-series-path delta_experiment/results/tinylora_longctx_1000v \
    --dynamicness-json datasets/panda_1000_480p/dynamic_degree.json \
    --captions-csv    datasets/panda_1000_480p/metadata.csv \
    --output-dir sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon

# ---------- A2 — side-by-side standard vs long-horizon  (~30 s) -----------
python3 scripts/compare_horizons_per_video.py \
    --standard-bundle    sweep_experiment/reports/per_video_analysis/2026-06-09 \
    --longhorizon-bundle sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon \
    --output-dir         sweep_experiment/reports/horizon_comparison/2026-06-11

# ---------- A3 — per-chunk ΔFVD sign analysis (both horizons)  (~30 s) ----
python3 scripts/analyze_per_chunk_fvd.py \
    --series-paths \
        sweep_experiment/results/panda_1000v_standard \
        delta_experiment/results/tinylora_panda_1000v_standard \
        sweep_experiment/results/panda_longctx_1000v \
        delta_experiment/results/tinylora_longctx_1000v \
    --baseline-method NOTTA \
    --output-dir      sweep_experiment/reports/horizon_comparison/2026-06-11/per_chunk_fvd

# ---------- A4 — file-based per-video loss-history aggregation  (~5 min) --
#   Long-horizon first (the primary gap; if time is tight, run only this one):
python3 scripts/aggregate_loss_history.py \
    --series-path           sweep_experiment/results/panda_longctx_1000v \
    --tinylora-series-path  delta_experiment/results/tinylora_longctx_1000v \
    --output-dir            sweep_experiment/reports/loss_history/2026-06-11/longhorizon \
    --psnr-threshold 0.5

#   Then standard horizon for the side-by-side reference (~5 min):
python3 scripts/aggregate_loss_history.py \
    --series-path           sweep_experiment/results/panda_1000v_standard \
    --tinylora-series-path  delta_experiment/results/tinylora_panda_1000v_standard \
    --output-dir            sweep_experiment/reports/loss_history/2026-06-11/standard \
    --psnr-threshold 0.5

# ---------- Commit + push the new bundles ---------------------------------
git add \
    sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon/ \
    sweep_experiment/reports/horizon_comparison/2026-06-11/ \
    sweep_experiment/reports/loss_history/2026-06-11/
git commit -m "analysis: long-horizon per-video + horizon comparison + per-chunk ΔFVD + loss-history"
git push origin main
```

---

## What this plan does NOT do

- No GPU work — the cluster compute nodes are unavailable.
- No new TTA runs of any kind.
- No re-merging of chunks (existing `merged_summary.json` files are already produced by `sweep_experiment/scripts/merge_chunks.py` and contain the per-chunk FVD list that A3 falls back to when individual `chunk_*/summary.json` files are unreachable).
- No re-running of `update_merged_with_vbench.py`, `build_paper_tables.py`, `extract_video_features_for_tta.py`, or the `correlate_tta_gain_with_features.py` pipeline — those either need fresh VBench compute on the cluster (`extract_video_features_for_tta.py`/`update_merged_with_vbench.py`) or already-completed inputs (`build_paper_tables.py`, which depends on merged summaries that are already paper-grade).
- No edits to TTA runners or sbatch wrappers — these are explicitly read-only reference material for this plan.

---

## Open questions for the user

1. **UCF series.** Should A1 also be run on `ucf101_932v_standard` / `ucf101_683v_longhorizon` to give a UCF counterpart to the Panda per-video analysis? (Default plan above: no — keep the focus on Panda where the population-level numbers are tightest.) The dynamicness JSON for UCF is at `datasets/ucf101_1000_480p/dynamic_degree.json` if so.
2. **TinyLoRA-NOPROMPT pairings.** The 2026-06-09 analysis flagged TinyLoRA NOPROMPT pairings as a TODO once the cluster returns. Should we extend A1/A4 with a `--methods` filter to ALSO include `*_NOPROMPT` siblings on the long-horizon regime if any exist on disk? (Default: include them automatically via auto-detection — the scripts already do this.)
3. **Per-chunk ΔFVD threshold for "interesting".** A4's per-method box-plot will show 10 chunks each. We currently report two-sided sign-test p-values; should the summary also surface a paired-bootstrap CI for mean ΔFVD per method? (Default: no — the sign-test is more interpretable and N=10 makes a parametric CI marginal at best.)
4. **Loss-history coverage on long-horizon.** A4 silently skips any per-video record where `early_stopping_info` is missing (e.g. methods run with `--es-disable`). Should we treat such methods as an error and abort, or continue with a "methods missing history" list in the summary? (Default: continue with a list — the long-horizon LORA_R8 / PANDA_TL_LAST24 runs may have shipped with ES disabled in some configurations.)
5. **Repo of long-horizon `_NOPROMPT` data on the cluster.** Standard-horizon `ADA_NOPROMPT` / `LORA_R8_TTA_NOPROMPT` exist on the cluster filesystem under `sweep_experiment/results/panda_1000v_standard/`, but I have no confirmation that the long-horizon counterparts (e.g. `ADA_S10_NOPROMPT` under `panda_longctx_1000v/`) exist. Please confirm before launching A1 — the script will auto-detect and include them if so, or skip them silently if not.
