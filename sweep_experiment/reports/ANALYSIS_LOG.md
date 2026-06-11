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
