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

## 2026-06-19 — H9 AdaSteer budget-grid pilot + analysis scaffolding
**Tags:** methodology, H9, in-flight
**Refs:**
- `scripts/sample_ood_quintile_videos.py` — OOD-quintile pilot list + symlink dataset
- `scripts/analyze_adasteer_budget_oracle.py` — per-video oracle over step×LR grid
- `sweep_experiment/configs/panda_1000v_adasteer_budget_grid.yaml` — full 20-config grid
- `sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh` — 12-config × 200-video pilot
- `sweep_experiment/sbatch/submit_adasteer_budget_1000v_chunked.sh` — optional full 1000v run

H9 (OOD-adaptive TTA budget) was the only open gating hypothesis after H1–H8
completed. Implemented the approved pilot scope: **12 configs** (LR 1e-3, 5e-3,
1e-2 × steps 2, 5, 10, 20) on **200 videos** (40 per OOD quintile), with the
**full 20-config** grid (adds LR 2.5e-3, 7.5e-3) documented for optional 1000v
follow-up. Fixed headline comparator: `S10_LR5e3` (same as `panda_1000v_standard/ADA`).

Cluster fire (after `git pull`):
```bash
python scripts/sample_ood_quintile_videos.py \
    --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \
    --source-dataset datasets/panda_1000_480p \
    --output-json sweep_experiment/lists/panda_ood_budget_pilot_videos.json \
    --create-dataset datasets/panda_ood_budget_pilot_480p

bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh
```

Post-merge analysis uses bootstrap CIs (same pattern as
`analyze_routing_win_magnitudes.py`). Check whether high-OOD quintiles prefer
more steps + lower LR (H9 prediction) vs the H5 falsification (higher OOD →
less ΔPSNR at fixed budget).

---

## 2026-06-14 — Gating Phase-0 Tier-1 extractors implemented (H-T1-1..4, H-T2-2/5)
**Tags:** methodology, gating-experiment, implementation
**Refs:**
- `scripts/extract_flow_shape_features.py` (H-T1-4: flow_max, flow_entropy, flow_max_over_mean on [0,48) @ 256×320)
- `scripts/extract_bpp_features.py` (H-T1-2)
- `scripts/extract_fft_features.py` (H-T1-3)
- `scripts/extract_vae_recerr_features.py` (H-T1-1)
- `scripts/derive_loss_variance.py` (H-T2-5 post-process)
- `scripts/compute_diffusion_ood_score.py` patch: `score_norm_caption_t*`, `score_norm_uncond_t*`, `mean_score_norm_*` (H-T2-2)
- `scripts/correlate_tta_gain_with_features.py` — optional `--flow-csv`, `--bpp-csv`, `--fft-csv`, `--vae-recerr-csv`, `--loss-var-csv`
- `scripts/sbatch/submit_per_video_feature_pipeline.sh` extended with stages 1d–1h

Implemented all unblocked gating-hypothesis extractors that can run on
existing `per_video_gains.csv` + `datasets/panda_1000_480p` without new
TTA sweeps. H-T1-4 now uses the correct TTA-visible window (48 frames,
not the legacy 28-frame `compute_dynamic_degree.py` path) and stores true
global `flow_max` (not p99). Correlation pipeline joins all new CSVs when
provided. Cluster fire: `git pull` then individual sbatch commands or the
updated `submit_per_video_feature_pipeline.sh` wrapper.

**Still blocked (not implemented here):** H-T2-3 full CFG-gap (extra forward
pass), H-T2-4 FLIPD/LID, Phase 1–3 analysis scripts
(`analyze_gating_univariate.py`, etc.), Phase 4 long-horizon (requires
RECOMMENDATION.md authorisation). H-T3-1/2 probe scripts exist on main
(`compute_tier3_probes.py`) — user already has job 10795485 running.

---

## 2026-06-18 — Bootstrap CIs + motion metrics for TTA-gain correlation
**Tags:** methodology, gating-experiment, implementation
**Refs:**
- `scripts/correlate_tta_gain_with_features.py` — `bootstrap_spearman_ci()`, `--bootstrap`, `--motion-csv`
- `scripts/extract_latent_motion_features.py` — `latent_temporal_l2_mean`, `pixel_mse_temporal_mean` on [0,48)
- `scripts/analyze_routing_win_magnitudes.py` — optional bootstrap CI for oracle mean uplift
- `scripts/sbatch/run_extract_latent_motion_features.sbatch`

Cluster (after `git pull`):
```bash
# Motion features (GPU)
sbatch scripts/sbatch/run_extract_latent_motion_features.sbatch

# Correlation with bootstrap + all aux CSVs
BOOTSTRAP=1 FLOW_CSV=... MOTION_CSV=... sbatch scripts/sbatch/run_correlate_tta_gain.sbatch

# Or full pipeline with bootstrap enabled
BOOTSTRAP=1 bash scripts/sbatch/submit_per_video_feature_pipeline.sh
```

`dino_temporal_l2_mean` remains in `video_features.csv` (Tier-1, online-actionable).
`flow_shape_features.csv` joins via existing `--flow-csv` (default in pipeline).

---
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

---

## 2026-06-27 — Per-video VBench++ cross-metric agreement script
**Tags:** methodology, in-flight
**Refs:**
- `scripts/analyze_per_video_vbench_agreement.py`
- `scripts/run_panda_vbench_agreement.sh`
- Output target: `sweep_experiment/reports/per_video_analysis/YYYY-MM-DD/vbench_agreement/`

After Panda 1000v retrieval VBench backfill completed (K5/K10 × SIM/RAND),
population means still show ≈0 ΔPSNR and mixed VBench shifts (Aes/Dyn↑,
IQ↓). Next diagnostic: **per-video** win/tie/loss on all 7 VBench++ dims
and **cross-metric agreement** with ΔPSNR / ΔSSIM / ΔLPIPS (FVD remains
population-only).

Run on cluster:
```bash
bash scripts/run_panda_vbench_agreement.sh
```
Then paste key tables from `vbench_agreement_summary.md` here once generated.

---

## 2026-06-28 — Oracle VBench++ suite + metric cache audit
**Tags:** methodology, oracle, efficiency
**Refs:**
- `scripts/per_video_metric_store.py` — shared wide-table loader + fingerprint cache
- `scripts/analyze_oracle_vbench.py` — method + budget config oracle on VBench++
- `scripts/plot_cross_metric_correlations.py` — OOD/ΔPSNR/ΔVBench heatmaps + method-level ΔFVD plot
- `scripts/run_oracle_analysis_suite.sh` — single entry point (reuses cache)

**Budget-oracle FVD status:** NOT computed. Pilot + 1000v budget runs used
``NO_SAVE_VIDEOS=1`` → ``run_budget_oracle_fvd`` job 11457714 failed (0 symlinks).
PSNR oracle uplift confirmed (~+0.85 dB pilot mean, ~+1.1 dB Q5 within-quintile);
**FVD ceiling for config-sliding oracle is unknown** until mp4s saved and
``run_budget_oracle_fvd.py`` succeeds.

**Method-oracle FVD (done):** job 11061632 → oracle_best_psnr FVD **149.57** vs
NOTTA **155.94** (−6.37).

**Cache / duplicate-work audit:**
| Pattern | Fix |
|---|---|
| Multiple scripts re-read ``per_video_vbench_gains.csv`` + OOD | ``load_or_build_wide_table()`` writes ``metric_cache/wide_metrics.csv`` |
| ``load_per_video_vbench`` per method in loops | Agreement script already loads once; budget VBench oracle loads per grid run only when ``--budget-series-root`` set |
| ``correlate_*`` + ``magnitude`` + ``oracle`` in one session | ``run_oracle_analysis_suite.sh`` shares ``--cache-dir`` |
| Budget FVD + method FVD | Separate symlink dirs; do not re-run ``eval_fvd`` if ``fvd.json`` exists (use ``--skip-build``) |
| VBench chunk join | Fixed in ``c5b6354`` (anchor-id alignment); all downstream scripts assume that CSV |

Run on cluster:
```bash
git pull
bash scripts/run_oracle_analysis_suite.sh
# Budget FVD ceiling (requires NO_SAVE_VIDEOS=0 re-run):
python3 sweep_experiment/scripts/run_budget_oracle_fvd.py \
  --series-root sweep_experiment/results/panda_ood_budget_pilot \
  --gt-cache gt_caches/panda_1000_longcat.npz
```

---

## 2026-06-30 — Pre-experiment oracle + VBench++ suite COMPLETE (budget FVD pending)
**Tags:** finding, oracle, VBench++, negative-result, in-flight
**Refs:**
- Cluster tag: `sweep_experiment/reports/per_video_analysis/2026-06-30/`
- Snapshot: `sweep_experiment/reports/local_archive/2026-06-30/SNAPSHOT.md` (**gitignored**, laptop only)
- Interpretation: [`experiment_outputs/2026-06-30.md`](experiment_outputs/2026-06-30.md)
- Dump script: `scripts/dump_analysis_reports.sh`
- Mp4 re-run jobs: **12082901–12082926** (`NO_SAVE_VIDEOS=0` budget pilot)

**Completed on cluster (N=999):**
| Deliverable | Key result |
|---|---|
| Method PSNR oracle | **18.287 dB** (+0.35 vs always-ADA) |
| VBench-total oracle (upper bound) | **0.776** mean total |
| Method FVD oracle | **149.57** vs NOTTA ~155.9 |
| Budget PSNR oracle (12-config grid) | **18.779 dB**, SSIM 0.6497, LPIPS 0.3281 |
| OOD → ΔAes (LoRA/retrieval) | ρ **−0.27 to −0.30** → supports **skip-gate**, not H5 |
| LoRA/retrieval Aes magnitude | ~93% win @ +0.05; cancel_ratio 2.0–2.5 |
| ΔPSNR vs ΔVBench | ρ ≈ 0.02–0.06 → **no predictive link** |

**Still open:** Budget-oracle **FVD** (mp4 jobs running; manifest still 0 symlinks).
Budget per-video VBench oracle **blocked** by `COMPUTE_VBENCH=0` on grid runs.

**Script fixes applied on cluster (not yet on GitHub main):**
- `analyze_oracle_vbench.py`: add `load_per_video_vbench` import
- `plot_cross_metric_correlations.py`: remove stray `arrays = ...` line in `correlation_matrix`

**Standard handoff command:**
```bash
bash scripts/dump_analysis_reports.sh 2026-06-30
```

---

## 2026-07-05 — Budget routing experiment suite @ N=200 (13 methods)
**Tags:** finding, negative-result, decision, H9, routing
**Refs:**
- `scripts/run_budget_routing_experiments.py`, commit `056edf8`
- Results: `sweep_experiment/reports/per_video_analysis/2026-07-05/budget_routing_experiments/`
- Table: `sweep_experiment/reports/paper_tables/2026-07-05_budget_routing_experiments_N200.md`

After linear VBench-total router failed bootstrap CI (~9% captured, includes 0),
ran 13 CPU routing experiments on existing 200v × 12-config pilot (no new videos).

**Total-VBench objective (comparable rows):**
- Best: **proxy_psnr_all 11.5%** captured (+0.016 vs fixed) — not deployable (needs all-config PSNR).
- Best deployable-ish: **probe_simulated 9.8%**, **baseline_linear 9.0%** — within noise of quintile gate (~8%).
- Nonlinear / pairwise / best-of-3 PSNR: **≤0% or negative** captured.
- Oracle ceiling unchanged: **+0.1402** mean VBench total (~100% captured).

**Per-dim routing:** `dim_imaging_quality` shows 98% captured on **IQ scale only**
(bug/footgun: dim trainers evaluate on dim matrix, not VBench total — do not cite as total-VBench win).
Other dims (Aes/Dyn/Subj) show 0% on their scales with negative policy gains.

**Decisions:**
1. **999v × 12 for total-VBench routing training: NO-GO** — no method separated from linear; CIs would remain overlapping.
2. Paper narrative: **oracle real, deployable routing hard**; PSNR–VBench decoupling confirmed at routing layer too.
3. Optional follow-up: re-score dim-router **picks** on VBench total (cheap offline); real probe-and-route needs inference not simulation.

**Known artifact:** `routing_experiments_bootstrap.md` baseline 18.9% is stale OOF from first failed Slurm submit; trust summary **9.0%**.

---

## 2026-07-05 — Recommended five-experiment program complete @ N=200
**Tags:** finding, negative-result, decision, H9, routing
**Refs:**
- `scripts/run_recommended_five_experiments.py`, commit `418180d`
- Results: `sweep_experiment/reports/per_video_analysis/2026-07-05/recommended_five_experiments/`

Ran the post-linear-router **five-experiment plan** 1:1 (Exp1 probe-and-route simulation,
Exp2 ΔDyn router, Exp3 pairwise, Exp4 NR proxy best-of-3, Exp5 stub).

**Results vs success bars:**
| Exp | Best result | Bar | Verdict |
|---|---|---|---|
| Exp1 probe | ridge 3-way **12.1%** total; commit **2.9%** total / **33%** Dyn | >25% total | **FAIL** total; Dyn commit partial |
| Exp2 ΔDyn | in-sample total **4.9%**; OOF ΔDyn negative headroom | beat 9% linear | **FAIL** on total VBench |
| Exp3 pairwise | −7.4% / −0.8% | — | **FAIL** |
| Exp4 NR proxy | −3% to −5.1%, Kendall τ≈0 | rank oracle | **FAIL** |
| Exp5 IQ-TTA | skipped | — | needs GPU + code |

**Decisions:**
1. **999v × 12 routing training: NO-GO** (confirmed across linear, nonlinear, probe, dyn, NR).
2. **GPU probe-and-route (Exp1 real inference): LOW ROI** unless chasing ~12%→15% marginal; simulation already uses probe PSNR.
3. **Dyn-only routing:** captures Dyn in-sample but **does not lift total VBench** — do not pivot paper to Dyn routing for population metrics.
4. **Exp5 IQ-constrained TTA** remains separate track for LoRA/retrieval IQ frontier (not budget-grid routing).

---

## 2026-07-05 — Gain prediction exp6–12: exp7 best honest OOF (12.8%); exp9 inflated
**Tags:** finding, negative-result, decision, routing
**Refs:**
- `scripts/run_vbench_gain_prediction_experiments.py` (commit 5a67a7a; exp9 OOF fix follow-up)
- `sweep_experiment/reports/per_video_analysis/2026-07-05/vbench_gain_prediction_experiments/`
- `sweep_experiment/reports/paper_tables/2026-07-05_vbench_gain_prediction_experiments.md`

Seven CPU experiments on pilot N=200 (12 AdaSteer configs). Oracle headroom +0.140 unchanged.

| Method | Captured % (total VBench) | Notes |
|---|---:|---|
| exp7 gain-probe ridge | **12.8** | Best **deployable** OOF; +0.7pp vs Exp1 3-way |
| exp11 tier3+probe 3-way | 12.1 | Tie prior best |
| exp10 DOVER proxy | 18.4 | **Upper bound** — GT Aes+IQ on S2/S10 probes |
| exp9 multitask Aes+IQ | 45.1† | **Invalid** — in-sample eval bug (not OOF) |
| exp6 kNN | 1.2 | Fail |
| exp8 abstain | −0.8 | Fail |
| exp12 trajectory | −0.2 | Fail |

**Finding:** Multitask proxy target (0.428·Aes+0.572·IQ) looked like a breakthrough at 45% but used in-sample ridge picks for total-VBench eval. Correct pipeline: OOF ridge on proxy → eval picks on total (fix pushed; rerun needed).

**Decision:** Still **NO-GO** on 999v×12 routing for total VBench (<25% bar). exp10 suggests probe+DOVER path may reach ~15–18% if frame-level proxy works — optional GPU follow-up, not scale-up.

---

## 2026-07-05 — exp9 OOF corrected: 7.6% total (98% on-proxy only)
**Tags:** finding, negative-result, routing
**Refs:** exp9 rerun post commit `6f6a75a`

Corrected exp9: **7.6%** total-VBench captured (17% match); **98%** on fused Aes+IQ proxy target. Proxy routing is excellent on-proxy but worse than exp7 (12.8%) on total. exp7 remains best deployable.

**In flight:** Track B Panda 1000v retrieval; Track C DOVER probe routing (exp13).

---

## 2026-07-05 — Track C (DOVER probe routing) cancelled by user
**Tags:** decision, routing
**Refs:** Slurm jobs submitted 2026-07-05 via `submit_tracks_b_and_c.sh`; cancelled same day.

User decided DOVER-on-probe routing (exp13) is **not worth GPU time** given routing NO-GO at ~12.8% and exp10 upper bound only ~18%. Track B (Panda 1000v retrieval) left running.

Cancel on cluster:
```bash
squeue -u $USER -h -o '%i %j' | awk '/dover/ {print $1}' | xargs -r scancel
```

---

## 2026-07-05 — Panda 1000v retrieval complete: SIM≈RAND null @ 999v
**Tags:** finding, negative-result, retrieval, paper-narrative
**Refs:** `paper_tables/2026-07-05_panda_1000v_retrieval.md`, `results/panda_1000v_retrieval/`

PSNR 17.87–17.90 (vs ADA 17.94); FVD 155–162 (vs ADA 153.4). SIM≈RAND (≤0.03 dB). Aes~0.442 (LoRA-like; confirm 7-dim). Retrieval not a headline win; 25K pool deprioritized.

---

## 2026-07-05 — Panda retrieval 7-dim VBench confirms LoRA-like tradeoff
**Tags:** finding, retrieval, paper-narrative
**Refs:** updated `paper_tables/2026-07-05_panda_1000v_retrieval.md`

Full backfill: all 4 methods have 7 dims × 999v. VB total 0.778–0.780 (SIM≈RAND). vs ADA: Aes +0.046, IQ −0.034, Dyn +0.03 — same sign pattern as LORA_R8. PSNR/FVD still do not beat single-video ADA. **Retrieval chapter closed for paper.**

---

## 2026-07-06 — Wave-1 predictor screen: NO-GO GPU; deployable cap ~13%
**Tags:** finding, negative-result, decision, routing, H9
**Refs:** `paper_tables/2026-07-06_wave1_predictor_screen.md`, `per_video_analysis/2026-07-06/wave1_predictor_experiments/`, commit fixing decision logic

Ran 7 CPU experiments on pilot N=200 before bed. **Best deployable:** exp16 kNN probe manifold **13.0%** captured (≈ exp7 ridge 12.8%). **Ceiling:** exp14_full **17.5%** using GT VBench Aes/IQ/Dyn on probe outputs — same non-deployable class as exp10 (18.4%). Probe-only PSNR+SSIM routing (exp14_deploy) **2.8%**. Tail-only gate: overall 1.0%, tail subset 24.1% @ 15% apply (below 30% GO bar). Per-dim fuse 5.8%. Feature screen (exp19): only flow×flickering pairs pass |ρ|≥0.2.

**Decision:** **NO-GO** Wave-2 GPU tonight (VideoAlign / CFG-gap / 999v retrain). Auto `wave1_decision.json` falsely GO'd on ceiling exp — corrected in script to split deployable vs ceiling. Paper line unchanged: oracle headroom real (+0.14 mean); honest offline routing ~13%; GT-probe ceiling ~17–18%.

---

## 2026-07-06 — VAE latent profile routing: null @ N=200 (overfit when stacked)
**Tags:** finding, negative-result, routing, H9, VAE
**Refs:** `vae_latent_profile_features.csv`, `vae_latent_profile_router/summary.md`, commit `766f48e`

Extracted **130-d** LongCat-VAE latent profiles (full/context/target pools on TTA-visible [0:48)) and re-ran OOF ridge budget router vs exp7 baseline. **baseline_exp7:** 12.8% (sanity match). **vae_profile_probe** (130 VAE + probe only): **12.2%** (−0.6pp). **vae_profile_full** (Phase-0 + VAE + probe, 177 feats): **4.2%** (−8.6pp) — classic small-N overfit; more dims hurt.

**Decision:** **CLOSED** VAE hand-pooling path for total-VBench routing. Do not scale 999v VAE-profile extraction for routing. Remaining honest ceiling is still **probe outputs scored by a learned quality model** (exp10/exp14_full ~17–18%), not richer latent CSVs.

---

## 2026-07-07 — Structured blocks A/B/C: video/caption dominates @ 20.8%
**Tags:** finding, routing, deploy, OOD, positive-result
**Refs:** `deploy_strict_router/summary.md`, `paper_tables/2026-07-07_deploy_router_structured_blocks.md`

OOF ridge ablation @ N=200 with blocks **A** (9-d video/caption), **B** (12-d diffusion-OOD), **C** (130-d VAE). **A alone: 20.8%** captured (+0.0291 vs fixed). **A+B (OOD allowed): 18.9%** (+0.0265, best match 21.0%). **C alone: 9.7%** (prior headline, now superseded). **B alone: 4.9%**. **A+B+C: 10.1%** (overfit). OOD adds match rate but **does not beat A** on captured headroom.

**Decision:** **Promote Block A (`video_caption_only`) as default deploy router**; use **A+B** when frozen DiT OOD pass is acceptable. Retire VAE-only and 51-d lab bundles for product narrative. Still **below 25%** internal bar but **~2×** prior best honest router.

---

## 2026-07-07 — Deploy-strict VAE router: 9.7% @ N=200 (headline deploy)
**Tags:** finding, routing, deploy, VAE, positive-result
**Refs:** `deploy_strict_router/summary.md`, `paper_tables/2026-07-07_deploy_strict_router_vae_only.md`, commit `1e163b9`+

Re-ran OOF ridge config picker with **only** `vae_latent_profile_features.csv` (130-d, LongCat `encode_video` on input video). **No** CLIP/DINO/OOD/Tier-3/probe/TTA-side metrics. **Result:** **9.7%** oracle headroom captured, **+0.0136 vs fixed S10**, 16.5% oracle-config match — **≥** the 51-d lab router (9.0%, +0.013) with a strictly inference-compatible feature set.

**Decision:** ~~Promote `vae_inference_embedding` as headline deploy router.~~ **Superseded 2026-07-07** by Block A @ 20.8%. VAE-only result stands as ablation (9.7%).

---

## 2026-07-06 — Deploy-strict router: VAE inference embedding ONLY (pending)
**Tags:** methodology, routing, deploy, VAE, pending
**Refs:** `run_deploy_strict_router_experiments.py`, `submit_deploy_strict_router.sh`, `paper_tables/2026-07-06_deploy_strict_router_PENDING.md`

User tightened deploy bar: router input = **only** the LongCat-VAE latent profile already computed for inference (`vae_latent_profile_features.csv`, ~130-d). **No** video_features.csv (CLIP/DINO/cuts), **no** Tier-3/OOD/probe/TTA-side metrics. Offline ridge labels still use pilot 12-config VBench matrix (calibration only). CPU eval: `vae_inference_embedding`. **Results pending** cluster run after push.

---

## 2026-07-07 — Cross-metric router eval (PSNR/SSIM/LPIPS/FVD) — script added, pending run
**Tags:** methodology, routing, metrics, pending
**Refs:** `scripts/analyze_deploy_router_aux_metrics.py`, `submit_deploy_router_aux_metrics.sh`, `paper_tables/2026-07-07_deploy_router_aux_metrics_PENDING.md`

User asked whether VBench-trained routers (Block A @ 20.8%, Block C @ 9.7%) also move **PSNR/SSIM/LPIPS** when we apply the OOF-predicted config per video. Added CPU script: re-run OOF ridge → lookup per-video metrics from existing `panda_ood_budget_pilot` outputs (no new generation). Reports mean policy vs fixed/NOTTA/oracles + **metric-specific captured %** + Spearman ρ(VBench gain, ΔPSNR). **FVD/FID:** per-video lookup invalid; script builds symlink policy dirs + optional `eval_fvd.py` (`RUN_FVD=1`). **Results pending** cluster CPU job; FVD pending mp4 availability.

**Decision:** Run CPU analysis first; if VBench captured % ≫ PSNR captured %, narrative = router optimizes perceptual VBench dims, not pixel fidelity. FVD row is the honest distributional check vs fixed S10.

---

## 2026-07-09 — Cross-metric router eval: VBench routing ≠ PSNR (CONFIRMED)
**Tags:** finding, routing, metrics, negative-result, positive-result
**Refs:** `deploy_router_aux_metrics/summary.md`, `paper_tables/2026-07-09_deploy_router_aux_metrics.md`, commit `7eed702`

OOF router-selected configs @ N=200, metrics looked up from existing grid outputs (no new gen). **Block A:** 20.8% VBench captured but **+0.009 dB PSNR** (1.2% PSNR-oracle headroom), ρ(VB gain, ΔPSNR)=**0.10**. **Oracle VBench** only +0.027 dB PSNR (3.5% cap) vs **oracle PSNR** +0.748 dB — VBench-optimal configs in this grid are not PSNR-optimal. **Block C:** −0.046 dB PSNR. SSIM/LPIPS slightly worse than fixed for routers. Fixed FVD 331.2 / FID 63.4; router FVD not run.

**Decision:** PI/paper story = route for **VBench perceptual bundle**, not reconstruction. Do not claim PSNR wins from VBench-trained router. Optional follow-up: `RUN_FVD=1` if mp4s exist; compare router symlink FVD to fixed 331.2.

---

## 2026-07-09 — PSNR-targeted router experiment (9-d Block A, pending)
**Tags:** methodology, routing, PSNR, pending
**Refs:** `scripts/run_deploy_psnr_router.py`, `submit_deploy_psnr_router.sh`

User asked whether 9-d handcrafted inputs can route for **PSNR gain** (cross-metric showed VBench router +0.009 dB). **Clarification:** poor PSNR transfer is primarily **objective mismatch** (VBench oracle +0.027 dB PSNR vs PSNR oracle +0.748 dB), not proven bad features. Added deploy-strict experiment: **same 9-d Block A**, ridge predicts **PSNR per config**, argmax PSNR. Compare PSNR captured % vs VBench router (1.2% PSNR / 20.8% VB). **Results pending** cluster run.

---

## 2026-07-09 — PSNR-targeted router: objective tradeoff confirmed @ N=200
**Tags:** finding, routing, PSNR, objective-tradeoff
**Refs:** `deploy_psnr_router/summary.md`, `paper_tables/2026-07-09_deploy_psnr_router.md`, commit `90c2ead`

Same 9-d Block A features; ridge target switched to **PSNR per config**. **Result:** +0.0539 dB vs fixed (**7.2%** PSNR oracle captured, 15.5% match) vs VBench router +0.009 dB (**1.2%** PSNR cap). **VBench side effect** only **5.6%** captured (vs **20.8%** VB-targeted). **Conclusion:** input format was not the PSNR problem — **wrong training objective** was; but 9-d still weak in absolute PSNR terms (7.2% cap). **Cannot maximize VB and PSNR with one 9-d picker.**

**Decision:** Headline deploy router stays **VBench-targeted Block A**. PSNR-targeted run is ablation / tradeoff evidence only.

---

## 2026-07-10 — VBench vs PSNR router pick alignment @ N=200
**Tags:** finding, routing, objective-tradeoff, alignment
**Refs:** `router_objective_alignment/summary.md`, `paper_tables/2026-07-10_router_objective_alignment.md`, commit `e3835f9`

OOF pick comparison (same 9-d Block A). **Pick agreement 12.5%** (25/200); **oracle agreement 15%**; when oracles agree routers agree only **10%**. Config Jaccard **0.75**. **But** realized metrics across picks: ρ(VB)=**0.995**, ρ(PSNR)=**0.987** — objectives diverge in **config label** space, not **outcome** space (flat local grid). On disagreeing videos, each router wins its own metric only **51–55%** (near coin-flip). Top agree pair: `S20_LR1e2`.

**Decision:** Narrative = routing escapes fixed S10 into a better grid **region**; fine objective (VB vs PSNR) swaps among near-tie configs. Supports keeping VB headline while explaining low PSNR transfer.

---

## 2026-07-10 — Budget 1000v pool audit: segment_pool @ 29,577 is the source
**Tags:** methodology, routing, scale-up, pending
**Refs:** `cluster_audit_budget_1000v_pools.sh`, user paste `/tmp/budget_1000v_audit.txt`

Cluster audit confirms **`datasets/panda_segment_pool`**: **29,577** mp4 + **caption_embeddings.npy (29577×384)**. `panda_pool_10k` empty. OOD CSVs exist only for **panda_1000 (999)** and **pilot (200)** — **not** segment pool. No `vae_latent_cache`. Partial `panda_ood_budget_1000v` (3 runs) used **`panda_1000_480p`**, not OOD-stratified pool — **do not continue** for router scale-up.

**Decision:** (1) GPU-score OOD on segment pool → (2) `sample_ood_quintile_videos.py --per-quintile 200` → `panda_ood_budget_1000v_480p` → (3) precompute router features + **VAE cache** (code TBD) → (4) submit **12-config** pilot grid @ 1000v to `panda_ood_budget_1000v` (new OOD-stratified series).

---

## 2026-07-11 — Preview 1000v from partial segment-pool OOD (~6K scored)
**Tags:** methodology, routing, scale-up, preview
**Refs:** job `13325919`, `sample_segment_pool_ood_preview_1000v.sh`

While full 29K OOD scoring runs, **~5885+ scored rows** suffice for `--per-quintile 200` (1000 total). Quintiles computed on **scored prefix only** (canonical `video_id` sort order — not random sample of pool). Acceptable for **router N=1000 preview** vs N=200 pilot; final paper set should re-sample from complete CSV.

**Decision:** Use `panda_ood_budget_1000v_preview_{480p,results,list}` — distinct from stale `panda_ood_budget_1000v` (3-run partial on `panda_1000_480p`). Re-sample final set when `wc -l` → 29578. Pipeline: `scripts/run_preview_1000v_pipeline.sh` + `submit_deploy_router_1000v_preview.sh`.

---

## 2026-07-14 — TTA runner audit: unused val holdout removed (affects all budget-grid numbers)
**Tags:** methodology, finding, decision
**Refs:** commit pushing run_delta_a/b/c, run_film_tta, run_norm_tune_tta, run_lora_tta, run_full_tta; audit in `experiment_outputs/2026-07-14.md` (13:20)

Expert ML audit of the shared TTA plumbing (`common.py`, `frame_window.py`, `early_stopping.py`) and all 8 runners. **No ground-truth leakage:** TTA window is strictly pre-anchor `[gen_start-tta_total, gen_start)` (explicit clamp), the conditioned flow-matching loss noises/scores only the target latents (cond tokens clean at t=0), generation conditioning comes from the eval clip's observed prefix (`training_entries[0] == eval_entry`), and future GT is read only post-generation for metrics (aligned `gen_output[num_cond:]` ↔ GT from `gen_start`).

**Finding (fixed):** `split_tta_latents` unconditionally carved a 25% val holdout via `es_holdout_fraction`, but the budget grid runs with `ES_DISABLE=1` and `anchor_reg_weight=0`, so the holdout was never consumed — every runner adapted on only ~75% of the observed frames. Batch/retrieval paths (`cl, tl, _`) discarded val outright, wasting it too. **Fix:** holdout is now `0.0` unless val will actually be used (single-video paths gate on `early_stopper is not None or anchor_reg_weight>0`; batch paths pass `0.0`).

Also fixed a delta-a-only inefficiency: it re-decoded the eval clip from disk for augmentation despite already holding it (now cached on CPU and reused). No numeric effect.

**Decision:** AdaSteer/LoRA/full budget-grid numbers produced BEFORE this commit trained on 75% of frames and are superseded. The pending preview-1000v **resweep** runs with the fix, so the paper's 1000v budget-grid numbers will reflect full-data adaptation. Any earlier pilot (N=200) budget numbers should be re-derived or explicitly caveated if cited alongside post-fix numbers. Do NOT mix pre- and post-fix budget-grid rows in the same table.

---

## 2026-07-14 — Defer 1000v budget grid to full-pool OOD resample; skip preview resweep
**Tags:** decision, methodology, routing, scale-up
**Refs:** `run_preview_1000v_pipeline.sh scope` output (this date), OOD job 13491658

`scope` on `panda_ood_budget_1000v_preview`: the 6 S10/S20 configs are 100% aligned to reference `S10_LR1e3` (997 videos); the 6 S2/S5 configs overlap only **11.5%** (115) with `∈retain=115` — every chunk ~1% overlap, i.e. they ran on the stale pre-symlink-fix video set. Pure-alignment rerun scope would be **6 configs / 60 jobs**.

**But** the aligned S10/S20 results predate the holdout fix (commit `29af8a2`) → they trained on 75% of adaptation frames. Rerunning only S2/S5 under the fixed code (100%) would produce a **mixed-protocol grid** (confounded per-video config comparison); a consistent grid would need all 12 → 120 jobs on a set that is discarded anyway (preview was sampled from the ~6K scored **prefix** of the segment pool, not full-pool quintiles).

**Decision (user):** WAIT for the full **29,578**-line segment-pool OOD scoring to finish (**19,512** as of 13:51; job 13491658 RUNNING ~11h; ~291 videos/h ⇒ ~1.5 day ETA). Then draw the FINAL 1000v set from the complete pool (correct quintile edges), build a **guarded** dataset, and run all 12 configs **once** under the fixed holdout protocol. The prefix-sampled preview is discarded — **do NOT resweep it**. Pipeline already validated by the N=200 pilot + partial preview (which caught the symlink instability and holdout bug), so nothing blocks on the preview router.

**Next when OOD → 29578:** resample → guarded dataset build → 12-config sweep → merge → audit (gated ≥900 intersection) → routers. If 13491658 TIMEOUTs first, resubmit via `scripts/sbatch/submit_segment_pool_ood.sh` (RESUME=1; do NOT hand-export env vars on a fresh login).

---

## 2026-07-15 — `canonical_video_id` truncates segment-pool YouTube ids (data-join bug)
**Tags:** bug, methodology, data-integrity, routing, scale-up
**Refs:** `scripts/caption_utils.py::canonical_video_id`, `scripts/sample_ood_quintile_videos.py`, sampler crash in `experiment_outputs/2026-07-15.md`

Firing the full-pool 1000v sample crashed: sampler wrote **999** (not 1000) ids and `create_pilot_dataset` raised `Missing source videos for 3 ids` (e.g. `E1_0`, `ETcLgl5_8`). Root cause: `_CANONICAL_PREFIX_RE = ^([A-Za-z][A-Za-z0-9]*_\d+)` was designed to strip synthetic method suffixes (`panda_0010_delta_a` → `panda_0010`), but Panda-70M segment files are `<youtubeID>_<segment>`, and when the **YouTube ID itself contains `_<digit>`** (e.g. `ETcLgl5_8xY_3`) the regex truncates mid-ID → `ETcLgl5_8`. Effects: (a) **collisions** — sibling segments of the same video collapse to one id (1000→999); (b) **unresolvable** — file is `ETcLgl5_8xY_3.mp4`, so `{canonical}.mp4` lookup fails.

**Latent risk (important):** every downstream table joins OOD score ↔ features ↔ PSNR/VBench on this canonical id. For the ~0.3% of segment-pool ids whose YouTube portion contains `_<digit>`, distinct segments share a key → **cross-contaminated rows**. Excluding them is therefore the *safe* choice, not just a convenience.

**Fix (this commit):** `sample_ood_quintile_videos.py` now (1) builds the set of on-disk `.mp4` stems, (2) drops rows whose canonical id is not an exact on-disk stem (removes the mangled/colliding ids), (3) dedups by canonical id, then samples — guaranteeing an exactly-reproducible, materializable N. `create_pilot_dataset` softened to warn+skip (no hard crash); the dataset stability guard remains the final count gate. Did NOT touch the global `canonical_video_id` (load-bearing across the repo).

**TODO (deferred, not paper-blocking):** properly fix `canonical_video_id` to strip only known method suffixes (`_delta_a`, `_lora`, `_notta`, …) instead of the greedy `<word>_<digit>` prefix, then audit whether any *already-produced* segment-pool feature/OOD joins silently merged colliding ids. Until then, the sampler-level exclusion keeps the 1000v set clean.

---

## 2026-07-19 — 12-config budget grid is population-flat at 1000v (router-motivating)
**Tags:** finding, paper-narrative, routing, scale-up, budget-grid
**Refs:** `sweep_experiment/results/panda_ood_budget_1000v_preview/*/merged_summary.json`, `paper_tables/2026-07-19_budget_grid_1000v_preview.md`, `experiment_outputs/2026-07-19.md`

The full 12-config AdaSteer step×LR grid (S{2,5,10,20} × LR{1e-3,5e-3,1e-2})
finished and merged on the N=1000 OOD-stratified preview pool. **Population
metrics are flat:** PSNR spans only 0.11 dB (19.372–19.486), SSIM 0.0038, LPIPS
0.0039, FVD 3.6 (65.2–68.8), FID 0.2. train time is the only thing that scales
(15→34→65→128 s with steps, 8.4×), buying no quality. The single visible trend is
that the most aggressive config S20_LR1e2 is *worst* on PSNR/SSIM/LPIPS — mild
over-adaptation. This reproduces the in-domain short-horizon saturation first seen
in `panda_1000v_standard`, now at 1000v on the OOD-preview pool.

**Why it matters:** a flat fixed-config mean is precisely the regime where a
per-video router must carry the result (cf. N=200 pilot: oracle PSNR routing
+0.95 dB vs no-TTA, +0.75 vs best fixed config; no config wins across OOD
quintiles). It also justifies the **13th "skip-TTA" router candidate**: if the
budget grid doesn't beat the mean, many clips are better left untouched. The
paper claim is NOT "config X wins" but "per-video routing over {12 configs +
skip} recovers oracle headroom that any fixed choice leaves on the table."

**Next:** merge NOTTA (jobs 14319937–946, same pool) → confirm AdaSteer≈NoTTA at
population level apples-to-apples → per-video oracle + learned-router analysis
across the 5 OOD quintiles (`analyze_adasteer_budget_oracle.py`).

---

## 2026-07-19 — SAVi-DNO LongCat sampler is broken (baseline unusable as-is)
**Tags:** bug, baseline, savi-dno, comparison-methods, blocker
**Refs:** `comparison_methods/scripts/savi_dno_longcat.py` (`_flow_euler_sample_differentiable`, `_dit_forward_step`, `generate_with_optimized_eps`), `experiment_outputs/2026-07-19.md` (A/B diagnostic, jobs 14259120/14259121)

Ran the SAVi-DNO 10-video sanity pair at production knobs (10 Euler / 10 rollout):
A (optimized) vs B (--no-optimize). Result: **A ≈ B** (PSNR 7.212 vs 7.202) and
**both catastrophic** (SSIM 0.04, LPIPS 0.96, FVD ~5400) against the AdaSteer grid's
PSNR ~19.4 / FVD ~66 on the same pool type. VBench subject/background consistency
≈ 0.95 with aesthetic 0.375 → the sampler produces internally-coherent but
GT-unrelated video = conditioning is not being applied.

**Two conclusions:** (1) the sequence-adaptive noise optimization is INERT in this
port (72 min of Adam → +0.01 dB); (2) the custom differentiable sampler
reimplements LongCat's conditioned flow-matching and gets it wrong. The standard
pipeline (NOTTA/AdaSteer) yields PSNR ~19 on the identical model, so the model is
fine — the bug is SAVi's sampler (candidate causes: per-token timestep /
num_cond_latents handling, sigma-direction / velocity sign, latent normalization,
and CFG-off during the differentiable rollout).

**Decision:** do NOT launch full SAVi-DNO (~110 GPU-pair-hours) until the sampler
is fixed and validated (predict_no_optimize vs generate_video_continuation on
identical cond frames must match). Open question for the paper: fix SAVi-DNO, or
drop it and rely on SlowFast-VGen/Temp-LoRA (short horizon) + TTC (long horizon).
The "we chose PSNR because SAVi-DNO reports it" lineage does NOT require SAVi-DNO
to ship if it can't be made correct.

---

## 2026-07-20 — SAVi-DNO root cause: sampler discretization, not conditioning
**Tags:** bug, baseline, savi-dno, comparison-methods, resolved-diagnosis
**Refs:** `comparison_methods/scripts/debug_savi_sampler.py` (job 14322111), `savi_dno_longcat.py:_flow_euler_sample_differentiable`, `experiment_outputs/2026-07-19.md` (2026-07-20 11:40 entry)

The bounded debug (REF standard pipeline vs custom sampler CFG off/on + a
conditioning-sensitivity probe) rules out the two cheap hypotheses and localizes the
bug. probe=0.44–0.68 (velocity changes when context latents are zeroed) => conditioning
IS applied. CUST0≈CUST1 (+0.2 dB) => CFG-off is NOT the cause. REF 12–15 dB vs CUST
8–9 dB on identical cond frames/prompt/geometry => the custom differentiable Euler
sampler is the problem, specifically its **discretization**: a 10-step shift-heavy
schedule with a huge penultimate step (σ 0.624→0.126→0) while the standard LongCat
pipeline uses ~19 steps. The Euler update itself is correct, so this is a step-count/
schedule mismatch (and possibly x0-anchored vs velocity-Euler stepping in the real
pipeline), NOT a formula error.

Combined with the earlier finding that the noise optimization is inert (A/B PSNR
identical), SAVi-DNO-on-LongCat is a hand-port that does not match the reference sampler.
Its native backbone is PVDM, and we have closer, working analogs (SlowFast-VGen/Temp-LoRA
for short horizon; TTC for long horizon). **Recommendation: drop SAVi-DNO as a LongCat
baseline** unless a 15-min matched-step (steps≈20) re-test closes the REF−CUST gap, in
which case the fix is just the default step count. Not a paper-blocking baseline either way.

---

## 2026-07-20 — 1000v-preview router: two data bugs fixed (NOTTA chunking + OOD CSV coverage)
**Tags:** bug, record-keeping, router, ood, notta, provenance
**Refs:** `lora_experiment/scripts/run_full_tta.py`, `sweep_experiment/sbatch/run_sweep.sbatch` (7a35aa4), `submit_notta_1000v_preview.sh` (6be10de), `experiment_outputs/2026-07-20.md`

The first `analyze_adasteer_budget_oracle.py` run on `panda_ood_budget_1000v_preview`
produced a table with `+nan` NOTTA deltas and N=35 OOD quintiles. Two independent bugs:

(A) The NOTTA baseline (METHOD=full) never merged: `run_full_tta.py` lacked
`--start-video-idx`/`--chunk-size`, and the `full)` branch of `run_sweep.sbatch`
never passed them, so all 10 chunks re-ran the full 1000 videos and hit the 8h wall
at ~216 (no `summary.json`). The delta_a grid arms were unaffected (they slice
`eval_videos[start:end]`). Fixed by adding the flags + slicing (mirrors delta_a) and
forwarding them in the sbatch; wall bumped 8h->14h. NOTTA resubmitted 10×100.

(B) The OOD-quintile join used `per_video_analysis/2026-07-12/diffusion_ood_scores.csv`,
which overlaps the swept set only 35/1000 (a stale/different 1000-sample). IDs are
identical `<youtube>_segNN` on both sides — a coverage, not format, mismatch. The
segment-pool CSV (`2026-07-10/diffusion_ood_scores_segment_pool.csv`, 29,379 rows)
overlaps 1000/1000 and is the authoritative source (preview was OOD-stratified from
that pool). Analysis will use the segment-pool CSV going forward.

Decision: the broken table was NOT committed to `paper_tables/` (would have enshrined
`+nan`/N=35). Regenerate after NOTTA merges. The valid, bug-independent findings stand:
population fixed-budget TTA is flat (all 12 configs within 0.11 dB), per-video oracle
uplift +0.382 dB [+0.337,+0.429] (median +0.144, tail-driven), the worst-population
config S20_LR1e2 is the most-picked oracle winner (30.6%), and PSNR-oracle routing
inflates FVD (383.9 vs ~66) — the routing objective is not free.

---

## 2026-07-21 — 5 routing tricks @ 1000v (PSNR): deployable routing ≈ no-TTA
tags: [router, psnr, 1000v, adapt-gate, probe, deployable]
refs: run_routing_tricks.py; experiment_outputs/2026-07-21.md (15:30);
per_video_analysis/2026-07-21/routing_tricks_psnr_1000v/

Ran the five deployable tricks (skip_augmented, route_for_metric, gain_target,
adapt_gate, probe_route) on PSNR over the 1000v preview grid, paired against the
now-present **in-pool** NOTTA (N=898 with NOTTA PSNR; grid N=998). Headline: on this
OOD-preview pool **no-TTA marginally beats fixed AdaSteer AND every deployable router**
(Δ-vs-NOTTA ≈ −0.015 dB; NOTTA ≈ fixed+0.03 dB) — all noise-level. This is the clean
in-pool restatement of the "AdaSteer ≈ No-TTA" saturation result (the 200v pilot's
+0.95 dB oracle-vs-NOTTA was cross-pool against panda_1000v_standard, not paired).

Three signals: (1) skip-awareness helps a hair — skip_augmented/gain_target beat
always-adapt (+0.0186 vs +0.0151 vs fixed) and elect to skip TTA on 58% of videos;
still below NOTTA (imperfect gate/pick). (2) probe_route (+0.09 dB vs NOTTA, 31.9% of
oracle) is the ONLY policy clearing no-TTA, but it is a semi-oracle upper bound (uses
actual probe PSNR/SSIM) costing ~4× inference — confirms static features can't route,
observed probes can, but not cheaply. (3) adapt_gate initially collapsed to always-adapt
because its label `config_oracle−NOTTA` is a max over 12 noisy configs (≈always > 0);
corrected to `fixed−NOTTA` (deployable, non-degenerate) and added combined per-OOD-quintile
Δ-vs-NOTTA to all tricks. Decision: PSNR remains ~unroutable-for-net-gain-vs-NOTTA at 1000v;
the surviving positive AdaSteer result is the matched-FVD win (job pending), not PSNR routing.

---

## 2026-07-21 — Methodology: "fixed" baseline = best population config (per metric); VBench is skip-averse
tags: [router, methodology, baseline, vbench, psnr, 1000v]
refs: run_routing_tricks.py; experiment_outputs/2026-07-21.md (15:55)

Decision (per research-partner instruction): every "Δ vs fixed" must compare against the
**best-performing single config on the same candidate pool for the relevant metric** — the
best-PSNR config for the PSNR router, the best-VBench-total config for the VBench router — NOT
a designated default (previously S10_LR5e3). This is the strongest no-per-video-routing
baseline. Implemented in `run_routing_tricks.py`: fixed = argmax_j population-mean of the
metric over the paired pool (≥1 config + NOTTA scored). Prior tricks numbers (15:55 log) used
S10; expect the small PSNR Δ-vs-fixed to shrink toward/below 0 against best-config.

Finding (v2, still vs S10 pending re-run): **VBench is un-routable AND skip-averse.** The
config-argmax router (route_for_metric −0.0069) and even the semi-oracle probe upper-bound
(−0.0033) sit at fixed/NOTTA on VBench-total (negligible on the ~tens raw-total scale), so
there is no deployable VBench routing win. Adding NOTTA as a 13th action is *net-negative*
for VBench (skip_augmented/gain_target −0.1276, adapt_gate −0.1197): VBench-total prefers some
adaptation, and skipping to no-TTA on 40–60% of videos costs quality. So for VBench neither
routing nor no-TTA beats fixed adaptation. For PSNR, routing ≈ fixed ≈ no-TTA; only the
observed-probe upper bound (+0.09 dB, 4× cost) clears no-TTA.

Scope note (audit of what was actually trained, to prevent overclaim): the clean feature-block
ablation (A / B / A+B / C / A+B+C) exists for the **VBench** deploy router
(`run_deploy_strict_router_experiments.py`, 12-config argmax, no NOTTA option); the **PSNR**
deploy router (`run_deploy_psnr_router.py`) used **Block A only** (12-config argmax, no NOTTA).
The 13-output NOTTA-skip action space exists only in `run_routing_tricks.py` (both metrics,
full feature set), not crossed with A/B/C. Missing subsets: A+C, B+C for both metrics; full
block ablation for PSNR. High-dim input was covered at 1000v via `run_budget_routing_experiments`
(~159-d merged + MLP/HGBM).

---

## 2026-07-21 — Full router matrix @ 1000v (7 blocks × {12,13} × {PSNR,VBench}); VBench oracle is fat-tail noise
tags: [router, matrix, ablation, psnr, vbench, oracle, variance, 1000v]
refs: run_router_full_matrix.py; paper_tables/2026-07-21_router_full_matrix_1000v.md;
router_full_matrix_1000v/router_full_matrix_summary.md

Filled the complete matrix (missing A+C, B+C subsets + full PSNR block ladder + 13-action
skip variants) on the 1000v preview, N=898 paired (config VBench/PSNR + NO-TTA scored) —
coverage confirmed fine (NOT the feared 70/config). Fixed = best population-mean config per
metric; oracle = augmented (max over 12 configs + NO-TTA) per partner instruction.

PSNR: all 14 cells negative vs best config AND vs NO-TTA (−0.004…−0.018). Skip option (13)
helps a hair, never clears 0. Best population PSNR config = S2_LR1e2 (LEAST-adaptive budget)
→ no-TTA ≈ minimal adaptation is PSNR-optimal. Un-routable across every feature block.

VBench: 12-action routers ~flat (≤ −0.007, cap ≈ −0.5%; config-oracle headroom only +0.098
≈ +1%). 13-action routers UNIFORMLY collapse to ≈ −0.13 across all 7 blocks — adding NO-TTA
as an action is structurally harmful, feature-independent.

Key mechanism (answers "why is a skip-capable router still < NO-TTA / is it hyperparameters?"):
the augmented oracle = 10.6005 is +1.03 over NO-TTA while config-oracle is only +0.098 over
fixed. One extra option (NO-TTA, mean 9.57) raising the per-video max by ~0.93 ⇒ NO-TTA's
per-video VBench has MUCH fatter tails than the tightly-clustered adapted configs. So (a)
12-action routers sit on the stable config cluster (~9.57, flat); (b) 13-action routers pick
NO-TTA ~59% on NOISY predictions and eat its downside tail (−0.13), while the oracle banks the
upside tail (+1.03) because it sees truth. The apparent VBench "oracle headroom" is therefore
max-of-a-fat-tailed-noisy-variable, NOT routable signal — a signal/variance ceiling, not a
tuning problem (λ CV-selected; ridge/MLP/HGBM/high-dim/pairwise + observed-probe all fail).

Testable follow-up (before citing +1.03 headroom): confirm NO-TTA VBench fat tail is genuine
(⇒ real "TTA reduces VBench variance / stabilizes quality" angle) vs a coverage/alignment
artifact. Probe = per-config VBench N + NO-TTA-vs-config per-video std/percentiles (queued).

---

## 2026-07-21 — RESOLVED: per-video oracle headroom is NOISE, not routable signal (both metrics)
tags: [router, routability, noise, oracle, vbench, psnr, negative-result, 1000v]
refs: diagnose_routability.py; routability_diag_1000v/routability_diag_summary.md;
paper_tables/2026-07-21_router_full_matrix_1000v.md (RESOLVED section)

Coverage probe: per-config VBench = 998 (complete); NOTTA VBench = 898 (100 missing = 1 chunk).
NOTTA vs CONFIG per-video marginals IDENTICAL (mean 9.570/9.570, std 1.860/1.848, matched
percentiles/min/max) → the fat aug-oracle is neither variance-reduction nor coverage artifact.

Routability diagnostic (N=898):
  PSNR : within_cfg_σ=0.2515 corr_cc=0.992 corr(notta,cfg)=0.998 oracle_gain/fixed=0.3575
         R²(gain|features)=−0.092  R²(gain|+probe)=−0.092
  VBench: within_cfg_σ=0.0579 corr_cc=0.998 corr(notta,cfg)=0.051 oracle_gain/fixed=0.0978
         R²(gain|features)=−0.082  R²(gain|+probe)=−0.082

Decisive reading: the per-video oracle gains are MAX-OVER-NOISE, not signal. (1) 12 configs
are ~identical per video (corr ≥0.99) so their per-config differences are noise; observed PSNR
oracle gain 0.36 dB ≈ pure-noise floor σ·E[max12]=0.41. (2) OOF R² predicting the per-video
oracle GAIN is NEGATIVE from the full 159-d stack AND with probe outcomes — no learnable
structure (explains all 28 matrix cells + 13 variants + 5 tricks failing). (3) VBench smoking
gun: corr(NOTTA,config)=0.051 (vs PSNR 0.998) — same clip, no-TTA VBench independent of adapted
VBench ⇒ per-video VBench-total (MUSIQ, no-reference) is scoring noise; the +1.03 aug-oracle is
max of two independent noise draws.

DECISION: stop per-video routing signal-hunting; it is a noise ceiling, not a features/models/
hyperparameters gap. Present as a clean negative result supporting "AdaSteer ≈ No-TTA → deploy a
single fixed config." N=898→~1000 backfill (100 NOTTA VBench, 1 chunk) is cosmetic only.

---

## 2026-08-04 — Binary TTA/no-TTA gate + initial-loss probe: RULED OUT on PSNR (1000v)
tags: [router, routability, noise, oracle, psnr, initial-loss, binary-gate, negative-result, 1000v]
refs: scripts/analyze_initial_loss_prediction.py;
sweep_experiment/reports/per_video_analysis/initial_loss_prediction_1000v.json;
paper_tables/2026-08-04_binary_gate_initial_loss_1000v.md

Direct test of two proposals: (Q1) can the CHEAP initial-TTA loss predict per-video PSNR gain;
(Q2) the simplified "route TTA vs no-TTA, then apply the best fixed config" gate. Fully offline
from existing budget-grid + NOTTA summary.json (N=900 common; 898 with finite PSNR gain).
Probe features from the shortest config (S2, whose final_loss = loss after 2 TTA steps):
final_loss(=base_loss here; base-loss≡total-loss so loss_reduction≡0), delta_norm, grad_norm.

Results (metric=PSNR):
  best fixed config = S2_LR1e2, mean gain -0.003 dB  (ALL 12 configs <=0 mean gain).
  always-fixed vs no-TTA (pop effect): -0.0028 dB [-0.0252,+0.0187]  -> null.
  PERFECT-gate vs always-fixed        : +0.0694 dB [+0.0542,+0.0872]
  noise floor E|g|/2                  : ~+0.069 dB  -> ceiling == noise floor.
  [ref] 12-config oracle              : +0.3547 dB (more noisy draws to max over).
  probe->binary-help predictability   : AUC ~0.50 for every feature; OOF ridge-probe AUC 0.508.
  OOF gate vs no-TTA / vs fixed        : +0.003 [-0.015,+0.021] / +0.006 [-0.005,+0.019] -> both null.
  Q1 corr(feature,gain)               : only final_loss CI-significant (Spearman -0.083
                                         [-0.148,-0.019]) but <1% variance; OOF ridge corr +0.059
                                         [-0.000,+0.117] (touches 0) -> no deployable regression.

Decisive reading: E[relu(-g)] = (E|g| - E[g])/2; with E[g]~0 the binary-gate ceiling collapses to
E|g|/2 = pure measurement noise. So even a PERFECT TTA/no-TTA oracle only "gains" the noise floor,
and the cheap probe predicts the gate at chance. The binary-gate simplification correctly removes
the 12->2 max-over-noise inflation but cannot pass the noise ceiling: there is no per-video signal.

DECISION: rule out the binary TTA/no-TTA gate (and initial-loss probes generally) for PSNR on
in-domain Panda. Reinforces the single-fixed-config recommendation. Open: (a) OOD/long-horizon
regimes where a real population effect may exist (E[g] != 0 would make the gate meaningful);
(b) seed-space best-of-k, where headroom comes from genuinely different videos, not noise.

---

## 2026-08-09 — Built: (a) genuinely-long-horizon sweep (~1 min), (b) EXP4 streaming anchored delta
tags: [long-horizon, streaming-delta, exp4, drift, native-geometry, sharding, build]
refs: delta_experiment/scripts/diag_longhorizon_drift.py;
delta_experiment/sbatch/submit_longhorizon_sweep.sh; delta_experiment/sbatch/run_longhorizon_drift.sbatch;
scripts/merge_drift_shards.py

Motivation: the 2026-08-08 native control showed drift is REAL but MILD at 6 native chunks
(=480 gen frames, ~30s: colorfulness +4%, sharpness +28%, PSNR -21%, LPIPS +96%). That sits at the
LOW end of what "long-horizon video continuation" reviewers expect (StreamingT2V ~2min/1200f;
Rolling Forcing multi-minute; LongCat's own design point ~1min). User: push to 25-50% of the field
ceiling, not the lower bound. Also build the streaming per-chunk delta the EXP-B null motivated.

(a) Long-horizon sweep. NUM_CHUNKS=12 @ native 13-cond/80-gen = 960 generated frames ~= 60s @16fps
    (~50% of StreamingT2V's 2min, ~= LongCat's 1-min design). One such video ~110 min @50 steps, so
    submit_longhorizon_sweep.sh SHARDS the pool across jobs: each shard gets its own OUTPUT_DIR +
    checkpoint (no race), all shards share NUM_VIDEOS+SEED so the video-list ordering is identical
    and START_VIDEO_IDX/CHUNK_SIZE slice it. Default POOL_N=8, SHARD_SIZE=2 -> 4 jobs (~4h each).
    scripts/merge_drift_shards.py pools successful records, recomputes per-chunk curves+verdict
    (same schema as the single-run summary -> plot_drift_curves.py works unchanged). GT-free drift
    signals survive GT running out (they always did); FVD/VBench-Long can be scored later off the
    saved stitched mp4s.

(b) EXP4 streaming anchored delta. diag_longhorizon_drift.py --method delta_stream: train delta0 on
    the real observed frames at chunk 0 (exact run_delta_a recipe), then BEFORE each subsequent
    chunk re-fit the delta on the most recent full [cond|gen] window (--stream-refit-steps, default
    5) and re-anchor: applied = (1-lambda)*refit + lambda*delta0 (--stream-blend lambda, default
    0.5). Anchoring to the real-data delta0 is the guard against the known failure mode (a purely
    self-supervised re-fit on the model's own drifting output could reinforce the drift). Hooks are
    removed during each re-fit (wrapper.forward adds delta via args; leaving hooks on would
    double-apply) and re-installed for generation; VAE+text-encoder offloaded during the few-step
    re-fit. Per-chunk delta norms logged to summary (stream_delta_norms).

STATUS: code built, byte-compiled, submitter dry-run verified. Runs pending. Next: launch the NOTTA
gating sweep (does mild native drift compound to a decisive effect at ~1 min?), then delta_stream at
the same geometry with paired seeds to test whether streaming re-anchoring flattens it where the
EXP-B fixed delta went stale.

---

## 2026-08-09 — GATING RESULT: native drift COMPOUNDS with horizon at ~60s (12 chunks, N=8)
tags: [long-horizon, drift, native-geometry, gating, notta, positive-result]
refs: sweep_experiment/results/longhorizon_sweep_notta_native_12ch/merged_summary.json;
sweep_experiment/reports/experiment_outputs/2026-08-09.md; scripts/merge_drift_shards.py

Merged 4 shards (N=8 videos x 12 native chunks = 960 gen frames ~= 60s @16fps, seed=42). GT-free
drift verdict (chunk1 -> chunk12), with the 30s/6-chunk native prelim (2026-08-08) for contrast:
  sharpness        +48%  (was +28% @ 6ch)   temporal_motion +45%  (was +8%)
  contrast         -16%  (was +2.8%)        colorfulness    +5.7% (was +4%)
All slopes consistent-signed + monotonic. => At a GENUINELY long horizon (~1 min, ~50% of the field
ceiling) native LongCat degrades meaningfully and MORE than at 30s: drift compounds. This is the
decisive long-horizon headroom that was ABSENT at short/native-6ch geometry.

CAVEAT: psnr/ssim/lpips "chunk1->last" spans only the first ~1-2 chunks (GT overlap runs out on the
short source clips), so their steep slopes (psnr -2.56/chunk over 2 points) are NOT the long-horizon
signal. Judge long-horizon drift by the GT-free curves only. N=8 is a gating sample; widen N once a
method shows signal.

DECISION: launch EXP4 delta_stream at the SAME geometry/seed (paired vs these 8 videos, same
per-chunk seeds). --stream-blend 0.5 anchor to the real-data delta0 guards against the per-chunk
re-fit chasing the rising sharpness/motion artifacts; escalate anchor to 0.6-0.7 if the delta
amplifies drift. This is the target the EXP-B fixed-delta null pointed to.

---

## 2026-08-09 — EXP4 streaming anchored delta: FIRST POSITIVE intervention (native ~60s, N=8)
tags: [long-horizon, streaming-delta, exp4, drift, native-geometry, positive-result, paired]
refs: sweep_experiment/results/longhorizon_sweep_delta_stream_native_12ch/merged_summary.json;
scripts/compare_drift_paired.py; sweep_experiment/reports/experiment_outputs/2026-08-09.md

delta_stream (refit_steps=5, blend/anchor=0.5) re-fits the AdaSteer delta each chunk on the most
recent generated window and re-anchors toward the real-data chunk-0 delta. Run at the SAME native
60s geometry + seed as the NOTTA gating run => paired per-video (chunk-1 baselines match to ~0.001).

Drift verdict (chunk1 -> chunk12), NOTTA vs stream-delta:
  sharpness (leading)  +48.0% (->0.0096)  ->  +24.8% (->0.0080)   ~HALVED
  colorfulness         +5.7%              ->  +0.4%               FLATTENED
  contrast (fade)      -16.4%             ->  -11.5%              ~30% less fade
  temporal_motion      +45.1% (->0.0341)  ->  +40.8% (->0.0341)   ~unchanged
  psnr/ssim/lpips      -14.7/-14.3/-4.6%  ->  -14.5/-14.2/-4.8%   tied (GT spans ~1-2 chunks; not LH signal)

READING: the anchored streaming delta shrinks the leading long-horizon drift mode (HF-artifact
accumulation) by ~half and flattens over-saturation, reduces contrast fade, WITHOUT amplifying
artifacts (lambda=0.5 anchor guard worked -> no drift-chasing). Motion instability is the one mode
it does not fix. This is the FIRST positive intervention result in the project; it directly answers
the EXP-B fixed-delta null (a moving target needs a moving correction).

CAVEATS: N=8 is gating; endpoint means are not a significance test. GT pixel metrics uninformative
at long horizon (GT overlap runs out). NEXT: run scripts/compare_drift_paired.py (per-video bootstrap
CI + sign-flip permutation on |drift| reduction). If CI excludes 0 on sharpness/colorfulness ->
promote to headline, widen N, sweep lambda (0.3/0.5/0.7) + refit_steps, and add FVD/VBench-Long on
the saved stitched mp4s. If null under the test -> report as promising-but-underpowered.

---

## 2026-08-09 (REBUTS the entry above) — EXP4 paired per-video test: NULL, not positive
tags: [long-horizon, streaming-delta, exp4, paired-test, negative-result, correction, self-supervised-flaw]
refs: scripts/compare_drift_paired.py;
sweep_experiment/results/longhorizon_sweep_delta_stream_native_12ch/paired/paired_stats.json;
sweep_experiment/reports/experiment_outputs/2026-08-09.md

The "FIRST POSITIVE" entry above judged EXP4 on POPULATION mean-curve endpoints. The correct
per-video paired test (bootstrap CI + sign-flip permutation on |drift|=|chunk12-chunk1|, N=8) says
it is NULL:
  signal            reduction(NOTTA-delta)   95% CI                p
  sharpness         -0.0015                  [-0.0038,+0.0007]     0.26
  temporal_motion   +0.0008                  [-0.0061,+0.0074]     0.88
  colorfulness      -0.0078                  [-0.0199,+0.0051]     0.32
  contrast          -0.0029                  [-0.0148,+0.0081]     0.66
No CI excludes 0; point estimates lean the WRONG way (delta drifts MORE per video) on 3/4 GT-free
signals. The population "flattening" was CANCELLATION: delta's mean-curve sharpness change (0.0016)
vs per-video mean |drift| (0.0074) = 4.6x gap (NOTTA 1.9x) -> delta raised per-video volatility that
averages flat. A flat mean curve here == added instability, not stability.

ROOT CAUSE: delta_stream re-fits each chunk by flow-matching to the model's OWN generated window, so
when generation drifts the refit target is the drifted frames -> the update partly REPRODUCES drift.
Only the lambda=0.5 delta0 anchor (trained on real chunk-0 frames) is a clean signal.

DECISION: EXP4 as built (lambda=0.5, refit_steps=5) is a clean negative under paired testing. Do NOT
sweep lambda upward (lambda->1 == the EXP-B fixed-delta null). Two live paths: (1) redesign the
per-chunk update to anchor to CLEAN chunk-0 context statistics / appearance (Pathwise-TTC-style
re-anchoring) instead of self-supervising on drifted output -- one real technical shot; (2) if that
also fails paired testing, consolidate the honest narrative: corrected native drift measurement +
"drift compounds with horizon" + a controlled catalogue of interventions (fixed delta, streaming
delta) that do not beat NOTTA per-video. Consistent with the project-wide pattern (PSNR router,
placement, TANGO): population movements that vanish under per-video paired tests.

---

## 2026-08-09 — Built: clean-anchored streaming re-fit (`--stream-target clean`) + length-extend knob
tags: [long-horizon, streaming-delta, exp4, clean-anchor, build, pathwise-ttc]
refs: delta_experiment/scripts/diag_longhorizon_drift.py;
delta_experiment/sbatch/submit_longhorizon_sweep.sh; delta_experiment/sbatch/run_longhorizon_drift.sbatch

Direct fix for the EXP4 root cause (self-supervising on the model's own drifted output). New
`delta_stream --stream-target clean`: at each chunk, CONDITION on the current drifted context (the
tail that will condition the next chunk) but FLOW-MATCH the delta toward the CLEAN chunk-0 real-frame
latents (cached from delta0 training, reused with no re-encode). The low-capacity bias thus learns
"from where you've drifted, steer back toward the clean distribution" -- a Pathwise-TTC-style
re-anchoring expressed through the AdaSteer delta rather than sampling-space guidance. Geometry
matches chunk-0 (cond=4 drifted latents + train=8 clean latents = 12). `--stream-blend` still
re-anchors the result toward delta0. Old behaviour preserved as `--stream-target generated` (the
null). Series name encodes the target (…_delta_stream_clean_native_12ch) so runs never collide.

Also exposed the length-extend fallback the user requested if clean-anchor fails: NUM_CHUNKS knob
(18=~72s, 24=~96s) with SHARD_SIZE=1 to stay in the 12h wall (~9.3 min/native chunk).

PLAN: run clean-anchored delta_stream at the SAME native 60s geometry/seed as the NOTTA + generated
runs (paired), then compare_drift_paired.py. Decision gate: CI excludes 0 on sharpness/colorfulness
=> real re-anchoring effect (widen N, sweep lambda/refit, add FVD/VBench-Long); null => extend
horizon (NUM_CHUNKS 18/24) to see if a bigger drift gap makes the correction detectable, else
consolidate the measurement + negative-results narrative.

---

## 2026-08-10 — Clean-anchored streaming delta: ALSO NULL (3rd delta variant to fail paired test)
tags: [long-horizon, streaming-delta, exp4, clean-anchor, negative-result, paired-test, mechanism-limit]
refs: sweep_experiment/results/longhorizon_sweep_delta_stream_clean_native_12ch/paired/paired_stats.json;
scripts/compare_drift_paired.py; sweep_experiment/reports/experiment_outputs/2026-08-10.md

delta_stream --stream-target clean (native 60s, N=8, paired vs NOTTA). Paired |drift| reduction:
  sharpness        -0.0014  [-0.0051,+0.0028]  p=0.53   (still favors NOTTA)
  temporal_motion  +0.0010  [-0.0077,+0.0087]  p=0.83
  colorfulness     +0.0015  [-0.0107,+0.0158]  p=0.84
  contrast         -0.0177  [-0.0659,+0.0164]  p=0.70   (WORSE: more fade)
  ssim (n=3)       -0.0006  [-0.0011,-0.0002]  p=0.25   * <- FALSE ALARM (n=3 degenerate CI; neg; p ns)
No GT-free CI excludes 0. Patched compare_drift_paired.py to suppress the "*" for n<5 so the ssim
artifact can't mislead.

WHAT CLEAN-ANCHORING DID: pushed saturation the intended direction (colorfulness pop +5.7% -> -8.5%,
paired point estimate flipped from -0.0078 in v1 to +0.0015) and flattened motion at POPULATION
(+45% -> +5.7%), but per-video |drift| barely moves (cancellation), and it OVERSHOT into more
contrast fade (paired -0.0177; pop contrast -20.9% vs NOTTA -16.4%). Net per-video: null.

CONCLUSION: three delta recipes now fail the per-video paired test at native 60s -- fixed
(2026-08-08 EXP-B), streaming-generated (2026-08-09), streaming-clean (this entry). A single global
AdaSteer bias vector can shift population-level color/motion statistics but cannot CONSISTENTLY
reduce per-video drift; it trades one axis (saturation) for another (contrast fade). This is a
mechanism/capacity limit, not an anchoring-recipe problem -- consistent with the project-wide
pattern (PSNR router, placement, TANGO, all deltas): population movements that vanish per-video.

DECISION: per the pre-committed fallback, extend the horizon (NUM_CHUNKS=18 ~90s / 24 ~120s field
ceiling) for BOTH NOTTA and clean-anchor (SHARD_SIZE=1) -- a bigger drift gap gives a real correction
more room and is easier to detect above N=8 noise, and strengthens the measurement story regardless.
If null again, commit to the measurement + honest-negative-results narrative (corrected native drift
measurement + drift compounds with horizon + a controlled catalogue of interventions that do not beat
NOTTA per-video). Do NOT keep permuting delta recipes.

---

## 2026-08-10 — Per-video heterogeneity is a NOISE ceiling: routing thread CLOSED
tags: [long-horizon, routing, heterogeneity, oracle-noise, negative-result, pivot, measurement]
refs: scripts/analyze_drift_heterogeneity.py; scripts/analyze_drift_per_video.py;
sweep_experiment/results/longhorizon_sweep_delta_stream_clean_native_12ch/per_video/heterogeneity.json

The per-video breakdown (2026-08-10 18:51) showed the intervention is heterogeneous (no-TTA best on
4/8 videos; delta net-harmful on sharpness) with a 23-39% per-video ORACLE gap -- tempting a router.
The heterogeneity gate kills it:
  cross-signal consistency: observed 0.312 vs shuffled-null 0.343 [0.229,0.500], p=0.71 (observed
    is BELOW the null mean) -> the best arm for a video does NOT agree across that video's own
    signals beyond chance. Not a video property; not routable.
  oracle vs best_fixed vs random: on every GT-free signal the perfect-router gain over the best
    fixed arm (sharpness +0.00202, motion +0.00536, colorful +0.00639, contrast +0.01357) is <= the
    noise-only min gain (0.00300 / 0.00577 / 0.00999 / 0.02043). The oracle gap is fully explained
    by min-over-3-noisy-arms selection -- identical to the 2026-08-04 PSNR-router noise-floor finding.

CONCLUSION: the AdaSteer-delta intervention line is a clean, well-controlled NEGATIVE at native
long-horizon: (a) fixed / streaming-generated / streaming-clean deltas all null under the paired
per-video test; (b) the per-video heterogeneity that could have justified a router is a noise ceiling
(consistency p=0.71), not realizable signal. Matches the project-wide pattern across PSNR router,
placement, TANGO.

DECISION: STOP permuting delta recipes / chasing a router. Pivot to the measurement + honest
negative-results paper: (1) corrected native drift measurement (naive short-window rollout massively
overstates drift; 2026-08-08 control) + drift compounds with horizon (2026-08-09); (2) a controlled
catalogue of interventions that do not beat NOTTA per-video. Re-purpose the horizon extension as the
MEASUREMENT CAPSTONE: NOTTA to ~90-120s (field-standard) for a reviewer-proof drift curve, + one
delta arm at that horizon to close "did you test long enough?". Not a delta rescue.

---

## 2026-08-10 — Time-scheduled (ramped) delta is CONTRAINDICATED; NOTTA-only capstone
tags: [long-horizon, delta, schedule, ramp, chunk-interaction, negative-result, pivot]
refs: scripts/analyze_delta_chunk_interaction.py;
sweep_experiment/results/longhorizon_sweep_delta_stream_clean_native_12ch/chunk_interaction/

Before spending a GPU-day on a ramped-gain delta (gamma_t small early -> large late), gated its
PREMISE on existing 12-chunk paired data: does the constant-delta's per-video paired effect cross
over (hurt early on near-clean content, help late on degraded content)? Result across 8 signal x arm
cells (N=8):
  - CROSSOVER in exactly 1 cell (gen/temporal_motion), but all per-chunk CIs straddle 0 and rel_eff
    is unstable (-94..-167) => noise.
  - Every SIGNIFICANT per-chunk cell (CI excludes 0) is NEGATIVE: the delta significantly HURTS
    (gen/sharpness ch4,6,7,9; clean/temporal ch7,8; clean/contrast ch5,6). No significant positives.
  - ANTI-crossover (effect worsens late) in 4/8 cells (sharpness both arms, clean/contrast,
    gen/colorfulness): a ramp raising gamma_t late would AMPLIFY harm where the model is worst.
So a schedule has no signal to exploit and is pointed against by the data. This is the 4th distinct
delta axis to fail (constant-fixed, streaming-generated, streaming-clean, time-scheduled) + routing
is a noise ceiling. The AdaSteer-delta intervention line is definitively CLOSED.

DECISION: run the NOTTA-ONLY measurement capstone (18ch ~90s native); do not build the ramped arm.
Commit fully to the measurement + negative-results narrative.

---

## 2026-08-10 — PIVOT: from steering-delta to TEST-TIME SEARCH (best-of-N drift verifier)
tags: [pivot, test-time-search, best-of-N, verifier, exposure-bias, literature, positive-direction]
refs: delta_experiment/scripts/diag_longhorizon_drift.py (method=bestof);
scripts/analyze_bestof_search.py; Video-T1 (ICCV'25, 2503.18942); MCTS-TTS (ICLR'26 sub);
Verifier Matters (BMVC'25); Pathwise TTC (2602.05871); History-Guided Video Diffusion / DFoT (ICLR'25);
Rolling Forcing (2509.25161)

DIRECTION CHANGE (user): stop framing toward a negative-results paper (no top venue publishes "method
X doesn't work"); use the nulls as a DIAGNOSIS and build a method that works, grounded in current
literature. The diagnosis: autoregressive drift is EXPOSURE BIAS (model conditions on its own degraded
output, a regime unseen in training). An additive bias in ACTIVATION space (AdaSteer delta, all 4
axes) cannot correct an INPUT-distribution shift -- which is exactly why every delta went stale/hurt.
Independently corroborated: Pathwise TTC (Feb 2026) documents that test-time PARAMETER optimization
"collapses" on long video and that the fix is sampling-space / conditioning-level correction -- our
clean-anchored delta re-fit null is the same phenomenon.

Literature scan (all training-free, fit our TTA framing):
  * Test-time search + verifier: Video-T1 (ICCV'25), MCTS-TTS (ICLR'26 sub), Verifier Matters (BMVC'25)
    -- reframe generation as search over noise; pick best candidate by a verifier.
  * Anchored sampling-space correction: Pathwise TTC (2026) -- swap drifted context -> clean anchor at
    low-noise refinement steps, re-noise, resume.
  * History guidance (CFG over context): DFoT / History-Guided Video Diffusion (ICLR'25).
  * Attention-sink anchoring: Rolling Forcing (2025).

DECISION (user picked): build TEST-TIME SEARCH first -- best-of-N per chunk with a GT-FREE DRIFT
VERIFIER (fastest to a positive number; reuses our validated monotonic drift signals + rollout infra).
Our contribution is the verifier: a physically-grounded, deployable (no future frames) drift score =
relative deviation of {sharpness, colorfulness, contrast, temporal_motion} from the initial REAL
conditioning-frame reference + a seam-continuity penalty. Candidate 0 reuses the NOTTA seed so
best-of-N is a STRICT SUPERSET of NOTTA (can only match/beat it), and every candidate is logged so a
post-hoc ORACLE (best candidate) bounds achievable headroom vs what the GT-free verifier captures.

EXTERNAL CONFIRMATION OF OUR DELTA NULL (Pathwise TTC 2602.05871, toy experiment sec 3.2 / Fig 4):
TTC asks the same question (fix long-video drift purely at inference) and reports that TEST-TIME
OPTIMIZATION FAILS. Two LoRA-at-test-time variants: (1) reconstruction reward on early frames ->
suppresses motion (collapses toward copying early content); (2) distribution-anchoring reward toward
the initial frames -> reward collapse into degenerate solutions violating the prior. Root causes they
name: unstable/ill-defined reward (drift is coupled semantics+appearance+motion; low-level reward kills
motion, high-level reward lacks frame-wise signal) + hypersensitivity of parameters to tiny test-time
gradients. Pivot: parameter-space TTO -> sampling-space correction. THIS IS OUR DELTA NULL, peer-
reviewed: our AdaSteer delta = parameter/activation-space optimization toward an anchor, same failure
for the same reasons. Their stated open problem "reward design for error accumulation" is exactly the
gap our GT-free drift verifier + per-chunk gate targets.

NOVELTY CAVEAT (be honest): plain best-of-N is NOT novel (TTC uses BoN N=5 as a baseline; Video-T1
built on it) and a straight TTC reimplementation is NOT novel. The novel contribution must be the
CONTROLLER: a drift-GATED, GT-free test-time controller that decides per-video/per-chunk WHETHER to
intervene (gate) and HOW (search actuator vs anchored-correction actuator), with the GT-free drift
verifier answering TTC's open "reward design" problem. BoN + TTC-correction are actuators inside it;
gating is the mechanism, not just a diagnostic. Pending user confirmation of this framing before
committing compute to the anchored-correction actuator (needs cluster-side pipeline access; common.py
+ LongCat pipeline are dehydrated locally).

GATING STILL APPLIES (statistical): the oracle-over-candidates in BoN is itself max-over-noise inflated
(best of k noisy draws), same trap as the 2026-08-04 PSNR-router. analyze_bestof_search.py now reports
verifier-pick vs RANDOM-pick vs oracle: the verifier has real signal only if chosen beats random; the
oracle-vs-random gap is the noise floor. Headline drift reduction vs NOTTA remains gated by the paired
sign-flip test (compare_drift_paired.py).

BUILD (this turn): added method=bestof to diag_longhorizon_drift.py (+ --search-k, --search-seam-weight),
threaded SEARCH_K/SEARCH_SEAM_WEIGHT through run_longhorizon_drift.sbatch + submit_longhorizon_sweep.sh
(method-aware SHARD_SIZE default 1 for bestof since per-chunk cost x k; series
longhorizon_sweep_bestof_k{K}_native_{C}ch). Added scripts/analyze_bestof_search.py (search activity,
verifier composite chosen-vs-cand0, TRUE-quality check on GT chunks: does the GT-free pick lift
PSNR/LPIPS and how much of the by-metric oracle it captures, per-signal oracle ceiling). Headline
end-of-rollout drift reduction vs NOTTA = compare_drift_paired.py vs the native 12ch NOTTA run.
NEXT: run bestof k=4 native 12ch N=8 (paired to longhorizon_sweep_notta_native_12ch), analyze.

---

## 2026-08-10 — Second actuator built: drift-GATED Pathwise-TTC (the controller, sampling-space)
tags: #ttc #controller #gating #sampling-space #actuator #build
refs: comparison_methods/scripts/{savi_dno_longcat.py,ttc_longcat.py},
delta_experiment/scripts/diag_longhorizon_drift.py, delta_experiment/sbatch/{run_longhorizon_drift.sbatch,submit_longhorizon_sweep.sh}

Recon (subagent) confirmed the shipped LongCatVideoPipeline exposes no per-step denoise handles, but the
repo already contains a self-contained engine (SAViDNO_LongCat) and a working single-window ungated
Pathwise-TTC sampler (TTC_LongCat). So the anchored-correction actuator did NOT need a from-scratch loop —
it needed integration into the multi-chunk rollout harness + the drift GATE.

Built into diag_longhorizon_drift.py: --method ttc (ungated appearance re-anchor to the clean first frame
during low-noise steps, sigma<=0.3) and --method ttc_gated (THE CONTROLLER: correct a chunk ONLY if its
INCOMING context's GT-free deviation from the real-frame reference exceeds --ttc-gate-threshold, else pass
through uncorrected). This is the same GT-free drift signal used by the bestof verifier, now used as a
per-chunk trigger — unifying both actuators (search + anchored-correction) under one gated controller, the
framing confirmed with the user.

Engineering correctness note (frame geometry): TTC_LongCat.sample decodes only the generated latents,
which for an 80-frame chunk yields 77 pixels (the shared VAE boundary frame is lost). In a chained rollout
that corrupts the conditioning tail size. FIX: added return_latents= to sample() and decode the FULL
[cond|gen] latent stack JOINTLY in the harness (4 cond + 20 gen = 24 latents -> 93 frames = 13 cond + 80
gen), exactly matching the pipeline geometry. Verified: prev_gen[num_gen:] = last 13 frames = num_cond.

Fair-comparison caveat (LOGGED so we don't fool ourselves): the TTC path re-encodes the pixel conditioning
tail each chunk (reencode-style), whereas the native NOTTA rollout uses KV-cache latent chaining. So the
honest paired baseline for TTC is `ttc --ttc-weight 0` on the SAME engine, NOT longhorizon_sweep_notta_
native_12ch. Both ttc arms and the ttc-w0 baseline share the reencode conditioning, so the paired sign-flip
test (compare_drift_paired.py) isolates the correction effect. Series: longhorizon_sweep_ttc_w<W>_* and
_ttcgated_w<W>_g<G>_*. Threaded TTC_* env through sbatch + submitter. All four files syntax-checked.
NEXT: after the bestof gate results land, sweep ttc-weight {0, 0.05, 0.1, 0.2} + ttc_gated, paired to w0.

---

## 2026-08-11 — best-of-N (k=4) FULL N=8: FIRST arm to PASS the credibility gate (verifier tracks true quality)
tags: [long-horizon, best-of-N, test-time-search, verifier, positive-result, credibility-gate, native-geometry, underpowered]
refs: sweep_experiment/results/longhorizon_sweep_bestof_k4_native_12ch/{merged_partial.json,search_analysis_partial/search_analysis.json,paired_partial/paired_stats.json};
scripts/analyze_bestof_search.py; scripts/compare_drift_paired.py;
sweep_experiment/reports/experiment_outputs/2026-08-11.md

best-of-4 GT-free drift verifier, native 13/80, 12 autoregressive chunks, seed=42, N=8 (all 8 shards
done), candidate 0 = NOTTA seed (strict superset). Two independent reads:

1. THE SELECTION MECHANISM WORKS (passes the gate routing FAILED). Search active: verifier picks a
   non-NOTTA candidate in 72/96 chunks (75%). On its own composite, chosen (14.69) beats random-pick
   (16.78) by +2.09 (vs cand0 17.04). CREDIBILITY TEST on the 11 GT-overlapping chunks (metric the
   verifier does NOT optimize): chosen PSNR 17.24 vs random 16.41 vs cand0 16.39 vs oracle-by-PSNR
   17.44 -> chosen-random = +0.833 dB, capturing 81% of the oracle-over-random gain (+1.028). LPIPS
   chosen-random = -0.0318 (oracle 0.278, chosen 0.283). Since random ~= cand0, this is REAL selection
   signal, not max-over-noise -- exactly the opposite of the PSNR-router (2026-08-04) and per-video
   routing (2026-08-10) threads, where chosen ~= random (noise ceiling). Per-signal oracle capture:
   sharpness 96% (+1.597/+1.664), temporal_motion 76% (+0.441/+0.581), contrast 29% (+0.014/+0.049),
   colorfulness 10% (+0.018/+0.177). The verifier is strong on the DOMINANT native-60s drift modes
   (sharpness, motion) and weak on the entangled one (color) -- consistent with every prior color result.

2. THE END-TO-END EFFECT IS DIRECTIONALLY-RIGHT BUT UNDERPOWERED. Paired |drift| reduction vs NOTTA
   (N=8, sign-flip): sharpness +0.0009 (p=0.62), temporal_motion +0.0046 (p=0.53) -- both lean the right
   way on the verifier's strong modes but no CI excludes 0; colorfulness -0.0013 (p=0.91); contrast
   -0.0125 (p=0.56, leans WRONG -- BoN does not fix the fade). GT metrics all lean right (psnr +0.10 dB,
   ssim +0.0127, lpips +0.0178; n=3, tiny). So per-chunk selection quality is demonstrated, but its
   conversion to a significant endpoint |drift|=|chunk12-chunk1| reduction is a POWER problem at N=8
   (2-point endpoint metric, high per-video variance, diluted by the weak color/contrast modes).

CONTRAST WITH THE CLOSED LINES: deltas failed BOTH the credibility check (no independent-metric gain)
AND the paired test; routing was a pure noise ceiling. best-of-N is the FIRST arm where the mechanism
provably lifts a held-out metric above the random-pick floor. That earns more compute.

DECISION: (1) SCALE N (the paired endpoint test is underpowered; the per-chunk gate already passes).
(2) Consider reweighting the verifier toward sharpness/motion (where it captures ~76-96% of oracle) or
adding an anchor-similarity term for color/contrast. (3) Efficiency + next actuator: latent-space
verifier (decode only the chosen candidate -> save (k-1)/k decodes) and the drift-gated TTC actuator.
This is NOT a paper number yet -- N=8 gating, GT chunks n=11 -- but it is the first credible positive
and reframes the controller narrative from "diagnosis of nulls" to "a working GT-free selection gate."

---

## 2026-08-14 — TTC w=0 first GPU run is GARBAGE: Euler sign bug in TTC_LongCat (same one SAViDNO already fixed)
tags: [ttc, bug, sampling-space, savi-dno, euler-sign, smoke-test]
refs: comparison_methods/scripts/ttc_longcat.py; comparison_methods/scripts/savi_dno_longcat.py
  (_flow_euler_sample_differentiable); sweep_experiment/reports/experiment_outputs/2026-08-14.md;
  jobs 15699080-083 / longhorizon_sweep_ttc_w0_native_12ch

TTC actuator's first GPU execution (w=0, shard_0000, 2 videos x 12 chunks) completed without a
traceback — joint [cond|gen] decode and chained rollout geometry are fine. The pixels are not.
On the same two videos that native NOTTA / best-of-N scored at PSNR 16-21 dB, TTC w=0 produced
PSNR 7.38 / 8.38, LPIPS ~0.94, and *identical* GT-free signals across a car clip and a watch clip
(sharpness stuck at 0.0021, motion ~0.130, colorfulness ~0.23, flat over 12 chunks). That is
decoded initial noise, not a continuation.

ROOT CAUSE: `TTC_LongCat.sample` still used the pre-fix SAViDNO Euler convention
`x_t = x_t + dt * v` and `x0 = x_t - sigma * v`. SAViDNO later documented that LongCat's
`generate_vc` negates the DiT output (`noise_pred = -noise_pred`) so the matching step is
`x_t = x_t - dt * v`, with clean estimate `x0 = x_t + sigma * v`. The old sign "stepped away
from clean and never denoised (output ~ decoded initial noise -> PSNR ~9 / SSIM ~0.05,
identical regardless of CFG/steps)" — verbatim the w=0 smoke-test signature.

FIX: flip Euler + x0 + v_corr in `ttc_longcat.py` to match SAViDNO/`generate_vc`. Do NOT
launch w=0.1 / ttc_gated on the broken sampler. Cancel remaining 1569908x shards; their
output is the same garbage and is NOT a paired baseline. Resubmit w=0 after pull as the
new smoke test — pass criterion is PSNR ~16-20 dB on chunk 1 of these videos and
per-video / per-chunk variation in the GT-free signals (not a constant 0.0021/0.130).

---

## 2026-08-15 — Switch long-horizon work onto the field's 1.3B streaming testbed
tags: [methodology, base-model, dataset, metrics, long-horizon, streaming, wan2.1, vbench-long]
refs: sweep_experiment/reports/paper_tables/2026-08-15_longhorizon_field_standard.md;
CausVid (Yin et al., CVPR 2025); Self-Forcing (Huang et al., NeurIPS 2025 Spotlight);
Pyramid Flow (Jin et al., ICLR 2025); FIFO-Diffusion (Kim et al., NeurIPS 2024);
FreeNoise (Qiu et al., ICLR 2024); One-Minute TTT (Dalal et al., CVPR 2025);
History-Guided / DFoT (Song et al., ICML 2025); VBench (Huang et al., CVPR 2024)

User decision: LongCat 13.6B is too expensive for the N we need now that the task is
long-horizon / streaming, and we should adopt the field's model + data + metrics.
Survey restricted to peer-reviewed 2024–2025 venue papers (no lightly-cited arXiv).

FINDING: the published streaming/long-horizon standard is **Wan2.1-T2V-1.3B**
(CausVid CVPR'25, Self-Forcing NeurIPS'25), eval on **VBench / VBench-Long** and
**MovieGen-128** prompts at 5 s / 10 s / 30 s, headline metrics = VBench quality
dims (subject/background consistency, flicker, motion smoothness, imaging/aesthetic,
dynamic degree) + human, not PSNR. Training-source clips in that literature are
3–10 s (MixKit, WebVid, OpenVid, Kinetics); nobody uses Panda short-clip
continuation as the long-horizon test. Self-Forcing explicitly reports quality
collapse when extrapolating past its 5 s train horizon — that is the headroom.

DECISION: switch the experimental stack to Wan2.1-1.3B (prefer CausVid/Self-Forcing
causal 1.3B checkpoint for streaming AR), VBench + MovieGen-128 prompts, VBench-Long
quality 7 as the paper headline. Keep LongCat results as the saturated-13B audit.
Keep best-of-N / gated TTC as the method (backbone-agnostic). Finish the already-
submitted LongCat TTC w0 v2 smoke only; do not launch more LongCat arms. Next:
Wan 1.3B NOTTA 5 s vs 30 s VBench-Long smoke on ~16 MovieGen prompts.

---

## 2026-08-15 — CORRECTION: stay in continuation / I2V; T2V was not required
tags: [methodology, continuation, i2v, correction]
refs: sweep_experiment/reports/paper_tables/2026-08-15_longhorizon_field_standard.md;
CausVid (CVPR 2025) I2V/V2V claims; VBench-I2V (official VBench++ extension);
History-Guided / DFoT (ICML 2025)

The previous entry recommended T2V because that is the *default task* of CausVid /
Self-Forcing / Pyramid Flow, not because continuation is invalid. User asked whether
we can stay in video continuation. Yes — and we should.

WHY T2V WAS SUGGESTED (and why that was the wrong coupling): those 1.3B streaming
papers generate from text (or a first frame treated as T2V-with-an-image). I
collapsed "switch to their small model" into "switch to their T2V task." Those are
independent knobs. Our scientific claim is exposure bias under *visual*
re-conditioning — a continuation problem. T2V-from-scratch removes the conditioning
tail our verifier, gate, and TTC anchor all read.

WHAT THE FIELD ALREADY OFFERS FOR CONTINUATION:
- CausVid (CVPR 2025) explicitly does streaming **I2V and V2V** on the same 1.3B
  causal student (zero-shot).
- **VBench-I2V** (VBench++ official): i2v_subject, i2v_background, camera_motion +
  the 6 quality dims. This is the conditioned analogue of VBench-Long.
- DFoT (ICML 2025) is video *prediction* (history frames → 64-frame rollout) with
  FVD on Kinetics-600 — the other published continuation-shaped setting.

REVISED STACK: Wan2.1-1.3B (CausVid/Self-Forcing causal ckpt) used as **I2V /
prefix-conditioned AR continuation**, eval on **VBench-I2V** at 5/10/30 s. Optional
second table: Kinetics-600 64-frame FVD (DFoT protocol). Do not move the paper to
T2V-from-scratch.

---

## 2026-08-15 — Cluster setup chain for Wan2.1-1.3B / Self-Forcing (do NOT reuse longcat env)
tags: [infra, wan, self-forcing, conda, sbatch]
refs: wan_experiment/README.md; wan_experiment/sbatch/{setup_env,download_assets,healthcheck}.sbatch;
wan_experiment/sbatch/submit_setup_chain.sh

The LongCat conda env (`/scratch/wc3013/conda-envs/longcat`, numpy 2.x / torch 2.6)
cannot host Self-Forcing (pins numpy==1.24.4, diffusers==0.31.0). Same reason we
already have a separate `vbench-backfill` env. New env: `conda-envs/self_forcing`.

Overnight chain (jobs 1+2 parallel, 3 afterok): (1) GPU env create + clone
Self-Forcing + pip + optional flash-attn; (2) CPU download Wan-AI/Wan2.1-T2V-1.3B
(~15 GB) + gdhe17/Self-Forcing DMD ckpt + VBench-I2V image suite (gdown; non-fatal
if Drive rate-limits); (3) GPU healthcheck writes
`wan_experiment/results/setup_healthcheck/report.json`. Submitter:
`bash wan_experiment/sbatch/submit_setup_chain.sh`. User can disconnect.

---

## 2026-08-15 — Wan setup_env failed on TensorRT extras; download already done
tags: [infra, wan, self-forcing, conda, pycuda]
refs: wan_experiment/sbatch/setup_env.sbatch; jobs 15772007/008/009

15772008 (download) COMPLETED in 3m53s: Wan2.1-T2V-1.3B 17G + self_forcing_dmd.pt
5.3G on disk. VBench-I2V gdown exited 1 on a Drive permission/rate-limit (131M
partial images remain; non-blocking). 15772007 (env) FAILED in 10m: official
Self-Forcing `requirements.txt` pulls `nvidia-pyindex` / `nvidia-tensorrt` /
`pycuda`; pycuda died with `cuda.h: No such file or directory`. 15772009
CANCELLED by afterok. Env skeleton + SF clone already exist — do not FORCE=1
wipe. Fix: strip those three lines before pip (inference unused). Resubmit
setup_env + healthcheck only; do not re-download.

---

## 2026-08-15 — setup_env TIMED OUT compiling flash-attn (15796574)
tags: [infra, wan, self-forcing, flash-attn, slurm]
refs: wan_experiment/sbatch/setup_env.sbatch; jobs 15796574/575

15796574 ran 2h00m on gh125 then TIMEOUT. Log is 60 lines of
`Building wheel for flash-attn ... still running`. TensorRT-skip worked;
`setup.py develop` never ran. 15796575 CANCELLED by afterok. Reason field
also shows `QOSMaxGRESPerUser` (job sat in Priority ~4.5h before starting).

Fix: skip flash-attn by default (`SKIP_FLASH=1`), run `setup.py develop`
first, drop `--gres` so setup is a CPU job (avoids the 2-GPU cap), 12m
`timeout` if someone sets `SKIP_FLASH=0`. Inference does not need flash-attn.
Resubmit setup_env + healthcheck only. Do not FORCE=1.

---

## 2026-08-16 — Wan / Self-Forcing healthcheck GREEN (15858269)
tags: [infra, wan, self-forcing, healthcheck]
refs: wan_experiment/results/setup_healthcheck/report.json; jobs 15858268/269

Required checks all passed. On disk and loadable: Wan2.1-T2V-1.3B config + T5
(22.7 GB across 2 files) + VAE + Self-Forcing DMD 5.3 GB (`generator_ema`).
VBench-I2V: 105 images found, 8 decoded (partial Drive download was enough).
Env: torch 2.13.0+cu130, CUDA yes, NVIDIA H200. `n_tensors=0` is a schema
quirk (ckpt top key is `generator_ema`, not `state_dict`/`model`) — file
loaded; unwrap that key in the runner.

Optional `smoke_t2v` failed in 16.6s: official Self-Forcing `inference.py`
does `from torchvision.io import write_video`, removed in torchvision bundled
with torch 2.13. Do not use that entry point. Write our own I2V / prefix-
conditioned continuation runner (imageio/av). Do not pin-downgrade torch
unless a forward pass actually breaks — 2.13+cu130 matches the H200 node.

Setup chain is closed. Next experiment: NOTTA 5 s vs 30 s VBench-I2V smoke
on ~16 images, then port best-of-N + gated TTC.

---

## 2026-08-16 — I2V continuation runner (NOTTA smoke first)
tags: [wan, self-forcing, i2v, continuation, infra]
refs: wan_experiment/scripts/run_i2v_continuation.py;
wan_experiment/sbatch/{run_i2v_notta.sbatch,submit_i2v_smoke.sh}

Built our own runner around official `CausalInferencePipeline.inference`
(`--i2v` path): resize 480×832, VAE-encode first frame, AR denoise with
KV cache, imageio mp4. Overrides: `independent_first_frame=true` (else a
1-frame prefix fails the block-size assert); KV cache enlarged past the
hardcoded 21-frame / 32760-token default (required even for 5 s = 22
latent frames, and mandatory for 30 s). `n_gen` rounded up to a multiple
of `num_frame_per_block=3`. Symlink `Self-Forcing/wan_models/Wan2.1-T2V-1.3B`
→ `/scratch/wc3013/wan-checkpoints/Wan2.1-T2V-1.3B` because wan_wrapper
hardcodes that relative path.

Gating smoke: 2 VBench-I2V images × 5 s, series `i2v_notta_smoke`. Do not
submit 16×{5,30}s until mp4s look like video. Then port best-of-N + gated
TTC onto this sampler.

---

## 2026-08-16 — Gating must not lose to always-on BoN / always-on TTC
tags: [methodology, gating, ablation, novelty, best-of-N, ttc]
refs: ANALYSIS_LOG 2026-08-10 pivot + 2026-08-11 bestof credibility;
user question 2026-08-16

User correctly flagged a load-bearing novelty risk: if the claimed
contribution is the GATE, then `gated` must be compared to **always
intervene**. If always-BoN or always-TTC matches or beats gated on
quality, the controller story is false (or collapses to "a verifier
for BoN," which we already said is not novel).

What we actually have, vs what we hypothesized:
- LongCat best-of-N k=4 was **always-on search**. cand0 = NOTTA, so the
  verifier can *soft-skip* by picking cand0 (it did on 25% of chunks).
  That is not the same as a hard incoming-context gate that skips the
  extra k-1 samples. We have never run gated-BoN vs always-BoN.
- `ttc` vs `ttc_gated` was designed as that ablation; TTC never passed
  a clean smoke, so we have **no** gated-vs-always quality number.
- The *reason* to expect gating to help is the closed delta line:
  intervening on non-drifted chunks can hurt (ramp contraindicated;
  significant chunks all negative). That argument is stronger for TTC
  (it rewrites the trajectory) than for BoN (cand0 is already in the
  pool; always-on BoN can still pick NOTTA).

LOCKED comparison on Wan (same seeds, same images, same horizon):
  NOTTA | always-BoN | gated-BoN | always-TTC | gated-TTC
  and, if both actuators are live, the joint controller (gate chooses
  skip / BoN / TTC). Headline: gated vs its always-on twin, paired.
Pass for the gate: quality ≥ always-on on the endpoint (VBench-I2V /
|drift|), and strictly cheaper (skipped interventions). A quality *loss*
vs always-on kills the gating claim even if we save compute. A quality
tie + compute win is a valid efficiency paper, not a "gating fixes
drift" paper — say that plainly. A quality win is the controller paper.

---

## 2026-08-16 — I2V smoke died on flash-attn assert; SDPA fallback
tags: [infra, wan, flash-attn, sdpa]
refs: job 15858704; wan/modules/attention.py:118; wan_experiment/scripts/run_i2v_continuation.py

`i2v_notta_smoke` n_ok=0. Both videos hit `assert FLASH_ATTN_2_AVAILABLE`
on the first DiT forward. Pipeline load succeeded. Cause: we skip
compiling flash-attn (15796574 TIMEOUT); Self-Forcing's `model.py`
imports `flash_attention` (hard assert), while the SDPA fallback is
only on `attention()`. Fix: monkeypatch both to PyTorch SDPA when
flash-attn is absent. Do not restart a 2h flash-attn compile. Resubmit
the same 2×5 s smoke.

---

## 2026-08-16 — I2V smoke OOM from 24×32760-token KV cache (15876397)
tags: [infra, wan, oom, kv-cache]
refs: job 15876397; wan_experiment/scripts/run_i2v_continuation.py

SDPA path reached the DiT; H200 filled to 138.10 / 139.80 GiB. That
allocation is `n_frames * pipeline.frame_seq_length` with
`frame_seq_length=32760` (WanDiffusionWrapper.seq_len = 21 frames of
1560 tokens), not 1560 tokens/frame. 24×32760×30×2×12×128×2 B ≈ 135 GB.
Fix: hardcode `FRAME_SEQ_PER_LATENT=1560`, print estimate, refuse >48 GB.
5 s cache should be ~7 GB. Resubmit 2×5 s smoke. Do not treat this as a
model-too-big problem.

---

## 2026-08-16 — 15877786 still 138 GB OOM; disable flex_attention compile
tags: [infra, wan, oom, torch-compile, flex-attention]
refs: job 15877786; wan/modules/causal_model.py (torch.compile max-autotune)

KV-cache cap held (job would have aborted a 135 GB cache). Same 138.10 GB
PyTorch fill, now in the denoise loop. Remaining suspect: Self-Forcing
compiles `flex_attention` with `max-autotune-no-cudagraphs` at import;
on H200 that autotune workspace can consume the card. Fix: set
`TORCH_COMPILE_DISABLE=1` before import, replace `flex_attention` with
eager, T5 via DynamicSwapInstaller, `low_memory=True`. Resubmit 2×5 s.

---

## 2026-08-16 — Stop blind I2V smokes; 138 GB persists (15879723)
tags: [infra, wan, oom]
refs: jobs 15876397, 15877786, 15879723; wan_experiment/scripts/probe_vram.py

Three smokes, three identical ~138 GB fills. KV-cache cap did not fire;
compile-disable did not shrink the footprint. Next is a load-only VRAM
probe (tensor shapes + memory_summary), not another 5 s generate. The
138 GB number still matches `24 × 32760 × 30 × K+V × 12 × 128 × bf16`
— something is still allocating a 24-frame cache at 32760 tokens/frame
even if our enlarge() print claims 1560.

---

## 2026-08-16 — 138 GB OOM was autograd, not the KV cache (15879723 head)
tags: [infra, wan, oom, autograd]
refs: wan_i2v_notta_15879723.out head; wan_experiment/scripts/run_i2v_continuation.py

Log head: after_load 3.09 GB, KV cache print `24 x 1560 = 37440 tokens,
est 6.90 GB`, after_kv_init 10.06 GB. Two denoise blocks print timesteps,
then 138 GB OOM. Official Self-Forcing `inference.py` line is
`torch.set_grad_enabled(False)`; we never set it. WanDiffusionWrapper
always passes `seq_len=32760` (pads every block). Autograd over a few
30-block padded forwards explains the fill. Fix: disable grad +
`inference_mode()`. KV-cache and compile-disable work was not wasted
(those were real bugs) but they were not this 138 GB. Resubmit 2×5 s
smoke. VRAM probe optional.

---

## 2026-08-16 — Wan I2V NOTTA smoke passed (15880611)
tags: [infra, wan, i2v, notta, smoke]
refs: job 15880611; wan_experiment/results/i2v_notta_smoke/h5s_shard0/

COMPLETED 0:0 in 2:55 on gh118. n_ok=2/2, 85 frames, 480×832, mp4s 5.9 MB
and 3.9 MB. Per-clip generate 11.99 s then 8.01 s. Autograd-off (`65ba50c`)
was the last OOM fix. Stack is now usable: Wan2.1-T2V-1.3B + Self-Forcing
causal DMD, official CausalInferencePipeline I2V path, imageio writer,
SDPA fallback, KV cache at 1560 tok/frame, compile disabled, T5
DynamicSwap, `inference_mode`. Next: eyeball one mp4 (first frame ≈ cond
image, not TTC-v1 noise), then 16×{5,30}s NOTTA, then port the GT-free
verifier and the required five-way:
NOTTA | always-BoN | gated-BoN | always-TTC | gated-TTC.
Do not reopen LongCat TTC. Do not rebuild the env.

---

## 2026-08-16 — Smoke first-frame MAE confirms I2V (5.56 / 3.71)
tags: [infra, wan, i2v, smoke, fidelity]
refs: job 15880611; /tmp/wan_first_{000,001}.png

Frame 0 vs resized cond jpg: bubbles MAE 5.56, pot MAE 3.71 (uint8).
That is VAE-roundtrip I2V, not decoded noise. Content gate passed.
Submit `i2v_notta_16v` 16×{5,30}s. Then port verifier / five-way.

---

## 2026-08-16 — Wan I2V NOTTA 16v passed at 5 s and 30 s
tags: [infra, wan, i2v, notta, timing]
refs: wan_experiment/results/i2v_notta_16v/; paper_tables/2026-08-16_wan_i2v_notta16.md

n_ok=16/16 at both horizons. 5 s: 85 px, mean 9.61 s/clip. 30 s: 481 px,
mean 38.32 s/clip (~1.27 s GPU per generated second). Stack is fast
enough for BoN (k=4 would be ~40 s at 5 s, ~150 s at 30 s). Quality
headroom is unknown: next measurement is first-1s vs last-1s GT-free
drift on these mp4s (`score_i2v_drift.py`). If 30 s is flat, do not
claim a controller win at this horizon — go longer or accept no-drift.
Do not implement the five-way until that table exists.

---

## 2026-08-17 — 30 s drift score Killed; 5 s motion number is invalid
tags: [infra, wan, drift, oom]
refs: score_i2v_drift.py first run; i2v_notta_16v

5 s finished. Mean tail/head (windows included frame 0): sharp +35%,
color +7%, contrast +10%, motion −31%. Do not cite the motion drop —
I2V frame 0 is the cond still, so the reference window contains the
still→video jump. Sharpness mean is outlier-dominated (004 +268%,
011 +126%). 30 s process was `Killed` on the login node: full-clip
float32 load is ~2.3 GB. Fix: stream first-1s-after-cond vs last-1s,
report mean and median. Rerun before any five-way decision.

---

## 2026-08-17 — Wan 30 s NOTTA drifts; five-way must be chunked
tags: [wan, drift, notta, five-way, gating]
refs: i2v_notta_16v/drift_head_tail.json; paper_tables/2026-08-17_wan_i2v_notta16_drift.md

N=16, same images/seed, 1 s after cond frame vs last 1 s. Medians:

| H | sharp | color | contrast | motion |
|---|---|---|---|---|
| 5 s | +11% | +9% | +9% | −14% |
| 30 s | +167% | +28% | −5% | −60% |

30 s sharpness up on 15/16; motion down on 15/16. Headroom is real.
Signature = sharpen + freeze (not LongCat's sharpen + motion inflation).
Cite medians. Means are outlier-pulled.

**Gating lock:** I2V t=0 incoming context is the cond still. A clip-level
gate has nothing to fire on, so gated-BoN = NOTTA and the required
comparison is vacuous. Five-way on Wan is a **chunked 30 s rollout**
(e.g. 6 × 5 s blocks, 21 gen latents each). At each chunk boundary,
score last-1s vs first-1s-after-cond reference + seam. Always-on
actuates every chunk; gated actuates only if relative |drift| exceeds
a threshold. cand0 of BoN = NOTTA seed for that chunk. Same 16 images,
seed 0, 30 s. Do not implement TTC until BoN chunk path generates
real video (Wan TTC-v1 lesson). Next code: chunked inference hook on
CausalInferencePipeline, then NOTTA vs always-BoN k=4 smoke on 2 clips.

---

## 2026-08-17 — Chunked Wan BoN runner (no TTC yet)
tags: [wan, bon, chunked, verifier]
refs: wan_experiment/scripts/run_i2v_chunked.py; i2v_verifier.py

Official `inference()` with `independent_first_frame=True` only KV-caches
`initial_latent[:, :1]`, so a growing prefix cannot be passed back through
that API. Runner replays committed latents at t=0 (same path as I2V init),
then denoises the next chunk. 30 s = 5 × 24 gen latents (~6 s). Chunk 0 is
always seed 0 (shared prefix + first-1s-after-cond reference). always-BoN
k=4 searches chunks 1–4; cand0 = seed 0. Verifier is the LongCat composite
(relative |dev| of sharpness/color/contrast/motion + seam). TTC is not
implemented — wait for this 2×30 s smoke to write real mp4s. Submit:
`wan_experiment/sbatch/submit_i2v_bon_smoke.sh`.

---

## 2026-08-17 — Chunked BoN smoke: search alive; seed the sampler
tags: [wan, bon, smoke, rng]
refs: jobs 15883525 / 15883526; paper_tables/2026-08-17_wan_i2v_chunked_bon_smoke.md

n_ok=2/2 both methods, 481-frame multi-MB mp4s. always-BoN left cand0
on 5/8 searchable chunks (bubbles 3, pot 2). NOTTA verifier scores rise
on later chunks (pot 1.77→4.81). Infra + search pass. Quality pairing
is invalid: chunk 0 cand0 already differs (3.305 vs 2.992) because
denoise `add_noise` used global `randn_like`. Fix: one Generator per
(cand, chunk) for init noise and step noise. Do not resubmit the 2-clip
smoke. Do not add TTC. Next 16v five-way only after the RNG fix.

---

## 2026-08-17 — Seed-invariant chunk RNG + gated-BoN 16v
tags: [wan, bon, rng, gated]
refs: run_i2v_chunked.py; submit_i2v_bon16.sh

Per-(cand, chunk) CUDA Generator for init noise and add_noise. Process
flags: cudnn deterministic, TF32 off, use_deterministic_algorithms
warn_only, CUBLAS_WORKSPACE_CONFIG, re-seed before each video. cand0
chunk i is identical across NOTTA / always-BoN / gated-BoN when the
committed prefix matches. gated-BoN searches iff incoming last-1s
composite > 2.0 (smoke NOTTA later chunks were 3–5; early ~1–2).
Submit 16v 30 s three-way, no TTC. First check: chunk-0 cand0 scores
must match across methods; if not, stop and do not scale.

---

## 2026-08-17 — 16v three-way: seed match; gate is a real controller
tags: [wan, bon, gated, 16v]
refs: jobs 15884598/599/600; paper_tables/2026-08-17_wan_i2v_bon16.md

n_ok=16/16. Chunk-0 cand0 scores identical on all 16 videos — RNG fix
held. always-BoN left cand0 on 43/64 chunks. gated-BoN fired 27/64,
skipped 4 videos entirely (those mp4s = NOTTA). Mean wall 84 / 267 /
211 s (gated 21% cheaper than always). Gate fires more on later chunks
as incoming drift grows. This is the first valid paired Wan comparison.
Do not claim a quality win until last-chunk composites are tabulated.
Locked test still stands: gated must not lose to always-on. No TTC yet.

---

## 2026-08-17 — Last-chunk: search works; gated is efficiency, not a quality win
tags: [wan, bon, gated, quality]
refs: paper_tables/2026-08-17_wan_i2v_bon16_lastchunk.md

Last-chunk composite (lower better), N=16: NOTTA 4.43 / always 3.23 /
gated 3.38. always−NOTTA −1.20 (14/16 better). gated−NOTTA −1.05.
gated−always +0.152 mean, −0.131 median, 6/16 better-or-tie. always-on
hurt NOTTA on 06 and 07; gated was better on both. Locked rule: this
is not a quality win vs always-on. Closest bucket is **tie + cheaper**
(gated 21% less wall). Do not claim gated beats always-on. Do not drop
gating. No TTC yet. Optional next: endpoint sharp/motion on the same
mp4s (`score_i2v_drift.py`) as a second metric.

---

## 2026-08-17 — Single gate threshold cannot beat always-on
tags: [wan, gated, threshold, diagnosis]
refs: paper_tables/2026-08-17_wan_gate_threshold_diagnosis.md

always-on first diverges at chunk 1 on 13/16 videos. T=2.0 skips that
chunk (incoming 0.2–1.3). Big gated misses (02, 03, 05, 09, 12) have
incoming 0.87–1.27 at first always-div. Correct skips (06, 07, always
hurt NOTTA) have incoming 0.20 and 0.68. No global T separates 0.68
from 0.87. Next: dump always-on chunk-1 candidate scores (shared
prefix, valid offline) and try a trend/Δincoming gate, not another
16v T-sweep. Do not “always search chunk 1.”

---

## 2026-08-17 — Hybrid gate: T=0.8 at chunk 1 + late T=2 + trend
tags: [wan, gated, hybrid]
refs: paper_tables/2026-08-17_wan_gate_threshold_diagnosis.md (chunk-1 dump)

ch1 best−cand0: 05 is −1.08 (all of its last-chunk gain). 03 is −0.01
(coin flip). 06/07 have local ch1 gains (−0.44, −0.22) but always-on
hurt the endpoint — local verifier ≠ last-chunk. 12 has zero ch1
headroom; miss is ch2 Δ=+0.55. Hybrid: fire chunk 1 if incoming>0.8;
else fire if incoming>2.0; else fire if Δincoming>0.5 and prev>0.5.
Keeps 06/07 early-skip. Next is implement + 2-clip smoke, not another
blind T sweep.

---

## 2026-08-17 — Hybrid gate implemented; 32v three-way ready
tags: [wan, gated, hybrid, 32v]
refs: run_i2v_chunked.py; submit_i2v_bon32_hybrid.sh;
paper_tables/2026-08-17_wan_i2v_hybrid_gate_spec.md

gated_bon now fires if (chunk==1 and incoming>0.8) or incoming>2.0 or
(Δincoming>0.5 and incoming_prev>0.5). T=2.0 remains the late level.
Per-chunk logs: incoming, prev, Δ, gate_reason (ch1/level/trend/skip),
incoming+outgoing per-signal |dev|, cand0 vs chosen, last-1s outgoing
composite. Sidecar `gate_trace.jsonl` one line per (video, chunk).
Analyzer: `analyze_i2v_bon.py`. Submit N=32 30 s three-way (re-run
NOTTA and always-on so the schema matches). Skip the 2-clip smoke —
16v already validated RNG. Hypothesis: if ch1-fire videos then follow
always-on and 06/07 stay skipped early, gated−always last-chunk mean
flips from +0.15 to about −0.2. Falsifiers: prefix diverges without
endpoint gain; trend fires 07 at a later chunk; local score still
fails to predict last-chunk. No TTC.

---

## 2026-08-17 — 32v hybrid: tie + 33% cheaper; do not cite means
tags: [wan, bon, gated, hybrid, 32v]
refs: paper_tables/2026-08-17_wan_i2v_bon32_hybrid.md;
wan_experiment/results/i2v_bon_32v_hybrid/

n_ok=32/32. Video 26 (spiral galaxy) last-chunk 85.63 for always and
gated vs NOTTA 5.06 — raw means are unusable. Cite medians: NOTTA
3.68 / always 2.97 / gated 3.04. Exclude-26 means: 3.92 / 3.08 /
3.04. gated−always −0.041 mean, 0 median, 9/10/13, 19/32
better-or-tie. Wall 92 / 258 / 173 s (gated 33% cheaper). Fired
66/128. First-16 NOTTA/always match `i2v_bon_16v` (4.429 / 3.226) —
pairing held. Hybrid flipped that slice's gated−always from +0.152
to −0.118 (drop 03 → −0.210). 05/02/09 matched always; 12 almost;
06/07/28/30 saved from always-hurt. Leftover misses: 17 never-fire,
03/24 ch1-then-skip, 26 search catastrophe (gate followed). Locked
read: still not a quality win vs always-on. Efficiency paper. Do not
drop gating. Next lever is stay-on hysteresis, not another T sweep.
No TTC.

---

## 2026-08-18 — Sticky gate: once fired, keep searching
tags: [wan, gated, sticky, 32v]
refs: run_i2v_chunked.py --gate-sticky; submit_i2v_bon32_sticky.sh;
paper_tables/2026-08-18_wan_i2v_sticky_gate_spec.md

Same three hybrid alarms. New flag --gate-sticky: after the first
search on a video, every later piece searches too (reason already_on
if no fresh alarm). Default off, so the hybrid 32v series stays
reproducible. Submit gated-only into i2v_bon_32v_sticky; pair against
hybrid do-nothing and always-search. Pass/fail: 03 and 24 should move
toward always-search; 06/07/28/30 must stay skipped on piece 1; watch
for a second 26-style explosion. Does not fix 17 (never wakes). No
test-time training.

---

## 2026-08-18 — Sticky 32v: 03/24 caught; became always-search
tags: [wan, gated, sticky, 32v]
refs: paper_tables/2026-08-18_wan_i2v_bon32_sticky.md;
wan_experiment/results/i2v_bon_32v_sticky/

n_ok=32/32. Median last-piece do-nothing 3.68 / always-search 2.97 /
sticky 2.99. sticky−always −0.012 mean, 0 median, 6/21/5. Fired
96/128 (28 already_on). Wall 256 vs always-search 258 s — the 33%
hybrid saving is gone. 03 and 24 now exact ties with always-search
(the intended fix). 06/07 still skipped on piece 1 and still beat
always-search. 30 was un-saved (copied always-search harm). No second
video-26 explosion. Hybrid’s two unique wins (11: 2.16, 16: 2.66)
were erased; local chosen−cand0 kept improving while the ending got
worse. 17 still never wakes. Locked read: stay-on works as designed
and is not a quality win. It is delayed-start always-search. Cite
hybrid if the claim is cheaper-at-same-typical-quality. Do not stack
another stay-on. No test-time training.

---

## 2026-08-18 — Videos 11/16: recovery skip vs lying pick-score
tags: [wan, gated, sticky, verifier, diagnosis]
refs: paper_tables/2026-08-18_wan_i2v_11_16_diagnosis.md

Hybrid unique wins (11: 2.16, 16: 2.66) came from sleeping after a
good piece-1 search. 11 recovered 2.38→1.11; 16 stayed healthy at
0.88. Stay-on forbade those skips, rebuilt always-search’s path, and
the pick-score kept reporting wins (11 piece 4 −4.11, 16 piece 4
−3.64) while last-second outgoing exploded (4.91, 7.45). Same shape
on 01 and 30. Opposite of 03/24, where incoming stayed high/flat and
later search was the win. Large piece-1 pick-score is the wrong stay-on
cue (11 had −1.10 and should sleep; 03 had −0.01 and should continue).
Useful cue: incoming after the search. Next lever is “search while
sick” (turn off on recovery / low incoming), not another forever
stay-on and not test-time training. Pick-score-on-the-tail is the
backup if the off-switch is not enough.

---

## 2026-08-18 — Search-while-sick implemented; 32v gated-only ready
tags: [wan, gated, sick, 32v]
refs: run_i2v_chunked.py --gate-sticky --gate-sick-min 1.0
--gate-recovery 0.5; submit_i2v_bon32_sick.sh;
paper_tables/2026-08-18_wan_i2v_sick_gate_spec.md

Same hybrid alarms + stay-on, but memory turns off if the last
search recovered incoming by more than 0.5 or outgoing last-second
is below 1.0. Knobs default to 0 so forever-sticky stays
reproducible. Gated-only series i2v_bon_32v_sick; pair against
hybrid do-nothing / always-search. Pass/fail: 11/16 near hybrid,
03/24 near always-search, 06/07 still skipped early, 30 back to
1.44, wall between 173 and 256 s. Trace caveat: 11 may search
piece 4 after a late alarm (prefix still hybrid through piece 2);
03 may turn off after piece 3 recovery. No test-time training.

---

## 2026-08-18 — Search-while-sick submitted; controller briefing
tags: [wan, gated, sick, briefing]
refs: job 15959146;
paper_tables/2026-08-18_wan_controller_briefing.md

User pulled d5b0804..52d1718 and submitted
submit_i2v_bon32_sick.sh. squeue: 15959146 h200_cour PD Priority.
Briefing written: setup, four GT-free signals + seam, incoming /
score / outgoing / recovery equations, hybrid vs stay-on headline
medians, in-flight checks. Cite medians. No test-time training.

---

## 2026-08-18 — Outcome eval must be official VBench, not the verifier
tags: [wan, methodology, vbench, eval, controller]
refs: paper_tables/2026-08-18_wan_i2v_official_eval_spec.md;
wan_experiment/scripts/score_i2v_vbench.py;
wan_experiment/scripts/analyze_i2v_vbench.py;
wan_experiment/sbatch/submit_i2v_vbench_hybrid32.sh

User lock: the controller still may not peek at ground truth when it
fires or picks. The finished videos must still be scored with common
metrics, or we cannot tell if do-nothing / always-search / gated-search
are improving, and we cannot tell if the handcrafted composite even
detects drift that helps performance. 11/16 already showed the
pick-score can lie.

These 32 clips are VBench-I2V stills — no paired 30 s GT, so no
PSNR/SSIM/LPIPS/FVD here. Official scorecard is VBench quality dims
(same family as Panda 1000v) on the existing hybrid mp4s. last5 first
(outcome window), then full 30 s (diluted by shared piece 0). Analyzer
prints the three-way table plus Spearman(last-chunk, each VBench dim);
expected sign if the verifier is a useful proxy is negative.

i2v_subject / i2v_background need vbench2_beta_i2v and are not in the
first job. A Panda-prefix Wan series is a later optional pixel audit,
not a rescoring of the current 32. No test-time training.

---

## 2026-08-18 — Two in-flight jobs: sick running, VBench queued
tags: [wan, jobs, sick, vbench]
refs: job 15959146; job 15959601;
paper_tables/2026-08-18_wan_in_flight_jobs.md

User pulled c6eef97 and submitted submit_i2v_vbench_hybrid32.sh.
squeue 11:47: 15959146 R 14:15 gh107 (i2v_bon_32v_sick);
15959601 PD Priority (hybrid VBench last5+full). Check-when-done
commands live in the in-flight table. No test-time training.

---

## 2026-08-18 — Search-while-sick 32v: checklist pass on handcrafted score
tags: [wan, gated, sick, 32v]
refs: job 15959146;
paper_tables/2026-08-18_wan_i2v_bon32_sick.md;
wan_experiment/results/i2v_bon_32v_sick/

n_ok=32/32, finished 14:24 EDT. Median last-piece do-nothing 3.679 /
always 2.966 / sick **2.764**. sick−always −0.155 mean, 0 median,
9/14/9. Wall 204 s (between hybrid 173 and sticky 256). Fired 84/128.

Pass/fail vs spec: 11 **1.830** (beat hybrid 2.157; sticky had 4.319);
16 **2.656** exact hybrid; 24 exact always 2.315; 03 **1.755** vs
always 1.567 (piece 4 off after 1.674→1.019 recovery, as predicted;
hybrid miss was +1.26); 06/07 still skipped on piece 1; 30 back to
1.444. 17 still never wakes. 26 still 85.63.

Locked read: first gated rule that keeps hybrid unique wins and
catches 24 without becoming always-search. Best handcrafted median so
far, 21% cheaper than always. Not a strict quality win (9–9). Do not
cite as paper quality until official VBench has all three methods.
No test-time training.

---

## 2026-08-18 — VBench job 15959601 scored do-nothing only
tags: [wan, vbench, infra, bug]
refs: job 15959601;
wan_experiment/sbatch/submit_i2v_vbench_hybrid32.sh

squeue gone; analyze failed: no always_bon / gated_bon joined.json.
Root cause: sbatch `--export=ALL,VIDEO_DIRS=a,b,c` — SLURM splits
export values on commas, so only the notta path survived. notta
last5+full completed (exit 0). do-nothing full-clip dynamic_degree
median 0.0 is a freeze hint, not a three-way result.

Fix: SERIES_DIR + space-separated METHODS, no commas. Rerun skips
existing notta files. Resubmit after cluster pull. No PSNR. No TTC.

---

## 2026-08-18 — VBench retry submitted job 15984561
tags: [wan, vbench, jobs]
refs: job 15984561; paper_tables/2026-08-18_wan_in_flight_jobs.md

User pulled 4116c3a and resubmitted. Analyze run immediately after
submit still failed (only notta joined.json). Expected. Wait for
always_bon and gated_bon last5+full. No TTC.

---

## 2026-08-18 — Official VBench: search/gating do not improve quality
tags: [wan, vbench, methodology, hybrid, verifier]
refs: jobs 15959601 15984561;
paper_tables/2026-08-18_wan_i2v_bon32_vbench_read.md;
paper_tables/2026-08-18_wan_i2v_bon32_vbench_last5.md;
paper_tables/2026-08-18_wan_i2v_bon32_vbench_full.md

Three-way official VBench on the hybrid 32 mp4s is in. Cite last5.

Do-nothing last5 imaging 68.17 vs always 66.43 vs gated 66.11;
background 0.957 vs 0.952 vs 0.952. Always wins aesthetic only
(0.548 vs do-nothing 0.535 vs gated 0.522). dynamic_degree median
0.0 for all three; always mean 0.250 vs 0.188 (≈2 extra dynamic
clips). Gated vs always win counts do not favor gating except a
small subject edge.

Spearman(last-chunk, imaging_quality) last5 is +0.229 / +0.243 /
+0.327. The handcrafted composite punishes sharpness deviation;
MUSIQ rewards sharpness. Most other |ρ| < 0.3. Motion smoothness
under search is the only useful negative ρ (−0.26 / −0.30).

Locked read: the handcrafted “search works” last-chunk table does
not survive official metrics. Gating is not a quality win. The
efficiency line is now “about the same VBench, slightly worse last5
Aes/IQ, 33% cheaper.” Do not treat sick’s better composite as a
quality win without scoring those mp4s — the signal may be
anti-aligned with IQ. No PSNR. No TTC.

---

## 2026-08-18 — last5 is diagnostic; BoN does not “worsen VBench++”
tags: [wan, vbench, methodology]
refs: paper_tables/2026-08-18_wan_i2v_bon32_vbench_read.md;
VBench-I2V; VBench-Long (vbench2_beta_long)

User asked whether last-5s-only is a common horizon protocol, and
whether we are claiming best-of-4 worsens VBench++.

last5 is **not** the field headline. VBench-I2V scores the full
native-length clip. VBench-Long scores the whole long video via
scene-split + fixed clips + slow/fast aggregation. Our 2026-08-15
lock was generate-at 5/10/30 s, not crop-the-tail of 30 s. last5
stays as a divergence diagnostic (shared piece 0 dilutes the full
clip). Paper tables that say “VBench++” should use the full 30 s
numbers.

Full-clip always vs do-nothing is a **tie** (Aes +0.006, subject
+0.007, IQ +0.04, motion −0.001, dynamic median 0). Do not write
“BoN worsens VBench++.” Fair narrower claim: last5 imaging drops
(68.2 → 66.4) while last5 Aes rises (0.535 → 0.548). Gating is
still not a quality win on either window. No TTC.

---

## 2026-08-18 — Full generated clip is the required VBench++ number
tags: [wan, vbench, methodology]
refs: score_i2v_vbench.py --clip full;
analyze_i2v_vbench.py --clip full;
run_i2v_vbench.sbatch CLIPS=full last5

User lock: always do the normal thing other papers do. Score the
**full generated clip** as the official comparable VBench++ table.
last5 may still run as a diagnostic (more pronounced tail), but it
is never the paper’s VBench++ number and must not replace full.
Defaults flipped: score/analyze `--clip full`; sbatch
`CLIPS=full last5` (full first so preemption still leaves the
comparable number). Hybrid 32 already has both windows; no rescore.
No TTC.

---

## 2026-08-18 — STOP: I2V-32 is not the field long-horizon protocol
**Tags:** decision, methodology, wan, long-horizon
**Owner:** agent
**Refs:**
- `paper_tables/2026-08-18_wan_protocol_stop.md`
- `paper_tables/2026-08-18_wan_i2v_bon32_vbench_read.md`
- Self-Forcing / Relax Forcing / FreqForcing / Self-Forcing++ / VBench-Long
- Rebuts 2026-08-15 “stay in I2V; T2V was not required” *for the
  standard-bench question only*

User asked to verify freeze / search / gating on a larger
industry-standard sample, and to halt if our basic setup is not what
recent similar papers report.

Halt. 5 s is not the long-horizon table — that part was already
correct — but **I2V-from-still × 32 VBench-I2V images × custom_input
VBench** is also not the long-horizon table.

What 2025–2026 long-horizon papers actually do on our model family
(Wan2.1-T2V-1.3B + Self-Forcing-style causal student): **T2V** from
text, AR continue from own KV cache, **N ≈ 128** MovieGen prompts
(often Qwen-refined), score with **VBench-Long** on the **full**
clip, horizon **30 s / 60 s / 120 s**. Self-Forcing uses 5 s as the
main table and 30 s as an extrapolation-failure demo. Official
VBench-I2V is a different protocol again (5 s / 81 frames on
Wan-I2V-14B).

What is fine about our run: 30 s length; Wan 1.3B causal. What is
not: task (still → animate vs text → self-continue), N (32 vs 128),
suite (`custom_input` vs VBench-Long / MovieGen).

**Decision:** do **not** submit I2V-32 or I2V-200 scale-up. The 32-clip
hybrid VBench stays as a discovery scorecard. Do not claim it as a
standard long-horizon result. No TTC.

---

## 2026-08-18 — Comparable verify is T2V 128 MovieGen + VBench-Long
**Tags:** decision, methodology, wan, vbench-long
**Owner:** agent
**Refs:** `paper_tables/2026-08-18_wan_t2v_vbenchlong_128_spec.md`

If we later verify freeze + search/gating against the field, copy
Relax Forcing / FreqForcing / Self-Forcing++:

- Wan 1.3B + Self-Forcing DMD
- **T2V** (not I2V-from-still)
- First 128 MovieGen prompts, Qwen-refined
  (`prompts/MovieGenVideoBench_extended.txt` on the Self-Forcing clone)
- 30 s required; 60 s optional second table
- Methods: do-nothing | always-BoN k=4 | gated-BoN
- Official score: **VBench-Long on the full clip**
- Series name: `t2v_bon_128v_vbenchlong`

This is a **new generate series**, not a rescore of the 32 stills.
`run_i2v_chunked.py` is I2V-only; the next agent must write a T2V
runner. **SPEC READY. Not submitted.** No job until explicitly
launched. No TTC.

---

## 2026-08-18 — Non-weight next methods (after a standard bench)
**Tags:** decision, methodology, wan, search, no-ttc
**Owner:** agent
**Refs:** `paper_tables/2026-08-18_wan_nonweight_next.md`

Brainstorm locked. Do not retune the I2V-32 sharpness-deviation gate
and call that a paper quality win. Do this on T2V 128 / VBench-Long.

Closest field language: Early Failure Detection (intervene only when
failure is predicted); CachedSearch (cheap explore, recommit winner);
BAG/NaviCache (gate NFEs); LatSearch (score partial latents); Temporal
Backtracking Search (rewind prefix). StreamingT2V: long AR stagnates.
History Guidance: vanilla history-CFG kills dynamics.

What our I2V-32 run forbids us to forget: composite anti-aligned with
IQ; `dynamic_degree` is 0/1 and seed-BoN barely leaves the freeze
attractor; pick-score can lie; first-second-after-a-still is a bad
motion target. Changing the seed does not change a collapsed
trajectory.

Try, in order, after the T2V bench exists: (1) failure-gated
CachedSearch; (2) motion / ImageReward verifier; (3) search
`{shift, cfg, sink}` not only seed; (4) prefix backtrack if outgoing
explodes. **No TTC / LoRA-at-test-time.**

---

## 2026-08-18 — Weekly briefing: model / dataset switch + long-horizon concepts
**Tags:** paper-narrative, methodology, wan
**Owner:** agent
**Refs:**
- `paper_tables/2026-08-18_week_switch_briefing.md`
- `paper_tables/2026-08-15_longhorizon_field_standard.md`
- canvas `week-switch-briefing.canvas.tsx`

User asked for a presentation of the past week *before* new-method
experiments: why we switched models, why we switched datasets, and the
key concepts in long-horizon generation, with citations.

The briefing compresses the 15 August field-standard memo plus the 18
August protocol stop. It is the setup talk, not the BoN/gating result
talk. Interactive canvas + dated markdown. No generate job.

---

## 2026-08-18 — T2V was not agreed; V2V continuation is allowed
**Tags:** decision, methodology, wan, continuation
**Owner:** agent
**Refs:**
- `paper_tables/2026-08-18_v2v_continuation_allowed.md`
- Rebuts 2026-08-18 “comparable verify is T2V 128 MovieGen” as the
  *only* next experiment
- Restates 2026-08-15 “stay in continuation / I2V; T2V was not required”

User: we did not agree on text-to-video; for long horizon, is
video-to-video not okay?

Yes. T2V 128 MovieGen + VBench-Long is the most *copied* 30–60 s table
(Relax Forcing / SF++ / FreqForcing). It is not a task lock. The 15
August correction already said the model switch and the T2V default are
independent knobs. Our claim is exposure bias under **visual**
re-conditioning. T2V-from-scratch removes the prefix the verifier/gate
read.

What we stopped on 18 August is **I2V-from-a-still** scale-up, not
visual continuation. A still has no motion (bad reference; freeze). A
real video prefix is the DFoT / SEINE / LongCat continuation setting
and is a valid long-horizon task. CausVid (CVPR 2025) already does
streaming I2V/V2V on Wan 1.3B (their V2V *table* is DAVIS translation,
not prefix-continuation). StreamingT2V’s ablation of “condition on last
frames, AR forever” is exactly V2V-style continuation — and they say it
stagnates.

Honest: there is no Relax Forcing–sized continuation leaderboard. A 30 s
V2V-prefix table is a continuation paper, not a T2V MovieGen cell.

**Decision:** T2V is one optional comparison, not required. V2V
prefix-continuation (real video history → 30 s AR, VBench on the full
clip) is allowed and is the closer match to the claim. I2V-still
scale-up stays closed. No job until the user picks. No TTC.

---

## 2026-08-18 — T2V 128 MovieGen compare is submit-ready
**Tags:** methodology, wan, t2v, jobs
**Owner:** agent
**Refs:**
- `wan_experiment/scripts/run_t2v_chunked.py`
- `wan_experiment/sbatch/submit_t2v_bon128.sh`
- `datasets/moviegen_128.txt`
- `paper_tables/2026-08-18_wan_t2v_vbenchlong_128_spec.md`

User: run T2V 128 anyway, as a standard bench to compare gating to
other methods.

Implemented a **new** T2V chunked runner (independent_first_frame=False,
no still, 6×21 latents ≈ 31.3 s). Not a flag on `run_i2v_chunked.py`.
Prompts: first 128 MovieGen (Qwen-extended if present on the SF clone,
else official VideoBench vendored). Methods: do-nothing | always-BoN
k=4 | hybrid gated-BoN. 4 shards. No TTC. V2V continuation still
allowed. I2V-32 scale-up still closed.

Not launched from this machine. Operator: `git pull`, then
`SMOKE=1 bash wan_experiment/sbatch/submit_t2v_bon128.sh`, then the
full 128. VBench-Long scoring is a later job.

---

## 2026-08-18 — VBench++ every 5 s window (trend)
**Tags:** methodology, wan, vbench
**Owner:** agent
**Refs:**
- `wan_experiment/scripts/score_i2v_vbench.py` (`wSTART_END` / `windows`)
- `wan_experiment/scripts/analyze_i2v_vbench_trend.py`
- `wan_experiment/sbatch/submit_i2v_vbench_windows.sh`

User asked for VBench++ not just last5 but every 5 s of the 30 s
generation, to plot the trend.

Implemented six windows on the existing hybrid 32 mp4s: 0–5 … 25–30.
Last window includes the leftover frame (481 px). Official comparable
number stays the full clip. last5 remains a separate diagnostic (last
80 frames, not exactly w25_30). No TTC. Not launched from this
machine: `bash wan_experiment/sbatch/submit_i2v_vbench_windows.sh`
then `analyze_i2v_vbench_trend.py`. Plot the canvas after the job.

---

## 2026-08-19 — VBench++ first-16 / last-16 on 5 s vs 30 s
**Tags:** methodology, wan, vbench
**Owner:** agent
**Refs:**
- `wan_experiment/scripts/score_i2v_vbench.py` (`first1` / `last1`)
- `wan_experiment/scripts/analyze_i2v_vbench_horizon.py`
- `wan_experiment/sbatch/submit_i2v_vbench_notta16.sh`
- `paper_tables/2026-08-19_wan_i2v_notta16_vbench_headtail_spec.md`

User asked whether VBench++ dims can use the same first-16-frame
window as the handpicked drift table, at both 5 s and 30 s.

Yes. `first1` = frames `[1:17]` (skip cond frame 0), `last1` = last
16 frames — identical to `score_i2v_drift.py` `WIN=16`. Also score
`full` and `first5` so 5 s full can sit next to the 30 s opening at
the official ~5 s duration.

These 16-frame clips are **diagnostics**, not official VBench++.
VBench quality dims assume ~5 s. `dynamic_degree` / motion
smoothness will be noisy. 5 s and 30 s were separate generates;
first1 is not a shared prefix. No new videos. No TTC. Not launched
from this machine: `bash wan_experiment/sbatch/submit_i2v_vbench_notta16.sh`
then `analyze_i2v_vbench_horizon.py`.

---

## 2026-08-19 — Hybrid 32 VBench windows job 16009916 submitted
**Tags:** jobs, wan, vbench
**Owner:** agent
**Refs:** job 16009916; cluster pull `4116c3a` → `eb51141`

Operator moved the two untracked paper tables aside and
`--ff-only` succeeded. `submit_i2v_vbench_windows.sh` accepted:
job **16009916**, series `i2v_bon_32v_hybrid`, methods
notta / always_bon / gated_bon, clips `w0_5` … `w25_30`.

FETCH_HEAD was `eb51141`. The first16/last16 commit `f48318b` was
not on the cluster yet, so `submit_i2v_vbench_notta16.sh` was not
run. Next: pull to `f48318b` and submit the 16v job. Do not
cancel 16009916. No TTC. No I2V scale-up.

---

## 2026-08-19 — Briefing slide: why Self-Forcing DMD, not vanilla Wan
**Tags:** methodology, wan, briefing
**Owner:** agent
**Refs:**
- `paper_tables/2026-08-18_week_switch_briefing.md` (new Slide 2)
- canvases/week-switch-briefing.canvas.tsx

User said the DMD choice was not clear. Added an early slide:
Wan 1.3B is the teacher; Self-Forcing DMD is the causal few-step
student we run. Reasons: AR + KV cache, 5 s train / 10–30 s collapse
(headroom), cost + comparable table vs Relax / SF++. We still load
Wan + T5 + VAE; we do not sample bidirectional 50-step Wan.

Same turn: squeue shows 16009916 PD QOSMaxGRESPerUser and 16010032
PD Priority. Leave both queued.

---

## 2026-08-19 — VBench window + 16v head-tail jobs DONE
**Tags:** jobs, wan, vbench
**Owner:** agent
**Refs:** jobs 16009916, 16010032

Both COMPLETED 0:0, started together at 02:30:53 on courtesy nodes
(gh119 / gh132) — the 2-way cap did not serialize them.

- 16009916: hybrid 32, 6 windows, 18/18 `joined.json`, 1h04, ~11 GB RSS.
  Last window gated `w25_30` n=32 ran=7 fail=0. Dynamic median still 0.
- 16010032: 16v NOTTA 5 s + 30 s, clips full/first5/first1/last1,
  8/8 `joined.json`, 23m. Last clip `last1` on 30 s n=16 ran=7 fail=0.

No paper table yet. Analyzer stdout not pasted. Do not cite the two
log-tail populations. Full-clip hybrid 32 remains the official VBench++
number. 16-frame clips stay diagnostic. No TTC. No I2V scale-up.

---

## 2026-08-19 — Window trend + 16-frame VBench vs handpicked drift
**Tags:** finding, wan, vbench, methodology
**Owner:** agent
**Refs:**
- jobs 16009916, 16010032
- `paper_tables/2026-08-19_wan_i2v_bon32_vbench_trend.md`
- `paper_tables/2026-08-19_wan_i2v_notta16_vbench_headtail.md`
- `paper_tables/2026-08-19_wan_i2v_vbench_windows_read.md`

Analyzer stdout pasted. Cite medians.

Hybrid 32 windows (N=32): piece 0 ties. Do-nothing aesthetic
0.651→0.538 and IQ 72.9→68.1 from 0–5 to 25–30. Search does not
reverse the decay; tail IQ favors do-nothing (68.14 vs 66.41 / 66.07).
`dynamic_degree` median is 0 in **every** window — freeze is not an
onset after piece 0; most clips already fail RAFT in the first 5 s.
Subject stays high (0.934→0.969).

16v first16/last16 (N=16): VBench Δrel does **not** reproduce
handpicked sharp +167% / motion −60%. Only aesthetic says the 30 s
tail is worse (−11.5% vs +1.8% at 5 s). IQ is mild (−2.3%). Subject /
smoothness / flicker go up (frozen tails look consistent). 5 s full ≈
30 s first5 (subject −0.001). Do not equate 16-frame VBench with the
handpicked table. Official hybrid-32 number remains the full-clip tie.
No PSNR. No TTC. I2V-32 scale-up stays closed.

---

## 2026-08-19 — Why 5 s VBench windows barely move (except aes/IQ)
**Tags:** methodology, wan, vbench
**Owner:** agent
**Refs:** `score_i2v_vbench.py` (`ensure_clip_dir` + `VBench.evaluate`);
Huang et al., VBench, CVPR 2024

User asked how VBench is calculated; the 0–5 vs 25–30 drop looked too
small.

We do **not** score “frame 0 vs frame 400.” For each window we cut a
new mp4 (`vbench_clips/wSTART_END/`) and run official `custom_input`
VBench on that file **alone**. Each dim is therefore an intra-clip
statistic on ~80 frames.

That saturates consistency / smoothness / flicker on a freeze: DINO
and CLIP cosine of nearby frames go up, AMT interpolation is easy,
flicker is low. `dynamic_degree` is 0/1 RAFT per clip, so the median
stays 0 if most clips were already still at 0–5 s. Only
`aesthetic_quality` (LAION, per-frame average) and `imaging_quality`
(MUSIQ, per-frame average) can fall with appearance. Those drops are
not tiny: do-nothing aes 0.651→0.538 (−17% relative), IQ 72.87→68.14
(−6.5%).

Evidence the long-range effect is real but hidden by windowing: full
30 s subject is **0.848**, while every 5 s window is **0.93–0.97**.
The window never asks whether the last 5 s still matches the first 5 s.
Handpicked drift does (first 1 s vs last 1 s of the same file).
VBench-Long exists because short-clip VBench on slices misses that.

Do not read “subject +0.035 from 0–5 to 25–30” as quality improving.

---

## 2026-08-19 — Headline the entire-clip 5 s vs 30 s VBench++ table
**Tags:** methodology, wan, vbench
**Owner:** agent
**Refs:** `paper_tables/2026-08-19_wan_i2v_notta16_vbench_fullclip.md`;
job 16010032 `vbench_full`

User asked for the typical VBench++ number: the **entire** 5 s clip and
the **entire** 30 s clip. Those scores were already in job 16010032
(`clip=full`) and buried under “unequal length.” Pulled out as the
citeable table.

N=16 NOTTA, median: subject 0.932 → **0.842** (−0.090), background
−0.045, aes −0.043, IQ −1.06. Smoothness/flicker slightly up. Dynamic
median 0 both; mean 0.250 → 0.438 (longer clip, more chance of one
RAFT-dynamic burst). Hybrid-32 30 s do-nothing subject 0.848 matches
this 16v 30 s 0.842.

5 s full = VBench-I2V-length. 30 s full = score the whole generation
(our hybrid lock; field long-horizon uses VBench-Long). Windows remain
diagnostics. No new job. No PSNR. No TTC.

---

## 2026-08-19 — Why dynamic_degree is ~0 throughout
**Tags:** methodology, wan, vbench
**Owner:** agent
**Refs:** VBench `dynamic_degree.py`; job 16009916 / 16010032

User asked why dynamic degree is so low in every window and on both
full clips.

It is a **per-video 0/1**, then a population fraction. RAFT top-5%
flow must beat `6.0 * min(H,W)/256` = **11.25 px** on our 480p, on
enough sampled pairs (~8 fps). Small I2V motion fails. Median 0 =
majority static. Mean 0.250 at 5 s = 8/32 (or 4/16). Already true in
the first 5 s — not a 30 s-only freeze by this test. Handpicked
`|Δframe|` can drop 60% without flipping the bit. VBench paper:
consistency vs dynamic degree trade off. I2V-from-still on Wan 1.3B
SF is the low-dynamic side of that trade-off.

---

## 2026-08-19 — I2V VBench++: we already have captions; missing I2V dims
**Tags:** methodology, wan, vbench, i2v
**Owner:** agent
**Refs:**
- `paper_tables/2026-08-19_vbench_i2v_what_papers_do.md`
- Huang et al., VBench++ Table III; `vbench2_beta_i2v` README
- `run_i2v_continuation.py` `_load_prompt_map` / `text_prompts=[prompt]`
- `score_i2v_vbench.py` (drops `i2v_subject` / `i2v_background`)

User asked whether VBench++ needs a paired text prompt, and what I2V
papers do.

Generation: we already pair each VBench-I2V still with its `caption`
from `i2v-bench-info.json`. DynamiCrafter / ConsistI2V / VideoCrafter-I2V
/ Wan2.1-I2V-14B do the same (image + `prompt_en`; Wan often Qwen-extends).
SVD-XT is on that leaderboard image-only (camera column blank).

Scoring: official I2V tables add `i2v_subject`, `i2v_background`,
`camera_motion` (image↔video, not prompt↔video except camera). Quality
7 does not use text. We scored quality 7 only (`custom_input`). That
does not make the run “promptless generation.” It does mean we are
not on the VBench-I2V leaderboard protocol (also: they submit ~16
frames / 2 s or 81 / 5 s, N in the hundreds).

Dynamic degree on that official I2V table is also often low
(ConsistI2V 18.6%, VideoCrafter-I2V 22.6%, I2VGen-XL 25.0%;
DynamiCrafter 47.4%, Animate-Anything 2.7%). Adding the prompt at
`evaluate()` will not raise ours. Next if we want the standard I2V
columns: score `i2v_subject` / `i2v_background` on existing mp4s.
For a high-dynamic table: T2V MovieGen-128, not more I2V stills.
No TTC. No I2V-32 scale-up.

---

## 2026-08-20 — V2V sampling-space bake-off (beyond gating)
**Tags:** decision, methodology, wan, v2v, search, no-ttc
**Owner:** agent
**Refs:**
- `paper_tables/2026-08-20_wan_v2v_sampling_bakeoff_spec.md`
- `paper_tables/2026-08-18_wan_nonweight_next.md`
- `paper_tables/2026-08-18_v2v_continuation_allowed.md`

User asked to move on from gating and run the sampling-space ideas.
Host locked to **V2V prefix-continuation** (Panda 2 s real prefix → 30 s
AR). ROI rank: flow `shift` (if it moves pixels) and a one-sided motion
verifier first; prefix backtrack as a tail fix; CFG likely null on DMD;
sink / CachedSearch / HG-f deferred.

Wave 1 methods: notta, seed_bon (control), motion_bon, shift_search
(conditional on probe), backtrack. N=2 smoke then N=8 parallel. Promote
past N=8 only if tail motion beats notta without IQ/subject collapse.
No TTC. No I2V-32 scale-up. T2V 128 stays optional.

---

## 2026-08-20 — V2V probe: shift/CFG are no-ops; drop shift_search
**Tags:** finding, methodology, wan, v2v
**Owner:** agent
**Refs:** jobs 16069897 / 16069898;
`paper_tables/2026-08-20_wan_v2v_probe.md`

Smoke NOTTA n=2 completed in 7 min (tail motion 0.0106). Probe n=2:
all 9 `(shift, cfg)` cells have identical motion 0.01626.
`apply_shift` / `apply_guidance` do not move pixels on Self-Forcing DMD.
Last-chunk composite 6336 is prefix-vs-tail scale clash — not a quality
number. Backtrack now ignores drift > 100 and keys off motion collapse.

N=8 wave 1: notta, seed_bon, motion_bon, backtrack. `SKIP_SHIFT=1`.
No CFG. No TTC. No I2V-32 scale-up.

---

## 2026-08-20 — V2V N=8: seed_bon raises tail motion; motion_bon/backtrack lose
**Tags:** finding, wan, v2v, search
**Owner:** agent
**Refs:** jobs 16092846–849;
`paper_tables/2026-08-20_wan_v2v_bakeoff_8v.md`

Paired N=8, medians. seed_bon (k=4 seeds, old deviation pick) tail
motion 0.0225 vs notta 0.0167 (**+35%**). motion_bon 0.0148 (−11%).
backtrack 0.0130 (−22%). Wall: notta 18 m, search ~51 m, backtrack 23 m.

On a real Panda prefix, seed search is not the I2V-still freeze
attractor. Greedy per-chunk `|Δframe|` does not compose into a more
dynamic 30 s tail. Backtrack (motion-collapse only after the 6336
guard) did not help.

Analyzer printed PROMOTE for seed_bon on motion alone. Official rule
still needs full-clip VBench IQ/subject. Do not scale. No TTC.

---

## 2026-08-20 — V2V N=8 VBench: seed_bon passes the promote rule
**Tags:** finding, decision, wan, v2v, vbench
**Owner:** agent
**Refs:** `paper_tables/2026-08-20_wan_v2v_bakeoff_8v_vbench.md`;
`v2v_panda_bakeoff_8v/*/vbench_full/joined.json`

Full-clip VBench medians, N=8. seed_bon vs notta: tail motion +35%
(0.0167→0.0225), IQ 67.98→67.38 (**−0.60**, under the ≥1.0 fail bar),
subject 0.5951→0.5956. `dynamic_degree` median **0→0.5** (4/8 vs 0/8;
notta mean 0.375 = 3/8). Locked rule **PASS**.

motion_bon: no tail-motion gain; IQ actually up (+0.79); HOLD.
backtrack: motion down, IQ **−2.94**; drop.

On a real prefix, k=4 seed search is a live sampling-space intervention.
Greedy motion pick and backtrack are not. Next allowed confirm: notta vs
seed_bon only, N=32. N=8 Dyn is a coin-flip. No TTC. No I2V-32. No
shift_search.

---

## 2026-08-20 — Recommend N=32 notta vs seed_bon only
**Tags:** decision, methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-20_wan_v2v_what_happened.md`

User asked what the bake-off meant and what to try next.

Plain reading: seed diversity on a real prefix is the only live
actuator we found. The “smart” ideas (motion pick, backtrack, shift,
CFG) failed or were no-ops. The win used the old drift pick score — so
the finding is “four seeds can leave the freeze on V2V,” not “we have a
good verifier.”

Recommend one confirm: N=32, notta vs seed_bon, same V2V protocol. If
it holds, next method work is cheaper seed search or attention sink.
If it dies, N=8 was noise. Not submitted until the user says go.

---

## 2026-08-20 — User said go: N=32 notta vs seed_bon
**Tags:** decision, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-20_wan_v2v_confirm_32_spec.md`;
`sbatch/submit_v2v_bakeoff.sh` `CONFIRM=1`

User approved the confirm. Two jobs only. No motion_bon / backtrack /
shift / TTC / I2V-32. First 32 Panda clips (N=8 is the prefix). Cite
medians after full-clip VBench. If the win dies, N=8 was noise.

---

## 2026-08-20 — Sampling-space hypotheses after V2V N=8
**Tags:** methodology, wan, v2v, hypothesis
**Owner:** agent
**Refs:** `paper_tables/2026-08-20_wan_v2v_sampling_hypotheses.md`

User asked what else is worth trying, inferred from the bake-off.

Main inference (H-match): seed_bon’s old two-sided score is “stay like
the real (moving) prefix.” That is why the same score froze on I2V
(reference = first second after a still) and unfroze on V2V. motion_bon
maximized twitch and lost; dead-tail backtrack lost; shift/CFG are dead.

Next tricks, sampling space only: (1) prefix-match hinge on motion,
N=8 paired with existing bake-off; (2) late-only / failure-gated
seeds; (3) attention sink of the prefix; (4) history-dropout search
(HG without CFG); (5) backtrack to last *good* chunk; (6) CachedSearch
only after N=32 confirms. Do not resubmit |Δframe|-max or dead-tail
backtrack. No TTC.

---

## 2026-08-20 — N=32 confirm in flight; tricks implemented
**Tags:** decision, wan, v2v, in-flight
**Owner:** agent
**Refs:** jobs 16113805 / 16113806; `submit_v2v_bakeoff.sh` `TRICKS=1`;
`paper_tables/2026-08-20_wan_v2v_tricks_8v_spec.md`

User pasted a successful `CONFIRM=1` submit after pulling `7371ea3`.
`v2v_panda_confirm_32v` notta=16113805, seed_bon=16113806. Do not scancel.

Same turn: “let’s test everything.” Implemented the six sampling-space
probes on the existing V2V runner (no TTC, no |Δframe|-max, no
dead-tail backtrack, no shift/CFG):

1. `hinge_bon` — k=4, `prefix_match_score` (appearance two-sided, motion hinge)
2. `late_bon` — seed_bon only if incoming motion < 0.7× prefix or last 2 chunks
3. `hist_drop` — full history vs last-3 latents vs extra seeds; hinge pick
4. `good_backtrack` — resample only if this chunk collapsed and the previous commit was good
5. `cached_bon` — seed_bon pick, KV replayed once then snapshotted
6. `sink` — k=1, replay prefix + last 21 latents (no rerope checkpoint)

N=8, same first-8 Panda videos as the bake-off. Submit with `TRICKS=1`.
2-way cap: extras queue behind the N=32 pair. Analyze with
`--baseline-dir .../v2v_panda_bakeoff_8v`.

---

## 2026-08-20 — TRICKS=1 queued (16115844–849)
**Tags:** in-flight, wan, v2v
**Owner:** agent
**Refs:** `experiment_outputs/2026-08-20.md` 22:26; jobs 16115844–849

Cluster pulled `fb11147` and submitted all six tricks. Confirm
16113805/806 still Running on gh116/gh119. Tricks all Pending
`QOSGrpGRES` on `h200_cour`. Courtesy did not bypass the 2-way cap
this time. No cancel. Start when a confirm GPU frees.

---

## 2026-08-21 — N=32 seed_bon dies; hist_drop is the only live trick
**Tags:** finding, negative-result, wan, v2v, paper-narrative
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_confirm_32_and_tricks.md`;
`experiment_outputs/2026-08-21.md` 02:21

Generate complete: confirm 32/32, tricks 6×8/8.

**seed_bon N=32 tail motion 0.01018 vs notta 0.01380 (−26%).** The N=8
+35% / Dyn 0→0.5 promote is a lucky prefix of the sorted Panda list.
Do not write the paper around seed-BoN. Last-chunk drift also fell
(20.5→18.5) — two-sided match quieted the tail, same failure mode as
I2V-from-still.

Tricks on the same 8 as the bake-off:
- `cached_bon` = bake-off seed_bon to printed precision (KV snapshot works).
- `sink` = bake-off notta to printed precision (replay-approx sink is dead).
- `hinge_bon` +11% vs notta, **−18% vs seed_bon** — H-match hinge is not
  why the 8 won.
- `late_bon` −10%, `good_backtrack` −22% (drift 97).
- `hist_drop` 0.02377 (**+42% vs notta, +6% vs seed_bon**). Only new
  motion win. Still N=8. VBench IQ/subject before any scale-up.

Next: full-clip VBench on confirm `{notta,seed_bon}` (document the kill)
and tricks `hist_drop`. No TTC. No hist_drop N=32 until VBench +
per-video check.

---

## 2026-08-21 — Retract N=32 seed_bon kill; hist_drop is broad on N=8
**Tags:** finding, methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_pervideo_retract.md`

The −26% confirm median compared unpaired `summary.json` rows. ~20
clips per method are `skipped: true` after preempt/requeue and have no
`tail_motion`. Printed n=32 was ok-count.

Honest pair (0020–0031 only): notta 0.01380 → seed_bon 0.01424
(**+3%, 5/12**). Tie. Retract KILL and the “lucky prefix” story until
sidecars for 0000–0019 are read.

hist_drop on the bake-off 8: **6/8 vs notta, 7/8 vs seed_bon** (0000
exact tie with seed). Losses vs notta are the two hottest prefixes.
vs seed_bon every non-tie is +0.0008…+0.0043 — increment, not a new
mode. Not one-clip. VBench still required before any scale-up.

---

## 2026-08-21 — Sidecar N=32: seed_bon −8.8% (12/32); VBench queued
**Tags:** finding, negative-result, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_confirm_32_sidecar.md`;
jobs 16122823 / 16122824

Sidecars recover all 32. Paired medians: notta 0.01353, seed_bon
0.01235 (**−8.8%**), **12/32** wins. Locked promote rule FAILS.

First 8 bit-match the bake-off (+34%, med 0.01675→0.02250). Last 24:
−19%, 8/24. Hot prefixes (notta≥0.020): **0/7** — two-sided match
damps already-moving tails. N=8 was two large lifts (0000, 0007) on a
list that over-weighted mid clips, not a general actuator.

Unpaired summary −26% stays retracted (skip stubs). Qualitative
conclusion is the same: do not write the paper around seed-BoN.

VBench submitted: 16122823 confirm `{notta,seed_bon}` full;
16122824 tricks `{hist_drop,hinge_bon}` full. Both PD QOSGrpGRES.
hist_drop still N=8 only; it increments the picker that just failed
confirm. No hist_drop-32 until IQ/subject land.

---

## 2026-08-21 — Bedtime pair: quiet_bon N=32 + tail_hist N=8
**Tags:** decision, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_next_bedtime.md`;
`submit_v2v_bakeoff.sh` `NEXT=1`

User asked what to run tonight without waiting for VBench.

Not hist_drop-32: that method increments the picker that just went
12/32 and 0/7 on hot prefixes.

Instead:
1. `quiet_bon` N=32 — k=4 only if real prefix_motion < 0.018, else
   k=1. Causal test of “search damps a living prefix.”
2. `tail_hist` N=8 — always last-3 latents, no search. Isolates the
   history axis from seed search.

VBench 16122823/824 already queued. These sit behind GRES. No TTC.

---

## 2026-08-21 — N=32 VBench: Dyn 0/0, subject +0.039; stop seed-family
**Tags:** finding, negative-result, wan, v2v, paper-narrative
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_overnight_vbench.md`;
jobs 16122823/824, 16124386/387

Official full-clip VBench n=32: notta vs seed_bon Dyn median **0 and 0**.
The N=8 0→0.5 flip is dead. seed_bon subject **0.665→0.705 (+0.039)**,
IQ −0.77 (under the 1.0 bar). Combined with sidecar tail −8.8%:
prefix-match search at this scale is an **identity-preserving damper**.

hist_drop / hinge N=8 VBench both PASS IQ+subject (hist IQ −0.15, hinge
+0.42) and Dyn 0.50 — the same N=8 coin-flip. Do **not** scale to 32.

tail_hist ≈ notta (+0.8%). hist_drop’s +42% was search, not short
history. quiet_bon n=32 tail 0.01089 vs notta 0.01353 (−19%). Gate
works on some hots; method still loses.

Default next: **stop V2V generate.** Paper table is the N=32 VBench
confirm. Optional later: `hot_tail` (short history only on hot
prefixes). No TTC. No hist_drop-32.

---

## 2026-08-21 — Picker insight: motion is a vote, not a constraint
**Tags:** methodology, wan, v2v, paper-narrative
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_picker_insight.md`

The two-sided sum `|Δsharp|+|Δcolor|+|Δcontrast|+|Δmotion|+seam` is
identity control. Dynamics only rise when notta undershoots the prefix
(0000/0007/0018). When notta is already in-band, appearance outvotes
motion → 0/7 hot, subject +0.039, Dyn 0.

Hinge too loose (no ceiling). late_bon skipped chunk 0 (where the
recoveries lived). quiet_bon gated on prefix quiet, not chunk collapse.
hist_drop = seed + tail candidate. tail_hist = different mode (helps
0002, kills seed lifts).

Next *score*: motion as [0.85, 1.15]×prefix **constraint**, appearance
as objective; else argmin |motion−prefix|. Resimulate on existing N=8
candidate logs before any GPU. No hist_drop-32.

---

## 2026-08-21 — Band resim: constraint would freeze still-prefix clips
**Tags:** finding, methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_band_resim.md`

N=8 cand-log resim. prefix_motion is often 0.0008–0.006 while chunk
motion is 0.02. Band [0.85, 1.15]×prefix is empty; fallback picks the
stillest seed (0000/0002/0003). 0007 prefix=0.070 is the real collapse
(notta 0.012, seed recovered 0.026).

quiet_bon inverted the gate: it searched still prefixes (where matching
damps) and skipped live ones (where recovery lives). That is the −19%.

Correct policy: search iff prefix is *live* (`>= ~0.012`); else notta.
Never two-sided-match motion to a still reference. Do not generate the
band constraint without that gate. Optional next: `live_bon` N=8 only.

---

## 2026-08-21 — Self-Forcing lineage brainstorm
**Tags:** methodology, wan, v2v, literature
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_sf_lineage_brainstorm.md`

Family: CausVid → Self-Forcing (our student) → SF++ (long DMD on
self-rollouts, no sink) / Rolling Forcing (windowed denoise + sink+rerope)
/ LongLive (sink + short window + KV-recache + long tuning). History
Guidance is the CFG-on-history cousin; CachedSearch is cost.

Long-horizon gains in this line are **trained**. LongLive: sinks are
dead until collapse is trained away — matches our replay-sink no-op.

Inference-only bets for us: (1) LongLive/RF V2V notta N=8 (is the host
the problem?), (2) `live_bon` on current SF, (3) prefix KV-recache only
on LongLive’s kernel. Do not train SF++/GRPO. Do not HG-CFG on DMD.

---

## 2026-08-21 — Submit all remaining lineage tests at once
**Tags:** methodology, wan, v2v, submit
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_lineage_overnight.md`

User asked to implement and submit everything, not in order. Suite:
`live_bon` + `live_hist` on SF (immediate); CPU download of LongLive
v1.0 + Rolling Forcing; then `longlive_notta`, `longlive_sink`,
`longlive_prefix_sink` (sink_size=9), `longlive_live_bon`,
`rolling_notta`; VBench full chained `afterany`. Series
`v2v_panda_lineage_8v`. Compare to bake-off notta. No TTC. No
hist_drop-32. 2-way H200 cap queues extras.

Command: `bash wan_experiment/sbatch/submit_v2v_lineage.sh` after
`git pull --ff-only origin main`.

---

## 2026-08-21 — Nine TTA/TTC ideas scored for sampling-space V2V
**Tags:** methodology, wan, v2v, literature
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_sampling_ideas.md`

User proposals ranked for *our* lock (sampling-space, no weight TTA).
Keep: (1) pseudo-future validation as a **gate** on seeds/policies, not
θ — `live_bon` is the cheap proxy in flight; (5) prefix anchor with
**appearance ≠ motion** (naive prefix-match already froze Dyn at N=32);
(3) noise-calibrated U_t as a trigger **after** an ε-stat probe on
4-step DMD; (9) router as the system once 1 and 5 have N=8 numbers.
Rewrite or drop: (2)(7) parameter TTA; (4) rolling ρ on vanilla SF
(shift/CFG were dead; maybe RF host); (6) lookahead on 4-step DMD
(= seed BoN); (8) horizon-increasing δ (`late_bon` already lost).
Do not submit new GPUs until lineage 16140808–816 finishes.

---

## 2026-08-21 — Implement and submit ideas 1/5/3 now
**Tags:** methodology, wan, v2v, submit
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_ideas_submit.md`

User overrode the wait-on-lineage line. Implement sampling-space
keepers and queue them today. Do **not** scancel 16140808–816.

Landed methods on vanilla SF, same N=8 Panda prefix set, new series
`v2v_panda_ideas_8v`:
- `appear_bon` / `live_appear`: idea 5. Drop `|Δmotion|` from the
  pick; appearance+seam only. live_appear also uses prefix_motion
  ≥ 0.012.
- `pseudo_gate` / `pseudo_appear`: idea 1. Split prefix 9 → A=6 +
  B=3. Generate B from A with k seeds, MAE vs real B. Search the
  30 s tail iff some extra seed beats notta on B. Else notta.
- `noise_probe` / `noise_bon`: idea 3. Log first-step
  (noisy−denoised) residual U_t. noise_bon searches extra seeds
  iff cand0 `eps_mean_abs` ≥ 0.04 (overridable). Appear pick.

Not implemented: weight TTA (#2/#7), rolling ρ on SF (#4),
lookahead beam (#6), horizon-increasing δ (#8), full router (#9).

Submit: `bash wan_experiment/sbatch/submit_v2v_ideas.sh` after
`git pull --ff-only origin main`. 2-way cap queues behind lineage.

---

## 2026-08-21 — Lineage generate health: 808/809/811 COMPLETED
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `experiment_outputs/2026-08-21.md` (12:08 dump)

Jobs **16140808** live_bon 31m, **16140809** live_hist 28m,
**16140810** download 65s, **16140811** longlive_notta 18m: all
COMPLETED 0:0, n_ok=8, 8 mp4s + summary. LongLive + RF ckpts on
disk. 812–815 still PENDING. Ideas 125–131 still PENDING.

Clip 0007 (the live-prefix recovery case, bake-off notta 0.01210):
live_bon 0.02617 **bit-matches seed_bon**; live_hist 0.02856
**bit-matches hist_drop**; longlive_notta 0.01499 (host did not
unstick this collapse). That is the live gate firing on the one
clip it should search. Population verdict needs the other 7
(still prefixes should match notta). Do not promote from 0007
alone. No VBench yet.

---

## 2026-08-21 — live_bon/live_hist +37% at N=8; do not scale yet
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_lineage_partial.md`

Paired N=8 vs bake-off notta. live_bon and live_hist median tail
0.0229 vs 0.0167 (+37%). last-chunk drift **exact notta 63.5052**.
Analyzer printed PROMOTE because VBench IQ/subject are missing and
it treats that as a pass.

This +37% is the same size as seed_bon-8 (+35% → 0.0225), which
later failed at N=32. Reconstruction: swapping only 0000 and 0007
to their seed/hist values and leaving the other six as notta yields
median 0.02293. 0007 logs already bit-match. last-chunk identity
says the still-prefix majority was skipped.

Honest status: **conditional**. Confirm 0002/0003 bit-match notta
(not seed_bon). Wait for VBench. Do **not** submit live_bon-32.
longlive_notta 0.0150 HOLD — host is not the Dyn fix.

---

## 2026-08-21 — live gate is net-nonnegative at N=8; 0000 FN
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_lineage_pervideo.md`

Per-video tails. Search iff prefix ≥ 0.012 fired on 0001 (0.022),
0006 (0.019), 0007 (0.070). Those bit-match seed_bon (live_bon) or
hist_drop (live_hist). Skips bit-match notta, including **0002/0003**
where seed_bon damped.

live_bon vs notta: 3 wins, 5 exact ties, **0 losses**. That is the
first search policy with this sign pattern on these 8.

0000 prefix 0.00638 was skipped — missed seed’s +0.013 recovery.
The 12:24 reconstruction (0000+0007 searched) is wrong. Median
0.0229 is 0006+0007 recoveries plus 0002/0003 kept high.

Do not scale to 32. VBench still missing. Do not retune live_min
while the rest of the wave is queued. longlive_notta still HOLD
(0002/0004 much worse than SF notta).

---

## 2026-08-21 — Stop N=8 fishing; live_bon-32 is the yes/no
**Tags:** methodology, wan, v2v, submit
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_live32_yesno.md`

User asked for a definite yes/no on a test method ASAP. More N=8
(threshold, appear, pseudo) cannot give that — seed_bon-8 already
lied. The test method is **live_bon / live_min=0.012**. One N=32
job vs existing confirm notta. YES = paired tail > notta AND stills
not mass-damped AND VBench IQ/subject hold. NO otherwise. No retune
after the shot. Lineage/ideas stay in the queue; they are not this
decision.

---

## 2026-08-21 — live_bon N=32 is NO
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_live32_verdict.md`

16147007/008 COMPLETED 0:0, 32 mp4s, VBench done. Script printed
PROMOTE +6% on “N=32”; that N is confirm summary stubs. Honest
pair on disk from `pair_v2v_tails` is **0020–0031 only** (n=12).
On those 12: skips exact notta; every search exact seed_bon; 2
wins / 4 losses. Mean tail −6%. VBench IQ −0.13 / subject +0.009
pass; Dyn 0/0.

**NO.** Live prefix ≠ collapse. Hot-live clips (0022/0027/0028)
are the N=32 damper. The skip is real; the controller is not.
Do not retune live_min. Do not scale ideas from N=8.

Lineage/ideas N=8 all done: rolling_notta is the only host
residue (+29%, Dyn 0.5, IQ hold). Not tonight.

---

## 2026-08-21 — Forward rolling + appear; audit mixed LongLive
**Tags:** methodology, wan, v2v, submit
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_forward.md`

User asked to move remaining potentials and investigate mixed
wins. Do **not** revive live_bon / live_hist / pseudo / noise_bon.

GPU forward (N=32 vs confirm notta): `rolling_notta` (host passed
N=8 on motion+Dyn+IQ) and `appear_bon` (different picker, N=8 Dyn
0.5 + IQ hold). Series `v2v_panda_forward_32v`.

N=8 diagnose only (no GPU): `longlive_prefix_sink` (is +84%
flicker?) and `longlive_notta` (is +IQ a freeze?). Fix IQ/motion
only if that paste says the gain is content, not junk.

---

## 2026-08-21 — Mixed diagnose: close prefix_sink / LL-notta; flicker tautology
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-21_wan_v2v_mixed_diagnose.md`

`diagnose_v2v_mixed.py` printed ρ(d_tail, d_flicker)=−1.0 on
**every** method. d_flick ≈ −d_tail. VBench flickering is not an
independent junk sensor. Ignore that READ line.

Independent reads: prefix_sink IQ drop>1 on **5/8** → close, do
not fix IQ. longlive_notta kills Dyn on every clip that had it →
close as a Dyn fix. live_hist Dyn 0.5 is 0007 at IQ −7.7 → close.
rolling 5/3, one IQ drop → N=32 job **16179112** stays.
appear 4/8 bit-match seed, median Δ tail −0.001 → job
**16179113** is a kill test, not a belief.

---

## 2026-08-22 — Collapse+band resim (zero GPU)
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_collapse_band_resim.md`

The leftover picker is not another live_min. It is: keep cand0
unless this chunk undershoots the prefix, then band / nearest
prefix. Still prefixes never match a hold. Run on existing
seed_bon-8, seed_bon-32, live_bon-32 sidecars. Earn a GPU only if
hots 0022/0027/0028 stay cand0 and 0007-class recoveries still
fire. Not a 30 s tail.

---

## 2026-08-22 — Collapse+band does not earn a GPU
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_collapse_band_read.md`

Resim on seed_bon-8/32 and live_bon-32. damp=0 vs cand0. Stills
skip. 0007-class recovers. 0027/0028 ch0 keep cand0. **0022 ch0
picks cand1 at 0.061** (prefix 0.068, c0 0.036) — same as seed_bon.
That is a real collapse + in-band recover, but the locked earn
rule was stay-cand0. Honor it. No generate. No retcon.

---

## 2026-08-22 — rolling-32 YES on locked bars; appear-32 NO
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_forward32_verdict.md`

Honest pair N=32 (sidecars). notta median 0.0135.

`appear_bon`: median +3%, mean −2%, 15/17, 12/32 bit-match
seed_bon, subject +0.065. **NO.** Close the appearance-pick arm.

`rolling_notta`: median +31%, mean +13%, 21/11, IQ +0.79, subject
+0.037, Aes +0.024, Dyn 0/0, 0/32 seed-match. Stills 15/6 not
mass-damped. **YES on the bars we locked for live_bon-32.** Not a
Dyn method. Not our controller — the host. First N=32 tail win
on this V2V protocol. Losses 0004 and 0027 are real.

---

## 2026-08-22 — Expand the host, not a controller
**Tags:** decision, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_rolling128_spec.md`

One method passed N=32 locked bars: `rolling_notta` (Rolling Forcing
student). That is a host change, not a sampling-space gate. appear /
live / seed stay closed. Next gate is N=128 on the same sorted Panda
prefix (32 is a prefix of 128), fresh SF notta + rolling in
`v2v_panda_rolling_128v`. Kill if median holds but mean or win-rate
flips. Still-prefix win-rate must stay ≥ 0.5. Dyn 0/0 does not
decide. 200/1000 wait on 128.

---

## 2026-08-22 — Cover leftover ideas; no weight TTA
**Tags:** decision, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_leftovers_cover.md`

RF host unlocked idea 4 (sample-specific rolling ρ) and idea 6
(lookahead). Those go to N=8 vs lineage `rolling_notta`. Trust-region
(#7) is a picker constraint inside `rolling_look` plus an offline
resim. Hybrid router (#9) is offline only (oracle + prefix rule).
Idea 2 is not a distinct GPU on the 9-latent protocol. Idea 8 stays
dead (`late_bon`). Do not revive appear/live/seed. Do not scale
leftovers from N=8.

---

## 2026-08-22 — Always-rolling beats a live-only gate
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_coverage_audit_read.md`

Coverage audit on disk. rolling N=32 still +31% / still 15/6 /
bit=notta 0. N=8 search Spearmans (+0.8…+0.9) die at 32. Offline
router: always rolling +31% vs prefix rule (still→notta,
live→rolling) **+9%**. The rule throws away the still-prefix host
wins. Oracle +46% picks seed/appear — not a method. Trust resim hit
0 chunks (`cands` vs `candidates`). U_t field absent on
noise_probe sidecars. Jobs 16209126–133 submitted. Do not gate RF
on live prefixes.

---

## 2026-08-22 — leftover ρ fails IQ; 128 tail holds
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_leftovers8_verdict.md`

Leftovers vs lineage rolling_notta (not SF notta). ρ knob moves
pixels: adapt = hi on stills, native on mid, lo on 0007. `rho_hi`
IQ −3.8 / flicker 0.971. `rho_lo` / `adapt` IQ −1.7 / −1.4. Idea 4
**NO**. `rolling_look` +6% tail, IQ +1.3, 5/3, damps 0001/0007.
N=8 HOLD only. 128 generate done: median 0.0136→0.0177 (+30%),
same as N=32. VBench 16209128 still pending. Do not scale leftovers.

---

## 2026-08-22 — rolling-128 motion bars hold; quality pending
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_rolling128_tails.md`

Honest 128-way sidecar pair. notta 0.01355 / rolling 0.01772.
Median +31%, mean +23%, 88/40. Stills 40/21, live 48/19. First 32
reproduce the N=32 pair exactly. Last 96 mean Δ is larger (+26%).
Worst losses still include 0004 and 0027. VBench 16209128 not in.
Do not cite YES. No leftover scale-up.

---

## 2026-08-22 — four cheap host hypotheses (H1–H4)
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_host_split_spec.md`

RF is not our method. Four cheap tests (no LoRA, no backprop).
H1: crossed pair `sf_roll` (SF θ + rolling window) vs `rf_chunk`
(RF θ + SF chunks). Host is now the ckpt, not the method prefix.
H2/H3: offline resim on existing 8/32/128 mp4s — argmax / veto on
generated chunk-0 motion. Not the prefix-motion gate (already lost).
H4: VAE re-encode last 9 latents (`sf_recache` / `rf_recache`),
unlike sink/tail_hist which only shortened attention. GPU N=8 only
for H1+H4. Do not scale. Do not call RF a controller.

---

## 2026-08-22 — host-split first run is N=32, not 8
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_host_split_spec.md`

User: N=8 is crazy small. Agreed — seed / live / look all printed
wins at 8 and died at 32. GPU series renamed
`v2v_panda_host_split_32v` before any job launched. Same first 32
as confirm/forward. Walls 8 h / 12 h. H2/H3 stay offline on the
existing 8/32/128 mp4s (already the right N). Do not launch an
8-video host-split.

---

## 2026-08-22 — H2 bake NO at 128; H3 veto median-ties RF
**Tags:** finding, wan, v2v, negative-result
**Owner:** agent
**Refs:** `paper_tables/2026-08-22_wan_v2v_host_switch_verdict.md`

Offline chunk-0 switch on existing mp4s. Cite N=128. H2 argmax
over-picks SF (73/128) and loses the median (−5.4% vs always-RF;
81/47 vs RF’s 88/40). H3 veto median ties RF; mean +3.4% / 92/36
is collapse salvage (0004/0027/0035/0044/0087) plus stolen RF
wins. N=8 +5% and N=32 +3.5/+4.7% are the lucky-N trap. ρ(Δc0,
Δtail) 0.62→0.51. Do not GPU a host router. Do not retune 0.8.
H1/H4 jobs 16215197–201 stay. Always-RF remains the host.

---

## 2026-08-23 — sf_roll sampler is live; rf_chunk failed; 128 VBench incomplete
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_v2v_host_split32_read.md`

H1: SF weights + RF window (`sf_roll`) N=32 tail 0.0281 vs notta
0.0135 (+108%, 28/4) and vs rolling 0.0178 (+58%, 27/5). IQ/subject
hold vs SF. Subject **fails vs RF host** (0.666 vs 0.702). Dyn
median 1.0. Sampler is not a no-op. Do not scale; twitch vs living
is open; `rf_chunk` FAILED 0 mp4 (H1 cross incomplete). H4:
`sf_recache` NO (−1%). `rf_recache` +6.6% vs host on 30/2, Dyn 0 —
VAE grain HOLD. 128 VBench: notta in (subj 0.648 IQ 70.20); rolling
cancelled at subject 91/128. Do not cite 128 YES.

---

## 2026-08-23 — rf_chunk died on kv_cache1; alias RF KV
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `wan_experiment/scripts/run_i2v_chunked.py`

16215198: RF `CausalInferencePipeline` has no `kv_cache1`.
`_reset_caches` did `if pipeline.kv_cache1 is None` and nn.Module
`__getattr__` raised. RF cache is `kv_cache_clean`. Fix aliases
both after first `_initialize_kv_cache`. Resubmit rf_chunk only
(`submit_v2v_rf_chunk.sh`). 128 rolling VBench resubmitted as
**16228045**. Do not scale sf_roll while the cross is incomplete.

---

## 2026-08-23 — H1 complete: mismatch twitch; 128 VBench preempts at 2h
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_v2v_host_split32_h1_read.md`

rf_chunk retry **16228103** 32/32; VBench **16228104** 7/7.
H1 crossed pair at N=32: `sf_roll` and `rf_chunk` both median tail
0.0281 vs SF 0.0135 (+108%) and vs RF host 0.0178 (+58%). W/L
28/4 and 30/2 vs SF; 27/5 and 29/3 vs RF. 0 exact bit-matches.
Both Dyn median 1.0, flicker 0.972. Subject fails vs rolling
(0.666 / 0.676 vs 0.702). Sampler is live; ckpt is not the only
knob; mismatching them is twitch, not a quality method. Matched
native RF remains the object. Do not scale either cross.
H4 unchanged (`sf_recache` NO, `rf_recache` HOLD). 128 VBench
**16228045** CANCELLED+ at 2h07 on flickering (same as 09128).
Resume skip-existing; join six written dims on login. Do not cite
128 YES.

---

## 2026-08-23 — rolling-128 locked bars pass on 6/7 VBench
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_v2v_rolling128_vbench6_read.md`

Login join after 16228045: subject/bg/aes/IQ/motion/dynamic all
n=128. Flickering still missing; GPU **16259396** submitted
(skip-existing). Medians vs SF notta: subject 0.687 vs 0.648
(+0.039), IQ 70.91 vs 70.20 (+0.71), aes 0.540 vs 0.507, Dyn
median 1.0 (mean 0.531) vs notta 0. Tails already +31% 88/40.
Locked promote rule PASSES. Official 7-dim waits on flickering.
N=32 rolling Dyn 0 does not automatically carry — first-32 vs
last-96 Dyn split still open. Still someone else's host. Do not
scale sf_roll / rf_chunk / leftovers. Do not resubmit 16259396.

---

## 2026-08-23 — rolling-128 Dyn median 1 is the last-96 slice
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_v2v_rolling128_dyn_split.md`

joined.json: 68/128 VBench-dynamic, mean 0.531, median 1.0.
first32: 14/32, mean 0.438, median 0.0 — matches the N=32
forward “Dyn 0” cite (14 < 16). last96: 54/96, mean 0.562,
median 1.0. N=32 was not “RF never dynamic”; the larger pool
crosses 50%. SF notta 128 Dyn median stays 0. Flickering job
16259396 PD QOSMaxGRESPerUser. Do not resubmit.

---

## 2026-08-23 — Family A rewind: offline first; SF stays the paper baseline
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_v2v_rf_sick_rewind_spec.md`

Do not drop Self-Forcing as the field baseline (Relax / Deep / Freq
ablate vs SF). RF native is a required comparison and the ablation
zero only if the method is a controller on RF. Next step is login
`resim_v2v_rf_chunk_trace.py` on the existing 128 pair: rewind has a
GPU job only if RF-losses vs SF are enriched for a late motion drop
(c5 < 0.8×c0). DROP is pre-registered. No retune. No look/recache/ρ.

---

## 2026-08-23 — family wave: test A/B/C/D at once
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_v2v_family_wave_spec.md`

User asked to test all four families together, not sequential GO.
Login: chunk-trace --only all (8/32/128). GPU N=32:
rf_rewind, rf_sick_search k=4, rf_pseudo k=4, rf_sink.
C is an RF sink pixel probe — not HG-f (not in repo).
Paper baseline stays SF. Ablation zero is rolling_notta.
DROP=0.8 pre-registered. 2-way H200 will queue.

---

## 2026-08-23 — H200 pack-2; VBench off H200
**Tags:** methodology, wan, cluster
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_gpu_batch_policy.md`

User asked again to fill the H200 or request a lesser GPU.
Candidate tensor-batch is off: 137-frame KV ≈ 39 GB, k=4 copies
miss 141 GB, and KV end-index is scalar. Do not shrink the cache
(H1 twitch). Generate stays H200 with `VIDEO_WORKERS=2` (two
independent videos, same pixels, MPS on). L40S is 48 GB — too
small for that KV. VBench moves to `--constraint=l40s` so it
stops occupying the 2-way H200 cap. Leave 16259396. If pack-2
OOMs, `VIDEO_WORKERS=1` on that method only.

---

## 2026-08-23 — rolling-128 official VBench 7/7
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_v2v_rolling128_vbench7_read.md`

Join-only after 16259396: both methods n=128 on all 7 dims.
Cite medians. rolling vs SF: subject +0.039, IQ +0.71, aes
+0.033, Dyn 0→1, flicker 0.986→0.982 (−0.0036), smoothness
−0.0009. Locked bars PASS. Flicker cost is not H1 twitch
(0.972). Still the RF host, not our controller. Family wave
is the next method test.

---

## 2026-08-23 — family pack-2 OOM; resume workers=1
**Tags:** methodology, wan, cluster
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_gpu_pack2_oom.md`

16261273–276 FAILED. mp4 15/16/16/5. Pattern = one of two
video-workers died (2×39 GB KV). 277 VBench afterany started
on incomplete dirs — scancel. Default VIDEO_WORKERS=1. VBench
dependency is afterok. Skip-existing resumes remaining videos.
Do not delete mp4s. Do not retry pack-2.

---

## 2026-08-23 — pack-2 OOM smoking gun: 127 GiB + 13 GiB
**Tags:** finding, wan, cluster
**Owner:** agent
**Refs:** `paper_tables/2026-08-23_wan_gpu_pack2_oom.md`

16261275.err: process 1841066 held 126.97 GiB; the sibling had
12.74 GiB and OOM'd on `module.to`. A single V2V generate
already fills the H200 (KV + activations + cache), so pack-2
is impossible and L40S 48 GB cannot host generate. H200 stays
for generate; VBench stays L40S. Do not retry pack-2.

---

## 2026-08-24 — family N=32: beat SF, not a scale-up vs RF
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_family32_verdict.md`

32/32 + official 7-dim. Analyzer PROMOTE vs SF is the RF host.
vs rolling_notta: rewind +7.7% HOLD (0027 recover, Dyn 0→1),
sick +6.5% HOLD, pseudo +1.3% / 18 exact **NO**, sink +24%
HOLD no-scale (subj −0.016, flicker 0.977, not HG-f). Locked
letter vs RF passes for all four; still do not scale. No TTC.
No I2V.

---

## 2026-08-24 — SF-hosted family: cite SF, implement on SF
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_sf_family_spec.md`

User: if SF is the paper baseline, the method must run on SF.
RF-hosted +X vs SF is the host. Next wave: sf_rewind /
sf_sick_search / sf_pseudo / sf_sink on native SF chunked.
Promote vs SF notta. RF rolling is a comparison row. No
sf_roll (H1). VIDEO_WORKERS=1. Do not scale RF family.

---

## 2026-08-24 — sf_rewind 16266878: score after increment
**Tags:** bug, wan, v2v
**Owner:** agent
**Refs:** job 16266878, `submit_v2v_sf_rewind_resume.sh`

Exit 2, n_ok=8/32. Accepted rewind scored `_score_cand` after
`committed += chunk_latents`, so `gen_only` was empty (numpy
empty-slice + IndexError on seam). The 8 survivors never
accepted. VBench 882 afterok cancelled. Fixed call order.
Leave sick/pseudo/sink (879–881). Resume rewind skip-existing.

---

## 2026-08-24 — SF-family dissection locked before harvest
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_sf_family_dissect.md`

Every arm gets a cell (A dead gate … H always-on harm) and a
next action. Coverage / conditional / quality / named wounds
before promote. A win is HOLD + invention sentence, not 128
tonight. A miss is a sensor or tax, not a dead end. No DROP
retune. No TTC.

---

## 2026-08-24 — SF-family N=32: first method-on-SF wins
**Tags:** finding, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_sf_family32_verdict.md`

Pseudo +37% vs SF (25/2/5), Dyn 0.5, IQ/subject hold, tail
0.0186 vs RF 0.0178. Fire 27/32 — loose gate. Not seed_bon-32
(drift pick, tail −9%). Rewind +6% with 12 later-freezes.
Sink +72%, subject −0.0195, flicker 0.977. Sick median −1%
despite 20/5/7. Do not scale. Next is always-motion-k=4 to
split gate vs pick.

---

## 2026-08-24 — GO always-motion-k=4 ablation
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_sf_always_search_spec.md`

User GO. `sf_always_search`: every chunk k=4, same motion+trust
pick as `sf_pseudo`, no prefix hold-out. Series
`v2v_panda_sf_always_32v`. Cite vs SF notta and vs pseudo.
Do not scale. No TTC.

---

## 2026-08-24 — Same-wave ablations + RF always + k=4
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_always_search_k.md`

User: launch obvious ablations in the same paste, not after
harvest. Gated ⇒ always-on + other-host twin. CachedSearch on
Wan 1.3B: BoN-4 budget, BoN-8 headline. This split stays k=4.
RF always was missing; added `rf_always_search`. k=8 is a later
width sweep.

---

## 2026-08-24 — Methods-since-switch talk locked
**Tags:** methodology, wan, briefing
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_methods_since_switch.md`

Anyone-readable walk-through of I2V discovery, failed prefix-match
search, RF host, SF-family widgets, gates, and locked metrics.
Always-search 16288113–115 not cited. RF 114 left squeue — sacct.

---

## 2026-08-24 — V2V text prompt was “panda NNNN”
**Tags:** bug, methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_panda_stem_prompt.md`

confirm_32v sidecars: `prompt_source=stem`, prompt `panda 0013`
etc. No captions.json in panda_1000_480p. Filename fallback
conditioned T5 on the dataset index. Pandas in the tail are
text takeover. Same-prompt method deltas still valid. Refuse
stem prompts for panda_*. Do not cite as caption-conditioned
V2V. Re-run only after real captions + GO.

---

## 2026-08-24 — V2V missed metadata.csv, not missing captions
**Tags:** bug, methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_panda_stem_prompt.md`

`panda_1000_480p/metadata.csv` is 357525 bytes (Apr 1). Videos are
under `videos/`. Runner only loaded JSON so sidecars stayed
`prompt_source=stem`. Hunt script stopped at empty `panda_100`
header CSV — ignore that 0-match. Next: confirm the four freeze
stems have real scene text, then GO a caption-conditioned re-run.
Always-search still the same-prompt ablation; do not scancel.

---

## 2026-08-24 — Panda V2V captions confirmed (first segment)
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_rerun_spec.md`

`metadata.csv` 1000/1000 list captions, 0 empty. Freeze stems are
kitchen / tomatoes / bathroom moisture / flashlight — not pandas.
First-segment resolve matches LongCat TTA. Caption-conditioned N=32
of notta + rolling + sf_pseudo + both always-search is WAITING GO.
Do not mix stem-prompt numbers into that table. No TTC. No I2V.

---

## 2026-08-24 — Caption replay GO (all prior V2V, waved)
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_rerun_spec.md`

User: rerun every prior V2V generate with real captions; stem
text likely hurt even the baseline. Hypothesis accepted: T5
“panda NNNN” fights kitchen/bathroom/flashlight prefixes.
WAVE=1 now (12 N=32 generate + VBench). WAVE=2 closed 32,
WAVE=3 leftover 8, WAVE=4 hosts 128 after harvest. New series
names. Stem always-search stays. No TTC. No I2V.

---

## 2026-08-24 — Caption WAVE=1 queued 16310318–330
**Tags:** jobs, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_rerun_spec.md`

Preflight passed (truck / kitchen / bookshelf / game cop).
12 generate + VBench afterok. Stem always-search 16288113–115
absent from this squeue — harvest later, keep as stem audit.
No WAVE=2 until a running sidecar shows metadata_csv.

---

## 2026-08-24 — Hold WAVE=2 (QOS cap + no sidecar)
**Tags:** jobs, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_rerun_spec.md`

18:44: 318/319 R 11m; 320–329 QOSMaxGRESPerUser. Do not submit
WAVE=2. First check notta sidecar prompt_source.

---

## 2026-08-24 — V2V gray flash is latent 21 (81-frame horizon)
**Tags:** bug, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_gray_flash.md`

Freeze-demo mp4s flash gray at 5.06–5.25 s on every clip. Pixel
MAE/sat drop on frames 81–84 = first VAE group past the native
21-latent / 81-frame / 5 s unit. Not prefix seam, not chunk seam,
not x264. In the file, not only the player.

---

## 2026-08-24 — AdaSteer confirmation on Wan SF V2V
**Tags:** methodology, wan, v2v, adasteer
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_adasteer_spec.md`

Port paper AdaSteer (t′=t+δ) onto CausalWan time_embedding.
Confirmation N=8: ada_fixed / ada_stream / ada_resid. Other LongCat
updates (retrieval, ES, Delta-B/C, LoRA, TTC) stay parked. Cite vs
caption notta. Do not scale a null. Queues behind caption WAVE=1.

---

## 2026-08-24 — AdaSteer 8v queued 16314667–670
**Tags:** jobs, wan, v2v, adasteer
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_adasteer_spec.md`

ada_fixed/stream/resid PD. Caption notta 318 still R 47m.
Rolling 319 left squeue. Do not harvest until metadata_csv + |δ|.

---

## 2026-08-24 — Caption WAVE=1 protocol PASS
**Tags:** jobs, wan, v2v, captions
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_wave1_hosts.md`

notta / rolling / rewind 32/32 COMPLETED 0:0. Every sidecar
`prompt_source=metadata_csv`. First prompt is the truck hood, not
`panda 0000`. Caption SF notta tail median 0.01164 (stem was
0.0135). Rolling 0.01423 (+22%). Rewind 0.01262 (+8%). Handcrafted
only — no VBench, no HOLD/NO. Stem tables stay audit. Sick-search
321 running. No WAVE=2. AdaSteer still cites this caption notta.

---

## 2026-08-24 — Caption host pair W/L
**Tags:** results, wan, v2v, captions
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_wave1_hosts.md`

Paired vs caption SF: rolling 23/9/0 Δmed +0.00273; rewind
23/5/4 Δmed +0.00075. Rolling win set wider than stem 21/11 even
though the median gap shrank (+22% vs +31%). Rewind still small
plus, fewer ties than stem 19/5/8. No VBench. No method call.

---

## 2026-08-24 — Caption search arms smoke (sick 6 / pseudo 3)
**Tags:** jobs, wan, v2v, captions
**Owner:** agent
**Refs:** experiment_outputs/2026-08-24.md 20:46

Both search jobs write `metadata_csv`. Sick gate skip-then-fire
is visible (0000: skip chunks 0–1, `sick_motion` pick 3/4). Pseudo
fires from the prefix (`pseudo_fire=True`, 6/6 chunks searched).
Not a harvest. Do not pair vs SF at n=6/3.

---

## 2026-08-24 — Caption outcomes table opened
**Tags:** results, wan, v2v, captions
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_wave1_outcomes.md`

Started filling caption WAVE=1 from finished generate. Hosts+rewind
N=32 tails are in. VBench columns blank until 16310330. Sick 321
left the queue — next harvest. AdaSteer 667–670 missing from 22:41
squeue; do not call. No HOLD/NO from tails. No WAVE=2.

---

## 2026-08-24 — Caption WAVE=1 generate harvest + AdaSteer crash
**Tags:** results, wan, v2v, captions, adasteer
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_wave1_outcomes.md`

Caption SF notta 0.01164. Sick **+0%** 19/4/9 (tie). Rewind +8%.
Sink +64% 31/1. RF host +22%. RF sick −1% vs host. RF always +25%
vs rolling. Pseudo 27/32 +28% — incomplete. AdaSteer 667–669
FAILED 2:0 in 3m, 0 mp4, metadata_csv loaded. Exit 2 = n_ok≠n.
Crash, not a Wan null. VBench still pending. No HOLD/NO.

---

## 2026-08-24 — AdaSteer crash was inference_mode
**Tags:** bug, wan, adasteer
**Owner:** agent
**Refs:** `wan_experiment/scripts/wan_adasteer.py`

16314667–669: `element 0 of tensors does not require grad`. Fit sat
inside `run_v2v_chunked`’s `torch.inference_mode()`. optimize() now
opens `inference_mode(False)` + `enable_grad`, clones the prefix, and
errors if the loss still has no grad_fn. Resubmit N=8 after pull.

---

## 2026-08-24 — AdaSteer N=8 resubmitted 16321558–563
**Tags:** jobs, wan, v2v, adasteer
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_adasteer_spec.md`

Cluster pulled `385f21a`. New jobs PD behind caption pseudo /
always. First R must print `|δ|=`. No N=32. No WAVE=2.

---

## 2026-08-25 — WAVE=1 down to always-search; AdaSteer left queue
**Tags:** jobs, wan, v2v, captions
**Owner:** agent
**Refs:** experiment_outputs/2026-08-25.md

squeue is only **16310324** (SF always R 2h51) and VBench 330.
Pseudo 322 and AdaSteer 558–563 are gone. Harvest 322. sacct
AdaSteer before calling a crash or a result. No WAVE=2.

---

## 2026-08-25 — Caption pseudo 32/32; AdaSteer IM-cache crash
**Tags:** results, bug, wan, v2v, adasteer
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_wave1_outcomes.md`

Pseudo COMPLETED: tail 0.01492 **+28%** vs caption SF, 23/0/9.
Always 29/32 still R. AdaSteer 558/560/562 FAILED: inplace update
to inference tensor (KV cache allocated under IM). Runner now skips
IM for AdaSteer methods and drops caches before fit. Resubmit N=8
after pull. No HOLD. No N=32.

---

## 2026-08-25 — AdaSteer N=8 resubmitted 16326033–036
**Tags:** jobs, wan, v2v, adasteer
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_adasteer_spec.md`

Cluster pulled `a1d294c`. New jobs 033/034/035 + VBench 036.
First R must print `|δ|=`. No N=32. No WAVE=2.

---

## 2026-08-25 — Caption generate off queue; VBench 330 scoring
**Tags:** jobs, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-24_wan_v2v_caption_wave1_outcomes.md`

squeue 02:23: only **16310330** R 1h10. Always **16310324** and
AdaSteer **16326033–035** left after Always 3h08 / Ada ~8–10 min.
036 gone. Do not fill Always-32 or AdaSteer from squeue. Harvest
then sacct. Do not swap stem VBench into caption tables. No WAVE=2.

---

## 2026-08-25 — Caption Always +39%; AdaSteer Wan NO; RF host flips
**Tags:** results, wan, v2v, caption, adasteer
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_always_adasteer.md`

Always 32/32 tail 0.01623 **+39%** 30/2 vs caption SF. Pseudo
+28% 23/0/9. Gate is not inert; no Always call until 330.
Provisional caption VBench: SF 0.700/71.54/0/0.989. RF
0.694/70.22 (−1.32 IQ, fails −1 bar). Stem “RF is the better
host on quality” does not copy under real captions. AdaSteer
8/8 `|δ|`≈0.84: IQ 42.7 / 51.5 / 17.8. **NO.** Do not scale.
This is a Wan measurement, not LongCat leakage.

---

## 2026-08-25 — Caption Prefix-match N=32 only (not a full relaunch)
**Tags:** jobs, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_prefix32_spec.md`

Slide VBench is stem except AdaSteer / Always tails / 3 provisional
caption host rows. WAVE=1 already has caption N=32 for SF/RF/family/
always. Do not resubmit those. Do not dump WAVE=2 (`sf_roll`,
quiet, host-split). Do not scale AdaSteer. Submit only
`seed_bon` / `live_bon` / `appear_bon` as `v2v_panda_caption_prefix_32v`.

---

## 2026-08-25 — Prefix wave queued 16328464–467
**Tags:** jobs, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_prefix32_spec.md`

Submitted. Preflight truck/kitchen/bookshelf/cop, 0 stem.
PD (None) not QOS. 330 left running. Bedtime: nothing else to
submit. Morning: sidecar `metadata_csv`, then harvest vs caption
notta. Do not scancel 16288113–115.

---

## 2026-08-25 — Slide caption audit; only crossed-host missing
**Tags:** jobs, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_slide_caption_audit.md`

Every slide controller is caption-submitted or has caption
metrics except Crossed host. AdaSteer has caption N=8. Prefix
16328464–467 queued. Family VBench 330 R. Submit `WAVE=cross`
only if they want that row overnight; it shares the 2-H200 cap
with prefix. Do not scale AdaSteer. Do not dump WAVE=2.

---

## 2026-08-25 — Caption queue empty; harvest before slide swap
**Tags:** jobs, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_slide_caption_audit.md`

Cross **16328612–614** submitted (preflight metadata_csv). Later
squeue empty. Sacct + harvest prefix/cross/WAVE=1 VBench before
replacing stem method cards. Do not invent dims from an empty
queue.

---

## 2026-08-25 — Caption Pseudo Dyn 0; 330 cancelled; leftover VBench
**Tags:** results, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_vbench_read.md`

16310330 CANCELLED after writing SF family n=32. Caption Pseudo
**Dyn 0** (stem was 0.50) with tail +28% and IQ/subject hold.
Sink subject 0.672 fails −0.02. Prefix seed −18%. Cross NO
(rf_chunk flick 0.975). Resume appear + leftover VBench only.
Do not cite stem Dyn 0.50 as caption.

---

## 2026-08-25 — Leftover jobs 16350479–481
**Tags:** jobs, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_vbench_read.md`

Submitted. 479 = Always + RF VBench (no dep). 480 = appear
resume. 481 = Prefix VBench afterok 480. This squeue paste
omitted 479 — confirm it exists. Cancel leftover only:
16350479 16350480 16350481.

---

## 2026-08-25 — Caption official almost complete; Always + Prefix in
**Tags:** results, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_official.md`

479 CANCELLED 2h17 after Always + RF (no rf_sink). Always
0.687/71.16/0, tail +39%, letter holds. Prefix seed subject
0.746 / tail −18% — identity damper under captions. live/appear
Dyn 0. Only `rf_sink` official dims missing. Do not cite stem
Pseudo Dyn 0.50.

---

## 2026-08-25 — WAVE=rfsink job 16358585
**Tags:** jobs, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_official.md`

Submitted after pull to `32bf3a6`. Preflight metadata_csv,
bad=0. One L40S VBench on caption `rf_sink` only. Second squeue
empty. Harvest before writing official dims. Do not fill from
stem flicker 0.977.

---

## 2026-08-25 — Caption official complete; rf_sink VBench in
**Tags:** results, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_official_complete.md`

16358585 COMPLETED 0:0 11m. rf_sink n=32 metadata_csv:
subject 0.709, IQ 70.15, Dyn 0, flicker 0.980. Tail +73% vs
SF (29/3), +42% vs RF. Subject holds vs SF; IQ −1.39 fails.
Opposite of SF sink (subject 0.672). Caption official N=32
complete. Do not cite stem flicker 0.977. No WAVE=2.

---

## 2026-08-25 — Dyn is percent of clips, not median
**Tags:** methodology, results, wan, vbench, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_dyn_percent.md`

VBench authors: official Dynamic Degree = mean of per-clip 0/1
= percent dynamic. We had been citing median (all 0). Caption
N=32 means: SF 7/32 (21.9%), Always 14/32 (43.8%), Pseudo 13/32
(40.6%), RF sink 5/32 (15.6%), crossed rf_chunk 24/32 (75.0%).
Do not cite caption Pseudo as Dyn 0. Subject/IQ/flicker stay
medians.

---

## 2026-08-25 — Caption wall time; Always is 3.1× SF
**Tags:** results, wan, v2v, caption
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_caption_wall_time.md`

Sidecar seconds, n=32. Cite medians. SF 113 s, Always 348 s
(flat), Pseudo 357 s with ~9 skips at 113 s (the exact-SF ties).
notta mean 196 s is 0002/0019 outliers. RF family 63–82 s.
Examples copied: panda_0000–0004, 0006 × {notta, always, pseudo}.

---

## 2026-08-27 — Intra N=8: SF crash, RF twins NO
**Tags:** results, negative-result, wan, v2v, intra-chunk
**Owner:** agent
**Refs:** `paper_tables/2026-08-27_wan_v2v_intra8_harvest.md`

Jobs 16371523–536. SF `sf_intra` / `sf_intra_always` FAILED exit
2:0 in ~3 min (smoke + N=8): 8 error json, 0 mp4. RF 8/8
`metadata_csv`. `rf_intra` ≡ `rf_intra_always` on every official
dim (subj 0.645, IQ 66.33, Dyn median 1, flick 0.983, tail 0.0169)
vs caption SF 0.700 / 71.54 / 0 / 0.989 / 0.0129. **NO.** Do not
retune 1.5×. Do not scale. VBench jobs failed (empty SF) but RF
joined.json exists. Need SF sidecar `error`/`traceback` before
resubmit.

---

## 2026-08-27 — SF intra crash: RF _restore_kv shadowed pipeline
**Tags:** bug, wan, v2v, intra-chunk
**Owner:** agent
**Refs:** sidecar 16371530; `run_v2v_chunked.py` `_restore_rf_kv`

`TypeError: CausalInferencePipeline is not iterable` at
`_restore_kv(pipeline, snap)` inside `_fill_sf_intra_chunk`. A later
RF helper reused the name and expected a KV list. Renamed to
`_restore_rf_kv`. `cached_bon` had the same trap. Resubmit
`WAVE=sf` only. RF harvest stays NO.

---

## 2026-08-27 — WAVE=sf intra resubmitted 16471672–677
**Tags:** in-flight, wan, v2v, intra-chunk
**Owner:** agent
**Refs:** user paste squeue

Smoke 672/673 + N=8 675/676 PD on H200 `QOSMaxGRESPerUser`.
VBench 674/677 wait. Pull was `183dfaf..db4fe2e`. Do not scancel.
Do not launch RF again.

---

## 2026-08-28 — In-chunk denoise hooks (lastmix / bpseudo / restep)
**Tags:** methodology, wan, v2v, denoise
**Owner:** agent
**Refs:** `paper_tables/2026-08-28_wan_v2v_denoise_hooks_spec.md`

User asked to implement the three in-chunk tests, not CachedSearch.
`sf_lastmix`: if the last of 4 DMD steps punches sharp/sat, keep
`0.5 * step3 + 0.5 * step4`. `sf_bpseudo`: hide the last committed
**block** (3 latents; DMD cannot write one latent), extra seed
rewrites B, restore, next block uses that seed if MAE wins.
`sf_restep`: if punch (or always), redo the last 2 of 4 steps.
Same-wave always-on + RF span twins. No `rf_bpseudo_always` (already
NO as `rf_intra_always`). Appear 1.5× locked. Do not scancel intra
16471672–677. Submit `WAVE=lastmix` first if GRES is tight.

---

## 2026-08-28 — lastmix submitted; do not scale prior NOs
**Tags:** methodology, wan, v2v, N-lock
**Owner:** agent
**Refs:** user paste squeue 16505827–837

Smoke lastmix 827/829 R; rest PD GRES. Official cite set is already
caption N=32 (SF / Always / Pseudo). Do not scale AdaSteer, Prefix,
sick, sink, RF intra, RF pseudo, appear, live, I2V, or TTC.
Intra and lastmix stay N=8 until harvest HOLD. The only pre-registered
larger-N is optional caption hosts N=128 (notta + rolling), not our
controller. Pseudo N=128 is a later paper-size call, not this week.

---

## 2026-08-28 — HOLD ≠ skip N=128; field long table is 128
**Tags:** methodology, paper-narrative, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-28_hold_vs_n128.md`, user question

User asked why HOLD methods are not promoted. Correction: we were
using HOLD as “not tonight / not the NOs.” That is not the same as
“N=32 is the paper table.” Long-horizon papers on this stack cite
~128 MovieGen + VBench-Long (Relax / SF++ / Freq / SGF). Caption
N=32 is discovery. Promote only SF + Pseudo + Always on caption
V2V (same-wave). T2V MovieGen-128 stays the optional field compare.
Do not submit 128 while lastmix 827–837 is on GRES. Do not promote
NOs.

---

## 2026-08-28 — GRES is not a reason to hold N=128 baselines
**Tags:** methodology, wan, v2v
**Owner:** agent
**Refs:** user question 03:43, `submit_v2v_caption128.sh`

User asked why not tonight, and said baselines at least should
be 128. Agreed. The only delay was H200 QOS — that is queue
position, not a protocol lock. Caption SF N=32 already finished;
N=128 is the same method on the next 96 videos of the same pool.
Paste `WAVE=hosts` (notta + rolling). Pseudo/Always may queue as
`WAVE=cite` without waiting for lastmix. Do not scancel 827–837.

---

## 2026-08-25 — Intra-chunk motion+appear probe spec
**Tags:** methodology, wan, v2v, intra-chunk
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_wan_v2v_intra_chunk_spec.md`

User saw sat/sharpen inside a denoising chunk. We cannot abort after
a seed is locked. New methods: `sf_intra` (3-latent block; fire if
motion < 0.8× prev OR sharp/color/sat > 1.5× prefix),
`sf_intra_always` (k=4 every block), RF twins at 21-latent span
(`rf_intra` / `rf_intra_always`). Caption N=8. Thresholds
pre-registered. Do not retune on the same 8. Not mid-step EFD.

---

## 2026-08-25 — Pseudo-future Search name + related work
**Tags:** paper-narrative, methodology, wan, v2v
**Owner:** agent
**Refs:** `paper_tables/2026-08-25_pseudo_future_search.md`

Paper name is **Pseudo-future Search** (short: Pseudo). The
held-out last 0.7 s of the real opening is a pseudo-future: a
stand-in label because we have no GT for the invented 30 s. Code
stays `sf_pseudo`. Neighbors: Early Failure Detection (when to
spend; mid-denoise RGB), CachedSearch (always search, cheaper
tries), speculative decode / LatSearch (verify inside one
generation). Intra-chunk abort is a real hole — we commit 21
latents sealed — and is a limitation paragraph, not the 2-month
method. Do not retune γ. Do not rename sidecars.

---
