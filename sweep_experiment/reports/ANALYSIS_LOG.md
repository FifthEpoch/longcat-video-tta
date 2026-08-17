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
