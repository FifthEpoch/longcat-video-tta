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

## 2026-07-21 — Per-VBench-dimension routability: no dimension is routable (short-horizon 1000v)

**Tags:** routing, vbench, per-dimension, negative-result, routability, literature
**Refs:** scripts/diagnose_routability_per_vbench_dim.py; routability_per_dim_1000v/; ANALYSIS_LOG 2026-07-21 (RESOLVED noise); INDEX router-matrix row

Ran the routability diagnostic per VBench dimension (all 7 raw dims) on
panda_ood_budget_1000v_preview (N=898 paired: 12 configs + NO-TTA scored).
Result: every dimension is un-routable.
- R2 (OOF ridge predicting per-video oracle gain) <= 0 for all 7 dims
  (best = dynamic_degree at -0.004; worst = aesthetic_quality -0.094).
- corr(NO-TTA, config-mean) ~ 0 on every dim (0.012-0.092) -> NO-TTA behaves
  as an independent noise draw per dimension, identical signature to
  VBench-total. This is why adding NO-TTA as a 13th action inflates the
  augmented oracle (max-over-independent-noise) without being routable.
- imaging_quality holds essentially all inter-config movement
  (within-cfg sigma=0.40, oracle gain 0.67) and thus drives VBench-total's
  apparent headroom -- but it too is max-over-noise (R2=-0.08).
- subject_consistency, which looked promising in LONG-CONTEXT Panda
  (2026-06-08 headline note), is flat on THIS short-horizon in-domain pool
  (sigma=0.0025, oracle 0.0035, R2=-0.05). The long-context regime is untested here.

Decision: per-dimension routing on short-horizon in-domain video is a dead end,
consistent with the VBench-total and PSNR routability conclusions. Real signal,
if any, lives in (a) the long-context/long-horizon drift regime and (b) DIVERSE
candidate generation (seed/noise variation), not hyperparameter-config routing.
Literature support (2026): SAVi-DNO's best-of-k-SEED oracle is a REAL +1.18 dB
PSNR (recoverable to +0.83), whereas our config oracle is noise because configs
corr~0.99; TANGO reports +3.1% VBench / -28% FVD via test-time noise optimization;
Video-T1 gains land in semantic dims (+10-19%), motion/imaging barely move.
Next testable step: best-of-k-seeds headroom on our model.

## 2026-07-21 — FVD: AdaSteer TTA HURTS distributional realism; routing helps vs fixed only (confounded, matched rerun pending)

**Tags:** fvd, routing, negative-result, tta-tradeoff, distribution-metric, confound
**Refs:** run_preview_1000v_matched_fvd.sbatch; run_pilot_matched_fvd_baselines.py (--intersect-with-notta); budget_oracle_fvd_1000v_preview/matched/

Matched-FVD on the 1000v OOD-preview pool, 3 policies, one GT cache, one protocol:
  always_notta      N=898  FVD  81.52   (best)
  oracle_best_psnr  N=998  FVD 168.68   (PSNR-router)
  fixed_S10_LR5e3   N=998  FVD 198.20   (worst)

Findings:
1. AdaSteer TTA badly degrades FVD: no-TTA 81.5 -> any-config ~168-198 (2-2.4x).
   Per-video PSNR/VBench gain is ~0, so on this pool AdaSteer is STRICTLY harmful
   (no fidelity gain, large realism loss). Mechanism = the classic TTA tradeoff
   also reported by SAVi-DNO: pixel/context-fitting improves pixelwise metrics
   but pushes samples off the real-video manifold, raising FVD.
2. Routing DOES move FVD in the right direction vs committing to one config
   (198.2 -> 168.7, ~30 pts) — the only place we have seen routing help a metric —
   but it is still far worse than no-TTA. So "router improves FVD" is true ONLY
   relative to a fixed config, not relative to no-TTA.

CONFOUND (blocking a clean claim): no-TTA scored on N=898, fixed/oracle on N=998
(100 NOTTA videos missing). FVD small-N bias is upward, so it cannot explain
no-TTA being the LOWEST; but the 100 extra videos in fixed/oracle (possibly the
hard/OOD ones) could inflate their FVD. Fix implemented: --intersect-with-notta
restricts all policies to the common 898 set (matched-N, re-eval only). Rerun
pending; will update this entry with matched numbers.

Narrative implication: FVD is the metric that separates policies here, and it says
DON'T adapt (on short-horizon in-domain). Combined with the routability result
(per-video config oracle is noise), the deployable recommendation on this regime is
NO-TTA. Real TTA wins in the literature come from long-horizon drift (TANGO -28% FVD)
and diverse-candidate generation, not from AdaSteer-config routing.

---

## 2026-07-21 — Generation seed confound in the 200v pilot's vs-NO-TTA comparison

**Tags:** seeds, reproducibility, pilot, methodology, oracle-headroom
**Refs:** scripts/verify_seed_alignment.py, sweep_experiment/sbatch/submit_notta_pilot.sh,
sweep_experiment/sbatch/submit_notta_1000v_preview.sh, delta_experiment/scripts/run_delta_a.py,
lora_experiment/scripts/run_full_tta.py

Per-generation seed is `base_seed(42) + position-within-chunk (+ rollout_step)`,
where the index is the video's position in its chunk's `eval_videos` slice
(the `enumerate` order == the order in summary.json `per_video_results`). So
per-video seeds line up across arms ONLY when the arms share the same pool
ordering AND chunking.

Verification (scripts/verify_seed_alignment.py, reconstructs per-video seed from
summary.json ordering):
- 1000v preview: config S10_LR5e3 vs in-series NOTTA -> 900/900 (100%) seed MATCH.
- 200v pilot: config S10_LR5e3 vs the NOTTA it was joined against
  (panda_1000v_standard/NOTTA, a DIFFERENT series) -> 1/200 (0.5%) match.

Consequences:
1. The paper-grade 1000v preview is seed-clean; all real vs-NO-TTA analysis
   should use it. No regeneration needed there.
2. The 200v pilot's per-video "oracle Δ vs NO-TTA", augmented oracle (adding
   NOTTA as a 13th option), and any Δ-vs-NOTTA chart are CONFOUNDED — NOTTA used
   different initial noise per video. The pilot's config-vs-config oracle (max
   over the 12) is still valid (those 12 are in-series, seed-matched to each other).
3. This is the mechanism behind the previously-noted VBench "augmented-oracle
   max-over-noise" inflation: the pilot NOTTA was statistically independent of
   the configs because it literally was an independent noise draw (different seed).

Fix: re-generate the pilot NO-TTA in-series with the pilot's exact 2×100 /
SEED=42 chunking (sweep_experiment/sbatch/submit_notta_pilot.sh), which aligns
seeds under the existing scheme, OR retire the pilot's vs-NOTTA numbers in favor
of the seed-clean 1000v preview. Going forward: never borrow NO-TTA cross-series;
always generate compared arms in-series with matched chunking (the preview
pipeline already does this).

---

## 2026-07-21 — Seed-clean 1000v vs-NO-TTA analysis: OOD-routing hypothesis does not hold

**Tags:** seeds, oracle-headroom, OOD-routing, max-over-noise, 1000v-preview, negative-result, methodology
**Refs:** scripts/dump_pilot_chart_data.py, scripts/render_pilot_charts_from_json.py,
scripts/analyze_adasteer_budget_oracle.py, scripts/verify_seed_alignment.py,
sweep_experiment/reports/per_video_analysis/2026-07-21/preview1k_chart_data_seedclean.json,
sweep_experiment/reports/per_video_analysis/2026-07-21/preview1k_ood_charts_seedclean/,
sweep_experiment/reports/per_video_analysis/2026-07-21/adasteer_budget_oracle_1000v_seedclean.md

Re-ran the full vs-NO-TTA analysis on the seed-clean 1000v preview
(panda_ood_budget_1000v_preview), using the IN-SERIES NOTTA (verified 100%
seed-matched to the 12 configs) instead of the confounded cross-series pilot
NOTTA. OOD scores joined via the full segment-pool CSV
(2026-07-10/diffusion_ood_scores_segment_pool.csv): ood_join=998/998,
n_psnr_pool=898 paired.

Findings (all on seed-matched noise):
1. PSNR oracle Δ vs NOTTA (max over 12 configs) is FLAT across OOD quintiles:
   Q1 +0.357, Q2 +0.365, Q3 +0.326, Q4 +0.344, Q5 +0.383 dB; pop mean +0.355 dB.
   No OOD gradient. (The seed confound was the only mechanism that could have
   manufactured a gradient; removing it leaves none.)
2. Per-video argmax is dominated by S20_LR1e2 in EVERY quintile — the most
   aggressive config (20 steps, LR 1e-2) and the WORST population PSNR (19.372,
   last of 12). High-variance arm winning the per-video max = textbook
   max-over-noise. Its share even shrinks with OOD (Q1 67 -> Q5 43 picks).
3. Oracle table: oracle +0.406 dB vs NOTTA, +0.382 dB vs fixed S10_LR5e3
   [95% CI +0.337, +0.429]; but fixed AdaSteer barely beats NOTTA (+0.024 dB).
4. DECISIVE negative: the realized quintile-adaptive policy (deploy modal-best
   config per quintile) = 19.372 dB vs 19.462 fixed = -0.089 dB (LOSES). The
   modal winner is S20_LR1e2 in all 5 quintiles => no quintile-conditional
   policy exists, and the config it names is the worst population performer.
5. VBench: dynamic_degree has the largest relative oracle headroom (+5.16%) but
   is a noisy near-binary metric with no OOD structure (Q3 +0.146, Q4 -0.017,
   SEMs ~= effects). aesthetic_quality +1.40%, imaging_quality +1.08%; none show
   an OOD gradient.

Conclusion: the OOD-stratified budget-routing headroom is a max-over-noise
artifact, not a routable OOD-adaptive signal — now demonstrated on clean,
seed-matched data. This supersedes the confounded 200v pilot's vs-NOTTA charts
(see 2026-07-21 seed-confound entry). Charts saved as PNGs under
per_video_analysis/2026-07-21/preview1k_ood_charts_seedclean/.

---

## 2026-07-22 — FVD comparisons across TTA arms are confounded (effect sign flips); generation is clean

**Tags:** FVD, confound, methodology, sample-size-bias, provenance, 1000v-preview, negative-result
**Refs:** delta_experiment/scripts/common.py (generate_video_continuation, evaluate_generation_metrics, OnlineFrechetAccumulator), sweep_experiment/scripts/eval_fvd.py, sweep_experiment/sbatch/run_preview_1000v_matched_fvd.sbatch

Investigated the suspicious "TTA doubles FVD" result. Code audit: NO-TTA
(run_full_tta.py) and TTA (run_delta_a.py) share the SAME common.py and the SAME
generation path — identical conditioning window (source video[gen_start-cond :
gen_start] = video[34:48]), identical seed (42 + local_idx + step_i), and both
score the identical generated tail gen_output[num_cond:num_cond+num_gen] =
video[48:62] for PSNR/SSIM/LPIPS AND online FVD. generate_video_continuation
returns cond+gen frames [N,H,W,3] in [0,1]. So TTA cannot structurally
restructure the output; it only applies a small weight delta. PSNR moves only
+0.02 dB => the videos are near-identical, so any large FVD change is not a real
TTA effect.

FVD numbers gathered (all confounded — do NOT cite as TTA effect):
- ONLINE merged_summary (gt_cached=n/a => per-video PAIRED reference):
  NOTTA fvd=157.0 @ N=375 (only 375 of ~900 videos accumulated!);
  configs fvd 66-69 @ N=998. => looks like TTA HALVES FVD.
- OFFLINE matched job (frozen preview cache, saved mp4s):
  always_notta fvd=81.5 @ N=898; fixed_S10_LR5e3 fvd=198 @ N=998;
  oracle_best_psnr fvd=168 @ N=998. => looks like TTA DOUBLES FVD.
The EFFECT SIGN FLIPS between the two computations => neither measures TTA's
real effect on FVD. Root confounds: (1) NO-TTA online FVD is incomplete
(375/900 videos) — a data-integrity bug in FVD accumulation/merge; (2) N
mismatch (375 or 898 vs 998) and FVD's strong small-N upward bias; (3) different
video subsets; (4) reference protocol differs (per-video paired online vs frozen
gt_cache offline); (5) provenance differs (raw float gen online vs lossy
H.264-reencoded mp4s from a separate save pass offline).

Also stale: the oracle-analysis FVD row (383.9) is the old N=200 pilot number
computed against panda_1000_longcat.npz, which the preview sbatch explicitly
flags as the WRONG reference for this pool. Ignore it.

Resolution (pending): one matched recompute — ALL arms scored on the SAME
video-ID set (INTERSECT_NOTTA=1 => --intersect-with-notta), SAME frozen preview
reference (gt_caches/panda_ood_budget_1000v_preview_longcat.npz), SAME mp4
provenance, so N is identical across arms. Until that lands, FVD cannot be used
to compare TTA vs NO-TTA. Prerequisite: re-accumulate NO-TTA's missing ~525 FVD
videos (or score all saved NOTTA mp4s offline) so the common set is well-sampled
(>=256).

---

## 2026-07-22 — Full evaluation-metric audit: VBench scores cond+gen (window bug); pixel clean; FVD window ok but comparison broken

**Tags:** audit, evaluation, leakage, VBench, FVD, FID, PSNR, fairness, methodology
**Refs:** sweep_experiment/reports/per_video_analysis/2026-07-22/eval_metric_audit.md,
delta_experiment/scripts/common.py, sweep_experiment/scripts/eval_vbench.py,
scripts/run_vbench_backfill.py, sweep_experiment/scripts/eval_fvd.py,
sweep_experiment/scripts/precompute_gt_features.py, sweep_experiment/scripts/merge_chunks.py

Audited every eval metric for (1) fairness and (2) generated-only scoring
(geometry: gen_start=48, cond=14, gen=14; pipeline returns 29 frames =
[14 cond | 15 gen]; saved mp4 = 29 frames).

- PSNR/SSIM/LPIPS: CLEAN. evaluate_generation_metrics scores gen_output[14:28] =
  video[48:62] vs source video[48:62]. Gen-only, no leakage, same window/GT for
  NO-TTA and TTA (paired, N=898 seed-clean). No action.
- FVD: window CORRECT (online accumulator, GT cache, and offline eval_fvd all use
  gen-only video[48:62]); merge_chunks sufficient-stats sum is correct. BUT the
  comparison is broken: online NO-TTA fvd=157@N=375 vs configs 66-69@N=998 (TTA
  "halves"), offline matched NO-TTA=81.5@898 vs fixed=198@998 (TTA "doubles") —
  sign FLIPS => confound (N/subset/reference-protocol/provenance), FVD is small-N
  biased. Data-integrity bug: NO-TTA online FVD accumulated only 375/~969 videos,
  chunk_5 summary missing/unmerged. Stale 383.9 oracle FVD row = old N=200 vs
  wrong reference (panda_1000_longcat.npz); ignore.
- FID: same gen-only window; same matched-N requirement as FVD.
- VBench++: WINDOW BUG. eval_vbench.py and run_vbench_backfill.py feed the ENTIRE
  mp4 ([14 cond | 15 gen]) to VBench(mode=custom_input) with NO cond-frame
  trimming. So all per-video VBench scores are ~half real conditioning footage:
  violates gen-only, inflates absolute values, compresses/muddies the
  TTA-vs-NO-TTA signal (explains tiny ~0.06 gains), and the mp4 cond region is the
  per-arm VAE reconstruction (not guaranteed identical across arms). winner_dim
  dynamic_degree +5.16% is cond+gen, not gen-only headroom.
- Train->eval leakage: NONE. run_delta_a clamps tta_total_frames to gen_start
  (explicit guard) => TTA trains video[0:48], scores disjoint video[48:62].

Actions: (1) VBench correctness fix — trim first cond frames into a gen-only clip
dir and re-run VBench backfill + oracle/router analysis for all arms; (2) FVD/FID
fairness — matched offline recompute (INTERSECT_NOTTA=1) so all arms share
N/reference/provenance; (3) recover/complete NO-TTA FVD or rely on offline
recompute from the 969 saved mp4s; (4) pixel metrics unchanged.

## 2026-07-22 — Eval-metric fix: VBench gen-only window + matched FVD/FID
tags: [eval, vbench, fvd, leakage, methodology]
refs: sweep_experiment/reports/per_video_analysis/2026-07-22/eval_metric_audit.md,
      sweep_experiment/reports/per_video_analysis/2026-07-22/FIX_AND_RECOMPUTE_RUNBOOK.md

Audit found VBench++ was scoring the FULL saved mp4 ([14 cond | 15 gen] = 29
frames) instead of generated-only frames — ~half of every VBench score was real
conditioning footage, contaminating absolute scores and the TTA-vs-NOTTA signal.
Pixel metrics (PSNR/SSIM/LPIPS) and the FVD/FID *window* were already gen-only
(video[48:62]); FVD *comparisons* were confounded (NOTTA online FVD accumulated
only ~375/969 videos, chunk_5 missing; per-video paired refs vs frozen cache; N
898 vs 998). TTA training uses video[0:48], disjoint from scored video[48:62] —
no train/eval leakage.

Fix (code, pushed this commit): scripts/make_geneval_clips.py trims the first 14
cond frames of each saved mp4 into videos_geneval/ (15-frame gen-only tail,
encoded identically to the pipeline writer; verified frame-exact on a synthetic
29-frame clip: out[0]==src[14], out[-1]==src[28]). run_vbench_backfill.py gained
--videos-subdir/--out-subdir; VBench re-runs on videos_geneval/ into
vbench_results_geneval/. load_per_video_vbench honors VBENCH_SUBDIR env so ALL
analysis consumers (oracle, router matrix, chart dumper) read gen-only with no
call-site edits. update_merged_with_vbench.py --deprecate-existing stashes old
full-clip means under merged_summary["vbench_fullclip_deprecated"] and rebuilds
"vbench" from gen-only. FVD/FID: no code change — recompute all 3 policies
offline vs the frozen preview GT cache on the common video set
(INTERSECT_NOTTA=1). Full ordered recompute in FIX_AND_RECOMPUTE_RUNBOOK.md.
All prior VBench-based numbers on panda_ood_budget_1000v_preview are superseded
pending the gen-only recompute; do not cite them.

## 2026-07-27 — FVD duplication bug in `_index_grid_videos` (invalidates offline fixed/oracle FVD)

**Tags:** fvd, bug, provenance, budget-grid, 1000v-preview
**Refs:** `sweep_experiment/scripts/build_budget_oracle_policy_dirs.py`; `sweep_experiment/scripts/run_pilot_matched_fvd_baselines.py`; `sweep_experiment/reports/budget_oracle_fvd_1000v_preview/`

The "FVD doubles under TTA" anomaly (fixed_S10_LR5e3 offline FVD=216.7 vs NOTTA=81.5) was a symlink-duplication artifact, NOT a real TTA effect. `_index_grid_videos` resolved config mp4s via `find_mp4`, whose `_numeric_id` fallback matched the trailing seg-number of the video id against the leading numeric prefix of the renamed config filenames (`{prefix}_{caption}_..._adasteer.mp4`). Those prefixes are non-unique and sparse (range 0-368 for a 100-video chunk; e.g. two files share prefix `0_`), so 898 distinct video ids collapsed onto **442 unique files**. FVD is a distribution metric; feeding ~2x duplicated clips inflated FVD. NOTTA was unaffected (its files keep raw-stem names and matched directly -> 898 unique).

**Impact:** All offline `fixed_*` and `oracle_best_psnr` matched-FVD numbers under `budget_oracle_fvd*` are INVALID and must be recomputed. Only `always_notta` offline FVD (81.5) and the online per-run FVDs (configs ~67-69) are trustworthy. PSNR/SSIM/LPIPS and VBench analyses are NOT affected — they read per-video metrics directly, never the symlink layer.

**Fix:** `_index_grid_videos` now resolves each record's mp4 by the per-video metric fingerprint `(psnr, ssim, lpips)` embedded in the filename (verified bijective on S10_LR5e3: 1000 records -> 998 matched -> 998 unique files, 0 collisions, 2 known failed-gen). Added a hard bijectivity guard that raises RuntimeError if `#ids != #unique files`, so a duplicated policy dir can never silently corrupt FVD again. Re-run matched FVD with `--clean` to regenerate the fixed/oracle policy dirs.

## 2026-07-27 — Corrected matched FVD (post duplication-bug fix)

**Tags:** fvd, corrected, budget-grid, 1000v-preview
**Refs:** `sweep_experiment/reports/paper_tables/2026-07-27_matched_fvd_1000v_corrected.md`; `sweep_experiment/reports/budget_oracle_fvd_1000v_preview/matched/`; commit `2b23b8b`

Re-ran matched FVD on the common set (N=898, `INTERSECT_NOTTA=1`) after fixing the `_index_grid_videos` duplication bug. Verified via `pilot_matched_fvd_summary.md`:

| Policy | N | FVD | Δ vs NO-TTA |
|---|---:|---:|---:|
| always_notta | 898 | 81.22 | — |
| fixed_S10_LR5e3 | 898 | 84.77 | +3.55 (+4.4%) |
| oracle_best_psnr | 898 | 72.28 | -8.94 (-11.0%) |

The pre-fix fixed/oracle values (fixed=198-217, oracle=169-184) are now formally SUPERSEDED and must not be cited. Conclusion: fixed AdaSteer is FVD-neutral (within run-to-run noise, consistent with online config FVDs ~67-69 and flat PSNR +0.02 dB); the PSNR-oracle modestly reduces pooled FVD (~-11%, upper bound). The "TTA doubles FVD" claim was entirely a symlink-duplication artifact.

## 2026-07-28 — Trained-router FVD (matched-N=898): headroom exists but is not routable

**Tags:** fvd, router, budget-grid, 1000v-preview, routability
**Refs:** `sweep_experiment/reports/paper_tables/2026-07-28_router_fvd_1000v.md`; `scripts/build_router_fvd_dirs.py`; `sweep_experiment/sbatch/run_router_fvd.sbatch`; commit `6a06ada`; job 14847128

FVD of the deployable PSNR- and VBench-motivated routers (5-fold OOF ridge, block A+B+C, gen-only VBench), composed per-video and scored on the matched N=898 pool (all dirs bijective, links=unique=898):

| Policy | FVD | Δ vs NO-TTA | Apply% |
|---|---:|---:|---:|
| always_notta | 81.22 | — | 0% |
| fixed_S10_LR5e3 | 84.77 | +3.55 | 100% |
| router_psnr_ABC_13act | 80.71 | -0.51 | 42.3% |
| router_psnr_ABC_12act | 82.19 | +0.97 | 100% |
| router_vbench_ABC_13act | 82.43 | +1.21 | 40.8% |
| router_vbench_ABC_12act | 82.67 | +1.45 | 100% |
| oracle_best_psnr | 72.28 | -8.94 | — |

Deployable routers are FVD-neutral (within +-1.5 of NO-TTA, all beat the fixed single config) but capture ~6% of the oracle's FVD headroom (-0.5 of the -8.9 gap). Allowing skip (13-action) helps FVD (stay near NO-TTA rather than the fixed config's 84.8); PSNR routers beat VBench routers on FVD. Conclusion consistent with PSNR/VBench routability (R^2(gain) <= 0): real oracle headroom, not routable from cheap features. Router picks are leakage-free OOF (deployable), unlike the oracle which needs GT.

## 2026-07-28 — Cross-metric per-policy table (FVD + PSNR + 7 VBench dims), matched N=898

**Tags:** fvd, psnr, vbench, router, oracle, cross-metric, 1000v-preview
**Refs:** `sweep_experiment/reports/paper_tables/2026-07-28_cross_metric_policy_1000v.md`; `scripts/build_cross_metric_policy_table.py`; commit `ab9b3de`

One row per policy across all three metric families (generated-only VBench):

- Best deployable per column: FVD -> router_psnr_13act 80.71 (<NO-TTA 81.22); PSNR -> NO-TTA 19.438 (all TTA policies slightly below); VB-mean(norm) -> router_psnr_13act 0.8150 (~tie NO-TTA 0.8133).
- oracle_best_psnr: FVD 72.28, PSNR 19.793 (+0.36), but VB-mean 0.8138 ~= NO-TTA -> oracle PSNR headroom does NOT transfer to VBench.
- VBench routers do not robustly win VBench dims (vbench_13act worst on Aes/Imaging).

Conclusion: real per-video oracle headroom on PSNR/FVD, not routable from cheap features, does not extend to VBench. Deployable AdaSteer (fixed or routed) ~= NO-TTA across all metric families.

## 2026-07-30 — Added two PUBLISHED noise-optimization TTA baselines (DNO, Direct Noise Optimization)

**Tags:** comparison-baselines, noise-optimization, related-work, code
**Refs:** comparison_methods/scripts/savi_dno_longcat.py, comparison_methods/sbatch/run_savi_dno_longcat.sbatch, comparison_methods/sbatch/run_noise_opt_baselines.sh

Decision: implement peer-reviewed noise-optimization TTA methods on the frozen
LongCat backbone for apples-to-apples comparison with AdaSteer / LoRA-TTA /
SAVi-DNO. Selected the two published, in-setting noise-optimization methods:
  - DNO — Karunratanakul et al., "Optimizing Diffusion Noise Can Serve As
    Universal Motion Priors", CVPR 2024 (arXiv:2312.11994), ~100 Scholar cites.
    Signature = decorrelation regularizer (keep optimized noise white/Gaussian).
  - Direct Noise Optimization — Tang et al., "Inference-Time Alignment of
    Diffusion Models with Direct Noise Optimization", ICML 2025 (arXiv:2405.18881).
    Signature = probability / "Gaussian-shell" regularizer (||z||^2 ~ d) to avoid
    out-of-distribution reward hacking.

Implementation: both reuse the existing differentiable SAVi-DNO LongCat Euler
sampler via a pluggable `--method {savi_dno,dno,direct_noise_opt}` flag; they
differ ONLY in the in-distribution regularizer (SAVi-DNO uses fresh-noise
interpolation p; DNO/Direct use explicit reg terms with noise_interp off). The
default `savi_dno` is byte-identical to prior behaviour (regularizer=none,
reg_weight=0, noise_interp=on) so existing SAVi-DNO results are unaffected.

Fairness note: DNO / Direct-Noise-Opt were designed to optimize noise against a
reward on the sample being generated; video prediction has no test-time future
reward, so we adopt SAVi-DNO's leakage-free protocol (adapt eps on an OBSERVED
history segment, then apply it to the UNSEEN scored future). This is the only
deployable, fair way to bring these reward-driven methods into prediction and
keeps them matched to AdaSteer.

Status: code only; GPU jobs not yet run. Submit via
comparison_methods/sbatch/run_noise_opt_baselines.sh on the 1000v preview pool
(requires that pool's video dataset dir + the shared preview GT cache for a
matched FVD/PSNR comparison).

## 2026-07-31 — Router is not winning from randomness (it is not winning at all)

**Tags:** router, significance, oracle, max-over-noise, routability, negative-result
**Refs:** scripts/router_significance_analysis.py; paper_tables/2026-07-31_router_significance_1000v.md

Ran a leakage-free significance/randomness analysis on the 1000v preview
(N=898, block A+B+C, 5-fold OOF ridge — identical picks to the router matrix) to
check, before the TTA-method comparison, whether the 12-config (and 13-action)
router beats randomness and whether the oracle headroom is real.

Result across all four variants (PSNR/VBench x 12/13-action): the router does
NOT win. (1) Every Delta-vs-fixed 95% CI includes 0 or is negative; the only
significant cell (VBench 13-action, p=0.048) is significant in the WRONG
direction (router worse than best fixed config, because the added NO-TTA action
hurts). (2) Shuffle-own-picks null p_shuf = 0.94-0.999 everywhere — randomly
reassigning which video gets which of the router's own picks does as well or
better ~94-99.9% of the time, so per-video targeting carries no signal;
argmax match% (0.02-0.085) is at/below the 1/12 chance floor. (3) The
config-oracle headroom (+0.358 dB PSNR, +0.098 VBench-raw) sits AT OR BELOW the
parametric max-over-noise floor (+0.428 dB, +0.099); "captured by router" is
negative. sigma estimated from the same values -> floor is conservative.

Decision: do not present a per-video AdaSteer config router as a positive
result on this grid; treat it as a clean negative result (per-video config
selection unlearnable; oracle gain is a noise artifact). Reframe the DNO /
Direct-Noise-Opt comparison at the population level (does any method move
PSNR/FVD/VBench vs NO-TTA more than AdaSteer's ~+0.02 dB fixed config?), not as
"router beats X." Consistent with the earlier routability diagnostic
(OOF R^2(gain) <= 0, corr(NO-TTA, config) ~ 0).

## 2026-07-31 — New direction beyond AdaSteer: best-of-k seed selection

**Tags:** direction, best-of-k, seed-selection, noise-space, headroom, routability
**Refs:** bestofk_experiment/scripts/run_bestofk_seeds.py;
scripts/analyze_bestofk_headroom.py; bestofk_experiment/sbatch/*

Rationale: the significance analysis showed parameter-delta TTA (AdaSteer grid)
produces near-identical videos, so its oracle headroom is max-over-noise and
unroutable. The lever with *demonstrated real* headroom in the literature
(SAVi-DNO, CVPR'24) is the INITIAL NOISE/seed: different seeds give genuinely
different continuations with a real quality spread. So we pivot to best-of-k
seed selection on the frozen model.

Built (offline/GPU, no model changes): (1) run_bestofk_seeds.py — generates K
continuations per video from K distinct seeds (candidate 0 == deployed
reference), records per-candidate PSNR/SSIM/LPIPS (gen-only window, no leakage)
+ cheap GT-free selector signals (seam continuity, motion, sharpness); reuses
common.py's generation+metric path so geometry matches the 1000v preview pool.
(2) analyze_bestofk_headroom.py — same CI/null machinery as the router
significance probe: (A) oracle best-of-k headroom vs seed0 and vs mean random
seed with bootstrap CI + sign-flip p; (B) routability via within-video
Spearman(signal, PSNR) and a leakage-free OOF ridge selector (fold by video)
with bootstrap CI, shuffle-picks null, random-seed null, and match% vs 1/k.
Verdict distinguishes real+routable / real-but-unroutable / no-headroom.
(3) sbatch runner + chunked submitter (prototype: first 200 videos, K=8).

Next: run the prototype, then decide whether a stronger (model-likelihood)
selector is needed if headroom is real but the cheap signals don't route it.

## 2026-07-31 — FVD is also null for deployable TTA (paired bootstrap CI)

**Tags:** fvd, confidence-interval, null-result, duplication-bug, oracle
**Refs:** sweep_experiment/scripts/fvd_bootstrap_ci.py;
paper_tables/2026-07-31_fvd_bootstrap_ci_1000v.md

Ran the FVD bootstrap CI (job 15044183, matched N=898, B=2000, gen-only window,
paired resample vs always_notta). Point FVD: notta 81.2, fixed 84.8, oracle 72.3.
Paired ΔFVD vs NO-TTA: fixed +3.15 [-5.79, +12.19] = NULL (CI spans 0); oracle
(best-PSNR/video) -10.37 [-21.79, -2.04] = improves but NON-DEPLOYABLE
(max-over-noise per the router significance analysis). So FVD is null for the
deployable fixed config, consistent with PSNR (+0.02 dB) and VBench nulls. This
also definitively closes the earlier "TTA doubles FVD" scare (fixed=216 in the
2026-07-27 matched run) as the _index_grid_videos symlink-duplication bug; on
de-duplicated data fixed FVD ~ NO-TTA. Caveat: absolute bootstrap FVD (~152) is
inflated by with-replacement duplication of clips (covariance bias); trust the
paired ΔFVD CI and the point FVD levels, not the absolute bootstrap mean.

## 2026-07-31 — Deployable routers are ALSO FVD-null (paired bootstrap CI)

**Tags:** fvd, confidence-interval, null-result, router, deployable
**Refs:** scripts/build_router_fvd_dirs.py; sweep_experiment/scripts/fvd_bootstrap_ci.py;
paper_tables/2026-07-31_router_fvd_bootstrap_ci_1000v.md

Extended the FVD bootstrap CI (job 15076548, matched N=898, B=2000, paired vs
always_notta) to the four DEPLOYABLE trained routers (PSNR/VBench x 12/13 act,
block A+B+C, leakage-free OOF ridge; VBench routers on gen-only scores). All four
ΔFVD-vs-NO-TTA CIs include 0: psnr_12 +0.27 [-7.70,+6.37]; psnr_13 -1.14
[-6.86,+3.06]; vbench_12 +0.75 [-9.56,+9.89]; vbench_13 +0.78 [-4.18,+4.45]. So
every deployable policy (fixed config + all routers) is FVD-null vs NO-TTA,
matching the PSNR/VBench nulls. Only the non-deployable PSNR-oracle moves FVD
(-10.37 [-21.79,-2.04]), consistent with the max-over-noise / unroutable finding.
Conclusion: no deployable FVD win from AdaSteer on the 1000v preview.

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
final_loss(=base_loss here; base-loss==total-loss so loss_reduction==0), delta_norm, grad_norm.

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

## 2026-08-04 — Long-horizon population effect: TTA real but negligible; FVD worse
tags: [population-effect, long-horizon, drift, fvd, vbench, ssim, negative-result, gate]
refs: scripts/analyze_population_effect.py;
per_video_analysis/popeffect_panda_longctx.json; popeffect_ucf_longhorizon.json;
popeffect_panda_std.json

Paired per-video delta (TTA - NOTTA) bootstrap CI + sign-flip, PSNR/SSIM/LPIPS + 7 VBench
dims, across regimes (ADA/ADA_S10 vs NOTTA):

panda_longctx_1000v (N=999): REAL effects on SSIM +0.0019 [+0.0006,+0.0032] p=.003,
  LPIPS +0.0033 [+0.0020,+0.0048] p<.001, subject_consistency +0.0010 [+0.0003,+0.0018]
  p=.011. PSNR null (+0.018, CI incl 0). FVD 278.71 -> 284.14 = +5.43 (TTA WORSE, point).
ucf101_683v_longhorizon (N=683): NO real effect on any metric (subj -0.0003 p=.052 border,
  wrong sign). FVD 181.59 -> 183.52 = +1.94 worse.
panda_1000v_standard (N=999, contrast): all null (as predicted). FVD 154.74 -> 153.36 = -1.38.

Reading: the "AdaSteer preserves subject-consistency in the drift regime" story is
statistically real (n=999 power) but PRACTICALLY NEGLIGIBLE (+0.001 subj, +0.002 SSIM,
~0.1-0.3%) and does NOT extend to FVD, which gets WORSE under TTA in every long-horizon
regime. So AdaSteer parameter-TTA is not an FVD lever in any regime we have. The binary
gate is flagged "meaningful" only in the E[delta]!=0 sense; effect sizes are too small to
build on. CAVEAT: long-ctx VBench is full-clip (pre window-bug-fix) so subj is directional;
SSIM/LPIPS/FVD are clean and agree.

DECISION: long-horizon does not rescue AdaSteer for FVD/VBench. Pivot weight to best-of-k
seed selection (real headroom from genuinely different videos; jobs 15284148-155 running).
Still closing the user's router question by scoring the literal binary {best-config, NO-TTA}
gate on FVD (build_router_fvd_dirs.py --actions 2).

---

## 2026-08-04 — Literature review: gate signal, vector placement, streaming (directions memo)
tags: [literature, tta-gating, steering-vector-placement, streaming, long-horizon, directions]
refs: sweep_experiment/reports/2026-08-04_literature_v2v_tta_directions.md

Deep review of V2V/continuation TTA to unblock the null AdaSteer story. Three findings:

1) Per-video "when to adapt" gates in the literature are NOT the final self-sup loss
   (which we proved unpredictive). They read cheaper GT-free signals DURING generation:
   intermediate-denoising decoded preview + quality critic (Early Failure Detection,
   2603.14320), next-frame surprise / Anticipative Head (Forget-Anticipate-Adapt,
   2606.26515), and predicted-noise gaussianity (TANGO, 2607.15849, which reports
   -28.3% FVD in AR video gen). => we probed the wrong signal; re-probe with these.

2) BIG architectural finding: our AdaSteer (DeltaAWrapper) injects ONE global delta into
   the timestep/AdaLN embedding, broadcast identically to every block. The steering
   literature (2512.24143 + LLM work) says the concept-rich, controllable region is the
   MID-LATE RESIDUAL STREAM (~60-75% depth), that single-early/single-late is ineffective,
   and residual stream >> other submodules; video-customization work (CustomTTT AAAI'25,
   Follow-Your-Motion, B-LoRA) localizes appearance vs motion to distinct layers/heads.
   Our global-AdaLN delta is close to the WORST insertion point => strongest mechanistic
   reason AdaSteer is null. Actionable: residual-stream delta on a mid-late band +
   appearance/motion split, ablated vs current global-AdaLN.

3) Streaming (user's "evolve bias one chunk at a time") is well supported: Self-Forcing
   (NeurIPS'25), LongLive (2509.22622), Rolling Forcing, Stream-T1 (2026, training-free
   chunkwise noise-propagation + reward pruning + memory sink). Frame evolving-AdaSteer
   as chunkwise local re-fit + first-chunk sink, triggered by noise-gaussianity.

DECISION: ranked next experiments in the memo — (1) re-probe gate with right signal,
(2) vector-placement ablation [highest upside], (3) TANGO noise-guidance FVD lever,
(4) streaming evolving-bias, (5) best-of-k -> FVD/VBench w/ multi-verifier.

---

## 2026-08-05 — EXP2 placement ablation: residual > adaln (REAL) but neither beats no-TTA
tags: [placement, adasteer, residual-stream, exp2, negative-result, real-but-tiny]
refs: sweep_experiment/reports/paper_tables/2026-08-05_placement_ablation_exp2.md;
per_video_analysis/popeffect_resid_vs_adaln.json; popeffect_resid_vs_notta.json;
popeffect_adaln_vs_notta.json; delta_experiment/scripts/run_delta_a.py (--delta-placement)

Series placement_ablation_panda, N=80 OOD-stratified preview, delta_a S10, identical
config across arms except injection site. Paired per-video Delta (better=+), bootstrap
CI + sign-flip p:

  RESID - ADALN (isolates placement): psnr +0.0485 [+0.0097,+0.0981] p=.013 REAL;
    ssim +0.0012 [+0.0001,+0.0026] p=.041 REAL; lpips +0.0022 p=.105 null.
  RESID - NOTTA: psnr +0.008 [-0.004,+0.022] p=.21 null; all null.
  ADALN - NOTTA: psnr -0.040 [-0.090,-0.002] p=.076 null (mildly hurts); all null.
  Means: RESID 19.305 ~= NOTTA 19.296 > ADALN 19.256.

Reading: placement IS a real lever (residual beats the global-AdaLN delta, p=.013), but
the whole effect is that the AdaLN delta mildly DEGRADES psnr and residual placement
REMOVES that harm, landing back at no-TTA. Residual does NOT beat no-TTA. The 2026-08-04
hypothesis (AdaLN = bad insertion site) is PARTIALLY confirmed: AdaLN is measurably worse
than the mid-late residual band, but fixing placement only recovers to neutral, it does
not unlock a gain. Methodological win: analyzer detected a real +0.05 dB effect at N=80
(p=.013), so the vs-NOTTA nulls are TRUE nulls, not underpowering.

Caveats: N=80; effect tiny (~0.05 dB); FVD NOT scored for arms (COMPUTE_FVD=0, NOTTA=157.05);
single delta smeared across the band — objective-specific appearance/motion multi-vector
(EXP2b) untested. DECISION: (a) score arms on FVD (cheap, clips saved); (b) a single-delta
same-objective multi-vector is low-value since single delta already reaches only neutral;
a real EXP2b needs OBJECTIVE-SPECIFIC losses (content vs motion), not just more vectors.
Higher-EV pivots remain EXP1 (better gate signal) and EXP3 (TANGO noise-guidance FVD lever).

---

## 2026-08-05 — EXP2 placement ablation: ALL-METRIC close-out (null on target metrics)
**Tags:** exp2, placement, adasteer, fvd, vbench, null-result
**Refs:** sweep_experiment/reports/paper_tables/2026-08-05_placement_ablation_exp2.md;
2026-08-04_literature_v2v_tta_directions.md; commit 61becb2

Completed the full metric set for the AdaSteer vector-placement ablation (adaln vs
mid-late residual, N=80 OOD-stratified preview, identical hp except --delta-placement).

- PSNR/SSIM: RESID > ADALN is REAL (+0.0485 dB p=0.013; +0.0012 SSIM p=0.041) but the
  effect is that global-AdaLN δ mildly HURTS (−0.04 dB vs no-TTA) and residual placement
  only RECOVERS to no-TTA (RESID−NOTTA null). Neither beats no-TTA.
- VBench++ 7-dim (gen-only, vbench_results_geneval): NO dimension moves (all CIs include
  0). Online 3-dim values were conditioning-contaminated (subject_consistency 0.941→0.955
  after gen-only recompute) — gen-only supersedes.
- FVD (matched-N=80, preview GT cache, 14/14 window): NOTTA 814.60; ADALN 807.92 (−6.68);
  RESID 808.87 (−5.73). Both <1% of baseline, no CI, and RANK-INCONSISTENT with pixel
  (ADALN edges RESID on FVD, opposite of PSNR/SSIM) → treated as null/noise.

VERDICT: placement is NOT the unlock. It is a real-but-negligible pixel lever and moves
neither VBench nor FVD trustworthily. Continue with EXP3 (TANGO noise-gaussianity FVD
guidance) and EXP1 (better per-video gate probe); deprioritize further placement work
except the untested objective-specific multi-vector design (EXP2b).

Tooling note: the placement arms' clips were metric-fingerprint renamed by
rename_videos.py to <segidx>_<slug>_PSNR-.._SSIM-.._LPIPS-.._adasteer.mp4; for ytid_segN
pools the <segidx> is a NON-unique seg number, which collided find_mp4's glob (80->42,
non-bijective FVD crash). Fixed in build_oracle_policy_dirs.py (commit 61becb2) by
resolving clips via the unique (psnr,ssim,lpips) fingerprint, with output_path/find_mp4
fallbacks (no regression for panda_XXXX runs).

## 2026-08-05 — EXP3 scaffold: TANGO predicted-noise-gaussianity guidance (FVD lever)
**Tags:** exp3, tango, fvd, sampling-guidance, savi_dno
**Refs:** comparison_methods/scripts/savi_dno_longcat.py; comparison_methods/sbatch/submit_exp3_tango.sh; comparison_methods/sbatch/run_savi_dno_longcat.sbatch; 2026-08-04_literature_v2v_tta_directions.md

Implemented a training-free, per-step TANGO-style gaussianity guidance inside the SAVi-DNO
differentiable Euler sampler. Rationale: AdaSteer's per-video delta does NOT move FVD (EXP2
null on FVD + all 7 VBench dims), so we need a DISTRIBUTION-level lever. TANGO nudges the
sampling trajectory so the per-step predicted noise stays ~ N(0, I).

Sign-convention derivation (CRITICAL — this codebase differs from textbook rectified flow):
the Euler update is `x_t -= dt*v_pred` with `dt = sigma_next - sigma_curr < 0`, and LongCat
NEGATES the raw DiT output before stepping, so the code's `v_pred = x0 - eps`. Solving the
interpolation `x_sigma = (1-sigma)*x0 + sigma*eps` gives:
  x0_hat  = x_t + sigma      * v_pred
  eps_hat = x_t - (1-sigma)  * v_pred
Guidance penalty G = mean(eps_hat)^2 + (std(eps_hat)-1)^2 [+ w_k*excess_kurt^2], analytic grad
wrt eps_hat (no DiT backward). Since d eps_hat/d v_pred = -(1-sigma), the descent step
`eps_hat -= lambda*grad` maps EXACTLY to `v_pred += lambda*grad`. Applied only for sigma in
[sigma_lo, sigma_hi] (default [0,0.9]) to avoid the 1/(1-sigma) blow-up near pure noise.

Isolation design: every arm uses the SAME sampler with `--no-optimize` + faithful LongCat
prediction (CFG on, 50 steps; the PVDM no-CFG/10-step recipe is garbage on this backbone). The
ONLY difference is the guidance, so control-vs-TANGO cleanly isolates it. Launcher
submit_exp3_tango.sh submits control + lambda in {0.02,0.05,0.1} at N=80 on the OOD preview pool
(cond=14/frames=28/gsf=48, seed=42 — lines up with AdaSteer/placement). N=80 is a STABILITY +
pixel/VBench-direction screen only (N=80 FVD is unreliable, cf. placement 814@N80 vs 157@N900);
re-run the best lambda at EXP3_N=512 for a trustworthy control-vs-TANGO FVD.

STATUS: code compiles + pushed; pilot not yet launched. Also this turn: chunk-aware placement
submitter (commit 38bd190) to scale EXP2 FVD to N=512 and settle whether residual placement
moves FVD out of the small-N noise band.

## 2026-08-06 — EXP2 CLOSED (reliable-N FVD): AdaSteer TTA degrades FVD; EXP3 TANGO bf16 no-op fixed
**Tags:** exp2, exp3, fvd, placement, tango, reliable-N, negative-result, bugfix
**Refs:** paper_tables/2026-08-06_placement_fvd_n500.md; experiment_outputs/2026-08-06.md

EXP2 reliable-N FVD (N=500, series placement_ablation_panda_512v, jobs 15410043-55):
NOTTA 139.914 / ADA_ADALN 149.825 (+9.910) / ADA_RESID 152.636 (+12.722). NOTTA 814->140 as N
80->500 confirms the N=80 FVD was small-N covariance rank-deficiency; 140 ~ headline 157@N900.
At reliable N BOTH AdaSteer placements DEGRADE FVD, monotone NOTTA<ADALN<RESID in BOTH the
mean-shift and the dominant trace (covariance) term. Residual is the WORST, not a rescue =>
"does residual placement help FVD" = NO. Supersedes the N=80 FVD "null" section of the
2026-08-05 table (the true sign is the OPPOSITE of the N=80 point estimate). EXP2 is now CLOSED
as a negative across ALL metrics (pixel: RESID>ADALN but neither beats NOTTA; VBench 7-dim: no
dim moves; FVD: TTA hurts, residual worst).

EXP3 TANGO pilot (N=80) returned byte-identical metrics across control + lambda{0.02,0.05,0.1}.
Logs proved the flag reached the sampler (tango_guidance=True, control OFF). Root cause: the
guidance magnitude was a bf16 NO-OP — the gaussianity gradient carries a 1/n factor (n~4e5), so
raw grad ~1e-6, x lambda(0.02) ~1e-7, which rounds to exactly 0 in bf16 (rel eps ~4e-3). FIX
(savi_dno_longcat.py _apply_tango_guidance): use only the gradient DIRECTION, rescaled to
lambda * per-sample ||v_pred|| (n-independent, bf16-safe); lambda is now a clean fractional
velocity perturbation toward N(0,I). Added EXP3_CONTROL=0 to submit_exp3_tango.sh to re-run only
the guided arms. Pilot re-run pending.
