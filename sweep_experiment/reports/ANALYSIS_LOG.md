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

