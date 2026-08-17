# AdaSteer Experiment Index

**Purpose:** Single source of truth for "what experiments exist, where their
results live, what is paper-quality vs discovery, and what remains to be
run." Every agent / human working on this paper should read this first.

**Update rule:** Append a row whenever a new experiment series completes,
update the Status / Findings columns when re-merged. NEVER delete rows
even if results are superseded — mark them `superseded` and keep them
for audit trail.

**Owners:** Wenchen (PI) and any active agent. Last updated: 2026-08-16.

---

## Headline 1000v paper-grade experiments (the 4 we'd publish today)

| Series | Dataset | N | Frames | Methods | Status | Cluster path | Paper table | Key finding |
|---|---|---|---|---|---|---|---|---|
| `panda_1000v_standard` | Panda-70M | 999 | 28 | NOTTA, ADA, LORA_R8_TTA, TL_BARE_R2, TL_TIED_R2 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/panda_1000v_standard/`, `delta_experiment/results/tinylora_panda_1000v_standard/` | Table 1 of [`paper_tables/2026-06-08_headline_1000v.md`](paper_tables/2026-06-08_headline_1000v.md) | AdaSteer ≈ NoTTA on every metric. LoRA shifts distribution (Aes↑, IQ↓). |
| `ucf101_932v_standard` | UCF-101 | 932 | 28 | NOTTA, ADA, LORA_R8_TTA, TL_BARE_R2, TL_TIED_R2 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/ucf101_932v_standard/`, `delta_experiment/results/tinylora_ucf101_932v_standard/` | Table 2 | Same saturation pattern. 932v not 1000v because some chunks failed. |
| `ucf101_932v_retrieval` | UCF-101 | 932 | 28 | K5_SIM, K5_RAND, K10_SIM, K10_RAND | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/ucf101_932v_retrieval/` | Table 2 | All 4 retrieval variants ≈ NOTTA. UCF class-block layout means SIM and RAND retrieve same-class neighbours. NOT a useful retrieval testbed. |
| `panda_1000v_retrieval` | Panda-70M | 999 | 28 | K5_SIM, K5_RAND, K10_SIM, K10_RAND | DONE + 7-dim VBench 2026-07-05 (pool `panda_2048_480p`) | `sweep_experiment/results/panda_1000v_retrieval/` | [`paper_tables/2026-07-05_panda_1000v_retrieval.md`](paper_tables/2026-07-05_panda_1000v_retrieval.md) | SIM≈RAND; PSNR/FVD ≤ ADA; LoRA-like Aes↑ IQ↓ (VB total≈0.778 vs ADA 0.773). |
| `panda_longctx_1000v` | Panda-70M | 999 | 76 | NOTTA, ADA_S10, LORA_R8, PANDA_TL_LAST24 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/panda_longctx_1000v/`, `delta_experiment/results/tinylora_longctx_1000v/` | Table 3 | Saturated at PSNR ~12.77. Subj drops 0.907→0.774 vs std (drift effect). AdaSteer preserves Subj (0.775); LoRA worsens it (0.757). |
| `ucf101_683v_longhorizon` | UCF-101 | 683 | 76 | NOTTA, ADA, LORA_R8_TTA | DONE + VBench backfilled (2026-06-08) | `sweep_experiment/results/ucf101_683v_longhorizon/` | Table 4 | All within 0.02 PSNR. LoRA Aes↑ (0.394→0.433), IQ↓ (0.450→0.430). 683 not 1000 because original chunked submit hit class-name skip. |

---

## Config-routing pilot (N=200 OOD-stratified) — current headline thread

| Series | Dataset | N | Methods | Status | Cluster path | Paper table | Key finding |
|---|---|---|---|---|---|---|---|
| `panda_ood_budget_pilot` | Panda-70M | 200 (40/quintile × 5 OOD) | 12 AdaSteer configs (S{2,5,10,20} × LR{1e-3,5e-3,1e-2}); NOTTA joined by id from `panda_1000v_standard/NOTTA` | DONE + 7-dim VBench backfilled | `sweep_experiment/results/panda_ood_budget_pilot/` | [`2026-07-15_pilot_config_vs_notta_routing.md`](paper_tables/2026-07-15_pilot_config_vs_notta_routing.md), [`2026-07-09_deploy_psnr_router.md`](paper_tables/2026-07-09_deploy_psnr_router.md), [`2026-07-09_deploy_router_aux_metrics.md`](paper_tables/2026-07-09_deploy_router_aux_metrics.md) | Same-200-video: oracle PSNR routing **+0.95 dB vs no-TTA**, **+0.75 vs best fixed config** (also ↑SSIM ↓LPIPS); no single config wins across OOD quintiles. Learned 9-d OOF router realizes ≈7.2% (PSNR) / ≈20.8% (VBench) of oracle gap. VBench/PSNR objectives decouple. |
| `panda_ood_budget_1000v_preview` | Panda segment pool | 1000 | same 12-config grid (+ NOTTA in flight) | **12 configs DONE + merged 2026-07-19** (10 chunks each, mp4s saved). NOTTA submitted 2026-07-19 (jobs 14319937–946). VBench backfill + routers pending. | `sweep_experiment/results/panda_ood_budget_1000v_preview/` | [`2026-07-19_budget_grid_1000v_preview.md`](paper_tables/2026-07-19_budget_grid_1000v_preview.md) | Population metrics FLAT across all 12 configs (PSNR 19.37–19.49, spread 0.11 dB); only train time scales (15→128 s). Aggressive S20_LR1e2 worst → mild over-adaptation. Motivates per-video router + 13th skip-TTA candidate. |
| `panda_ood_budget_1000v_preview` (router matrix) | Panda segment pool | 898 paired | 7 feature blocks (A/B/C + subsets) × {12,13 actions} × {PSNR,VBench}; NOTTA VBench backfilled | DONE 2026-07-21 (offline OOF ridge) | same series; features `per_video_analysis/2026-07-12` | [`2026-07-21_router_full_matrix_1000v.md`](paper_tables/2026-07-21_router_full_matrix_1000v.md), [`2026-07-21_router_1000v_feature_model_suite.md`](paper_tables/2026-07-21_router_1000v_feature_model_suite.md) | **No deployable router beats best fixed config or no-TTA** on either metric, any block. PSNR all cells −0.004…−0.018; best PSNR config = S2 (least-adaptive). VBench 12-action ~flat (cap −0.5%), 13-action uniformly −0.13 (skip-averse). Aug-oracle +1.03 over no-TTA but is max-of-fat-tailed-noise ⇒ un-routable (signal ceiling, not tuning). Open: verify NO-TTA VBench fat tail = genuine variance-reduction vs coverage artifact. |

---

## Long-horizon drift + test-time control (2026-08 — current headline thread)

Native LongCat window (13-cond/80-gen), true autoregressive rollout (feed the
model's own generated tail back). Judge by GT-free per-chunk drift + per-video
paired sign-flip test (`compare_drift_paired.py`); GT pixel metrics span only
~1-2 chunks (source clips short) so they are gating, not paper numbers. All N=8
here is a GATING sample.

| Series | N | Chunks / horizon | Method | Status | Cluster path | Key finding |
|---|---|---|---|---|---|---|
| `longhorizon_sweep_notta_native_12ch` | 8 | 12 / ~60 s | NOTTA baseline | DONE 2026-08-09 | `sweep_experiment/results/longhorizon_sweep_notta_native_12ch/` | Native drift COMPOUNDS with horizon: sharpness +48%, motion +45%, contrast −16% (vs +28/+8/+3 at 6ch/30 s). Headroom is real at ~1 min. |
| `longhorizon_sweep_delta_stream_native_12ch` | 8 | 12 / ~60 s | AdaSteer δ re-fit each chunk on generated window | DONE 2026-08-09 | same root | NULL under paired test (p≥0.26); population "flattening" was cancellation (raised per-video volatility). |
| `longhorizon_sweep_delta_stream_clean_native_12ch` | 8 | 12 / ~60 s | δ re-fit toward clean chunk-0 latents | DONE 2026-08-10 | same root | NULL (p≥0.53); fixes saturation, overshoots contrast fade. 3rd delta recipe to fail; delta line CLOSED (+ ramp contraindicated, routing = noise ceiling). |
| `longhorizon_sweep_bestof_k4_native_12ch` | 8 | 12 / ~60 s | best-of-4 GT-free drift verifier (cand0 = NOTTA seed) | DONE 2026-08-11 | `sweep_experiment/results/longhorizon_sweep_bestof_k4_native_12ch/` | **FIRST credible positive.** Verifier picks non-NOTTA 75%; on 11 GT chunks chosen beats RANDOM by **+0.833 dB PSNR** (81% of by-PSNR oracle), −0.032 LPIPS — passes the credibility gate routing failed. Per-signal oracle capture: sharpness 96%, motion 76%, contrast 29%, color 10%. BUT end-to-end paired |drift| vs NOTTA not yet significant at N=8 (sharpness/motion lean right, contrast wrong). Worth SCALING. |

---

## Wan 1.3B I2V continuation (2026-08 — current stack)

LongCat 13.6B stays as the saturated-large-model audit. Paper methods move
to Wan2.1-T2V-1.3B + Self-Forcing causal DMD, **I2V continuation** (not
T2V-from-scratch). Required comparison once the verifier is ported:
NOTTA | always-BoN | gated-BoN | always-TTC | gated-TTC.

| Series | N | Horizon | Method | Status | Cluster path | Key finding |
|---|---|---|---|---|---|---|
| `i2v_notta_smoke` | 2 | 5 s (85 px) | NOTTA | **DONE 2026-08-16** job 15880611 | `wan_experiment/results/i2v_notta_smoke/h5s_shard0/` | First working generate. n_ok=2, mp4s 5.9/3.9 MB, 8–12 s/clip. Frame-0 MAE vs cond 5.56 / 3.71 (I2V, not noise). Autograd-off fixed the 138 GB OOM. |
| `i2v_notta_16v` | 16 | 5 s + 30 s | NOTTA | READY TO SUBMIT | `wan_experiment/results/i2v_notta_16v/` | `wan_experiment/sbatch/submit_i2v_notta16.sh` |

Timing notes: [`paper_tables/2026-08-16_wan_i2v_smoke.md`](paper_tables/2026-08-16_wan_i2v_smoke.md)

---

## Missing / not-yet-run experiments (paper-blocking or paper-relevant)

| Series | Why it's needed | Cluster status | Decision |
|---|---|---|---|
| `panda_1000v_retrieval` (K5/K10 × SIM/RAND) | UCF retrieval is uninformative due to class-block layout. Panda hash-ordered pool would give a clean retrieval signal. | **DONE** 2026-07-05 (40 jobs, pool `panda_2048_480p`, 999v merged). | **CLOSED** — SIM≈RAND; no PSNR/FVD win vs ADA. See [`paper_tables/2026-07-05_panda_1000v_retrieval.md`](paper_tables/2026-07-05_panda_1000v_retrieval.md). |
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
