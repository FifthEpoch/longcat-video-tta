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
| `*_NOPROMPT` (in `panda_1000v_standard`, `ucf101_1000v_standard`, `tinylora_*_1000v_standard`) | 1000 | ADA_NOPROMPT, LORA_R8_TTA_NOPROMPT, TL_BARE_R2_NOPROMPT, TL_TIED_R2_NOPROMPT | "TTA without text prompt" ablation: drop caption only at TTA training time, keep caption at inference. Tests whether visual-only TTA gives different gains than visual+text TTA. | In-flight (queued 2026-06-09 via `submit_standard_1000v_noprompt.sh`) |

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
| 2. Panda 25K segment pool build (extends existing 3.3K pool to ~22-25K via full metadata) | 2026-06-09 (12:30 AM UTC+8) | 10617270 (FAILED 49 s — csv field-size-limit + per-source resume bug; see ANALYSIS_LOG 2026-06-09 entry "Panda 25K segment-pool build"). Relaunch pending against patched `scripts/build_panda_segment_pool.py` with `SOURCE_METADATA=datasets/panda_metadata_full/panda70m_training_full.csv`. | ~4-12 h on 16 CPU workers; idempotent (per-(source, chunk_index) resume; existing 3,302 clips preserved) | After done: verify `ls datasets/panda_segment_pool/videos/*.mp4 \| wc -l` ≈ 22-25K, then submit step 3 |
| 3. Panda 25K embedding precompute | After step 2 | TBD | ~30 min on 1 GPU | After done: verify `caption_embeddings.npy` shape ≈ (25000+, 384), then launch step 4 |
| 4. Panda 1000v retrieval sweep (40 jobs, K5/K10 × SIM/RAND, against 25K pool) | After step 3 | TBD | ~3 days with 2-way GPU cap | Merge: `python sweep_experiment/scripts/merge_chunks.py --results-dir sweep_experiment/results/panda_1000v_retrieval --recursive`; then `python scripts/update_merged_with_vbench.py --series-dir sweep_experiment/results/panda_1000v_retrieval --force`; then `python scripts/build_paper_tables.py --regime panda_std --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_panda_retrieval_followup.md` |
| 5. Standard-horizon "TTA without text prompt" ablation (80 jobs: 4 methods × 2 datasets × 10 chunks) | 2026-06-09 | TBD | ~12-16 h per chunk (matches headline standard-horizon walls) | Submit: `bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh`. Merge into the existing series dirs: `python sweep_experiment/scripts/merge_chunks.py --results-dir sweep_experiment/results/panda_1000v_standard --recursive` (and the matching `ucf101_1000v_standard`, `delta_experiment/results/tinylora_{panda,ucf101}_1000v_standard`). Then rebuild the standard-horizon paper table to add the *_NOPROMPT rows: `python scripts/build_paper_tables.py --regime panda_std --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_headline_1000v_noprompt.md` (and the `ucf_std` variant). |

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
