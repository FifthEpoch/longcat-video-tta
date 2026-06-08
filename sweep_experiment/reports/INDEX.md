# AdaSteer Experiment Index

**Purpose:** Single source of truth for "what experiments exist, where their
results live, what is paper-quality vs discovery, and what remains to be
run." Every agent / human working on this paper should read this first.

**Update rule:** Append a row whenever a new experiment series completes,
update the Status / Findings columns when re-merged. NEVER delete rows
even if results are superseded — mark them `superseded` and keep them
for audit trail.

**Owners:** Wenchen (PI) and any active agent. Last updated: 2026-06-08.

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

---

## Datasets

| Name | Cluster path | N | Notes |
|---|---|---|---|
| Panda 1000v eval | `datasets/panda_1000_480p/` (videos under `videos/`, captions in `metadata.csv`) | 1000 | Used for all Panda eval runs |
| Panda 100v eval | `datasets/panda_100_480p/` | 100 | Discovery |
| UCF-101 1000v eval | `datasets/ucf101_1000_480p/` (videos under `videos/`) | 1000 | Used for `ucf101_932v_*` runs |
| UCF-101 test eval | `datasets/ucf101_test_480p/` | (varies) | Older runs |
| Panda 2048-clip retrieval pool | `datasets/panda_2048_480p/` | 2048 | Built early-May. Has clip-level captions. |
| Panda segment retrieval pool | `datasets/panda_segment_pool/` | ~3000 | Phase 2A segment extraction (mid-late May). Clean per-segment captions. **Pending: scale to 25K via full Panda-70M metadata.** |
| UCF-101 max retrieval pool | `datasets/ucf101_pool_max/` | ~26000 | v2 chunked: each UCF clip cut into ~3 non-overlapping chunks. Built late-May. |

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
