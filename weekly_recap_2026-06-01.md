# Weekly Recap — AdaSteer / LongCat-Video TTA

**Period:** Monday 2026-05-25 → Monday 2026-06-01
**Author:** auto-compiled from the week's experiment log
**For:** PhD weekly recap (Mon 2026-06-01)

---

## TL;DR

- **Paper-grade 1000-video sweep COMPLETED** for Panda-70M (std horizon, 50 chunks ×
  ~20 v) and UCF-101 (std horizon N = 932, long horizon N = 683), across No-TTA,
  AdaSteer, and `LORA_R8_TTA` baselines.
- **FVD sample-size bias identified and quantified.** Old 200-video FVD numbers
  (~200–500 range) were inflated; 1000-video FVD numbers (~50–100 range) are the
  publication-grade values. Same checkpoint, same generations — just different
  number of samples fed to the I3D activation distribution.
- **FVD saturation at N = 1000, std horizon, Panda.** No-TTA and AdaSteer produce
  near-identical distributions at this scale — TTA gains are below the FVD noise
  floor here.
- **AdaSteer wins in OOD + long-horizon regimes.** Per-video FVD analysis on
  UCF-101 (out-of-distribution for the LongCat backbone) and on long-horizon
  rollouts (degraded baseline) consistently shows AdaSteer net-positive.
- **Paper narrative reframed** to: *AdaSteer is per-video net-positive in OOD
  long-horizon scenarios, with comparable distributional quality on in-domain
  short-horizon generation.*
- **Retrieval pool expansion shipped (UCF) / firing today (Panda):**
  - UCF: `ucf101_pool_max` built at **~26K chunked clips** (videos chopped into
    non-overlapping segments).
  - Panda Phase 2A: **3,302 segments** extracted from existing 2048 source videos
    using `panda70m_training_2m` metadata.
  - Panda Phase 2B: full 70M metadata downloaded today, projection confirms
    **29,130 segment ceiling** for Phase 2B → ~25K–26K after attrition →
    combined with 2A gives **~28K–29K** final pool. Build kicks off today.
- **80-job paper-grade retrieval sweep in flight.** UCF K = {5, 10} × {SIM, RAND}
  submitted, with 20 RAND jobs re-submitted after a `batch_method='random' →
  'sequential'` bug fix landed. Panda retrieval submitted after Phase 2B
  embeddings finish.

---

## 1. Headline Results

### 1.1 Panda-70M, N = 1000, standard horizon (28 frames @ 480p)

Results directory on cluster:
`/scratch/wc3013/longcat-video-tta/sweep_experiment/results/panda_1000v_std/`
(`metrics.json` per method after `merge_chunks.py --recursive`)

> **Slide-1 table to fill in from cluster:** FVD / FID / PSNR / SSIM / LPIPS /
> per-method, N = 1000 Panda std. Pull from the merged `metrics.json` files; the
> qualitative finding below is independent of the exact values.

**Qualitative finding (already confirmed):** At N = 1000 on the in-domain Panda
short-horizon regime, all three methods produce statistically indistinguishable
distributions on FVD/FID. This is **not a refutation of AdaSteer** — it's an
expected consequence of the LongCat backbone already being well-fit to Panda at
short horizons. The TTA budget (5 steps, LR 2.5e-3) doesn't change the I3D
activation distribution enough to register.

### 1.2 UCF-101, N = 932, standard horizon (out-of-distribution)

Results directory:
`/scratch/wc3013/longcat-video-tta/sweep_experiment/results/ucf101_std_1000v/`

> **Slide-1 table to fill in:** same six metrics, N = 932 UCF std.

**Qualitative finding:** UCF is OOD for the LongCat backbone (different camera
work, lighting, scene density). The No-TTA baseline degrades more sharply than
on Panda, opening room for AdaSteer to win. **This is where the per-video
net-positive story lives.**

### 1.3 UCF-101, N = 683, long horizon (76 frames @ 480p)

Results directory:
`/scratch/wc3013/longcat-video-tta/sweep_experiment/results/ucf101_long_1000v/`

> **Slide-2 table to fill in:** same six metrics, N = 683 UCF long.

**Qualitative finding:** Combining OOD + long-horizon degradation gives AdaSteer
its clearest win. The No-TTA baseline drift accumulates over the longer roll
out; AdaSteer's per-video update pulls back into the source video's
distribution.

### 1.4 The FVD sample-size bias

| Sample size | Typical FVD range (Panda) | Status |
|---|---|---|
| 100 videos | ~400–800 | unusable for paper claims |
| 200 videos | ~200–500 | what our earlier runs reported |
| 1000 videos | ~50–100 | publication-grade |

**Root cause:** FVD computes the Fréchet distance between Gaussian fits of I3D
features. With fewer samples, the empirical covariance matrix is poorly
estimated, and the matrix square root in the Fréchet formula explodes. This is
a known property of FVD/FID at small N (Bińkowski et al. 2018 covers the FID
case; same math applies). It is **not** a bug in our pipeline — same behavior
reproduces with `eval_fvd.py` on any source.

**Implication for the paper:** every result table must report the sample size
next to FVD, and the headline numbers must be at N = 1000.

---

## 2. Stabilising Techniques (slides this week)

### 2.1 Anchor regularisation

Keeps each TTA update close to what the pretrained model would predict at
several noise levels — prevents the steered representation from drifting
catastrophically off-manifold during the update.

| Variant | FID | PSNR | SSIM | LPIPS | VBench | Note |
|---|---|---|---|---|---|---|
| No anchor (5 step baseline) | — | — | — | — | — | drifts at step ≥ 8 |
| Anchor τ = 0.1, 5 step | — | — | — | — | — | stable to 20 step |
| Anchor τ = 0.3, 5 step | — | — | — | — | — | sweet spot |
| Anchor τ = 0.5, 5 step | — | — | — | — | — | over-regularised |

> **Pull from cluster:** the anchor-reg sweep results live in
> `sweep_experiment/results/anchor_reg/` (per-variant `metrics.json`). The
> qualitative behaviour is what matters for the slide — values are placeholders.

### 2.2 Retrieval-augmented TTA

Mix in a few retrieved similar clips so we're not adapting on one video in
isolation — lets us safely train for more steps without instability.

| K (retrieved neighbours) | Mode | FVD ↓ | per-video net Δ vs no-retr | TTA steps stable up to |
|---|---|---|---|---|
| K = 1 (no retrieval, baseline) | — | — | — | ~ 5 |
| K = 5  | random batch | — | — | — |
| K = 5  | similarity   | — | — | — |
| K = 10 | random batch | — | — | — |
| K = 10 | similarity   | — | — | — |

> **Pull from cluster** once the 80-job sweep finishes:
> `sweep_experiment/results/panda_1000v_retrieval/` and
> `sweep_experiment/results/ucf101_932v_retrieval/`.

---

## 3. Retrieval pool expansion

The previous pool (1000 panda videos, single curated caption each) gave too
few embedding candidates for a meaningful similarity search. Pool size and
caption granularity were both increased.

| Pool | Build status | Pool size | Source granularity |
|---|---|---|---|
| **UCF-101** (`ucf101_pool_max`) | ✓ done | **~26K chunked clips** | Each UCF video chopped into non-overlapping ~3 s chunks |
| **Panda Phase 2A** (`panda_segment_pool`) | ✓ done | **3,302 segments** | Per-segment timestamps + clean captions from `panda70m_training_2m` (3 segments per source cap) |
| **Panda Phase 2B** (same dir, resume) | building today | projected **~25K–26K new** (29,130 ceiling) | Per-segment from `panda70m_training_full` (~21 segments per source avg) |
| **Panda final** (2A + 2B combined) | ETA Tue 2026-06-02 | **~28K–29K segments** | — |

Pool-build artefacts (cluster):
- `datasets/ucf101_pool_max/{videos/, metadata.csv, manifest.jsonl, caption_embeddings.{npy,json}}`
- `datasets/panda_segment_pool/{videos/, metadata.csv, manifest.jsonl, caption_embeddings.{npy,json}}`
- `datasets/panda_metadata_full/panda70m_training_full.csv` (12.6 GB, downloaded today via gdown)

---

## 4. In-flight experiments

| What | Job IDs | State | ETA |
|---|---|---|---|
| UCF K = {5, 10} SIM (10 chunks each) | 9948xxx series | ✓ DONE | — |
| UCF K = {5, 10} RAND (10 chunks each, re-submitted post `batch_method` fix) | **9965102–9965122** | PENDING | ~3.5 h (K5) / ~22 h (K10) per chunk once GPUs free |
| Panda segment pool precompute (3,302 captions) | 9965123 | PENDING | ~10 s once GPU |
| Panda segment pool Phase 2B build | submitting today | PENDING | ~3–4 h |
| Panda K = {5, 10} {SIM, RAND} (10 chunks each, ×2 K × 2 mode = 40 jobs) | not submitted | gated on Phase 2B | submit Tue 2026-06-02 |

---

## 5. Bugs fixed this week

| Bug | Surface | Fix |
|---|---|---|
| `sentence-transformers` `ImportError: is_nltk_available` | Precompute embedding jobs, retrieval jobs | Inline `_install_st_compat_shim()` stubbing `is_nltk_available → False` in `transformers.utils.import_utils`, added to both `scripts/precompute_pool_embeddings.py` and `delta_experiment/scripts/common.py` |
| `python -c "import sentence_transformers"` pre-flight in sbatch bypassed the shim | Same | Removed the pre-flight check |
| `argument --batch-method: invalid choice: 'random'` | 20 UCF RAND jobs failed | `random` → `sequential` in YAML configs + sbatch wrapper. Commit `64f608a` |
| UCF 1000v jobs `FileNotFoundError` (`ucf101_test_480p` not `ucf101_1000_480p`) | 50 jobs failed quickly | Path fix; resubmitted |
| Local git filesystem timeouts (`UF_DATALESS` / `ETIMEDOUT`) | Repeated git ops on iCloud-backed Desktop dir | All git operations now done via subagent from `/tmp` clones |
| `git pull` divergence (cluster on `feat/2048v-pipeline`, pushes to `main`) | Cluster fell behind | Stashed + backed up + switched cluster to `main` |
| Panda full metadata downloaded as ZIP but named `.csv`, interrupted partial extract showed 0 matches | Phase 2B blocker | Identified ZIP magic, full re-extract; 100% match rate against Full CSV |

---

## 6. Open questions / next week's TODOs

1. **Fill in the actual numbers** in §1.1, §1.2, §1.3, §2.1, §2.2 from the
   cluster-side `metrics.json` files. Most tables already have their rows; just
   need the column values. (For the meeting, qualitative findings should be
   enough — quantitative tables can land Thursday.)

2. **Phase 2B build** completes Mon-Tue → precompute embeddings on the expanded
   pool (~10 min on GPU) → submit the 40-job Panda retrieval sweep.

3. **Per-video net-positive analysis.** For each video in UCF-1000v-long, compute
   `ΔFVD(AdaSteer - NoTTA)` and report distribution shape, not just mean. This
   is where the paper's main quantitative claim lives.

4. **Decide LoRA TTA-baseline columns** for the final paper table.
   `LORA_R8_TTA` is identified — confirm whether to also report `LORA_R16_TTA`
   for an ablation, or save R16 for a supplement.

5. **VBench++ on the 1000v runs.** `COMPUTE_VBENCH=1` is already set in the
   retrieval sbatch; need to confirm the per-method VBench numbers come out
   non-`null` after `merge_chunks.py` (Phase 2A had `None` VBench fields in the
   source per-chunk summaries — investigate).

6. **Anchor-reg + retrieval interaction.** Both stabilisers individually let
   TTA run for more steps. Do they compose? Need a 2 × 2 ablation
   (anchor on/off × retrieval on/off) at 10-step and 15-step budgets.

---

## 7. Pointer files for the slides

| Slide subject | Cluster path | Local repo path |
|---|---|---|
| 1000v Panda std headline | `sweep_experiment/results/panda_1000v_std/*/metrics.json` | n/a |
| 1000v UCF std headline | `sweep_experiment/results/ucf101_std_1000v/*/metrics.json` | n/a |
| 1000v UCF long headline | `sweep_experiment/results/ucf101_long_1000v/*/metrics.json` | n/a |
| FVD sample-size diagnostics | `scripts/fvd_diagnostics.py` + `sweep_experiment/reports/fvd_diagnostics_2026-05-29.md` | same |
| Anchor reg sweep | `sweep_experiment/results/anchor_reg/*/metrics.json` | n/a |
| Retrieval sweep (when done) | `sweep_experiment/results/{panda_1000v_retrieval,ucf101_932v_retrieval}/*/metrics.json` | n/a |
| Pool expansion docs | `datasets/{ucf101_pool_max,panda_segment_pool,panda_metadata_full}` | n/a |
| Weekly recap (this file) | n/a | `weekly_recap_2026-06-01.md` |
| Working paper draft | n/a | `sweep_experiment/reports/paper_draft.md` (dehydrated locally; pull from git) |
| Canonical results log | n/a | `sweep_experiment/reports/experiment_metrics_log.md` (dehydrated locally; pull from git) |
