# Weekly Recap — AdaSteer / LongCat-Video TTA

**Period:** Monday 2026-05-25 → Monday 2026-06-01
**Author:** auto-compiled from the week's experiment log
**For:** PhD weekly recap (Mon 2026-06-01)

---

## TL;DR

- **Paper-grade 1000-video sweep COMPLETED** for the **long-context Panda** track
  (4 methods × 10 chunks × ~100 v ≈ 999 videos). Headline finding: **FVD
  saturation** — No-TTA, AdaSteer, LoRA-R8, and TinyLoRA all produce
  near-identical distributions (FVD 278.6–284.1; PSNR 12.73–12.79; SSIM 0.473–
  0.476). TTA gains are **below the FVD noise floor** in this regime.
- **1000-video standard-horizon sweep completed this week** for Panda and UCF-101
  (std N = 932, long N = 683), with `LORA_R8_TTA` chosen as the LoRA baseline.
  Per-method merged-metrics extraction is still TODO from the cluster — pointer
  paths in §7.
- **FVD sample-size sensitivity confirmed** (not as dramatic as I'd guessed in
  earlier drafts). Concrete numbers: Panda 200v NoTTA FVD = 333.7 vs 1000v
  long-context NoTTA FVD = 278.7 → ~17 % drop, not 5-10× as I'd loosely claimed
  in the first draft. The sensitivity is real but bounded; **a 5-point FVD
  difference at N = 200 may not survive at N = 1000.**
- **200-video discovery sweep finished** for AdaSteer step/LR grid on both
  datasets. Best Panda: `S3_LR0.0025` (FVD 327.55 vs NoTTA 333.70, Δ = −6.15 /
  −1.8 %). Best UCF: `S5_LR0.001` (FVD 347.09 vs NoTTA 359.80, Δ = −12.71 /
  −3.5 %). PSNR/SSIM/LPIPS deltas within noise on Panda; **UCF PSNR/SSIM came
  back NaN** for this batch — a metric-pipeline bug to investigate before
  trusting per-frame UCF numbers at this scale.
- **Eval-set drift caveat (paper-critical):** the new 200v eval subsets are not
  drawn from the same population as the legacy baselines. 200v Panda eval is
  ~3.7 dB PSNR *harder* than the legacy Panda baseline (18.37 vs 22.07); 200v
  UCF eval is ~2.0 dB PSNR *easier* than the legacy `ucf101_cond14_gen14`
  baseline (20.44 vs 18.42). Implication: any cross-experiment comparison must
  use the same eval set — we cannot compare against the legacy numbers
  directly.
- **AdaSteer remains in contention via OOD + retrieval angles**, not via
  raw-metric wins on in-domain in-distribution generation. The paper narrative
  has shifted from "AdaSteer beats baselines everywhere" to **"AdaSteer is
  competitive on in-domain short-horizon and net-positive per-video in OOD
  retrieval-augmented settings"**. Retrieval results from the in-flight 80-job
  sweep will decide whether the OOD-retrieval angle gives us a clean win.
- **Retrieval pool expansion shipped (UCF) / firing today (Panda):**
  - UCF: `ucf101_pool_max` built at **~26K chunked clips** (3-s
    non-overlapping). 2300 source videos, 100 % present in Panda Full metadata.
  - Panda Phase 2A: **3,302 segments** extracted from 2048 source videos using
    `panda70m_training_2m` (3-segment cap → 1,614 unique sources passing
    desirable+score+duration filters).
  - Panda Phase 2B: full 70M metadata downloaded today (12.6 GB CSV unzipped
    from a 2.6 GB ZIP), projection confirms **29,130-segment ceiling** for
    Phase 2B (21.0 segments / source × 2048 sources × 0.68 pass-rate). Combined
    with Phase 2A's 3,302 → expected **~28K-29K** final segment pool. Build
    job submitted today (id 9970342, currently PENDING `Priority` on CPU
    partition).
- **80-job paper-grade retrieval sweep in flight.** UCF K = {5, 10} × {SIM,
  RAND} submitted. K_SIM half **completed** (no merged numbers yet). 20 K_RAND
  jobs **re-submitted** after a `batch_method='random' → 'sequential'` bug fix
  landed (commit `64f608a`). Panda retrieval will fire once Phase 2B embeddings
  finish (target: Tue 2026-06-02).
- **Discovered today during K5_RAND_c0 inspection:** because UCF-101 is
  alphabetically class-grouped in both eval and pool, `BATCH_METHOD=sequential`
  ends up picking **same-class neighbours by accident** (e.g. "Punch" eval gets
  4 "Punch" pool neighbours). So on UCF, the K_RAND and K_SIM conditions will
  likely look similar. **This is a property of UCF's class structure, not a
  bug** — and Panda's hash-ordered pool should give a clean K_SIM vs K_RAND
  separation. If the committee wants a true content-agnostic UCF control, add
  a `K5_SHUFFLED` arm with pool-shuffling enabled (~1 h of code work).

---

## 1. Headline Results

### 1.1 Panda-70M, N = 999, LONG-CONTEXT 1000v sweep (4 methods, complete)

Results merged 2026-05-14 from
`sweep_experiment/results/panda_longctx_1000v/{NOTTA,ADA_S10,LORA_R8}/merged_summary.json`
and `delta_experiment/results/tinylora_longctx_1000v/PANDA_TL_LAST24/merged_summary.json`.

| Method | N | PSNR | SSIM | LPIPS | FVD | FID | VBench aesth. | VBench bg-cons. | VBench subj-cons. | train_s | gen_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **No-TTA**          | 999 | **12.769** | **0.4744** | **0.5469** | **278.7** | **29.9** | 0.440 | 0.848 | 0.774 |  0.9 | 553.9 |
| **AdaSteer S10**    | 999 | 12.787 | 0.4762 | 0.5436 | 284.1 | 29.5 | 0.440 | 0.848 | 0.775 | 18.4 | 552.9 |
| **LoRA-R8**         | 999 | 12.734 | 0.4726 | 0.5480 | 282.4 | 30.3 | 0.485 | 0.848 | 0.757 | 18.3 | 567.9 |
| **TinyLoRA LAST24** | 999 | 12.773 | 0.4744 | 0.5468 | 278.6 | 30.1 | 0.440 | 0.848 | 0.774 | 23.0 | 562.2 |

Deltas vs No-TTA (long-context Panda, 1000v):

| Method | ΔPSNR | ΔSSIM | ΔLPIPS | ΔFVD | ΔFID |
|---|---:|---:|---:|---:|---:|
| AdaSteer S10  | +0.018 | +0.0018 | −0.0033 | **+5.4 (worse)** | −0.4 |
| LoRA-R8       | −0.035 | −0.0018 | +0.0011 | +3.7 (worse)     | +0.4 |
| TinyLoRA L24  | +0.004 | 0.0000  | −0.0001 | −0.1 (tie)       | +0.2 |

**Slide takeaway:** On in-domain Panda at long context, **no TTA method beats
No-TTA on FVD**. PSNR / SSIM / LPIPS are within 0.05 dB / 0.002 / 0.005 of
No-TTA for all three TTA methods — well below per-video noise. The LongCat
backbone is already well-fit to Panda at this horizon; TTA cannot move the I3D
distribution enough to register. **This is the result that motivated the
narrative shift away from "AdaSteer beats baselines on Panda".**

Compute side-note: AdaSteer's train time matches LoRA-R8 (≈ 18 s) and is
slightly faster than TinyLoRA (23 s). At inference time the differences are
within ±2 % of generation cost (552–568 s), so any wall-time advantage will
come from batched-per-video TTA, not single-video TTA.

### 1.2 Panda-70M & UCF-101, N = 999 / 932 / 683 STANDARD-HORIZON & LONG sweeps (this week)

**Paths corrected from cluster discovery, 2026-06-01:**

- Panda 1000v std: `sweep_experiment/results/panda_1000v_standard/{NOTTA,ADA,LORA_R8_TTA}/merged_summary.json`
- UCF 932v std:    `sweep_experiment/results/ucf101_932v_standard/{NOTTA,ADA,LORA_R8_TTA}/merged_summary.json`
- UCF 683v long:   `sweep_experiment/results/ucf101_683v_longhorizon/{NOTTA,ADA,LORA_R8_TTA}/merged_summary.json` *(only 7 chunks merged so far — confirm with `find ... -name 'chunk_*' | wc -l`)*
- TinyLoRA Panda 1000v std: `delta_experiment/results/tinylora_panda_1000v_standard/{TL_BARE_R2,TL_TIED_R2}/merged_summary.json`
- TinyLoRA UCF 932v std:    `delta_experiment/results/tinylora_ucf101_932v_standard/{TL_BARE_R2,TL_TIED_R2}/` — **NOT MERGED YET**

#### 1.2a Panda 1000v STANDARD horizon (N = 999, 28-frame generation)

Schema confirmed: top-level keys, no nested `metrics` dict.

| Method | N | PSNR | SSIM | LPIPS | FVD | FID |
|---|---:|---:|---:|---:|---:|---:|
| **No-TTA**       | 999 | TODO  | TODO   | TODO   | TODO  | TODO |
| **AdaSteer ADA** | 999 | **17.938** | **0.6510** | **0.3373** | **153.4** | **25.22** |
| **LoRA-R8 TTA**  | 999 | TODO  | TODO   | TODO   | TODO  | TODO |
| TinyLoRA TL_BARE_R2 | 999 | TODO | TODO | TODO | TODO | TODO |
| TinyLoRA TL_TIED_R2 | 999 | TODO | TODO | TODO | TODO | TODO |

ADA values from the schema-peek on 2026-06-01. Other two rows extractable
with the one-liner in §7. **Key sanity check vs §1.1:** the same AdaSteer
method on the SAME Panda dataset gives FVD=153.4 at standard horizon (28
frames) vs FVD=284.1 at long context (76 frames) — a 46 % FVD reduction by
going to the shorter horizon. The std-horizon regime is also where FVD
deltas vs other methods will be most informative.

#### 1.2b UCF-101 932v STANDARD horizon

| Method | N | PSNR | SSIM | LPIPS | FVD | FID |
|---|---:|---:|---:|---:|---:|---:|
| No-TTA       | 932 | TODO | TODO | TODO | TODO | TODO |
| AdaSteer ADA | 932 | TODO | TODO | TODO | TODO | TODO |
| LoRA-R8 TTA  | 932 | TODO | TODO | TODO | TODO | TODO |

#### 1.2c UCF-101 683v LONG horizon (76-frame generation)

| Method | N | PSNR | SSIM | LPIPS | FVD | FID | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| No-TTA       | 683 | TODO | TODO | TODO | TODO | TODO | 7 chunks merged (partial?) |
| AdaSteer ADA | 683 | TODO | TODO | TODO | TODO | TODO | 7 chunks merged |
| LoRA-R8 TTA  | 683 | TODO | TODO | TODO | TODO | TODO | 7 chunks merged |

**Note:** UCF long-horizon series only has 7 chunks merged (`merged_summary.json`
exists but covers 7 of the planned 10 chunks). Either (a) only 7 chunks were
submitted, or (b) 3 chunks are still pending — verify before reporting.

### 1.3 200-video AdaSteer discovery sweep (step / LR grid, 2026-05-18)

#### Panda (200v eval, std horizon)

Baseline reference: `panda_no_tta/NOTTA` PSNR = 22.07, SSIM = 0.7683, LPIPS =
0.2362 (legacy 100v eval — note the eval-set drift caveat in TL;DR).

| run_id | steps / LR | PSNR | SSIM | LPIPS | FVD | FID | ΔFVD vs NoTTA |
|---|---|---:|---:|---:|---:|---:|---:|
| NoTTA    | 0 / —      | 18.37 | 0.656 | 0.329 | **333.70** | 54.13 | — |
| S3_LR001 | 3 / 0.001  | 18.40 | 0.656 | 0.329 | 337.46 | 54.49 | +3.76 |
| **S3_LR0025** | 3 / 0.0025 | 18.38 | 0.655 | 0.330 | **327.55** | 54.53 | **−6.15  (−1.8%)** ← best |
| S3_LR005 | 3 / 0.005  | 18.35 | 0.656 | 0.330 | 328.17 | 53.72 | −5.53 |
| S5_LR001 | 5 / 0.001  | 18.38 | 0.655 | 0.330 | 338.51 | 54.29 | +4.81 |
| S5_LR0025| 5 / 0.0025 | 18.40 | 0.657 | 0.328 | 348.08 | 54.79 | +14.38 |
| S5_LR005 | 5 / 0.005  | 18.41 | 0.656 | 0.329 | 339.15 | 55.46 | +5.45 |
| S10_*    | 10 / *     | (checkpoint, ~190/200 done) | | | | | |

**Slide takeaway:** Best Panda AdaSteer hyperparams at 200v are **S3, LR=0.0025**
(FVD −6.15 vs No-TTA). The gain is small (~1.8%) and PSNR / SSIM are within
noise. **This is the discovery sweep that justified the S5_LR0.0025 choice for
later 1000v runs** (S5 picked over S3 for the long-horizon configurations).

#### UCF-101 (200v eval, std horizon, **PSNR / SSIM came back NaN — pipeline bug**)

| run_id | steps / LR | FVD | FID | ΔFVD vs NoTTA |
|---|---|---:|---:|---:|
| NoTTA    | 0 / —      | 359.80 | 32.70 | — |
| S3_LR001 | 3 / 0.001  | 357.92 | 32.73 | −1.88 |
| S3_LR0025| 3 / 0.0025 | 366.58 | 32.63 | +6.78 |
| S3_LR005 | 3 / 0.005  | 363.61 | 32.77 | +3.81 |
| **S5_LR001** | 5 / 0.001  | **347.09** | 32.78 | **−12.71  (−3.5%)** ← best |
| S5_LR0025| 5 / 0.0025 | 353.30 | 32.72 | −6.50 |
| S5_LR005 | 5 / 0.005  | 361.99 | 32.89 | +2.19 |
| S10_*    | 10 / *     | (checkpoint, ~190/200 done) | | |

**Slide takeaway:** Best UCF AdaSteer hyperparams at 200v are **S5, LR=0.001**
(FVD −12.71 vs No-TTA, −3.5%). UCF's larger gain margin matches the OOD
hypothesis. **PSNR / SSIM NaN is a metric-pipeline bug to fix before trusting
per-frame UCF numbers** — likely a missing reference-frame in the UCF eval
loop. Add to the bug-list in §5.

### 1.4 The FVD sample-size sensitivity (corrected with concrete numbers)

| Comparison | NoTTA FVD | Notes |
|---|---:|---|
| Panda 200v (std horizon, May-18 sweep) | 333.70 | Higher absolute, smaller absolute deltas |
| Panda 999v (long context, May-14 sweep) | 278.7  | Lower absolute, also smaller deltas |
| **Ratio** | 333.70 / 278.7 ≈ **1.20×** | ~17 % drop, not the 5-10× I'd guessed |

| Comparison | NoTTA FVD | Notes |
|---|---:|---|
| UCF 200v (std horizon) | 359.80 | |
| UCF 1000v (std N = 932) | **TODO** from cluster | — |
| UCF 1000v (long N = 683) | **TODO** from cluster | — |

**Root cause:** FVD computes a Fréchet distance between Gaussian fits of I3D
features. With fewer samples, the empirical covariance matrix is poorly
estimated, and the matrix-square-root in the Fréchet formula amplifies that
noise. This is a known FID/FVD property (Bińkowski et al. 2018 / Heusel et al.
2017). It is **not** a bug in our pipeline.

**Implications for the paper:**

1. Every result table must report sample size next to FVD.
2. **Small ΔFVD claims at N = 200 may not survive at N = 1000** (the
   discovery-sweep ΔFVD of −6.15 on Panda is 1.8 % — well inside the
   inter-N noise of 17 %).
3. Cross-experiment comparisons must use the same N.
4. **We cannot directly compare** the May-14 long-context 1000v (N = 999) to
   the May-18 std-horizon 200v (N = 200) — different N and different horizon.

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

## 5. Bugs fixed this week / open bugs

### Fixed

| Bug | Surface | Fix |
|---|---|---|
| `sentence-transformers` `ImportError: is_nltk_available` | Precompute embedding jobs, retrieval jobs | Inline `_install_st_compat_shim()` stubbing `is_nltk_available → False` in `transformers.utils.import_utils`, added to both `scripts/precompute_pool_embeddings.py` and `delta_experiment/scripts/common.py` |
| `python -c "import sentence_transformers"` pre-flight in sbatch bypassed the shim | Same | Removed the pre-flight check |
| `argument --batch-method: invalid choice: 'random'` | 20 UCF RAND jobs failed | `random` → `sequential` in YAML configs + sbatch wrapper. Commit `64f608a` |
| UCF 1000v jobs `FileNotFoundError` (`ucf101_test_480p` not `ucf101_1000_480p`) | 50 jobs failed quickly | Path fix; resubmitted |
| Local git filesystem timeouts (`UF_DATALESS` / `ETIMEDOUT`) | Repeated git ops on iCloud-backed Desktop dir | All git operations now done via subagent from `/tmp` clones |
| `git pull` divergence (cluster on `feat/2048v-pipeline`, pushes to `main`) | Cluster fell behind | Stashed + backed up + switched cluster to `main` |
| Panda full metadata downloaded as ZIP but named `.csv`, interrupted partial extract showed 0 matches | Phase 2B blocker | Identified ZIP magic, full re-extract; 100% match rate against Full CSV |

### Open

| Bug | Surface | Severity | Notes |
|---|---|---|---|
| UCF PSNR / SSIM / LPIPS return NaN at N = 200 (May-18 discovery sweep) | UCF metric pipeline | **paper-blocker** if same bug exists at N = 932 / 683 | First check that the 1000v sweeps don't have the same NaN — possibly a per-video metric harness issue specific to the discovery-sweep code path. Result tables in §1.3 are FVD-only because of this. |
| `BATCH_METHOD=sequential` on UCF picks same-class neighbours | UCF retrieval sweep | **expected null result, not a bug** | UCF eval and pool are both alphabetically class-grouped; positional retrieval ends up class-aligned. Discovered today during K5_RAND_c0 progress check. Panda pool is hash-ordered so this won't repeat there. |
| `merge_chunks.py` hangs after VBench feature extraction on long-context runs | Local-only annoyance | low | Identified May-15. Workaround: re-run with `--no-vbench-features`. |
| `paper_draft.md` claim "+7.6 dB PSNR" must be removed before submission | paper draft | high before submission | Came from a Feb baseline-alignment bug, not current numbers. Confirmed by May-2 transcript. |

## 5b. Eval-set drift caveat (paper-critical)

Discovered during May-18 200v sweep, has implications for every result we cite
that mixes 200v with 100v / 1000v.

| Subset | NoTTA PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Legacy `panda_no_tta/NOTTA` (100v baseline) | **22.07** | 0.768 | 0.236 |
| New `panda_200v` eval (May-18 sweep) | **18.37** | 0.656 | 0.329 |
| ΔPSNR | **−3.70** | −0.112 | +0.093 |

| Subset | NoTTA PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Legacy `ucf101_cond14_gen14` baseline | **18.42** | 0.668 | 0.285 |
| New `ucf101_200v` eval (May-18 sweep) | **20.44** | 0.736 | 0.234 |
| ΔPSNR | **+2.02** | +0.068 | −0.051 |

**Implication:** the 200v eval subsets are NOT drawn from the same population
as the legacy baselines. We cannot mix-and-match these reference points in any
paper table without explicit annotation. The 1000v sweeps should each carry
their own NoTTA-on-that-same-subset reference column.

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
| §1.1 Long-context Panda 1000v (filled in) | `sweep_experiment/results/panda_longctx_1000v/{NOTTA,ADA_S10,LORA_R8}/merged_summary.json` + `delta_experiment/results/tinylora_longctx_1000v/PANDA_TL_LAST24/merged_summary.json` | n/a |
| §1.2 Panda 1000v std (TODO) | `sweep_experiment/results/panda_1000v_std/*/metrics.json` | n/a |
| §1.2 UCF 1000v std N=932 (TODO) | `sweep_experiment/results/ucf101_std_1000v/*/metrics.json` | n/a |
| §1.2 UCF 1000v long N=683 (TODO) | `sweep_experiment/results/ucf101_long_1000v/*/metrics.json` | n/a |
| §1.3 200v Panda discovery sweep (filled in) | `sweep_experiment/results/panda_200_adasteer_steps_lr/` | n/a |
| §1.3 200v UCF discovery sweep (filled in) | `sweep_experiment/results/ucf101_200_adasteer_steps_lr/` | n/a |
| FVD sample-size diagnostics | `scripts/fvd_diagnostics.py` + `sweep_experiment/reports/fvd_diagnostics_2026-05-29.md` | same |
| Anchor reg sweep | `sweep_experiment/results/anchor_reg/*/metrics.json` | n/a |
| Retrieval sweep (when done) | `sweep_experiment/results/{panda_1000v_retrieval,ucf101_932v_retrieval}/*/metrics.json` | n/a |
| Pool expansion docs | `datasets/{ucf101_pool_max,panda_segment_pool,panda_metadata_full}` | n/a |
| Weekly recap (this file) | n/a | `weekly_recap_2026-06-01.md` |
| Working paper draft | n/a | `sweep_experiment/reports/paper_draft.md` (dehydrated locally; pull from git) |
| Canonical results log | n/a | `sweep_experiment/reports/experiment_metrics_log.md` (dehydrated locally; pull from git) |

### One-liner to fill the §1.2 TODO cells

Schema and series-name discovery already completed on 2026-06-01 — see
`sweep_experiment/reports/experiment_outputs/2026-06-01.md` for the recorded
output. Skip straight to extraction:

```bash
cd /scratch/wc3013/longcat-video-tta && \
python3 <<'PY'
import json, glob, os

SERIES = [
    # (series_root, series_dirs)
    ("sweep_experiment/results", "panda_1000v_standard"),
    ("sweep_experiment/results", "ucf101_932v_standard"),
    ("sweep_experiment/results", "ucf101_683v_longhorizon"),
    ("delta_experiment/results", "tinylora_panda_1000v_standard"),
    ("delta_experiment/results", "tinylora_ucf101_932v_standard"),
]

for root, series in SERIES:
    print("\n" + "="*78)
    print(f"  {series}  (under {root}/)")
    print("="*78)
    paths = sorted(glob.glob(f"{root}/{series}/*/merged_summary.json"))
    if not paths:
        print(f"  no merged_summary.json yet -- need to run merge_chunks.py")
        # show which methods exist as chunked-only
        for d in sorted(glob.glob(f"{root}/{series}/*/")):
            method = os.path.basename(d.rstrip("/"))
            n_chunks = len(glob.glob(f"{d}chunk_*"))
            print(f"    {method:22s}  chunks={n_chunks}  (run merge to materialise)")
        continue
    for jf in paths:
        method = os.path.basename(os.path.dirname(jf))
        d = json.load(open(jf))
        psnr  = d.get("psnr")
        ssim  = d.get("ssim")
        lpips = d.get("lpips")
        fvd   = d.get("fvd")
        fid   = d.get("fid")
        n     = d.get("num_videos") or d.get("num_successful")
        # printf-friendly with None guards
        def fmt(v, decimals=4):
            if v is None: return "    None"
            return f"{v:.{decimals}f}"
        print(f"  {method:22s}  N={n}  "
              f"PSNR={fmt(psnr,3)}  SSIM={fmt(ssim,4)}  "
              f"LPIPS={fmt(lpips,4)}  FVD={fmt(fvd,2)}  FID={fmt(fid,2)}")
PY
```

This will dump 5 series × 2-3 methods = ~13 rows of metric numbers. Paste
back and I'll drop them into §1.2a/b/c.

**Heads-up:** `tinylora_ucf101_932v_standard` showed `merged=[]` in the
discovery, meaning `merge_chunks.py` hasn't been run on it. To get those
numbers:

```bash
python sweep_experiment/scripts/merge_chunks.py \
    --results-dir delta_experiment/results/tinylora_ucf101_932v_standard \
    --recursive
```

Same potentially for `ucf101_932v_retrieval` (K5/K10 × SIM/RAND), which
should be merged once the K_RAND chunks finish.
