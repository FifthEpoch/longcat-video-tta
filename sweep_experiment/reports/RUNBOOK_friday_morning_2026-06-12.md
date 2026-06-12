# Runbook — Friday 2026-06-12 cluster restart

**Status:** READY — execute when cluster is back online (expected ~2026-06-12 morning; full unavailability since 2026-06-09 07:00 covers BOTH compute and login nodes).
**Authorised by:** user 2026-06-11.
- Gating plan Phases 0–3 green-lit (`PLAN_gating_experiment_2026-06-11.md` §8, all four open decisions resolved 2026-06-11; Phase 4 explicitly gated on Phase-3 `RECOMMENDATION.md` review).
- Offline-investigation suite A1–A5 ready to run (`PLAN_offline_investigations_2026-06-11.md`, login-node CPU only).
- NOPROMPT close-out: smoke job 10618645 was submitted 2026-06-09 (Panda × ADA_NOPROMPT × chunk 0); full 80-job sweep is *pending smoke confirmation* per INDEX.md row 6.
- VBench backfill: discovery + mass-submit wrappers already on disk (`scripts/discover_vbench_backfill_targets.py` + `scripts/submit_vbench_backfill_all.sh`); fires after Track A's NOPROMPT jobs finish to fold the missing VBench dims into the NOPROMPT `merged_summary.json` files.

**Total wallclock estimate:** ~5–7 days end-to-end. Critical path is **A2 (NOPROMPT sweep) → A3 (VBench backfill) → paper-table rebuild**, ~6–7 days with the 2-way GPU cap on chunked TTA + 8-way parallelism cap on VBench backfill jobs. Everything else (A1 Phase-0, Track B login-node analyses, A1 correlation) completes inside the first ~6 GPU h and ~15 CPU min respectively.

**One-shot summary:** fire Track A on the login node within minutes of the cluster coming back (~6 GPU h + a 5–7 day chunked TTA sweep), kick Track B on the same login session while Track A is in the queue (~15 min CPU total), then submit Track C as soon as Track A's NOPROMPT jobs finish (no manual chaining — Track C reads the merged summaries Track A writes).

---

## 0. Pre-flight (5 min, login node)

```bash
cd /scratch/${USER}/longcat-video-tta
git pull origin main
git log -1 --pretty=format:'%h %s'   # confirm head is the runbook commit
ls scripts/sbatch/run_compute_tier3_probes.sbatch \
   scripts/compute_tier3_probes.py \
   sweep_experiment/reports/RUNBOOK_friday_morning_2026-06-12.md
```

Quick cluster-health probe (each step takes seconds; abort the runbook if any return errors):

```bash
sinfo -p stake_a100,interactive,gen   # confirm partitions are UP
squeue -u "${USER}" | head             # confirm scheduler is responsive
ls /scratch/${USER}/longcat-video-checkpoints \
   /scratch/${USER}/conda-envs/longcat   # confirm scratch is mounted
nvidia-smi 2>/dev/null | head -1 || true # login-node may not have a GPU; fine
```

Verify the Phase-0 input files are intact (these are the only inputs the GPU jobs cannot reproduce on their own):

```bash
ls datasets/panda_1000_480p/metadata.csv \
   datasets/panda_1000_480p/dynamic_degree.json \
   sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv
wc -l sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv  # expect 1000 (header + 999 videos)
```

If anything above fails, **STOP and triage**; the rest of the runbook assumes a clean baseline.

---

## 1. Track A — GPU sbatch jobs (submit immediately, run in parallel)

### A1. Gating Phase 0 — feature extraction + diffusion-OOD + Tier-3 probes (~3 GPU jobs in parallel + 1 chained correlation; ~3–4 wallclock hours)

Submits 3 GPU jobs (Stage 1a / 1b / 1c — fully independent, all three only depend on the dataset) plus a CPU correlation job that auto-chains behind all three via `--dependency=afterok:1a:1b:1c`.

```bash
cd /scratch/${USER}/longcat-video-tta
bash scripts/sbatch/submit_per_video_feature_pipeline.sh
```

Expected stdout (the wrapper prints the 4 job IDs and the dependency graph; copy these into your notes):

```
Submitted feature extraction       : <EXTRACT_JID>   (stage 1a, h200, ~25 min)
Submitted diffusion-OOD computation: <OOD_JID>       (stage 1b, h200, ~2-3 h)
Submitted Tier-3 probe computation : <TIER3_JID>     (stage 1c, h200, ~2-3 h)
Submitted correlation              : <CORR_JID>      (CPU, after EXTRACT + OOD + TIER3, ~5-10 min)
```

**Wallclock breakdown:** 1a ~25 min, 1b ~2–3 h, 1c ~2–3 h (mirrors 1b — same base-model load, 6 forward passes per video + 3 backwards + 3 Adam steps; the sbatch wrapper caps at 4 h), correlation ~10 min once all three are done. Track A1 end-to-end ~3–4 wallclock h (sequential within stages 1b/1c since they share the same h200 queue, in parallel with 1a; correlation auto-fires when the longest-running stage-1 job completes).

**Outputs (cluster paths):**
- `sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv` ← stage 1a
- `sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv` ← stage 1b
- `sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features.csv` ← stage 1c (NEW)
- `sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation/{correlation_table.{md,csv}, top_features_per_method.md, summary.md, plot_<feature>.png}` ← stage 2

**Monitor:**

```bash
squeue -u "${USER}" -j <EXTRACT_JID>,<OOD_JID>,<TIER3_JID>,<CORR_JID>
squeue -u "${USER}" | grep -E 'extract_video_features|compute_diffusion_ood|compute_tier3_probes|correlate_tta_gain'
```

**Live logs:**

```bash
tail -f sweep_experiment/logs/compute_tier3_probes_<TIER3_JID>.out
tail -f sweep_experiment/logs/compute_diffusion_ood_<OOD_JID>.out
tail -f sweep_experiment/logs/extract_video_features_<EXTRACT_JID>.out
tail -f sweep_experiment/logs/correlate_tta_gain_<CORR_JID>.out
```

**Fallback knobs (don't use unless something fails):**

```bash
# Skip the OOD job (e.g. if stage 1b already finished in an earlier run):
SKIP_OOD=1   bash scripts/sbatch/submit_per_video_feature_pipeline.sh

# Skip the Tier-3 job (e.g. for a fast re-run of just the Tier-1 correlation):
SKIP_TIER3=1 bash scripts/sbatch/submit_per_video_feature_pipeline.sh

# Skip BOTH (correlation joins only the feature CSV):
SKIP_OOD=1 SKIP_TIER3=1 bash scripts/sbatch/submit_per_video_feature_pipeline.sh

# Resume Tier-3 / OOD after a partial run (re-uses the existing CSV's video_ids):
RESUME=1 bash scripts/sbatch/submit_per_video_feature_pipeline.sh
```

### A2. NOPROMPT sweep close-out — standard-horizon Panda + UCF, all 4 methods (~5–7 wallclock days; up to 80 jobs)

Per INDEX.md row 6, smoke job 10618645 (`ADA_NOPROMPT × Panda × chunk_0`, ~8 h wall) was submitted 2026-06-09 but the cluster went down before completion. The full 80-job sweep is gated on smoke-test success.

**Step A2.0 — verify the smoke ran to a sane state during the cluster shutdown** (it may have either completed inside the maintenance window or been requeued):

```bash
sacct -j 10618645 --format=JobID,JobName,State,ExitCode,Elapsed,End -P
# Look for:
#   - State=COMPLETED + ExitCode=0:0  → smoke OK; proceed to A2.1 full submit.
#   - State=CANCELLED/TIMEOUT/FAILED  → re-fire smoke (A2.0a below) before A2.1.
ls -la /scratch/${USER}/longcat-video-tta/sweep_experiment/results/panda_1000v_standard/ADA_NOPROMPT/chunk_0/summary.json 2>&1
# If the file exists and looks sane (size > 0, `python3 -c 'import json; json.load(open(...))'`
# parses), smoke is good.
```

Sanity-check the smoke summary if present:

```bash
python3 -c "
import json, sys
with open('sweep_experiment/results/panda_1000v_standard/ADA_NOPROMPT/chunk_0/summary.json') as f:
    s = json.load(f)
n = len(s.get('results', s.get('per_video_results', [])))
print(f'n_videos={n}  fvd={s.get(\"fvd\")}  fid={s.get(\"fid\")}')
print('TTA caption flag (must be true / TTA_DISABLE_CAPTION=1):',
      s.get('args', {}).get('tta_disable_caption'))
"
```

**Step A2.0a — re-fire smoke ONLY if the 2026-06-09 smoke is missing / failed** (otherwise skip directly to A2.1):

```bash
DRY_RUN=0 NUM_CHUNKS=1 ONLY_DATASET=panda ONLY_METHODS="ADA_NOPROMPT" \
    bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh
# Wait for the resulting chunk_0/summary.json to materialise (~8 h), THEN proceed to A2.1.
```

**Step A2.1 — full 80-job NOPROMPT sweep** (4 methods × 2 datasets × 10 chunks):

```bash
bash sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh
# Submits 80 jobs named t1knp_<dataset>_<METHOD>_c<chunk>.
# Wallclock per job: ADA / LoRA ~8-12 h ; TinyLoRA ~12-16 h.
# With the 2-way GPU cap the full sweep takes ~5-7 wallclock days.
```

Methods (per `sweep_experiment/sbatch/submit_standard_1000v_noprompt.sh` lines 178–193, locked to the headline LR/rank):

| Run ID | Method | Hyperparams |
|---|---|---|
| `ADA_NOPROMPT` | delta_a | `DELTA_STEPS=10 / 5` (panda / ucf), `DELTA_LR=5.0e-3 / 2.5e-3` |
| `LORA_R8_TTA_NOPROMPT` | lora | `LORA_RANK=8, LORA_ALPHA=16, NUM_STEPS=10, LEARNING_RATE=5.0e-5, WARMUP_STEPS=3, WEIGHT_DECAY=0.01, TARGET_BLOCKS=all` |
| `TL_BARE_R2_NOPROMPT` | tinylora | `SVD_RANK=2, N_TIE=1, TARGET_PRESET=qkv_proj, TARGET_BLOCKS=all, TTA_STEPS=20, TTA_LR=1e-3` |
| `TL_TIED_R2_NOPROMPT` | tinylora | `SVD_RANK=2, N_TIE=48, TARGET_PRESET=qkv_proj, TARGET_BLOCKS=all, TTA_STEPS=20, TTA_LR=1e-3` |

**Outputs** (NOPROMPT runs land alongside their prompted siblings — same series dir, suffixed run-id):
- `sweep_experiment/results/panda_1000v_standard/{ADA_NOPROMPT, LORA_R8_TTA_NOPROMPT}/chunk_*/summary.json`
- `sweep_experiment/results/ucf101_1000v_standard/{ADA_NOPROMPT, LORA_R8_TTA_NOPROMPT}/chunk_*/summary.json`
- `delta_experiment/results/tinylora_panda_1000v_standard/{TL_BARE_R2_NOPROMPT, TL_TIED_R2_NOPROMPT}/chunk_*/summary.json`
- `delta_experiment/results/tinylora_ucf101_1000v_standard/{TL_BARE_R2_NOPROMPT, TL_TIED_R2_NOPROMPT}/chunk_*/summary.json`

**Monitor:**

```bash
squeue -u "${USER}" | grep '^[0-9]* t1knp_' | wc -l           # remaining jobs
squeue -u "${USER}" | grep '^[0-9]* t1knp_' | awk '{print $5}' | sort | uniq -c   # state breakdown
```

**Step A2.2 — merge chunks once each `<series>/<METHOD>_NOPROMPT/` has all 10 `chunk_*/summary.json` files** (run this once per series; idempotent):

```bash
python sweep_experiment/scripts/merge_chunks.py \
    --results-dir sweep_experiment/results/panda_1000v_standard --recursive
python sweep_experiment/scripts/merge_chunks.py \
    --results-dir sweep_experiment/results/ucf101_1000v_standard --recursive
python sweep_experiment/scripts/merge_chunks.py \
    --results-dir delta_experiment/results/tinylora_panda_1000v_standard --recursive
python sweep_experiment/scripts/merge_chunks.py \
    --results-dir delta_experiment/results/tinylora_ucf101_1000v_standard --recursive
```

Track C (VBench backfill, §3 below) is the natural follow-up — it reads the merged summaries this step writes.

---

## 2. Track B — Login-node analyses (CPU-only; run in parallel with Track A)

These reproduce the A1–A4 sequence from `PLAN_offline_investigations_2026-06-11.md`. They do **not** share GPU resources with Track A and can run from the same login session immediately after submitting Track A. Total wall ≤ 15 min on the login node CPU.

```bash
cd /scratch/${USER}/longcat-video-tta
git pull origin main   # in case Track A pushed anything

# ---------- B1 — long-horizon per-video analysis (~5 min) ------------------
python3 scripts/analyze_per_video_tta_gain.py \
    --series-path sweep_experiment/results/panda_longctx_1000v \
    --tinylora-series-path delta_experiment/results/tinylora_longctx_1000v \
    --dynamicness-json datasets/panda_1000_480p/dynamic_degree.json \
    --captions-csv    datasets/panda_1000_480p/metadata.csv \
    --output-dir sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon

# ---------- B2 — side-by-side standard vs long-horizon (~30 s) -------------
python3 scripts/compare_horizons_per_video.py \
    --standard-bundle    sweep_experiment/reports/per_video_analysis/2026-06-09 \
    --longhorizon-bundle sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon \
    --output-dir         sweep_experiment/reports/horizon_comparison/2026-06-11

# ---------- B3 — per-chunk ΔFVD sign analysis, both regimes (~30 s) --------
python3 scripts/analyze_per_chunk_fvd.py \
    --series-paths \
        sweep_experiment/results/panda_1000v_standard \
        delta_experiment/results/tinylora_panda_1000v_standard \
        sweep_experiment/results/panda_longctx_1000v \
        delta_experiment/results/tinylora_longctx_1000v \
    --baseline-method NOTTA \
    --output-dir      sweep_experiment/reports/horizon_comparison/2026-06-11/per_chunk_fvd

# ---------- B4 — file-based per-video loss-history aggregation (~5-10 min) -
# Long-horizon first (the primary gap; safe to ctrl-c after this one if time is tight):
python3 scripts/aggregate_loss_history.py \
    --series-path           sweep_experiment/results/panda_longctx_1000v \
    --tinylora-series-path  delta_experiment/results/tinylora_longctx_1000v \
    --output-dir            sweep_experiment/reports/loss_history/2026-06-11/longhorizon \
    --psnr-threshold 0.5

# Standard horizon (the side-by-side reference; ~5 min):
python3 scripts/aggregate_loss_history.py \
    --series-path           sweep_experiment/results/panda_1000v_standard \
    --tinylora-series-path  delta_experiment/results/tinylora_panda_1000v_standard \
    --output-dir            sweep_experiment/reports/loss_history/2026-06-11/standard \
    --psnr-threshold 0.5

# ---------- Commit + push the offline bundles ------------------------------
git add \
    sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon/ \
    sweep_experiment/reports/horizon_comparison/2026-06-11/ \
    sweep_experiment/reports/loss_history/2026-06-11/
git commit -m "analysis: long-horizon per-video + horizon comparison + per-chunk ΔFVD + loss-history (offline bundle 2026-06-12)"
git push origin main
```

---

## 3. Track C — VBench backfill on NOPROMPT methods (queues after Track A's NOPROMPT jobs finish; ~1–2 wallclock days with the 8-way parallelism cap)

Per INDEX.md row 2 of "Headline 1000v paper-grade experiments", the two prompted methods (`ADA`, `LORA_R8_TTA`) and the two TinyLoRA prompted methods already have full 7-dim VBench backfilled (2026-06-05). The four NOPROMPT siblings (which A2 produces) need their VBench dims folded in for the paper-table rebuild.

**Step C.0 — verify A2's NOPROMPT `merged_summary.json` files exist before discovering targets** (Track C is **strictly sequential** after A2.2 — the discovery script requires `merged_summary.json` to read existing-vs-missing dims):

```bash
ls sweep_experiment/results/panda_1000v_standard/{ADA_NOPROMPT,LORA_R8_TTA_NOPROMPT}/merged_summary.json
ls sweep_experiment/results/ucf101_1000v_standard/{ADA_NOPROMPT,LORA_R8_TTA_NOPROMPT}/merged_summary.json
ls delta_experiment/results/tinylora_panda_1000v_standard/{TL_BARE_R2_NOPROMPT,TL_TIED_R2_NOPROMPT}/merged_summary.json
ls delta_experiment/results/tinylora_ucf101_1000v_standard/{TL_BARE_R2_NOPROMPT,TL_TIED_R2_NOPROMPT}/merged_summary.json
```

**Step C.1 — one-time env / cache setup** (already done on past cluster sessions; idempotent):

```bash
bash scripts/setup_vbench_backfill_env.sh
```

**Step C.2 — discover which method dirs need backfill** (writes a TSV the mass-submit reads):

```bash
python3 scripts/discover_vbench_backfill_targets.py \
    --output sweep_experiment/reports/vbench_backfill_targets_2026-06-12_noprompt.tsv \
    --only-needs-backfill
cat sweep_experiment/reports/vbench_backfill_targets_2026-06-12_noprompt.tsv
# Expected: 8 rows (4 NOPROMPT methods × 2 datasets), each with the missing VBench dims listed.
```

**Step C.3 — mass-submit the backfill jobs** (one sbatch per method dir, throttled to 8 in flight):

```bash
TARGETS_FILE=sweep_experiment/reports/vbench_backfill_targets_2026-06-12_noprompt.tsv \
MAX_PARALLEL=8 \
    bash scripts/submit_vbench_backfill_all.sh
# Expected: ~8 jobs named vb_<series>_<METHOD>_NOPROMPT.
# Per-job wallclock: ~3-8 h on h200 (varies with missing-dim count).
# Total wall with MAX_PARALLEL=8 and 8 jobs: ~1 wallclock day.
```

**Step C.4 — fold the new VBench dims into each `merged_summary.json`** (run once all `vb_*` jobs in C.3 complete):

```bash
for d in $(cut -f1 sweep_experiment/reports/vbench_backfill_targets_2026-06-12_noprompt.tsv | tail -n +2); do
    python3 scripts/update_merged_with_vbench.py --method-dir "$d"
done
```

**Step C.5 — rebuild the paper tables to incorporate the now-complete NOPROMPT rows:**

```bash
python scripts/build_paper_tables.py --regime panda_std \
    --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_headline_1000v_noprompt.md
python scripts/build_paper_tables.py --regime ucf_std \
    --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_headline_1000v_ucf_noprompt.md

git add sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_headline_1000v_noprompt.md \
        sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_headline_1000v_ucf_noprompt.md \
        sweep_experiment/reports/vbench_backfill_targets_2026-06-12_noprompt.tsv \
        sweep_experiment/results/panda_1000v_standard/*_NOPROMPT/merged_summary.json \
        sweep_experiment/results/ucf101_1000v_standard/*_NOPROMPT/merged_summary.json \
        delta_experiment/results/tinylora_panda_1000v_standard/*_NOPROMPT/merged_summary.json \
        delta_experiment/results/tinylora_ucf101_1000v_standard/*_NOPROMPT/merged_summary.json
git commit -m "paper-tables: NOPROMPT close-out + VBench backfill folded into merged summaries"
git push origin main
```

---

## 4. Track D — Recipe-modification & TTOM control (Friday afternoon, conditional on Phase 0-3 results)

Two waves, both **independent of Phase 0–3** (can fire any time after the cluster restart). D1 is ready to submit; D2 is BLOCKED on the missing sbatch wrapper `sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh`. Track D is **additive** to Tracks A/B/C, not a replacement. The "conditional on Phase 0–3 results" qualifier in the heading refers to the *paper-narrative* gating (whether the gating recommendation lands first changes which way we tell the Track D story), not to the scheduling gate — the jobs themselves do not read any Phase 0–3 output.

### D1. Modification 1 smoke-test — anchor-frame x0 consistency loss (~2 GPU h on H200; single chunk × 100 videos)

- **Status:** ready; sbatch wrapper exists.
- **Command:**

```bash
bash sweep_experiment/sbatch/submit_smoke_x0_loss.sh
```

- **What it submits:** single-chunk `LORA_R8_TTA` on Panda 1000v with `ANCHOR_X0_WEIGHT=1.0`; 100 videos, chunk 0. Uses the exact headline `LORA_R8_TTA` hyperparameters (rank=8 / α=16 / lr=5e-5 / 10 steps / wd=0.01 / max-grad-norm=10) so the *only* changing variable vs the headline cell is the x0 loss term.
- **Output:** `sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA_X0_W1.0/chunk_0/`.
- **Dependency:** none (can fire any time after cluster restart; **not** gated on Phase 0–3).
- **Decision rule (verbatim from `LITERATURE_tta_recipe_modifications_2026-06-12.md` §1):**
  - Median \|ΔPSNR\| > 0.5 dB vs `LORA_R8_TTA/chunk_0` (either direction) → scale up to full 4-method × 4-λ × 10-chunk sweep (~80 GPU-h, sbatch wrapper TBD).
  - NaN gradients OR \|ΔPSNR\| < 0.05 dB → loss formulation is not the binding constraint; pivot to Modification 2 (VAE-decoder-only TTA).
- **What success on this wave gives the paper:** evidence that the v-prediction-only loss was the binding constraint (Sangare CVPR 2026 critique applies), justifying the recipe change as a paper contribution.

### D2. TTOM iteration-saturation sweep (~125 GPU h serial on H200; ~1500 TTA runs)

- **Status:** spec'd but **sbatch wrapper does not yet exist**. Must be implemented before submission.
- **Spec (from `PAPER_FRAGMENT_ttom_positioning_2026-06-12.md` §"Suggested control"):**
  - `--tta-steps ∈ {10, 20, 40, 80, 160}` × {ADA, LORA_R8_TTA, TL_BARE_R2} × stratified ~100-video Panda 1000v subset.
  - ≈ 1500 TTA runs ≈ 125 GPU-h serial on H200.
  - Plot ΔPSNR / ΔLPIPS / ΔFVD vs. iteration count; the 16-iter regression point in TTOM Table 8 motivates including the high-iter end of the sweep deliberately as an over-shoot.
- **Dependency:** none (can fire any time after cluster restart; **not** gated on Phase 0–3); BLOCKED on sbatch-wrapper implementation.
- **Decision rule:**
  - TTOM-style saturate-then-degrade crossover observed → shared mechanism with TTOM; paper claim is "we reproduce TTOM's saturation-then-degradation in a different setting, mechanism is over-optimization".
  - Monotonic-flat curve at noise floor → distinct mechanism (per-video noise floor); paper claim is "TTOM's mechanism does not transfer; per-video reconstructive TTA is rate-limited by a different bottleneck".
- **What success on this wave gives the paper:** pre-empts the obvious reviewer challenge ("did you just not run enough TTA iterations?") and turns the TTOM citation into either a confirmation or a contrast finding.
- **TODO before this wave can fire:** write `sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh` (~100 LOC, mirrors `submit_standard_1000v_chunked.sh` but iterates over `TTA_STEPS` env-var grid and uses a 100-video subset). Defer until either Wave D1 produces a positive signal OR the user explicitly authorizes the wrapper.

---

## 5. Dependency graph (ASCII)

Times are wallclock from Friday-morning T=0 (login node, immediately after `git pull`).

```
T=0          T~25min       T~3h          T~4h            T~5-7 days       T~8-9 days
─────────────────────────────────────────────────────────────────────────────────────────
A1a extract ─┐
A1b OOD ─────┤
A1c Tier3 ──┐│
            ├┴─→ A1 corr ─→ Phase 1+ univariate / multivariate gating analysis (CPU; gated on A1 corr)
            │                                                              (~1-2 day analysis pass; not in this runbook — fires after RECOMMENDATION.md authorisation)
            │
A2.0 smoke OK? ─→ A2.1 NOPROMPT 80-job sweep ─→ A2.2 merge ─→ C.1-C.4 VBench backfill ─→ C.5 paper-table rebuild
                  (~5-7 days with 2-way GPU cap)              (~1 day with 8-way cap)

B1 → B2 → B3 → B4 (CPU; ~15 min total; runs alongside everything in Track A from T=0)
└─→ git commit + push offline-investigation bundle

D1 smoke (anchor-x0 loss; LORA_R8_TTA × Panda chunk_0; ~2 GPU h on H200; NOT gated on Phase 0–3)
└─→ if median |ΔPSNR| > 0.5 dB: scale up to 4-method × 4-λ × 10-chunk sweep (~80 GPU h; wrapper TBD)
└─→ if NaN grads OR |ΔPSNR| < 0.05 dB: pivot to Modification 2 (VAE-decoder-only TTA)

D2 TTOM iteration-saturation sweep (3 methods × 5 tta-steps × ~100 videos = ~1500 runs ≈ ~125 GPU h serial; NOT gated on Phase 0–3)
└─→ BLOCKED on submit_ttom_iteration_sweep.sh (defer until D1 positive OR user authorises wrapper)
```

---

## 6. Critical path

**A2.1 NOPROMPT 80-job sweep (~5–7 wallclock days)** dominates the critical path. Every Track-C step (VBench backfill, paper-table rebuild) and the eventual NOPROMPT row in the paper headline table is gated on A2.1 completion.

- A1 (gating Phase 0) runs in ~3–4 h and feeds the **separate** gating-experiment analysis workstream that fires later (Phase 1+; not in this runbook — gated on Phase-3 `RECOMMENDATION.md` review per `PLAN_gating_experiment_2026-06-11.md` §8 Decision 1).
- B1–B4 (offline investigations) finish inside the first ~15 CPU min on the login node and are independent of every GPU job.
- C (VBench backfill) is strictly sequential after A2.2 and adds ~1 day to A2's ~5–7 days.

**If GPU queue capacity is tighter than the 2-way cap assumed above** (e.g. shared cluster contention), A2.1 stretches proportionally. A1 and Track C are not affected — they fit in the standard h200 partition with no special priority.

---

## 7. After completion

Each track has its own "what lands in git" line; running the runbook end-to-end produces these commits.

**Track A1 (gating Phase 0):** The Phase-0 CSVs land in `sweep_experiment/reports/per_video_analysis/2026-06-09/` (already-existing dir). The wrapper does NOT auto-commit Phase-0 CSVs; commit them by hand once correlation finishes:

```bash
git add sweep_experiment/reports/per_video_analysis/2026-06-09/{video_features,diffusion_ood_scores,tier3_probe_features}.csv \
        sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation/
git commit -m "phase0: feature + diffusion-OOD + Tier-3 probe CSVs + correlation report (gating plan §3.1)"
git push origin main
```

**Track A2 → A2.2 (NOPROMPT sweep merged):** The merged_summary.json files for the 4 NOPROMPT methods × 2 datasets are written in place; commit them so Track C reads from `origin/main`:

```bash
git add sweep_experiment/results/{panda,ucf101}_1000v_standard/*_NOPROMPT/merged_summary.json \
        delta_experiment/results/tinylora_{panda,ucf101}_1000v_standard/*_NOPROMPT/merged_summary.json
git commit -m "merge: NOPROMPT close-out (4 methods × 2 datasets × 10 chunks)"
git push origin main
```

**Track B (offline investigations):** committed inside §2 ("Commit + push the offline bundles" block).

**Track C (VBench backfill + paper tables):** committed inside §3 step C.5.

**Next phase triggered by A1 completion:** the gating-experiment Phase 1 (univariate analysis) and Phase 2 (multivariate analysis) authorised under `PLAN_gating_experiment_2026-06-11.md` §6. Those phases are CPU-only and run from a separate analysis session — they are NOT in this runbook.

**Next phase triggered by C.5:** the paper-table refresh is the natural input to the gating plan's Phase-3 `RECOMMENDATION.md` (§3.4 of the gating plan); the recommendation gates Phase 4 (long-horizon validation), which requires a separate user authorisation per Decision 1.

---

## 8. What is NOT in this runbook (deferred or out of scope)

- **Gating-experiment Phases 1–3.** Univariate / multivariate / Pareto analysis is CPU-only and runs from a separate analysis session after A1's CSVs land (`PLAN_gating_experiment_2026-06-11.md` §3.2–§3.4). The scripts `analyze_gating_univariate.py` / `analyze_gating_multivariate.py` / `build_gating_pareto.py` are themselves a separate implementation task — none of them exist on disk today.
- **Gating-experiment Phase 4 (long-horizon validation).** Explicitly gated on Phase-3 `RECOMMENDATION.md` review per Decision 1 in `PLAN_gating_experiment_2026-06-11.md` §8. Requires separate user authorisation after the standard-horizon recommendation is in.
- **Panda retrieval pipeline (INDEX.md rows 2–4 of "Pending merges and in-flight sweeps").** Steps 2 (25K pool build), 3 (embedding precompute), 4 (40-job retrieval sweep), and 5 (retrieval × NOPROMPT ablation) are a parallel workstream. They are unblocked by the cluster restart but are not authorised by the user yet for this Friday; INDEX.md keeps the running status.
- **Discovery / ablation runs.** Anything in INDEX.md's "Active discovery / ablation experiments" table is read-only audit material; nothing fires from there on Friday.
- **VBench backfill for already-complete methods.** Track C runs `discover_vbench_backfill_targets.py --only-needs-backfill` so it picks up only the NOPROMPT close-out targets; the prompted methods (ADA, LORA_R8_TTA, TL_BARE_R2, TL_TIED_R2) were already backfilled 2026-06-05 (INDEX.md row 1) and remain untouched.
- **Anything in INDEX.md not authorised by name above.**
