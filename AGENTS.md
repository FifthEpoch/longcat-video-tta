# AGENTS.md — persistent index & workflow rules

**Purpose:** This file is the canonical entry point for any AI agent (Claude,
Cursor, etc.) picking up work on this project. Read it FIRST before any
substantive task. Update it whenever a new persistent artifact is created.

---

## 1. Persistent files & where to find them

| What | Path | Notes |
|---|---|---|
| **This index file** | `AGENTS.md` | Updated as artifacts are added |
| **Cluster & sbatch onboarding guide** | `docs/CLUSTER_SBATCH_GUIDE.md` | Self-contained guide for a brand-new agent: cluster quirks (account flag, /scratch, conda/PYTHONHOME), how to write sbatch jobs, and ready-to-use fine-tune + long-horizon continuation recipes. |
| **Master experiment index** | `sweep_experiment/reports/INDEX.md` | **Single source of truth** for what experiments exist + cluster paths. Read this first when picking up work. |
| **Analysis log (decisions/findings)** | `sweep_experiment/reports/ANALYSIS_LOG.md` | Append-only log of paper-relevant findings and decisions. NEVER edit past entries. |
| **Paper-ready tables** | `sweep_experiment/reports/paper_tables/YYYY-MM-DD_<name>.md` | One Markdown file per table set, dated. Reproducible via `scripts/build_paper_tables.py`. |
| **Weekly recap (current week)** | `weekly_recap_YYYY-MM-DD.md` | One per Monday meeting. Latest: `weekly_recap_2026-06-01.md` |
| **Daily experimental-output log** | `sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md` | Append every pasted output (raw + interpretation) |
| **Canonical results memory (legacy)** | `sweep_experiment/reports/experiment_metrics_log.md` | Long-form running log. Superseded by INDEX.md + ANALYSIS_LOG.md as of 2026-06-08, but kept for history. |
| **Paper draft** | `sweep_experiment/reports/paper_draft.md` | LaTeX-aligned narrative + result placeholders. Often dehydrated locally. |
| **Paper LaTeX** | `paper/main.tex`, `paper/sections/*.tex`, `paper/refs.bib` | Real submission source |
| **Run registry** | `experiment_tracker/run_registry.yaml` | Job-ID ↔ result-dir mapping |
| **Cluster repo root** | `/scratch/wc3013/longcat-video-tta/` | All results & raw data live here. Local repo is mostly views. |
| **Wan 1.3B / Self-Forcing setup** | `wan_experiment/README.md` | **32v hybrid DONE** (handcrafted-score efficiency only). Official VBench next: `submit_i2v_vbench_hybrid32.sh`. Sick job 15959146 in flight. Do **not** add TTC yet. |

## 2. CRITICAL workflow rules

### 2a. iCloud / `UF_DATALESS` gotcha

The local repo at `/Users/macrohard/Desktop/longcat-video-tta/` lives on iCloud
Drive. Files routinely dehydrate (`UF_DATALESS`) and `.git` ops time out
(`ETIMEDOUT`). **Never** run `git add / commit / push` directly from the local
working tree. Use the subagent pattern:

1. Write/edit files in `/Users/macrohard/Desktop/longcat-video-tta/` with the
   normal file tools — this works because writing materializes the file.
2. Dispatch a `shell` subagent that does `git clone --depth=1` into `/tmp`,
   `cp` the local files into the clone, then commits and pushes from `/tmp`.
3. Subagent prompt template is in §5 of this file.

After pushing, the user pulls on the cluster:
```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
```

### 2b. Save EVERY pasted experimental output

When the user pastes terminal/cluster output:

1. **Append the raw output** verbatim to today's
   `sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md` with a
   timestamped section header.
2. **Add a 1-3 line interpretation** below the raw block.
3. If the output contains paper-grade metrics, also update the relevant
   weekly recap table AND the master `INDEX.md`.
4. If a new fact emerges that future agents must know (a path, a bug, a
   workflow change), add it to AGENTS.md.

If the date file doesn't exist yet, create it with the standard header
template in §4.

### 2b-bis. Record-keeping commandments (ADDED 2026-06-08 after the user
called out repeated record-keeping failures)

These are NON-NEGOTIABLE. Every agent that touches this repo must follow
them. The user has explicitly cited "bad record keeping" as a paper-blocking
issue. Past failures included: stale `merged_summary.json` numbers leaking
into recap tables; results not saved locally; no clear mapping from cluster
paths to paper tables; no audit trail when narrative pivots happened.

1. **Whenever you produce a table that the user might cite in the paper or
   meeting,** save it as a dated Markdown file under
   `sweep_experiment/reports/paper_tables/YYYY-MM-DD_<short_name>.md` AND
   push it to GitHub the same turn. Do NOT just emit it inline in chat
   and move on.

2. **Whenever a new experiment series finishes,** add a row to
   `sweep_experiment/reports/INDEX.md`. Include cluster path, methods,
   N, frames, status, and key finding. Update existing rows when a series
   is re-merged or backfilled.

3. **Whenever you reach a methodology decision or paper-narrative finding,**
   append an entry to `sweep_experiment/reports/ANALYSIS_LOG.md` with date,
   tags, refs, and 5–15 line body. Past entries are immutable; rebut with
   a new entry.

4. **Whenever the user pastes raw cluster output,** the rule from §2b
   applies (append to `experiment_outputs/YYYY-MM-DD.md`). DO NOT skip
   this even if the output seems "uninteresting" — context that looks
   trivial today is what unblocks debugging next week.

5. **Every paper-grade table must be regenerable from cluster data.**
   Use `scripts/build_paper_tables.py --regime <regime>` to produce
   tables from `merged_summary.json` files. If you produce a table by
   any other means (manual edit, ad-hoc calculation), document it in
   `ANALYSIS_LOG.md` so future agents know not to overwrite it.

6. **Push every record-keeping update to GitHub the same turn.** The
   local iCloud workspace is unreliable; GitHub is the persistence layer.
   If you wrote to `INDEX.md` / `ANALYSIS_LOG.md` / `paper_tables/*.md`,
   the next thing you do is dispatch the subagent push (§5).

7. **If you find a stale or wrong number in a published table,** add a
   new dated table file (don't edit the old one — keep the audit trail)
   AND add an entry to `ANALYSIS_LOG.md` explaining what was stale and why.

8. **NEVER delete generated videos without a manifest + metric capture.**
   The `*.mp4` files are the ONLY source from which frame-based metrics
   (VBench++, FID, FVD, future perceptual metrics) can ever be recomputed;
   once deleted a run is frozen at whatever is already in `merged_summary.json`.
   Before deleting any generated videos:
   (a) run `scripts/build_run_manifest.py` (records per-run provenance +
       metrics + a sha1 pool fingerprint proving which runs share a test set);
   (b) confirm each run's needed metrics — ESPECIALLY VBench — are captured,
       and backfill on the saved frames first if not;
   (c) curate a few matched examples into `figure_bank/` via
       `scripts/curate_figure_bank.py` (keys on the normalized video index so
       NOTTA/ADA/LoRA filename schemes align);
   (d) delete only via `scripts/cleanup_generated_videos.sh` — use PURGE
       allowlist mode (delete only named series), which refuses to run without
       a manifest and hard-protects `datasets/`, `**/figure_bank/`,
       `baseline_experiment/results/gt_clips_*`, and `LongCat-Video/`.
   Comparison-method / T2V baselines (PVDM, DFoT, OpenSora, LongCat-T2V) have
   NO stored metrics as of 2026-07-17 — do not delete their frames until their
   FVD/FID/etc. are computed and saved.

### 2c. Local repo is mostly dehydrated

Don't waste tokens trying to read files like
`sweep_experiment/reports/experiment_metrics_log.md` from the local filesystem
— they're almost certainly dehydrated. To read them, do one of:

- Ask the user to `cat` it on the cluster and paste the output.
- Dispatch a subagent that clones from GitHub into `/tmp` and reads from there.
- Read the raw blob via `gh api` (slower).

For writing: writing **creates** the file fresh on disk (no dehydration
issue), so writing to `experiment_outputs/2026-06-01.md` is fine even if the
parent directory is dehydrated — the directory rehydrates as the file appears.

### 2d. Cluster series-name conventions (CONFIRMED 2026-06-01)

The cluster's `sweep_experiment/results/` directory naming convention for
recent paper-grade work:

| Series dir | What it is | Methods present |
|---|---|---|
| `panda_1000v_standard/` | Panda-70M N=999 std horizon | NOTTA, ADA, LORA_R8_TTA |
| `panda_longctx_1000v/` | Panda-70M N=999 LONG context | NOTTA, ADA_S10, LORA_R8 |
| `ucf101_932v_standard/` | UCF-101 N=932 std horizon | NOTTA, ADA, LORA_R8_TTA |
| `ucf101_683v_longhorizon/` | UCF-101 N=683 long horizon | NOTTA, ADA, LORA_R8_TTA (7 chunks — partial?) |
| `ucf101_932v_retrieval/` | UCF retrieval sweep (this week) | K5/K10 × SIM/RAND, NOT YET MERGED |

TinyLoRA lives separately under `delta_experiment/results/`:
- `tinylora_panda_1000v_standard/` (TL_BARE_R2, TL_TIED_R2 — merged)
- `tinylora_ucf101_932v_standard/` (TL_BARE_R2, TL_TIED_R2 — NOT merged)
- `tinylora_longctx_1000v/` (PANDA_TL_LAST24 — merged, used in §1.1 of recap)

### 2d-bis. Comparison baselines (ADDED 2026-07-17)

Three TTA comparison baselines for AdaSteer, all leakage-free and pinned to the
paper pools + shared metric code:

| Baseline | Horizon | Entry point | Notes |
|---|---|---|---|
| **SAVi-DNO** (noise opt) | short (14+14@48) | `comparison_methods/scripts/savi_dno_longcat.py` + `sbatch/run_savi_dno_longcat.sbatch` | DEFAULT is now the **fair leakage-free** protocol (adapt noise on observed history, predict unseen future). `--oracle-leak` (env `SAVI_ORACLE_LEAK=1`) reproduces the OLD behaviour = optimizes noise against the scored future = ORACLE upper bound only. The previous default leaked GT — do NOT cite pre-2026-07-17 SAVi-DNO numbers. |
| **SlowFast-VGen** (Temp-LoRA) | short | `lora_experiment/scripts/run_temp_lora_tta.py`, `METHOD=temp_lora` in run_sweep | Config `sweep_experiment/configs/panda_1000v_temp_lora.yaml`. Temp-LoRA fast-learns sequentially (warm-start streams over observed context, then per-rollout-chunk updates on self-generated frames). Uses the standard summary/FVD/VBench plumbing. |
| **TTC** (pathwise correction) | **LONG only** (14+79@14) | `comparison_methods/scripts/ttc_longcat.py` + `sbatch/run_ttc_longcat.sbatch` | Training-free; re-anchors appearance to first frame at low-noise steps. Do NOT report short-horizon. Knobs (sigma-threshold/cadence/weight) are a LongCat adaptation of the T2V TTC paper — tune/validate before citing. `--no-correction` gives the matched baseline arm. |

All three were syntax/dry-run validated 2026-07-17 but NOT yet cluster-run;
smoke-test before full 1000v.

### 2e. `merged_summary.json` schema (CONFIRMED 2026-06-01)

Top-level keys (no nested `metrics` dict):

```
psnr, psnr_std, ssim, ssim_std, lpips, lpips_std
fvd, fvd_num_chunks, fvd_num_videos, fvd_num_ref_videos, fvd_per_chunk
fid, fid_num_frames_gen, fid_num_frames_ref, fid_per_chunk
num_chunks, num_successful, num_videos
vbench, vbench_num_chunks
avg_train_time, avg_gen_time, avg_total_time
```

Per-method `merged_summary.json` lives at:
`{sweep,delta}_experiment/results/<series>/<METHOD>/merged_summary.json`

## 3. Active project state (snapshot — keep current)

**Date:** Updated 2026-08-18.

- **Paper target:** CVPR 2027.
- **Method stack (current):** Wan2.1-T2V-1.3B + Self-Forcing causal DMD,
  I2V continuation. Contribution is a drift-gated GT-free test-time
  controller. Required comparison (same seeds/images/horizon):
  NOTTA | always-BoN | gated-BoN | always-TTC | gated-TTC.
  **Must be chunked** (e.g. 6×5 s on a 30 s rollout). Clip-level gate
  at t=0 is vacuous (incoming context = cond still). LongCat 13.6B
  stays the saturated-large-model audit. Do not launch more LongCat TTC.
- **Wan drift (2026-08-17, N=16):** 5 s median sharp +11% / motion −14%
  (mild). **30 s median sharp +167% / motion −60%** (15/16 each).
  Signature = sharpen + freeze. Table:
  `paper_tables/2026-08-17_wan_i2v_notta16_drift.md`.
- **Wan 16v three-way (2026-08-17):** last-chunk NOTTA 4.43 / always
  3.23 / gated 3.38. Search works. Gated vs always is **not** a
  quality win (mean +0.152, 6/16 better-or-tie); median slightly
  favors gated; always-on hurt 2/16. Honest line: efficiency
  controller that keeps most of the search gain. No TTC yet.
- **Wan hybrid 32v (2026-08-17):** cite medians, not means (video 26
  = 85.6). Last-chunk median NOTTA 3.68 / always 2.97 / gated 3.04.
  gated−always −0.041 / 0, 19/32 better-or-tie, **33% cheaper**.
  First-16 hybrid flipped T=2.0 +0.15 → −0.12. Efficiency on the
  **handcrafted score only** — official VBench not scored yet. Table:
  `paper_tables/2026-08-17_wan_i2v_bon32_hybrid.md`.
- **Wan sticky 32v (2026-08-18):** 03/24 caught (exact ties with
  always-search). 21/32 exact ties overall. Wall 256 vs 258 s —
  spent the hybrid 33% saving. Erased hybrid wins on 11 and 16.
  Not a quality win. Hybrid remains the efficiency method. 11/16
  diagnosis: hybrid slept after recovery; stay-on rebuilt
  always-search while the pick-score lied about the tail. Next
  lever: search-while-sick (turn off on recovery). Table:
  `paper_tables/2026-08-18_wan_i2v_11_16_diagnosis.md`.
- **Wan search-while-sick (2026-08-18):** `--gate-sticky` plus
  `--gate-sick-min 1.0` / `--gate-recovery 0.5`. Job **15959146**
  submitted (Priority). Briefing:
  `paper_tables/2026-08-18_wan_controller_briefing.md`. No TTC.
- **Official outcome eval (2026-08-18, paper-blocking):** controller
  stays GT-free at decision time; finished mp4s must be scored with
  VBench quality dims. No PSNR on these 32 stills (no 30 s GT).
  Submit `wan_experiment/sbatch/submit_i2v_vbench_hybrid32.sh` after
  cluster pull. Spec:
  `paper_tables/2026-08-18_wan_i2v_official_eval_spec.md`.
- **LongCat audit (closed):** short-horizon in-domain 14→14 saturated;
  native AR long-horizon drifts; AdaSteer delta + routing closed;
  BoN k=4 N=8 passed credibility gate as always-on search, not a hard
  incoming-context gate.
- **In-flight cluster jobs** (as of 2026-08-18 11:39): **15959146**
  `i2v_bon_32v_sick` (Priority). Official VBench hybrid job not
  submitted until cluster `git pull` + `submit_i2v_vbench_hybrid32.sh`.

## 4. Daily-log template

When creating a new `experiment_outputs/YYYY-MM-DD.md`, use this header:

```markdown
# Experimental Outputs — YYYY-MM-DD

This file accumulates every cluster output the user pasted on this date,
plus 1-3 line interpretation of each. Raw blocks are preserved verbatim
so future agents can re-analyze.

---

## HH:MM — <short title>

**Source:** <user paste / cluster command that produced it>
**Run:** <jobID or series_name if known>

```
<RAW OUTPUT HERE>
```

**Interpretation:** <1-3 lines>
**Action taken:** <what we did with this data>

---
```

## 5. Subagent push-template (copy-paste-ready)

When pushing local changes via a `shell` subagent, use this prompt skeleton
(fill in the file list and commit message):

```
You are running shell commands on macOS to push <N> files to the GitHub repo
https://github.com/FifthEpoch/longcat-video-tta.git on the `main` branch. The
user's local clone at /Users/macrohard/Desktop/longcat-video-tta is on iCloud
Drive and chronically hits UF_DATALESS / ETIMEDOUT errors during git ops, so
you MUST do everything inside /tmp.

Files to push:
  1. <relative/path/from/repo/root>
  2. ...

Steps:
1. WORK=$(mktemp -d -t longcat-push-XXXXX) && cd "$WORK" && \
   git clone --depth=1 https://github.com/FifthEpoch/longcat-video-tta.git repo && \
   cd repo
2. mkdir -p <dirs needed> && \
   cp /Users/macrohard/Desktop/longcat-video-tta/<file1> <file1> && \
   cp ...
3. git status (confirm only expected files appear)
4. git add <file list>
5. Write commit message to /tmp/msg.txt with cat <<'MSGEOF' (do NOT inline
   heredoc into `git commit -m "$(cat <<...)"` — the wrapper shell breaks it)
6. git commit -F /tmp/msg.txt
7. git push origin main
8. git log -3 --pretty=format:"%h %s"
9. Cleanup
```

## 6. Where the user is in the meeting cycle

- **Mondays:** weekly recap meeting with PhD partner (next: today, 2026-06-01).
  Each Monday a fresh `weekly_recap_YYYY-MM-DD.md` is generated.
- **Thursdays/Fridays:** PI updates as needed.
- **Paper deadline (target):** CVPR 2027 submission window (~Nov 2026).
