# AGENTS.md — persistent index & workflow rules

**Purpose:** This file is the canonical entry point for any AI agent (Claude,
Cursor, etc.) picking up work on this project. Read it FIRST before any
substantive task. Update it whenever a new persistent artifact is created.

---

## 1. Persistent files & where to find them

| What | Path | Notes |
|---|---|---|
| **This index file** | `AGENTS.md` | Updated as artifacts are added |
| **Weekly recap (current week)** | `weekly_recap_YYYY-MM-DD.md` | One per Monday meeting. Latest: `weekly_recap_2026-06-01.md` |
| **Daily experimental-output log** | `sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md` | Append every pasted output (raw + interpretation) |
| **Canonical results memory** | `sweep_experiment/reports/experiment_metrics_log.md` | Long-form running log. Often dehydrated locally (iCloud). For edits, pull from git via subagent. |
| **Paper draft** | `sweep_experiment/reports/paper_draft.md` | LaTeX-aligned narrative + result placeholders. Often dehydrated locally. |
| **Paper LaTeX** | `paper/main.tex`, `paper/sections/*.tex`, `paper/refs.bib` | Real submission source |
| **Run registry** | `experiment_tracker/run_registry.yaml` | Job-ID ↔ result-dir mapping |
| **Cluster repo root** | `/scratch/wc3013/longcat-video-tta/` | All results & raw data live here. Local repo is mostly views. |

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
   weekly recap table.
4. If a new fact emerges that future agents must know (a path, a bug, a
   workflow change), add it to AGENTS.md.

If the date file doesn't exist yet, create it with the standard header
template in §4.

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

**Date:** Updated 2026-06-01.

- **Paper target:** CVPR 2027.
- **Headline finding (today):** AdaSteer is competitive (not net-positive) on
  in-domain in-distribution generation. Wins come from OOD (UCF) and
  retrieval-augmented settings. See `weekly_recap_2026-06-01.md` for the
  latest pivot.
- **In-flight cluster jobs** (as of 2026-06-01 15:00 EDT):
  - Phase 2B Panda segment-pool build (9970342) — pending `Priority`
  - 20 UCF K_RAND retrieval chunks (9965102–9965122) — 2 running, 18 queued
  - All other RAND chunks gated on `QOSMaxGRESPerUser` (~2 concurrent GPU cap)
- **Pending decisions:** whether to add a `K5_SHUFFLED` true-random UCF
  control (~1 h of code); whether to fix the UCF PSNR/SSIM NaN bug before
  trusting per-frame UCF results.

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
