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
| **Pseudo-future Search note** | `sweep_experiment/reports/paper_tables/2026-08-25_pseudo_future_search.md` | **DROPPED as paper title 2026-09-04.** Code `sf_pseudo` stays for the cite-128 ablation. |
| **Drop Pseudo + next territories** | `sweep_experiment/reports/paper_tables/2026-09-04_drop_pseudo_next_territories.md` | Outcome atlas. Territories A (new student) / B (analysis) / C (new control). No GPU until the user picks. |
| **Gate neighbors + publishability** | `sweep_experiment/reports/paper_tables/2026-09-01_gate_neighbors_publishability.md` | EFD / SDVG / Video-T1 / CachedSearch / LatSearch. 13% vs Always is not a quality paper. |
| **RF schedule neighbors** | `sweep_experiment/reports/paper_tables/2026-09-01_rf_noise_schedule_neighbors.md` | Deep / Relax / Ms. / Stream / Reward / FIFO. Most RF follow-ons are memory. TTA cousins: lookahead, shallower / local-steep diagonal. |
| **RF non-linear timestep list** | `sweep_experiment/reports/paper_tables/2026-09-01_rf_nonlinear_schedule.md` | **DONE / NO.** Linger / dump Imaging Quality died. Harvest: `2026-09-04_wan_v2v_caption_schedule8_harvest.md`. Do not start 8-GPU DMD. |
| **Schedule8 harvest** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_schedule8_harvest.md` | 16855778–780 COMPLETED. Native floor 556. Linger −10% IQ 66.34. Dump +39% IQ 68.14. Both **NO**. |
| **SF / RF shared experiment machine** | `sweep_experiment/reports/paper_tables/2026-09-04_sf_rf_common_impl.md` | Both papers: unroll inference + holistic DMD. Mix + context noise are diagnostic, not the paper idea. |
| **SF / RF KV + compute audit** | `sweep_experiment/reports/paper_tables/2026-09-04_sf_rf_kv_opt_audit.md` | Quality KV / sink / RoPE / window already on. Do not retune `enlarge_kv` / `local_attn` / extra sink on cite hosts. |
| **Mix + context-noise spec** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_mixctx_spec.md` | Caption N=8. `rf_mix` / `sf_mix` + always + `*_ctx`. Do not remake cite-128. |
| **FIFO + lock-score spec** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_fifo_tscore_spec.md` | Caption N=8. `rolling_fifo` + `rf_tscore` / `sf_tscore`. 1.3B freeze-score, not Wan-14B. |
| **Mix+ctx harvest** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_mixctx_harvest.md` | 16931124–130 COMPLETED. All six **NO**. |
| **FIFO+tscore harvest** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_fifo_tscore_harvest.md` | 16931441–447 COMPLETED. All six **NO**. Gated tscore = host identity. |
| **LPIPS + aligned FVD spec** | `sweep_experiment/reports/paper_tables/2026-09-01_wan_v2v_lpips_fvd_spec.md` | Fill existing pixel jsons. I3D on 30 s tails. **DONE 16738784.** Harvest: `2026-09-04_wan_v2v_cite128_lpips_fvd.md`. |
| **Cite-128 LPIPS/FVD** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_cite128_lpips_fvd.md` | SF 0.745 / 410. Pseudo 0.753 / **405**. Always 0.751 / 425. RF 0.762 / 436; last16 RF **1108**. |
| **Cite-128 full grid (2026-09-04)** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_cite128_all_metrics.md` | VBench + pixels + LPIPS + FVD. Supersedes open cells in the 2026-08-31 grid. |
| **Denoise-hooks spec** | `sweep_experiment/reports/paper_tables/2026-08-28_wan_v2v_denoise_hooks_spec.md` | lastmix / bpseudo / restep. Caption N=8. `WAVE=lastmix` first. |
| **Pseudo-next N=8 spec** | `sweep_experiment/reports/paper_tables/2026-08-31_wan_v2v_pseudo_next8_spec.md` | Cheapen + re-gate. Caption N=8. **NO.** Harvest: `2026-08-31_wan_v2v_pseudo_next8_harvest.md`. |
| **Caption leftover ρ spec** | `sweep_experiment/reports/paper_tables/2026-09-01_wan_v2v_caption_leftovers_spec.md` | Stem leftovers are panda-infected. Caption N=8 ρ / look only. Do not remake cite-128. |
| **Caption leftover harvest** | `sweep_experiment/reports/paper_tables/2026-09-01_wan_v2v_caption_leftovers_harvest.md` | 16734909–913 COMPLETED. All four **NO**. Real captions did not save Imaging Quality. |
| **Cite-128 wall (job/96)** | `sweep_experiment/reports/paper_tables/2026-09-01_wan_v2v_cite128_wall.md` | 108 / 47 / 294 / 354. Not n=32 sidecar 196 / 45 / 304 / 348. |
| **Weekly recap (current week)** | `weekly_recap_YYYY-MM-DD.md` | One per Monday meeting. Latest: `weekly_recap_2026-09-01.md` |
| **Daily experimental-output log** | `sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md` | Append every pasted output (raw + interpretation) |
| **Canonical results memory (legacy)** | `sweep_experiment/reports/experiment_metrics_log.md` | Long-form running log. Superseded by INDEX.md + ANALYSIS_LOG.md as of 2026-06-08, but kept for history. |
| **Paper draft** | `sweep_experiment/reports/paper_draft.md` | LaTeX-aligned narrative + result placeholders. Often dehydrated locally. |
| **Paper LaTeX** | `paper/main.tex`, `paper/sections/*.tex`, `paper/refs.bib` | Real submission source |
| **Run registry** | `experiment_tracker/run_registry.yaml` | Job-ID ↔ result-dir mapping |
| **Cluster repo root** | `/scratch/wc3013/longcat-video-tta/` | All results & raw data live here. Local repo is mostly views. |
| **Wan 1.3B / Self-Forcing setup** | `wan_experiment/README.md` | I2V-32 is **discovery only**. Official VBench **DONE** (full-clip tie). **Do not scale I2V-32.** Current next: V2V Panda bake-off (`2026-08-20_wan_v2v_sampling_bakeoff_spec.md`). T2V 128 is optional. Do **not** add TTC. |

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

### 2b-quater. Jobs that left `squeue` (ADDED 2026-09-01)

`squeue` only shows what is still queued or running. A finished
job vanishes. If a known JobID is missing from the next
`squeue` paste, **immediately** `sacct -j <ID>` and harvest
disk. Do not wait for the user to notice. Caption leftover
16734909–911 left the queue at 15:20 and were not harvested
until the user called it out.

### 2b-ter. Same-wave ablations (ADDED 2026-08-24)

When submitting a **gated** method, the same paste must also
launch the obvious twins: **always-on** (no gate) and the
**other host** if the claim is host-specific. Do not wait for
harvest to invent the ablation. Harvest still decides the call.
k stays locked to the family width (k=4) unless a dated spec
says otherwise. CachedSearch headline N=8 is a later width
sweep, not a harvest retune.

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

**Date:** Updated 2026-09-04.

- **Paper target:** CVPR 2027.
- **Paper method (2026-09-04):** Pseudo-future Search is **dropped**
  as the title. A 13% gate on Always-search is not a CVPR idea.
  Next fork: territories A (new student) / B (analysis paper) /
  C (new frozen-weight control). No GPU until the user picks.
  `paper_tables/2026-09-04_drop_pseudo_next_territories.md`.
- **Method stack (current):** Wan2.1-T2V-1.3B + Self-Forcing causal DMD.
  I2V-32 30 s is a **discovery / stress** run, not the field
  long-horizon table. **Do not scale I2V-32 or I2V-200.** **Do not
  add TTC / LoRA-at-test-time.** LongCat 13.6B stays the
  saturated-large-model audit. Do not launch more LongCat TTC.
- **Protocol stop (2026-08-18):** recent long-horizon papers on this
  model family are **T2V** self-continuation, **128 MovieGen**
  (Qwen-refined), **VBench-Long**, 30 s / 60 s. Ours was I2V-from-still,
  N=32, `custom_input`. Length 30 s and Wan 1.3B are fine; task / N /
  suite are not. Stop:
  `paper_tables/2026-08-18_wan_protocol_stop.md`.
- **Task lock (2026-08-18, user correction):** T2V was **not** agreed.
  I2V-from-still scale-up stays closed. **V2V prefix-continuation is
  allowed** and is the closer match to the claim (visual history →
  long AR). T2V 128 MovieGen is only an optional comparison to Relax
  Forcing–style tables. Note:
  `paper_tables/2026-08-18_v2v_continuation_allowed.md`.
- **Optional T2V compare (SUBMIT-READY, not launched):** T2V 30 s,
  128 MovieGen, do-nothing | always-BoN | gated-BoN. New runner
  `wan_experiment/scripts/run_t2v_chunked.py`. Submit:
  `SMOKE=1 bash wan_experiment/sbatch/submit_t2v_bon128.sh` then
  `bash wan_experiment/sbatch/submit_t2v_bon128.sh`. Spec:
  `paper_tables/2026-08-18_wan_t2v_vbenchlong_128_spec.md`.
- **V2V caption bug (2026-08-24):** Panda pool had
  `metadata.csv` (1000/1000 list captions). The runner only
  loaded JSON, so every finished V2V arm used filename stems
  (`panda 0013`). Real 0013 caption is a bathroom stain.
  Tail→panda is T5 takeover. Same-prompt deltas still hold.
  Runner now reads `metadata.csv`. Caption WAVE=1
  **protocol PASS** (`prompt_source=metadata_csv`, 0 stem).
  WAVE=1 generate **32/32** all arms. Always **16310324**
  COMPLETED tail **+39%** 30/2 (Pseudo +28% 23/0/9). VBench
  **16310330** still R; notta/rolling/rewind written (SF
  0.700/71.54/0/0.989; RF IQ −1.32 vs SF). AdaSteer N=8
  **NO** (16326033–036 COMPLETED): `|δ|`≈0.84, IQ 43/51/18.
  Do not scale AdaSteer. Do not mix stem-prompt numbers into
  caption tables.
  Outcomes: `paper_tables/2026-08-24_wan_v2v_caption_wave1_outcomes.md`.
  Spec: `paper_tables/2026-08-24_wan_v2v_caption_rerun_spec.md`.
- **Current next (2026-09-04):** User picks A / B / C. Do not
  cheapen Pseudo. Do not scale mix / FIFO / tscore / ρ / list.
  Caption official (historical):
  N=32 **DONE**. Cite Dyn as **percent of clips** (VBench official),
  not median. SF 21.9% (7/32), Pseudo **40.6%** (13/32), Always
  **43.8%** (14/32). `rf_sink` 0.709 / 70.15 / 15.6% / 0.980.
  Prefix-match NO. AdaSteer N=8 **NO**. Table:
  `paper_tables/2026-08-25_wan_v2v_caption_dyn_percent.md`.
  Method note (name + related work + intra-chunk hole):
  `paper_tables/2026-08-25_pseudo_future_search.md`.
  Paper name: **Pseudo-future Search** (code `sf_pseudo`).
  In-chunk: scored arms **NO** (lastmix / sf_bpseudo / rf_restep
  identity). SF intra + SF restep + RF bpseudo still **FAILED**.
  Caption-128 hosts **DONE**: SF 0.666 / 72.07 / **Dyn 32.8%
  (42/128)**; rolling +33% tail / 0.685 / 71.52 / Dyn **28.9%**.
  Cite 128 **COMPLETE**: Pseudo tail **0.0157 ≈ RF 0.0158**;
  official **Dyn 47.7% (61/128)** / 0.660 / 72.38. Always
  **50.8% (65/128)** / 0.661 / 72.19. Gate **90 fire / 38 skip**
  costs 4 Dyn clips. SF 32.8% (42). RF 28.9% (37).   Pixel PSNR/SSIM **DONE** (**16702323**): SF **9.25** /
  RF 7.98 / Pseudo **9.22** / Always **9.21**. Search ≈ SF
  on reconstruction; Rolling −1.28 dB. Headline stays
  VBench + Dyn%. Mid-chunk rewrite **CLOSED**. Pseudo-next
  N=8 **NO** (CachedSearch slower; re-gate no lift). Tables:
  `paper_tables/2026-08-31_wan_v2v_cite128_complete.md`,
  `paper_tables/2026-09-04_wan_v2v_cite128_all_metrics.md`
  (VBench 7/7; pixel suite **DONE**),
  `paper_tables/2026-09-04_wan_v2v_cite128_lpips_fvd.md`,
  `paper_tables/2026-09-01_wan_v2v_cite128_pixel.md`,
  `paper_tables/2026-08-31_wan_v2v_pseudo_next8_harvest.md`,
  `paper_tables/2026-08-31_wan_v2v_keep_intra_closed.md`.
  GPU: `paper_tables/2026-08-23_wan_gpu_batch_policy.md`.
  Success bar + neighbors (2026-08-30): RF quality, cost <<
  always-search; mid-chunk rewrite closed.
  `paper_tables/2026-08-30_wan_success_and_neighbors.md`.
  Beat-RF path (2026-08-30): not seed search. Intervene at
  window-exit (context noise / next-block noise / softer sink).
  `paper_tables/2026-08-30_wan_rf_intervene.md`.
  RF descendants (2026-09-01): Deep / Relax / Reward /
  Forcing-KV rewrite **memory**. Stream Forcing and
  Ms. Forcing are the schedule papers (need a student).
  TTA-legal cousins: FIFO lookahead, shallower diagonal,
  local next-block bump, context noise. Note:
  `paper_tables/2026-09-01_rf_noise_schedule_neighbors.md`.
  Next on Pseudo (2026-08-31): CachedSearch / re-gate **NO**.
  Gate is almost free; cheapen is still the paper move, but
  **not** this CPU KV snap. Search-early or prune k. Or RF
  window-exit. Pixel skip-existing **landed**. LPIPS/FVD
  **DONE 16738784**: SF 0.745 / 410; Pseudo 0.753 / **405**;
  Always 0.751 / 425; RF 0.762 / 436 (last16 **1108**).
  Caption leftover **NO**. Schedule8 linger/dump **NO**.
  Mix+ctx **NO**. FIFO **NO**. Gated lock-score = host identity.
  Pseudo-future Search **dropped as title**. Distill is
  territory A if the user picks it — do not start 8-GPU DMD
  tonight. Harvests:
  `paper_tables/2026-09-04_wan_v2v_caption_mixctx_harvest.md`,
  `paper_tables/2026-09-04_wan_v2v_caption_fifo_tscore_harvest.md`,
  `paper_tables/2026-09-04_drop_pseudo_next_territories.md`.
- **N=32 leftover (closed):** `appear_bon` NO. `rolling_notta` YES
  on locked tail+quality bars (Dyn 0). Host, not our controller.
  Verdict: `paper_tables/2026-08-22_wan_v2v_forward32_verdict.md`.
- **Next methods (no weights):** motion verifier + `{shift,cfg}` probe
  + prefix backtrack now live on V2V. CachedSearch / sink / HG-f wait.
  Memo: `paper_tables/2026-08-18_wan_nonweight_next.md`.
- **Week briefing (2026-08-18):** model + dataset switch with citations,
  plus long-horizon concepts. Setup talk, not the method talk.
  `paper_tables/2026-08-18_week_switch_briefing.md`.
- **Methods-since-switch talk (2026-08-24):** every widget, question,
  hypothesis, papers, gates, and real N=32 numbers. Anyone-readable.
  `paper_tables/2026-08-24_wan_methods_since_switch.md`. Canvas:
  `wan-methods-since-switch`.
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
  **handcrafted score only**. Official VBench (full clip) is a tie —
  see the Official VBench bullet. Table:
  `paper_tables/2026-08-17_wan_i2v_bon32_hybrid.md`.
- **Wan sticky 32v (2026-08-18):** 03/24 caught (exact ties with
  always-search). 21/32 exact ties overall. Wall 256 vs 258 s —
  spent the hybrid 33% saving. Erased hybrid wins on 11 and 16.
  Not a quality win. Hybrid remains the efficiency method. 11/16
  diagnosis: hybrid slept after recovery; stay-on rebuilt
  always-search while the pick-score lied about the tail. Next
  lever: search-while-sick (turn off on recovery). Table:
  `paper_tables/2026-08-18_wan_i2v_11_16_diagnosis.md`.
- **Wan search-while-sick (2026-08-18):** Job **15959146 DONE**.
  Checklist pass on the handcrafted score. Median 2.764 vs always
  2.966 / hybrid 3.036. 11/16 recovered, 24 exact always, wall 204 s.
  9/14/9 — not a strict quality win. Table:
  `paper_tables/2026-08-18_wan_i2v_bon32_sick.md`. No TTC.
- **VBench protocol (locked 2026-08-18):** always score the **full
  generated clip**. That is the comparable number. last5 is optional
  diagnostic only — never the paper’s “VBench++” table. Defaults:
  `CLIPS=full last5`, analyze `--clip full`.
- **Official VBench (2026-08-18, DONE, hybrid 32):** full-clip is a
  tie (Aes 0.587/0.593/0.591, IQ 71.24/71.28/71.19, dynamic median
  0). last5 IQ drop is diagnostic only. Verifier anti-aligned with
  IQ on last5 (ρ +0.23 to +0.33). Read:
  `paper_tables/2026-08-18_wan_i2v_bon32_vbench_read.md`. No PSNR.
  No TTC.
- **LongCat audit (closed):** short-horizon in-domain 14→14 saturated;
  native AR long-horizon drifts; AdaSteer delta + routing closed;
  BoN k=4 N=8 passed credibility gate as always-on search, not a hard
  incoming-context gate.
- **In-flight cluster jobs** (as of 2026-09-04 13:11):
  Mix+ctx **16931124–130 DONE / NO**. FIFO+tscore
  **16931441–447 DONE / NO**. Caption leftover **DONE / NO**.
  LPIPS+FVD **16738784 DONE**. Schedule8 **DONE / NO**.
  Do not remake cite-128. Do not WAVE=3. Do not start 8-GPU DMD.
  **No I2V. No TTC.**
- **VBench 5 s windows (DONE 16009916):** hybrid 32. Aes 0.651→0.538,
  IQ 72.9→68.1 (do-nothing). Search does not reverse it. Dynamic
  median 0 every window. Full clip stays official.
  `paper_tables/2026-08-19_wan_i2v_bon32_vbench_trend.md`
  All 7 dims in one grid:
  `paper_tables/2026-08-19_wan_i2v_bon32_vbench_alldims.md`
- **VBench 16v 5 s vs 30 s + first16/last16 (DONE 16010032):**
  **Cite entire clips:** 5 s full vs 30 s full, subject 0.932→0.842.
  `paper_tables/2026-08-19_wan_i2v_notta16_vbench_fullclip.md`
  16-frame VBench does **not** copy handpicked sharp/motion. Only
  aesthetic Δrel matches “30 s worse” (−11.5% vs +1.8%).
  `paper_tables/2026-08-19_wan_i2v_notta16_vbench_headtail.md`
  Read: `paper_tables/2026-08-19_wan_i2v_vbench_windows_read.md`.

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
