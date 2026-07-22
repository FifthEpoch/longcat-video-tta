# Eval-metric fix + full recompute runbook (2026-07-22)

Single ordered runbook to fix the evaluation bugs found in the metric audit
(`eval_metric_audit.md`, same folder) and recompute **everything** on the
seed-clean 1000v preview series
(`sweep_experiment/results/panda_ood_budget_1000v_preview`). Run top-to-bottom
on the cluster. Do not skip steps; each later step depends on earlier ones.

## What was broken and what the fix does

| Metric | Bug | Fix |
|---|---|---|
| **VBench++** | Scored the **entire mp4** `[14 cond \| 15 gen]` (29 frames). ~half the score was real conditioning footage → contaminated absolute scores and muddied TTA-vs-NOTTA. | Trim the first 14 (cond) frames → `videos_geneval/` (15 gen frames), re-run VBench on those into `vbench_results_geneval/`. |
| **FVD/FID** | Comparisons broken: NOTTA online FVD accumulated only ~375/969 videos (chunk_5 missing); configs used per-video paired refs while offline used a frozen cache; N mismatch (898 vs 998). Window itself was correct (`video[48:62]`). | One offline recompute for **all 3 policies** vs the **same frozen preview GT cache**, intersected to the **common video set** (`INTERSECT_NOTTA=1`). |
| **PSNR/SSIM/LPIPS** | Clean — already score gen-only `video[48:62]`, seed-matched, paired. | No change. |
| **TTA train leakage** | None — TTA trains on `video[0:48]`, disjoint from scored `video[48:62]`. | No change. |

Geometry (fixed everywhere): `gen_start=48`, `num_cond_frames=14`,
`num_gen_frames=14`. Saved mp4 = 29 frames (14 cond + 15 gen). Pixel/FVD score
`[14:28]`; gen-only VBench scores the 15-frame tail `[14:29]`.

Frame-exactness of the trim was verified locally on a synthetic 29-frame clip:
`out.shape==15`, `out[0]==src[14]`, `out[-1]==src[28]`.

## Code changes (already pushed)

- `scripts/make_geneval_clips.py` **(new)** — trims cond frames → `videos_geneval/`,
  encoded identically to the pipeline writer (imageio + libx264, quality=9).
- `scripts/run_vbench_backfill.py` — `--videos-subdir` / `--out-subdir`.
- `scripts/analyze_per_video_vbench_agreement.py` — `load_per_video_vbench`
  honors `VBENCH_SUBDIR` env (redirects **all** analysis consumers to gen-only).
- `scripts/update_merged_with_vbench.py` — `--vbench-subdir` / `--videos-subdir`
  / `--deprecate-existing` (stashes old full-clip means under
  `vbench_fullclip_deprecated`, rebuilds `vbench` from gen-only).
- `run_vbench_backfill.sbatch` + `submit_budget_1000v_preview_vbench_backfill.sh`
  — `VIDEOS_SUBDIR`/`OUT_SUBDIR` passthrough, `GENEVAL=1` shortcut, NOTTA added.

---

## Runbook

```bash
cd /scratch/wc3013/longcat-video-tta
git pull --ff-only origin main

SERIES=sweep_experiment/results/panda_ood_budget_1000v_preview
ARMS="NOTTA S2_LR1e3 S2_LR5e3 S2_LR1e2 S5_LR1e3 S5_LR5e3 S5_LR1e2 \
S10_LR1e3 S10_LR5e3 S10_LR1e2 S20_LR1e3 S20_LR5e3 S20_LR1e2"
```

### Step 1 — Trim gen-only clips for all 13 arms (longcat env; CPU, fast)

```bash
conda activate /scratch/wc3013/conda-envs/longcat
for a in $ARMS; do
  echo "=== trim $a ==="
  python3 scripts/make_geneval_clips.py --method-dir "$SERIES/$a" --num-cond-frames 14
done
# sanity: every videos_geneval mp4 must have exactly 15 frames
f=$(find "$SERIES/NOTTA" -path '*/videos_geneval/*.mp4' | head -1)
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of csv=p=0 "$f"   # -> 15
```

### Step 2 — VBench on GEN-ONLY clips, all 13 arms (vbench-backfill env; GPU)

```bash
# submits one job per arm; reads videos_geneval/, writes vbench_results_geneval/
GENEVAL=1 FORCE=1 bash sweep_experiment/sbatch/submit_budget_1000v_preview_vbench_backfill.sh
# wait for all 13 to finish:
squeue -u wc3013 | grep vb_prev1k
```

### Step 3 — Fold gen-only VBench into merged_summary (deprecate old), all arms

```bash
conda activate /scratch/wc3013/conda-envs/longcat
for a in $ARMS; do
  python3 scripts/update_merged_with_vbench.py \
    --method-dir "$SERIES/$a" --force --deprecate-existing \
    --vbench-subdir vbench_results_geneval --videos-subdir videos_geneval
done
# spot-check: old means preserved, new gen-only means in place
jq '.vbench, (.vbench_fullclip_deprecated // "none")' "$SERIES/NOTTA/merged_summary.json"
```

### Step 4 — FVD/FID matched offline recompute (frozen cache, common N; GPU)

```bash
# 3 policies (always_notta / fixed_S10_LR5e3 / oracle_best_psnr) on the SAME
# common video set, ONE frozen preview GT cache. GT cache already exists.
INTERSECT_NOTTA=1 SKIP_GT_CACHE=1 \
  sbatch sweep_experiment/sbatch/run_preview_1000v_matched_fvd.sbatch
# when done:
BASE=sweep_experiment/reports/budget_oracle_fvd_1000v_preview
cat "$BASE/matched/pilot_matched_fvd_summary.md"
for p in matched/always_notta matched/fixed_S10_LR5e3 matched/oracle_best_psnr; do
  jq -r '"\(input_filename): fvd=\(.fvd) N=\(.num_valid_pairs)"' "$BASE/$p/fvd.json"
done   # all three N must be EQUAL (matched)
```

### Step 5 — Re-run ALL vs-NOTTA analyses on the corrected data

`VBENCH_SUBDIR=vbench_results_geneval` redirects every per-video VBench consumer
(oracle, router matrix, chart dumper) to the gen-only results.

```bash
export VBENCH_SUBDIR=vbench_results_geneval
D=$(date +%Y-%m-%d)
OODCSV=sweep_experiment/reports/per_video_analysis/2026-07-10/diffusion_ood_scores_segment_pool.csv

# 5a. PSNR + VBench oracle uplift (in-series seed-clean NOTTA baseline)
python3 scripts/analyze_adasteer_budget_oracle.py --bootstrap \
  --series-root "$SERIES" --baseline-series-root "$SERIES" \
  --ood-csv "$OODCSV" \
  --output "sweep_experiment/reports/per_video_analysis/$D/adasteer_budget_oracle_1000v_geneval.md"

# 5b. Full router matrix (blocks A/B/C/ABC x {12,13} x {PSNR,VBench}), gen-only VBench
python3 scripts/run_router_full_matrix.py \
  --series-root "$SERIES" --metrics psnr vbench \
  --output-dir "sweep_experiment/reports/per_video_analysis/$D/router_full_matrix_1000v_geneval"

# 5c. Chart data JSON (winner VBench dim, OOD-quintile deltas) — gen-only.
#     (dump script prints JSON to stdout; baseline-series-root=SERIES -> in-series NOTTA)
python3 scripts/dump_pilot_chart_data.py \
  --series-root "$SERIES" --baseline-series-root "$SERIES" --ood-csv "$OODCSV" \
  > "sweep_experiment/reports/per_video_analysis/$D/chart_data_1000v_geneval.json"
```

Paste `chart_data_1000v_geneval.json` back and the charts get re-rendered locally
with `scripts/render_pilot_charts_from_json.py --label "1000v (gen-only, seed-clean)"`.

### Step 6 — Verification gate (all must pass before citing any number)

```bash
# (a) gen-only clips are 15 frames everywhere
for a in $ARMS; do
  bad=$(for f in $(find "$SERIES/$a" -path '*/videos_geneval/*.mp4'); do
    n=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "$f"); [ "$n" = 15 ] || echo "$f:$n";
  done | head -3); [ -n "$bad" ] && echo "$a BAD: $bad" || echo "$a ok(15f)";
done
# (b) merged_summary carries deprecated audit key + gen-only vbench
for a in $ARMS; do
  jq -e '.vbench_fullclip_deprecated and .vbench.aesthetic_quality' \
    "$SERIES/$a/merged_summary.json" >/dev/null && echo "$a vbench ok" || echo "$a vbench MISSING";
done
# (c) matched FVD: identical N across the 3 policies (printed in Step 4)
```

## Notes / gotchas

- Old full-clip VBench (`vbench_results/`) is **kept** for audit; nothing is
  deleted. The deprecated population means live under
  `merged_summary["vbench_fullclip_deprecated"]`.
- If Step 2's `vbench-backfill` conda env is missing:
  `bash scripts/setup_vbench_backfill_env.sh` (once).
- FVD is distribution-level (not per-video) → only the 3-policy comparison is
  meaningful; do not route on it.
- Re-record: after Step 5, add the new dated tables to `INDEX.md` and an
  `ANALYSIS_LOG.md` entry, then push (§2b-bis of AGENTS.md).
