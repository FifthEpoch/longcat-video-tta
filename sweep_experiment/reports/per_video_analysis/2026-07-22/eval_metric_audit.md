# Evaluation-metric audit — fairness & generated-only-window check (2026-07-22)

**Scope:** verify every evaluation metric used to compare TTA arms vs NO-TTA is
(1) fair / apples-to-apples, and (2) scores **only the generated portion**
`video[48:62]` — no leakage of the conditioning frames `video[34:48]` or the TTA
training region `video[0:48]`.

**Geometry (all arms):** `gen_start_frame=48`, `num_cond_frames=14`,
`num_frames=28` → `num_gen=14`. The pipeline rounds up to a VAE-valid length
(`num_frames_valid=29`), so `generate_video_continuation` returns **29** frames =
`[14 cond | 15 gen]`. Saved mp4s therefore contain 29 frames.

**Code paths audited:** `delta_experiment/scripts/common.py`
(`generate_video_continuation`, `evaluate_generation_metrics`,
`OnlineFrechetAccumulator`), `sweep_experiment/scripts/eval_fvd.py`,
`sweep_experiment/scripts/precompute_gt_features.py`,
`sweep_experiment/scripts/merge_chunks.py`,
`sweep_experiment/scripts/eval_vbench.py`, `scripts/run_vbench_backfill.py`,
`delta_experiment/scripts/run_delta_a.py`, `lora_experiment/scripts/run_full_tta.py`.

---

## Summary verdict

| Metric | Scored window | Reference | Leakage? | Fair (apples-to-apples)? | Verdict |
|---|---|---|---|---|---|
| **PSNR / SSIM / LPIPS** | `gen_output[14:28]` = `video[48:62]` (gen-only) | source `video[48:62]` | **No** | Yes (same window, same GT, paired) | ✅ CLEAN |
| **FVD** | `gen_output[14:28]` = `video[48:62]` (gen-only) | GT `video[48:62]` (paired online, or frozen cache) | **No** | **No** — N/subset/provenance mismatch | ⚠️ WINDOW OK, COMPARISON BROKEN |
| **FID** | per-frame over `gen_output[14:28]` | GT `video[48:62]` | **No** | **No** — same N caveat as FVD | ⚠️ WINDOW OK, COMPARISON BROKEN |
| **VBench++** | **entire mp4 = `[14 cond | 15 gen]` (29 frames)** | n/a (no-reference / first-frame) | **YES — includes 14 real cond frames** | Partially (same cond both arms, but signal contaminated) | ❌ WINDOW WRONG |
| TTA training region | trains on `video[0:48]` | — | disjoint from scored `video[48:62]` (clamped) | — | ✅ NO TRAIN→EVAL LEAKAGE |

---

## 1. Pixel metrics — PSNR / SSIM / LPIPS  ✅ CLEAN

`evaluate_generation_metrics` (common.py) slices
`gen_frames = gen_output[num_cond_frames : num_cond_frames + num_gen_frames]`
(= the 14 generated frames), and reads GT by decoding the source video, skipping
`gen_start_frame` frames, then taking `num_gen_frames` → `video[48:62]`, LANCZOS
resized to the gen resolution. Per-frame PSNR/SSIM/LPIPS then averaged.

- **Window:** generated-only. Conditioning frames are dropped on the gen side;
  GT starts exactly at the anchor (48), i.e. the true unseen future.
- **Leakage:** none. The scored GT `video[48:62]` was never shown to the model
  (conditioning is `video[34:48]`), and never used in TTA (`video[0:48]`).
- **Fairness:** NO-TTA (`run_full_tta.py`) and TTA (`run_delta_a.py`) call the
  identical function with identical geometry and the same per-video GT, on the
  same paired video set. Seed-clean pool: N=898 paired.

**No action needed.**

## 2. FVD  ⚠️ window correct, comparison broken

- **Window (correct):** `OnlineFrechetAccumulator.update` feeds I3D exactly
  `gen_output[14:28]` = `video[48:62]`. The GT cache
  (`precompute_gt_features.py`, `load_gt_frames_longcat(gen_start=48,
  num_gen=14)`) is `video[48:62]`. Offline `eval_fvd.py` skips the first 14
  frames of each saved mp4 and scores the tail. **All three agree — gen-only, no
  leakage.**
- **Merge (correct):** `merge_chunks.py` sums per-chunk sufficient stats
  (`gen_sum/gen_cov/gen_count`, `ref_sum/ref_cov/ref_count`) — a valid
  running-moment merge.
- **Comparison (BROKEN):** every existing FVD comparison mismatches sample size,
  subset, reference protocol, or provenance, and the apparent TTA effect **flips
  sign**:
  - Online `merged_summary.json` (per-video paired ref, `gt_cached=n/a`):
    NO-TTA **fvd=157.0 @ N=375**; configs **66–69 @ N=998** → looks like TTA
    *halves* FVD.
  - Offline matched job (frozen preview cache, saved mp4s):
    `always_notta=81.5 @ N=898`, `fixed=198 @ N=998`, `oracle=168 @ N=998` →
    looks like TTA *doubles* FVD.
  - Sign flip ⇒ neither measures a real effect. FVD is strongly small-N biased,
    so 157 @ 375 vs 66 @ 998 is largely the N gap.
- **Data-integrity bug:** NO-TTA online FVD accumulated only **375 of ~969**
  videos. Per-chunk `fvd_num_videos` = 45/41/61/17/63/–/64/60/16/8 and
  **`chunk_5`'s summary is missing** (never merged). This is a run/merge
  completion problem specific to the NO-TTA re-run, not a windowing bug.
- **Stale number:** the oracle-analysis FVD row **383.9** is the old N=200 pilot
  computed against `panda_1000_longcat.npz` — the WRONG reference for this pool
  (the preview sbatch flags this explicitly). Ignore it.

**Fix:** one matched offline recompute — score ALL arms from saved mp4s
(NO-TTA 969, configs 1000) against the frozen preview cache
`gt_caches/panda_ood_budget_1000v_preview_longcat.npz`, restricted to the common
video-ID set (`INTERSECT_NOTTA=1` → `--intersect-with-notta`) so N, reference,
and provenance are identical across arms.

## 3. FID  ⚠️ same as FVD

Per-frame InceptionV3 over the same generated slice `gen_output[14:28]`; GT is
`video[48:62]` frames from the cache. Window is gen-only (no leakage). Same
matched-N requirement as FVD for a fair comparison.

## 4. VBench++  ❌ window wrong (evaluates cond + gen)

Both `eval_vbench.py` and `scripts/run_vbench_backfill.py` pass the **entire
saved mp4 directory** to `VBench.evaluate(videos_path=..., mode=custom_input)`.
The saved mp4 is `[14 cond | 15 gen]` = 29 frames, and **nothing trims the
conditioning frames**. So every per-video VBench score in the analysis was
computed over **~14 real conditioning frames + ~15 generated frames**, i.e. the
metric is roughly half real-input content.

Consequences:
1. **Violates the generated-only requirement.** Unlike pixel/FVD (gen-only),
   VBench includes the real input prefix.
2. **Contaminated absolute values.** Dimensions like `subject_consistency`,
   `background_consistency`, `motion_smoothness`, `temporal_flickering`,
   `dynamic_degree` are averaged over frames that are ½ real footage, inflating
   them toward real-video quality.
3. **Compressed / muddied TTA-vs-NO-TTA signal.** The identical real prefix is a
   large shared component in both arms, shrinking measured differences (consistent
   with the tiny ~0.06 overall VBench gains seen). Worse, the mp4 "cond" region is
   the VAE reconstruction under each arm's (possibly TTA-perturbed) weights, so it
   is not even guaranteed identical across arms — the prefix can itself differ.
4. The `winner_dim=dynamic_degree (+5.16%)` result is over cond+gen, so it mixes
   real-footage motion with generated motion and cannot be read as
   generated-only headroom.

**Fix:** trim the first `num_cond_frames` (14; or 15 to also drop the extra
VAE-rounding frame — decide once and document) from each saved mp4 into a
`videos_geneval/` dir, then re-run `run_vbench_backfill.py` on the trimmed clips
for all arms. Re-derive per-video VBench and re-run the VBench oracle/router
analysis on the gen-only scores. (Check each dimension's minimum-frame
requirement; ~14–15 frames is sufficient for the dims we use, but confirm RAFT-
based `dynamic_degree` / `motion_smoothness` behave on 14 frames.)

## 5. Train→eval leakage  ✅ none

`run_delta_a.py` clamps `tta_total_frames` to `gen_start_frame` (explicit
"avoid GT leakage" guard), so TTA trains on `video[0:48]` while scoring uses the
disjoint `video[48:62]`. The conditioning window `video[34:48]` is the model's
input, not a scored target. No train/condition frames enter any scored metric on
the pixel/FVD/FID side.

---

## Action items (priority order)

1. **VBench (correctness):** regenerate gen-only clips (trim cond frames) and
   re-run VBench backfill for NO-TTA + all 12 configs; re-derive per-video VBench
   and re-run the VBench oracle/routing analysis. This is the substantive fix —
   current VBench numbers are cond-contaminated.
2. **FVD/FID (fairness):** run the matched offline recompute
   (`INTERSECT_NOTTA=1 sbatch run_preview_1000v_matched_fvd.sbatch`) so all arms
   share N/reference/provenance; then retire the online 375-video NO-TTA FVD and
   the stale 383.9 row.
3. **Data integrity:** re-accumulate / re-merge NO-TTA FVD (recover `chunk_5`,
   complete the ~525 missing videos) OR simply rely on the offline recompute from
   the 969 saved mp4s.
4. **Pixel metrics:** no change — already gen-only, fair, leakage-free.
