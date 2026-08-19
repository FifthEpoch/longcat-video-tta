# Wan I2V official outcome eval — spec (no numbers yet)

**Date:** 2026-08-18.
**Why this exists:** the controller must stay ground-truth-free at
*decision* time. That does **not** excuse scoring the *finished videos*
only with the handcrafted composite. Videos 11/16 already showed the
pick-score can report a win while the last second dies. We will not
know if do-nothing / always-search / gated-search are improving by
common standards until those mp4s are judged with official metrics.

---

## Split that is now locked

| When | Allowed to see GT / official metrics? |
|---|---|
| Gate fire / candidate pick (controller loop) | **No.** Incoming / score / outgoing only. |
| After the mp4 is written (outcome eval) | **Yes.** VBench quality dims, later I2V dims. |

The handcrafted score is a **controller signal**, not the paper's
quality claim. If Spearman(last-chunk, VBench) is near zero, the
signal is not detecting drift that helps standard performance.

---

## What we cannot compute on these 32 clips

The current set is **VBench-I2V stills + captions**. There is no
paired 30-second real video.

Do **not** report PSNR / SSIM / LPIPS / FVD against “the real
continuation.” Those numbers do not exist here.

A later optional audit can point Wan at Panda-70M / UCF prefixes
(those GT videos already live under the LongCat eval sets). That is
a new generate series, not a rescoring of the current 32.

---

## What we will compute

Official VBench quality dimensions (same family as the Panda 1000v
tables), in `custom_input` mode, `vbench-backfill` env:

| Dimension | Why it is here |
|---|---|
| `subject_consistency` | identity over the clip |
| `background_consistency` | scene hold |
| `aesthetic_quality` | LAION aesthetic |
| `imaging_quality` | MUSIQ; maps to our sharpen story |
| `motion_smoothness` | AMT smoothness |
| `dynamic_degree` | RAFT; maps to our freeze story |
| `temporal_flickering` | extra; used on Panda backfill |

Windows (locked 2026-08-18): **always score the full generated clip.**
That is the comparable VBench++ number. last5 is optional diagnostic.

| Clip | Frames | Role |
|---|---|---|
| `full` | entire generated clip (481 frames / 30 s here) | **Required. Official comparable number.** What other papers report. |
| `last5` | last 5 s (~80 frames) | Optional diagnostic. Methods diverge here. Do not label this “VBench++.” |
| `first5` | first 5 s | Optional pairing sanity (should nearly match across methods). |

`i2v_subject` / `i2v_background` need `vbench2_beta_i2v` and
name-matched stills. Not in the first job. `camera_motion` needs
labeled camera prompts; skip.

---

## Required comparison

Same 32 images, seed 0, 30 s, already on disk:

`wan_experiment/results/i2v_bon_32v_hybrid/{notta,always_bon,gated_bon}_h30s_shard0/`

After search-while-sick finishes, score
`i2v_bon_32v_sick/gated_bon_h30s_shard0/` the same way and pair
against the hybrid do-nothing / always-search VBench numbers.

---

## How to run

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_vbench_hybrid32.sh
```

One sequential GPU job, **full then last5**, three method dirs. Uses
`vbench-backfill`, not `self_forcing`.

```bash
python wan_experiment/scripts/analyze_i2v_vbench.py \
    --series-dir wan_experiment/results/i2v_bon_32v_hybrid \
    --clip full \
    --out sweep_experiment/reports/paper_tables/$(date +%F)_wan_i2v_bon32_vbench_full.md
```

---

## How to read the result (before numbers exist)

- **Gating quality win:** gated median ≥ always on most dims, and
  per-video gated>always beats gated<always.
- **Efficiency-only (current GT-free story):** VBench tie + gated
  cheaper. That story is allowed only after VBench agrees it is a tie.
- **Verifier useful:** Spearman ρ(last-chunk, VBench) **negative**
  on `subject_consistency` / `dynamic_degree` / `imaging_quality`.
- **Verifier lying or misaligned:** ρ ≈ 0, or **positive** ρ on
  imaging / dynamic (official dims may reward raw sharpness or
  motion; our composite punishes *deviation* from the first second).
- Cite medians. Video 26 is an 85.6 last-chunk outlier.

No numbers in this file. The dated `*_vbench_full.md` table is the
comparable result. `*_vbench_last5.md` is extra.
