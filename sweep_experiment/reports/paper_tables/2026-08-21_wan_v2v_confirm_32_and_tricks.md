# V2V generate read — N=32 confirm + N=8 tricks

**Date:** 2026-08-21
**Status:** generate DONE, both series. Full-clip VBench **not scored yet**.
**Cite medians. Official quality number is still full-clip VBench.**
**Do not write the paper around seed-BoN.**

Sources: user dump 2026-08-21 02:21. `summary.json` on disk for every
method. Paired print died on skipped rows missing `tail_motion` — medians
below are from `summary.json` ok-rows (n matches mp4 count).

## 1. N=32 confirm — seed_bon FAILS

Series: `v2v_panda_confirm_32v`
Jobs: 16113805 notta, 16113806 seed_bon. 32/32 mp4s both dirs.

| Method | N | tail \|Δframe\| | vs notta | last-chunk drift ↓ |
|---|---:|---:|---:|---:|
| notta | 32 | 0.01380 | — | 20.49 |
| seed_bon | 32 | **0.01018** | **−26%** | 18.45 |

Locked rule: promote only if tail motion **beats** notta. **FAIL.**

N=8 bake-off seed_bon was +35% (0.02250 vs 0.01675) and Dyn median 0→0.5
(4/8). That was a lucky prefix of the sorted Panda list. The same method
on the full 32 **reduces** tail motion. Last-chunk drift also dropped
(20.5→18.5) — that is the old I2V failure mode: the two-sided score
matches a quieter tail, not a living one.

**Decision:** seed_bon is not a quality method. Do not scale it. Do not
center the paper on “four seeds + prefix match.” VBench on this pair is
only to document that the N=8 Dyn flip dies; it cannot rescue a −26%
motion result.

## 2. N=8 tricks — vs bake-off notta / seed_bon

Same first 8 videos as `v2v_panda_bakeoff_8v`. Bake-off controls:
notta 0.01675 / seed_bon 0.02250 (last-chunk drift 63.51 / 53.03).

| Method | tail \|Δframe\| | vs notta | vs seed_bon | last-chunk drift | Verdict |
|---|---:|---:|---:|---:|---|
| notta (bake-off) | 0.01675 | — | — | 63.51 | baseline |
| seed_bon (bake-off) | 0.02250 | +34% | — | 53.03 | N=8 only; killed at N=32 |
| **hist_drop** | **0.02377** | **+42%** | **+6%** | 40.45 | **only new motion win** — VBench next |
| cached_bon | 0.02250 | +34% | **0.000** | **53.03** | **exact seed_bon clone** (KV snap works) |
| hinge_bon | 0.01852 | +11% | −18% | 50.18 | H-match hinge **lost to two-sided** |
| sink | 0.01675 | **0.000** | — | **63.51** | **exact notta clone** (sink is a no-op) |
| late_bon | 0.01502 | −10% | — | 57.29 | HOLD |
| good_backtrack | 0.01300 | −22% | — | 97.52 | HOLD — same class as dead-tail backtrack |

Sanity:
- `cached_bon` matched bake-off `seed_bon` to all printed decimals. Snapshot
  restore is live. Efficiency method, not a new quality bet — and the
  quality bet just died at N=32.
- `sink` matched bake-off `notta` to all printed decimals. Prefix+window
  replay without a rerope/sink attention hook does not change pixels.

## 3. What the hypotheses did

**H-match (hinge vs two-sided).** On the same 8 videos where two-sided
seed_bon was +34%, hinge-on-motion was only +11%. The N=8 win was not
“stop rewarding extra twitch.” Two-sided was the better picker *on that
lucky set*, and both lose at N=32.

**H-horizon (late_bon).** Searching only on motion-collapse / last two
chunks lost 10% vs always-on. The N=8 seed_bon gain was not concentrated
in the tail in a way this gate captured.

**History dropout.** Only method that beat both notta and seed_bon on
tail motion at N=8. This is the one live sampling-space axis left.
N=8 is still a coin-flip. Do **not** scale past 8 until full-clip VBench
passes IQ ≥ notta−1.0 and subject ≥ notta−0.02.

**Good-chunk backtrack.** −22%, last-chunk drift 97. Rewind-and-resample
is still the wrong family.

**CachedSearch-style KV snap.** Works (bit-match on the score we print).
Ships the method we just killed. Keep the hook; do not advertise it as a
quality win.

**Attention sink (replay approx).** Dead on this checkpoint. Need the
NVIDIA `sink+window+rerope` kernel/ckpt before this hypothesis gets
another GPU.

## 4. What to run next (quality)

1. Full-clip VBench on `v2v_panda_confirm_32v` `{notta,seed_bon}` —
   document the kill (especially Dyn median).
2. Full-clip VBench on `v2v_panda_tricks_8v` `hist_drop` (and `hinge_bon`
   if we want the H-match autopsy). Pair IQ/subject against bake-off
   notta `joined.json`.
3. Per-video tail_motion for hist_drop vs notta vs seed_bon on the 8 —
   one clip must not be the whole +42%.
4. Do **not** submit hist_drop N=32 until (2)+(3) pass.
5. No TTC. No motion_bon. No more dead-tail backtrack. No shift/CFG.

## 5. How to score VBench

```bash
# confirm N=32 (document the kill)
sbatch --account=torch_pr_36_mren --time=08:00:00 \
  --export=ALL,SERIES_DIR=/scratch/wc3013/longcat-video-tta/wan_experiment/results/v2v_panda_confirm_32v,METHODS="notta seed_bon",CLIPS=full \
  wan_experiment/sbatch/run_i2v_vbench.sbatch

# hist_drop (+ hinge) on the 8
sbatch --account=torch_pr_36_mren --time=04:00:00 \
  --export=ALL,SERIES_DIR=/scratch/wc3013/longcat-video-tta/wan_experiment/results/v2v_panda_tricks_8v,METHODS="hist_drop hinge_bon",CLIPS=full \
  wan_experiment/sbatch/run_i2v_vbench.sbatch
```

Space-separated `METHODS`. No commas in `--export`.
