# Host-split N=32 H1 complete (2026-08-23 20:08)

Jobs: **16215197** sf_roll 35m 0:0 32/32;
**16228103** rf_chunk retry 1h04 0:0 **32/32** (after kv_cache1 alias);
**16228104** rf_chunk VBench 27m 0:0, 7/7 dims;
**16215199** sf_recache 1h02; **16215200** rf_recache 37m.
128 rolling VBench **16228045** CANCELLED+ at 2h07 during
`temporal_flickering` (same wall as **16209128**). `joined.json`
still missing. Dims through `dynamic_degree` are on disk.

Cite medians. Pair is vs confirm notta **and** forward
`rolling_notta` (same 32). Analyzer PROMOTE is vs SF notta only —
wrong baseline for RF-lineage arms.

## H1 — weights vs sampler (COMPLETE)

| Method | θ | unroll | tail med | vs SF | vs RF | W/L vs SF | W/L vs RF | Subj | IQ | Dyn | Flicker | Call |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| notta | SF | chunks | 0.0135 | — | −24% | — | 11/21 | 0.665 | 69.65 | 0 | 0.986 | SF host |
| rolling_notta | RF | window | 0.0178 | +31% | — | 21/11 | — | **0.702** | **70.44** | 0 | — | RF host |
| **sf_roll** | SF | window | **0.0281** | **+108%** | **+58%** | 28/4 | 27/5 | 0.666 | 70.09 | **1.0** | 0.972 | twitch; do not scale |
| **rf_chunk** | RF | chunks | **0.0281** | **+108%** | **+58%** | **30/2** | **29/3** | 0.676 | 69.85 | **1.0** | 0.972 | twitch; do not scale |

Neither cross bit-matches the other (20/12; 0 exact). Both pass
locked bars **vs SF notta** (IQ/subj hold, Dyn 0→1). Both **fail
subject vs RF host** (0.666 / 0.676 vs 0.702; −0.036 / −0.026).
Background/flicker slip vs SF (0.796–0.805 / 0.972 vs 0.825 / 0.986).

rf_chunk losses vs rolling: 0002 (tie-ish 0.03295 vs 0.03301),
0021, 0031. vs notta: 0004, 0018.

**H1 call:** Sampler is **not** a no-op. Ckpt is **not** the only
thing. Mismatching θ and unroll (either direction) overshoots into
**twitch** (tail ~0.028, Dyn 1, flicker 0.972). The quality-preserving
object is still **matched native RF** (RF θ + RF window, tail 0.0178,
Dyn 0, subj 0.702). Do **not** scale `sf_roll` or `rf_chunk`. Do not
call either “our method.” Watch a few mp4s before any narrative that
Dyn 1 is living motion (0001 / 0004 / 0007 / 0025).

## H4 — VAE recache (unchanged)

| Method | vs own host tail | W/L | IQ vs host | Subj vs host | Dyn | Call |
|---|---:|---|---:|---:|---:|---|
| sf_recache | **−1.1%** vs notta | 15/17 | +0.34 | −0.001 | 0 | **NO** (no-op) |
| rf_recache | **+6.6%** vs rolling | **30/2** | −0.52 | −0.003 | 0 | **HOLD** |

`rf_recache` lifts almost every clip a little; Dyn stays 0. That is
VAE-roundtrip grain, not living motion. Do not scale.

H2/H3 stay **NO** (N=128 offline). Do not GPU a chunk-0 router.

## Full-clip VBench N=32 (host-split series)

| Method | subject | background | aesthetic | IQ | smoothness | dynamic | flicker |
|---|---:|---:|---:|---:|---:|---:|---:|
| notta | 0.6652 | 0.8248 | 0.5068 | 69.65 | 0.9923 | 0.000 | 0.9863 |
| sf_roll | 0.6655 | 0.7963 | 0.5433 | 70.09 | 0.9853 | 1.000 | 0.9720 |
| rf_chunk | 0.6764 | 0.8053 | 0.5151 | 69.85 | 0.9833 | 1.000 | 0.9719 |
| sf_recache | 0.6642 | 0.8160 | 0.5185 | 70.00 | 0.9921 | 0.000 | 0.9859 |
| rf_recache | 0.6985 | 0.8237 | 0.5299 | 69.92 | 0.9900 | 0.000 | 0.9809 |

`rolling_notta` VBench stays on `v2v_panda_forward_32v` (subj 0.702,
IQ 70.44, Dyn 0).

## 128 quality (still incomplete)

| Method | N | tail med | Subj | IQ | Dyn |
|---|---:|---:|---:|---:|---:|
| notta | 128 | 0.0136 | **0.648** | **70.20** | 0 |
| rolling_notta | 128 | 0.0177 (+31%) | **missing joined** | **missing joined** | per-dim through Dyn written |

**16228045** and **16209128** both died at **~2h06–2h07** despite a
12 h wall. Treat as preemption (`#SBATCH --comment="preemption=yes"`).
A full 7-dim 128 pass does not fit in one 2 h slice. Resume with
skip-existing so only `temporal_flickering` (~25 min) runs.
Check subject n: 09128 died at 91/128; if that file was skipped,
drop it before resubmit.

Do **not** call 128 YES. Tails +31% hold. notta quality is in.

## Closed this series

Do not scale `sf_roll`, `rf_chunk`, `sf_recache`, or `rf_recache`.
No H2/H3 GPU. No `live_min` / 0.8 retune. No TTC. No I2V scale-up.
Do not rebrand RF as our controller.

## Next (cluster)

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
JOIN_ONLY=1 bash wan_experiment/sbatch/submit_v2v_rolling128_vbench.sh
bash wan_experiment/sbatch/submit_v2v_rolling128_vbench.sh
```
