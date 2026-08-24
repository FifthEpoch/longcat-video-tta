# rolling-128 official VBench 7/7 (2026-08-23 21:49)

Login `--join-only` after **16259396** COMPLETED 0:0 (5m36s).
Both methods **n=128** on every dim. Cite **medians**. Official
VBench++ is the **full generated clip**.

This is **someone else’s host** (RF DMD + native rolling sampler,
k=1). Not AdaSteer / TTA / our controller.

## Locked bars — PASS

| Clause | N=128 rolling vs SF notta | Call |
|---|---|---|
| median tail > notta | 0.01772 vs 0.01355 (+31%, 88/40) | **Yes** |
| IQ ≥ notta − 1.0 | 70.91 vs 70.20 (**+0.71**) | **Yes** |
| subject ≥ notta − 0.02 | 0.687 vs 0.648 (**+0.039**) | **Yes** |

## Full-clip VBench (7/7)

| Method | N | subject | background | aesthetic | IQ | smoothness | dynamic | flicker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| notta (SF) | 128 | 0.6482 | 0.8049 | 0.5073 | 70.20 | **0.9919** | 0.000 | **0.9858** |
| rolling_notta (RF) | 128 | **0.6871** | **0.8093** | **0.5403** | **70.91** | 0.9910 | **1.000** | 0.9822 |

| Dim | rolling − SF | Note |
|---|---|---|
| subject | **+0.039** | locked bar |
| background | +0.004 | |
| aesthetic | **+0.033** | |
| IQ | **+0.71** | locked bar |
| smoothness | −0.0009 | noise |
| dynamic | 0 → **1** | mean 0.53; first-32 median still 0 |
| flicker | −0.0036 | not H1 twitch (those were 0.972) |

Flicker is a small cost, not a fail. H1 crosses (`sf_roll` /
`rf_chunk`) were 0.972 / Dyn 1 / subject down vs this host.

## Dyn split (unchanged)

| Split | n_dyn | mean | median |
|---|---:|---:|---:|
| first 32 | 14/32 | 0.438 | **0** |
| last 96 | 54/96 | 0.562 | **1** |
| all 128 | 68/128 | 0.531 | **1** |

N=32 “Dyn stays 0” is the first-32 slice. Do not rewrite the
N=32 forward table.

## What this is not

- Not a reason to scale `sf_roll` / `rf_chunk`.
- Not `rf_recache` / leftovers / H2 / H3.
- Not “our TTA.” Paper baseline stays SF notta; this is the
  required RF comparison.
- Family wave 16261273–277 is a separate controller test on RF.

Source: login join-only 2026-08-23 21:49. Regenerable from
`v2v_panda_rolling_128v/{notta,rolling_notta}_h30s_shard0/vbench_full/joined.json`.
