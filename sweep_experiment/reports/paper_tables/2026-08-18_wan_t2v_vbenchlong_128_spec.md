# Spec — standard long-horizon verify (not submitted)

**Status:** SUBMIT-READY (2026-08-18). User asked to run T2V 128 as an
optional standard compare for gating vs other methods. Not a task lock.
V2V continuation remains allowed. I2V-32 scale-up stays closed. No TTC.

Cluster (after `git pull`):
```
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 bash wan_experiment/sbatch/submit_t2v_bon128.sh   # 2 prompts first
bash wan_experiment/sbatch/submit_t2v_bon128.sh           # then 128
```

**Why this exists:** if we verify freeze / BoN / gating against the
field, copy Relax Forcing / FreqForcing / Self-Forcing++.

---

## Protocol (locked)

| Knob | Value |
|---|---|
| Model | Wan2.1-T2V-1.3B + Self-Forcing causal DMD (existing `self_forcing` env) |
| Task | **T2V** from text, then AR continue from own KV cache |
| Prompts | First **128** MovieGen prompts, Qwen-refined |
| Prompt file (cluster, once present) | Self-Forcing `prompts/MovieGenVideoBench_extended.txt` (same source as official SF inference) or Qwen2.5-7B-Instruct refine of MovieGen 128 |
| Horizons | **30 s** required (Relax Forcing grid). **60 s** optional second table |
| Methods | do-nothing \| always-BoN k=4 \| gated-BoN (same three-way) |
| Official score | **VBench-Long on the full clip** (scene-split + slow/fast). last5 diagnostic only |
| Dims | subject, background, aesthetic, imaging, motion_smoothness, dynamic_degree, flickering |
| Seed | 0, paired across methods |
| Series name | `t2v_bon_128v_vbenchlong` |

## What this is not

- Not I2V-from-still.
- Not VBench-I2V 5 s / 14B.
- Not Kinetics FVD (that is the DFoT visual-prefix table, optional later).
- Not a rescore of `i2v_bon_32v_hybrid`.

## Implementation notes for the next agent (when launched)

1. Confirm `MovieGenVideoBench_extended.txt` on the Self-Forcing clone
   (`/scratch/wc3013/third_party/Self-Forcing/prompts/`). Take lines 1–128.
2. T2V runner: official Self-Forcing `inference.py` path for chunk 0,
   then the existing chunked KV-replay loop **without** an image cond
   (`run_i2v_chunked.py` is I2V-only today). Do not call this a small
   flag flip — it is a new runner.
3. Three methods, same seeds / prompts / horizon. Piece 0 shared.
4. Score with VBench-Long (`vbench2_beta_long` / `vbench-backfill` if
   the env has it). Full clip required. `custom_input` on 32 stills is
   the wrong scorer.
5. Wall: 128 × 30 s × always-search will be large. Shard. 2-way H200
   cap. Do not start TTC.

## Analyze (after a future job)

Cite full-clip VBench-Long medians vs Relax Forcing / Self-Forcing
baselines in those papers. Dynamic degree: report **mean** (fraction
dynamic) and median; median 0 is a 0/1 artifact.

Submit command: see status block above. Runner:
`wan_experiment/scripts/run_t2v_chunked.py` (new file; not a flag on
`run_i2v_chunked.py`). Prompts: first 128 of MovieGen VideoBench
(Qwen-extended if present on the Self-Forcing clone, else vendored
`datasets/moviegen_128.txt`). Horizon: 6 × 21 latents (~501 frames,
~31.3 s) to match the Self-Forcing 5 s unit. 4 shards × 3 methods.
Official score after generate: VBench-Long on the full clip (not yet
submitted).
