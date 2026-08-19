# Wan 1.3B / Self-Forcing cluster setup

Overnight, no-SSH setup for the **continuation / I2V** stack on Wan2.1-T2V-1.3B
+ the Self-Forcing (NeurIPS 2025) causal checkpoint.

**Do not install this into `/scratch/wc3013/conda-envs/longcat`.** That env is
numpy 2.x / torch 2.6 (LongCat). Self-Forcing pins `numpy==1.24.4` and
`diffusers==0.31.0`. We already split VBench into `vbench-backfill` for the
same reason. New env: `/scratch/wc3013/conda-envs/self_forcing`.

## One command (login node)

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_setup_chain.sh
```

That submits three dependent SLURM jobs and prints the job IDs. You can
disconnect. When job 3 finishes, read:

```
wan_experiment/results/setup_healthcheck/report.json
wan_experiment/slurm_log/wan_healthcheck_<jid>.out
```

| Job | Script | Partition | Wall | What |
|---|---|---|---|---|
| 1 | `setup_env.sbatch` | CPU (flash-attn skipped) | 1 h | conda env + clone Self-Forcing + pip |
| 2 | `download_assets.sbatch` | CPU | 4 h | Wan2.1-1.3B (~15 GB) + SF DMD ckpt + VBench-I2V images |
| 3 | `healthcheck.sbatch` | GPU | 1 h | load weights, decode 8 I2V images, optional 1-clip T2V smoke |

Jobs 1 and 2 run **in parallel**. Job 3 waits for both (`afterok`).

**If download already succeeded** (Wan dir + `self_forcing_dmd.pt` present) and
only the env job failed, do **not** rerun the full chain. Pull, then:

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
J1=$(sbatch --parsable --account=torch_pr_36_mren wan_experiment/sbatch/setup_env.sbatch)
J3=$(sbatch --parsable --account=torch_pr_36_mren --dependency=afterok:${J1} \
    wan_experiment/sbatch/healthcheck.sbatch)
echo "setup_env=${J1}  healthcheck=${J3}"
```

Do **not** `FORCE=1` the env unless you intend to wipe it.

Known setup failures (both already patched):
- 15772007: official `requirements.txt` builds `pycuda` / `nvidia-pyindex`
  (`cuda.h` missing). Those three lines are stripped. Unused by inference.
- 15796574: `TIMEOUT` at 2h compiling optional `flash-attn`. Setup is now
  CPU-only and skips flash-attn by default (`SKIP_FLASH=1`). `setup.py develop`
  runs before any optional compile.

## Canonical paths

| Thing | Path |
|---|---|
| Conda env | `/scratch/wc3013/conda-envs/self_forcing` |
| Self-Forcing clone | `/scratch/wc3013/third_party/Self-Forcing` |
| Wan2.1-T2V-1.3B | `/scratch/wc3013/wan-checkpoints/Wan2.1-T2V-1.3B` |
| Self-Forcing DMD | `/scratch/wc3013/wan-checkpoints/self_forcing_dmd.pt` |
| VBench-I2V images | `/scratch/wc3013/longcat-video-tta/datasets/vbench_i2v/` |

Re-run any step alone (`FORCE=1` to redo). All three scripts are idempotent.

## Setup status (2026-08-16)

Healthcheck job **15858269** passed all required checks. Official
`inference.py` smoke failed (`torchvision.io.write_video` removed in the
torch 2.13 wheel). That is expected; the continuation runner writes
video via imageio and loads the DMD ckpt from the `generator_ema` key.

## First experiment — NOTTA I2V smoke (2 images × 5 s)

**PASSED 2026-08-16, job 15880611.** `n_ok=2`, 85-frame 480×832 mp4s
(5.9 MB / 3.9 MB), generate 11.99 s then 8.01 s. Job wall 2:55 including
load. The 138 GB OOMs were autograd (`65ba50c`).

```bash
# already done; keep for reruns
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_smoke.sh
```

First-frame MAE vs cond jpg: **5.56 / 3.71** (I2V pass; login node has
no ffmpeg — use `imageio.get_reader().get_data(0)`).

## 16 images × {5 s, 30 s} NOTTA

**PASSED 2026-08-16.** `n_ok=16` at both horizons. Mean generate+write
9.61 s (5 s) and 38.32 s (30 s).

```bash
# already done; keep for reruns
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_notta16.sh
```

## Next — GT-free drift on those mp4s (CPU, login node)

Do this before porting BoN/TTC. If 30 s is flat vs 5 s, there is no
controller headroom at this horizon.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
/scratch/wc3013/conda-envs/self_forcing/bin/python \
    wan_experiment/scripts/score_i2v_drift.py \
    --dir wan_experiment/results/i2v_notta_16v/h5s_shard0 \
    --dir wan_experiment/results/i2v_notta_16v/h30s_shard0
```

**DONE 2026-08-17.** 30 s median sharp **+167%**, motion **−60%**.
5 s is mild (+11% / −14%).

## Next — chunked NOTTA vs always-BoN k=4 (2 × 30 s)

Official `inference()` only KV-caches I2V frame 0, so the runner
(`run_i2v_chunked.py`) replays committed latents then denoises the next
chunk. 30 s = 5 × 24 gen latents. Chunk 0 is seed 0 (shared prefix).
always-BoN searches chunks 1–4. No TTC in this smoke.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_bon_smoke.sh
```

**PASSED 2026-08-17** (15883525/526): n_ok=2, search left cand0 on 5/8
chunks. Not a paired quality result (unseeded add_noise).

## Next — 16v 30 s NOTTA | always-BoN | gated-BoN

cand0 is seed-invariant (per-chunk CUDA Generator + deterministic
flags). gated-BoN fires when incoming last-1s composite > 2.0.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_bon16.sh
```

**DONE 2026-08-17** (15884598/599/600). Chunk-0 cand0 matched 16/16.
Last-chunk NOTTA 4.43 / always 3.23 / gated 3.38. Search works. Gated
vs always is **not** a quality win (mean +0.152). Efficiency line.

## Next — 32v hybrid gate (ch1 / late / trend)

`gated_bon` fires if `(chunk==1 and incoming>0.8)` or `incoming>2.0`
or `(Δincoming>0.5 and incoming_prev>0.5)`. Per-chunk traces go to
`gate_trace.jsonl`. Re-runs NOTTA and always-on so the schema matches.
No TTC.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_bon32_hybrid.sh
```

After the three jobs finish:

```bash
python wan_experiment/scripts/analyze_i2v_bon.py \
    --series-dir wan_experiment/results/i2v_bon_32v_hybrid
```

**DONE 2026-08-17.** Cite medians (video 26 = 85.6): NOTTA 3.68 /
always 2.97 / gated 3.04. gated−always −0.041 / 0, 19/32
better-or-tie, 33% cheaper. First-16 pairing held; hybrid flipped
T=2.0 +0.15 → −0.12. Still efficiency, not a quality win. Table:
`sweep_experiment/reports/paper_tables/2026-08-17_wan_i2v_bon32_hybrid.md`.

## Next — sticky gated-search (same 32 videos)

Once any alarm fires, later pieces keep searching. Same three alarms
as the hybrid run. Only gated-search is re-run; compare against the
hybrid do-nothing and always-search jobs. No test-time training.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_bon32_sticky.sh
```

After it finishes:

```bash
python wan_experiment/scripts/analyze_i2v_bon.py \
    --series-dir wan_experiment/results/i2v_bon_32v_sticky \
    --baseline-dir wan_experiment/results/i2v_bon_32v_hybrid
```

**DONE 2026-08-18.** 03/24 caught (exact ties). 21/32 exact ties with
always-search. Wall 256 vs 258 s — spent the 33% hybrid saving.
Erased hybrid’s unique wins on 11 and 16. Not a quality win. Hybrid
stays the efficiency method. Table:
`sweep_experiment/reports/paper_tables/2026-08-18_wan_i2v_bon32_sticky.md`.

## Next — search while sick (same 32 videos)

Stay-on after an alarm, but turn memory off if the last second
recovered by more than 0.5 or is now below 1.0. Gated-search only.
No test-time training.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_bon32_sick.sh
```

After it finishes:

```bash
python wan_experiment/scripts/analyze_i2v_bon.py \
    --series-dir wan_experiment/results/i2v_bon_32v_sick \
    --baseline-dir wan_experiment/results/i2v_bon_32v_hybrid
```

## Next — official VBench on the hybrid 32 mp4s (paper-blocking)

The controller stays GT-free. The finished videos do not. The
handcrafted last-piece score is not a paper quality claim (11/16
already showed it can lie). These 32 stills have no 30 s GT, so no
PSNR. Score do-nothing / always-search / gated-search with VBench
quality dims. **Always score the full clip** (comparable VBench++).
last5 is optional diagnostic. Default job order: full then last5.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_i2v_vbench_hybrid32.sh
```

```bash
python wan_experiment/scripts/analyze_i2v_vbench.py \
    --series-dir wan_experiment/results/i2v_bon_32v_hybrid \
    --clip full
```

**DONE 2026-08-18** (15959601 + 15984561). Full-clip VBench++ is a
tie (cite this vs other papers). last5 exists as a diagnostic.
Read: `sweep_experiment/reports/paper_tables/2026-08-18_wan_i2v_bon32_vbench_read.md`.

## Search-while-sick 32v — DONE 2026-08-18 (job 15959146)

Checklist pass on the handcrafted score. Median 2.764 vs always 2.966
/ hybrid 3.036. 11=1.830, 16=2.656, 24 exact always, wall 204 s.
Table: `sweep_experiment/reports/paper_tables/2026-08-18_wan_i2v_bon32_sick.md`.
Not a paper quality claim until VBench has all three methods.

Runner: `wan_experiment/scripts/run_i2v_continuation.py`
(official CausalInferencePipeline I2V path; `independent_first_frame=true`;
KV cache enlarged past the 21-frame default; PyTorch SDPA if flash-attn
is missing — job 15858704 died on `assert FLASH_ATTN_2_AVAILABLE`).
Must run with `torch.set_grad_enabled(False)` + `inference_mode`
(job 15879723 filled the H200 with grads on).
