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
chunks. Not a paired quality result (chunk 0 cand0 already differed).
Sampler RNG is now seeded. Pull before any 16v run. No TTC yet.

Runner: `wan_experiment/scripts/run_i2v_continuation.py`
(official CausalInferencePipeline I2V path; `independent_first_frame=true`;
KV cache enlarged past the 21-frame default; PyTorch SDPA if flash-attn
is missing — job 15858704 died on `assert FLASH_ATTN_2_AVAILABLE`).
Must run with `torch.set_grad_enabled(False)` + `inference_mode`
(job 15879723 filled the H200 with grads on).
