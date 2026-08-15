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
| 1 | `setup_env.sbatch` | GPU (flash-attn compile) | 2 h | conda env + clone Self-Forcing + pip |
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

Do **not** `FORCE=1` the env unless you intend to wipe it. The 15772007 failure
was `pip install -r requirements.txt` building `pycuda` / `nvidia-pyindex`
(TensorRT extras; `cuda.h` missing). Setup now strips those three lines.
Inference does not need them.

## Canonical paths

| Thing | Path |
|---|---|
| Conda env | `/scratch/wc3013/conda-envs/self_forcing` |
| Self-Forcing clone | `/scratch/wc3013/third_party/Self-Forcing` |
| Wan2.1-T2V-1.3B | `/scratch/wc3013/wan-checkpoints/Wan2.1-T2V-1.3B` |
| Self-Forcing DMD | `/scratch/wc3013/wan-checkpoints/self_forcing_dmd.pt` |
| VBench-I2V images | `/scratch/wc3013/longcat-video-tta/datasets/vbench_i2v/` |

Re-run any step alone (`FORCE=1` to redo). All three scripts are idempotent.
