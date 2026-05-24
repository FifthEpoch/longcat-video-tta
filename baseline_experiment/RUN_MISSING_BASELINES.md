# Run Missing Baseline Configs

The summary report compares TTA runs to **no-TTA baseline** runs with the same cond/gen frame counts. The following baseline configs are **not yet run**; once they exist under `baseline_experiment/results/cond{N}_gen{M}/`, the extractor will fill PSNR_bl, SSIM_bl, LPIPS_bl for the corresponding experiments.

## Missing configs

| cond | gen | Needed for |
|------|-----|------------|
| 7    | 14  | Exp 3 Training Frames (TF_*2: 7 cond, 14 gen) |
| 24   | 14  | Exp 3 Training Frames (TF_*4: 24 cond, 14 gen) |
| 14   | 2   | Exp 4 Generation Horizon (GH_*1: 14 cond, 2 gen) |
| 14   | 30  | Exp 4 Generation Horizon (GH_*3: 14 cond, 30 gen) |
| 14   | 58  | Exp 4 Generation Horizon (GH_*4: 14 cond, 58 gen) |

**Note:** `cond14_gen14` already exists and is used for Exp 4 GH_*2 and standard sweeps.

## Option A: SLURM (one job per config)

From the **repo root** (and with `REPO` set to your repo path if different from the sbatch default):

```bash
cd /path/to/LongCat-Video-Experiment   # or your REPO

# Exp 3 missing baselines
sbatch --account torch_pr_36_mren --export=NUM_COND=7,NUM_GEN=14 baseline_experiment/sbatch/run_baseline_sweep.sbatch
sbatch --account torch_pr_36_mren --export=NUM_COND=24,NUM_GEN=14 baseline_experiment/sbatch/run_baseline_sweep.sbatch

# Exp 4 missing baselines
sbatch --account torch_pr_36_mren --export=NUM_COND=14,NUM_GEN=2 baseline_experiment/sbatch/run_baseline_sweep.sbatch
sbatch --account torch_pr_36_mren --export=NUM_COND=14,NUM_GEN=30 baseline_experiment/sbatch/run_baseline_sweep.sbatch
sbatch --account torch_pr_36_mren --export=NUM_COND=14,NUM_GEN=58 baseline_experiment/sbatch/run_baseline_sweep.sbatch
```

The sbatch uses `OUTPUT_DIR="${REPO}/baseline_experiment/results/cond${NUM_COND}_gen${NUM_GEN}"`. If your sbatch uses a different `REPO` (e.g. `SCRATCH`-based), ensure `REPO` points to the same tree where you run `check_all_progress.sh` so that `PROJECT_ROOT/baseline_experiment/results` contains these dirs.

## Option B: Direct Python (no SLURM)

From repo root, with the same data/model paths you use for other baselines:

```bash
# Set these to match your environment
CHECKPOINT_DIR=/path/to/longcat-video-checkpoints
DATA_DIR=/path/to/LongCat-Video-Experiment/datasets/panda_100_480p
RESULTS=baseline_experiment/results

python baseline_experiment/scripts/run_baseline.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "${RESULTS}/cond7_gen14" \
  --num-cond-frames 7 --num-gen-frames 14 --gen-start-frame 32 \
  --resolution 480p --num-inference-steps 50 --guidance-scale 4.0 \
  --seed 42 --max-videos 100 --save-videos

python baseline_experiment/scripts/run_baseline.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "${RESULTS}/cond24_gen14" \
  --num-cond-frames 24 --num-gen-frames 14 --gen-start-frame 32 \
  --resolution 480p --num-inference-steps 50 --guidance-scale 4.0 \
  --seed 42 --max-videos 100 --save-videos

python baseline_experiment/scripts/run_baseline.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "${RESULTS}/cond14_gen2" \
  --num-cond-frames 14 --num-gen-frames 2 --gen-start-frame 32 \
  --resolution 480p --num-inference-steps 50 --guidance-scale 4.0 \
  --seed 42 --max-videos 100 --save-videos

python baseline_experiment/scripts/run_baseline.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "${RESULTS}/cond14_gen30" \
  --num-cond-frames 14 --num-gen-frames 30 --gen-start-frame 32 \
  --resolution 480p --num-inference-steps 50 --guidance-scale 4.0 \
  --seed 42 --max-videos 100 --save-videos

python baseline_experiment/scripts/run_baseline.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "${RESULTS}/cond14_gen58" \
  --num-cond-frames 14 --num-gen-frames 58 --gen-start-frame 32 \
  --resolution 480p --num-inference-steps 50 --guidance-scale 4.0 \
  --seed 42 --max-videos 100 --save-videos
```

After these complete, re-run the summary script so baseline columns are filled for Exp 3 (7/14 and 24/14) and Exp 4 (14/2, 14/30, 14/58).
