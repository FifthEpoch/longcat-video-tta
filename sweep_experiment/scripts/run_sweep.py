#!/usr/bin/env python3
"""
TTA Hyperparameter Sweep Runner

Reads a sweep config YAML and submits one sbatch job per sweep row,
passing all hyperparameters as environment variables to the unified
sweep sbatch template.

Usage:
  # Submit all runs in a series config:
  python sweep_experiment/scripts/run_sweep.py \
      --config sweep_experiment/configs/series01_full_lr.yaml \
      --account torch_pr_36_mren

  # Dry-run (print commands without submitting):
  python sweep_experiment/scripts/run_sweep.py \
      --config sweep_experiment/configs/series01_full_lr.yaml \
      --account torch_pr_36_mren \
      --dry-run

  # Submit only specific run IDs:
  python sweep_experiment/scripts/run_sweep.py \
      --config sweep_experiment/configs/series01_full_lr.yaml \
      --account torch_pr_36_mren \
      --run-ids F1 F3

  # Override data directory (e.g. for UCF101):
  python sweep_experiment/scripts/run_sweep.py \
      --config sweep_experiment/configs/series01_full_lr.yaml \
      --account torch_pr_36_mren \
      --data-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_100_480p
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not found, installing...", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml


# Mapping from config keys to the environment variable names expected
# by run_sweep.sbatch.
_KEY_TO_ENV = {
    # Common
    "num_cond_frames": "NUM_COND_FRAMES",
    "num_frames": "NUM_FRAMES",
    "gen_start_frame": "GEN_START_FRAME",
    "tta_total_frames": "TTA_TOTAL_FRAMES",
    "tta_context_frames": "TTA_CONTEXT_FRAMES",
    "num_inference_steps": "NUM_INFERENCE_STEPS",
    "guidance_scale": "GUIDANCE_SCALE",
    "resolution": "RESOLUTION",
    "seed": "SEED",
    "max_videos": "MAX_VIDEOS",
    "batch_videos": "BATCH_VIDEOS",
    # Full / LoRA shared
    "learning_rate": "LEARNING_RATE",
    "num_steps": "NUM_STEPS",
    "warmup_steps": "WARMUP_STEPS",
    "weight_decay": "WEIGHT_DECAY",
    "max_grad_norm": "MAX_GRAD_NORM",
    # LoRA specific
    "lora_rank": "LORA_RANK",
    "lora_alpha": "LORA_ALPHA",
    "target_modules": "TARGET_MODULES",
    "lora_target_blocks": "LORA_TARGET_BLOCKS",
    "target_ffn": "TARGET_FFN",
    # Delta shared
    "delta_steps": "DELTA_STEPS",
    "delta_lr": "DELTA_LR",
    # Delta-B
    "num_groups": "NUM_GROUPS",
    # Delta-C
    "delta_mode": "DELTA_MODE",
    # Full-model specific
    "optimizer": "OPTIMIZER",
    # Early stopping
    "es_disable": "ES_DISABLE",
    "es_check_every": "ES_CHECK_EVERY",
    "es_patience": "ES_PATIENCE",
    "es_anchor_sigmas": "ES_ANCHOR_SIGMAS",
    "es_noise_draws": "ES_NOISE_DRAWS",
    "es_strategy": "ES_STRATEGY",
    "es_holdout_fraction": "ES_HOLDOUT_FRACTION",
    # Control flags
    "skip_generation": "SKIP_GENERATION",
}

# Method name mapping (config -> env)
_METHOD_MAP = {
    "full": "full",
    "lora": "lora",
    "delta_a": "delta_a",
    "delta_b": "delta_b",
    "delta_c": "delta_c",
}


def load_config(path: str) -> dict:
    """Load and validate a sweep YAML config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required = ["method", "series", "series_name", "fixed", "sweep"]
    for key in required:
        if key not in cfg:
            print(f"ERROR: Missing required key '{key}' in {path}", file=sys.stderr)
            sys.exit(1)

    if cfg["method"] not in _METHOD_MAP:
        print(f"ERROR: Unknown method '{cfg['method']}'. "
              f"Valid: {list(_METHOD_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    return cfg


def build_env_vars(
    method: str,
    series_name: str,
    run_id: str,
    fixed: dict,
    run_overrides: dict,
    data_dir: str | None = None,
    output_base: str | None = None,
) -> dict[str, str]:
    """Build the environment variable dict for a single sbatch submission."""
    env = {
        "METHOD": _METHOD_MAP[method],
        "RUN_ID": run_id,
        "SERIES_NAME": series_name,
    }

    if data_dir:
        env["DATA_DIR"] = data_dir

    if output_base:
        env["OUTPUT_DIR"] = f"{output_base}/{series_name}/{run_id}"

    # Merge fixed + run overrides (run overrides win)
    merged = {**fixed, **{k: v for k, v in run_overrides.items() if k != "run_id"}}

    for key, value in merged.items():
        env_name = _KEY_TO_ENV.get(key)
        if env_name is None:
            print(f"WARNING: Unknown config key '{key}', skipping.", file=sys.stderr)
            continue

        # Handle boolean flags: only set env var if True
        if isinstance(value, bool):
            if value:
                env[env_name] = "1"
            # If False, don't set (empty string = disabled in sbatch)
            continue

        env[env_name] = str(value)

    return env


def submit_job(
    sbatch_path: str,
    env_vars: dict[str, str],
    account: str,
    job_name: str | None = None,
    time_limit: str | None = None,
    mem: str | None = None,
    dry_run: bool = False,
) -> str | None:
    """Submit a single sbatch job. Returns the job ID or None on dry-run."""
    # Build --export string: ALL + our custom vars
    export_parts = ["ALL"]
    for k, v in sorted(env_vars.items()):
        export_parts.append(f"{k}={v}")
    export_str = ",".join(export_parts)

    cmd = ["sbatch", f"--account={account}", f"--export={export_str}"]

    if job_name:
        cmd.append(f"--job-name={job_name}")

    if time_limit:
        cmd.append(f"--time={time_limit}")

    if mem:
        cmd.append(f"--mem={mem}")

    cmd.append(sbatch_path)

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return None

    print(f"  Submitting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: sbatch failed: {result.stderr.strip()}", file=sys.stderr)
        return None

    # Parse job ID from "Submitted batch job 12345"
    output = result.stdout.strip()
    print(f"  -> {output}")
    job_id = output.split()[-1] if output else None
    return job_id


def estimate_time(method: str, run_overrides: dict, fixed: dict) -> str:
    """Estimate wall-clock time for a run based on method and step count."""
    merged = {**fixed, **{k: v for k, v in run_overrides.items() if k != "run_id"}}

    if method == "full":
        steps = merged.get("num_steps", 20)
        if steps <= 10:
            return "12:00:00"
        elif steps <= 40:
            return "18:00:00"
        else:
            return "24:00:00"
    elif method == "lora":
        steps = merged.get("num_steps", 20)
        if steps <= 10:
            return "10:00:00"
        elif steps <= 40:
            return "18:00:00"
        else:
            return "24:00:00"
    elif method in ("delta_a", "delta_b", "delta_c"):
        steps = merged.get("delta_steps", 20)
        if steps <= 20:
            return "8:00:00"
        elif steps <= 50:
            return "12:00:00"
        else:
            return "18:00:00"
    return "24:00:00"


def estimate_mem(method: str) -> str:
    """Estimate memory requirement based on method."""
    if method == "full":
        return "256G"
    elif method == "lora":
        return "192G"
    else:  # delta_a, delta_b, delta_c
        return "192G"


def main():
    parser = argparse.ArgumentParser(
        description="Submit TTA sweep jobs from a YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to sweep config YAML")
    parser.add_argument("--account", type=str, required=True,
                        help="SLURM account (e.g. torch_pr_36_mren)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    parser.add_argument("--run-ids", nargs="+", type=str, default=None,
                        help="Only submit specific run IDs (e.g. F1 F3)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (e.g. for UCF101)")
    parser.add_argument("--output-base", type=str, default=None,
                        help="Override output base directory")
    parser.add_argument("--sbatch-template", type=str,
                        default="sweep_experiment/sbatch/run_sweep.sbatch",
                        help="Path to sbatch template")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method = cfg["method"]
    series_name = cfg["series_name"]
    fixed = cfg["fixed"]
    sweep_rows = cfg["sweep"]

    # Filter to requested run IDs
    if args.run_ids:
        sweep_rows = [r for r in sweep_rows if r["run_id"] in args.run_ids]
        if not sweep_rows:
            print(f"ERROR: No matching run IDs found. Available: "
                  f"{[r['run_id'] for r in cfg['sweep']]}", file=sys.stderr)
            sys.exit(1)

    print(f"{'=' * 70}")
    print(f"TTA Sweep: {cfg.get('description', series_name)}")
    print(f"{'=' * 70}")
    print(f"  Method      : {method}")
    print(f"  Series      : {cfg['series']} ({series_name})")
    print(f"  Runs        : {len(sweep_rows)}")
    if cfg.get("depends_on"):
        print(f"  Depends on  : Series {cfg['depends_on']}")
        print(f"  NOTE: Check that dependent series have completed and "
              f"PLACEHOLDER values in this config have been updated!")
    print(f"{'=' * 70}")
    print()

    submitted = []
    for row in sweep_rows:
        run_id = row["run_id"]
        job_name = f"tta_{series_name}_{run_id}"

        env_vars = build_env_vars(
            method=method,
            series_name=series_name,
            run_id=run_id,
            fixed=fixed,
            run_overrides=row,
            data_dir=args.data_dir,
            output_base=args.output_base,
        )

        time_limit = estimate_time(method, row, fixed)
        mem = estimate_mem(method)

        print(f"Run {run_id}:")
        # Print the swept hyperparameters
        for k, v in row.items():
            if k != "run_id":
                print(f"    {k}: {v}")

        job_id = submit_job(
            sbatch_path=args.sbatch_template,
            env_vars=env_vars,
            account=args.account,
            job_name=job_name,
            time_limit=time_limit,
            mem=mem,
            dry_run=args.dry_run,
        )

        submitted.append({"run_id": run_id, "job_id": job_id})
        print()

    print(f"{'=' * 70}")
    print(f"Summary: {len(submitted)} jobs {'would be ' if args.dry_run else ''}submitted")
    print(f"{'=' * 70}")
    for s in submitted:
        status = s["job_id"] if s["job_id"] else "(dry-run)"
        print(f"  {s['run_id']:>6} -> {status}")

    if not args.dry_run:
        print(f"\nMonitor with: squeue -u $USER --name='tta_{series_name}_*'")
        print(f"Results will be in: sweep_experiment/results/{series_name}/")


if __name__ == "__main__":
    main()
