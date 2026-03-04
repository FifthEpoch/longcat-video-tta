#!/usr/bin/env python3
"""
Consolidated experiment results exporter.

Replaces: aggregate_sweep.py, build_comparison_tables.py, find_best_configs.py,
          extract_summary.py, check_all_progress.sh

Walks every results directory, reads summary.json and config.json, matches
each TTA run to the appropriate baseline, and outputs:
  1. all_results.json  — structured JSON with runs, series metadata, missing baselines
  2. Console report    — series-by-series tables with fixed params and per-run metrics

Usage:
    cd /path/to/LongCat-Video-Experiment          # or PROJECT_ROOT on cluster
    PROJECT_ROOT=$PWD python sweep_experiment/scripts/export_all_results.py

    # JSON-only (no console tables, writes to stdout):
    PROJECT_ROOT=$PWD python sweep_experiment/scripts/export_all_results.py --json-only > all_results.json

    # Specify output file (default: all_results.json in project root):
    PROJECT_ROOT=$PWD python sweep_experiment/scripts/export_all_results.py -o my_results.json
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    Path(__file__).resolve().parent.parent.parent,
))
RESULTS_DIRS = [
    PROJECT_ROOT / "sweep_experiment" / "results",
    PROJECT_ROOT / "sweep_experiment" / "results_no_preempt",
    PROJECT_ROOT / "baseline_experiment" / "results",
    PROJECT_ROOT / "backbone_experiment" / "opensora" / "results",
    PROJECT_ROOT / "backbone_experiment" / "cogvideo" / "results",
]
CONFIG_DIR = PROJECT_ROOT / "sweep_experiment" / "configs"
DEFAULT_OUTPUT = PROJECT_ROOT / "all_results.json"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: COLLECT RUNS
# ═══════════════════════════════════════════════════════════════════════

def safe_mean(xs):
    return statistics.mean(xs) if xs else None

def safe_std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0 if xs else None

def safe_median(xs):
    return statistics.median(xs) if xs else None


def extract_run(run_dir: Path) -> Optional[dict]:
    """Extract a single run directory into a flat record."""
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint.json"

    if not summary_path.exists():
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                ck = json.load(f)
            return {
                "status": "in_progress",
                "series": run_dir.parent.name,
                "run_id": run_dir.name,
                "videos_done": ck.get("next_idx", 0),
            }
        return None

    with open(summary_path) as f:
        s = json.load(f)

    rec: Dict[str, Any] = {
        "status": "complete",
        "series": run_dir.parent.name,
        "run_id": run_dir.name,
        "path": str(run_dir),
    }

    is_baseline = "metrics" in s and "results" not in s
    results = s.get("results", [])
    ok = [r for r in results if r.get("success", False)]

    method = s.get("method", "")
    if not method and config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            method = cfg.get("method", "")
        except (json.JSONDecodeError, OSError):
            pass
    rec["method"] = method

    if is_baseline:
        rec["n_ok"] = s.get("num_successful", 0)
        rec["n_total"] = s.get("num_videos", 0)
    else:
        rec["n_ok"] = len(ok)
        rec["n_total"] = s.get("num_videos", len(results))

    if is_baseline:
        m = s.get("metrics", {})
        for key in ("psnr", "ssim", "lpips"):
            if key in m and m[key]:
                rec[f"{key}_mean"] = m[key].get("mean")
                rec[f"{key}_std"] = m[key].get("std")
                rec[f"{key}_median"] = m[key].get("median")
        t = s.get("timing", {}).get("per_video_inference_s", {})
        if t:
            rec["gen_time_mean"] = t.get("mean")
            rec["gen_time_std"] = t.get("std")
        rec["train_time_mean"] = 0.0
        rec["train_time_std"] = 0.0
    else:
        psnrs = [r["psnr"] for r in ok if "psnr" in r and r["psnr"] is not None and not math.isnan(r["psnr"])]
        ssims = [r["ssim"] for r in ok if "ssim" in r and r["ssim"] is not None and not math.isnan(r["ssim"])]
        lpipss = [r["lpips"] for r in ok if "lpips" in r and r["lpips"] is not None and not math.isnan(r["lpips"])]
        train_ts = [r["train_time"] for r in ok if "train_time" in r]
        gen_ts = [r["gen_time"] for r in ok if "gen_time" in r]
        total_ts = [r.get("total_time", r.get("train_time", 0) + r.get("gen_time", 0))
                    for r in ok if "train_time" in r]

        rec["psnr_mean"] = safe_mean(psnrs)
        rec["psnr_std"] = safe_std(psnrs)
        rec["psnr_median"] = safe_median(psnrs)
        rec["ssim_mean"] = safe_mean(ssims)
        rec["ssim_std"] = safe_std(ssims)
        rec["lpips_mean"] = safe_mean(lpipss)
        rec["lpips_std"] = safe_std(lpipss)
        rec["train_time_mean"] = safe_mean(train_ts)
        rec["train_time_std"] = safe_std(train_ts)
        rec["gen_time_mean"] = safe_mean(gen_ts)
        rec["gen_time_std"] = safe_std(gen_ts)
        rec["total_time_mean"] = safe_mean(total_ts)

        losses = [r["final_loss"] for r in ok
                  if "final_loss" in r and r["final_loss"] is not None
                  and not math.isnan(r["final_loss"])]
        rec["final_loss_mean"] = safe_mean(losses)

    if not is_baseline:
        es_infos = [r.get("early_stopping_info") for r in ok
                    if r.get("early_stopping_info")]
        if es_infos:
            stopped = [e for e in es_infos if e.get("stopped_early", False)]
            best_steps = [e["best_step"] for e in es_infos if "best_step" in e]
            rec["es_stopped_count"] = len(stopped)
            rec["es_total_count"] = len(es_infos)
            rec["es_best_step_mean"] = safe_mean(best_steps)

    config_keys = [
        "delta_steps", "delta_lr", "num_groups", "delta_target",
        "delta_mode", "delta_dim",
        "learning_rate", "num_steps", "warmup_steps",
        "lora_rank", "lora_alpha",
        "total_params", "trainable_params",
        "norm_target", "norm_steps", "norm_lr",
        "film_mode", "film_steps", "film_lr",
        "num_cond_frames", "num_frames", "gen_start_frame",
        "tta_total_frames", "tta_context_frames",
        "num_inference_steps", "guidance_scale", "resolution",
        "max_videos", "seed",
        "batch_videos", "retrieval_pool_dir",
        "optimizer", "weight_decay", "max_grad_norm",
        "lora_target_blocks", "target_modules",
        "es_disable", "es_check_every", "es_patience",
        "es_anchor_sigmas", "es_noise_draws", "es_strategy",
        "es_holdout_fraction",
        "clip_gate_enabled", "clip_gate_threshold", "clip_gate_model",
        "clip_gate_sample_frames", "clip_gate_aggregation",
        "clip_gate_sampling_mode", "clip_gate_late_fraction",
        "clip_gate_log_only", "clip_gate_fail_open",
    ]
    for k in config_keys:
        if k in s and s[k] is not None:
            rec[k] = s[k]

    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if k2 not in rec or rec[k2] is None:
                            rec[k2] = v2
                else:
                    if k not in rec or rec[k] is None:
                        rec[k] = v
        except (json.JSONDecodeError, OSError):
            pass

    clip_stats = s.get("clip_gate_stats")
    if isinstance(clip_stats, dict):
        rec["clip_skip_rate"] = clip_stats.get("skip_rate")
        rec["clip_num_skipped"] = clip_stats.get("num_skipped")
        rec["clip_num_scored"] = clip_stats.get("num_scored")

    trainable = rec.get("trainable_params")
    if trainable is None:
        method_lower = str(rec.get("method", "")).lower()
        if "delta_a" in method_lower:
            rec["trainable_params"] = 512
        elif "delta_b" in method_lower:
            groups = rec.get("num_groups", 1)
            dim = rec.get("delta_dim") or 512
            rec["trainable_params"] = groups * dim
        elif "delta_c" in method_lower:
            rec["trainable_params"] = 4096

    if rec.get("train_time_mean") and rec.get("gen_time_mean"):
        rec["train_gen_ratio"] = rec["train_time_mean"] / rec["gen_time_mean"]

    return rec


def collect_all_runs() -> List[dict]:
    """Phase 1: walk result directories and extract all runs."""
    all_records: List[dict] = []
    seen_paths: set = set()

    for results_dir in RESULTS_DIRS:
        if not results_dir.exists():
            continue
        for series_dir in sorted(results_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            if (series_dir / "summary.json").exists():
                rec = extract_run(series_dir)
                if rec and str(series_dir) not in seen_paths:
                    seen_paths.add(str(series_dir))
                    all_records.append(rec)
                continue
            for run_dir in sorted(series_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if run_dir.name == "generated_videos":
                    continue
                if str(run_dir) in seen_paths:
                    continue
                rec = extract_run(run_dir)
                if rec:
                    seen_paths.add(str(run_dir))
                    all_records.append(rec)

    return all_records


# ═══════════════════════════════════════════════════════════════════════
# HELPERS: infer cond/gen frames
# ═══════════════════════════════════════════════════════════════════════

_EXP3_COND_MAP = {"1": 2, "2": 7, "3": 14, "4": 24}
_EXP4_GEN_MAP = {"1": 2, "2": 14, "3": 30, "4": 58}


def _infer_num_gen(rec: dict) -> Optional[int]:
    """Return the number of generated frames for a run."""
    ncond = rec.get("num_cond_frames")
    nframes = rec.get("num_frames")
    if ncond is not None and nframes is not None:
        return nframes - ncond
    return None


def _infer_cond_gen_for_run(rec: dict) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort inference of (num_cond, num_gen) from record."""
    ncond = rec.get("num_cond_frames")
    ngen = _infer_num_gen(rec)
    series = rec.get("series", "")
    rid = rec.get("run_id", "")

    if ncond is not None and ngen is not None:
        return ncond, ngen

    # Exp3: suffix digit -> cond count, gen=14
    if "exp3_train_frames" in series:
        suffix = rid[-1] if rid else ""
        c = _EXP3_COND_MAP.get(suffix)
        if c is not None:
            return c, 14

    # Exp4: suffix digit -> gen count, cond=14
    if "exp4_gen_horizon" in series:
        suffix = rid[-1] if rid else ""
        g = _EXP4_GEN_MAP.get(suffix)
        if g is not None:
            return 14, g

    # Ratio sweep
    if series == "adasteer_ratio_sweep":
        ratio_map = {
            "AR1": (2, 14), "AR2": (2, 14), "AR3": (2, 14),
            "AR4": (14, 14), "AR5": (14, 14),
            "AR6": (24, 14), "AR7": (24, 14), "AR8": (24, 14),
        }
        if rid in ratio_map:
            return ratio_map[rid]

    # Delta-A optimized
    if series == "delta_a_optimized":
        if rid == "DAO1":
            return 24, 14
        if rid == "DAO2":
            return 14, 14

    # Baseline experiment: parse from run_id like cond14_gen14
    m = re.match(r"(?:ucf101_)?cond(\d+)_gen(\d+)", rid)
    if m:
        return int(m.group(1)), int(m.group(2))
    if rid == "baseline_480p":
        return 2, 14

    # Default for standard experiments
    if ncond is not None:
        return ncond, 14
    return 14, 14


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: BASELINE MATCHING
# ═══════════════════════════════════════════════════════════════════════

def _build_baseline_index(all_records: List[dict]) -> dict:
    """Build lookup structures for baseline matching."""
    idx: Dict[str, Any] = {
        "no_tta_panda": {},     # (cond,gen) -> record  from panda_no_tta_continuation
        "no_tta_ucf101": {},    # (cond,gen) -> record  from ucf101_no_tta
        "baseline_panda": {},   # (cond,gen) -> record  from baseline_experiment cond*_gen*
        "baseline_ucf101": {},  # (cond,gen) -> record  from baseline_experiment ucf101_cond*_gen*
        "es_disable": None,     # es_ablation_disable record
        "by_series_run": {},    # (series, run_id) -> record
    }

    for r in all_records:
        if r.get("status") != "complete":
            continue
        series = r.get("series", "")
        rid = r.get("run_id", "")
        cond, gen = _infer_cond_gen_for_run(r)

        idx["by_series_run"][(series, rid)] = r

        if series == "panda_no_tta_continuation":
            if cond is not None and gen is not None:
                idx["no_tta_panda"][(cond, gen)] = r

        if series == "ucf101_no_tta":
            if cond is not None and gen is not None:
                idx["no_tta_ucf101"][(cond, gen)] = r

        if series == "results":
            if "ucf101" in rid:
                if cond is not None and gen is not None:
                    idx["baseline_ucf101"][(cond, gen)] = r
            else:
                if cond is not None and gen is not None:
                    idx["baseline_panda"][(cond, gen)] = r

        if series == "es_ablation_disable":
            idx["es_disable"] = r

    return idx


def _is_ucf101_series(series: str) -> bool:
    return "ucf101" in series


def _is_no_tta(rec: dict) -> bool:
    s = rec.get("series", "")
    return s in ("panda_no_tta_continuation", "ucf101_no_tta") or (
        rec.get("method", "") == "" and rec.get("series") == "results"
    )


def _is_baseline_series(series: str) -> bool:
    return series in ("results", "panda_no_tta_continuation", "ucf101_no_tta")


_COND_GEN_SWEEP_PARAMS = {"num_cond_frames", "num_frames", "num_gen_frames"}


def match_baselines(all_records: List[dict]) -> Tuple[List[dict], List[str]]:
    """Phase 2: attach per-run baseline metrics to each TTA run.

    Key principle: every run is matched to the no-TTA baseline with the
    SAME cond/gen frame configuration.  For series that sweep cond or gen
    counts (exp3, exp4, ratio_sweep, …) we use baseline_experiment entries
    so all runs within the series are compared against the same methodology.
    For standard series (fixed cond=14, gen=14) we use panda_no_tta /
    ucf101_no_tta as before.  No fallback to a mismatched cond/gen baseline.
    """
    idx = _build_baseline_index(all_records)
    missing_set: set = set()

    for r in all_records:
        if r.get("status") != "complete":
            continue
        if _is_no_tta(r):
            continue

        series = r.get("series", "")
        rid = r.get("run_id", "")
        cond, gen = _infer_cond_gen_for_run(r)
        baseline = None
        bl_source = None

        # ES ablation: baseline is the ES-disabled run
        if series.startswith("es_ablation_") and series != "es_ablation_disable":
            baseline = idx["es_disable"]
            bl_source = "es_ablation_disable/ES_D1"

        elif cond is not None and gen is not None:
            is_ucf = _is_ucf101_series(series)

            series_info = SERIES_INFO.get(series, {})
            sweeps = set(series_info.get("sweep_params", []))
            varies_cond_gen = bool(sweeps & _COND_GEN_SWEEP_PARAMS)

            if is_ucf:
                baseline = idx["baseline_ucf101"].get((cond, gen))
                if baseline:
                    bl_source = f"baseline:ucf101_cond{cond}_gen{gen}"
                else:
                    baseline = idx["no_tta_ucf101"].get((cond, gen))
                    if baseline:
                        bl_source = "ucf101_no_tta/UCF_NOTTA"
            elif varies_cond_gen:
                baseline = idx["baseline_panda"].get((cond, gen))
                if baseline:
                    bl_source = f"baseline:cond{cond}_gen{gen}"
            else:
                baseline = idx["no_tta_panda"].get((cond, gen))
                if baseline:
                    bl_source = "panda_no_tta/NOTTA"
                else:
                    baseline = idx["baseline_panda"].get((cond, gen))
                    if baseline:
                        bl_source = f"baseline:cond{cond}_gen{gen}"

            if baseline is None:
                prefix = "ucf101_" if is_ucf else ""
                missing_set.add(f"{prefix}cond{cond}_gen{gen}")

        if baseline:
            r["baseline_psnr"] = baseline.get("psnr_mean")
            r["baseline_ssim"] = baseline.get("ssim_mean")
            r["baseline_lpips"] = baseline.get("lpips_mean")
            r["baseline_source"] = bl_source
        else:
            r["baseline_psnr"] = None
            r["baseline_ssim"] = None
            r["baseline_lpips"] = None
            r["baseline_source"] = None

    return all_records, sorted(missing_set)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: SERIES METADATA
# ═══════════════════════════════════════════════════════════════════════

SERIES_INFO = {
    "full_lr_sweep": {
        "purpose": "Full-model TTA learning rate sweep (SGD)",
        "sweep_params": ["learning_rate"],
    },
    "full_iter_sweep": {
        "purpose": "Full-model TTA iteration count sweep",
        "sweep_params": ["num_steps"],
    },
    "full_long_train": {
        "purpose": "Full-model long training (500 steps, 30 videos, ES)",
        "sweep_params": [],
    },
    "full_long_train_100v": {
        "purpose": "Full-model long training on 100 videos",
        "sweep_params": ["learning_rate", "num_steps"],
    },
    "delta_a_lr_sweep": {
        "purpose": "AdaSteer-1 (Delta-A, G=1) learning rate sweep",
        "sweep_params": ["delta_lr"],
    },
    "delta_a_iter_sweep": {
        "purpose": "AdaSteer-1 (Delta-A) iteration count sweep",
        "sweep_params": ["delta_steps"],
    },
    "delta_a_long_train": {
        "purpose": "AdaSteer-1 long training (200 steps, 30 videos, ES)",
        "sweep_params": [],
    },
    "delta_a_optimized": {
        "purpose": "AdaSteer-1 with optimized configs (24 cond / 100-step extended)",
        "sweep_params": ["num_cond_frames", "delta_steps"],
    },
    "delta_a_norm_combined": {
        "purpose": "AdaSteer + NormTune combined",
        "sweep_params": [],
    },
    "delta_a_equiv_verify": {
        "purpose": "Verify Delta-A equals Delta-B(G=1)",
        "sweep_params": ["delta_lr"],
    },
    "delta_b_groups_sweep": {
        "purpose": "AdaSteer multi-group (Delta-B) group count sweep",
        "sweep_params": ["num_groups"],
    },
    "delta_b_lr_sweep": {
        "purpose": "AdaSteer multi-group (G=4) learning rate sweep",
        "sweep_params": ["delta_lr"],
    },
    "delta_b_iter_sweep": {
        "purpose": "AdaSteer multi-group (G=1) iteration sweep",
        "sweep_params": ["delta_steps"],
    },
    "delta_b_low_lr": {
        "purpose": "AdaSteer multi-group (G=1) low LR exploration",
        "sweep_params": ["delta_lr"],
    },
    "delta_b_hidden_sweep": {
        "purpose": "AdaSteer hidden-state residual target sweep",
        "sweep_params": ["num_groups", "delta_lr", "delta_target"],
    },
    "delta_c_lr_sweep": {
        "purpose": "Output residual (Delta-C) learning rate sweep",
        "sweep_params": ["delta_lr"],
    },
    "delta_c_iter_sweep": {
        "purpose": "Output residual (Delta-C) iteration sweep",
        "sweep_params": ["delta_steps"],
    },
    "lora_rank_sweep": {
        "purpose": "LoRA rank sweep (r=1..32, all blocks)",
        "sweep_params": ["lora_rank", "lora_alpha"],
    },
    "lora_iter_sweep": {
        "purpose": "LoRA iteration sweep (r=1)",
        "sweep_params": ["num_steps"],
    },
    "lora_constrained_sweep": {
        "purpose": "LoRA constrained: target last N blocks, vary alpha",
        "sweep_params": ["lora_target_blocks", "lora_alpha"],
    },
    "lora_ultra_constrained": {
        "purpose": "LoRA ultra-constrained: last 1-2 blocks, very low alpha",
        "sweep_params": ["lora_target_blocks", "lora_alpha"],
    },
    "lora_builtin_comparison": {
        "purpose": "LoRA: custom vs diffusers built-in implementation",
        "sweep_params": ["use_builtin_lora"],
    },
    "lora_ultra_long_train": {
        "purpose": "Ultra-constrained LoRA extended training (50 steps, ES)",
        "sweep_params": ["lora_target_blocks", "lora_alpha"],
    },
    "adasteer_groups_5step": {
        "purpose": "AdaSteer group sweep at 5 training steps",
        "sweep_params": ["num_groups"],
    },
    "adasteer_ratio_sweep": {
        "purpose": "AdaSteer cond-frames x groups sweep (token-to-param ratio)",
        "sweep_params": ["num_cond_frames", "num_groups"],
    },
    "adasteer_extended_data": {
        "purpose": "AdaSteer with longer input videos (5s / 8s windows)",
        "sweep_params": ["gen_start_frame", "num_groups"],
    },
    "adasteer_param_sweep": {
        "purpose": "AdaSteer parameter dimension sweep (delta_dim, groups)",
        "sweep_params": ["delta_dim", "num_groups"],
    },
    "adasteer_long_train_es": {
        "purpose": "AdaSteer long training with ES (100-200 steps, lower LR)",
        "sweep_params": ["delta_lr", "delta_steps", "num_cond_frames"],
    },
    "best_methods_proper_es": {
        "purpose": "Best methods with properly tuned early stopping",
        "sweep_params": [],
    },
    "norm_tune_sweep": {
        "purpose": "NormTune (TENT-style) norm layer sweep",
        "sweep_params": ["norm_target", "norm_lr"],
    },
    "film_adapter_sweep": {
        "purpose": "FiLM adapter sweep (adaLN modulation correction)",
        "sweep_params": ["num_groups", "film_lr"],
    },
    "es_ablation_disable": {
        "purpose": "ES ablation baseline: early stopping DISABLED",
        "sweep_params": [],
    },
    "es_ablation_check_freq": {
        "purpose": "ES ablation: vary check frequency",
        "sweep_params": ["es_check_every"],
    },
    "es_ablation_patience": {
        "purpose": "ES ablation: vary patience",
        "sweep_params": ["es_patience"],
    },
    "es_ablation_sigmas": {
        "purpose": "ES ablation: vary anchor sigma set",
        "sweep_params": ["es_anchor_sigmas"],
    },
    "es_ablation_noise_draws": {
        "purpose": "ES ablation: vary noise draws per check",
        "sweep_params": ["es_noise_draws"],
    },
    "es_ablation_holdout": {
        "purpose": "ES ablation: vary holdout fraction",
        "sweep_params": ["es_holdout_fraction"],
    },
    "exp3_train_frames_full": {
        "purpose": "Conditioning frames ablation (full-model TTA)",
        "sweep_params": ["num_cond_frames"],
    },
    "exp3_train_frames_lora": {
        "purpose": "Conditioning frames ablation (LoRA TTA)",
        "sweep_params": ["num_cond_frames"],
    },
    "exp3_train_frames_delta_a": {
        "purpose": "Conditioning frames ablation (AdaSteer-1)",
        "sweep_params": ["num_cond_frames"],
    },
    "exp3_train_frames_delta_b": {
        "purpose": "Conditioning frames ablation (AdaSteer multi-group)",
        "sweep_params": ["num_cond_frames"],
    },
    "exp3_train_frames_delta_c": {
        "purpose": "Conditioning frames ablation (Delta-C)",
        "sweep_params": ["num_cond_frames"],
    },
    "exp4_gen_horizon_full": {
        "purpose": "Generation horizon ablation (full-model TTA)",
        "sweep_params": ["num_frames"],
    },
    "exp4_gen_horizon_lora": {
        "purpose": "Generation horizon ablation (LoRA TTA)",
        "sweep_params": ["num_frames"],
    },
    "exp4_gen_horizon_delta_a": {
        "purpose": "Generation horizon ablation (AdaSteer-1)",
        "sweep_params": ["num_frames"],
    },
    "exp4_gen_horizon_delta_b": {
        "purpose": "Generation horizon ablation (AdaSteer multi-group)",
        "sweep_params": ["num_frames"],
    },
    "exp4_gen_horizon_delta_c": {
        "purpose": "Generation horizon ablation (Delta-C)",
        "sweep_params": ["num_frames"],
    },
    "exp5_batch_size_delta_a": {
        "purpose": "Retrieval-augmented batch-size ablation (AdaSteer-1, Panda)",
        "sweep_params": ["batch_videos"],
    },
    "exp5_batch_size_full": {
        "purpose": "Retrieval-augmented batch-size ablation (full-model, Panda)",
        "sweep_params": ["batch_videos"],
    },
    "exp5_batch_size_lora": {
        "purpose": "Retrieval-augmented batch-size ablation (LoRA, Panda)",
        "sweep_params": ["batch_videos"],
    },
    "exp5_batch_size_delta_a_ucf101": {
        "purpose": "Retrieval-augmented batch-size ablation (AdaSteer-1, UCF-101)",
        "sweep_params": ["batch_videos"],
    },
    "exp5_batch_size_full_ucf101": {
        "purpose": "Retrieval-augmented batch-size ablation (full-model, UCF-101)",
        "sweep_params": ["batch_videos"],
    },
    "exp5_batch_size_lora_ucf101": {
        "purpose": "Retrieval-augmented batch-size ablation (LoRA, UCF-101)",
        "sweep_params": ["batch_videos"],
    },
    "ucf101_full": {
        "purpose": "Cross-dataset: full-model TTA on UCF-101",
        "sweep_params": [],
    },
    "ucf101_delta_a": {
        "purpose": "Cross-dataset: AdaSteer-1 on UCF-101",
        "sweep_params": [],
    },
    "ucf101_lora": {
        "purpose": "Cross-dataset: LoRA TTA on UCF-101",
        "sweep_params": [],
    },
    "ucf101_no_tta": {
        "purpose": "Cross-dataset: no-TTA baseline on UCF-101",
        "sweep_params": [],
    },
    "panda_no_tta_continuation": {
        "purpose": "No-TTA baseline (video continuation, 0 training steps)",
        "sweep_params": [],
    },
    "results": {
        "purpose": "Baseline experiment results (no TTA, various cond/gen)",
        "sweep_params": [],
    },
}

# Keys to exclude from "fixed params" display (always present, not informative)
_SUPPRESS_FIXED = {
    "seed", "resolution", "guidance_scale", "num_inference_steps",
    "max_videos", "gen_start_frame", "tta_total_frames", "tta_context_frames",
    "method", "status", "series", "run_id", "path", "n_ok", "n_total",
    "psnr_mean", "psnr_std", "psnr_median", "ssim_mean", "ssim_std",
    "lpips_mean", "lpips_std", "train_time_mean", "train_time_std",
    "gen_time_mean", "gen_time_std", "total_time_mean", "final_loss_mean",
    "es_stopped_count", "es_total_count", "es_best_step_mean",
    "total_params", "trainable_params", "train_gen_ratio",
    "baseline_psnr", "baseline_ssim", "baseline_lpips", "baseline_source",
    "videos_done", "weight_decay", "max_grad_norm", "optimizer",
    "retrieval_pool_dir",
}


def build_series_metadata(all_records: List[dict]) -> dict:
    """Phase 3: compute fixed and sweep params from actual run data."""
    series_groups: Dict[str, List[dict]] = defaultdict(list)
    for r in all_records:
        if r.get("status") == "complete":
            series_groups[r["series"]].append(r)

    metadata = {}
    for series_name, runs in sorted(series_groups.items()):
        info = SERIES_INFO.get(series_name, {})
        methods = set(r.get("method", "") for r in runs)
        method_str = ", ".join(sorted(methods)) if methods else "unknown"

        dataset = "ucf101" if _is_ucf101_series(series_name) else "panda"

        declared_sweep = info.get("sweep_params", [])

        # Compute actually-fixed params: values identical across all runs
        all_keys = set()
        for r in runs:
            all_keys.update(r.keys())
        relevant_keys = all_keys - _SUPPRESS_FIXED

        fixed_params = {}
        for k in sorted(relevant_keys):
            vals = [r.get(k) for r in runs if r.get(k) is not None]
            if not vals:
                continue
            if len(set(str(v) for v in vals)) == 1:
                fixed_params[k] = vals[0]

        metadata[series_name] = {
            "purpose": info.get("purpose", f"Series: {series_name}"),
            "method": method_str,
            "dataset": dataset,
            "n_runs": len(runs),
            "sweep_params": declared_sweep,
            "fixed_params": fixed_params,
        }

    return metadata


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def _fmt(v, width=7, prec=4):
    if v is None:
        return "-".center(width)
    if isinstance(v, float):
        s = f"{v:.{prec}f}"
        return s[:width].rjust(width)
    return str(v)[:width].rjust(width)


def _fmt_delta(run_val, bl_val, width=7, prec=4):
    if run_val is None or bl_val is None:
        return "-".center(width)
    d = run_val - bl_val
    sign = "+" if d >= 0 else ""
    s = f"{sign}{d:.{prec}f}"
    return s[:width].rjust(width)


def _get_sweep_val(rec: dict, sweep_params: list) -> str:
    """Format the swept parameter values for a run."""
    parts = []
    for p in sweep_params:
        v = rec.get(p)
        if v is None:
            cond, gen = _infer_cond_gen_for_run(rec)
            if p == "num_cond_frames":
                v = cond
            elif p in ("num_frames", "num_gen_frames"):
                v = gen
        if v is not None:
            parts.append(f"{v}")
        else:
            parts.append("-")
    return " / ".join(parts) if parts else "-"


# Compact key name mapping for table headers
_KEY_SHORT = {
    "learning_rate": "LR",
    "delta_lr": "LR",
    "num_steps": "steps",
    "delta_steps": "steps",
    "num_groups": "G",
    "lora_rank": "rank",
    "lora_alpha": "alpha",
    "lora_target_blocks": "blocks",
    "num_cond_frames": "cond",
    "num_frames": "nframes",
    "num_gen_frames": "gen",
    "delta_target": "target",
    "delta_dim": "dim",
    "batch_videos": "K",
    "es_check_every": "chk_freq",
    "es_patience": "patience",
    "es_anchor_sigmas": "sigmas",
    "es_noise_draws": "draws",
    "es_holdout_fraction": "holdout",
    "norm_target": "norm_tgt",
    "norm_lr": "LR",
    "film_lr": "LR",
    "film_mode": "mode",
    "use_builtin_lora": "builtin",
}


def print_series_table(series_name: str, runs: List[dict], meta: dict):
    """Print one series as a formatted table."""
    purpose = meta.get("purpose", series_name)
    method = meta.get("method", "?")
    fixed = meta.get("fixed_params", {})
    sweep_params = meta.get("sweep_params", [])

    print(f"\n{'=' * 80}")
    print(f"  {series_name} ({meta.get('n_runs', len(runs))} runs)")
    print(f"{'=' * 80}")
    print(f"  Purpose: {purpose}")
    print(f"  Method : {method}  |  Dataset: {meta.get('dataset', 'panda')}")

    # Show fixed params compactly
    if fixed:
        fixed_parts = []
        for k, v in sorted(fixed.items()):
            short = _KEY_SHORT.get(k, k)
            fixed_parts.append(f"{short}={v}")
        line = "  Fixed  : " + ", ".join(fixed_parts)
        if len(line) > 120:
            half = len(fixed_parts) // 2
            print("  Fixed  : " + ", ".join(fixed_parts[:half]))
            print("           " + ", ".join(fixed_parts[half:]))
        else:
            print(line)

    # Analyse baseline variation across runs
    _bl_info = {}
    for r in runs:
        src = r.get("baseline_source")
        if src:
            _bl_info[r.get("run_id", "?")] = {
                "source": src,
                "psnr": r.get("baseline_psnr"),
                "ssim": r.get("baseline_ssim"),
                "lpips": r.get("baseline_lpips"),
            }
    _unique_sources = set(v["source"] for v in _bl_info.values())
    _has_missing_bl = any(
        r.get("baseline_source") is None
        and r.get("n_ok", 0) > 0
        and not _is_no_tta(r)
        for r in runs
    )
    _show_per_run_bl = len(_unique_sources) > 1 or (
        _has_missing_bl and len(_unique_sources) >= 1
    )

    if len(_unique_sources) == 1 and not _has_missing_bl:
        src = next(iter(_unique_sources))
        info = next(v for v in _bl_info.values())
        parts = []
        if info["psnr"] is not None: parts.append(f"PSNR={info['psnr']:.4f}")
        if info["ssim"] is not None: parts.append(f"SSIM={info['ssim']:.4f}")
        if info["lpips"] is not None: parts.append(f"LPIPS={info['lpips']:.4f}")
        print(f"  Baseln : {src}  ({', '.join(parts)})")
    elif _show_per_run_bl:
        print(f"  Baseln : per-run (varies by cond/gen)")
    elif _has_missing_bl:
        print(f"  Baseln : MISSING")
    print()

    # Build sweep header
    sweep_hdr = " / ".join(_KEY_SHORT.get(p, p) for p in sweep_params) if sweep_params else "config"

    # Table header
    hdr = (f"  {'run_id':>8} | {sweep_hdr:>12} | {'N':>3} | {'cond':>4} {'gen':>3} "
           f"| {'PSNR':>7} {'dPSNR':>7} | {'SSIM':>7} {'dSSIM':>7} "
           f"| {'LPIPS':>7} {'dLPIPS':>7} | {'train_s':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in sorted(runs, key=lambda x: x.get("run_id", "")):
        rid = r.get("run_id", "?")
        sweep_val = _get_sweep_val(r, sweep_params) if sweep_params else rid
        n = r.get("n_ok", 0)
        cond, gen = _infer_cond_gen_for_run(r)

        psnr = r.get("psnr_mean")
        ssim = r.get("ssim_mean")
        lpips = r.get("lpips_mean")
        bl_psnr = r.get("baseline_psnr")
        bl_ssim = r.get("baseline_ssim")
        bl_lpips = r.get("baseline_lpips")
        train_s = r.get("train_time_mean")

        row = (f"  {rid:>8} | {sweep_val:>12} | {n:>3} | {cond or '-':>4} {gen or '-':>3} "
               f"| {_fmt(psnr)} {_fmt_delta(psnr, bl_psnr)} "
               f"| {_fmt(ssim)} {_fmt_delta(ssim, bl_ssim)} "
               f"| {_fmt(lpips)} {_fmt_delta(lpips, bl_lpips, prec=4)} "
               f"| {_fmt(train_s, prec=1)}")
        print(row)

    if _show_per_run_bl:
        print()
        print("  Baselines used:")
        for r in sorted(runs, key=lambda x: x.get("run_id", "")):
            if r.get("n_ok", 0) == 0:
                continue
            rid = r.get("run_id", "?")
            src = r.get("baseline_source")
            if src:
                bl_psnr = r.get("baseline_psnr")
                note = f" (PSNR={bl_psnr:.2f})" if bl_psnr is not None else ""
                print(f"    {rid} -> {src}{note}")
            else:
                c, g = _infer_cond_gen_for_run(r)
                print(f"    {rid} -> MISSING (need cond{c}_gen{g})")


def print_report(all_records: List[dict], metadata: dict, missing: List[str]):
    """Phase 4: print the full console report."""
    series_groups: Dict[str, List[dict]] = defaultdict(list)
    for r in all_records:
        if r.get("status") == "complete":
            series_groups[r["series"]].append(r)

    # Define display order by experiment category
    order = [
        # Baselines
        "panda_no_tta_continuation", "results",
        # Full-model
        "full_lr_sweep", "full_iter_sweep", "full_long_train", "full_long_train_100v",
        # AdaSteer-1
        "delta_a_lr_sweep", "delta_a_iter_sweep", "delta_a_long_train",
        "delta_a_optimized", "delta_a_norm_combined", "delta_a_equiv_verify",
        # AdaSteer multi-group
        "delta_b_groups_sweep", "delta_b_lr_sweep", "delta_b_iter_sweep",
        "delta_b_low_lr", "delta_b_hidden_sweep",
        "adasteer_groups_5step", "adasteer_ratio_sweep",
        "adasteer_extended_data", "adasteer_param_sweep",
        "adasteer_long_train_es", "best_methods_proper_es",
        # Delta-C
        "delta_c_lr_sweep", "delta_c_iter_sweep",
        # LoRA
        "lora_rank_sweep", "lora_iter_sweep",
        "lora_constrained_sweep", "lora_ultra_constrained",
        "lora_builtin_comparison", "lora_ultra_long_train",
        # Naive methods
        "norm_tune_sweep", "film_adapter_sweep",
        # ES ablation
        "es_ablation_disable", "es_ablation_check_freq", "es_ablation_patience",
        "es_ablation_sigmas", "es_ablation_noise_draws", "es_ablation_holdout",
        # Exp3 / Exp4
        "exp3_train_frames_full", "exp3_train_frames_lora",
        "exp3_train_frames_delta_a", "exp3_train_frames_delta_b", "exp3_train_frames_delta_c",
        "exp4_gen_horizon_full", "exp4_gen_horizon_lora",
        "exp4_gen_horizon_delta_a", "exp4_gen_horizon_delta_b", "exp4_gen_horizon_delta_c",
        # Exp5
        "exp5_batch_size_delta_a", "exp5_batch_size_full", "exp5_batch_size_lora",
        "exp5_batch_size_delta_a_ucf101", "exp5_batch_size_full_ucf101", "exp5_batch_size_lora_ucf101",
        # UCF-101
        "ucf101_no_tta", "ucf101_full", "ucf101_delta_a", "ucf101_lora",
    ]

    printed = set()
    for sn in order:
        if sn in series_groups and sn in metadata:
            print_series_table(sn, series_groups[sn], metadata[sn])
            printed.add(sn)

    # Print any remaining series not in the order list
    for sn in sorted(series_groups.keys()):
        if sn not in printed and sn in metadata:
            print_series_table(sn, series_groups[sn], metadata[sn])

    # Summary
    complete = sum(1 for r in all_records if r.get("status") == "complete")
    in_prog = sum(1 for r in all_records if r.get("status") == "in_progress")
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: {complete} complete + {in_prog} in-progress = {len(all_records)} total")
    print(f"{'=' * 80}")

    if missing:
        print(f"\n  MISSING BASELINES (no-TTA results needed for comparison):")
        for m in missing:
            print(f"    - {m}")
        print()

    # In-progress runs
    ip = [r for r in all_records if r.get("status") == "in_progress"]
    if ip:
        print(f"\n  IN-PROGRESS RUNS ({len(ip)}):")
        for r in sorted(ip, key=lambda x: (x.get("series", ""), x.get("run_id", ""))):
            print(f"    - {r['series']}/{r['run_id']}: {r.get('videos_done', '?')} videos done")
        print()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Export all experiment results")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help=f"Output JSON file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--json-only", action="store_true",
                        help="Only write JSON to stdout, no console tables")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Read runs from existing JSON instead of scanning result dirs")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    print("Phase 1: Collecting runs...", file=sys.stderr)

    if args.from_json:
        src = Path(args.from_json)
        with open(src) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            all_records = raw
        elif isinstance(raw, dict) and "runs" in raw:
            all_records = raw["runs"]
        else:
            all_records = []
        print(f"  Loaded {len(all_records)} records from {src}", file=sys.stderr)
    else:
        all_records = collect_all_runs()

    complete = sum(1 for r in all_records if r.get("status") == "complete")
    in_prog = sum(1 for r in all_records if r.get("status") == "in_progress")
    print(f"  Found {complete} complete + {in_prog} in-progress = {len(all_records)} total",
          file=sys.stderr)

    print("Phase 2: Matching baselines...", file=sys.stderr)
    all_records, missing_baselines = match_baselines(all_records)
    matched = sum(1 for r in all_records
                  if r.get("status") == "complete" and r.get("baseline_source"))
    print(f"  Matched {matched} runs to baselines, {len(missing_baselines)} gaps",
          file=sys.stderr)

    print("Phase 3: Building series metadata...", file=sys.stderr)
    metadata = build_series_metadata(all_records)
    print(f"  {len(metadata)} series", file=sys.stderr)

    # Phase 4: output
    output_data = {
        "series_metadata": metadata,
        "runs": all_records,
        "missing_baselines": missing_baselines,
    }

    if args.json_only:
        json.dump(output_data, sys.stdout, indent=2, default=str)
    else:
        print(f"Phase 4: Writing JSON to {output_path}...", file=sys.stderr)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"  Wrote {output_path}", file=sys.stderr)

        print("\n" + "=" * 80, file=sys.stderr)
        print("  CONSOLE REPORT", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

        print_report(all_records, metadata, missing_baselines)


if __name__ == "__main__":
    main()
