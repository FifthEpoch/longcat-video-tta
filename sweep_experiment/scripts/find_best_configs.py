#!/usr/bin/env python3
"""
Experiment 1: Best Config Selection and Parameter Efficiency Chart

Reads all summary.json files from sweep results, selects the best
hyperparameter configuration per TTA method, and generates:
  1. best_configs.json — used by subsequent experiments
  2. A bar chart of performance (PSNR/SSIM/LPIPS) vs. tunable parameter count

Usage:
    python sweep_experiment/scripts/find_best_configs.py \
        --results-dir sweep_experiment/results \
        --output-dir sweep_experiment/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


METHOD_PARAM_COUNTS = {
    "full": 13.6e9,
    "lora": None,  # depends on rank; computed dynamically
    "delta_a": 512,
    "delta_b": None,  # depends on num_groups; computed dynamically
    "delta_c": 16,
}

SERIES_TO_METHOD = {
    "full_lr_sweep": "full",
    "full_iter_sweep": "full",
    "lora_rank_sweep": "lora",
    "lora_lr_sweep": "lora",
    "lora_targets_sweep": "lora",
    "lora_iter_sweep": "lora",
    "delta_a_lr_sweep": "delta_a",
    "delta_a_iter_sweep": "delta_a",
    "delta_b_groups_sweep": "delta_b",
    "delta_b_lr_sweep": "delta_b",
    "delta_c_lr_sweep": "delta_c",
    "delta_c_iter_sweep": "delta_c",
}


def estimate_lora_params(rank: int, target_modules: str = "qkv,proj",
                         target_ffn: bool = False, num_blocks: int = 48,
                         hidden_dim: int = 3072) -> int:
    """Estimate total trainable parameters for a LoRA config."""
    modules_per_block = 0
    targets = target_modules.split(",") if target_modules else ["qkv", "proj"]
    for t in targets:
        t = t.strip()
        if t in ("qkv", "q", "k", "v"):
            modules_per_block += 3 if t == "qkv" else 1
        elif t == "proj":
            modules_per_block += 1

    if target_ffn:
        modules_per_block += 2

    params_per_adapter = 2 * rank * hidden_dim
    return modules_per_block * num_blocks * params_per_adapter


def estimate_delta_b_params(num_groups: int, dim: int = 512) -> int:
    return num_groups * dim


def load_all_summaries(results_dir: str) -> List[Dict[str, Any]]:
    """Walk the results directory tree and load all summary.json files."""
    entries = []
    results_path = Path(results_dir)

    for series_dir in sorted(results_path.iterdir()):
        if not series_dir.is_dir():
            continue
        series_name = series_dir.name

        for run_dir in sorted(series_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with open(summary_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  WARNING: Failed to load {summary_path}: {e}",
                      file=sys.stderr)
                continue

            method = data.get("method", SERIES_TO_METHOD.get(series_name, "unknown"))
            run_id = run_dir.name

            successful = [r for r in data.get("results", [])
                          if r.get("success", False)]

            if not successful:
                print(f"  Skipping {series_name}/{run_id}: no successful videos")
                continue

            psnr_vals = [r["psnr"] for r in successful
                         if "psnr" in r and not np.isnan(r["psnr"])]
            ssim_vals = [r["ssim"] for r in successful
                         if "ssim" in r and not np.isnan(r["ssim"])]
            lpips_vals = [r["lpips"] for r in successful
                          if "lpips" in r and not np.isnan(r["lpips"])]

            if not psnr_vals:
                print(f"  Skipping {series_name}/{run_id}: no valid PSNR values")
                continue

            num_params = METHOD_PARAM_COUNTS.get(method, 0)
            if method == "lora":
                rank = data.get("lora_rank", 8)
                target_modules = data.get("target_modules", "qkv,proj")
                target_ffn = data.get("target_ffn", False)
                num_params = estimate_lora_params(rank, target_modules, target_ffn)
            elif method == "delta_b":
                num_groups = data.get("num_groups", 4)
                num_params = estimate_delta_b_params(num_groups)

            entry = {
                "series_name": series_name,
                "run_id": run_id,
                "method": method,
                "num_params": num_params,
                "psnr_mean": float(np.mean(psnr_vals)),
                "psnr_std": float(np.std(psnr_vals)),
                "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
                "ssim_std": float(np.std(ssim_vals)) if ssim_vals else float("nan"),
                "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
                "lpips_std": float(np.std(lpips_vals)) if lpips_vals else float("nan"),
                "num_successful": len(successful),
                "num_total": len(data.get("results", [])),
                "avg_train_time": data.get("avg_train_time", 0),
                "config": {k: v for k, v in data.items()
                           if k not in ("results", "method")},
            }
            entries.append(entry)

    return entries


def select_best_per_method(entries: List[Dict]) -> Dict[str, Dict]:
    """Select the best run per method, ranked by PSNR > SSIM > LPIPS."""
    by_method = defaultdict(list)
    for e in entries:
        by_method[e["method"]].append(e)

    best = {}
    for method, runs in sorted(by_method.items()):
        runs_sorted = sorted(
            runs,
            key=lambda r: (
                r["psnr_mean"],
                r.get("ssim_mean", 0) if not np.isnan(r.get("ssim_mean", 0)) else 0,
                -(r.get("lpips_mean", 1) if not np.isnan(r.get("lpips_mean", 1)) else 1),
            ),
            reverse=True,
        )
        winner = runs_sorted[0]
        best[method] = winner
        print(f"  Best {method:>8}: {winner['series_name']}/{winner['run_id']} "
              f"PSNR={winner['psnr_mean']:.2f} "
              f"SSIM={winner['ssim_mean']:.4f} "
              f"LPIPS={winner['lpips_mean']:.4f} "
              f"({winner['num_successful']}/{winner['num_total']} videos)")

    return best


def generate_bar_chart(best: Dict[str, Dict], output_path: str):
    """Generate a bar chart of metrics vs. method (ordered by param count)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed, skipping chart generation.",
              file=sys.stderr)
        return

    methods_sorted = sorted(best.keys(),
                            key=lambda m: best[m].get("num_params", 0))
    labels = []
    psnr_vals = []
    psnr_errs = []
    ssim_vals = []
    lpips_vals = []
    param_counts = []

    for m in methods_sorted:
        info = best[m]
        nparams = info.get("num_params", 0)
        if nparams >= 1e9:
            plabel = f"{nparams/1e9:.1f}B"
        elif nparams >= 1e6:
            plabel = f"{nparams/1e6:.1f}M"
        elif nparams >= 1e3:
            plabel = f"{nparams/1e3:.0f}K"
        else:
            plabel = str(int(nparams))

        labels.append(f"{m}\n({plabel})")
        psnr_vals.append(info["psnr_mean"])
        psnr_errs.append(info["psnr_std"])
        ssim_vals.append(info.get("ssim_mean", 0))
        lpips_vals.append(info.get("lpips_mean", 0))
        param_counts.append(nparams)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(labels))
    width = 0.6
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    axes[0].bar(x, psnr_vals, width, yerr=psnr_errs, color=colors,
                capsize=5, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("PSNR (dB) ↑", fontsize=12)
    axes[0].set_title("Peak Signal-to-Noise Ratio", fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, ssim_vals, width, color=colors,
                edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("SSIM ↑", fontsize=12)
    axes[1].set_title("Structural Similarity", fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, lpips_vals, width, color=colors,
                edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("LPIPS ↓", fontsize=12)
    axes[2].set_title("Perceptual Distance", fontsize=13)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("TTA Method Performance vs. Tunable Parameters",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select best TTA configs and generate parameter efficiency chart")
    parser.add_argument("--results-dir", type=str,
                        default="sweep_experiment/results",
                        help="Directory containing sweep results")
    parser.add_argument("--output-dir", type=str,
                        default="sweep_experiment/results",
                        help="Directory to write best_configs.json and chart")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 1: Best Config Selection")
    print("=" * 70)

    print("\nLoading sweep results...")
    entries = load_all_summaries(args.results_dir)
    print(f"  Found {len(entries)} valid runs")

    if not entries:
        print("ERROR: No valid results found. Ensure sweeps have completed.",
              file=sys.stderr)
        sys.exit(1)

    print("\nSelecting best config per method...")
    best = select_best_per_method(entries)

    output_json = os.path.join(args.output_dir, "best_configs.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(best, f, indent=2, default=str)
    print(f"\nBest configs saved to {output_json}")

    chart_path = os.path.join(args.output_dir, "param_efficiency_chart.png")
    generate_bar_chart(best, chart_path)

    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Method':<12} {'Run':<8} {'#Params':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 60)
    for m in sorted(best.keys(),
                    key=lambda k: best[k].get("num_params", 0)):
        info = best[m]
        nparams = info.get("num_params", 0)
        if nparams >= 1e9:
            ps = f"{nparams/1e9:.1f}B"
        elif nparams >= 1e6:
            ps = f"{nparams/1e6:.1f}M"
        else:
            ps = str(int(nparams))
        print(f"{m:<12} {info['run_id']:<8} {ps:<12} "
              f"{info['psnr_mean']:>8.2f} "
              f"{info['ssim_mean']:>8.4f} "
              f"{info['lpips_mean']:>8.4f}")


if __name__ == "__main__":
    main()
