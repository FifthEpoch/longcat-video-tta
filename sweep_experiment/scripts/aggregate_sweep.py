#!/usr/bin/env python3
"""
Aggregate TTA sweep results and produce comparison tables + plots.

Scans sweep_experiment/results/<series_name>/<run_id>/summary.json for
all completed runs, collects metrics, and produces:

  1. CSV table: method, run_id, trainable_params, lr, steps, PSNR, SSIM, LPIPS, train_time
  2. Pareto frontier plot: PSNR vs. trainable parameters
  3. Overfitting curves: PSNR vs. training steps (one line per method at best LR)
  4. LR sensitivity curves: PSNR vs. learning rate (one line per method)

Usage:
  # Aggregate all series:
  python sweep_experiment/scripts/aggregate_sweep.py \
      --results-dir sweep_experiment/results \
      --output-dir sweep_experiment/plots

  # Aggregate specific series:
  python sweep_experiment/scripts/aggregate_sweep.py \
      --results-dir sweep_experiment/results \
      --output-dir sweep_experiment/plots \
      --series full_lr_sweep lora_rank_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Trainable parameter estimates
# ============================================================================
# These are approximate counts based on LongCat-Video's architecture:
#   hidden_size=4096, depth=48, adaln_tembed_dim=512, out_channels=16

_METHOD_NORMALIZE = {
    "full_tta": "full",
    "lora_tta": "lora",
    "full": "full",
    "lora": "lora",
    "delta_a": "delta_a",
    "delta_b": "delta_b",
    "delta_c": "delta_c",
}


def estimate_params(method: str, config: dict) -> int:
    """Estimate the number of trainable parameters for a given method+config."""
    if method == "full":
        return config.get("total_params", 13_600_000_000)

    elif method == "lora":
        rank = config.get("lora_rank", 8)
        target_ffn = config.get("target_ffn", False)
        # Per block: qkv (4096->12288) + proj (4096->4096) = 2 layers in self-attn
        #            q (4096->4096) + kv (4096->8192) + proj (4096->4096) = 3 in cross-attn
        # Total attention params per block: 5 LoRA adapters
        # Each LoRA: 2 * rank * (in + out) ... but stored as down+up
        # Actually: down = rank*in, up = rank*out for each adapter

        blocks = 48
        # Self-attention
        sa_qkv = rank * 4096 + rank * 12288  # down + up
        sa_proj = rank * 4096 + rank * 4096
        # Cross-attention
        xa_q = rank * 4096 + rank * 4096
        xa_kv = rank * 4096 + rank * 8192
        xa_proj = rank * 4096 + rank * 4096

        per_block = sa_qkv + sa_proj + xa_q + xa_kv + xa_proj

        if target_ffn:
            # ffn.w1 (4096->16384), w2 (16384->4096), w3 (4096->16384)
            ffn_w1 = rank * 4096 + rank * 16384
            ffn_w2 = rank * 16384 + rank * 4096
            ffn_w3 = rank * 4096 + rank * 16384
            per_block += ffn_w1 + ffn_w2 + ffn_w3

        return blocks * per_block

    elif method == "delta_a":
        return 512  # adaln_tembed_dim

    elif method == "delta_b":
        num_groups = config.get("num_groups", 4)
        return num_groups * 512  # num_groups * adaln_tembed_dim

    elif method == "delta_c":
        return 16  # out_channels

    return 0


# ============================================================================
# Method display names and colors
# ============================================================================
METHOD_DISPLAY = {
    "full": "Full-Model",
    "lora": "LoRA",
    "delta_a": "Delta-A (Global)",
    "delta_b": "Delta-B (Per-Layer)",
    "delta_c": "Delta-C (Output)",
}

METHOD_COLORS = {
    "full": "#E8573A",
    "lora": "#3B7DD8",
    "delta_a": "#2ECC71",
    "delta_b": "#F39C12",
    "delta_c": "#9B59B6",
}

METHOD_MARKERS = {
    "full": "s",
    "lora": "o",
    "delta_a": "^",
    "delta_b": "D",
    "delta_c": "v",
}


# ============================================================================
# Data collection
# ============================================================================
def collect_results(results_dir: Path, series_filter: Optional[List[str]] = None) -> List[dict]:
    """Scan results directory and collect all completed run summaries."""
    rows = []

    if not results_dir.exists():
        print(f"WARNING: Results directory does not exist: {results_dir}", file=sys.stderr)
        return rows

    for series_dir in sorted(results_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        series_name = series_dir.name

        if series_filter and series_name not in series_filter:
            continue

        for run_dir in sorted(series_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            summary_path = run_dir / "summary.json"

            if not summary_path.exists():
                print(f"  SKIP: {series_name}/{run_id} (no summary.json)")
                continue

            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  SKIP: {series_name}/{run_id} (invalid summary: {e})")
                continue

            # Normalize method name (full_tta -> full, lora_tta -> lora)
            raw_method = summary.get("method", "")
            method = _METHOD_NORMALIZE.get(raw_method, raw_method)
            if not method:
                for candidate in ("full", "lora", "delta_a", "delta_b", "delta_c"):
                    if candidate in series_name:
                        method = candidate
                        break

            # The TTA scripts write a flat summary with per-video results
            # in summary["results"]. Try to extract metrics from there.
            per_video = summary.get("results", [])
            successful = [r for r in per_video if r.get("success", False)]

            # Try nested metrics format first, then compute from per-video
            metrics = summary.get("metrics", {})
            if metrics and "psnr" in metrics:
                psnr_mean = metrics["psnr"].get("mean", float("nan"))
                psnr_std = metrics["psnr"].get("std", float("nan"))
                ssim_mean = metrics["ssim"].get("mean", float("nan"))
                ssim_std = metrics["ssim"].get("std", float("nan"))
                lpips_mean = metrics["lpips"].get("mean", float("nan"))
                lpips_std = metrics["lpips"].get("std", float("nan"))
            else:
                psnr_vals = [r["psnr"] for r in successful if "psnr" in r and r["psnr"] is not None]
                ssim_vals = [r["ssim"] for r in successful if "ssim" in r and r["ssim"] is not None]
                lpips_vals = [r["lpips"] for r in successful if "lpips" in r and r["lpips"] is not None]
                psnr_mean = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
                psnr_std = float(np.std(psnr_vals)) if psnr_vals else float("nan")
                ssim_mean = float(np.mean(ssim_vals)) if ssim_vals else float("nan")
                ssim_std = float(np.std(ssim_vals)) if ssim_vals else float("nan")
                lpips_mean = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
                lpips_std = float(np.std(lpips_vals)) if lpips_vals else float("nan")

            # Extract config — TTA scripts use flat keys at the top level
            config = summary.get("config", summary)

            # Count errors
            errors = [r for r in per_video if not r.get("success", False)]
            error_msgs = list({r.get("error", "unknown")[:80] for r in errors[:5]}) if errors else []

            row = {
                "series": series_name,
                "run_id": run_id,
                "method": method,
                "trainable_params": estimate_params(method, config),
                "learning_rate": config.get("learning_rate",
                                            config.get("delta_lr", 0)),
                "num_steps": config.get("num_steps",
                                        config.get("delta_steps", 0)),
                "psnr_mean": psnr_mean,
                "psnr_std": psnr_std,
                "ssim_mean": ssim_mean,
                "ssim_std": ssim_std,
                "lpips_mean": lpips_mean,
                "lpips_std": lpips_std,
                "avg_train_time": summary.get("avg_train_time", 0),
                "num_successful": summary.get("num_successful", len(successful)),
                "num_videos": summary.get("num_videos", len(per_video)),
                "num_errors": len(errors),
                "error_samples": error_msgs,
            }

            if method == "lora":
                row["lora_rank"] = config.get("lora_rank", 0)
                row["target_ffn"] = config.get("target_ffn", False)

            if method == "delta_b":
                row["num_groups"] = config.get("num_groups", 0)

            rows.append(row)

    return rows


# ============================================================================
# CSV output
# ============================================================================
def write_csv(rows: List[dict], output_path: Path):
    """Write collected results to CSV."""
    if not rows:
        print("WARNING: No rows to write to CSV")
        return

    fieldnames = [
        "series", "run_id", "method", "trainable_params",
        "learning_rate", "num_steps",
        "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
        "lpips_mean", "lpips_std",
        "avg_train_time", "num_successful", "num_videos", "num_errors",
    ]

    # Add optional fields if present
    if any("lora_rank" in r for r in rows):
        fieldnames.insert(4, "lora_rank")
    if any("num_groups" in r for r in rows):
        fieldnames.insert(4, "num_groups")
    if any("target_ffn" in r for r in rows):
        fieldnames.insert(4, "target_ffn")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"  CSV saved: {output_path}")


# ============================================================================
# Console table
# ============================================================================
def print_table(rows: List[dict]):
    """Print a formatted comparison table to stdout."""
    if not rows:
        print("No results found.")
        return

    def _fmt(val, fmt_str):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "   N/A"
        return f"{val:{fmt_str}}"

    print()
    print(f"{'Series':<25} {'Run':>6} {'Method':<12} {'Params':>12} "
          f"{'LR':>10} {'Steps':>6} "
          f"{'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Time(s)':>8} "
          f"{'OK/Tot':>8}")
    print("-" * 130)

    for row in sorted(rows, key=lambda r: (r["method"], r["series"], r["run_id"])):
        params_str = _format_params(row["trainable_params"])
        ok = row.get("num_successful", 0)
        tot = row.get("num_videos", 0)
        print(f"{row['series']:<25} {row['run_id']:>6} {row['method']:<12} "
              f"{params_str:>12} {row['learning_rate']:>10.1e} "
              f"{row['num_steps']:>6} "
              f"{_fmt(row['psnr_mean'], '>8.2f')} {_fmt(row['ssim_mean'], '>8.4f')} "
              f"{_fmt(row['lpips_mean'], '>8.4f')} {row['avg_train_time']:>8.1f} "
              f"{ok:>3}/{tot:<3}")

    print("-" * 130)

    # Show error samples if any runs have errors
    error_runs = [r for r in rows if r.get("num_errors", 0) > 0]
    if error_runs:
        print(f"\nRuns with errors ({len(error_runs)}):")
        for r in error_runs[:10]:
            msgs = r.get("error_samples", [])
            msg_str = msgs[0] if msgs else "unknown"
            print(f"  {r['series']}/{r['run_id']}: "
                  f"{r['num_errors']} errors, e.g.: {msg_str}")
    print()


def _format_params(n: int) -> str:
    """Format parameter count with appropriate suffix."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


# ============================================================================
# Plots
# ============================================================================
def _has_valid_data(rows: List[dict], metric_key: str = "psnr_mean") -> bool:
    """Check if any row has a valid (non-NaN) metric value."""
    return any(
        not np.isnan(row.get(metric_key, float("nan")))
        for row in rows
    )


def _filter_valid(rows: List[dict], metric_key: str = "psnr_mean") -> List[dict]:
    """Return only rows with valid (non-NaN) values for a metric."""
    return [r for r in rows if not np.isnan(r.get(metric_key, float("nan")))]


def plot_pareto(rows: List[dict], output_dir: Path):
    """Plot PSNR vs. trainable parameters (Pareto frontier)."""
    valid = _filter_valid(rows, "psnr_mean")
    if not valid:
        print("  SKIP plot_pareto: no valid PSNR data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = {}
    for row in valid:
        m = row["method"]
        if m not in methods:
            methods[m] = {"params": [], "psnr": [], "labels": []}
        methods[m]["params"].append(max(row["trainable_params"], 1))
        methods[m]["psnr"].append(row["psnr_mean"])
        methods[m]["labels"].append(row["run_id"])

    for method, data in sorted(methods.items()):
        color = METHOD_COLORS.get(method, "#333333")
        marker = METHOD_MARKERS.get(method, "o")
        label = METHOD_DISPLAY.get(method, method)

        ax.scatter(
            data["params"], data["psnr"],
            c=color, marker=marker, s=80, label=label,
            edgecolors="white", linewidths=0.5, zorder=5,
        )

        for i, txt in enumerate(data["labels"]):
            ax.annotate(
                txt, (data["params"][i], data["psnr"][i]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.7,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("TTA Performance vs. Capacity: Pareto Frontier", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "pareto_psnr_vs_params.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def plot_lr_sensitivity(rows: List[dict], output_dir: Path):
    """Plot PSNR vs. learning rate for each method."""
    valid = _filter_valid(rows, "psnr_mean")
    if not valid:
        print("  SKIP plot_lr_sensitivity: no valid PSNR data")
        return

    lr_series = {}
    for row in valid:
        m = row["method"]
        if m not in lr_series:
            lr_series[m] = {"lr": [], "psnr": [], "psnr_std": []}
        lr_series[m]["lr"].append(row["learning_rate"])
        lr_series[m]["psnr"].append(row["psnr_mean"])
        lr_series[m]["psnr_std"].append(row.get("psnr_std", 0))

    if not lr_series:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, data in sorted(lr_series.items()):
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_DISPLAY.get(method, method)

        sorted_idx = np.argsort(data["lr"])
        lrs = np.array(data["lr"])[sorted_idx]
        psnrs = np.array(data["psnr"])[sorted_idx]
        stds = np.array(data["psnr_std"])[sorted_idx]
        stds = np.nan_to_num(stds, nan=0.0)

        ax.errorbar(
            lrs, psnrs, yerr=stds,
            color=color, marker="o", markersize=6,
            label=label, capsize=3, linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("TTA Performance vs. Learning Rate", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "lr_sensitivity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def plot_iter_curves(rows: List[dict], output_dir: Path):
    """Plot PSNR vs. training steps for each method (overfitting curves)."""
    valid = _filter_valid(rows, "psnr_mean")
    if not valid:
        print("  SKIP plot_iter_curves: no valid PSNR data")
        return

    iter_data = {}
    for row in valid:
        m = row["method"]
        if m not in iter_data:
            iter_data[m] = {"steps": [], "psnr": [], "psnr_std": []}
        iter_data[m]["steps"].append(row["num_steps"])
        iter_data[m]["psnr"].append(row["psnr_mean"])
        iter_data[m]["psnr_std"].append(row.get("psnr_std", 0))

    if not iter_data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, data in sorted(iter_data.items()):
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_DISPLAY.get(method, method)

        sorted_idx = np.argsort(data["steps"])
        steps = np.array(data["steps"])[sorted_idx]
        psnrs = np.array(data["psnr"])[sorted_idx]
        stds = np.nan_to_num(np.array(data["psnr_std"])[sorted_idx], nan=0.0)

        ax.errorbar(
            steps, psnrs, yerr=stds,
            color=color, marker="o", markersize=6,
            label=label, capsize=3, linewidth=1.5,
        )

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("TTA Performance vs. Training Iterations", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "iter_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def plot_capacity_curve(rows: List[dict], output_dir: Path):
    """Plot PSNR vs. capacity for Delta-B groups sweep and LoRA rank sweep."""
    valid = _filter_valid(rows, "psnr_mean")
    if not valid:
        print("  SKIP plot_capacity_curve: no valid PSNR data")
        return

    capacity_data = {}
    for row in valid:
        m = row["method"]
        if m in ("delta_b", "lora", "delta_a", "delta_c"):
            if m not in capacity_data:
                capacity_data[m] = {"params": [], "psnr": [], "psnr_std": [], "labels": []}
            capacity_data[m]["params"].append(max(row["trainable_params"], 1))
            capacity_data[m]["psnr"].append(row["psnr_mean"])
            capacity_data[m]["psnr_std"].append(row.get("psnr_std", 0))
            if m == "delta_b":
                capacity_data[m]["labels"].append(f"g={row.get('num_groups', '?')}")
            elif m == "lora":
                capacity_data[m]["labels"].append(f"r={row.get('lora_rank', '?')}")
            else:
                capacity_data[m]["labels"].append(row["run_id"])

    if not capacity_data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, data in sorted(capacity_data.items()):
        color = METHOD_COLORS.get(method, "#333333")
        marker = METHOD_MARKERS.get(method, "o")
        label = METHOD_DISPLAY.get(method, method)

        sorted_idx = np.argsort(data["params"])
        params = np.array(data["params"])[sorted_idx]
        psnrs = np.array(data["psnr"])[sorted_idx]
        stds = np.nan_to_num(np.array(data["psnr_std"])[sorted_idx], nan=0.0)
        labels = np.array(data["labels"])[sorted_idx]

        ax.errorbar(
            params, psnrs, yerr=stds,
            color=color, marker=marker, markersize=8,
            label=label, capsize=3, linewidth=1.5,
        )

        for i, txt in enumerate(labels):
            ax.annotate(
                txt, (params[i], psnrs[i]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.7,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("TTA Performance vs. Model Capacity", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "capacity_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def plot_all_metrics_pareto(rows: List[dict], output_dir: Path):
    """Create a 3-panel plot: PSNR, SSIM, LPIPS vs. trainable params."""
    has_any = any(
        _has_valid_data(rows, k)
        for k in ("psnr_mean", "ssim_mean", "lpips_mean")
    )
    if not has_any:
        print("  SKIP plot_all_metrics_pareto: no valid metric data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("psnr_mean", "PSNR (dB)", "higher is better"),
        ("ssim_mean", "SSIM", "higher is better"),
        ("lpips_mean", "LPIPS", "lower is better"),
    ]

    for ax, (metric_key, ylabel, direction) in zip(axes, metrics):
        valid = _filter_valid(rows, metric_key)
        if not valid:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")
            ax.set_title(f"{ylabel} ({direction})", fontsize=11)
            continue

        methods = {}
        for row in valid:
            m = row["method"]
            if m not in methods:
                methods[m] = {"params": [], "val": []}
            methods[m]["params"].append(max(row["trainable_params"], 1))
            methods[m]["val"].append(row[metric_key])

        for method, data in sorted(methods.items()):
            color = METHOD_COLORS.get(method, "#333333")
            marker = METHOD_MARKERS.get(method, "o")
            label = METHOD_DISPLAY.get(method, method)

            ax.scatter(
                data["params"], data["val"],
                c=color, marker=marker, s=60, label=label,
                edgecolors="white", linewidths=0.5, zorder=5,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Trainable Params", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel} ({direction})", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("TTA Sweep: All Metrics vs. Capacity", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = output_dir / "all_metrics_pareto.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate TTA sweep results into tables and plots",
    )
    parser.add_argument("--results-dir", type=str,
                        default="sweep_experiment/results",
                        help="Root directory containing series/run_id/ subdirs")
    parser.add_argument("--output-dir", type=str,
                        default="sweep_experiment/plots",
                        help="Directory for output CSV and plots")
    parser.add_argument("--series", nargs="*", type=str, default=None,
                        help="Only aggregate specific series (by directory name)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"Aggregating TTA sweep results")
    print(f"{'=' * 70}")
    print(f"  Results dir : {results_dir}")
    print(f"  Output dir  : {output_dir}")
    print()

    rows = collect_results(results_dir, args.series)

    if not rows:
        print("No results found. Run some sweep jobs first!")
        return

    print(f"\nCollected {len(rows)} results")

    # Print table
    print_table(rows)

    # Write CSV
    csv_path = output_dir / "sweep_results.csv"
    write_csv(rows, csv_path)

    # Generate plots
    print("\nGenerating plots...")
    plot_pareto(rows, output_dir)
    plot_lr_sensitivity(rows, output_dir)
    plot_iter_curves(rows, output_dir)
    plot_capacity_curve(rows, output_dir)
    plot_all_metrics_pareto(rows, output_dir)

    # Save combined JSON
    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")

    print(f"\n{'=' * 70}")
    print(f"Aggregation complete. {len(rows)} results processed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
