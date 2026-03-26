#!/usr/bin/env python3
"""
Compare TTA methods: Baseline, LoRA, Delta-A/B/C.

Aggregates summary.json / eval_summary.json from each method's results
directory and outputs a comparison table + JSON.

Usage:
    python compare_methods.py \\
        --baseline-dir results/baseline/eval \\
        --lora-dir results/lora_best/eval \\
        --delta-a-dir results/delta_a \\
        --delta-b-dir results/delta_b \\
        --delta-c-dir results/delta_c \\
        --output-dir results/comparison
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_summary(result_dir: str) -> Optional[Dict]:
    """Load a summary.json or eval_summary.json from a results directory."""
    for fname in ["eval_summary.json", "summary.json", "metrics_summary.json"]:
        path = os.path.join(result_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def extract_eval_metrics(summary: Dict) -> Dict:
    """Extract standardized metrics from a summary dict."""
    metrics = {
        "num_videos": summary.get("num_videos", summary.get("num_valid", 0)),
    }

    # Eval-style summary (from evaluate_delta.py)
    for key in ["avg_psnr", "std_psnr", "avg_ssim", "std_ssim",
                 "avg_lpips", "std_lpips", "avg_gen_time"]:
        if key in summary:
            metrics[key] = summary[key]

    # Training-time summary (from run_delta_*.py)
    if "avg_train_time" in summary:
        metrics["avg_train_time"] = summary["avg_train_time"]

    # Method-specific info
    if "method" in summary:
        metrics["method"] = summary["method"]
    if "delta_steps" in summary:
        metrics["delta_steps"] = summary["delta_steps"]
    if "delta_lr" in summary:
        metrics["delta_lr"] = summary["delta_lr"]

    # Per-video loss stats (from training)
    if "results" in summary:
        train_results = [r for r in summary["results"] if "final_loss" in r and r["final_loss"] is not None]
        if train_results:
            losses = [r["final_loss"] for r in train_results]
            metrics["avg_final_loss"] = float(np.mean(losses))
            metrics["std_final_loss"] = float(np.std(losses))
            metrics["num_trained"] = len(train_results)

            # Delta norms
            if "delta_norm" in train_results[0]:
                norms = [r["delta_norm"] for r in train_results]
                metrics["avg_delta_norm"] = float(np.mean(norms))
            elif "delta_norms" in train_results[0]:
                all_norms = [np.mean(r["delta_norms"]) for r in train_results]
                metrics["avg_delta_norm"] = float(np.mean(all_norms))
            elif "delta_out_norm" in train_results[0]:
                norms = [r["delta_out_norm"] for r in train_results]
                metrics["avg_delta_norm"] = float(np.mean(norms))

            # Train times
            train_times = [r.get("train_time", 0) for r in train_results if "train_time" in r]
            if train_times:
                metrics["avg_train_time"] = float(np.mean(train_times))
                metrics["std_train_time"] = float(np.std(train_times))

            # Early stopping stats
            es_entries = [r.get("early_stopping_info") for r in train_results
                          if r.get("early_stopping_info")]
            if es_entries:
                stopped_early = sum(1 for e in es_entries if e.get("stopped_early", False))
                best_steps = [e["best_step"] for e in es_entries if "best_step" in e]
                metrics["early_stop_rate"] = stopped_early / len(es_entries)
                metrics["avg_best_step"] = float(np.mean(best_steps)) if best_steps else None

    return metrics


def format_table(methods: Dict[str, Dict]) -> str:
    """Format a comparison table as a string."""
    # Header
    cols = ["Method", "Videos", "Avg Loss", "δ Norm", "Train(s)",
            "PSNR", "SSIM", "LPIPS", "ES Rate"]
    widths = [15, 7, 10, 10, 9, 8, 8, 8, 8]

    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)

    lines = [header, sep]

    for name, m in methods.items():
        row = [
            name[:15].ljust(15),
            str(m.get("num_trained", m.get("num_videos", "-"))).ljust(7),
            f"{m.get('avg_final_loss', 0):.4f}".ljust(10) if "avg_final_loss" in m else "-".ljust(10),
            f"{m.get('avg_delta_norm', 0):.4f}".ljust(10) if "avg_delta_norm" in m else "-".ljust(10),
            f"{m.get('avg_train_time', 0):.1f}".ljust(9) if "avg_train_time" in m else "-".ljust(9),
            f"{m.get('avg_psnr', 0):.2f}".ljust(8) if "avg_psnr" in m else "-".ljust(8),
            f"{m.get('avg_ssim', 0):.4f}".ljust(8) if "avg_ssim" in m else "-".ljust(8),
            f"{m.get('avg_lpips', 0):.4f}".ljust(8) if "avg_lpips" in m else "-".ljust(8),
            f"{m.get('early_stop_rate', 0):.1%}".ljust(8) if "early_stop_rate" in m else "-".ljust(8),
        ]
        lines.append(" | ".join(row))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare TTA methods")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Path to baseline eval results")
    parser.add_argument("--lora-dir", type=str, default=None,
                        help="Path to LoRA eval results")
    parser.add_argument("--delta-a-dir", type=str, default=None,
                        help="Path to Delta-A results")
    parser.add_argument("--delta-b-dir", type=str, default=None,
                        help="Path to Delta-B results")
    parser.add_argument("--delta-c-dir", type=str, default=None,
                        help="Path to Delta-C results")
    parser.add_argument("--output-dir", type=str, default="results/comparison",
                        help="Where to write comparison output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect methods
    method_dirs = {
        "Baseline": args.baseline_dir,
        "LoRA": args.lora_dir,
        "Delta-A": args.delta_a_dir,
        "Delta-B": args.delta_b_dir,
        "Delta-C": args.delta_c_dir,
    }

    methods = {}
    for name, d in method_dirs.items():
        if d and os.path.isdir(d):
            summary = load_summary(d)
            if summary:
                metrics = extract_eval_metrics(summary)
                metrics["source_dir"] = d
                methods[name] = metrics
                print(f"✓ Loaded {name} from {d}")
            else:
                print(f"✗ No summary found in {d}")
        else:
            if d:
                print(f"✗ Directory not found: {d}")

    if not methods:
        print("\nNo methods to compare. Provide at least one --*-dir argument.")
        return

    # Print table
    print("\n" + "=" * 110)
    print("TTA Method Comparison")
    print("=" * 110)
    table = format_table(methods)
    print(table)
    print("=" * 110)

    # Save comparison
    comparison = {
        "methods": methods,
        "table": table,
    }

    output_path = os.path.join(args.output_dir, "comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved to: {output_path}")

    # Also save table as text
    table_path = os.path.join(args.output_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write("TTA Method Comparison\n")
        f.write("=" * 110 + "\n")
        f.write(table + "\n")
        f.write("=" * 110 + "\n")
    print(f"Table saved to: {table_path}")


if __name__ == "__main__":
    main()
