#!/usr/bin/env python3
"""
Unified comparison of all video prediction methods.

Loads summary.json from each method's results and our LongCat TTA results,
then prints a comprehensive comparison table.

Usage:
    python compare_all.py \
        --results-dir comparison_methods/results \
        --longcat-dir sweep_experiment/results
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_metric(summary, key, default=None):
    if summary is None:
        return default
    if key in summary:
        return summary[key]
    avg_key = "avg_" + key
    if avg_key in summary:
        return summary[avg_key]
    return default


def fmt(val, decimals=4):
    if val is None:
        return "---"
    if isinstance(val, str):
        return val
    return "%.*f" % (decimals, val)


def print_table(rows, title=""):
    if not rows:
        return
    headers = list(rows[0].keys())
    col_widths = {}
    for h in headers:
        col_widths[h] = len(h)
        for row in rows:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))

    header_line = "  ".join(h.rjust(col_widths[h]) for h in headers)
    sep_line = "  ".join("-" * col_widths[h] for h in headers)

    if title:
        print()
        print("=" * len(header_line))
        print(title)
        print("=" * len(header_line))
    print(header_line)
    print(sep_line)
    for row in rows:
        line = "  ".join(str(row.get(h, "")).rjust(col_widths[h]) for h in headers)
        print(line)
    print()


def print_latex_table(rows, title=""):
    if not rows:
        return
    headers = list(rows[0].keys())
    print()
    print("%% LaTeX table: " + title)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{%s}" % title)
    col_fmt = "l" + "r" * (len(headers) - 1)
    print("\\begin{tabular}{%s}" % col_fmt)
    print("\\toprule")
    print(" & ".join(headers) + " \\\\")
    print("\\midrule")
    for row in rows:
        vals = [str(row.get(h, "---")) for h in headers]
        print(" & ".join(vals) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--longcat-dir", default="")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    methods = []

    # --- External methods ---
    for name, subdir, res, frames in [
        ("PVDM (baseline)", "pvdm_baseline", "256x256", "16+16"),
        ("PVDM+SAVi-DNO (10s)", "savi_dno_s10", "256x256", "16+16"),
        ("PVDM+SAVi-DNO (50s)", "savi_dno_s50", "256x256", "16+16"),
        ("DFoT (K600)", "dfot_k600", "128x128", "5+12"),
    ]:
        s = load_summary(results_dir / subdir / "summary.json")
        if s:
            methods.append((name, s, res, frames))

    # --- LongCat TTA methods ---
    if args.longcat_dir:
        lc_dir = Path(args.longcat_dir)

        notta = load_summary(lc_dir / "ucf500_notta" / "NOTTA" / "summary.json")
        if notta:
            methods.append(("LongCat No-TTA", notta, "832x480", "14+14"))

        for series in ["ucf500_tta_baseline", "ucf500_tta_es_clip"]:
            for mid in ["FULL", "LORA", "DBL4"]:
                s = load_summary(lc_dir / series / mid / "summary.json")
                if s:
                    tag = "base" if "baseline" in series else "ES+CLIP"
                    methods.append(("LongCat %s (%s)" % (mid, tag), s, "832x480", "14+14"))

        for p2 in sorted(lc_dir.glob("phase2_*")):
            for rd in sorted(p2.iterdir()):
                s = load_summary(rd / "summary.json")
                if s:
                    methods.append(("LongCat P2/%s" % rd.name, s, "832x480", "14+14"))

    if not methods:
        print("No results found.")
        sys.exit(1)

    rows = []
    for name, summary, resolution, frames in methods:
        n = get_metric(summary, "num_successful", 0)
        rows.append({
            "Method": name,
            "Res": resolution,
            "Frames": frames,
            "n": str(n),
            "PSNR": fmt(get_metric(summary, "psnr")),
            "SSIM": fmt(get_metric(summary, "ssim")),
            "LPIPS": fmt(get_metric(summary, "lpips")),
            "FVD": fmt(get_metric(summary, "fvd"), 2),
            "FID": fmt(get_metric(summary, "fid"), 2),
            "Time(s)": fmt(get_metric(summary, "total_time"), 1),
        })

    title = "UCF-101 Video Prediction: Cross-Method Comparison"

    original_stdout = sys.stdout
    output_file = None
    if args.output:
        output_file = open(args.output, "w")
        sys.stdout = output_file

    print_table(rows, title)

    # Config summary
    config_rows = []
    for name, summary, resolution, frames in methods:
        method = get_metric(summary, "method", "?")
        steps = get_metric(summary, "num_steps") or get_metric(summary, "ddim_steps", "?")
        lr = get_metric(summary, "learning_rate") or get_metric(summary, "lr", "?")
        config_rows.append({
            "Method": name,
            "Res": resolution,
            "Cond+Gen": frames,
            "Steps": str(steps),
            "LR": str(lr),
        })
    print_table(config_rows, "Configuration Summary")

    if args.latex:
        print_latex_table(rows, title)

    if output_file:
        sys.stdout = original_stdout
        output_file.close()
        print("Report saved to: %s" % args.output)


if __name__ == "__main__":
    main()
