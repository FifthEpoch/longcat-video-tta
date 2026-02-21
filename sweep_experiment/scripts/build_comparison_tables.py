#!/usr/bin/env python3
"""
Build comparison tables aggregating all experiment results.

Produces three tables:
  Table A: Panda-70M (100 videos) -- per-video metrics + VBench++
  Table B: UCF-101 (~2500 videos) -- includes FVD
  Table C: Cross-backbone results (Panda-70M)

Usage:
    python sweep_experiment/scripts/build_comparison_tables.py \
        --results-dir sweep_experiment/results \
        --best-configs sweep_experiment/results/best_configs.json \
        --output-dir sweep_experiment/results/comparison
"""
import argparse, json, os, sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(summary: dict) -> dict:
    """Extract mean PSNR/SSIM/LPIPS from a summary.json."""
    successful = [r for r in summary.get("results", []) if r.get("success", False)]
    if not successful:
        return {"psnr": None, "ssim": None, "lpips": None, "n": 0}
    
    psnr = [r["psnr"] for r in successful if "psnr" in r and r["psnr"] is not None and not np.isnan(r["psnr"])]
    ssim = [r["ssim"] for r in successful if "ssim" in r and r["ssim"] is not None and not np.isnan(r["ssim"])]
    lpips = [r["lpips"] for r in successful if "lpips" in r and r["lpips"] is not None and not np.isnan(r["lpips"])]
    
    return {
        "psnr": f"{np.mean(psnr):.2f}" if psnr else "—",
        "ssim": f"{np.mean(ssim):.4f}" if ssim else "—",
        "lpips": f"{np.mean(lpips):.4f}" if lpips else "—",
        "n": len(successful),
    }


def build_table_a(results_dir: str, best_configs: dict) -> str:
    """Table A: Panda-70M results with VBench++ scores."""
    rows = []
    
    # No-TTA baselines
    rows.append(("LongCat no-TTA", "None", "0", "—", "—", "—", "—", "—"))
    rows.append(("Open-Sora no-TTA", "None", "0", "—", "—", "—", "—", "—"))
    rows.append(("DFoT (guidance)", "Guidance", "0", "—", "—", "—", "—", "—"))
    
    # Our methods
    method_info = {
        "delta_c": ("Ours: Delta-C", "Param TTA", "16"),
        "delta_a": ("Ours: Delta-A", "Param TTA", "512"),
        "delta_b": ("Ours: Delta-B", "Param TTA", "~2K"),
        "lora": ("Ours: LoRA", "Param TTA", "~1.8M"),
        "full": ("Ours: Full-Model", "Param TTA", "13.6B"),
    }
    
    for method, (name, adapt, params) in method_info.items():
        if method in best_configs:
            cfg = best_configs[method]
            psnr = f"{cfg.get('psnr_mean', 0):.2f}" if cfg.get("psnr_mean") else "—"
            ssim = f"{cfg.get('ssim_mean', 0):.4f}" if cfg.get("ssim_mean") else "—"
            lpips = f"{cfg.get('lpips_mean', 0):.4f}" if cfg.get("lpips_mean") else "—"
            rows.append((name, adapt, params, psnr, ssim, lpips, "(run)", "(run)"))
        else:
            rows.append((name, adapt, params, "TBD", "TBD", "TBD", "(run)", "(run)"))
    
    # Format as markdown table
    header = "| Method | Adaptation | #Params | PSNR | SSIM | LPIPS | VBench++ (subject) | VBench++ (motion) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def build_table_b(results_dir: str) -> str:
    """Table B: UCF-101 results including FVD."""
    header = "| Method | Adaptation Type | FVD | SSIM | PSNR | Source |"
    sep = "|---|---|---|---|---|---|"
    rows = [
        ("CVP", "Purpose-built predictor", "(paper)", "(paper)", "(paper)", "CVPR 2025"),
        ("SAVi-DNO", "Noise optimization", "(paper)", "(paper)", "(paper)", "arXiv 2025"),
        ("LongCat no-TTA", "None", "(run)", "(run)", "(run)", "Ours"),
        ("Ours: Delta-A", "Param TTA (512)", "(run)", "(run)", "(run)", "Ours"),
        ("Ours: LoRA", "Param TTA (~1.8M)", "(run)", "(run)", "(run)", "Ours"),
        ("Ours: Full-Model", "Param TTA (13.6B)", "(run)", "(run)", "(run)", "Ours"),
    ]
    
    # Check for existing FVD results
    for fvd_file in Path(results_dir).rglob("*fvd*.json"):
        data = load_json(str(fvd_file))
        if data and "fvd" in data:
            print(f"  Found FVD result: {fvd_file} -> FVD={data['fvd']:.2f}")
    
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_table_c(results_dir: str) -> str:
    """Table C: Cross-backbone results."""
    header = "| Backbone | Method | PSNR | SSIM | LPIPS |"
    sep = "|---|---|---|---|---|"
    rows = [
        ("LongCat 13.6B", "Best TTA", "(done)", "(done)", "(done)"),
        ("Open-Sora v2.0", "Best TTA", "(Exp 6)", "(Exp 6)", "(Exp 6)"),
        ("CogVideoX-5B", "Best TTA", "(Exp 6)", "(Exp 6)", "(Exp 6)"),
    ]
    
    # Check for backbone results
    for backbone_dir in ["opensora", "cogvideo"]:
        summary_files = list(Path(results_dir).parent.parent.glob(
            f"backbone_experiment/{backbone_dir}/results/*/summary.json"))
        for sf in summary_files:
            data = load_json(str(sf))
            if data:
                metrics = extract_metrics(data)
                print(f"  Found backbone result: {sf}")
    
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build comparison tables")
    parser.add_argument("--results-dir", default="sweep_experiment/results")
    parser.add_argument("--best-configs", default="sweep_experiment/results/best_configs.json")
    parser.add_argument("--output-dir", default="sweep_experiment/results/comparison")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    best_configs = load_json(args.best_configs) or {}
    
    print("=" * 70)
    print("Building Comparison Tables")
    print("=" * 70)
    
    # Table A
    print("\n## Table A: Panda-70M (100 videos)")
    table_a = build_table_a(args.results_dir, best_configs)
    print(table_a)
    
    # Table B
    print("\n## Table B: UCF-101 (~2,500 videos)")
    table_b = build_table_b(args.results_dir)
    print(table_b)
    
    # Table C
    print("\n## Table C: Cross-backbone (Panda-70M)")
    table_c = build_table_c(args.results_dir)
    print(table_c)
    
    # Save all tables to a markdown file
    output = os.path.join(args.output_dir, "comparison_tables.md")
    with open(output, "w") as f:
        f.write("# TTA Experiment Comparison Tables\n\n")
        f.write("## Table A: Panda-70M (100 videos) — per-video metrics + VBench++\n\n")
        f.write(table_a + "\n\n")
        f.write("## Table B: UCF-101 (~2,500 videos) — includes FVD\n\n")
        f.write(table_b + "\n\n")
        f.write("## Table C: Cross-backbone results (Panda-70M)\n\n")
        f.write(table_c + "\n")
    
    print(f"\nTables saved to {output}")
    
    # Save raw data as JSON
    raw = {
        "best_configs": best_configs,
        "tables": {
            "table_a": table_a,
            "table_b": table_b,
            "table_c": table_c,
        }
    }
    raw_path = os.path.join(args.output_dir, "comparison_data.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)


if __name__ == "__main__":
    main()
