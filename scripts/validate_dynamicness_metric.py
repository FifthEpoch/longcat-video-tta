#!/usr/bin/env python3
"""Validate our RAFT mean optical-flow score against VBench's official
Dynamic Degree score on the same set of videos.

VBench (Huang et al., CVPR 2024) is the de-facto benchmark suite for
video generation. Its Dynamic Degree dimension also uses RAFT, but with
fixed thresholds and a binary "dynamic / static" classification per video.
We use the *continuous* mean flow magnitude. If the two correlate strongly,
we get a one-line citation hook ("our continuous score correlates ρ=X.XX
with VBench's Dynamic Degree on the same N videos").

VBench's full_info JSON files live at
    sweep_experiment/results/<series>/<method>/chunk_*/vbench_results/
        vbench_dynamic_degree_full_info.json

Each file holds a list of records like:
    {"prompt_en": "panda_0010_delta_a",
     "dimension": ["dynamic_degree"],
     "video_list": ["/abs/path/.../panda_0010_delta_a.mp4"]}

VBench's `_eval_results.json` companion file holds the per-video scores.
We try both naming conventions and merge across chunks.

Output: scatter plot + correlation stats + JSON summary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: str) -> str:
    if s is None:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


def _try_load_eval_results(eval_path: Path) -> Dict[str, float]:
    """VBench's eval_results.json may have a few schemas; try the common ones."""
    if not eval_path.exists():
        return {}
    try:
        d = json.load(open(eval_path))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {eval_path}: {e}", file=sys.stderr)
        return {}

    out: Dict[str, float] = {}
    # Schema 1: {"dynamic_degree": [score, [{"video_path":..., "video_results":...}, ...]]}
    if isinstance(d, dict):
        for k, v in d.items():
            if not isinstance(v, list) or len(v) < 2:
                continue
            inner = v[1]
            if not isinstance(inner, list):
                continue
            for rec in inner:
                if not isinstance(rec, dict):
                    continue
                vp = rec.get("video_path") or rec.get("video") or rec.get("prompt_en")
                score = rec.get("video_results")
                if score is None:
                    score = rec.get("score") or rec.get("video_score")
                if vp is not None and score is not None:
                    try:
                        out[_canonical_video_id(vp)] = float(score)
                    except (TypeError, ValueError):
                        pass
    return out


def _try_load_full_info(full_info_path: Path) -> Dict[str, float]:
    """full_info.json is mostly metadata, but some VBench versions store
    per-video scores there too."""
    if not full_info_path.exists():
        return {}
    try:
        d = json.load(open(full_info_path))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {full_info_path}: {e}", file=sys.stderr)
        return {}
    out: Dict[str, float] = {}
    if isinstance(d, list):
        for rec in d:
            if not isinstance(rec, dict):
                continue
            score = (rec.get("video_results")
                     or rec.get("score")
                     or rec.get("dynamic_degree_score")
                     or rec.get("video_score"))
            vp = (rec.get("video_path")
                  or (rec.get("video_list") or [None])[0]
                  or rec.get("prompt_en"))
            if vp is not None and score is not None:
                try:
                    out[_canonical_video_id(vp)] = float(score)
                except (TypeError, ValueError):
                    pass
    return out


def collect_vbench_dynamic(method_dir: Path) -> Tuple[Dict[str, float], int]:
    """Return ({canonical_vid: vbench_dyn_score}, n_chunks_seen)."""
    merged: Dict[str, float] = {}
    chunks = sorted(method_dir.glob("chunk_*/vbench_results"))
    for chunk_dir in chunks:
        eval_p = chunk_dir / "vbench_dynamic_degree_eval_results.json"
        full_p = chunk_dir / "vbench_dynamic_degree_full_info.json"
        scores = _try_load_eval_results(eval_p)
        if not scores:
            scores = _try_load_full_info(full_p)
        if not scores:
            print(f"[warn] no dynamic-degree scores parsed from {chunk_dir}",
                  file=sys.stderr)
            continue
        merged.update(scores)
    return merged, len(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-dir", required=True, type=Path,
                    help="A method directory (e.g. "
                         "sweep_experiment/results/panda_1000v_standard/NOTTA) "
                         "whose chunk_*/vbench_results/ holds the VBench "
                         "Dynamic Degree files.")
    ap.add_argument("--dynamic-degree-json", required=True, type=Path,
                    help="Output JSON from scripts/compute_dynamic_degree.py")
    ap.add_argument("--flow-key", default="mean_flow",
                    choices=["mean_flow", "max_flow"])
    ap.add_argument("--output-png", required=True, type=Path)
    ap.add_argument("--title", default="")
    ap.add_argument("--save-stats-json", type=Path, default=None)
    args = ap.parse_args()

    # ---- VBench scores -----------------------------------------------------
    vbench_scores, n_chunks = collect_vbench_dynamic(args.method_dir)
    print(f"VBench: {len(vbench_scores)} per-video scores from "
          f"{n_chunks} chunks under {args.method_dir}")
    if not vbench_scores:
        print("[error] no VBench dynamic-degree scores found.", file=sys.stderr)
        return 2

    # ---- our RAFT scores ---------------------------------------------------
    dd = json.load(open(args.dynamic_degree_json))
    flow_by_vid: Dict[str, float] = {}
    for vid, info in dd["videos"].items():
        if "error" in info or info.get(args.flow_key) is None:
            continue
        flow_by_vid[_canonical_video_id(vid)] = float(info[args.flow_key])
    print(f"RAFT: {len(flow_by_vid)} per-video scores "
          f"(key={args.flow_key}, model={dd.get('model')}).")

    # ---- intersect ---------------------------------------------------------
    common = sorted(set(flow_by_vid.keys()) & set(vbench_scores.keys()))
    print(f"Common videos: {len(common)}")
    if len(common) < 30:
        print("[warn] very few common videos; correlation will be unstable.",
              file=sys.stderr)

    raft = np.array([flow_by_vid[v] for v in common], dtype=float)
    vb   = np.array([vbench_scores[v] for v in common], dtype=float)

    # ---- correlations ------------------------------------------------------
    pearson = float(np.corrcoef(raft, vb)[0, 1])
    rx, ry = np.argsort(np.argsort(raft)), np.argsort(np.argsort(vb))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    print(f"\nRAFT mean_flow vs VBench Dynamic Degree:")
    print(f"  Pearson  r = {pearson:+.3f}")
    print(f"  Spearman ρ = {spearman:+.3f}")

    # If VBench is binary (0/1), also report sensitivity / specificity at a
    # threshold sweep on our continuous score.
    binarized = set(np.unique(np.round(vb, 4)).tolist())
    is_binary = binarized.issubset({0.0, 1.0}) and len(binarized) <= 2

    threshold_table = []
    if is_binary:
        # sweep RAFT thresholds, report precision/recall vs vbench label
        thresholds = np.quantile(raft, np.linspace(0.05, 0.95, 19))
        for t in thresholds:
            pred = raft >= t
            label = vb > 0.5
            tp = int((pred & label).sum())
            fp = int((pred & ~label).sum())
            fn = int((~pred & label).sum())
            tn = int((~pred & ~label).sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            threshold_table.append({
                "raft_threshold": float(t),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec, "recall": rec, "f1": f1,
            })

    # ---- plot --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if is_binary:
        # boxplot of RAFT score split by VBench label
        ax.boxplot([raft[vb < 0.5], raft[vb > 0.5]],
                   labels=["VBench=0 (static)", "VBench=1 (dynamic)"],
                   showfliers=True)
        ax.set_ylabel(f"Our continuous score ({args.flow_key}, RAFT)")
        ax.set_title(
            f"RAFT continuous flow vs VBench Dynamic Degree (binary)\n"
            f"Spearman ρ = {spearman:+.3f}, n={len(common)}"
        )
        ax.set_yscale("symlog", linthresh=0.1)
    else:
        ax.scatter(vb, raft, s=8, alpha=0.4, edgecolors="none")
        ax.set_xlabel("VBench Dynamic Degree score")
        ax.set_ylabel(f"Our score ({args.flow_key}, RAFT)")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(
            f"Pearson r = {pearson:+.3f}  Spearman ρ = {spearman:+.3f}  "
            f"n = {len(common)}"
        )
    ax.grid(True, alpha=0.3)
    if args.title:
        fig.suptitle(args.title, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95] if args.title else None)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160, bbox_inches="tight")
    print(f"\nWrote figure: {args.output_png}")

    # ---- save stats --------------------------------------------------------
    if args.save_stats_json is None:
        args.save_stats_json = args.output_png.with_suffix(".stats.json")
    with open(args.save_stats_json, "w") as f:
        json.dump({
            "title": args.title,
            "flow_key": args.flow_key,
            "n_common_videos": len(common),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "vbench_is_binary": bool(is_binary),
            "threshold_sweep": threshold_table,
            "method_dir": str(args.method_dir),
        }, f, indent=2)
    print(f"Wrote stats JSON: {args.save_stats_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
