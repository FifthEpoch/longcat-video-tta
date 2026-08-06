#!/usr/bin/env python3
"""Consolidated matched-N all-metric table for the EXP2 placement arms.

Compares NO-TTA vs normal AdaSteer (adaln) vs residual AdaSteer on EVERY metric
we track — PSNR, SSIM, LPIPS, FVD, and all 7 VBench++ dimensions (+ a normalized
overall) — on a SINGLE common video set so N is identical across every metric and
every method.

Where each metric comes from
----------------------------
- pixel (psnr/ssim/lpips): per-video from each method's SOURCE run dir
  (chunk_*/summary.json), via analyze_per_video_tta_gain.load_per_video_metrics.
- VBench (7 dims, gen-only): per-video from the FVD POLICY dirs
  (<fvd_root>/<METHOD>/vbench_results_geneval), via
  analyze_per_video_vbench_agreement.load_per_video_vbench. Those policy dirs hold
  exactly the FVD common-N clips symlinked as <canonical_id>.mp4, so computing
  VBench there guarantees the same set for all three methods AND fills the
  otherwise-missing NO-TTA gen-only VBench.
- FVD: set-level, read from <fvd_root>/<METHOD>/fvd.json (the reliable N=500 run).
  If the per-video common set S' is SMALLER than the FVD N, FVD is flagged as
  computed on a superset and (with --eval-fvd + --gt-cache) can be recomputed on
  exactly S'.

The common set S' = ids present with pixel AND all requested VBench dims in ALL
THREE methods. Every reported per-video mean is over S', so N matches.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    load_per_video_vbench,
)

PIXEL_KEYS = ["psnr", "ssim", "lpips"]
# imaging_quality is reported on a 0-100 scale; everything else is 0-1. Normalize
# for the "overall" mean only (per-dim columns show raw values).
_IMG_QUALITY_SCALE = {"imaging_quality": 100.0}


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _read_fvd_json(method_dir: Path) -> Optional[Dict]:
    fp = method_dir / "fvd.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text())
    except Exception:  # noqa: BLE001
        return None


def _fvd_value(blob: Optional[Dict]) -> Optional[float]:
    if not blob:
        return None
    for k in ("fvd", "FVD", "fvd_value", "score"):
        if k in blob and blob[k] is not None:
            try:
                return float(blob[k])
            except (TypeError, ValueError):
                pass
    return None


def _fvd_n(blob: Optional[Dict]) -> Optional[int]:
    if not blob:
        return None
    for k in ("num_videos", "n", "N", "fvd_num_videos", "num_gen"):
        if k in blob and blob[k] is not None:
            try:
                return int(blob[k])
            except (TypeError, ValueError):
                pass
    return None


def _recompute_fvd_on_ids(
    method_dir: Path, ids: List[str], gt_cache: Path, python: str,
    num_cond: int, num_gen: int, tmp_tag: str,
) -> Optional[float]:
    """Symlink S' clips into a subset dir and run eval_fvd on exactly S'."""
    src_videos = method_dir / "videos"
    if not src_videos.is_dir():
        print(f"  [fvd-recompute] no videos/ under {method_dir}", file=sys.stderr)
        return None
    subset = method_dir / f"videos_matchedN_{tmp_tag}"
    subset.mkdir(parents=True, exist_ok=True)
    for p in subset.glob("*.mp4"):
        p.unlink()
    linked = 0
    for vid in ids:
        src = src_videos / f"{vid}.mp4"
        if src.exists() or src.is_symlink():
            os.symlink(src.resolve(), subset / f"{vid}.mp4")
            linked += 1
    if linked < 40:
        print(f"  [fvd-recompute] only {linked} clips linked for {method_dir.name}",
              file=sys.stderr)
        return None
    out_json = method_dir / f"fvd_matchedN_{tmp_tag}.json"
    cmd = [
        python, str(_REPO / "sweep_experiment/scripts/eval_fvd.py"),
        "--gen-dir", str(subset), "--gt-cache", str(gt_cache),
        "--num-cond-frames", str(num_cond), "--num-gen-frames", str(num_gen),
        "--min-videos", "40", "--output", str(out_json), "--device", "cuda", "--force",
    ]
    print("  [fvd-recompute]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"  [fvd-recompute] failed: {exc}", file=sys.stderr)
        return None
    return _fvd_value(_read_fvd_json_path(out_json))


def _read_fvd_json_path(fp: Path) -> Optional[Dict]:
    try:
        return json.loads(fp.read_text())
    except Exception:  # noqa: BLE001
        return None


def _fmt(v: Optional[float], nd: int) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fvd-root", required=True, type=Path,
                    help="Dir with <METHOD>/{videos,fvd.json,vbench_results_geneval}.")
    ap.add_argument("--notta-label", default="NOTTA")
    ap.add_argument("--adaln-label", default="ADA_ADALN")
    ap.add_argument("--resid-label", default="ADA_RESID")
    ap.add_argument("--pixel-notta", required=True, type=Path,
                    help="SOURCE run dir for NO-TTA pixel metrics (has summary.json).")
    ap.add_argument("--pixel-adaln", required=True, type=Path)
    ap.add_argument("--pixel-resid", required=True, type=Path)
    ap.add_argument("--vbench-subdir", default="vbench_results_geneval")
    ap.add_argument("--eval-fvd", action="store_true",
                    help="Recompute FVD on the exact per-video common set S' "
                         "(needs --gt-cache). Otherwise FVD is read from fvd.json.")
    ap.add_argument("--gt-cache", type=Path, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--num-cond-frames", type=int, default=14)
    ap.add_argument("--num-gen-frames", type=int, default=14)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    os.environ["VBENCH_SUBDIR"] = args.vbench_subdir

    methods = [
        (args.notta_label, args.pixel_notta),
        (args.adaln_label, args.pixel_adaln),
        (args.resid_label, args.pixel_resid),
    ]

    # Load per-video pixel (source dirs) + VBench (policy dirs).
    pixel: Dict[str, Dict[str, Dict[str, float]]] = {}
    vbench: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, pix_dir in methods:
        pixel[label] = load_per_video_metrics(pix_dir)
        vb_method_dir = args.fvd_root / label
        vbench[label] = load_per_video_vbench(vb_method_dir)
        print(f"[load] {label}: pixel ids={len(pixel[label])} "
              f"(src {pix_dir})  vbench ids={len(vbench[label])} (policy {vb_method_dir})")

    # Common set S' = ids with pixel + all 7 VBench dims in ALL three methods.
    def _ok_pixel(d: Dict[str, float]) -> bool:
        return all(d.get(k) is not None for k in PIXEL_KEYS)

    def _ok_vbench(d: Dict[str, float]) -> bool:
        return all(d.get(k) is not None for k in VBENCH_DIMS)

    id_sets = []
    for label, _ in methods:
        ids = {v for v, d in pixel[label].items() if _ok_pixel(d)}
        ids &= {v for v, d in vbench[label].items() if _ok_vbench(d)}
        id_sets.append(ids)
    common = sorted(set.intersection(*id_sets)) if id_sets else []
    n = len(common)
    print(f"[common] matched-N set S' (pixel + all 7 VBench in all 3 methods): N={n}")
    if n == 0:
        print("ERROR: empty common set — check vbench_results_geneval exists for all "
              "three methods under --fvd-root.", file=sys.stderr)
        return 2

    # Aggregate.
    rows = []
    for label, _ in methods:
        row: Dict[str, Optional[float]] = {"method": label}
        for k in PIXEL_KEYS:
            row[k] = _mean([pixel[label][v][k] for v in common])
        for d in VBENCH_DIMS:
            row[d] = _mean([vbench[label][v][d] for v in common])
        # normalized overall (map imaging_quality 0-100 -> 0-1)
        norm_vals = []
        for d in VBENCH_DIMS:
            val = row[d]
            if val is None:
                continue
            norm_vals.append(val / _IMG_QUALITY_SCALE.get(d, 1.0))
        row["vbench_overall"] = _mean(norm_vals)

        fvd_blob = _read_fvd_json(args.fvd_root / label)
        fvd_full = _fvd_value(fvd_blob)
        fvd_full_n = _fvd_n(fvd_blob)
        if args.eval_fvd and args.gt_cache is not None and (fvd_full_n or 0) != n:
            row["fvd"] = _recompute_fvd_on_ids(
                args.fvd_root / label, common, args.gt_cache, args.python,
                args.num_cond_frames, args.num_gen_frames, tmp_tag=f"n{n}",
            )
            row["fvd_n"] = n
        else:
            row["fvd"] = fvd_full
            row["fvd_n"] = fvd_full_n
        rows.append(row)

    # Emit markdown.
    notta = rows[0]
    lines: List[str] = []
    lines.append("# EXP2 placement arms — consolidated matched-N all-metric table")
    lines.append("")
    lines.append(f"**Matched N = {n}** videos, identical set across all three methods "
                 f"for every per-video metric (PSNR/SSIM/LPIPS + 7 VBench dims).")
    fvd_ns = {r["fvd_n"] for r in rows if r.get("fvd_n") is not None}
    if fvd_ns == {n}:
        lines.append(f"FVD computed on the same N={n} set.")
    else:
        lines.append(f"FVD is a set-level metric read from fvd.json at "
                     f"N={sorted(fvd_ns)} (superset of the per-video S'={n}; "
                     f"re-run with --eval-fvd --gt-cache to force N={n}).")
    lines.append("")
    lines.append("Higher is better: PSNR, SSIM, all VBench dims. Lower is better: "
                 "LPIPS, FVD. imaging_quality is 0–100; vbench_overall normalizes it "
                 "to 0–1 before averaging the 7 dims.")
    lines.append("")

    cols = (["psnr", "ssim", "lpips", "fvd"]
            + list(VBENCH_DIMS) + ["vbench_overall"])
    nd = {"psnr": 4, "ssim": 4, "lpips": 4, "fvd": 3, "imaging_quality": 4,
          "vbench_overall": 4}
    header = "| metric | " + " | ".join(r["method"] for r in rows) + " | Δ ADALN−NOTTA | Δ RESID−NOTTA |"
    sep = "|---|" + "|".join(["---:"] * (len(rows) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in cols:
        vals = [r.get(c) for r in rows]
        d_adaln = (vals[1] - vals[0]) if (vals[1] is not None and vals[0] is not None) else None
        d_resid = (vals[2] - vals[0]) if (vals[2] is not None and vals[0] is not None) else None
        digits = nd.get(c, 4)
        cells = " | ".join(_fmt(v, digits) for v in vals)
        lines.append(f"| {c} | {cells} | "
                     f"{_fmt(d_adaln, digits)} | {_fmt(d_resid, digits)} |")
    lines.append("")
    lines.append(f"FVD N per method: " + ", ".join(
        f"{r['method']}={r.get('fvd_n')}" for r in rows))
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
