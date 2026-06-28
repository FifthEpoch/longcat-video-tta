#!/usr/bin/env python3
"""Per-video VBench++ deltas vs NOTTA and cross-metric win/loss agreement.

Answers:
  1. Do some videos win and others degrade on VBench++ under TTA / retrieval?
  2. When a video wins or loses on PSNR (or SSIM / LPIPS), does the same
     video tend to win/lose on VBench dimensions?

Loads per-video PSNR/SSIM/LPIPS from chunk ``summary.json`` (via
``analyze_per_video_tta_gain.load_per_video_metrics``) and per-video VBench
scores from ``chunk_*/vbench_results/vbench_<dim>_eval_results.json``.

Typical usage (Panda standard + retrieval, cluster):

    python3 scripts/analyze_per_video_vbench_agreement.py \\
        --baseline-dir sweep_experiment/results/panda_1000v_standard/NOTTA \\
        --method-dirs \\
            sweep_experiment/results/panda_1000v_standard/ADA \\
            sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA \\
            sweep_experiment/results/panda_1000v_retrieval/K5_SIM \\
            sweep_experiment/results/panda_1000v_retrieval/K5_RAND \\
            sweep_experiment/results/panda_1000v_retrieval/K10_SIM \\
            sweep_experiment/results/panda_1000v_retrieval/K10_RAND \\
        --output-dir sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/vbench_agreement

Outputs:
  - ``per_video_vbench_gains.csv``   wide table (baseline + deltas)
  - ``vbench_agreement_summary.md``  win/tie/loss + cross-metric agreement
  - ``delta_psnr_vs_vbench_scatter.png``  (optional grid scatter)

Dependencies: numpy, matplotlib (plot only).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_per_video_tta_gain import (  # noqa: E402
    _canonical_video_id,
    _coerce_float,
    load_per_video_metrics,
    pearson_r,
    spearman_rho,
)

VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
    "temporal_flickering",
]

# (column_suffix, higher_is_better, default_tie_threshold)
METRIC_SPECS: List[Tuple[str, bool, float]] = [
    ("psnr", True, 0.5),
    ("ssim", True, 0.01),
    ("lpips", False, 0.01),
    *[(d, True, 0.01) for d in VBENCH_DIMS],
]


BACKFILL_DIMS = [
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
    "temporal_flickering",
]


def _parse_eval_results_file(eval_path: Path) -> Dict[str, float]:
    """Parse per-video scores from ``vbench_<dim>_eval_results.json``.

    Mirrors ``validate_dynamicness_metric._try_load_eval_results`` but merges
    every dimension key present in the file (robust to dim-name quirks).
    """
    if not eval_path.exists():
        return {}
    try:
        parsed = json.loads(eval_path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {eval_path}: {exc}", file=sys.stderr)
        return {}

    out: Dict[str, float] = {}
    if not isinstance(parsed, dict):
        return out

    for _key, body in parsed.items():
        if not isinstance(body, list) or len(body) < 2:
            continue
        inner = body[1]
        if isinstance(inner, list):
            for rec in inner:
                if not isinstance(rec, dict):
                    continue
                vp = (
                    rec.get("video_path")
                    or rec.get("video")
                    or rec.get("path")
                    or rec.get("prompt_en")
                )
                score = rec.get("video_results")
                if score is None:
                    score = rec.get("score") or rec.get("video_score")
                vid = _canonical_video_id(vp if vp else "")
                val = _coerce_float(score)
                if vid and val is not None:
                    out[vid] = val
        elif isinstance(inner, dict):
            for path, score in inner.items():
                vid = _canonical_video_id(str(path))
                val = _coerce_float(score)
                if vid and val is not None:
                    out[vid] = val
    return out


def _parse_full_info_file(full_info_path: Path) -> Dict[str, float]:
    """Fallback: some VBench versions store per-video scores in full_info.json."""
    if not full_info_path.exists():
        return {}
    try:
        parsed = json.loads(full_info_path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {full_info_path}: {exc}", file=sys.stderr)
        return {}

    out: Dict[str, float] = {}
    if not isinstance(parsed, list):
        return out
    for rec in parsed:
        if not isinstance(rec, dict):
            continue
        score = (
            rec.get("video_results")
            or rec.get("score")
            or rec.get("video_score")
            or rec.get("dynamic_degree_score")
        )
        vp = (
            rec.get("video_path")
            or (rec.get("video_list") or [None])[0]
            or rec.get("prompt_en")
            or rec.get("video")
        )
        vid = _canonical_video_id(vp if vp else "")
        val = _coerce_float(score)
        if vid and val is not None:
            out[vid] = val
    return out


def _parse_vbench_per_video(parsed: dict, dim: str) -> Dict[str, float]:
    """Return {canonical_video_id -> score} from one eval_results.json blob."""
    body = parsed.get(dim)
    if body is None and len(parsed) == 1:
        body = next(iter(parsed.values()))
    if body is None:
        return {}

    per_video = None
    if isinstance(body, list) and len(body) >= 2:
        per_video = body[1]
    elif isinstance(body, list) and body and isinstance(body[0], dict):
        per_video = body[0]
    elif isinstance(body, (list, dict)):
        per_video = body

    out: Dict[str, float] = {}
    if isinstance(per_video, list):
        for item in per_video:
            if not isinstance(item, dict):
                continue
            path = (
                item.get("video_path")
                or item.get("video")
                or item.get("path")
                or item.get("prompt_en")
            )
            score = item.get("video_results", item.get("video_score", item.get("score")))
            vid = _canonical_video_id(path if path else "")
            val = _coerce_float(score)
            if vid and val is not None:
                out[vid] = val
    elif isinstance(per_video, dict):
        for path, score in per_video.items():
            vid = _canonical_video_id(str(path))
            val = _coerce_float(score)
            if vid and val is not None:
                out[vid] = val
    return out


def load_per_video_vbench(method_dir: Path) -> Dict[str, Dict[str, float]]:
    """Return {video_id -> {dim -> score}} merged across chunks."""
    merged: Dict[str, Dict[str, float]] = {}
    chunk_dirs = sorted(method_dir.glob("chunk_*"))
    if not chunk_dirs:
        vb_root = method_dir / "vbench_results"
        if vb_root.is_dir():
            chunk_dirs = [method_dir]
    for chunk_dir in chunk_dirs:
        vb_dir = chunk_dir / "vbench_results"
        if not vb_dir.is_dir():
            continue
        for dim in VBENCH_DIMS:
            eval_p = vb_dir / f"vbench_{dim}_eval_results.json"
            full_p = vb_dir / f"vbench_{dim}_full_info.json"
            scores = _parse_eval_results_file(eval_p)
            if not scores:
                scores = _parse_full_info_file(full_p)
            for vid, score in scores.items():
                merged.setdefault(vid, {})[dim] = score
    return merged


def vbench_dim_counts(vb: Dict[str, Dict[str, float]]) -> Dict[str, int]:
    counts = {d: 0 for d in VBENCH_DIMS}
    for dims in vb.values():
        for d in dims:
            if d in counts:
                counts[d] += 1
    return counts


def select_active_dims(
    all_vb: Dict[str, Dict[str, Dict[str, float]]],
    *,
    require_dims: Optional[List[str]] = None,
    min_videos: int = 50,
) -> List[str]:
    """Dims with per-video scores in every method dir (>= min_videos each)."""
    if require_dims:
        return list(require_dims)
    active = set(VBENCH_DIMS)
    for vb in all_vb.values():
        counts = vbench_dim_counts(vb)
        active &= {d for d in VBENCH_DIMS if counts[d] >= min_videos}
    if active:
        return sorted(active)
    # Fall back to backfill dims (usually have per-video lists).
    active = set(BACKFILL_DIMS)
    for vb in all_vb.values():
        counts = vbench_dim_counts(vb)
        active &= {d for d in BACKFILL_DIMS if counts[d] >= min_videos}
    return sorted(active)


def print_diagnose(
    baseline_name: str,
    baseline_dir: Path,
    methods: List[Tuple[str, Path]],
    baseline_pv: Dict,
    baseline_vb: Dict,
    method_pv: Dict,
    method_vb: Dict,
) -> None:
    print("\n=== Per-video coverage diagnose ===")
    dirs = [(baseline_name, baseline_pv, baseline_vb)] + [
        (n, method_pv[n], method_vb[n]) for n, _ in methods
    ]
    for label, pv, vb in dirs:
        psnr_n = sum(1 for m in pv.values() if m.get("psnr") is not None)
        counts = vbench_dim_counts(vb)
        print(f"\n{label} ({psnr_n} PSNR videos):")
        for d in VBENCH_DIMS:
            print(f"  {d:25s} {counts[d]:4d} per-video scores")
        if vb:
            sample = next(iter(vb))
            print(f"  sample vbench id: {sample!r}")
        if pv:
            sample_p = next(iter(pv))
            print(f"  sample psnr id:   {sample_p!r}")
    print("\nHint: original 3 VBench dims often store aggregate-only scores.")
    print("Re-backfill with --force if a dim shows 0 per-video counts:")
    print("  python scripts/run_vbench_backfill.py --method-dir <DIR> \\")
    print("    --dimensions subject_consistency background_consistency aesthetic_quality \\")
    print("    --force")
    print()


def _method_label(method_dir: Path) -> str:
    return method_dir.name


def _delta(method_val: Optional[float], base_val: Optional[float]) -> float:
    if method_val is None or base_val is None:
        return float("nan")
    mv = _coerce_float(method_val)
    bv = _coerce_float(base_val)
    if mv is None or bv is None:
        return float("nan")
    return float(mv - bv)


def _row_delta(row: dict, col: str) -> float:
    v = row.get(col)
    if v is None:
        return float("nan")
    x = _coerce_float(v)
    return float("nan") if x is None else float(x)


def _classify(delta: float, threshold: float, higher_is_better: bool) -> str:
    if math.isnan(delta):
        return "missing"
    if higher_is_better:
        if delta > threshold:
            return "win"
        if delta < -threshold:
            return "loss"
        return "tie"
    if delta < -threshold:
        return "win"
    if delta > threshold:
        return "loss"
    return "tie"


def _win_loss_tie_counts(
    deltas: np.ndarray, threshold: float, higher_is_better: bool,
) -> Tuple[int, int, int, int]:
    wins = ties = losses = missing = 0
    for d in deltas:
        c = _classify(float(d), threshold, higher_is_better)
        if c == "win":
            wins += 1
        elif c == "tie":
            ties += 1
        elif c == "loss":
            losses += 1
        else:
            missing += 1
    return wins, ties, losses, missing


def _outcome_agreement(
    ref_outcomes: List[str], cmp_outcomes: List[str],
) -> Dict[str, float]:
    """Agreement rates between two outcome lists (same length)."""
    valid = [
        (a, b) for a, b in zip(ref_outcomes, cmp_outcomes)
        if a != "missing" and b != "missing"
    ]
    n = len(valid)
    if n == 0:
        return {"n": 0, "exact": float("nan"), "sign": float("nan"),
                "win_win": float("nan"), "loss_loss": float("nan")}
    exact = sum(1 for a, b in valid if a == b) / n
    sign_match = sum(
        1 for a, b in valid
        if (a == "win" and b == "win") or (a == "loss" and b == "loss")
    ) / n
    ww = sum(1 for a, b in valid if a == "win" and b == "win") / n
    ll = sum(1 for a, b in valid if a == "loss" and b == "loss") / n
    return {"n": n, "exact": exact, "sign": sign_match,
            "win_win": ww, "loss_loss": ll}


def build_wide_table(
    video_ids: List[str],
    baseline_name: str,
    baseline_pv: Dict[str, Dict[str, Optional[float]]],
    baseline_vb: Dict[str, Dict[str, float]],
    methods: List[Tuple[str, Path]],
    active_dims: List[str],
    method_pv: Optional[Dict[str, Dict[str, Dict[str, Optional[float]]]]] = None,
    method_vb: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> Tuple[List[dict], List[str]]:
    method_names = [m for m, _ in methods]
    basic = ("psnr", "ssim", "lpips")
    fieldnames = ["video_id"]
    for spec in basic:
        fieldnames.append(f"{baseline_name}_{spec}")
    for name in method_names:
        for spec in basic:
            fieldnames.append(f"{name}_{spec}")
            fieldnames.append(f"{name}_d{spec}")
    for spec in active_dims:
        fieldnames.append(f"{baseline_name}_{spec}")
    for name in method_names:
        for spec in active_dims:
            fieldnames.append(f"{name}_{spec}")
            fieldnames.append(f"{name}_d{spec}")

    rows: List[dict] = []
    if method_pv is None:
        method_pv = {n: load_per_video_metrics(p) for n, p in methods}
    if method_vb is None:
        method_vb = {n: load_per_video_vbench(p) for n, p in methods}

    for vid in video_ids:
        row: dict = {"video_id": vid}
        base_m = baseline_pv.get(vid, {})
        base_v = baseline_vb.get(vid, {})
        for spec in basic:
            row[f"{baseline_name}_{spec}"] = base_m.get(spec)
        for spec in active_dims:
            row[f"{baseline_name}_{spec}"] = base_v.get(spec)
        for name in method_names:
            mpv = method_pv[name].get(vid, {})
            mvb = method_vb[name].get(vid, {})
            for spec in basic:
                raw = mpv.get(spec)
                d = _delta(raw, base_m.get(spec))
                row[f"{name}_{spec}"] = raw
                row[f"{name}_d{spec}"] = d if not math.isnan(d) else None
            for spec in active_dims:
                raw = mvb.get(spec)
                d = _delta(raw, base_v.get(spec))
                row[f"{name}_{spec}"] = raw
                row[f"{name}_d{spec}"] = d if not math.isnan(d) else None
        rows.append(row)
    return rows, fieldnames


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k)) for k in fieldnames})


def write_summary_md(
    path: Path,
    baseline_name: str,
    method_names: List[str],
    rows: List[dict],
    psnr_thresholds: Sequence[float],
    vbench_threshold: float,
    active_dims: List[str],
) -> None:
    lines: List[str] = []
    n = len(rows)
    lines.append("# Per-video VBench++ agreement vs NOTTA")
    lines.append("")
    lines.append(f"- Baseline: `{baseline_name}`")
    lines.append(f"- Methods: {', '.join(f'`{m}`' for m in method_names)}")
    lines.append(f"- Common videos (intersection): **{n}**")
    lines.append(f"- VBench dims analyzed: {', '.join(f'`{d}`' for d in active_dims)}")
    lines.append(f"- PSNR win/tie/loss thresholds (dB): {', '.join(str(t) for t in psnr_thresholds)}")
    lines.append(f"- VBench / SSIM / LPIPS tie threshold: **{vbench_threshold}**")
    lines.append("")
    lines.append(
        "Interpretation: population means can be ≈0 while many videos win and "
        "lose on each metric. **Exact agreement** = same win/tie/loss label; "
        "**Sign agreement** = both win or both lose (ties ignored)."
    )
    lines.append("")

    for thr in psnr_thresholds:
        lines.append(f"## Win / tie / loss counts (PSNR threshold = {thr} dB)")
        lines.append("")
        lines.append("| method | win | tie | loss | missing | win% | loss% |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name in method_names:
            deltas = np.array([_row_delta(r, f"{name}_dpsnr") for r in rows], dtype=float)
            w, t, l, m = _win_loss_tie_counts(deltas, thr, True)
            denom = max(w + t + l, 1)
            lines.append(
                f"| `{name}` | {w} | {t} | {l} | {m} | {100*w/denom:.1f}% | {100*l/denom:.1f}% |"
            )
        lines.append("")

    lines.append(f"## Win / tie / loss (VBench++, threshold = {vbench_threshold})")
    lines.append("")
    lines.append("| method | dim | win | tie | loss | win% | loss% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name in method_names:
        for dim in active_dims:
            deltas = np.array([_row_delta(r, f"{name}_d{dim}") for r in rows], dtype=float)
            w, t, l, _m = _win_loss_tie_counts(deltas, vbench_threshold, True)
            denom = max(w + t + l, 1)
            lines.append(
                f"| `{name}` | {dim} | {w} | {t} | {l} | {100*w/denom:.1f}% | {100*l/denom:.1f}% |"
            )
    lines.append("")

    lines.append(f"## Win / tie / loss (SSIM / LPIPS, threshold = {vbench_threshold})")
    lines.append("")
    lines.append("| method | metric | win | tie | loss | win% | loss% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name in method_names:
        for spec, hib, thr in [("ssim", True, vbench_threshold), ("lpips", False, vbench_threshold)]:
            deltas = np.array([_row_delta(r, f"{name}_d{spec}") for r in rows], dtype=float)
            w, t, l, _m = _win_loss_tie_counts(deltas, thr, hib)
            denom = max(w + t + l, 1)
            lines.append(
                f"| `{name}` | {spec} | {w} | {t} | {l} | {100*w/denom:.1f}% | {100*l/denom:.1f}% |"
            )
    lines.append("")
    lines.append(
        "> **FVD is population-only** in our pipeline (no per-video FVD). "
        "Cross-metric agreement here uses PSNR / SSIM / LPIPS + VBench++."
    )
    lines.append("")

    lines.append("## Spearman ρ(ΔPSNR, ΔVBench dim) per method")
    lines.append("")
    lines.append("| method | " + " | ".join(active_dims) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(active_dims)) + "|")
    for name in method_names:
        dpsnr = np.array([_row_delta(r, f"{name}_dpsnr") for r in rows], dtype=float)
        cells = []
        for dim in active_dims:
            dv = np.array([_row_delta(r, f"{name}_d{dim}") for r in rows], dtype=float)
            rho = spearman_rho(dpsnr, dv)
            cells.append(f"{rho:+.3f}" if rho is not None else "n/a")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines.append("")

    ref_thr = psnr_thresholds[0] if psnr_thresholds else 0.5
    lines.append(f"## Cross-metric agreement with ΔPSNR (threshold PSNR={ref_thr} dB, VBench={vbench_threshold})")
    lines.append("")
    lines.append(
        "For each method, classify every video on PSNR and each other metric, "
        "then report **exact** (same win/tie/loss) and **sign** (win+win or loss+loss) agreement."
    )
    lines.append("")
    for name in method_names:
        lines.append(f"### `{name}`")
        lines.append("")
        dpsnr = np.array([_row_delta(r, f"{name}_dpsnr") for r in rows], dtype=float)
        psnr_out = [_classify(float(d), ref_thr, True) for d in dpsnr]
        lines.append("| metric | N | exact agree | sign agree | win∩win | loss∩loss |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        cross_metrics: List[Tuple[str, bool, float]] = [
            ("ssim", True, vbench_threshold),
            ("lpips", False, vbench_threshold),
        ] + [(d, True, vbench_threshold) for d in active_dims]
        for spec, hib, thr in cross_metrics:
            t = thr
            deltas = np.array([_row_delta(r, f"{name}_d{spec}") for r in rows], dtype=float)
            outcomes = [_classify(float(d), t, hib) for d in deltas]
            ag = _outcome_agreement(psnr_out, outcomes)
            lines.append(
                f"| `{spec}` | {ag['n']} | {ag['exact']:.3f} | {ag['sign']:.3f} | "
                f"{ag['win_win']:.3f} | {ag['loss_loss']:.3f} |"
            )
        lines.append("")

    lines.append("## VBench total Δ vs ΔPSNR")
    lines.append("")
    lines.append("VBench total ≈ mean of analyzed VBench dims.")
    lines.append("")
    lines.append("| method | mean Δ total | ρ(ΔPSNR, Δtotal) | win% on Δtotal |")
    lines.append("|---|---:|---:|---:|")
    for name in method_names:
        dpsnr = np.array([_row_delta(r, f"{name}_dpsnr") for r in rows], dtype=float)
        dtot = []
        for r in rows:
            vals = [r.get(f"{name}_d{d}") for d in active_dims]
            floats = [_coerce_float(v) for v in vals]
            if any(v is None for v in floats):
                dtot.append(float("nan"))
            else:
                dtot.append(float(np.mean(floats)))  # type: ignore[arg-type]
        dtot_arr = np.array(dtot, dtype=float)
        rho = spearman_rho(dpsnr, dtot_arr)
        w, t, l, _ = _win_loss_tie_counts(dtot_arr, vbench_threshold, True)
        denom = max(w + t + l, 1)
        mu = float(np.nanmean(dtot_arr)) if np.isfinite(dtot_arr).any() else float("nan")
        rho_s = f"{rho:+.3f}" if rho is not None else "n/a"
        lines.append(
            f"| `{name}` | {mu:+.5f} | {rho_s} | {100*w/denom:.1f}% |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scatter_grid(
    path: Path,
    method_names: List[str],
    rows: List[dict],
    vbench_dims: Sequence[str],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_methods = len(method_names)
    n_dims = len(vbench_dims)
    fig, axes = plt.subplots(n_methods, n_dims, figsize=(3.2 * n_dims, 2.8 * n_methods),
                             squeeze=False)
    for i, name in enumerate(method_names):
        dpsnr = np.array([_row_delta(r, f"{name}_dpsnr") for r in rows], dtype=float)
        for j, dim in enumerate(vbench_dims):
            ax = axes[i, j]
            dv = np.array([_row_delta(r, f"{name}_d{dim}") for r in rows], dtype=float)
            mask = ~(np.isnan(dpsnr) | np.isnan(dv))
            if mask.sum() >= 2:
                ax.scatter(dpsnr[mask], dv[mask], s=6, alpha=0.35, linewidths=0)
                rho = spearman_rho(dpsnr, dv)
                ax.set_title(f"{name}\n{dim}\nρ={rho:+.2f}" if rho is not None else dim,
                             fontsize=8)
            else:
                ax.set_title(f"{name}\n{dim}\nn/a", fontsize=8)
            if i == n_methods - 1:
                ax.set_xlabel("ΔPSNR")
            if j == 0:
                ax.set_ylabel(f"Δ{dim[:8]}")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.axvline(0, color="0.7", lw=0.5)
    fig.suptitle("Per-video ΔPSNR vs ΔVBench (vs NOTTA)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--baseline-dir", type=Path,
        default=Path("sweep_experiment/results/panda_1000v_standard/NOTTA"),
    )
    ap.add_argument(
        "--method-dirs", type=Path, nargs="+", required=True,
        help="Method directories to compare against baseline.",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--psnr-thresholds", type=float, nargs="+", default=[0.5, 0.1],
        help="Win/tie/loss thresholds for ΔPSNR in dB.",
    )
    ap.add_argument(
        "--vbench-threshold", type=float, default=0.01,
        help="Win/tie/loss threshold for ΔVBench dims (0–1 scale).",
    )
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--diagnose", action="store_true",
        help="Print per-method PSNR/VBench coverage and exit (no analysis).",
    )
    ap.add_argument(
        "--require-dims", nargs="+", default=None,
        help="Require these VBench dims (default: auto-detect from data).",
    )
    ap.add_argument(
        "--min-vbench-videos", type=int, default=50,
        help="Min per-video count per dim to include in auto-detect.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    baseline_dir = args.baseline_dir.resolve()
    if not baseline_dir.is_dir():
        print(f"[error] baseline dir missing: {baseline_dir}", file=sys.stderr)
        return 2

    baseline_name = _method_label(baseline_dir)
    methods: List[Tuple[str, Path]] = []
    for p in args.method_dirs:
        pr = p.resolve()
        if not pr.is_dir():
            print(f"[warn] skip missing method dir: {pr}", file=sys.stderr)
            continue
        methods.append((_method_label(pr), pr))
    if not methods:
        print("[error] no valid method dirs", file=sys.stderr)
        return 2

    baseline_pv = load_per_video_metrics(baseline_dir)
    baseline_vb = load_per_video_vbench(baseline_dir)
    method_pv = {n: load_per_video_metrics(p) for n, p in methods}
    method_vb = {n: load_per_video_vbench(p) for n, p in methods}

    all_vb = {baseline_name: baseline_vb, **method_vb}
    if args.diagnose:
        print_diagnose(
            baseline_name, baseline_dir, methods,
            baseline_pv, baseline_vb, method_pv, method_vb,
        )
        return 0

    active_dims = select_active_dims(
        all_vb,
        require_dims=args.require_dims,
        min_videos=args.min_vbench_videos,
    )
    if not active_dims:
        print("[error] no VBench dims with per-video scores found", file=sys.stderr)
        print_diagnose(
            baseline_name, baseline_dir, methods,
            baseline_pv, baseline_vb, method_pv, method_vb,
        )
        return 2

    def _complete_vbench(vb: Dict[str, Dict[str, float]]) -> set:
        return {vid for vid, dims in vb.items() if all(d in dims for d in active_dims)}

    def _complete_psnr(pv: Dict[str, Dict[str, Optional[float]]]) -> set:
        return {vid for vid, m in pv.items() if m.get("psnr") is not None}

    common = _complete_psnr(baseline_pv) & _complete_vbench(baseline_vb)
    for name, _ in methods:
        common &= _complete_psnr(method_pv[name]) & _complete_vbench(method_vb[name])

    if not common:
        print(
            f"[error] empty video intersection (need PSNR + VBench dims: {active_dims})",
            file=sys.stderr,
        )
        print_diagnose(
            baseline_name, baseline_dir, methods,
            baseline_pv, baseline_vb, method_pv, method_vb,
        )
        return 2

    video_ids = sorted(common)
    print(f"Common videos: {len(video_ids)}  |  VBench dims: {', '.join(active_dims)}")

    rows, fieldnames = build_wide_table(
        video_ids, baseline_name, baseline_pv, baseline_vb, methods,
        active_dims,
        method_pv=method_pv, method_vb=method_vb,
    )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_video_vbench_gains.csv", rows, fieldnames)
    write_summary_md(
        out_dir / "vbench_agreement_summary.md",
        baseline_name,
        [n for n, _ in methods],
        rows,
        args.psnr_thresholds,
        args.vbench_threshold,
        active_dims,
    )
    if not args.no_plots:
        plot_scatter_grid(
            out_dir / "delta_psnr_vs_vbench_scatter.png",
            [n for n, _ in methods],
            rows,
            active_dims,
        )

    print(f"Wrote {out_dir / 'per_video_vbench_gains.csv'}")
    print(f"Wrote {out_dir / 'vbench_agreement_summary.md'}")
    if not args.no_plots:
        print(f"Wrote {out_dir / 'delta_psnr_vs_vbench_scatter.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
