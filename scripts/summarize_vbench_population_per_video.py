#!/usr/bin/env python3
"""Population + per-video VBench++ breakdown vs NOTTA.

Complements ``analyze_per_video_vbench_agreement.py`` with paper-style tables:
  - Population means (from ``merged_summary.json``)
  - Per-video mean Δ, win/tie/loss % (vs NOTTA baseline)

Example (Panda 1000v, cluster):

    python3 scripts/summarize_vbench_population_per_video.py \\
        --baseline-dir sweep_experiment/results/panda_1000v_standard/NOTTA \\
        --output sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/vbench_breakdown.md \\
        --method NOTTA:sweep_experiment/results/panda_1000v_standard/NOTTA:No-TTA \\
        --method ADA:sweep_experiment/results/panda_1000v_standard/ADA:AdaSteer \\
        --method LORA:sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA:LoRA-R8 \\
        --method K5_SIM:sweep_experiment/results/panda_1000v_retrieval/K5_SIM:AdaSteer+K5 SIM \\
        --method K5_RAND:sweep_experiment/results/panda_1000v_retrieval/K5_RAND:AdaSteer+K5 RAND \\
        --method K10_SIM:sweep_experiment/results/panda_1000v_retrieval/K10_SIM:AdaSteer+K10 SIM \\
        --method K10_RAND:sweep_experiment/results/panda_1000v_retrieval/K10_RAND:AdaSteer+K10 RAND

Or use the Panda preset:

    bash scripts/run_panda_vbench_breakdown.sh
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    _coerce_float,
    _delta,
    _win_loss_tie_counts,
    load_per_video_vbench,
    select_active_dims,
)

DIM_SHORT = {
    "subject_consistency": "Subj",
    "background_consistency": "BG",
    "aesthetic_quality": "Aes",
    "motion_smoothness": "Motn",
    "dynamic_degree": "Dyn",
    "imaging_quality": "IQ",
    "temporal_flickering": "Flick",
}


def _load_population_vbench(method_dir: Path) -> Dict[str, float]:
    ms = method_dir / "merged_summary.json"
    if not ms.exists():
        return {}
    blob = json.loads(ms.read_text())
    vb = blob.get("vbench") or {}
    out: Dict[str, float] = {}
    for d in VBENCH_DIMS:
        v = vb.get(d)
        if isinstance(v, (int, float)):
            out[d] = float(v)
    return out


def _vbench_total(scores: Dict[str, float]) -> Optional[float]:
    vals = [scores.get(d) for d in VBENCH_DIMS]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_delta(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}"


def _parse_method_arg(s: str) -> Tuple[str, Path, str]:
    """``ID:PATH:LABEL`` or ``ID:PATH`` (label=id)."""
    parts = s.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"expected ID:PATH[:LABEL], got {s!r}")
    mid, path = parts[0], Path(parts[1])
    label = parts[2] if len(parts) == 3 else mid
    return mid, path, label


def _per_video_stats(
    baseline_vb: Dict[str, Dict[str, float]],
    method_vb: Dict[str, Dict[str, float]],
    video_ids: Sequence[str],
    dims: Sequence[str],
    threshold: float,
) -> Dict[str, dict]:
    """Per-dim mean Δ and win/tie/loss counts."""
    out: Dict[str, dict] = {}
    for dim in dims:
        deltas = []
        for vid in video_ids:
            d = _delta(
                (method_vb.get(vid) or {}).get(dim),
                (baseline_vb.get(vid) or {}).get(dim),
            )
            if not math.isnan(d):
                deltas.append(d)
        arr = np.array(deltas, dtype=float) if deltas else np.array([], dtype=float)
        w, t, l, m = _win_loss_tie_counts(arr, threshold, True) if arr.size else (0, 0, 0, 0)
        denom = max(w + t + l, 1)
        out[dim] = {
            "mean_delta": float(arr.mean()) if arr.size else float("nan"),
            "win": w, "tie": t, "loss": l, "missing": m,
            "win_pct": 100.0 * w / denom,
            "tie_pct": 100.0 * t / denom,
            "loss_pct": 100.0 * l / denom,
            "n": int(arr.size),
        }
    return out


def _intersection_ids(
    baseline_vb: Dict[str, Dict[str, float]],
    methods_vb: Dict[str, Dict[str, Dict[str, float]]],
    dims: Sequence[str],
) -> List[str]:
    def complete(vb: Dict[str, Dict[str, float]]) -> set:
        return {v for v, dm in vb.items() if all(d in dm for d in dims)}

    common = complete(baseline_vb)
    for vb in methods_vb.values():
        common &= complete(vb)
    return sorted(common)


def build_report(
    methods: List[Tuple[str, Path, str]],
    baseline_dir: Path,
    *,
    vbench_threshold: float = 0.01,
) -> str:
    baseline_dir = baseline_dir.resolve()
    baseline_id = baseline_dir.name
    pop: Dict[str, Dict[str, float]] = {}
    pvb: Dict[str, Dict[str, Dict[str, float]]] = {}

    for mid, path, _label in methods:
        pop[mid] = _load_population_vbench(path)
        pvb[mid] = load_per_video_vbench(path)

    base_pop = pop.get(baseline_id) or _load_population_vbench(baseline_dir)
    base_pvb = pvb.get(baseline_id) or load_per_video_vbench(baseline_dir)
    base_total = _vbench_total(base_pop)

    all_vb = {mid: pvb[mid] for mid, _, _ in methods if mid in pvb}
    active_dims = select_active_dims(all_vb, min_videos=50)
    video_ids = _intersection_ids(base_pvb, all_vb, active_dims)
    n = len(video_ids)

    lines: List[str] = []
    lines.append("# VBench++ population & per-video breakdown (Panda 1000v)")
    lines.append("")
    lines.append(f"- **Date:** {date.today().isoformat()}")
    lines.append(f"- **Eval set:** Panda 1000v (N={n} common videos with full VBench)")
    lines.append(f"- **Baseline:** `{baseline_dir}`")
    lines.append(f"- **Per-video threshold:** ±{vbench_threshold} on Δ vs NOTTA")
    lines.append(f"- **VBench total:** mean of {len(active_dims)} dims "
                 f"({', '.join(DIM_SHORT[d] for d in active_dims)})")
    lines.append("")
    lines.append("Retrieval runs use **AdaSteer (`delta_a`)** with K neighbors from "
                 "`panda_2048_480p`; LoRA is a separate TTA baseline without retrieval.")
    lines.append("")

    # --- Population ---
    lines.append("## A. Population-level VBench++ (merged 999-video mean)")
    lines.append("")
    hdr = "| Method | Label | " + " | ".join(DIM_SHORT[d] for d in VBENCH_DIMS) + " | Total | ΔTotal vs NOTTA |"
    sep = "|---|---|" + "|".join(["---:"] * len(VBENCH_DIMS)) + "|---:|---:|"
    lines.append(hdr)
    lines.append(sep)
    for mid, _path, label in methods:
        scores = pop.get(mid, {})
        total = _vbench_total(scores)
        dtot = (total - base_total) if total is not None and base_total is not None else None
        cells = [_fmt(scores.get(d)) for d in VBENCH_DIMS]
        lines.append(
            f"| `{mid}` | {label} | " + " | ".join(cells) +
            f" | {_fmt(total)} | {_fmt_delta(dtot)} |"
        )
    lines.append("")

    lines.append("### Population Δ vs NOTTA (by dim)")
    lines.append("")
    lines.append("| Method | " + " | ".join(DIM_SHORT[d] for d in VBENCH_DIMS) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(VBENCH_DIMS)) + "|")
    for mid, _path, label in methods:
        if mid == baseline_id:
            continue
        scores = pop.get(mid, {})
        cells = []
        for d in VBENCH_DIMS:
            bv = base_pop.get(d)
            mv = scores.get(d)
            if bv is None or mv is None:
                cells.append("—")
            else:
                cells.append(_fmt_delta(mv - bv))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Per-video mean delta ---
    lines.append("## B. Per-video mean Δ vs NOTTA")
    lines.append("")
    lines.append("| Method | " + " | ".join(DIM_SHORT[d] for d in active_dims) + " | Mean Δ total |")
    lines.append("|---|" + "|".join(["---:"] * len(active_dims)) + "|---:|")
    for mid, _path, label in methods:
        if mid == baseline_id:
            continue
        stats = _per_video_stats(base_pvb, pvb[mid], video_ids, active_dims, vbench_threshold)
        cells = [_fmt_delta(stats[d]["mean_delta"]) for d in active_dims]
        dtot_vals = []
        for vid in video_ids:
            vals = []
            for d in active_dims:
                raw = _delta(
                    (pvb[mid].get(vid) or {}).get(d),
                    (base_pvb.get(vid) or {}).get(d),
                )
                if not math.isnan(raw):
                    vals.append(raw)
            if len(vals) == len(active_dims):
                dtot_vals.append(float(np.mean(vals)))
        mu = float(np.mean(dtot_vals)) if dtot_vals else float("nan")
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {_fmt_delta(mu)} |")
    lines.append("")

    # --- Per-video win/loss ---
    lines.append(f"## C. Per-video win / tie / loss vs NOTTA (±{vbench_threshold})")
    lines.append("")
    lines.append("**win% / tie% / loss%** per dimension (NOTTA omitted — baseline by definition).")
    lines.append("")

    for mid, _path, label in methods:
        if mid == baseline_id:
            continue
        lines.append(f"### {label} (`{mid}`)")
        lines.append("")
        lines.append("| Dim | win% | tie% | loss% |")
        lines.append("|---|---:|---:|---:|")
        stats = _per_video_stats(base_pvb, pvb[mid], video_ids, active_dims, vbench_threshold)
        for d in active_dims:
            s = stats[d]
            lines.append(
                f"| {DIM_SHORT[d]} | {s['win_pct']:.1f}% | {s['tie_pct']:.1f}% | {s['loss_pct']:.1f}% |"
            )
        lines.append("")

    # --- Compact comparison ---
    lines.append("## D. Compact comparison (per-video win% / loss%)")
    lines.append("")
    lines.append("| Dim | AdaSteer | LoRA-R8 | K5 SIM | K5 RAND | K10 SIM | K10 RAND |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    compare_cols: List[Tuple[str, str]] = [
        ("ADA", "AdaSteer"),
        ("LORA", "LoRA-R8"),
        ("K5_SIM", "K5 SIM"),
        ("K5_RAND", "K5 RAND"),
        ("K10_SIM", "K10 SIM"),
        ("K10_RAND", "K10 RAND"),
    ]
    id_set = {mid for mid, _, _ in methods}
    for d in active_dims:
        cells = []
        for mid, _col in compare_cols:
            if mid not in id_set or mid not in pvb or mid == baseline_id:
                cells.append("—")
                continue
            st = _per_video_stats(base_pvb, pvb[mid], video_ids, [d], vbench_threshold)[d]
            cells.append(f"{st['win_pct']:.0f}/{st['loss_pct']:.0f}")
        lines.append(f"| {DIM_SHORT[d]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("*Cells show win% / loss% (ties omitted for brevity).*")
    lines.append("")

    lines.append("## E. Reading guide")
    lines.append("")
    lines.append("1. **Population ≈ 0, per-video spread** — AdaSteer: aggregate Subj/Motn/Flick "
                 "flat; IQ ~50/50 win-loss at video level.")
    lines.append("2. **Frontier shift** — LoRA & AdaSteer+retrieval: ~92% videos win on Aes, "
                 "~75% lose on IQ (retrieval); population and per-video agree.")
    lines.append("3. **Retrieval ≈ LoRA on Aes/IQ**, differs on Subj/Dyn/motion/temporal.")
    lines.append("4. **K5 vs K10, SIM vs RAND** — nearly identical per-video (same table row patterns).")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", type=Path,
                    default=Path("sweep_experiment/results/panda_1000v_standard/NOTTA"))
    ap.add_argument("--method", action="append", required=True, type=_parse_method_arg,
                    metavar="ID:PATH:LABEL")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vbench-threshold", type=float, default=0.01)
    args = ap.parse_args()

    report = build_report(args.method, args.baseline_dir,
                            vbench_threshold=args.vbench_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
