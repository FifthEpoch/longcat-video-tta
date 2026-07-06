"""Shared helpers for learned-verifier probe routing on the budget pilot."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from scripts.budget_routing_common import labeled_mask
from scripts.train_vbench_headroom_router import eval_config_pick_policy

PROBE2 = ("S2_LR5e3", "S10_LR5e3")
FULL3 = ("S5_LR5e3", "S10_LR5e3", "S20_LR1e3")
FULL_MAP = {"S2_LR5e3": "S5_LR5e3", "S10_LR5e3": "S10_LR5e3"}

_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def canonical_video_id(name: str) -> str:
    stem = Path(name).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


def iter_probe_mp4s(series_root: Path, run_id: str) -> Iterator[Tuple[str, Path]]:
    root = series_root / run_id
    if not root.is_dir():
        return
    for mp4 in sorted(root.rglob("*.mp4")):
        if "videos" not in mp4.parts:
            continue
        yield canonical_video_id(mp4.name), mp4


def score_csv_columns(backend: str) -> List[str]:
    if backend == "videoscore":
        return ["vq", "tc", "dd", "tva", "fc", "mean5"]
    if backend == "videoreward":
        return ["vq", "mq", "ta", "overall"]
    if backend == "visionreward":
        return ["score"]
    raise ValueError(f"unknown backend: {backend}")


def load_score_csv(path: Path, backend: str) -> Dict[str, Dict[str, float]]:
    """Return {video_id: {dim: value}}."""
    dims = score_csv_columns(backend)
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id") or ""
            if not vid:
                continue
            rec: Dict[str, float] = {}
            for d in dims:
                v = row.get(d)
                if v is None or v == "":
                    continue
                try:
                    rec[d] = float(v)
                except ValueError:
                    continue
            if rec:
                out[vid] = rec
    return out


def merge_score_shards(
    csv_dir: Path,
    run_id: str,
    backend: str,
) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {}
    pattern = f"{run_id}_{backend}_shard*.csv"
    paths = sorted(csv_dir.glob(pattern))
    if not paths:
        paths = sorted(csv_dir.glob(f"{run_id}_shard*.csv"))
    for p in paths:
        merged.update(load_score_csv(p, backend))
    return merged


def load_scores_table(
    csv_dir: Path,
    run_ids: Sequence[str],
    backend: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return {run_id: {video_id: {dim: value}}}."""
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rid in run_ids:
        table[rid] = merge_score_shards(csv_dir, rid, backend)
    return table


def build_verifier_probe_features(
    bundle: dict,
    video_ids: Sequence[str],
    base_X: np.ndarray,
    feat_names: List[str],
    scores_by_run: Dict[str, Dict[str, Dict[str, float]]],
    *,
    probe_runs: Sequence[str] = PROBE2,
    dims: Optional[Sequence[str]] = None,
    include_deltas: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Append probe verifier scores (+ optional deltas vs fixed probe) to Phase-0 block."""
    grid = bundle["grid_runs"]
    fixed_rid = bundle["fixed_run"]
    out = base_X.copy()
    names = list(feat_names)

    if dims is None:
        all_dims: List[str] = []
        for rid in probe_runs:
            for d in scores_by_run.get(rid, {}).values():
                all_dims.extend(d.keys())
                break
        dims = sorted(set(all_dims))

    fixed_scores = scores_by_run.get(fixed_rid, {})

    cols: List[np.ndarray] = []
    for rid in probe_runs:
        if rid not in grid:
            continue
        run_scores = scores_by_run.get(rid, {})
        for dim in dims:
            vec = np.array([
                run_scores.get(vid, {}).get(dim, float("nan")) for vid in video_ids
            ], dtype=float)
            cols.append(vec)
            names.append(f"ver_{rid}_{dim}")
            if include_deltas and rid != fixed_rid:
                fvec = np.array([
                    fixed_scores.get(vid, {}).get(dim, float("nan")) for vid in video_ids
                ], dtype=float)
                cols.append(vec - fvec)
                names.append(f"ver_{rid}_d{dim}")

    if cols:
        out = np.column_stack([out] + cols)
    return out, names


def route_from_probe_verifier(
    video_ids: Sequence[str],
    grid: Sequence[str],
    scores_by_run: Dict[str, Dict[str, Dict[str, float]]],
    *,
    dims: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
    probe_runs: Sequence[str] = PROBE2,
) -> np.ndarray:
    """Rank-based probe routing: best probe → mapped full config."""
    n = len(video_ids)
    picks = np.full(n, -1, dtype=int)
    probe_js = [grid.index(r) for r in probe_runs if r in grid]
    w = weights or {d: 1.0 / max(len(dims), 1) for d in dims}

    for i, vid in enumerate(video_ids):
        best_j, best_s = -1, float("-inf")
        for j in probe_js:
            rid = grid[j]
            rec = scores_by_run.get(rid, {}).get(vid, {})
            s = 0.0
            ok = False
            for dim in dims:
                v = rec.get(dim)
                if v is None or not math.isfinite(v):
                    continue
                s += w.get(dim, 0.0) * float(v)
                ok = True
            if ok and s > best_s:
                best_s, best_j = s, j
        if best_j < 0:
            continue
        rid = grid[best_j]
        target = FULL_MAP.get(rid, rid)
        if target in grid:
            picks[i] = grid.index(target)
    return picks


def eval_probe_route_policy(
    picks: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    mask: Optional[np.ndarray] = None,
) -> dict:
    if mask is None:
        mask = labeled_mask(fixed_vb, Y)
    valid = mask & (picks >= 0)
    pol = eval_config_pick_policy(picks[valid], Y[valid], fixed_vb[valid], grid)
    oracle_idx = np.nanargmax(Y[valid], axis=1)
    pol["oof_oracle_match_rate"] = float(np.mean(picks[valid] == oracle_idx))
    pol["n_scored"] = int(valid.sum())
    pol["n_labeled"] = int(mask.sum())
    return pol


def write_result_row(
    out_dir: Path,
    row: dict,
    *,
    title: str,
    references: Optional[List[str]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = row.get("experiment", "result")
    (out_dir / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    cap = row.get("captured_pct")
    cap_s = f"{cap:.1f}" if cap is not None else "—"
    mr = row.get("match_rate")
    mr_s = f"{100 * mr:.1f}" if mr is not None else "—"
    lines = [
        f"# {title}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Captured % | {cap_s} |",
        f"| Oracle match % | {mr_s} |",
        f"| Videos scored | {row.get('n_scored', '—')} / {row.get('n_videos', '—')} |",
        f"| Features | {row.get('n_features', '—')} |",
        "",
    ]
    if references:
        lines.append("## Reference")
        lines.append("")
        lines.extend(references)
        lines.append("")
    (out_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
