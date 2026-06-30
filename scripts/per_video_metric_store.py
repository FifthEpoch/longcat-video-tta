#!/usr/bin/env python3
"""Shared per-video metric loading + optional disk cache.

Avoids re-parsing chunk summaries / CSVs when multiple analysis scripts run
in the same pipeline (oracle VBench, correlation plots, predictor tables).

Cache layout (under ``--cache-dir``):
  * ``wide_metrics.parquet`` or ``wide_metrics.csv`` — merged per-video table
  * ``meta.json`` — source paths + mtime fingerprints

Scripts should call ``load_or_build_wide_table()`` once per pipeline run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    load_per_video_vbench,
)
from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402

BASIC_METRICS = ("psnr", "ssim", "lpips")
OOD_DEFAULT_COL = "mean_diffusion_loss_caption"


def _coerce(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _fingerprint(paths: Sequence[Path]) -> str:
    parts = []
    for p in paths:
        if p.exists():
            st = p.stat()
            parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
        else:
            parts.append(f"{p}:missing")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


@dataclass
class WideTable:
    video_ids: List[str]
    rows: Dict[str, Dict[str, float]]
    methods: List[str]
    columns: List[str]

    def column(self, name: str) -> np.ndarray:
        return np.array([self.rows[v].get(name, float("nan")) for v in self.video_ids], dtype=float)


def load_gains_csv(path: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Load ``per_video_gains.csv`` or ``per_video_vbench_gains.csv``."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        methods: List[str] = []
        for fn in fieldnames:
            if fn.endswith("_dpsnr"):
                methods.append(fn[: -len("_dpsnr")])
            else:
                for dim in VBENCH_DIMS:
                    if fn.endswith(f"_d{dim}"):
                        m = fn[: -len(f"_d{dim}")]
                        if m not in methods:
                            methods.append(m)
                        break
        methods = sorted(set(methods))
        rows: Dict[str, Dict[str, float]] = {}
        for r in reader:
            vid = (r.get("video_id") or "").strip()
            if not vid:
                continue
            rows[vid] = {k: _coerce(v) for k, v in r.items() if k != "video_id"}
    return rows, methods


def load_ood_column(path: Path, col: str = OOD_DEFAULT_COL) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vid = (r.get("video_id") or "").strip()
            if vid:
                out[vid] = _coerce(r.get(col))
    return out


def load_features_columns(path: Path, cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    want = set(cols)
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vid = (r.get("video_id") or "").strip()
            if not vid:
                continue
            out[vid] = {c: _coerce(r.get(c)) for c in want if c in r}
    return out


def merge_wide(
    *,
    gains_rows: Dict[str, Dict[str, float]],
    ood: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Dict[str, float]]] = None,
) -> WideTable:
    video_ids = sorted(gains_rows.keys())
    if ood:
        video_ids = sorted(set(video_ids) & set(ood.keys()))
    if extra:
        for d in extra.values():
            video_ids = sorted(set(video_ids) & set(d.keys()))
    methods: List[str] = []
    for row in gains_rows.values():
        for k in row:
            if k.endswith("_dpsnr"):
                methods.append(k[: -len("_dpsnr")])
    methods = sorted(set(methods))
    rows: Dict[str, Dict[str, float]] = {}
    columns: set = set()
    for vid in video_ids:
        merged: Dict[str, float] = dict(gains_rows.get(vid, {}))
        if ood:
            merged[f"ood_{OOD_DEFAULT_COL}"] = ood.get(vid, float("nan"))
            columns.add(f"ood_{OOD_DEFAULT_COL}")
        if extra:
            for prefix, d in extra.items():
                for ck, cv in d.get(vid, {}).items():
                    key = f"{prefix}_{ck}" if not ck.startswith(prefix) else ck
                    merged[key] = cv
                    columns.add(key)
        rows[vid] = merged
        columns.update(merged.keys())
    return WideTable(
        video_ids=video_ids,
        rows=rows,
        methods=methods,
        columns=sorted(columns),
    )


def _cache_valid(cache_dir: Path, fp: str) -> bool:
    meta = cache_dir / "meta.json"
    if not meta.exists():
        return False
    try:
        blob = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return blob.get("fingerprint") == fp


def save_wide_cache(cache_dir: Path, wide: WideTable, fp: str, sources: Dict[str, str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "wide_metrics.csv"
    fieldnames = ["video_id"] + wide.columns
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for vid in wide.video_ids:
            row = {"video_id": vid}
            row.update({k: wide.rows[vid].get(k, "") for k in wide.columns})
            w.writerow(row)
    meta = {"fingerprint": fp, "n_videos": len(wide.video_ids), "sources": sources}
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_wide_cache(cache_dir: Path) -> Optional[WideTable]:
    csv_path = cache_dir / "wide_metrics.csv"
    if not csv_path.exists():
        return None
    rows, methods = load_gains_csv(csv_path)
    video_ids = sorted(rows.keys())
    cols = sorted({k for r in rows.values() for k in r})
    return WideTable(video_ids=video_ids, rows=rows, methods=methods, columns=cols)


def load_or_build_wide_table(
    *,
    gains_csv: Path,
    ood_csv: Optional[Path] = None,
    features_csv: Optional[Path] = None,
    feature_cols: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> WideTable:
    sources = {"gains_csv": str(gains_csv)}
    paths = [gains_csv]
    if ood_csv:
        paths.append(ood_csv)
        sources["ood_csv"] = str(ood_csv)
    if features_csv:
        paths.append(features_csv)
        sources["features_csv"] = str(features_csv)
    fp = _fingerprint(paths)

    if cache_dir and not force_rebuild and _cache_valid(cache_dir, fp):
        cached = load_wide_cache(cache_dir)
        if cached:
            return cached

    gains_rows, _ = load_gains_csv(gains_csv)
    ood = load_ood_column(ood_csv) if ood_csv and ood_csv.exists() else None
    extra = None
    if features_csv and features_csv.exists() and feature_cols:
        extra = {"feat": load_features_columns(features_csv, feature_cols)}

    wide = merge_wide(gains_rows=gains_rows, ood=ood, extra=extra)
    if cache_dir:
        save_wide_cache(cache_dir, wide, fp, sources)
    return wide


def vbench_total(row: Dict[str, float], method: str) -> float:
    vals = [_coerce(row.get(f"{method}_d{d}")) for d in VBENCH_DIMS]
    if any(math.isnan(v) for v in vals):
        return float("nan")
    return float(np.mean(vals))


def oracle_pick_max(
    row: Dict[str, float],
    methods: Sequence[str],
    metric_suffix: str,
    *,
    higher_is_better: bool = True,
) -> Tuple[Optional[str], float]:
    """Return (winning_method, metric_value). ``metric_suffix`` e.g. ``_psnr`` or ``_dpsnr``."""
    best_m: Optional[str] = None
    best_v = float("-inf") if higher_is_better else float("inf")
    for m in methods:
        v = _coerce(row.get(f"{m}{metric_suffix}"))
        if math.isnan(v):
            continue
        if higher_is_better and v > best_v:
            best_v, best_m = v, m
        elif not higher_is_better and v < best_v:
            best_v, best_m = v, m
    if best_m is None:
        return None, float("nan")
    return best_m, best_v


def spearman_rho(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return None
    x, y = xs[mask], ys[mask]
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if den <= 0:
        return None
    return float((rx * ry).sum() / den)
