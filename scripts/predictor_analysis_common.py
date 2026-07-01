#!/usr/bin/env python3
"""Shared loaders and outcome definitions for predictor-transfer analysis."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS
from scripts.correlate_tta_gain_with_features import (
    FEATURE_INTERPRETATIONS,
    TIER1_FEATURES,
    load_bpp_csv,
    load_features_csv,
    load_flow_csv,
    load_fft_csv,
    load_loss_var_csv,
    load_motion_csv,
    load_ood_csv,
    load_tier3_csv,
    load_vae_recerr_csv,
    spearman_rho,
)
from scripts.per_video_metric_store import load_gains_csv as load_gains_float
from scripts.per_video_metric_store import vbench_total
from scripts.summarize_vbench_population_per_video import DIM_SHORT

METHODS_DEFAULT: Tuple[str, ...] = (
    "ADA",
    "LORA_R8_TTA",
    "K5_SIM",
    "K5_RAND",
    "K10_SIM",
    "K10_RAND",
)

GATE_METHODS: Tuple[str, ...] = ("ADA", "LORA_R8_TTA")

BASIC_OUTCOMES: Tuple[str, ...] = ("psnr", "ssim", "lpips")
VBENCH_OUTCOMES: Tuple[str, ...] = VBENCH_DIMS + ("vbench_total",)

PASS_RHO = 0.2
PASS_MIN_METHODS = 2


def _coerce(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


@dataclass(frozen=True)
class OutcomeSpec:
    key: str
    label: str
    col_suffix: str  # e.g. _dpsnr, _daesthetic_quality, _dvbench_total


def outcome_specs() -> List[OutcomeSpec]:
    specs: List[OutcomeSpec] = [
        OutcomeSpec("psnr", "ΔPSNR", "_dpsnr"),
        OutcomeSpec("ssim", "ΔSSIM", "_dssim"),
        OutcomeSpec("lpips", "ΔLPIPS", "_dlpips"),
    ]
    for dim in VBENCH_DIMS:
        specs.append(
            OutcomeSpec(dim, f"Δ{DIM_SHORT.get(dim, dim)}", f"_d{dim}")
        )
    specs.append(OutcomeSpec("vbench_total", "ΔVBench total", "_dvbench_total"))
    return specs


def enrich_vbench_totals(
    rows: Dict[str, Dict[str, float]], methods: Sequence[str]
) -> None:
    for vid, row in rows.items():
        for method in methods:
            if method == "NOTTA":
                continue
            col = f"{method}_dvbench_total"
            if col in row and not math.isnan(_coerce(row.get(col))):
                continue
            row[col] = vbench_total(row, method) - vbench_total(row, "NOTTA")


def load_vbench_gains(path: Path) -> Tuple[List[str], Dict[str, Dict[str, float]], List[str]]:
    rows, methods = load_gains_float(path)
    methods = [m for m in methods if m != "NOTTA"]
    enrich_vbench_totals(rows, methods)
    video_ids = sorted(rows.keys())
    return video_ids, rows, methods


def notta_baseline_predictors() -> List[Tuple[str, str]]:
    """(column_name, short_label) predictors from NOTTA outputs."""
    preds: List[Tuple[str, str]] = [
        ("NOTTA_psnr", "NOTTA PSNR"),
        ("NOTTA_ssim", "NOTTA SSIM"),
        ("NOTTA_lpips", "NOTTA LPIPS"),
    ]
    for dim in VBENCH_DIMS:
        preds.append((f"NOTTA_{dim}", f"NOTTA {DIM_SHORT.get(dim, dim)}"))
    return preds


def join_feature_tables(
    *,
    features_csv: Path,
    ood_csv: Optional[Path] = None,
    tier3_csv: Optional[Path] = None,
    flow_csv: Optional[Path] = None,
    bpp_csv: Optional[Path] = None,
    fft_csv: Optional[Path] = None,
    vae_recerr_csv: Optional[Path] = None,
    motion_csv: Optional[Path] = None,
    loss_var_csv: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, float]], List[str], Dict[str, str]]:
    """Merge Phase-0 feature CSVs → {vid -> {feat -> float}}, feature names, tier map."""
    merged: Dict[str, Dict[str, float]] = {}
    feature_names: List[str] = []
    tiers: Dict[str, str] = {}

    def _merge_source(src: Dict[str, Dict], cols: Iterable[str], tier: str) -> None:
        for c in cols:
            if c not in feature_names:
                feature_names.append(c)
            tiers[c] = tier
        for vid, row in src.items():
            bucket = merged.setdefault(vid, {})
            for c in cols:
                bucket[c] = _coerce(row.get(c))

    base = load_features_csv(features_csv)
    tier1_present = [c for c in TIER1_FEATURES if any(c in r for r in base.values())]
    _merge_source(base, tier1_present, "T1")

    if ood_csv and ood_csv.exists():
        ood_rows, ood_cols = load_ood_csv(ood_csv)
        _merge_source(ood_rows, ood_cols, "OOD")

    optional_loaders = [
        (tier3_csv, lambda p: load_tier3_csv(p), "T3P"),
        (flow_csv, lambda p: load_flow_csv(p)[0:2], "T1"),
        (bpp_csv, lambda p: load_bpp_csv(p)[0:2], "T1"),
        (fft_csv, lambda p: load_fft_csv(p)[0:2], "T1"),
        (vae_recerr_csv, lambda p: load_vae_recerr_csv(p)[0:2], "T1"),
        (motion_csv, lambda p: load_motion_csv(p)[0:2], "T1"),
        (loss_var_csv, lambda p: load_loss_var_csv(p)[0:2], "OOD"),
    ]
    for path, loader, tier in optional_loaders:
        if path and path.exists():
            rows, cols = loader(path)
            _merge_source(rows, cols, tier)

    return merged, feature_names, tiers


def intersect_videos(*id_sets: Iterable[str]) -> List[str]:
    common: Optional[set] = None
    for s in id_sets:
        st = set(s)
        common = st if common is None else common & st
    return sorted(common or [])


def outcome_column(method: str, spec: OutcomeSpec) -> str:
    if spec.key == "vbench_total":
        return f"{method}_dvbench_total"
    if spec.key in BASIC_OUTCOMES:
        return f"{method}_d{spec.key}"
    return f"{method}_d{spec.key}"


def compute_rho_grid(
    video_ids: Sequence[str],
    predictors: Dict[str, np.ndarray],
    gains: Dict[str, Dict[str, float]],
    methods: Sequence[str],
    specs: Sequence[OutcomeSpec],
) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    for pred_name, pred_vals in predictors.items():
        for method in methods:
            for spec in specs:
                col = outcome_column(method, spec)
                y = np.array(
                    [_coerce(gains.get(vid, {}).get(col)) for vid in video_ids],
                    dtype=float,
                )
                rho = spearman_rho(pred_vals, y)
                rows_out.append({
                    "predictor": pred_name,
                    "method": method,
                    "outcome": spec.key,
                    "outcome_label": spec.label,
                    "rho": rho,
                    "n": int(np.sum(~np.isnan(pred_vals) & ~np.isnan(y))),
                })
    return rows_out


def passes_gate(
    rho_rows: Sequence[Dict[str, object]],
    predictor: str,
    outcome: str,
    *,
    methods: Sequence[str] = GATE_METHODS,
    rho_threshold: float = PASS_RHO,
    min_methods: int = PASS_MIN_METHODS,
) -> Tuple[bool, List[str]]:
    hits: List[str] = []
    for method in methods:
        for row in rho_rows:
            if row["predictor"] != predictor or row["outcome"] != outcome:
                continue
            if row["method"] != method:
                continue
            rho = row["rho"]
            if rho is not None and abs(float(rho)) >= rho_threshold:
                hits.append(method)
                break
    return len(hits) >= min_methods, hits


def write_rho_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["predictor", "method", "outcome", "outcome_label", "rho", "n"],
        )
        w.writeheader()
        for r in rows:
            out = dict(r)
            rho = out.get("rho")
            out["rho"] = f"{rho:.6f}" if rho is not None else ""
            w.writerow(out)


def format_rho(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.3f}"


def predictor_interp(name: str) -> str:
    if name.startswith("NOTTA_"):
        return f"Baseline score before TTA ({name[6:]})"
    return FEATURE_INTERPRETATIONS.get(name, name)
