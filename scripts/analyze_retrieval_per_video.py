#!/usr/bin/env python3
"""Per-video analysis for batch-retrieval TTA vs No-TTA baseline.

Loads per-video PSNR/SSIM/LPIPS from chunk ``summary.json`` files (or
``merged_summary.json``), joins against a separate No-TTA baseline series,
and reports distributional effects — not just population means.

Typical usage (UCF template — merged retrieval results exist on cluster):

    python3 scripts/analyze_retrieval_per_video.py \\
        --series-root sweep_experiment/results/ucf101_932v_retrieval \\
        --baseline-series-root sweep_experiment/results/ucf101_932v_standard \\
        --baseline NOTTA \\
        --methods K5_SIM K5_RAND K10_SIM K10_RAND \\
        --output-dir sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/ucf_retrieval

Panda 1000v retrieval (once ``panda_1000v_retrieval/`` is populated):

    python3 scripts/analyze_retrieval_per_video.py \\
        --series-root sweep_experiment/results/panda_1000v_retrieval \\
        --baseline-series-root sweep_experiment/results/panda_1000v_standard \\
        --baseline NOTTA \\
        --output-dir sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/panda_retrieval

Optional oracle routing ceiling (same eval set, from standard sweep gains):

    python3 scripts/analyze_retrieval_per_video.py ... \\
        --oracle-gains-csv sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv

Outputs:
  - ``per_video_retrieval_gains.csv``
  - ``retrieval_per_video_summary.md``

Dependencies: numpy only (no pandas / scipy).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_per_video_tta_gain import (  # noqa: E402
    _canonical_video_id,
    _coerce_float,
    _records_from_blob,
    autodiscover_methods,
    load_per_video_metrics,
    pearson_r,
    quantile_bin_assign,
    spearman_rho,
)

_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")

_SIM_SCALAR_KEYS = (
    "mean_neighbor_sim",
    "mean_neighbour_sim",
    "neighbor_sim_mean",
    "neighbour_sim_mean",
    "retrieval_mean_sim",
    "batch_neighbor_sim_mean",
    "batch_neighbour_sim_mean",
    "neighbor_similarity_mean",
    "neighbour_similarity_mean",
)

_SIM_LIST_KEYS = (
    "neighbor_similarities",
    "neighbour_similarities",
    "neighbor_sims",
    "neighbour_sims",
    "retrieval_neighbor_sims",
    "retrieval_neighbour_sims",
)


def _infer_baseline_series_root(series_root: Path) -> Path:
    name = series_root.name.lower()
    if "ucf" in name:
        return Path("sweep_experiment/results/ucf101_932v_standard")
    if "panda" in name:
        return Path("sweep_experiment/results/panda_1000v_standard")
    return Path("sweep_experiment/results/panda_1000v_standard")


def _summary_candidates(method_dir: Path) -> List[Path]:
    candidates = sorted(method_dir.glob("chunk_*/summary.json"))
    if not candidates:
        candidates = sorted(method_dir.glob("chunk_*/results.json"))
    if not candidates:
        for flat_name in ("merged_summary.json", "summary.json"):
            flat = method_dir / flat_name
            if flat.exists():
                candidates = [flat]
                break
    return candidates


def _extract_neighbor_sim(record: dict) -> Optional[float]:
    for key in _SIM_SCALAR_KEYS:
        v = _coerce_float(record.get(key))
        if v is not None:
            return v
    for key in _SIM_LIST_KEYS:
        raw = record.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            vals = [_coerce_float(x) for x in raw]
            vals = [x for x in vals if x is not None]
            if vals:
                return float(np.mean(vals))
    retrieval = record.get("retrieval")
    if isinstance(retrieval, dict):
        for key in _SIM_SCALAR_KEYS + _SIM_LIST_KEYS:
            if key in retrieval:
                nested = {"k": retrieval[key]}
                if key in _SIM_SCALAR_KEYS:
                    v = _coerce_float(retrieval[key])
                    if v is not None:
                        return v
                elif isinstance(retrieval[key], (list, tuple)):
                    vals = [_coerce_float(x) for x in retrieval[key]]
                    vals = [x for x in vals if x is not None]
                    if vals:
                        return float(np.mean(vals))
    return None


def load_per_video_records(method_dir: Path) -> Dict[str, dict]:
    """Return {video_id -> merged per-video record dict}."""
    out: Dict[str, dict] = {}
    for cf in _summary_candidates(method_dir):
        try:
            with cf.open() as f:
                blob = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {cf}: {e}", file=sys.stderr)
            continue
        for r in _records_from_blob(blob):
            vid_raw = (
                r.get("video_name")
                or r.get("video_id")
                or r.get("video")
                or r.get("video_path")
                or r.get("path")
            )
            vid = _canonical_video_id(vid_raw if vid_raw is not None else "")
            if not vid:
                continue
            row = dict(r)
            row["psnr"] = _coerce_float(r.get("psnr", r.get("avg_psnr")))
            row["ssim"] = _coerce_float(r.get("ssim", r.get("avg_ssim")))
            row["lpips"] = _coerce_float(r.get("lpips", r.get("avg_lpips")))
            row["neighbor_sim"] = _extract_neighbor_sim(r)
            out[vid] = row
    return out


def _resolve_methods(
    series_root: Path,
    explicit: Optional[List[str]],
) -> List[Tuple[str, Path]]:
    if explicit:
        out: List[Tuple[str, Path]] = []
        for name in explicit:
            cand = series_root / name
            if cand.is_dir():
                out.append((name, cand))
            else:
                print(f"[warn] method dir missing: {cand}", file=sys.stderr)
        return out
    return [(n, series_root / n) for n in autodiscover_methods(series_root)]


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _stats(arr: Sequence[float]) -> Tuple[int, float, float, float, float, float]:
    a = np.asarray([x for x in arr if x is not None and not math.isnan(x)], dtype=float)
    if a.size == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    return (
        int(a.size),
        float(a.mean()),
        float(np.median(a)),
        float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        float(np.percentile(a, 25)),
        float(np.percentile(a, 75)),
    )


def _win_loss_counts(delta: np.ndarray) -> Tuple[int, int, int]:
    finite = delta[~np.isnan(delta)]
    helped = int((finite > 0).sum())
    hurt = int((finite < 0).sum())
    flat = int((finite == 0).sum())
    return helped, hurt, flat


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool]]:
    a = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if a.size == 0:
        return None, None, None, None
    mean = float(a.mean())
    if a.size < 2:
        return mean, None, None, None
    rng = np.random.default_rng(seed)
    boot = [float(a[rng.integers(0, a.size, size=a.size)].mean()) for _ in range(n_boot)]
    boot_arr = np.asarray(boot, dtype=np.float64)
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    excludes_zero = bool((ci_lo > 0.0) or (ci_hi < 0.0))
    return mean, ci_lo, ci_hi, excludes_zero


def paired_bootstrap_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """Bootstrap CI for mean(a - b) on paired finite entries."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = ~(np.isnan(aa) | np.isnan(bb))
    d = aa[mask] - bb[mask]
    n = int(d.size)
    if n == 0:
        return None, None, None, 0
    mean = float(d.mean())
    if n < 2:
        return mean, None, None, n
    rng = np.random.default_rng(seed)
    boot = [float(d[rng.integers(0, n, size=n)].mean()) for _ in range(n_boot)]
    boot_arr = np.asarray(boot, dtype=np.float64)
    return mean, float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5)), n


def load_oracle_uplift(gains_csv: Path, baseline: str) -> Dict[str, float]:
    """Return {video_id -> oracle_best_psnr - baseline_psnr} from gains CSV."""
    if not gains_csv.exists():
        print(f"[warn] oracle gains CSV not found: {gains_csv}", file=sys.stderr)
        return {}
    methods = ("NOTTA", "ADA", "LORA_R8_TTA")
    out: Dict[str, float] = {}
    with gains_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id", "").strip()
            if not vid:
                continue
            psnrs = []
            for m in methods:
                key = f"{m}_psnr"
                v = row.get(key, "")
                if v in ("", None):
                    continue
                try:
                    psnrs.append(float(v))
                except ValueError:
                    continue
            base_key = f"{baseline}_psnr"
            try:
                base = float(row[base_key])
            except (KeyError, ValueError):
                continue
            if not psnrs:
                continue
            out[vid] = max(psnrs) - base
    return out


def _fmt(v: Optional[float], decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _method_pair_key(k: int, mode: str) -> str:
    return f"K{k}_{mode}"


def build_rows(
    video_ids: List[str],
    baseline_name: str,
    baseline_pv: Dict[str, Dict[str, Optional[float]]],
    method_records: Dict[str, Dict[str, dict]],
) -> List[dict]:
    rows: List[dict] = []
    for vid in video_ids:
        base = baseline_pv.get(vid, {})
        row: dict = {
            "video_id": vid,
            f"{baseline_name}_psnr": base.get("psnr"),
            f"{baseline_name}_ssim": base.get("ssim"),
            f"{baseline_name}_lpips": base.get("lpips"),
        }
        for method, recs in method_records.items():
            rec = recs.get(vid, {})
            for metric in ("psnr", "ssim", "lpips"):
                row[f"{method}_{metric}"] = rec.get(metric)
                if metric == "psnr":
                    row[f"{method}_dpsnr"] = _delta(rec.get("psnr"), base.get("psnr"))
                elif metric == "ssim":
                    row[f"{method}_dssim"] = _delta(rec.get("ssim"), base.get("ssim"))
                else:
                    row[f"{method}_dlpips"] = _delta(rec.get("lpips"), base.get("lpips"))
            ns = rec.get("neighbor_sim")
            if ns is not None:
                row[f"{method}_neighbor_sim"] = ns
        rows.append(row)
    return rows


def write_csv(out_path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if math.isnan(v):
                return ""
            return f"{v:.6f}"
        return str(v)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fieldnames})


def build_markdown_report(
    *,
    series_root: Path,
    baseline_series_root: Path,
    baseline_name: str,
    methods: List[str],
    rows: List[dict],
    oracle_uplift: Dict[str, float],
    n_bins: int = 5,
) -> str:
    lines: List[str] = [
        "# Batch-retrieval per-video analysis",
        "",
        f"- **Retrieval series:** `{series_root}`",
        f"- **Baseline series:** `{baseline_series_root}` / `{baseline_name}`",
        f"- **Methods:** {', '.join(f'`{m}`' for m in methods)}",
        f"- **N videos (intersection):** {len(rows)}",
        "",
    ]

    dpsnr_by_method: Dict[str, np.ndarray] = {}
    for m in methods:
        dpsnr_by_method[m] = np.asarray(
            [r.get(f"{m}_dpsnr") if r.get(f"{m}_dpsnr") is not None else float("nan") for r in rows],
            dtype=float,
        )

    lines.extend([
        "## ΔPSNR distribution (method − NOTTA)",
        "",
        "| method | N | mean | median | std | p25 | p75 | helped | hurt | flat | p(helped) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for m in methods:
        d = dpsnr_by_method[m]
        n, mean, med, std, p25, p75 = _stats(d.tolist())
        helped, hurt, flat = _win_loss_counts(d)
        p_help = helped / n if n else float("nan")
        lines.append(
            f"| `{m}` | {n} | {_fmt(mean)} | {_fmt(med)} | {_fmt(std)} | "
            f"{_fmt(p25)} | {_fmt(p75)} | {helped} | {hurt} | {flat} | {p_help:.3f} |"
        )
    lines.append("")

    base_psnr = np.asarray(
        [r.get(f"{baseline_name}_psnr") if r.get(f"{baseline_name}_psnr") is not None else float("nan")
         for r in rows],
        dtype=float,
    )
    finite_base = base_psnr[~np.isnan(base_psnr)]
    if finite_base.size >= n_bins:
        bin_idx, edges = quantile_bin_assign(finite_base, n_bins)
        lines.extend([
            f"## ΔPSNR by baseline-PSNR quintile (N={len(rows)})",
            "",
            "Tests whether retrieval helps low-PSNR (hard) clips more than easy ones.",
            "",
        ])
        for m in methods:
            d = dpsnr_by_method[m]
            lines.append(f"### `{m}`")
            lines.append("")
            lines.append("| quintile | baseline PSNR range | N | mean ΔPSNR |")
            lines.append("|---|---|---:|---:|")
            for b in range(n_bins):
                mask = bin_idx == b
                if not mask.any():
                    continue
                lo = edges[b] if b < len(edges) else float("nan")
                hi = edges[b + 1] if b + 1 < len(edges) else float("nan")
                vals = d[mask]
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                lines.append(
                    f"| Q{b + 1} | [{_fmt(lo)} , {_fmt(hi)}] | {vals.size} | {_fmt(float(vals.mean()))} |"
                )
            lines.append("")

    lines.extend([
        "## SIM vs RAND (paired per-video ΔPSNR)",
        "",
        "Positive mean(SIM−RAND) ⇒ similarity retrieval beats content-agnostic batching.",
        "",
        "| pair | N paired | mean Δ(SIM−RAND) | bootstrap 95% CI | CI excludes 0? |",
        "|---|---:|---:|---|---|",
    ])
    pairs = [(5, "SIM", "RAND"), (10, "SIM", "RAND")]
    for k, sim_mode, rand_mode in pairs:
        sim = _method_pair_key(k, sim_mode)
        rand = _method_pair_key(k, rand_mode)
        if sim not in methods or rand not in methods:
            continue
        sim_d = dpsnr_by_method[sim]
        rand_d = dpsnr_by_method[rand]
        mean_diff, ci_lo, ci_hi, n = paired_bootstrap_mean_diff(sim_d, rand_d)
        excl = "yes" if mean_diff is not None and ci_lo is not None and ci_hi is not None and ((ci_lo > 0) or (ci_hi < 0)) else "no"
        ci_str = f"[{_fmt(ci_lo)}, {_fmt(ci_hi)}]" if ci_lo is not None else "—"
        lines.append(
            f"| `{sim}` vs `{rand}` | {n} | {_fmt(mean_diff)} | {ci_str} | {excl} |"
        )
    lines.append("")

    sim_methods = [m for m in methods if m.endswith("_SIM")]
    if sim_methods:
        lines.extend([
            "## Neighbour caption similarity vs ΔPSNR",
            "",
            "Uses per-video neighbour similarity if logged in chunk summaries "
            "(fields like `mean_neighbor_sim` or `neighbor_similarities`).",
            "",
            "| method | N w/ sim | Pearson r | Spearman ρ |",
            "|---|---:|---:|---:|",
        ])
        for m in sim_methods:
            xs = []
            ys = []
            for r in rows:
                sim = r.get(f"{m}_neighbor_sim")
                dps = r.get(f"{m}_dpsnr")
                if sim is None or dps is None:
                    continue
                xs.append(float(sim))
                ys.append(float(dps))
            if len(xs) < 3:
                lines.append(f"| `{m}` | {len(xs)} | — | — |")
                continue
            xarr = np.asarray(xs, dtype=float)
            yarr = np.asarray(ys, dtype=float)
            lines.append(
                f"| `{m}` | {len(xs)} | {_fmt(pearson_r(xarr, yarr), 4)} | "
                f"{_fmt(spearman_rho(xarr, yarr), 4)} |"
            )
        lines.append("")
    else:
        lines.extend([
            "## Neighbour caption similarity vs ΔPSNR",
            "",
            "_No `_SIM` methods in this run, or neighbour similarity was not logged "
            "in summary.json per-video records._",
            "",
        ])

    if oracle_uplift:
        oracle_vals = []
        for r in rows:
            vid = r["video_id"]
            if vid in oracle_uplift:
                oracle_vals.append(oracle_uplift[vid])
        lines.extend([
            "## Oracle routing ceiling (optional)",
            "",
            f"Oracle = per-video best PSNR among NOTTA/ADA/LORA minus `{baseline_name}` "
            f"(from `--oracle-gains-csv`). Compare retrieval mean ΔPSNR to this ceiling.",
            "",
        ])
        if oracle_vals:
            n, mean, med, std, p25, p75 = _stats(oracle_vals)
            lines.append(
                f"- Oracle uplift: mean={_fmt(mean)} dB, median={_fmt(med)} dB, "
                f"N={n}"
            )
            lines.append("")
            lines.append("| method | mean ΔPSNR | % of oracle ceiling (mean) |")
            lines.append("|---|---:|---:|")
            oracle_mean = mean
            for m in methods:
                d = dpsnr_by_method[m]
                m_mean = float(np.nanmean(d)) if np.any(~np.isnan(d)) else float("nan")
                pct = (100.0 * m_mean / oracle_mean) if oracle_mean and not math.isnan(m_mean) else float("nan")
                lines.append(f"| `{m}` | {_fmt(m_mean)} | {_fmt(pct, 1)}% |")
            lines.append("")
        else:
            lines.append("_Oracle CSV provided but no overlapping video IDs._")
            lines.append("")

    lines.extend([
        "## Head-to-head win rates (ΔPSNR)",
        "",
        "Fraction of videos where method A beats method B (strictly higher ΔPSNR).",
        "",
        "| A | B | A wins | B wins | ties |",
        "|---|---|---:|---:|---:|",
    ])
    for i, ma in enumerate(methods):
        for mb in methods[i + 1:]:
            a_w = b_w = ties = 0
            da = dpsnr_by_method[ma]
            db = dpsnr_by_method[mb]
            for x, y in zip(da, db):
                if math.isnan(x) or math.isnan(y):
                    continue
                if x > y:
                    a_w += 1
                elif y > x:
                    b_w += 1
                else:
                    ties += 1
            lines.append(f"| `{ma}` | `{mb}` | {a_w} | {b_w} | {ties} |")
    lines.append("")

    lines.extend([
        "## Interpretation checklist",
        "",
        "1. **Population mean ≈ 0 is not the whole story** — check helped/hurt counts "
        "and ΔPSNR std; large opposing tails can cancel in the mean.",
        "2. **SIM vs RAND paired test** — on UCF, expect null (class-block pool); "
        "on Panda with hash-ordered pool, a SIM advantage would support topical retrieval.",
        "3. **Neighbour-sim correlation** — positive r suggests gains track caption similarity "
        "(only meaningful when SIM actually retrieves diverse neighbours).",
        "4. **Oracle ceiling** — if retrieval mean Δ ≪ oracle uplift, routing policy "
        "has headroom even when batch-retrieval alone is neutral.",
        "",
    ])
    return "\n".join(lines)


def discover_data_status(series_root: Path, baseline_series_root: Path, baseline: str) -> None:
    print(f"=== Data discovery ===")
    print(f"Retrieval root: {series_root}  exists={series_root.exists()}")
    if series_root.exists():
        methods = autodiscover_methods(series_root)
        print(f"  methods with summaries: {methods or '(none)'}")
        for m in methods:
            mdir = series_root / m
            n_chunks = len(list(mdir.glob("chunk_*/summary.json")))
            merged = (mdir / "merged_summary.json").exists()
            print(f"    {m:16s} chunks={n_chunks:>2d} merged={merged}")
    else:
        print("  (retrieval series directory missing — run merge after sweep completes)")

    print(f"Baseline root:  {baseline_series_root}  exists={baseline_series_root.exists()}")
    base_dir = baseline_series_root / baseline
    if base_dir.exists():
        n = len(load_per_video_metrics(base_dir))
        merged = (base_dir / "merged_summary.json").exists()
        print(f"  {baseline}: per-video records={n} merged={merged}")
    else:
        print(f"  baseline method dir missing: {base_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--series-root", type=Path,
        default=Path("sweep_experiment/results/ucf101_932v_retrieval"),
        help="Retrieval experiment root (method subdirs).",
    )
    ap.add_argument(
        "--baseline-series-root", type=Path, default=None,
        help="Standard sweep root containing NOTTA (default: inferred from series name).",
    )
    ap.add_argument("--baseline", default="NOTTA", help="Baseline method subdir name.")
    ap.add_argument(
        "--methods", nargs="*", default=None,
        help="Retrieval methods (default: auto-detect under --series-root).",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("sweep_experiment/reports/per_video_analysis/_retrieval_unspecified"),
    )
    ap.add_argument(
        "--oracle-gains-csv", type=Path, default=None,
        help="Optional per_video_gains.csv for oracle routing ceiling.",
    )
    ap.add_argument("--n-bins", type=int, default=5, help="Baseline-PSNR quantile bins.")
    ap.add_argument(
        "--discover-only", action="store_true",
        help="Print what exists under series/baseline roots and exit.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    baseline_series_root = args.baseline_series_root
    if baseline_series_root is None:
        baseline_series_root = _infer_baseline_series_root(args.series_root)

    discover_data_status(args.series_root, baseline_series_root, args.baseline)
    if args.discover_only:
        return 0

    method_specs = _resolve_methods(args.series_root, args.methods)
    if not method_specs:
        print("[error] no retrieval methods found.", file=sys.stderr)
        return 2

    baseline_dir = baseline_series_root / args.baseline
    if not baseline_dir.is_dir():
        print(f"[error] baseline dir missing: {baseline_dir}", file=sys.stderr)
        return 2

    baseline_pv = load_per_video_metrics(baseline_dir)
    baseline_pv = {vid: row for vid, row in baseline_pv.items() if row.get("psnr") is not None}
    print(f"Baseline `{args.baseline}`: {len(baseline_pv)} videos with PSNR")

    method_names = [n for n, _ in method_specs]
    method_records: Dict[str, Dict[str, dict]] = {}
    for name, mdir in method_specs:
        recs = load_per_video_records(mdir)
        recs = {vid: r for vid, r in recs.items() if r.get("psnr") is not None}
        method_records[name] = recs
        n_sim = sum(1 for r in recs.values() if r.get("neighbor_sim") is not None)
        print(f"  {name:16s}  videos={len(recs):>4d}  w/neighbor_sim={n_sim}")

    common: Optional[set] = set(baseline_pv.keys())
    for recs in method_records.values():
        common &= set(recs.keys())
    video_ids = sorted(common or set())
    print(f"Intersection: {len(video_ids)} videos")
    if not video_ids:
        print("[error] empty intersection.", file=sys.stderr)
        return 2

    rows = build_rows(video_ids, args.baseline, baseline_pv, method_records)

    fieldnames = ["video_id", f"{args.baseline}_psnr", f"{args.baseline}_ssim", f"{args.baseline}_lpips"]
    for m in method_names:
        for suffix in ("psnr", "ssim", "lpips", "dpsnr", "dssim", "dlpips"):
            fieldnames.append(f"{m}_{suffix}")
        if any(r.get(f"{m}_neighbor_sim") is not None for r in rows):
            fieldnames.append(f"{m}_neighbor_sim")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_video_retrieval_gains.csv"
    write_csv(csv_path, rows, fieldnames)
    print(f"Wrote {csv_path}")

    oracle_uplift: Dict[str, float] = {}
    if args.oracle_gains_csv:
        oracle_uplift = load_oracle_uplift(args.oracle_gains_csv, args.baseline)
        print(f"Oracle uplift loaded for {len(oracle_uplift)} videos")

    report = build_markdown_report(
        series_root=args.series_root,
        baseline_series_root=baseline_series_root,
        baseline_name=args.baseline,
        methods=method_names,
        rows=rows,
        oracle_uplift=oracle_uplift,
        n_bins=args.n_bins,
    )
    md_path = args.output_dir / "retrieval_per_video_summary.md"
    md_path.write_text(report, encoding="utf-8")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
