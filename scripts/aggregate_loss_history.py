#!/usr/bin/env python3
"""Per-video TTA loss-history extraction from on-disk chunk summaries.

Background
----------
The TTA runners (``delta_experiment/scripts/run_tinylora.py``,
``run_delta_a.py``, ``run_delta_b.py``, ``run_delta_c.py``,
``run_norm_tune_tta.py``, ``run_film_tta.py``,
``lora_experiment/scripts/run_lora_tta.py``) all populate a per-step
``losses`` list inside ``optimize_*``, but only the ``final_loss`` value
makes it into the per-video result dict. The full per-step training-loss
trajectory is therefore NOT recoverable from the JSON outputs — it only
appears in slurm stdout.

HOWEVER, when early stopping is enabled (the default for all of these
runners), the per-video result dict DOES persist a held-out anchor-loss
trajectory under ``result['early_stopping_info']['loss_history']``,
sampled at ``check_every`` step intervals (default every 2 steps). This
is the *anchor* (held-out validation) loss, not the training loss, but
it is the loss the runner actually uses to decide which checkpoint to
keep — i.e. the right quantity to ask "does this loss decrease for
winning videos and stay flat for losing ones?".

This script reads that history out of every chunk's ``summary.json``,
joins it with per-video ΔPSNR (computed against the same series's
baseline method, default ``NOTTA``), and produces:

  per_video_loss_summary.csv   — one row per (method, video_id): n_checks,
                                  initial_loss, best_loss, final_loss,
                                  loss_decrease_pct, best_step,
                                  stopped_early, baseline_psnr,
                                  method_psnr, dpsnr, winner_band
  per_video_loss_curves.csv    — long format (method, video_id, step,
                                  anchor_loss) for plotting
  loss_curves_<METHOD>.png     — overlaid loss curves colored by
                                  winner/loser band
  loss_decrease_vs_dpsnr.png   — scatter of loss_decrease_pct vs ΔPSNR,
                                  one subplot per method
  summary.md                    — correlation table (Pearson + Spearman)
                                  between loss_decrease_pct and ΔPSNR,
                                  per method, plus per-band group stats

Inputs
------
``<series>/<METHOD>/chunk_*/summary.json`` for every method under the
series root. Auto-detects methods.

CPU-only — numpy + matplotlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Canonical video-id extraction (mirrors analyze_per_video_tta_gain.py)
# ---------------------------------------------------------------------------
_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: Optional[str]) -> str:
    if not s:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


def _coerce_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


# ---------------------------------------------------------------------------
# Per-video record extraction
# ---------------------------------------------------------------------------
def _records_from_blob(blob) -> List[dict]:
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    if not isinstance(blob, dict):
        return []
    for key in ("results", "per_video_results", "per_video"):
        v = blob.get(key)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
    return []


def _vid_of(r: dict) -> str:
    raw = (r.get("video_name") or r.get("video_id") or r.get("video")
           or r.get("video_path") or r.get("path"))
    return _canonical_video_id(raw if raw is not None else "")


def load_method_per_video(method_dir: Path) -> Dict[str, dict]:
    """Read all chunk_*/summary.json and return {video_id: record}.

    Falls back to merged_summary.json if individual chunk files are missing
    (note: merged_summary.json does NOT contain per-video results — only
    the chunks do).
    """
    out: Dict[str, dict] = {}
    chunks = sorted(method_dir.glob("chunk_*/summary.json"))
    for cf in chunks:
        try:
            with cf.open() as f:
                blob = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {cf}: {e}", file=sys.stderr)
            continue
        for r in _records_from_blob(blob):
            vid = _vid_of(r)
            if not vid:
                continue
            out[vid] = r
    return out


def autodiscover_methods(series_path: Path) -> List[str]:
    if not series_path.exists():
        return []
    out: List[str] = []
    for sub in sorted(p for p in series_path.iterdir() if p.is_dir()):
        if any(sub.glob("chunk_*/summary.json")):
            out.append(sub.name)
    return out


def _coerce_loss_history(es_info) -> List[Tuple[int, float]]:
    """early_stopping_info['loss_history'] is List[Tuple[int, float]] in JSON
    but ``json.load`` returns List[List[int, float]]. Coerce to (int, float)."""
    if not es_info or not isinstance(es_info, dict):
        return []
    raw = es_info.get("loss_history") or []
    out: List[Tuple[int, float]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            step = int(entry[0])
            loss = float(entry[1])
        except (TypeError, ValueError):
            continue
        if math.isnan(loss) or math.isinf(loss):
            continue
        out.append((step, loss))
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def pearson_r(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return None
    x = xs[mask].astype(np.float64)
    y = ys[mask].astype(np.float64)
    sx, sy = x - x.mean(), y - y.mean()
    den = math.sqrt(float((sx * sx).sum()) * float((sy * sy).sum()))
    if den <= 0:
        return None
    return float((sx * sy).sum() / den)


def spearman_rho(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return None
    def _ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(a.size, dtype=np.float64)
        uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sum_ranks = np.zeros(uniq.size, dtype=np.float64)
            np.add.at(sum_ranks, inv, ranks)
            avg_ranks = sum_ranks / counts
            ranks = avg_ranks[inv]
        return ranks
    return pearson_r(_ranks(xs[mask]), _ranks(ys[mask]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


def plot_loss_curves(
    plt, out_path: Path, method: str,
    rows: List[dict], threshold: float, max_curves_per_band: int = 30,
):
    """Loss curves coloured by winner / middle / loser band.

    Each row in ``rows`` has keys ``loss_history`` (list of (step, loss)),
    ``dpsnr`` (float). Plot at most ``max_curves_per_band`` curves per band
    to avoid overplotting on N=999 videos.
    """
    bands = {
        "winners (Δ>+t)": [],
        "middle (|Δ|≤t)": [],
        "losers (Δ<−t)": [],
    }
    for r in rows:
        if not r.get("loss_history"):
            continue
        d = r.get("dpsnr")
        if d is None or math.isnan(d):
            continue
        if d > threshold:
            bands["winners (Δ>+t)"].append(r)
        elif d < -threshold:
            bands["losers (Δ<−t)"].append(r)
        else:
            bands["middle (|Δ|≤t)"].append(r)

    colors = {
        "winners (Δ>+t)": "tab:green",
        "middle (|Δ|≤t)": "lightgrey",
        "losers (Δ<−t)": "tab:red",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted_any = False
    rng = np.random.default_rng(seed=0)
    for band, items in bands.items():
        if not items:
            continue
        if len(items) > max_curves_per_band:
            pick = rng.choice(len(items), size=max_curves_per_band, replace=False)
            items_plot = [items[i] for i in pick]
        else:
            items_plot = items
        for r in items_plot:
            steps = [s for s, _ in r["loss_history"]]
            losses = [l for _, l in r["loss_history"]]
            if not steps:
                continue
            ax.plot(steps, losses, color=colors[band], alpha=0.35,
                    linewidth=1.0)
            plotted_any = True
        ax.plot([], [], color=colors[band],
                label=f"{band}  N={len(items)} (showing ≤{max_curves_per_band})")
    if not plotted_any:
        plt.close(fig)
        return False
    ax.set_xlabel("TTA step")
    ax.set_ylabel("held-out anchor loss")
    ax.set_title(f"{method} — per-video anchor-loss trajectories by ΔPSNR band "
                 f"(threshold ±{threshold:g} dB)")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_loss_decrease_vs_dpsnr(
    plt, out_path: Path, per_method_rows: Dict[str, List[dict]],
):
    methods = [m for m, rs in per_method_rows.items() if rs]
    if not methods:
        return False
    n = len(methods)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
        squeeze=False, sharey=False,
    )
    for k, m in enumerate(methods):
        ax = axes[k // ncols][k % ncols]
        xs = []
        ys = []
        for r in per_method_rows[m]:
            ld = r.get("loss_decrease_pct")
            dp = r.get("dpsnr")
            if ld is None or dp is None:
                continue
            if math.isnan(ld) or math.isnan(dp):
                continue
            xs.append(ld)
            ys.append(dp)
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.scatter(xs, ys, s=10, alpha=0.5, color="tab:blue", edgecolor="none")
        r = pearson_r(xs, ys)
        rho = spearman_rho(xs, ys)
        if r is not None or rho is not None:
            ax.set_title(f"{m}  N={xs.size}  r={r:+.3f}  ρ={rho:+.3f}"
                         if (r is not None and rho is not None)
                         else f"{m}  N={xs.size}")
        else:
            ax.set_title(f"{m}  N={xs.size}")
        if k // ncols == nrows - 1:
            ax.set_xlabel("loss decrease % = (initial − best) / initial")
        if k % ncols == 0:
            ax.set_ylabel(r"$\Delta$PSNR (dB)")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_curves_csv(out_path: Path, per_method_rows: Dict[str, List[dict]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "video_id", "step", "anchor_loss"])
        for m, rows in per_method_rows.items():
            for r in rows:
                vid = r["video_id"]
                for step, loss in r.get("loss_history", []):
                    w.writerow([m, vid, step, f"{loss:.6f}"])


def write_summary_csv(out_path: Path, per_method_rows: Dict[str, List[dict]]):
    fieldnames = [
        "method", "video_id", "n_checks", "best_step",
        "initial_loss", "best_loss", "final_loss",
        "loss_decrease_pct", "stopped_early",
        "baseline_psnr", "method_psnr", "dpsnr", "winner_band",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m, rows in per_method_rows.items():
            for r in rows:
                w.writerow({
                    "method": m,
                    "video_id": r["video_id"],
                    "n_checks": r.get("n_checks", 0),
                    "best_step": r.get("best_step", ""),
                    "initial_loss": "" if r.get("initial_loss") is None else f"{r['initial_loss']:.6f}",
                    "best_loss":    "" if r.get("best_loss") is None else f"{r['best_loss']:.6f}",
                    "final_loss":   "" if r.get("final_loss") is None else f"{r['final_loss']:.6f}",
                    "loss_decrease_pct": "" if r.get("loss_decrease_pct") is None else f"{r['loss_decrease_pct']:.6f}",
                    "stopped_early": int(bool(r.get("stopped_early", False))),
                    "baseline_psnr": "" if r.get("baseline_psnr") is None else f"{r['baseline_psnr']:.4f}",
                    "method_psnr":   "" if r.get("method_psnr")   is None else f"{r['method_psnr']:.4f}",
                    "dpsnr":         "" if r.get("dpsnr")         is None else f"{r['dpsnr']:+.4f}",
                    "winner_band":   r.get("winner_band", ""),
                })


def write_summary_md(
    out_path: Path, args: argparse.Namespace, baseline_name: str,
    per_method_rows: Dict[str, List[dict]],
    methods_no_history: List[str], total_videos_seen: int,
):
    lines: List[str] = []
    lines.append("# Per-video TTA loss-history aggregation")
    lines.append("")
    lines.append(f"- Series:               `{args.series_path}`")
    if args.tinylora_series_path:
        lines.append(f"- TinyLoRA series:      `{args.tinylora_series_path}`")
    lines.append(f"- Baseline (for ΔPSNR): `{baseline_name}`")
    lines.append(f"- Winner band threshold: ±{args.psnr_threshold:g} dB")
    lines.append(f"- Total per-video records seen: {total_videos_seen}")
    if methods_no_history:
        lines.append(f"- Methods with NO early-stopping loss history "
                     f"(likely ran with --es-disable, so the per-step "
                     f"trajectory is unrecoverable from JSON — would need "
                     f"slurm stdout): "
                     + ", ".join(f"`{m}`" for m in methods_no_history))
    lines.append("")
    lines.append(
        "## Methodology\n\n"
        "The loss reported here is the *held-out anchor loss* — the flow-"
        "matching loss recomputed on the held-out 25 %% of the conditioning "
        "latents under a fixed noise draw seeded by the video id (see "
        "`delta_experiment/scripts/early_stopping.py`). It is sampled every "
        "`check_every` training steps (default 2), and is the same quantity "
        "the early-stopping logic uses to decide which checkpoint to keep. "
        "It is NOT the training loss (which is not persisted to JSON; only "
        "the FINAL training loss is). For the mechanism question (do "
        "winning videos see their loss decrease and losing ones see it "
        "stay flat or rise?), the anchor loss is the correct quantity to "
        "inspect — the training loss has step-to-step σ-randomisation noise."
    )
    lines.append("")
    lines.append("## Per-method summary")
    lines.append("")
    lines.append("| method | N | mean n_checks | mean initial_loss | "
                 "mean best_loss | mean loss_decrease_pct | "
                 "stopped_early % | r(loss_dec, ΔPSNR) ρ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for m, rows in per_method_rows.items():
        finite = [r for r in rows if r.get("loss_decrease_pct") is not None
                  and r.get("dpsnr") is not None
                  and not math.isnan(r["dpsnr"])]
        if not finite:
            lines.append(f"| `{m}` | 0 | — | — | — | — | — | — |")
            continue
        ldec = np.array([r["loss_decrease_pct"] for r in finite], dtype=float)
        dpsnr = np.array([r["dpsnr"] for r in finite], dtype=float)
        n_checks = np.array([r.get("n_checks", 0) for r in finite], dtype=float)
        init_losses = np.array(
            [r["initial_loss"] for r in finite if r["initial_loss"] is not None],
            dtype=float,
        )
        best_losses = np.array(
            [r["best_loss"] for r in finite if r["best_loss"] is not None],
            dtype=float,
        )
        stopped = np.array(
            [int(bool(r.get("stopped_early"))) for r in finite],
            dtype=float,
        )
        r = pearson_r(ldec, dpsnr)
        rho = spearman_rho(ldec, dpsnr)
        rstr = (f"{r:+.3f} ({rho:+.3f})"
                if (r is not None and rho is not None) else "n/a")
        lines.append(
            f"| `{m}` | {len(finite)} | {n_checks.mean():.1f} | "
            f"{init_losses.mean():.5f} | {best_losses.mean():.5f} | "
            f"{ldec.mean() * 100:.2f}% | {stopped.mean() * 100:.1f}% | "
            f"{rstr} |"
        )
    lines.append("")
    lines.append("## Per-method winner/loser band group stats")
    lines.append("")
    lines.append(f"`band` is determined by ΔPSNR against `{baseline_name}` at "
                 f"threshold ±{args.psnr_threshold:g} dB:\n"
                 f"- `winners` = ΔPSNR > +t\n"
                 f"- `middle`  = |ΔPSNR| ≤ t\n"
                 f"- `losers`  = ΔPSNR < −t\n")
    lines.append("")
    lines.append("| method | band | N | mean initial_loss | mean best_loss | "
                 "mean loss_decrease_pct | mean ΔPSNR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for m, rows in per_method_rows.items():
        for band in ("winners", "middle", "losers"):
            sub = [r for r in rows if r.get("winner_band") == band
                   and r.get("loss_decrease_pct") is not None]
            if not sub:
                continue
            il = np.array(
                [r["initial_loss"] for r in sub if r["initial_loss"] is not None],
                dtype=float,
            )
            bl = np.array(
                [r["best_loss"] for r in sub if r["best_loss"] is not None],
                dtype=float,
            )
            ld = np.array(
                [r["loss_decrease_pct"] for r in sub], dtype=float,
            )
            dp = np.array(
                [r["dpsnr"] for r in sub if r.get("dpsnr") is not None],
                dtype=float,
            )
            il_s = f"{il.mean():.5f}" if il.size else "—"
            bl_s = f"{bl.mean():.5f}" if bl.size else "—"
            dp_s = f"{dp.mean():+.3f}" if dp.size else "—"
            lines.append(
                f"| `{m}` | {band} | {len(sub)} | {il_s} | {bl_s} | "
                f"{ld.mean() * 100:.2f}% | {dp_s} |"
            )
    lines.append("")
    lines.append(
        "**Reading guide.** If winners have systematically larger "
        "`loss_decrease_pct` than losers and middles, the anchor loss IS the "
        "mechanism — TTA actually adapts toward each winner's latent target "
        "and stalls on losers. If all three bands have similar mean "
        "`loss_decrease_pct`, the per-video ΔPSNR is decoupled from the "
        "loss the runner optimised — i.e. the loss is satisfied on every "
        "video but the resulting checkpoint helps some and hurts others "
        "for reasons not visible to the held-out anchor loss."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--series-path", type=Path, required=True,
        help="Primary series root (one subdir per method, each with "
             "chunk_*/summary.json).",
    )
    ap.add_argument(
        "--tinylora-series-path", type=Path, default=None,
        help="Optional second series root (e.g. tinylora_*). Default: none.",
    )
    ap.add_argument(
        "--baseline-method", default="NOTTA",
        help="Method used as baseline for ΔPSNR computation. Default: NOTTA.",
    )
    ap.add_argument(
        "--methods", nargs="*", default=None,
        help="Optional explicit method list. Default: auto-detect.",
    )
    ap.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write summary.md, the two CSVs, and the PNGs.",
    )
    ap.add_argument(
        "--psnr-threshold", type=float, default=0.5,
        help="ΔPSNR band threshold (in dB) for winner/middle/loser grouping.",
    )
    ap.add_argument(
        "--max-curves-per-band", type=int, default=30,
        help="Cap on per-band curves overplotted to avoid 999×3 spaghetti.",
    )
    return ap.parse_args()


def _resolve_method_paths(
    series_path: Path,
    tinylora_series_path: Optional[Path],
    explicit_methods: Optional[List[str]],
) -> List[Tuple[str, Path]]:
    if explicit_methods:
        out: List[Tuple[str, Path]] = []
        for name in explicit_methods:
            cp = series_path / name
            if cp.exists():
                out.append((name, cp))
                continue
            if tinylora_series_path and (tinylora_series_path / name).exists():
                out.append((name, tinylora_series_path / name))
                continue
            print(f"[warn] method {name} not found under {series_path} or "
                  f"{tinylora_series_path}; skipping", file=sys.stderr)
        return out
    seen: Dict[str, Path] = {}
    for n in autodiscover_methods(series_path):
        seen.setdefault(n, series_path / n)
    if tinylora_series_path is not None:
        for n in autodiscover_methods(tinylora_series_path):
            seen.setdefault(n, tinylora_series_path / n)
    return sorted(seen.items(), key=lambda kv: kv[0])


def main() -> int:
    args = _parse_args()
    if args.tinylora_series_path is not None:
        s = str(args.tinylora_series_path).strip()
        if s in ("", ".", "none", "null", "None") or not args.tinylora_series_path.exists():
            args.tinylora_series_path = None

    print("=== aggregate_loss_history ===")
    print(f"Series:           {args.series_path}")
    print(f"TinyLoRA series:  {args.tinylora_series_path}")
    print(f"Baseline:         {args.baseline_method}")
    print(f"Output:           {args.output_dir}")
    print(f"ΔPSNR threshold:  ±{args.psnr_threshold:g} dB")
    print()

    method_specs = _resolve_method_paths(
        args.series_path, args.tinylora_series_path, args.methods,
    )
    if not method_specs:
        print("[error] no methods discovered; abort.", file=sys.stderr)
        return 2
    print("Discovered methods:")
    for n, p in method_specs:
        n_chunks = len(list(p.glob("chunk_*/summary.json")))
        print(f"  {n:30s}  chunks={n_chunks:>2d}  ({p})")

    pv_by_method: Dict[str, Dict[str, dict]] = {}
    for n, p in method_specs:
        pv_by_method[n] = load_method_per_video(p)
        print(f"  loaded {n:30s}  per-video records: {len(pv_by_method[n])}")
    print()

    if args.baseline_method not in pv_by_method:
        print(f"[error] baseline {args.baseline_method!r} not in discovered "
              f"methods; abort.", file=sys.stderr)
        return 2

    baseline_pv = pv_by_method[args.baseline_method]

    per_method_rows: Dict[str, List[dict]] = {}
    methods_no_history: List[str] = []
    total_seen = 0

    for m, pv in pv_by_method.items():
        if m == args.baseline_method:
            continue
        rows: List[dict] = []
        any_history = False
        for vid, rec in pv.items():
            total_seen += 1
            es = rec.get("early_stopping_info")
            history = _coerce_loss_history(es)
            if not history:
                continue
            any_history = True

            losses = [l for _, l in history]
            initial_loss = losses[0] if losses else None
            best_loss = float(min(losses)) if losses else None
            final_loss = _coerce_float(rec.get("final_loss"))
            loss_decrease_pct: Optional[float] = None
            if initial_loss is not None and best_loss is not None and initial_loss > 0:
                loss_decrease_pct = (initial_loss - best_loss) / initial_loss

            base_rec = baseline_pv.get(vid, {})
            base_psnr = _coerce_float(base_rec.get("psnr", base_rec.get("avg_psnr")))
            m_psnr    = _coerce_float(rec.get("psnr", rec.get("avg_psnr")))
            dpsnr: Optional[float]
            if base_psnr is None or m_psnr is None:
                dpsnr = None
                band = ""
            else:
                dpsnr = m_psnr - base_psnr
                if dpsnr > args.psnr_threshold:
                    band = "winners"
                elif dpsnr < -args.psnr_threshold:
                    band = "losers"
                else:
                    band = "middle"

            stopped = False
            best_step = None
            if isinstance(es, dict):
                stopped = bool(es.get("stopped_early", False))
                if es.get("best_step") is not None:
                    try:
                        best_step = int(es["best_step"])
                    except (TypeError, ValueError):
                        best_step = None

            rows.append({
                "video_id": vid,
                "n_checks": len(history),
                "loss_history": history,
                "initial_loss": initial_loss,
                "best_loss":    best_loss,
                "final_loss":   final_loss,
                "loss_decrease_pct": loss_decrease_pct,
                "stopped_early": stopped,
                "best_step": best_step,
                "baseline_psnr": base_psnr,
                "method_psnr":   m_psnr,
                "dpsnr":         dpsnr if dpsnr is not None else float("nan"),
                "winner_band":   band,
            })

        if not any_history:
            methods_no_history.append(m)
        per_method_rows[m] = rows
        print(f"  [{m}] {len(rows)} videos with anchor-loss history "
              f"out of {len(pv)} per-video records")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSVs --------------------------------------------------------------
    curves_csv = args.output_dir / "per_video_loss_curves.csv"
    write_curves_csv(curves_csv, per_method_rows)
    print(f"\nWrote {curves_csv}")

    summary_csv = args.output_dir / "per_video_loss_summary.csv"
    write_summary_csv(summary_csv, per_method_rows)
    print(f"Wrote {summary_csv}")

    # ---- plots -------------------------------------------------------------
    plt = _setup_matplotlib()
    for m, rows in per_method_rows.items():
        if not rows:
            continue
        out_p = args.output_dir / f"loss_curves_{m}.png"
        if plot_loss_curves(
            plt, out_p, m, rows, args.psnr_threshold,
            max_curves_per_band=args.max_curves_per_band,
        ):
            print(f"Wrote {out_p}")

    scatter_path = args.output_dir / "loss_decrease_vs_dpsnr.png"
    if plot_loss_decrease_vs_dpsnr(plt, scatter_path, per_method_rows):
        print(f"Wrote {scatter_path}")

    # ---- summary.md --------------------------------------------------------
    md_path = args.output_dir / "summary.md"
    write_summary_md(
        md_path, args, args.baseline_method, per_method_rows,
        methods_no_history, total_seen,
    )
    print(f"Wrote {md_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
