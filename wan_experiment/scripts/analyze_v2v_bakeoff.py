#!/usr/bin/env python3
"""Paired V2V bake-off: tail motion + last-chunk scores + optional VBench.

Promote a method past N=8 only if median tail motion beats notta AND
imaging / subject do not collapse (when VBench joined.json is present).

    python wan_experiment/scripts/analyze_v2v_bakeoff.py \
        --series-dir wan_experiment/results/v2v_panda_bakeoff_8v
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METHODS = (
    "notta", "seed_bon", "motion_bon", "shift_search", "backtrack",
    "hinge_bon", "late_bon", "hist_drop", "good_backtrack",
    "cached_bon", "sink", "quiet_bon", "tail_hist",
    "live_bon", "live_hist",
    "longlive_notta", "longlive_sink", "longlive_live_bon",
    "longlive_prefix_sink", "rolling_notta",
    "rolling_rho_lo", "rolling_rho_hi", "rolling_adapt", "rolling_look",
    "sf_roll", "rf_chunk", "sf_recache", "rf_recache",
    "rf_rewind", "rf_sick_search", "rf_pseudo", "rf_sink",
    "sf_rewind", "sf_sick_search", "sf_pseudo", "sf_always_search",
    "rf_always_search", "sf_sink",
    "sf_intra", "sf_intra_always", "rf_intra", "rf_intra_always",
    "sf_lastmix", "sf_lastmix_always", "sf_bpseudo", "sf_bpseudo_always",
    "sf_restep", "sf_restep_always",
    "rf_lastmix", "rf_lastmix_always", "rf_bpseudo",
    "rf_restep", "rf_restep_always",
    "appear_bon", "live_appear", "pseudo_gate", "pseudo_appear",
    "noise_probe", "noise_bon",
    "ada_fixed", "ada_stream", "ada_resid",
)
VBENCH_DIMS = (
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "dynamic_degree",
    "temporal_flickering",
)


def _usable(rec: dict) -> bool:
    return bool(
        rec.get("ok")
        and not rec.get("skipped")
        and rec.get("tail_motion") is not None
    )


def _load_rows(series_dir: Path, method: str, allow_partial: bool = False) -> list[dict]:
    """Sidecars first. summary.json skip-stubs have ok=True and no tail."""
    rows = []
    seen: set[str] = set()
    dirs = sorted(series_dir.glob(f"{method}_h*s_shard*"))
    for d in dirs:
        for p in sorted(d.glob("*.json")):
            if p.name in {"summary.json", "joined.json"} or "vbench" in p.name:
                continue
            try:
                rec = json.loads(p.read_text())
            except Exception:
                continue
            if not _usable(rec):
                continue
            key = rec.get("file_name") or rec.get("stem") or p.stem
            if key in seen:
                continue
            seen.add(str(key))
            rows.append(rec)
        summary = d / "summary.json"
        if not summary.is_file():
            continue
        try:
            data = json.loads(summary.read_text())
        except Exception:
            continue
        for rec in data.get("rows") or []:
            if not _usable(rec):
                continue
            key = rec.get("file_name") or rec.get("stem") or rec.get("mp4")
            if not key or str(key) in seen:
                continue
            seen.add(str(key))
            rows.append(rec)
    if rows or allow_partial:
        return rows
    return rows


def _key(row: dict) -> str:
    return row.get("file_name") or row.get("stem") or row.get("mp4")


def _median(xs: list[float] | None):
    xs = [x for x in (xs or []) if x is not None]
    if not xs:
        return None
    return statistics.median(xs)


def _load_vbench(method_dir: Path, clip: str = "full") -> dict[str, float] | None:
    for name in (
        f"vbench_{clip}/joined.json",
        f"vbench_{clip}/summary.json",
        "vbench_full/joined.json",
    ):
        p = method_dir / name
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        out = {}
        pop = data.get("population") or {}
        src = data.get("medians") or data.get("metrics") or {}
        for dim in VBENCH_DIMS:
            cell = pop.get(dim)
            if isinstance(cell, dict) and cell.get("median") is not None:
                out[dim] = float(cell["median"])
            elif dim in src and src[dim] is not None:
                if isinstance(src[dim], dict) and src[dim].get("median") is not None:
                    out[dim] = float(src[dim]["median"])
                else:
                    out[dim] = float(src[dim])
            elif dim in data and data[dim] is not None and not isinstance(data[dim], dict):
                out[dim] = float(data[dim])
        if out:
            return out
    return None


def analyze(
    series_dir: Path,
    clip: str = "full",
    baseline_dir: Path | None = None,
    allow_partial: bool = False,
) -> dict:
    by = {}
    for m in METHODS:
        rows = _load_rows(series_dir, m, allow_partial=allow_partial)
        if rows:
            by[m] = {_key(r): r for r in rows}
    if "notta" not in by and baseline_dir is not None:
        rows = _load_rows(baseline_dir, "notta", allow_partial=allow_partial)
        if rows:
            by["notta"] = {_key(r): r for r in rows}
    if "notta" not in by:
        raise FileNotFoundError(
            f"no notta rows under {series_dir}"
            + (f" or {baseline_dir}" if baseline_dir else "")
        )
    keys = sorted(by["notta"])
    for m, mapping in by.items():
        keys = [k for k in keys if k in mapping]

    method_stats = {}
    for m, mapping in by.items():
        tail = [mapping[k].get("tail_motion") for k in keys]
        last = [mapping[k].get("last_chunk_score") for k in keys]
        last_m = [mapping[k].get("last_chunk_motion_score") for k in keys]
        dirs = sorted(series_dir.glob(f"{m}_h*s_shard*"))
        if m == "notta" and not dirs and baseline_dir is not None:
            dirs = sorted(baseline_dir.glob(f"{m}_h*s_shard*"))
        vb = _load_vbench(dirs[0], clip) if dirs else None
        method_stats[m] = {
            "n": len(keys),
            "n_on_disk": len(mapping),
            "tail_motion_median": _median(tail),
            "last_chunk_score_median": _median(last),
            "last_chunk_motion_median": _median(last_m),
            "vbench": vb,
        }

    notta_mot = method_stats["notta"]["tail_motion_median"]
    notta_vb = method_stats["notta"].get("vbench") or {}
    decisions = {}
    for m, st in method_stats.items():
        if m == "notta":
            decisions[m] = "baseline"
            continue
        mot = st["tail_motion_median"]
        motion_win = (
            mot is not None and notta_mot is not None and mot > notta_mot
        )
        iq_ok = True
        subj_ok = True
        vb = st.get("vbench") or {}
        if vb and notta_vb:
            if "imaging_quality" in vb and "imaging_quality" in notta_vb:
                iq_ok = vb["imaging_quality"] >= notta_vb["imaging_quality"] - 1.0
            if "subject_consistency" in vb and "subject_consistency" in notta_vb:
                subj_ok = vb["subject_consistency"] >= notta_vb["subject_consistency"] - 0.02
        if motion_win and iq_ok and subj_ok:
            decisions[m] = "PROMOTE"
        elif motion_win and not (iq_ok and subj_ok):
            decisions[m] = "FAIL (motion win, quality collapse)"
        elif mot is None:
            decisions[m] = "PENDING generate"
        else:
            decisions[m] = "HOLD (no motion gain)"

    lines = [
        "# V2V sampling-space bake-off",
        "",
        f"Series: `{series_dir}`  paired N={len(keys)}  cite medians."
        + ("  **PARTIAL** (running job / per-video json)." if allow_partial else ""),
        "",
        "| Method | tail motion | last-chunk drift ↓ | last-chunk motion ↑ | decision |",
        "|---|---:|---:|---:|---|",
    ]
    for m in METHODS:
        if m not in method_stats:
            continue
        st = method_stats[m]
        def fmt(x, nd=4):
            return "—" if x is None else f"{x:.{nd}f}"
        lines.append(
            f"| {m} | {fmt(st['tail_motion_median'])} | "
            f"{fmt(st['last_chunk_score_median'])} | "
            f"{fmt(st['last_chunk_motion_median'])} | {decisions.get(m, '—')} |"
        )
    if any(st.get("vbench") for st in method_stats.values()):
        lines += [
            "",
            "## Full-clip VBench (when scored)",
            "",
            "| Method | " + " | ".join(VBENCH_DIMS) + " |",
            "|---|" + "|".join(["---:" ] * len(VBENCH_DIMS)) + "|",
        ]
        for m in METHODS:
            st = method_stats.get(m)
            if not st or not st.get("vbench"):
                continue
            vb = st["vbench"]
            cells = []
            for d in VBENCH_DIMS:
                v = vb.get(d)
                cells.append("—" if v is None else f"{v:.4f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    else:
        lines += [
            "",
            "VBench not scored yet. After generate:",
            "",
            "```",
            "python wan_experiment/scripts/score_i2v_vbench.py --clip full \\",
            f"  --video-dir {series_dir}/<method>_h30s_shard0",
            "```",
        ]
    lines += [
        "",
        "## Decision rule",
        "",
        "Promote past N=8 only if median tail motion beats `notta` **and**",
        "imaging quality is not worse by ≥1.0 and subject consistency is not",
        "worse by ≥0.02. A freeze-looking smoothness win does not count.",
        "",
    ]
    return {
        "n": len(keys),
        "methods": method_stats,
        "decisions": decisions,
        "markdown": "\n".join(lines) + "\n",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", required=True, type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="optional series that has notta (e.g. bakeoff_8v)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="read per-video json if summary.json is not written yet")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--clip", default="full")
    args = ap.parse_args()
    result = analyze(
        args.series_dir, clip=args.clip, baseline_dir=args.baseline_dir,
        allow_partial=args.allow_partial,
    )
    print(result["markdown"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
