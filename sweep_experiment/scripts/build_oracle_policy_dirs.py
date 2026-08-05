#!/usr/bin/env python3
"""Build per-policy video directories for Phase-1 oracle upper-bound FVD.

Each policy composes one saved generated clip per eval video (symlinked as
``panda_XXXX.mp4``) by reading per-video metrics from ``per_video_gains.csv``
and locating the corresponding mp4 under ``chunk_*/videos/`` in each method
run.  **Never** falls back to GT source videos — missing outputs are skipped
with a loud error so FVD cannot silently become 0 (gen == ref).

Policies (default set):
  always_notta          — always use NOTTA output
  always_ada            — always use ADA output
  always_lora           — always use LORA_R8_TTA output
  oracle_best_psnr      — per-video argmax PSNR among NOTTA / ADA / LORA
  oracle_skip_ada_nonpos — ADA if ΔPSNR>0 else NOTTA
  oracle_skip_both_nonpos — best PSNR among {ADA,LORA} with ΔPSNR>0 else NOTTA
  oracle_top50_ada_dpsnr — ADA on top-50% ADA ΔPSNR videos, else NOTTA

Usage:
    python sweep_experiment/scripts/build_oracle_policy_dirs.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv \\
        --series-root sweep_experiment/results/panda_1000v_standard \\
        --output-root sweep_experiment/reports/phase1_oracle_fvd
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.caption_utils import canonical_video_id

_METHOD_DIRS = {
    "NOTTA": "NOTTA",
    "ADA": "ADA",
    "LORA_R8_TTA": "LORA_R8_TTA",
}

_METHOD_SUFFIX = {
    "LORA_R8_TTA": "_lora",
}


def _numeric_id(video_name: str) -> Optional[int]:
    """``panda_0867`` -> 867 (matches post-rename ``867_*_<method>.mp4``)."""
    m = re.search(r"(\d+)$", video_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _load_chunk_summary_order(summary: dict) -> Dict[str, int]:
    """Return {video_name: results-array index} from a chunk summary.json."""
    out: Dict[str, int] = {}
    pv = summary.get("per_video_results") or summary.get("results") or []
    for i, r in enumerate(pv):
        v = r.get("video_name") or r.get("video_id")
        if v:
            out[v] = i
    return out


def find_mp4(
    videos_dir: Path,
    video_name: str,
    idx_by_name: Dict[str, int],
    *,
    prefer_suffix: Optional[str] = None,
) -> Optional[Path]:
    """Locate generated mp4 for ``video_name`` (handles post-rename outputs).

    Mirrors ``scripts/build_cover_image_filmstrips.py`` — NOTTA/AdaSteer
    runs rename outputs to ``<idx>_<caption-slug>_..._<method>.mp4`` while
    LoRA keeps ``<video_name>_lora.mp4``.

    When ``prefer_suffix`` is set (e.g. ``"_lora"`` for LORA_R8_TTA), that
    variant is tried before the bare ``<video_name>.mp4`` so a stray GT copy
    named ``panda_XXXX.mp4`` cannot shadow ``panda_XXXX_lora.mp4``.
    """
    if not videos_dir.is_dir():
        return None
    if prefer_suffix:
        pref = videos_dir / f"{video_name}{prefer_suffix}.mp4"
        if pref.exists():
            return pref
    direct = videos_dir / f"{video_name}.mp4"
    if direct.exists():
        return direct
    pre = sorted(videos_dir.glob(f"{video_name}*.mp4"))
    if pre:
        return pre[0]
    nid = _numeric_id(video_name)
    if nid is not None:
        num_glob = sorted(videos_dir.glob(f"{nid}_*.mp4"))
        if num_glob:
            return num_glob[0]
    idx = idx_by_name.get(video_name)
    if idx is not None:
        post = sorted(videos_dir.glob(f"{idx}_*.mp4"))
        if post:
            return post[0]
    sub = list(videos_dir.glob(f"*{video_name}*.mp4"))
    if len(sub) == 1:
        return sub[0]
    return None


def _is_under_dir(path: Path, parent: Optional[Path]) -> bool:
    if parent is None:
        return False
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def index_method_videos(
    series_root: Path,
    method: str,
    *,
    ref_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Map canonical ``panda_XXXX`` -> absolute path to saved generated mp4."""
    run_dir = series_root / method
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Method dir not found: {run_dir}")

    ref_resolved = ref_dir.resolve() if ref_dir else None
    datasets_root = (_REPO_ROOT / "datasets").resolve()
    out: Dict[str, Path] = {}

    chunk_dirs = sorted(run_dir.glob("chunk_*/"))
    if not chunk_dirs:
        chunk_dirs = [run_dir]

    for chunk_dir in chunk_dirs:
        videos_dir = chunk_dir / "videos"
        summary_path = chunk_dir / "summary.json"
        if not summary_path.exists() and chunk_dir == run_dir:
            summary_path = run_dir / "merged_summary.json"
        if not summary_path.exists():
            continue

        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        idx_by_name = _load_chunk_summary_order(summary)

        for rec in summary.get("results", []):
            if not rec.get("success", False):
                continue
            vname = rec.get("video_name", "")
            vid = canonical_video_id(vname)
            if not vid:
                continue

            candidates: List[Path] = []
            # Prefer the exact per-record output_path first: it is authoritative
            # and 1:1, avoiding find_mp4's fuzzy-glob collisions (e.g. the EXP2
            # placement arms, where {nid}_*/{idx}_* globs collapse many records
            # onto the same file). Guarded by .exists(), so runs that renamed
            # their outputs post-summary (1000v NOTTA/AdaSteer) fall back to
            # find_mp4 exactly as before — no regression.
            op = rec.get("output_path")
            if op and videos_dir.is_dir():
                p = Path(op).resolve()
                if p.exists() and p.suffix.lower() == ".mp4" and _is_under_dir(p, videos_dir):
                    candidates.append(p)

            mp4 = find_mp4(
                videos_dir, vname, idx_by_name,
                prefer_suffix=_METHOD_SUFFIX.get(method),
            )
            if mp4 is not None:
                candidates.append(mp4.resolve())

            chosen: Optional[Path] = None
            for p in candidates:
                if _is_under_dir(p, ref_resolved):
                    continue
                if _is_under_dir(p, datasets_root):
                    continue
                chosen = p
                break

            if chosen is not None:
                out[vid] = chosen

    return out


def _load_gains(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _float_or_none(val) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pick_always(method: str) -> Callable[[dict], str]:
    def _fn(row: dict) -> str:
        return method
    return _fn


def _pick_oracle_best_psnr(row: dict) -> str:
    cands = []
    for m in ("NOTTA", "ADA", "LORA_R8_TTA"):
        psnr = _float_or_none(row.get(f"{m}_psnr"))
        if psnr is not None:
            cands.append((psnr, m))
    if not cands:
        return "NOTTA"
    cands.sort(reverse=True)
    return cands[0][1]


def _pick_skip_ada_nonpos(row: dict) -> str:
    dpsnr = _float_or_none(row.get("ADA_dpsnr"))
    return "ADA" if dpsnr is not None and dpsnr > 0 else "NOTTA"


def _pick_skip_both_nonpos(row: dict) -> str:
    pos = []
    for m in ("ADA", "LORA_R8_TTA"):
        dpsnr = _float_or_none(row.get(f"{m}_dpsnr"))
        psnr = _float_or_none(row.get(f"{m}_psnr"))
        if dpsnr is not None and dpsnr > 0 and psnr is not None:
            pos.append((psnr, m))
    if not pos:
        return "NOTTA"
    pos.sort(reverse=True)
    return pos[0][1]


def _make_top50_ada_picker(rows: List[dict]) -> Callable[[dict], str]:
    scored: List[Tuple[float, str]] = []
    for row in rows:
        vid = row.get("video_id", "")
        dpsnr = _float_or_none(row.get("ADA_dpsnr"))
        if vid and dpsnr is not None:
            scored.append((dpsnr, vid))
    scored.sort(reverse=True)
    n = len(scored)
    cutoff = scored[n // 2][0] if n else float("inf")
    top_vids = {vid for val, vid in scored if val >= cutoff}

    def _fn(row: dict) -> str:
        return "ADA" if row.get("video_id") in top_vids else "NOTTA"
    return _fn


POLICY_FNS: Dict[str, Callable[[dict], str]] = {
    "always_notta": _pick_always("NOTTA"),
    "always_ada": _pick_always("ADA"),
    "always_lora": _pick_always("LORA_R8_TTA"),
    "oracle_best_psnr": _pick_oracle_best_psnr,
    "oracle_skip_ada_nonpos": _pick_skip_ada_nonpos,
    "oracle_skip_both_nonpos": _pick_skip_both_nonpos,
}


def build_policy_dir(
    policy_name: str,
    picker: Callable[[dict], str],
    rows: List[dict],
    video_index: Dict[str, Dict[str, Path]],
    output_root: Path,
    *,
    ref_dir: Optional[Path] = None,
    clean: bool,
) -> Tuple[Path, int, List[str]]:
    out_dir = output_root / policy_name / "videos"
    if clean and out_dir.exists():
        for p in out_dir.glob("*.mp4"):
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    missing: List[str] = []
    ref_resolved = ref_dir.resolve() if ref_dir else None

    for row in rows:
        vid = row.get("video_id", "")
        if not vid:
            continue
        method = picker(row)
        src = video_index.get(method, {}).get(vid)
        if src is None:
            missing.append(f"{vid}->{method}")
            continue
        if ref_resolved is not None:
            try:
                src.relative_to(ref_resolved)
                missing.append(f"{vid}->GT_COLLISION({src.name})")
                continue
            except ValueError:
                pass
        dst = out_dir / f"{vid}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        linked += 1

    manifest = {
        "policy": policy_name,
        "linked_videos": linked,
        "missing": len(missing),
        "videos_dir": str(out_dir),
    }
    with (output_root / policy_name / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir, linked, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Build oracle policy video dirs")
    ap.add_argument(
        "--gains-csv",
        type=Path,
        default=Path(
            "sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv"
        ),
    )
    ap.add_argument(
        "--series-root",
        type=Path,
        default=Path("sweep_experiment/results/panda_1000v_standard"),
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("sweep_experiment/reports/phase1_oracle_fvd"),
    )
    ap.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="GT source dir (used only to reject gen==ref symlinks)",
    )
    ap.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help="Subset of policy names (default: all)",
    )
    ap.add_argument("--clean", action="store_true", help="Remove old symlinks first")
    ap.add_argument(
        "--min-linked", type=int, default=900,
        help="Fail if any policy links fewer than this many videos (default 900)",
    )
    args = ap.parse_args()

    if not args.gains_csv.exists():
        print(f"ERROR: gains CSV not found: {args.gains_csv}", file=sys.stderr)
        return 2

    rows = _load_gains(args.gains_csv)
    if not rows:
        print("ERROR: empty gains CSV", file=sys.stderr)
        return 2

    ref_dir = args.ref_dir
    if ref_dir is None:
        ref_dir = Path("datasets/panda_1000_480p/videos")

    video_index: Dict[str, Dict[str, Path]] = {}
    for method in _METHOD_DIRS:
        video_index[method] = index_method_videos(
            args.series_root, method, ref_dir=ref_dir,
        )
        print(f"  indexed {method}: {len(video_index[method])} videos")

    policy_names = args.policies or list(POLICY_FNS) + ["oracle_top50_ada_dpsnr"]

    exit_code = 0
    for pname in policy_names:
        if pname == "oracle_top50_ada_dpsnr":
            picker = _make_top50_ada_picker(rows)
        elif pname in POLICY_FNS:
            picker = POLICY_FNS[pname]
        else:
            print(f"WARNING: unknown policy {pname!r}, skipping", file=sys.stderr)
            continue

        out_dir, linked, missing = build_policy_dir(
            pname, picker, rows, video_index, args.output_root,
            ref_dir=ref_dir, clean=args.clean,
        )
        print(f"{pname}: linked {linked} videos -> {out_dir}")
        if missing:
            exit_code = 1
            print(f"  MISSING {len(missing)} (first 10): {missing[:10]}", file=sys.stderr)
        if linked < args.min_linked:
            exit_code = 1
            print(
                f"  ERROR: {pname} linked {linked} < --min-linked={args.min_linked}",
                file=sys.stderr,
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
