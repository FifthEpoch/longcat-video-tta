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


_METHOD_SUFFIXES = (
    "_lora.mp4",
    "_full.mp4",
    "_no-TTA.mp4",
    "_adasteer.mp4",
    "_delta_a.mp4",
    "_tinylora.mp4",
)


def find_mp4(
    videos_dir: Path,
    video_name: str,
    idx_by_name: Dict[str, int],
) -> Optional[Path]:
    """Locate generated mp4 for ``video_name`` (handles post-rename outputs).

    Chunk ``videos/`` dirs often contain bare ``panda_XXXX.mp4`` GT source
    copies alongside generated outputs.  Always prefer method-specific
    suffixes (``_full.mp4``, ``_delta_a.mp4``, ``_lora.mp4``) and post-rename
    ``<idx>_*.mp4`` files before the bare GT name — otherwise offline FVD
    scores the wrong temporal region and inflates ~60 pts vs headline.
    """
    if not videos_dir.is_dir():
        return None

    for suffix in _METHOD_SUFFIXES:
        p = videos_dir / f"{video_name}{suffix}"
        if p.exists():
            return p

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

    pre = sorted(
        p for p in videos_dir.glob(f"{video_name}*.mp4")
        if p.name != f"{video_name}.mp4"
    )
    if pre:
        return pre[0]

    # Never fall back to bare ``panda_XXXX.mp4`` in chunk dirs — those are
    # almost always GT source copies and inflate offline FVD ~60 pts.
    return None


def _mp4_readable(path: Path, *, min_frames: int = 28) -> bool:
    """Return True if PyAV can open and decode at least *min_frames* frames.

    Oracle FVD scores frames [14:28] (14 cond + 14 gen).  Truncated outputs
    (``moov atom not found``) fail here so eval_fvd does not silently drop
    pairs and shrink ``num_valid_pairs``.
    """
    try:
        import av
    except ImportError:
        return True  # defer to eval_fvd when av unavailable at build time

    try:
        container = av.open(str(path))
    except Exception:
        return False

    n = 0
    try:
        for _ in container.decode(video=0):
            n += 1
            if n >= min_frames:
                break
    except Exception:
        return False
    finally:
        container.close()
    return n >= min_frames


def _is_under_dir(path: Path, parent: Optional[Path]) -> bool:
    if parent is None:
        return False
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


_FINGERPRINT_RE = re.compile(
    r"_PSNR-(-?\d+\.\d{3})_SSIM-(-?\d+\.\d{3})_LPIPS-(-?\d+\.\d{3})_"
)


def _build_metric_fingerprint_index(
    videos_dir: Path,
) -> Dict[Tuple[str, str, str], Optional[Path]]:
    """Map ``(psnr, ssim, lpips)`` 3-decimal fingerprint -> mp4.

    ``rename_videos.py`` rewrites saved clips to
    ``<idx>_<slug>_..._PSNR-x.xxx_SSIM-x.xxx_LPIPS-x.xxx_..._<method>.mp4`` where
    ``<idx>`` is the trailing digits of ``video_name``. For ``ytid_segN`` pools
    (e.g. the EXP2 placement arms) that trailing number is the *seg index*, which
    is NOT unique, so glob-by-index collapses many records onto one file. The
    embedded metric fingerprint IS a unique per-video key, so we resolve by it.
    Colliding fingerprints map to ``None`` so we never mis-resolve.
    """
    idx: Dict[Tuple[str, str, str], Optional[Path]] = {}
    if not videos_dir.is_dir():
        return idx
    for p in videos_dir.glob("*.mp4"):
        m = _FINGERPRINT_RE.search(p.name)
        if not m:
            continue
        key = (m.group(1), m.group(2), m.group(3))
        idx[key] = None if key in idx else p
    return idx


def _record_fingerprint_key(rec: dict) -> Optional[Tuple[str, str, str]]:
    """Build the ``(psnr, ssim, lpips)`` 3-decimal key from a summary record."""
    psnr, ssim, lpips_ = rec.get("psnr"), rec.get("ssim"), rec.get("lpips")
    if psnr is None or ssim is None or lpips_ is None:
        return None
    try:
        return (f"{float(psnr):.3f}", f"{float(ssim):.3f}", f"{float(lpips_):.3f}")
    except (TypeError, ValueError):
        return None


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
        fp_index = _build_metric_fingerprint_index(videos_dir)

        for rec in summary.get("results", []):
            if not rec.get("success", False):
                continue
            vname = rec.get("video_name", "")
            vid = canonical_video_id(vname)
            if not vid:
                continue

            candidates: List[Path] = []
            # 1) Exact per-record output_path (authoritative, 1:1). Guarded by
            #    .exists(): runs that renamed their outputs post-summary (all the
            #    metric-fingerprint renames) fall through to (2)/(3).
            op = rec.get("output_path")
            if op and videos_dir.is_dir():
                p = Path(op).resolve()
                if p.exists() and p.suffix.lower() == ".mp4" and _is_under_dir(p, videos_dir):
                    candidates.append(p)

            # 2) Metric-fingerprint match for rename_videos.py outputs. Robust
            #    where find_mp4's {idx}_* glob collides — e.g. ytid_segN pools
            #    (EXP2 placement arms) whose trailing "index" is a non-unique seg
            #    number. The (psnr,ssim,lpips) triple is a unique per-video key.
            fp_key = _record_fingerprint_key(rec)
            if fp_key is not None:
                fp_hit = fp_index.get(fp_key)
                if fp_hit is not None:
                    candidates.append(fp_hit.resolve())

            # 3) Legacy glob heuristics (bare panda_XXXX naming, etc.).
            mp4 = find_mp4(videos_dir, vname, idx_by_name)
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
        if not _mp4_readable(src):
            missing.append(f"{vid}->UNREADABLE({src.name})")
            print(
                f"  SKIP unreadable mp4: {vid} <- {src}",
                file=sys.stderr,
            )
            continue
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
