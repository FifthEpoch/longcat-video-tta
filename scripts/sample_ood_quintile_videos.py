#!/usr/bin/env python3
"""Sample a stratified OOD-quintile pilot eval set for the AdaSteer budget grid (H9).

Reads ``diffusion_ood_scores.csv`` (from ``compute_diffusion_ood_score.py``),
assigns each video to an OOD quintile by ``mean_diffusion_loss_caption``, and
draws ``--per-quintile`` videos per bin (default 40 → 200 total).  Optionally
materialises a symlinked pilot dataset compatible with ``run_delta_a`` /
``run_sweep.sbatch``.

Outputs:
  * ``pilot_videos.json`` — retain-list shape with quintile metadata
  * ``pilot_videos.txt``    — one canonical ``video_id`` per line
  * (optional) filtered ``metadata.csv`` + ``videos/`` symlinks under
    ``--dataset-dir``

Usage:
    python scripts/sample_ood_quintile_videos.py \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --source-dataset datasets/panda_1000_480p \\
        --output-json sweep_experiment/lists/panda_ood_budget_pilot_videos.json \\
        --create-dataset datasets/panda_ood_budget_pilot_480p
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.caption_utils import canonical_video_id

DEFAULT_OOD = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv"
)
DEFAULT_SOURCE = _REPO_ROOT / "datasets/panda_1000_480p"
DEFAULT_JSON = _REPO_ROOT / "sweep_experiment/lists/panda_ood_budget_pilot_videos.json"

OOD_COL = "mean_diffusion_loss_caption"


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return float("nan")
    return float(v)


def load_ood_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: List[dict] = []
    for r in rows:
        vid = canonical_video_id(r.get("video_id", ""))
        ood = _f(r, OOD_COL)
        if not vid or np.isnan(ood):
            continue
        out.append({"video_id": vid, OOD_COL: ood, **r})
    return out


def _available_stems(source_dataset: Path) -> set:
    """Set of on-disk video filename stems (what ``_resolve_video_path`` matches).

    A row's (already-canonical) ``video_id`` is resolvable iff a file named
    ``{video_id}.mp4`` exists. Segment-pool ids whose YouTube portion contains
    ``_<digit>`` get truncated by ``canonical_video_id`` (e.g. ``ETcLgl5_8xY_3``
    -> ``ETcLgl5_8``), so they will NOT appear here and are dropped — which is
    correct: those ids collide under canonicalization and would cross-contaminate
    the downstream feature/metric joins that key on the same canonical id.
    """
    stems: set = set()
    for d in (source_dataset / "videos", source_dataset):
        if d.is_dir():
            for p in d.glob("*.mp4"):
                stems.add(p.stem)
    return stems


def clean_rows(
    rows: List[dict],
    *,
    available_stems: Optional[set],
) -> Tuple[List[dict], int, int]:
    """Dedup by canonical id and (if known) keep only on-disk-resolvable ids.

    Returns (clean_rows, n_dropped_missing, n_dropped_duplicate).
    """
    seen: set = set()
    clean: List[dict] = []
    n_missing = 0
    n_dup = 0
    for r in rows:
        vid = r["video_id"]
        if available_stems is not None and vid not in available_stems:
            n_missing += 1
            continue
        if vid in seen:
            n_dup += 1
            continue
        seen.add(vid)
        clean.append(r)
    return clean, n_missing, n_dup


def quintile_assign(
    values: Sequence[float],
    n_bins: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_index per row, bin_edges). Low OOD = Q1."""
    arr = np.asarray(values, dtype=float)
    edges = np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(len(arr), dtype=int), edges
    # digitize: 0 .. n_bins-1
    idx = np.digitize(arr, edges[1:-1], right=False)
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx, edges


def sample_per_quintile(
    rows: List[dict],
    *,
    per_quintile: int,
    n_bins: int,
    seed: int,
) -> Tuple[List[dict], np.ndarray]:
    ood_vals = [r[OOD_COL] for r in rows]
    bin_idx, edges = quintile_assign(ood_vals, n_bins=n_bins)
    rng = random.Random(seed)
    selected: List[dict] = []
    by_bin: Dict[int, List[dict]] = {b: [] for b in range(n_bins)}
    for i, r in enumerate(rows):
        by_bin[int(bin_idx[i])].append(r)

    for b in range(n_bins):
        pool = by_bin[b]
        if not pool:
            continue
        pool = sorted(pool, key=lambda x: x["video_id"])
        k = min(per_quintile, len(pool))
        picks = rng.sample(pool, k) if len(pool) > k else pool
        for p in picks:
            selected.append(
                {
                    **p,
                    "ood_quintile": b + 1,
                    "ood_quintile_label": f"Q{b + 1}",
                }
            )
    selected.sort(key=lambda x: (x["ood_quintile"], x["video_id"]))
    return selected, edges


def _load_metadata_by_stem(meta_path: Path) -> Dict[str, dict]:
    by_stem: Dict[str, dict] = {}
    with meta_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            fname = row.get("filename", row.get("video_path", ""))
            stem = Path(fname).stem
            by_stem[stem] = row
            cid = canonical_video_id(stem)
            if cid:
                by_stem[cid] = row
    return by_stem


def _resolve_video_path(source_dataset: Path, video_id: str) -> Optional[Path]:
    for candidate in (
        source_dataset / "videos" / f"{video_id}.mp4",
        source_dataset / f"{video_id}.mp4",
    ):
        if candidate.exists():
            return candidate
    matches = sorted(source_dataset.rglob(f"{video_id}.mp4"))
    return matches[0] if matches else None


def create_pilot_dataset(
    source_dataset: Path,
    dataset_dir: Path,
    selected: Sequence[dict],
) -> None:
    meta_src = source_dataset / "metadata.csv"
    if not meta_src.exists():
        raise FileNotFoundError(f"metadata.csv not found: {meta_src}")

    meta_by_stem = _load_metadata_by_stem(meta_src)
    videos_dir = dataset_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rows_out: List[dict] = []
    missing_meta: List[str] = []
    missing_video: List[str] = []

    for entry in selected:
        vid = entry["video_id"]
        src_vp = _resolve_video_path(source_dataset, vid)
        if src_vp is None:
            missing_video.append(vid)
            continue
        dst_vp = videos_dir / src_vp.name
        if dst_vp.exists() or dst_vp.is_symlink():
            dst_vp.unlink()
        dst_vp.symlink_to(src_vp.resolve())

        meta_row = meta_by_stem.get(vid) or meta_by_stem.get(src_vp.stem)
        if meta_row is None:
            missing_meta.append(vid)
            continue
        rows_out.append(meta_row)

    # Selection pre-filters to resolvable ids, so misses here should be ~0.
    # Warn + skip rather than nuke a multi-minute dataset build; the downstream
    # dataset guard enforces the final video count.
    if missing_video:
        print(
            f"[warn] skipped {len(missing_video)} ids with no source video "
            f"(first: {missing_video[:5]})",
            file=sys.stderr,
        )
    if missing_meta:
        print(
            f"[warn] skipped {len(missing_meta)} ids with no metadata row "
            f"(first: {missing_meta[:5]})",
            file=sys.stderr,
        )
    if not rows_out:
        raise RuntimeError(
            "No videos materialized — source dataset ids do not match selection."
        )

    fieldnames = list(rows_out[0].keys())
    with (dataset_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    for aux in ("dynamic_degree.json",):
        src_aux = source_dataset / aux
        if src_aux.exists():
            dst_aux = dataset_dir / aux
            if dst_aux.exists() or dst_aux.is_symlink():
                dst_aux.unlink()
            dst_aux.symlink_to(src_aux.resolve())


def write_outputs(
    selected: Sequence[dict],
    *,
    output_json: Path,
    output_txt: Optional[Path],
    ood_csv: Path,
    source_dataset: Path,
    per_quintile: int,
    seed: int,
    edges: np.ndarray,
) -> None:
    all_ids = sorted({e["video_id"] for e in selected})
    quintile_counts: Dict[str, int] = {}
    for e in selected:
        lbl = e["ood_quintile_label"]
        quintile_counts[lbl] = quintile_counts.get(lbl, 0) + 1

    payload = {
        "purpose": "H9 AdaSteer budget-grid OOD-quintile pilot eval set",
        "ood_csv": str(ood_csv),
        "ood_column": OOD_COL,
        "source_dataset": str(source_dataset),
        "per_quintile": per_quintile,
        "seed": seed,
        "n_selected": len(selected),
        "quintile_counts": quintile_counts,
        "ood_bin_edges": [float(x) for x in edges.tolist()],
        "videos": [
            {
                "video_id": e["video_id"],
                "ood_quintile": e["ood_quintile"],
                "ood_quintile_label": e["ood_quintile_label"],
                OOD_COL: e[OOD_COL],
            }
            for e in selected
        ],
        "all": all_ids,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_json} ({len(all_ids)} videos)")

    if output_txt:
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text("\n".join(all_ids) + "\n", encoding="utf-8")
        print(f"Wrote {output_txt}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sample OOD-quintile-stratified pilot videos for H9 budget sweep",
    )
    ap.add_argument("--ood-csv", type=Path, default=DEFAULT_OOD)
    ap.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Optional one-id-per-line list (default: <json>.txt)",
    )
    ap.add_argument("--per-quintile", type=int, default=40)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--create-dataset",
        type=Path,
        default=None,
        help="Write symlink pilot dataset (metadata.csv + videos/) here",
    )
    args = ap.parse_args()

    if not args.ood_csv.exists():
        print(f"[error] OOD CSV not found: {args.ood_csv}", file=sys.stderr)
        return 2
    if not args.source_dataset.is_dir():
        print(f"[error] source dataset not found: {args.source_dataset}", file=sys.stderr)
        return 2

    rows = load_ood_rows(args.ood_csv)
    if not rows:
        print("[error] no valid OOD rows", file=sys.stderr)
        return 2

    # Dedup collisions + drop ids with no on-disk video so the sampled set is
    # exactly reproducible and materializable (segment-pool canonicalization
    # can truncate/collide YouTube-style ids — see clean_rows docstring).
    available = _available_stems(args.source_dataset)
    n_before = len(rows)
    rows, n_missing, n_dup = clean_rows(
        rows, available_stems=available if available else None,
    )
    print(
        f"[info] candidate rows: {n_before} -> {len(rows)} "
        f"(dropped {n_missing} unresolvable, {n_dup} duplicate canonical ids; "
        f"{len(available)} videos on disk)",
        file=sys.stderr,
    )
    if not rows:
        print("[error] no resolvable OOD rows after cleaning", file=sys.stderr)
        return 2

    selected, edges = sample_per_quintile(
        rows,
        per_quintile=args.per_quintile,
        n_bins=args.n_bins,
        seed=args.seed,
    )

    out_txt = args.output_txt
    if out_txt is None:
        out_txt = args.output_json.with_suffix(".txt")

    write_outputs(
        selected,
        output_json=args.output_json,
        output_txt=out_txt,
        ood_csv=args.ood_csv,
        source_dataset=args.source_dataset,
        per_quintile=args.per_quintile,
        seed=args.seed,
        edges=edges,
    )

    for lbl in sorted({e["ood_quintile_label"] for e in selected}):
        n = sum(1 for e in selected if e["ood_quintile_label"] == lbl)
        print(f"  {lbl}: {n} videos")

    if args.create_dataset:
        if args.create_dataset.exists():
            shutil.rmtree(args.create_dataset)
        create_pilot_dataset(args.source_dataset, args.create_dataset, selected)
        print(f"Created pilot dataset: {args.create_dataset}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
