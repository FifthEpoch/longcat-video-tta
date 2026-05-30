#!/usr/bin/env python3
"""
deploy_pool_extensions_v3.py
============================

Cluster-side deploy script for the **embedding-cache** upgrade. This is a
focused delta on top of v2 (the chunked UCF builder + same-source
exclusion patches).

Why v3:
  ``delta_experiment/scripts/common.py::build_retrieval_pool`` currently
  encodes every pool caption fresh on every TTA job. With a 25-30K-entry
  pool that costs 30-60 seconds per job (model load + encode). Across a
  50-job sweep that's 25-50 wall-clock minutes spent re-encoding the
  same captions over and over.

What v3 changes:

  PATCH  delta_experiment/scripts/common.py::build_retrieval_pool
           - Adds an optional ``pool_dir: Optional[Path] = None`` argument.
           - If ``pool_dir`` is None the function infers it from
             ``pool_entries[0]['video_path']`` (.../<pool_dir>/videos/foo.mp4).
           - If ``<pool_dir>/caption_embeddings.npy`` exists with row
             count matching ``len(pool_entries)`` AND the sidecar
             ``<pool_dir>/caption_embeddings.json`` records the same
             ``model_name`` the caller requested, the cache is loaded
             directly. Otherwise we silently fall back to the original
             on-the-fly ``st_model.encode(...)`` path -- behaviour for
             un-cached pools is unchanged.

The cache files are produced by the companion script
``scripts/precompute_pool_embeddings.py`` (also pushed in this commit).

Idempotent: re-running ``--apply`` after the patch is already in place
reports "already applied" instead of failing.

Usage:

    python scripts/deploy_pool_extensions_v3.py            # dry-run
    python scripts/deploy_pool_extensions_v3.py --apply    # patch

After applying, pre-compute embeddings for an existing pool:

    sbatch --account=torch_pr_36_mren \\
        --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/datasets/ucf101_pool_max \\
        delta_experiment/sbatch/precompute_pool_embeddings.sbatch
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List


# ============================================================================
# Helpers (mirrors v2 deploy patterns)
# ============================================================================


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def apply_patch(
    target: Path, old: str, new: str, *, apply: bool, label: str = "",
) -> str:
    target = target.resolve()
    label = label or str(target)
    if not target.exists():
        return f"  [SKIP] {label}: target missing -> {target}"
    content = target.read_text(encoding="utf-8")
    if old in content:
        if new in content:
            return (f"  [skip] {label} (both old and new present?? "
                    f"-- inspect manually)")
        new_content = content.replace(old, new, 1)
        if apply:
            target.write_text(new_content, encoding="utf-8")
            return (f"  [PATCH] {label}\n"
                    f"          old md5: {_md5(old)}\n"
                    f"          new md5: {_md5(new)}")
        return (f"  [dry-run] would patch {label}\n"
                f"          old md5: {_md5(old)}\n"
                f"          new md5: {_md5(new)}")
    if new in content:
        return f"  [skip] {label} (already applied)"
    return (f"  [SKIP] {label}: anchor text not found.\n"
            f"          Inspect {target} manually -- it may have a different\n"
            f"          baseline (e.g. v2 was applied but build_retrieval_pool\n"
            f"          was already modified by a separate hand-edit).")


# ============================================================================
# common.py patch: build_retrieval_pool with cache loading
# ============================================================================

#: The exact pre-v3 source of build_retrieval_pool. Must match byte-for-byte.
BUILD_RETRIEVAL_OLD = '''def build_retrieval_pool(
    pool_entries: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[np.ndarray, Any]:
    """Pre-compute normalised sentence embeddings for a pool of videos.

    Only used when ``batch_method='similarity'``. The ``sentence_transformers``
    import is performed lazily here so that ``random`` neighbour sampling
    does not require the package to be importable.

    Returns (embeddings, sentence_transformer_model) so the model can be
    reused for encoding query captions without reloading.
    """
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model_name)
    captions = [v.get("caption", "") for v in pool_entries]
    embeddings = st_model.encode(
        captions, show_progress_bar=True, normalize_embeddings=True,
    )
    print(f"  Retrieval pool: {len(pool_entries)} videos, "
          f"embedding dim={embeddings.shape[1]}")
    return embeddings, st_model
'''


BUILD_RETRIEVAL_NEW = '''def build_retrieval_pool(
    pool_entries: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
    pool_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Any]:
    """Pre-compute (or load cached) normalised sentence embeddings for a pool.

    Only used when ``batch_method='similarity'``. The ``sentence_transformers``
    import is performed lazily here so that ``random`` neighbour sampling
    does not require the package to be importable.

    Caching:
        If ``pool_dir`` is given (or can be inferred from
        ``pool_entries[0]['video_path']`` -- standard layout has
        ``<pool_dir>/videos/<file>.mp4``), and a file
        ``<pool_dir>/caption_embeddings.npy`` exists whose row count equals
        ``len(pool_entries)`` and whose sidecar
        ``<pool_dir>/caption_embeddings.json`` records the same
        ``model_name``, the cached array is returned directly (no
        re-encode). The cache files are produced by
        ``scripts/precompute_pool_embeddings.py``.

        On any cache mismatch (size, model, sidecar absent or unreadable)
        the function silently falls back to fresh on-the-fly encoding,
        preserving the original behaviour for un-cached pools.

    Returns (embeddings, sentence_transformer_model) so the model can be
    reused for encoding query captions without reloading.
    """
    import json as _json

    if pool_dir is None and pool_entries:
        first_path = pool_entries[0].get("video_path", "")
        if first_path:
            inferred = Path(first_path).resolve().parent.parent
            if (inferred / "metadata.csv").exists():
                pool_dir = inferred

    cached_emb = None
    if pool_dir is not None:
        cache_path = Path(pool_dir) / "caption_embeddings.npy"
        sidecar_path = Path(pool_dir) / "caption_embeddings.json"
        if cache_path.exists():
            try:
                cand_emb = np.load(cache_path)
            except (OSError, ValueError) as exc:
                print(f"  Retrieval pool: cache load failed ({exc}); "
                      f"re-encoding")
                cand_emb = None
            if cand_emb is not None:
                cand_model = None
                if sidecar_path.exists():
                    try:
                        cand_model = _json.loads(
                            sidecar_path.read_text(encoding="utf-8")
                        ).get("model")
                    except (OSError, ValueError):
                        cand_model = None
                if cand_emb.shape[0] != len(pool_entries):
                    print(f"  Retrieval pool: cache row-count "
                          f"{cand_emb.shape[0]} != {len(pool_entries)}; "
                          f"re-encoding")
                elif cand_model is not None and cand_model != model_name:
                    print(f"  Retrieval pool: cache model '{cand_model}' "
                          f"!= requested '{model_name}'; re-encoding")
                else:
                    cached_emb = cand_emb.astype(np.float32, copy=False)
                    print(f"  Retrieval pool: loaded cached embeddings "
                          f"from {cache_path} "
                          f"(shape={cached_emb.shape}, "
                          f"model={cand_model or model_name})")

    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model_name)

    if cached_emb is not None:
        print(f"  Retrieval pool: {len(pool_entries)} videos, "
              f"embedding dim={cached_emb.shape[1]} (cached)")
        return cached_emb, st_model

    captions = [v.get("caption", "") for v in pool_entries]
    embeddings = st_model.encode(
        captions, show_progress_bar=True, normalize_embeddings=True,
    )
    print(f"  Retrieval pool: {len(pool_entries)} videos, "
          f"embedding dim={embeddings.shape[1]}")
    return embeddings, st_model
'''


# ============================================================================
# Driver
# ============================================================================

PATCHES: List[Dict[str, str]] = [
    {
        "path": "delta_experiment/scripts/common.py",
        "label": "common.py: build_retrieval_pool with cached embeddings",
        "old": BUILD_RETRIEVAL_OLD,
        "new": BUILD_RETRIEVAL_NEW,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply patches in-place. Without this flag the script runs as "
             "a dry-run and only previews what would change.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root (default: parent dir of this script).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    print("=" * 72)
    print("deploy_pool_extensions_v3.py")
    print("=" * 72)
    print(f"  repo root : {repo_root}")
    print(f"  mode      : {'APPLY' if args.apply else 'dry-run'}")
    print()
    print(f"Plan: apply {len(PATCHES)} patch(es).")
    print()
    print("Patches:")
    failures = 0
    for p in PATCHES:
        msg = apply_patch(
            repo_root / p["path"], p["old"], p["new"],
            apply=args.apply, label=p["label"],
        )
        print(msg)
        if "[SKIP]" in msg and "anchor text not found" in msg:
            failures += 1

    print()
    print("=" * 72)
    if failures:
        print(f"WARNING: {failures} patch(es) could not find their anchor "
              f"text. Inspect manually before retrying.")
        if args.apply:
            return 1
    if args.apply:
        print("OK -- changes applied. Next steps:")
        print("")
        print("  # Once a pool is built, pre-compute its caption embeddings:")
        print("  sbatch --account=torch_pr_36_mren \\")
        print("      --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/datasets/ucf101_pool_max \\")
        print("      delta_experiment/sbatch/precompute_pool_embeddings.sbatch")
        print("")
        print("  # Then a retrieval-augmented sweep that points at the pool")
        print("  # via `retrieval_pool_dir:` will auto-load the cached")
        print("  # embeddings (no fresh encode at TTA-job startup).")
    else:
        print("dry-run only. Re-run with --apply to write the patch.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
