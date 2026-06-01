#!/usr/bin/env python3
"""
Pre-compute and cache caption embeddings for a retrieval pool.

Produces ``<POOL_DIR>/caption_embeddings.npy`` (N x D L2-normalised float32)
and ``<POOL_DIR>/caption_embeddings.json`` (sidecar with model name +
shape, used by ``build_retrieval_pool`` to validate the cache).

Why pre-compute:
  ``delta_experiment/scripts/common.py::build_retrieval_pool`` previously
  encoded every pool caption fresh on every TTA job (~30-60 s for a 25K
  pool). After the v3 deploy, ``build_retrieval_pool`` checks for a
  matching cache alongside ``metadata.csv`` and loads it directly. With a
  50-job sweep that's 25-50 minutes saved.

Caching rules enforced by the loader:
  - ``<pool_dir>/caption_embeddings.npy`` row count must equal
    ``len(pool_entries)``.
  - Sidecar ``caption_embeddings.json`` must record the same model name
    that the caller requested (default ``all-MiniLM-L6-v2``).
  - Otherwise the loader silently falls back to fresh encoding.

Usage:
  python scripts/precompute_pool_embeddings.py \\
      --pool-dir <pool_dir> [--model all-MiniLM-L6-v2] [--batch-size 256]

Environment fallback:
  POOL_DIR : if set, used as the default for ``--pool-dir`` (so the
             sbatch wrapper can pass it through ``--export``).

Submit on cluster:
  sbatch --account=torch_pr_36_mren \\
      --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/datasets/ucf101_pool_max \\
      delta_experiment/sbatch/precompute_pool_embeddings.sbatch
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


def _install_st_compat_shim() -> None:
    """Stub `is_nltk_available` on `transformers.utils.import_utils`.

    sentence-transformers >= 2.3 imports `is_nltk_available` (and the
    paired `NLTK_IMPORT_ERROR`) from `transformers.utils.import_utils`
    at module-import time, but `transformers < 4.40` does not expose
    those symbols. The longcat conda env pins `transformers==4.33.2`
    (a hard requirement of the diffusion model), so importing
    sentence_transformers raises ImportError before any user code runs.

    We never use `DenoisingAutoEncoderDataset` (the only consumer of
    nltk inside sentence-transformers), so a benign stub returning
    False is sufficient. Idempotent.
    """
    try:
        import transformers.utils.import_utils as _iu  # type: ignore
        if not hasattr(_iu, "is_nltk_available"):
            _iu.is_nltk_available = lambda: False  # type: ignore[attr-defined]
        if not hasattr(_iu, "NLTK_IMPORT_ERROR"):
            _iu.NLTK_IMPORT_ERROR = (
                "nltk shim installed by precompute_pool_embeddings "
                "(transformers<4.40 has no is_nltk_available)."
            )
    except Exception:
        pass


def _read_captions(metadata_csv: Path) -> List[str]:
    captions: List[str] = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "caption" not in (reader.fieldnames or []):
            print(f"ERROR: 'caption' column not in {metadata_csv}",
                  file=sys.stderr)
            print(f"  Columns present: {reader.fieldnames}", file=sys.stderr)
            sys.exit(2)
        for row in reader:
            captions.append(row.get("caption") or "")
    return captions


def _resolve_pool_dir(arg_value: str | None) -> Path | None:
    candidate = arg_value or os.environ.get("POOL_DIR")
    if not candidate:
        return None
    return Path(candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pool-dir",
        type=str,
        default=None,
        help="Pool directory containing metadata.csv. Falls back to "
             "POOL_DIR env var.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (default %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path (default <pool_dir>/caption_embeddings.npy).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if a matching cache already exists.",
    )
    args = parser.parse_args()

    pool_dir = _resolve_pool_dir(args.pool_dir)
    if pool_dir is None:
        print("ERROR: --pool-dir is required (or set POOL_DIR env var).",
              file=sys.stderr)
        return 2
    metadata_csv = pool_dir / "metadata.csv"
    if not metadata_csv.exists():
        print(f"ERROR: {metadata_csv} does not exist.", file=sys.stderr)
        return 2

    out_path = (
        Path(args.output).resolve() if args.output
        else pool_dir / "caption_embeddings.npy"
    )
    sidecar = out_path.with_suffix(".json")

    print("=" * 70)
    print("precompute_pool_embeddings")
    print("=" * 70)
    print(f"  pool dir   : {pool_dir}")
    print(f"  metadata   : {metadata_csv}")
    print(f"  model      : {args.model}")
    print(f"  output     : {out_path}")
    print(f"  sidecar    : {sidecar}")
    print(f"  batch size : {args.batch_size}")
    print(f"  force      : {args.force}")
    print()

    if out_path.exists() and not args.force:
        sidecar_data = {}
        if sidecar.exists():
            try:
                sidecar_data = json.loads(
                    sidecar.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                sidecar_data = {}
        try:
            existing = np.load(out_path, mmap_mode="r")
            print(f"  Cache present: shape={existing.shape}, "
                  f"sidecar_model={sidecar_data.get('model')}")
        except (OSError, ValueError) as exc:
            print(f"  Cache exists but unreadable ({exc}); will overwrite.")
            existing = None
        if existing is not None and sidecar_data.get("model") == args.model:
            print("  Same model -> skipping (use --force to re-encode).")
            return 0
        print("  Different/missing model in sidecar -> re-encoding.")

    print("[1/3] Loading captions ...", flush=True)
    captions = _read_captions(metadata_csv)
    print(f"      {len(captions)} captions loaded.", flush=True)

    if len(captions) == 0:
        print("ERROR: 0 captions in metadata.csv; refusing to write empty "
              "cache.", file=sys.stderr)
        return 2

    print(f"[2/3] Loading model '{args.model}' ...", flush=True)
    t0 = time.time()
    _install_st_compat_shim()
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(args.model)
    print(f"      Model ready in {time.time() - t0:.1f}s.", flush=True)

    print(f"[3/3] Encoding {len(captions)} captions "
          f"(batch_size={args.batch_size}) ...", flush=True)
    t0 = time.time()
    embeddings = st_model.encode(
        captions,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"      Encoded in {elapsed:.1f}s "
          f"({len(captions) / max(elapsed, 1e-3):.1f} captions/s).",
          flush=True)
    print(f"      Shape: {embeddings.shape}, dtype: {embeddings.dtype}.")

    embeddings = np.ascontiguousarray(embeddings.astype(np.float32, copy=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    sidecar.write_text(
        json.dumps(
            {
                "model": args.model,
                "n_entries": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "dtype": str(embeddings.dtype),
                "normalized": True,
                "metadata_csv": str(metadata_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    size_mb = embeddings.nbytes / 1024 / 1024
    print()
    print("=" * 70)
    print(f"DONE. Wrote {out_path} ({size_mb:.1f} MB)")
    print(f"      Sidecar: {sidecar}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
