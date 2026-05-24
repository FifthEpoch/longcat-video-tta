#!/bin/bash
# ==============================================================================
# Prepare Panda-70M metadata on the LOGIN node (has internet access)
# ==============================================================================
# Run this BEFORE submitting the download sbatch job.
# It downloads the official Panda-70M metadata CSV and converts it to
# a clean JSONL that the download script can use.
#
# Usage (on login node):
#   bash datasets/prepare_metadata.sh
# ==============================================================================

set -euo pipefail

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
DATASET_DIR="${PROJECT_ROOT}/datasets/panda_100"
META_JSONL="${DATASET_DIR}/panda70m_metadata_clean.jsonl"

# Load conda
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH_BASE}/conda-envs/longcat"

mkdir -p "${DATASET_DIR}"
cd "${PROJECT_ROOT}"

echo "=============================================================================="
echo "Preparing Panda-70M metadata"
echo "=============================================================================="

# Try multiple sources in order of reliability
python3 - <<'PYEOF'
import json
import sys
import os

DATASET_DIR = os.environ.get("DATASET_DIR", "/scratch/wc3013/longcat-video-tta/datasets/panda_100")
META_JSONL = os.environ.get("META_JSONL", f"{DATASET_DIR}/panda70m_metadata_clean.jsonl")

def try_huggingface():
    """Try loading from HuggingFace datasets (multiple possible names)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets library not installed")
        return None

    # Try multiple possible dataset names
    candidates = [
        "snap-research/Panda-70M",
        "iejMac/Panda-70M",
        "tsb0601/Panda-70M",
        "Awiny/Panda-70M",
    ]
    for name in candidates:
        try:
            print(f"  Trying HuggingFace: {name} ...")
            ds = load_dataset(name, split="train", streaming=True)
            rows = []
            for i, item in enumerate(ds):
                if i >= 50000:
                    break
                rows.append(dict(item))
                if (i + 1) % 10000 == 0:
                    print(f"    Loaded {i+1} rows...")
            if rows:
                print(f"  Success: {name} ({len(rows)} rows)")
                return rows
        except Exception as e:
            print(f"  Failed: {name} ({e})")
            continue
    return None


def try_google_drive():
    """Try downloading from official Panda-70M Google Drive."""
    try:
        import subprocess
        # Official Panda-70M training split 0
        # Source: https://github.com/snap-research/Panda-70M
        gdrive_ids = [
            "1k7NzU6wVNZYl6NxOhLXE7Hz7OrpzNLgB",  # from original Open-Sora setup
        ]
        for gid in gdrive_ids:
            tmp_path = f"{DATASET_DIR}/panda70m_raw_download.tmp"
            url = f"https://drive.google.com/uc?id={gid}"
            print(f"  Trying Google Drive: {gid} ...")
            try:
                import gdown
                gdown.download(id=gid, output=tmp_path, quiet=False)
            except ImportError:
                subprocess.run(["pip", "install", "gdown", "--quiet"], check=True)
                import gdown
                gdown.download(id=gid, output=tmp_path, quiet=False)

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1000:
                print(f"  Failed: download too small or missing")
                continue

            # Try to parse as CSV or JSONL
            rows = parse_file(tmp_path)
            if rows:
                os.remove(tmp_path)
                return rows
            else:
                print(f"  Failed to parse downloaded file")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception as e:
        print(f"  Google Drive failed: {e}")
    return None


def try_panda70m_github_csv():
    """Download Panda-70M training split CSV from the official GitHub releases."""
    import subprocess
    import csv
    import io

    # The Panda-70M CSV files are at:
    # https://github.com/snap-research/Panda-70M/blob/main/splitting/
    # But the actual data CSVs are on Google Drive.
    # Try to grab the sample/test split which is smaller.
    urls = [
        # Panda-70M test split (smaller, more likely to work)
        "https://raw.githubusercontent.com/snap-research/Panda-70M/main/splitting/panda70m_testing.csv",
    ]
    for url in urls:
        try:
            print(f"  Trying GitHub: {url} ...")
            import urllib.request
            with urllib.request.urlopen(url, timeout=60) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(content))
            rows = []
            for row in reader:
                rows.append(row)
                if len(rows) >= 50000:
                    break
            if rows:
                print(f"  Success: {len(rows)} rows from GitHub")
                return rows
        except Exception as e:
            print(f"  Failed: {e}")
    return None


def parse_file(path):
    """Try to parse a file as JSONL or CSV."""
    import csv
    import gzip

    # Check if gzipped
    with open(path, "rb") as f:
        magic = f.read(2)

    rows = []

    # Try JSONL (gzipped or plain)
    try:
        if magic == b"\x1f\x8b":
            opener = lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            opener = lambda: open(path, "r", encoding="utf-8", errors="replace")
        with opener() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        if len(rows) > 100:
            print(f"  Parsed as JSONL: {len(rows)} rows")
            return rows
    except Exception:
        pass

    # Try CSV
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= 50000:
                    break
        if len(rows) > 100:
            print(f"  Parsed as CSV: {len(rows)} rows")
            return rows
    except Exception:
        pass

    return None


def normalize_and_save(rows, output_path):
    """Normalize rows and save as clean JSONL."""
    import re
    clean = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        video_id = r.get("videoID", "") or r.get("video_id", "") or r.get("id", "")
        url = r.get("url", "") or r.get("video_url", "")
        caption = r.get("caption", "") or r.get("text", "") or "video"

        # Extract video ID from URL if needed
        if not video_id and url:
            m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
            if m:
                video_id = m.group(1)

        if not video_id:
            continue

        # Parse timestamp
        timestamp = r.get("timestamp", {})
        if isinstance(timestamp, str):
            try:
                timestamp = json.loads(timestamp.replace("'", '"'))
            except Exception:
                timestamp = {}

        entry = {
            "videoID": video_id,
            "url": url or f"https://www.youtube.com/watch?v={video_id}",
            "caption": caption,
            "timestamp": timestamp if isinstance(timestamp, dict) else {},
        }
        clean.append(entry)

    with open(output_path, "w") as f:
        for entry in clean:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved {len(clean)} clean entries to {output_path}")
    return clean


# ── Main ──────────────────────────────────────────────────────────────────
print("Attempting to download Panda-70M metadata...\n")

# Check if clean metadata already exists
if os.path.exists(META_JSONL):
    with open(META_JSONL) as f:
        count = sum(1 for _ in f)
    if count > 1000:
        print(f"Clean metadata already exists: {META_JSONL} ({count} rows)")
        print("Delete it to re-download.")
        sys.exit(0)

rows = None

# Method 1: HuggingFace
print("Method 1: HuggingFace datasets")
rows = try_huggingface()

# Method 2: GitHub raw CSV (test split)
if not rows:
    print("\nMethod 2: GitHub raw CSV")
    rows = try_panda70m_github_csv()

# Method 3: Google Drive
if not rows:
    print("\nMethod 3: Google Drive")
    rows = try_google_drive()

if not rows:
    print("\nERROR: All metadata sources failed!")
    print("Please download Panda-70M metadata manually from:")
    print("  https://github.com/snap-research/Panda-70M")
    print(f"Save as CSV/JSONL at: {META_JSONL}")
    sys.exit(1)

# Normalize and save
normalize_and_save(rows, META_JSONL)
print("\nDone! You can now submit the download job:")
print(f"  sbatch --account=torch_pr_36_mren datasets/download_panda70m.sbatch")

PYEOF

echo ""
echo "=============================================================================="
echo "Metadata preparation complete"
echo "=============================================================================="
