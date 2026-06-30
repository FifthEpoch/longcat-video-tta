#!/usr/bin/env bash
# Save a pasted cluster dump into gitignored local_archive/YYYY-MM-DD/
# Usage:
#   bash scripts/ingest_local_archive_dump.sh 2026-06-30 < cluster_output.txt
#   bash scripts/ingest_local_archive_dump.sh 2026-06-30 cluster_output.txt
set -euo pipefail

DATE_TAG="${1:?Usage: ingest_local_archive_dump.sh DATE_TAG [input.txt]}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="${REPO_ROOT}/sweep_experiment/reports/local_archive/${DATE_TAG}"
INPUT="${2:-}"

mkdir -p "${ARCHIVE}/reports"

if [[ -n "${INPUT}" ]]; then
  cp "${INPUT}" "${ARCHIVE}/cluster_dump.txt"
else
  cat > "${ARCHIVE}/cluster_dump.txt"
fi

python3 "${REPO_ROOT}/scripts/split_cluster_dump.py" "${ARCHIVE}/cluster_dump.txt" "${ARCHIVE}"

echo "Archived to ${ARCHIVE}"
ls -la "${ARCHIVE}" "${ARCHIVE}/reports"
