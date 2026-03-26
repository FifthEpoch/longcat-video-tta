#!/usr/bin/env bash
# ==============================================================================
# pull_from_cluster.sh — Transfer data from NYU Torch /scratch to local machine
#
# Uses the Data Transfer Node (DTN) as recommended by NYU HPC docs:
#   https://services.rt.nyu.edu/docs/hpc/storage/data_transfers/
#
# NOTE: You will be prompted to authenticate via Microsoft device login
#       (PIN at https://microsoft.com/devicelogin) on each connection.
#
# Usage examples:
#   ./pull_from_cluster.sh                          # list what's in /scratch/wc3013
#   ./pull_from_cluster.sh datasets/panda_100       # pull panda_100 dataset
#   ./pull_from_cluster.sh delta_experiment/results  # pull delta results
#   ./pull_from_cluster.sh --dry-run datasets/       # preview without downloading
#   ./pull_from_cluster.sh --all                     # pull EVERYTHING (careful!)
# ==============================================================================
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
NETID="wc3013"
DTN_HOST="dtn.torch.hpc.nyu.edu"          # NYU Torch Data Transfer Node
REMOTE_BASE="/scratch/${NETID}"
LOCAL_BASE="${HOME}/Downloads/scratch_wc3013"

# rsync flags:
#   -a  archive mode (recursive, preserve permissions/times/symlinks)
#   -v  verbose
#   -z  compress during transfer
#   -h  human-readable sizes
#   --progress  show per-file progress
RSYNC_OPTS=(-avzh --progress)

# ─── Parse arguments ────────────────────────────────────────────────────────
DRY_RUN=false
LIST_ONLY=false
SUBPATH=""

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [SUBPATH]

Transfer data from ${REMOTE_BASE}/<SUBPATH> on NYU Torch HPC
to ${LOCAL_BASE}/<SUBPATH> on your local machine.

Arguments:
  SUBPATH           Subdirectory or file under /scratch/${NETID} to pull.
                    If omitted, lists the contents of /scratch/${NETID}.

Options:
  --dry-run, -n     Show what would be transferred without actually copying.
  --all             Pull the ENTIRE /scratch/${NETID} directory (use with care).
  --list, -l        List contents of the remote path without transferring.
  --local-dir DIR   Override the local destination directory.
                    Default: ${LOCAL_BASE}
  --exclude PAT     Exclude files matching PAT (can be repeated).
  -h, --help        Show this help message.

Examples:
  $(basename "$0")                                  # list /scratch/${NETID}
  $(basename "$0") datasets/panda_100               # pull panda_100 videos
  $(basename "$0") --dry-run delta_experiment       # preview delta_experiment transfer
  $(basename "$0") --exclude '*.mp4' datasets/      # pull datasets minus videos
  $(basename "$0") --all --dry-run                  # preview pulling everything
EOF
}

EXCLUDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --all)
            SUBPATH=""
            LIST_ONLY=false
            shift
            # We'll handle --all specially below
            ALL_MODE=true
            ;;
        --list|-l)
            LIST_ONLY=true
            shift
            ;;
        --local-dir)
            LOCAL_BASE="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDES+=(--exclude "$2")
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            print_usage
            exit 1
            ;;
        *)
            SUBPATH="$1"
            shift
            ;;
    esac
done

ALL_MODE="${ALL_MODE:-false}"

# ─── Determine remote and local paths ───────────────────────────────────────
REMOTE_PATH="${REMOTE_BASE}"
LOCAL_PATH="${LOCAL_BASE}"

if [[ -n "$SUBPATH" ]]; then
    # Strip leading/trailing slashes for consistency
    SUBPATH="${SUBPATH#/}"
    SUBPATH="${SUBPATH%/}"
    REMOTE_PATH="${REMOTE_BASE}/${SUBPATH}"
    LOCAL_PATH="${LOCAL_BASE}/${SUBPATH}"
fi

REMOTE_URI="${NETID}@${DTN_HOST}:${REMOTE_PATH}"

# ─── If no subpath and not --all, just list remote contents ──────────────────
if [[ -z "$SUBPATH" && "$ALL_MODE" == "false" ]] || [[ "$LIST_ONLY" == "true" ]]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Listing: ${REMOTE_PATH}"
    echo "║  Host:    ${DTN_HOST}"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Connecting to DTN (you may need to authenticate via device login)..."
    echo ""
    ssh "${NETID}@${DTN_HOST}" "ls -lhA ${REMOTE_PATH}/ 2>/dev/null || echo 'Path not found: ${REMOTE_PATH}'"
    echo ""
    echo "To download a subdirectory, run:"
    echo "  $(basename "$0") <subdirectory_name>"
    exit 0
fi

# ─── Dry run or actual transfer ─────────────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
    RSYNC_OPTS+=(--dry-run)
fi

# Create local destination
mkdir -p "${LOCAL_PATH}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  NYU Torch HPC → Local Data Transfer"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Remote: ${REMOTE_URI}/"
echo "║  Local:  ${LOCAL_PATH}/"
echo "║  DTN:    ${DTN_HOST}"
if [[ "$DRY_RUN" == "true" ]]; then
echo "║  Mode:   DRY RUN (no files will be copied)"
else
echo "║  Mode:   TRANSFER"
fi
if [[ ${#EXCLUDES[@]} -gt 0 ]]; then
echo "║  Excludes: ${EXCLUDES[*]}"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Connecting to DTN (you may need to authenticate via device login)..."
echo ""

# Add trailing slash to remote to copy contents (not the directory itself)
rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]+"${EXCLUDES[@]}"}" \
    -e ssh \
    "${REMOTE_URI}/" \
    "${LOCAL_PATH}/"

RSYNC_EXIT=$?

echo ""
if [[ $RSYNC_EXIT -eq 0 ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "✓ Dry run complete. Re-run without --dry-run to transfer."
    else
        echo "✓ Transfer complete!"
        echo "  Files saved to: ${LOCAL_PATH}/"
        # Show summary
        if command -v du &>/dev/null; then
            TOTAL_SIZE=$(du -sh "${LOCAL_PATH}" 2>/dev/null | cut -f1)
            echo "  Total size: ${TOTAL_SIZE}"
        fi
    fi
else
    echo "✗ rsync exited with code ${RSYNC_EXIT}"
    exit $RSYNC_EXIT
fi
