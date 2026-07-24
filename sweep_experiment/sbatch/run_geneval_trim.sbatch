#!/bin/bash
#SBATCH --job-name=geneval_trim
#SBATCH --output=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/geneval_trim_%A_%a.out
#SBATCH --error=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/geneval_trim_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-12

# ============================================================================
# Step 1 (sbatch): trim conditioning frames -> videos_geneval/ for one arm.
# Job ARRAY: one task per arm (0..12). CPU-only (no GPU).
#
# Self-healing: before trimming, deletes any existing videos_geneval clip that
# does NOT decode to exactly NUM_GEN_FRAMES frames (removes partial/corrupt
# clips left by an interrupted run), then make_geneval_clips.py recreates them.
# Idempotent: good existing clips are skipped.
#
# Submit (all arms):
#   sbatch sweep_experiment/sbatch/run_geneval_trim.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SERIES="${SERIES:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_GEN_FRAMES="${NUM_GEN_FRAMES:-15}"   # frames expected in each gen-only clip

ARMS=(
    NOTTA
    S2_LR1e3 S2_LR5e3 S2_LR1e2
    S5_LR1e3 S5_LR5e3 S5_LR1e2
    S10_LR1e3 S10_LR5e3 S10_LR1e2
    S20_LR1e3 S20_LR5e3 S20_LR1e2
)

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH_BASE}/conda-envs/longcat"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

IDX="${SLURM_ARRAY_TASK_ID:-0}"
ARM="${ARMS[$IDX]}"
METHOD_DIR="${SERIES}/${ARM}"

echo "============================================================"
echo "geneval trim  task=${IDX}  arm=${ARM}"
echo "  method_dir : ${METHOD_DIR}"
echo "  drop first : ${NUM_COND_FRAMES} cond frames  (expect ${NUM_GEN_FRAMES} gen frames)"
echo "============================================================"

if [ ! -d "${METHOD_DIR}" ]; then
    echo "ERROR: ${METHOD_DIR} does not exist" >&2
    exit 2
fi

# --- pre-clean: drop any partial/corrupt gen-only clip (!= NUM_GEN_FRAMES) ---
removed=0
while IFS= read -r f; do
    n=$(ffprobe -v error -count_frames -select_streams v:0 \
          -show_entries stream=nb_read_frames -of csv=p=0 "$f" 2>/dev/null || echo "x")
    if [ "$n" != "${NUM_GEN_FRAMES}" ]; then
        rm -f "$f"; removed=$((removed + 1))
    fi
done < <(find "${METHOD_DIR}" -path '*/videos_geneval/*.mp4' 2>/dev/null)
echo "pre-clean: removed ${removed} partial/corrupt gen-only clip(s)"

# --- trim (skips good existing clips) ---
python3 scripts/make_geneval_clips.py \
    --method-dir "${METHOD_DIR}" \
    --num-cond-frames "${NUM_COND_FRAMES}"

# --- verify: every gen-only clip must be exactly NUM_GEN_FRAMES ---
bad=0
while IFS= read -r f; do
    n=$(ffprobe -v error -count_frames -select_streams v:0 \
          -show_entries stream=nb_read_frames -of csv=p=0 "$f" 2>/dev/null || echo "x")
    if [ "$n" != "${NUM_GEN_FRAMES}" ]; then
        echo "  BAD ${f}: ${n} frames"; bad=$((bad + 1))
    fi
done < <(find "${METHOD_DIR}" -path '*/videos_geneval/*.mp4' 2>/dev/null)
n_out=$(find "${METHOD_DIR}" -path '*/videos_geneval/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
echo "arm=${ARM}: gen-only clips=${n_out}  bad(!=${NUM_GEN_FRAMES})=${bad}"
[ "${bad}" = "0" ] || { echo "ERROR: ${bad} bad clips in ${ARM}" >&2; exit 1; }
echo "arm=${ARM}: OK"
