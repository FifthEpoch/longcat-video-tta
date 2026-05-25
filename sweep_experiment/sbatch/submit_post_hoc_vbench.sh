#!/bin/bash
# ============================================================================
# Post-hoc VBench++ evaluation for the 2048-video sweep.
#
# The headline 2048v jobs already run VBench inline (COMPUTE_VBENCH=1 in
# submit_standard_2048v_chunked.sh), but this script provides a way to
# (re-)evaluate VBench separately:
#
#   - As a recovery path when an inline VBench evaluation fails or is
#     truncated by job preemption.
#   - To evaluate a *different* set of VBench dimensions than the
#     default inline ones.
#   - To re-run VBench on the merged (across-chunk) video sets rather
#     than per-chunk.
#
# Usage:
#   bash sweep_experiment/sbatch/submit_post_hoc_vbench.sh
#
#   # Specific results dir only:
#   RESULTS_ROOTS="sweep_experiment/results/panda_2048v" \
#       bash sweep_experiment/sbatch/submit_post_hoc_vbench.sh
#
#   # Custom dimensions (space-separated, must match vbench's names):
#   VBENCH_DIMENSIONS="subject_consistency motion_smoothness temporal_flickering aesthetic_quality" \
#       bash sweep_experiment/sbatch/submit_post_hoc_vbench.sh
#
#   # Dry run:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_post_hoc_vbench.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-0}"

# Default VBench dimensions to evaluate (matches the inline default).
VBENCH_DIMENSIONS="${VBENCH_DIMENSIONS:-subject_consistency motion_smoothness temporal_flickering aesthetic_quality imaging_quality}"

# Results directories to scan for chunk_*/videos/ subdirectories.
RESULTS_ROOTS="${RESULTS_ROOTS:-sweep_experiment/results/panda_2048v sweep_experiment/results/ucf101_2048v delta_experiment/results/tinylora_panda_2048v delta_experiment/results/tinylora_ucf101_2048v}"

# Per-chunk VBench wall time (5 dimensions x ~128 videos ~= 1-2h).
TIME="${TIME:-04:00:00}"

SBATCH_TPL="$(mktemp -t vbench_posthoc.XXXXXX.sbatch)"
cat > "${SBATCH_TPL}" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=sweep_experiment/slurm_log/vbench_%x_%j.out
#SBATCH --error=sweep_experiment/slurm_log/vbench_%x_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
export VBENCH_CACHE_DIR="${SCRATCH_BASE}/.cache/vbench"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

# Variables expected from --export:
#   VIDEO_DIR (chunk videos dir)
#   OUTPUT_JSON (vbench output path)
#   DIMS (space-separated dimensions list)

echo "VBench++ post-hoc evaluation"
echo "  videos     : ${VIDEO_DIR}"
echo "  output     : ${OUTPUT_JSON}"
echo "  dimensions : ${DIMS}"
echo ""

"${PYTHON}" sweep_experiment/scripts/eval_vbench.py \
    --video-dir "${VIDEO_DIR}" \
    --output "${OUTPUT_JSON}" \
    --dimensions ${DIMS}

echo "Done: ${OUTPUT_JSON}"
SBATCH_EOF

echo "============================================================"
echo "Post-hoc VBench++ submission"
echo "============================================================"
echo "  dimensions: ${VBENCH_DIMENSIONS}"
echo "  dry run   : ${DRY_RUN}"
echo "  roots     : ${RESULTS_ROOTS}"
echo "============================================================"
echo ""

count=0
for root in ${RESULTS_ROOTS}; do
    abs_root="${PROJECT_ROOT}/${root}"
    if [ ! -d "${abs_root}" ]; then
        echo "  skip: ${abs_root} not found"
        continue
    fi
    # Find every chunk_*/videos/ that has at least one .mp4
    while IFS= read -r videos_dir; do
        n_vids=$(find "${videos_dir}" -maxdepth 1 -name '*.mp4' -type f | head -n 1 | wc -l)
        if [ "${n_vids}" = "0" ]; then continue; fi

        chunk_dir="$(dirname "${videos_dir}")"
        output_json="${chunk_dir}/vbench_scores_posthoc.json"
        run_id="$(basename "$(dirname "${chunk_dir}")")"
        chunk_id="$(basename "${chunk_dir}")"
        job_name="vbench_${run_id}_${chunk_id}"

        cmd=(sbatch
            --account="${ACCOUNT}"
            --job-name="${job_name}"
            --time="${TIME}"
            --export="ALL,VIDEO_DIR=${videos_dir},OUTPUT_JSON=${output_json},DIMS=${VBENCH_DIMENSIONS}"
            "${SBATCH_TPL}")

        if [ "${DRY_RUN}" = "1" ]; then
            echo "[DRY] ${cmd[*]}"
        else
            "${cmd[@]}"
        fi
        count=$((count + 1))
    done < <(find "${abs_root}" -type d -name videos)
done

echo ""
echo "Submitted ${count} VBench post-hoc jobs."
echo "Monitor with: squeue -u \$USER --name='vbench_*'"
echo "Output files: <chunk_dir>/vbench_scores_posthoc.json"
echo ""
echo "(Template at ${SBATCH_TPL} is kept; safe to delete after queue drains.)"
