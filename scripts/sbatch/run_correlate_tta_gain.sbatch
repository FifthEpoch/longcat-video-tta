#!/bin/bash
#SBATCH --job-name=correlate_tta_gain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/logs/correlate_tta_gain_%j.out
#SBATCH --error=sweep_experiment/logs/correlate_tta_gain_%j.err

# ============================================================================
# Per-video TTA-gain x video-feature correlation analysis.
#
# Joins the per-video gains CSV (from analyze_per_video_tta_gain.py) with the
# per-video features CSV (from extract_video_features_for_tta.py), then
# computes Pearson/Spearman correlations and writes plots + summary tables
# to --output-dir. CPU-only; finishes in minutes.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

trap 'echo "ERROR on line ${LINENO}: command \"${BASH_COMMAND}\" failed with exit code $?" >&2' ERR

# ============================================================================
# Path setup
# ============================================================================
SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"

# ============================================================================
# Inputs / outputs (override via --export=ALL,VAR=value when submitting)
# ============================================================================
GAINS_CSV="${GAINS_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv}"
FEATURES_CSV="${FEATURES_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation}"

# ============================================================================
# Banner
# ============================================================================
echo "=============================================================================="
echo "Per-video TTA-gain x feature correlation"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID:-<unset>}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Resolved configuration:"
echo "  USER          : ${USER}"
echo "  SCRATCH_BASE  : ${SCRATCH_BASE}"
echo "  PROJECT_ROOT  : ${PROJECT_ROOT}"
echo "  CONDA_ENV     : ${CONDA_ENV}"
echo "  GAINS_CSV     : ${GAINS_CSV}"
echo "  FEATURES_CSV  : ${FEATURES_CSV}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "=============================================================================="

# ============================================================================
# Environment setup
#
# CRITICAL: module load anaconda3/... sets PYTHONHOME / PYTHONPATH to the
# system anaconda, which shadows the conda env's sys.path. We MUST unset
# both AFTER `conda activate`, even for CPU-only jobs that depend on
# numpy/scipy/matplotlib from the conda env.
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

if [ -d "${CONDA_ENV}" ]; then
    conda activate "${CONDA_ENV}"
    echo "Activated conda: ${CONDA_ENV}"
else
    echo "ERROR: Conda environment not found at ${CONDA_ENV}" >&2
    exit 1
fi

unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found at ${PYTHON}" >&2
    exit 1
fi
echo "Using: ${PYTHON} ($("${PYTHON}" --version 2>&1))"

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "sweep_experiment/logs"

# ============================================================================
# Sanity-check inputs
# ============================================================================
if [ ! -f "${GAINS_CSV}" ]; then
    echo "ERROR: GAINS_CSV not found: ${GAINS_CSV}" >&2
    exit 1
fi
if [ ! -f "${FEATURES_CSV}" ]; then
    echo "ERROR: FEATURES_CSV not found: ${FEATURES_CSV}" >&2
    exit 1
fi

# ============================================================================
# Run correlation
# ============================================================================
echo "Starting correlation analysis..."

"${PYTHON}" scripts/correlate_tta_gain_with_features.py \
    --gains-csv "${GAINS_CSV}" \
    --features-csv "${FEATURES_CSV}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=============================================================================="
echo "Correlation analysis complete"
echo "=============================================================================="
echo "End time   : $(date)"
echo "Output dir : ${OUTPUT_DIR}"
echo "=============================================================================="
