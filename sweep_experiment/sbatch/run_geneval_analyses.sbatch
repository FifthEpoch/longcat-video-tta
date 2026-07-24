#!/bin/bash
#SBATCH --job-name=geneval_analyses
#SBATCH --output=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/geneval_analyses_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/geneval_analyses_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ============================================================================
# Step 5 (sbatch): re-run ALL vs-NOTTA analyses on the corrected data.
# VBENCH_SUBDIR=vbench_results_geneval redirects every per-video VBench consumer
# (oracle, router matrix, chart dumper) to the generated-only results.
# CPU-only. Depends on the fold job (afterok) and, ideally, the matched-FVD job.
#
# Submit:
#   sbatch sweep_experiment/sbatch/run_geneval_analyses.sbatch
# Outputs land under sweep_experiment/reports/per_video_analysis/${DATE_TAG}/.
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SERIES="${SERIES:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
OOD_CSV="${OOD_CSV:-sweep_experiment/reports/per_video_analysis/2026-07-10/diffusion_ood_scores_segment_pool.csv}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH_BASE}/conda-envs/longcat"

cd "${PROJECT_ROOT}"
OUT="sweep_experiment/reports/per_video_analysis/${DATE_TAG}"
mkdir -p "${OUT}" sweep_experiment/slurm_log

# All per-video VBench reads go to the generated-only results dir.
export VBENCH_SUBDIR=vbench_results_geneval
echo "VBENCH_SUBDIR=${VBENCH_SUBDIR}"
echo "OUT=${OUT}"

echo "=== 5a. PSNR + VBench oracle uplift (seed-clean, in-series NOTTA) ==="
python3 scripts/analyze_adasteer_budget_oracle.py --bootstrap \
  --series-root "${SERIES}" --baseline-series-root "${SERIES}" \
  --ood-csv "${OOD_CSV}" \
  --output "${OUT}/adasteer_budget_oracle_1000v_geneval.md"

echo "=== 5b. Full router matrix (A/B/C/ABC x {12,13} x {PSNR,VBench}) ==="
python3 scripts/run_router_full_matrix.py \
  --series-root "${SERIES}" --metrics psnr vbench \
  --output-dir "${OUT}/router_full_matrix_1000v_geneval"

echo "=== 5c. Chart data JSON (winner VBench dim, OOD-quintile deltas) ==="
python3 scripts/dump_pilot_chart_data.py \
  --series-root "${SERIES}" --baseline-series-root "${SERIES}" \
  --ood-csv "${OOD_CSV}" \
  > "${OUT}/chart_data_1000v_geneval.json"

echo ""
echo "Done. Artifacts:"
echo "  ${OUT}/adasteer_budget_oracle_1000v_geneval.md"
echo "  ${OUT}/router_full_matrix_1000v_geneval/"
echo "  ${OUT}/chart_data_1000v_geneval.json"
