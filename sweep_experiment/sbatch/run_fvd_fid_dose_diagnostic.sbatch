#!/bin/bash
#SBATCH --job-name=fvd_dose_diag
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/fvd_dose_diag_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/fvd_dose_diag_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h200:1

# ============================================================================
# FVD/FID dose-response diagnostic for the "fixed TTA triples FVD" puzzle.
#
# For a ladder of arms spanning TTA dose (steps x LR), compute FVD **and** FID
# directly from each arm's own videos/ dir vs the SAME frozen preview GT cache,
# same gen-only window ([14:28]). eval_fvd now also reports the FVD split:
#   FVD = mean_term (||mu_gen-mu_ref||^2)  +  trace_term (covariance mismatch).
#
# This answers:
#   (a) Does running on videos/ directly reproduce the matched fixed=216.7?
#       -> if yes, the symlink policy layer is exonerated (effect is in the clips).
#   (b) Does FVD scale monotonically with TTA dose?
#       -> monotonic dose-response = real distribution shift;
#          all-TTA-high-regardless = systematic artifact to chase.
#   (c) Is the blow-up in the MEAN term (systematic feature shift, consistent
#       with flat PSNR) or the TRACE term (variance/covariance change)?
#   (d) Does per-frame FID move too, or only temporal FVD?
#
# Submit:
#   sbatch sweep_experiment/sbatch/run_fvd_fid_dose_diagnostic.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SERIES="${SERIES:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"
OUT_ROOT="${OUT_ROOT:-sweep_experiment/reports/budget_oracle_fvd_1000v_preview/dose_diagnostic}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_GEN_FRAMES="${NUM_GEN_FRAMES:-14}"

# Dose ladder: baseline + (gentle -> aggressive) across steps and LR.
ARMS="${ARMS:-NOTTA S2_LR1e3 S2_LR1e2 S10_LR5e3 S20_LR1e3 S20_LR1e2}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH_BASE}/conda-envs/longcat"
export LD_LIBRARY_PATH="${SCRATCH_BASE}/conda-envs/longcat/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_ROOT}" sweep_experiment/slurm_log

echo "============================================================"
echo "FVD/FID dose diagnostic"
echo "  series   : ${SERIES}"
echo "  gt_cache : ${GT_CACHE}"
echo "  arms     : ${ARMS}"
echo "  window   : cond=${NUM_COND_FRAMES} gen=${NUM_GEN_FRAMES} (tail [14:28])"
echo "============================================================"

[ -f "${GT_CACHE}" ] || { echo "ERROR: GT cache missing: ${GT_CACHE}" >&2; exit 2; }

for a in ${ARMS}; do
    gen_dir="${SERIES}/${a}/videos"
    # videos/ live under chunk_*/videos — flatten via a temp symlink dir if needed.
    if [ ! -d "${gen_dir}" ]; then
        flat="${OUT_ROOT}/_flat_${a}/videos"
        rm -rf "${OUT_ROOT}/_flat_${a}"; mkdir -p "${flat}"
        n=0
        while IFS= read -r f; do
            ln -sf "$(readlink -f "$f")" "${flat}/$(basename "$f")"; n=$((n+1))
        done < <(find "${SERIES}/${a}" -path '*/videos/*.mp4' 2>/dev/null)
        gen_dir="${flat}"
        echo "[${a}] flattened ${n} mp4s -> ${gen_dir}"
    fi
    out_json="${OUT_ROOT}/${a}_fvd_fid.json"
    echo "=== ${a} ==="
    python sweep_experiment/scripts/eval_fvd.py \
        --gen-dir "${gen_dir}" \
        --gt-cache "${GT_CACHE}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-gen-frames "${NUM_GEN_FRAMES}" \
        --compute-fid --force \
        --min-videos 256 \
        --output "${out_json}" \
        --device cuda
done

echo ""
echo "===== DOSE-RESPONSE SUMMARY ====="
python3 - "${OUT_ROOT}" ${ARMS} <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); arms = sys.argv[2:]
print(f"{'arm':<12} {'N':>5} {'FVD':>10} {'mean_term':>11} {'trace_term':>11} {'FID':>9}")
for a in arms:
    p = root / f"{a}_fvd_fid.json"
    if not p.is_file():
        print(f"{a:<12} (no json)"); continue
    b = json.loads(p.read_text())
    print(f"{a:<12} {b.get('num_valid_pairs',0):>5} "
          f"{b.get('fvd',float('nan')):>10.3f} "
          f"{(b.get('fvd_mean_term') or float('nan')):>11.3f} "
          f"{(b.get('fvd_trace_term') or float('nan')):>11.3f} "
          f"{(b.get('fid') if b.get('fid') is not None else float('nan')):>9.3f}")
print()
print("Read: if FVD rises monotonically with dose (NOTTA<S2<...<S20) it's a real")
print("distribution shift; if mean_term dominates, it's a systematic feature shift")
print("(consistent with flat per-video PSNR); if FID tracks FVD, it's per-frame too.")
PY
echo "Per-arm JSON under: ${OUT_ROOT}/"
