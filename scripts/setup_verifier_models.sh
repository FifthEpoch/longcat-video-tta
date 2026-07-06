#!/usr/bin/env bash
# One-time setup for learned verifier scoring (Options 1–2).
#
#   bash scripts/setup_verifier_models.sh
#
# Installs Python deps and clones third-party repos under /scratch/$USER/third_party/.
set -euo pipefail

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USER}}"
THIRD="${THIRD:-${SCRATCH_BASE}/third_party}"
ENV_PATH="${ENV_PATH:-${SCRATCH_BASE}/conda-envs/longcat}"

mkdir -p "${THIRD}"

echo "=== Verifier model setup ==="
echo "THIRD_PARTY=${THIRD}"

module purge 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "${ENV_PATH}" 2>/dev/null || echo "WARN: conda env ${ENV_PATH} not found; using current python"
fi

pip install -q --upgrade pip
pip install -q transformers accelerate av pillow einops
pip install -q "mantis-llava @ git+https://github.com/TIGER-AI-Lab/Mantis.git" || {
  echo "WARN: mantis-llava install failed — VideoScore backend may not work"
}

# VideoAlign / VideoReward (Option 2)
if [ ! -d "${THIRD}/VideoAlign" ]; then
  git clone --depth=1 https://github.com/KlingAIResearch/VideoAlign.git "${THIRD}/VideoAlign"
fi
export VIDEOALIGN_ROOT="${THIRD}/VideoAlign"
if [ ! -d "${VIDEOALIGN_ROOT}/checkpoints" ]; then
  echo ""
  echo "Download VideoReward checkpoint into ${VIDEOALIGN_ROOT}/checkpoints"
  echo "  huggingface-cli download KlingTeam/VideoReward --local-dir ${VIDEOALIGN_ROOT}/checkpoints"
fi

# VisionReward (optional, slow)
if [ ! -d "${THIRD}/VisionReward" ]; then
  git clone --depth=1 https://github.com/THUDM/VisionReward.git "${THIRD}/VisionReward" || true
fi
export VISIONREWARD_ROOT="${THIRD}/VisionReward"

cat <<EOF

Setup complete. Export before scoring:

  export VIDEOALIGN_ROOT=${VIDEOALIGN_ROOT}
  export VISIONREWARD_ROOT=${VISIONREWARD_ROOT}
  export VIDEOSCORE_MODEL=TIGER-Lab/VideoScore

Then submit:
  bash sweep_experiment/sbatch/submit_verifier_options_pilot.sh

EOF
