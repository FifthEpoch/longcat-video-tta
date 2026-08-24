#!/bin/bash
# VBench only for v2v_panda_sf_family_32v. Use after rewind 16267992
# is already in the queue (do NOT resubmit rewind).
#
# afterok only on jobs still in squeue. Finished siblings are already
# on disk; including them caused "Job dependency problem" (16266882
# replacement).
#
#   bash wan_experiment/sbatch/submit_v2v_sf_family_vbench.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-v2v_panda_sf_family_32v}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"
ROLL_BASE="${ROLL_BASE:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_forward_32v/rolling_notta_h30s_shard0}"
NOTTA_DIR="${NOTTA_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0}"
REWIND_JOB="${REWIND_JOB:-16267992}"
SIBLINGS="${SIBLINGS:-16266879 16266880 16266881}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${ROOT}/sf_rewind_h30s_shard0 ${ROOT}/sf_sick_search_h30s_shard0 ${ROOT}/sf_pseudo_h30s_shard0 ${ROOT}/sf_sink_h30s_shard0 ${NOTTA_DIR} ${ROLL_BASE}"

DEPS=("${REWIND_JOB}")
for j in ${SIBLINGS}; do
    if squeue -j "${j}" -h -o '%i' 2>/dev/null | grep -q .; then
        DEPS+=("${j}")
    else
        echo "skip finished sibling ${j} (not in squeue)"
    fi
done
DEP=$(IFS=:; echo "${DEPS[*]}")

VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterok:${DEP}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterok ${DEP}"
echo "Do not resubmit rewind. 16267992 is already the resume."
