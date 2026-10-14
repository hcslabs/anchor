#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ANCHOR_PYTHON:-${REPOSITORY}/.venv/bin/python}"
MODEL="${ANCHOR_POSE_MODEL:-${REPOSITORY}/inputs/models/pose_landmarker_lite.task}"
URDF="${ANCHOR_KHI_URDF:-${REPOSITORY}/khi_source/g1_29dof_pc2.urdf}"
EMBODIMENT="${ANCHOR_EMBODIMENT:-${REPOSITORY}/integrations/khi_g1/unitree_g1_29dof.embodiment.json}"
OUTPUT="${ANCHOR_OUTPUT:-${REPOSITORY}/generated/closed_loop_$(date -u +%Y%m%dT%H%M%SZ)}"

VIDEOS=(
  "${REPOSITORY}/inputs/videos/GRAB_official_lateral_offset_reach_grasp_lift_phone.mp4"
  "${REPOSITORY}/inputs/videos/KIT_FruitBars_FullBody_Side_Grasp.mp4"
  "${REPOSITORY}/inputs/videos/KIT_CrackerBox_FullBody_Direct_Grasp.mp4"
)

for required in "${PYTHON}" "${MODEL}" "${URDF}" "${EMBODIMENT}" "${VIDEOS[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "MISSING_REQUIRED_INPUT=${required}" >&2
    exit 2
  fi
done

arguments=(
  -m anchor.cli closed-loop-khi
  --model "${MODEL}"
  --urdf "${URDF}"
  --embodiment "${EMBODIMENT}"
  --profile khi_tabletop_right
  --sample-fps 5
  --output-dir "${OUTPUT}"
)
for video in "${VIDEOS[@]}"; do
  arguments+=(--video "${video}")
done
if [[ -n "${ANCHOR_PHYSICS_FEEDBACK:-}" ]]; then
  arguments+=(--feedback-json "${ANCHOR_PHYSICS_FEEDBACK}")
fi

export PYTHONPATH="${REPOSITORY}/pipeline/src${PYTHONPATH:+:${PYTHONPATH}}"
printf 'ANCHOR_EXECUTION_SCOPE=simulation_and_non_actuating_validation_only\n'
printf 'ANCHOR_OUTPUT=%s\n' "${OUTPUT}"
"${PYTHON}" "${arguments[@]}"
printf 'ANCHOR_RECEIPT=%s\n' "${OUTPUT}/anchor_closed_loop_receipt_v1.json"
