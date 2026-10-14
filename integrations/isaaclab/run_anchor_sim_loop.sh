#!/usr/bin/env bash
set -euo pipefail

ROOT="${ANCHOR_ROOT:-/workspace/anchor-khi-closed-loop-20261014}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/ubuntu/IsaacLab}"
TRAJECTORY="${ANCHOR_TRAJECTORY:-$ROOT/generated/closed_loop_final/candidate_01_GRAB_official_lateral_offset_reach_grasp_lift_phone/trajectory_balanced.json}"
OUT="${ANCHOR_SIM_OUTPUT:-$ROOT/generated/isaaclab_policy_physics}"

mkdir -p "$OUT"
export PYTHONPATH="$ROOT/pipeline/src:${PYTHONPATH:-}"

"$ROOT/.venv/bin/python" -m anchor.sim_cli \
  build-surrogate \
  --source "$ROOT/khi_source/g1_29dof_pc2.urdf" \
  --output "$OUT/g1_29dof_collision_surrogate.urdf"

"$ISAACLAB_ROOT/isaaclab.sh" -p "$ROOT/integrations/isaaclab/train_motion_prior.py" \
  --trajectory "$TRAJECTORY" \
  --output "$OUT/policy" \
  --epochs "${ANCHOR_POLICY_EPOCHS:-1200}" \
  --device cuda

"$ISAACLAB_ROOT/isaaclab.sh" -p "$ROOT/integrations/isaaclab/g1_reference_playback.py" \
  --trajectory "$TRAJECTORY" \
  --policy "$OUT/policy/anchor_phase_policy.pt" \
  --policy-receipt "$OUT/policy/anchor_motion_prior_policy_receipt_v1.json" \
  --urdf "$OUT/g1_29dof_collision_surrogate.urdf" \
  --surrogate-receipt "$OUT/g1_29dof_collision_surrogate.receipt.json" \
  --output "$OUT/anchor_isaaclab_physics_playback_receipt_v1.json" \
  --headless --device cuda:0

echo "ANCHOR_SIM_LOOP_COMPLETE=$OUT"

