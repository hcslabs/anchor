"""Release-gated Anchor loop from video motion to production candidate."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ProductionGates:
    maximum_policy_joint_rmse_rad: float = 0.05
    maximum_physics_joint_rmse_rad: float = 0.12
    minimum_upright_fraction: float = 0.99
    maximum_joint_limit_violations: int = 0
    minimum_grasp_success_rate: float = 0.70
    minimum_hold_s: float = 5.0


def evaluate_production_candidate(
    lineage: dict[str, Any],
    policy: dict[str, Any] | None,
    physics: dict[str, Any] | None,
    task: dict[str, Any] | None,
    gates: ProductionGates | None = None,
) -> dict[str, Any]:
    """Evaluate evidence without ever authorizing physical execution."""

    gates = gates or ProductionGates()
    checks = {
        "lineage": bool(lineage.get("technical_pass") or lineage.get("technical_lineage_complete")),
        "policy": bool(policy)
        and float(policy.get("fit", {}).get("joint_rmse_rad", float("inf"))) <= gates.maximum_policy_joint_rmse_rad,
        "physics_mapping_exact": bool(physics) and bool(physics.get("exact_embodiment_mapping")),
        "physics_tracking": bool(physics)
        and float(physics.get("metrics", {}).get("joint_tracking_rmse_rad", float("inf")))
        <= gates.maximum_physics_joint_rmse_rad,
        "physics_upright": bool(physics)
        and float(physics.get("metrics", {}).get("upright_fraction", 0.0)) >= gates.minimum_upright_fraction,
        "joint_limits": bool(physics)
        and int(physics.get("metrics", {}).get("joint_limit_violation_count", 1))
        <= gates.maximum_joint_limit_violations,
        "task_success": bool(task)
        and float(task.get("grasp_success_rate", 0.0)) >= gates.minimum_grasp_success_rate,
        "hold": bool(task) and float(task.get("minimum_hold_s", 0.0)) >= gates.minimum_hold_s,
    }
    if not checks["lineage"]:
        next_action = "repair_video_pose_trajectory_lineage"
    elif not checks["policy"]:
        next_action = "retrain_motion_prior_policy"
    elif not checks["physics_mapping_exact"]:
        next_action = "install_exact_robot_collision_asset_and_actuator_model"
    elif not checks["physics_tracking"] or not checks["physics_upright"] or not checks["joint_limits"]:
        next_action = "physics_residual_learning_with_domain_randomization"
    elif not checks["task_success"] or not checks["hold"]:
        next_action = "contact_policy_training_and_reward_revision"
    else:
        next_action = "supervised_hardware_release_review"
    return {
        "schema_version": 1,
        "kind": "anchor_production_gate_receipt_v1",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "thresholds": asdict(gates),
        "next_action": next_action,
        "production_release_ready": all(checks.values()),
        "physical_robot_execution_authorized": False,
    }

