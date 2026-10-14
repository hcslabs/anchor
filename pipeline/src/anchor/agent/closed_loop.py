"""Bounded, auditable Anchor feedback loop for KHI retargeting proofs.

This is an agent control surface, not an unconstrained self-modifying system.
It extracts human motion once, evaluates several named retargeting recipes,
selects the best technically valid result, and exports KHI artifacts only when
all lineage and kinematic gates pass.  Physics feedback can be ingested to
produce the next dataset/reward action without authorizing robot execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from anchor.enrich.human_pose import extract_human_motion
from anchor.integrations.khi_g1 import (
    export_khi_object_reference,
    export_khi_robot_reference,
    validate_khi_replay_pair,
)
from anchor.retarget.embodiment import retarget_embodiment_file
from anchor.retarget.hcs_g1 import RetargetSettings


@dataclass(frozen=True)
class ClosedLoopThresholds:
    minimum_detection_ratio: float = 0.35
    minimum_frame_count: int = 5
    maximum_mean_marker_rmse_m: float = 0.020
    maximum_marker_rmse_m: float = 0.050
    maximum_velocity_limit_ratio: float = 1.0


RECIPES = (
    (
        "balanced",
        RetargetSettings(),
    ),
    (
        "stability_first",
        RetargetSettings(
            damping=0.025,
            posture_weight=0.001,
            continuity_weight=0.012,
            maximum_iterations=100,
            maximum_step_rad=0.12,
        ),
    ),
    (
        "fit_first",
        RetargetSettings(
            damping=0.012,
            posture_weight=0.0005,
            continuity_weight=0.006,
            maximum_iterations=140,
            maximum_step_rad=0.14,
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_quality(
    human_motion: dict[str, Any],
    trajectory: dict[str, Any],
    thresholds: ClosedLoopThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or ClosedLoopThresholds()
    human_quality = human_motion["quality"]
    robot_quality = trajectory["quality"]
    checks = {
        "detection_ratio": float(human_quality["detection_ratio"])
        >= thresholds.minimum_detection_ratio,
        "frame_count": int(robot_quality["frame_count"]) >= thresholds.minimum_frame_count,
        "mean_marker_rmse": float(robot_quality["mean_marker_rmse_m"])
        <= thresholds.maximum_mean_marker_rmse_m,
        "maximum_marker_rmse": float(robot_quality["maximum_marker_rmse_m"])
        <= thresholds.maximum_marker_rmse_m,
        "joint_limits": bool(robot_quality["all_joint_limits_satisfied"]),
        "velocity_limits": bool(robot_quality["all_velocity_limits_satisfied"])
        and float(robot_quality["maximum_joint_velocity_limit_ratio"])
        <= thresholds.maximum_velocity_limit_ratio,
        "uniform_timeline": bool(human_quality.get("uniform_timeline_preserved")),
        "no_premade_robot_motion": trajectory.get("uses_premade_robot_motion") is False,
    }
    score = (
        float(robot_quality["mean_marker_rmse_m"])
        + 0.25 * float(robot_quality["maximum_marker_rmse_m"])
        + 0.002 * float(robot_quality["maximum_joint_velocity_limit_ratio"])
        + 0.05 * (1.0 - min(float(human_quality["detection_ratio"]), 1.0))
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "score": score,
        "thresholds": asdict(thresholds),
    }


def interpret_physics_feedback(feedback: dict[str, Any] | None) -> dict[str, Any]:
    if not feedback:
        return {
            "feedback_present": False,
            "physics_qualified": False,
            "next_action": "run_khi_isaac_lab_physics_evaluation",
        }
    if feedback.get("kind") != "anchor_khi_physics_feedback_v1":
        raise ValueError("physics feedback must use anchor_khi_physics_feedback_v1")
    collision_count = int(feedback.get("table_collision_count", 0))
    success_rate = float(feedback.get("grasp_success_rate", 0.0))
    hold_s = float(feedback.get("minimum_hold_s", 0.0))
    if collision_count > 0:
        action = "select_or_collect_lateral_clearance_motion_and_raise_collision_gate"
    elif success_rate < 0.70 or hold_s < 5.0:
        action = "revise_containment_lift_hold_reward_priorities_then_resimulate"
    else:
        action = "promote_to_supervised_hardware_candidate"
    return {
        "feedback_present": True,
        "physics_qualified": collision_count == 0 and success_rate >= 0.70 and hold_s >= 5.0,
        "next_action": action,
        "observed": {
            "table_collision_count": collision_count,
            "grasp_success_rate": success_rate,
            "minimum_hold_s": hold_s,
        },
    }


def run_closed_loop(
    *,
    videos: list[str | Path],
    model_path: str | Path,
    urdf_path: str | Path,
    embodiment_path: str | Path,
    output_directory: str | Path,
    motion_profile: str = "khi_tabletop_right",
    sample_fps: float = 5.0,
    feedback_path: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    feedback = None
    if feedback_path is not None:
        feedback = json.loads(Path(feedback_path).read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for candidate_index, raw_video in enumerate(videos, 1):
        video = Path(raw_video).resolve()
        candidate_root = output_root / f"candidate_{candidate_index:02d}_{video.stem}"
        candidate_root.mkdir(parents=True, exist_ok=True)
        human_path = candidate_root / "human_motion_v1.json"
        human = extract_human_motion(
            video, model_path, human_path, sample_fps=sample_fps
        )
        attempts: list[dict[str, Any]] = []
        for recipe_name, settings in RECIPES:
            trajectory_path = candidate_root / f"trajectory_{recipe_name}.json"
            trajectory = retarget_embodiment_file(
                human_path,
                urdf_path,
                embodiment_path,
                trajectory_path,
                profile=motion_profile,
                settings=settings,
            )
            gate = evaluate_quality(human, trajectory)
            attempts.append(
                {
                    "recipe": recipe_name,
                    "settings": asdict(settings),
                    "trajectory_path": str(trajectory_path),
                    "trajectory_sha256": _sha256(trajectory_path),
                    "quality": trajectory["quality"],
                    "gate": gate,
                }
            )
        passing = [attempt for attempt in attempts if attempt["gate"]["passed"]]
        pool = passing or attempts
        selected = min(pool, key=lambda attempt: attempt["gate"]["score"])
        candidates.append(
            {
                "video_path": str(video),
                "video_sha256": _sha256(video),
                "human_motion_path": str(human_path),
                "human_motion_sha256": _sha256(human_path),
                "human_quality": human["quality"],
                "attempts": attempts,
                "selected_recipe": selected["recipe"],
                "selected_trajectory_path": selected["trajectory_path"],
                "selected_trajectory_sha256": selected["trajectory_sha256"],
                "passed": selected["gate"]["passed"],
                "selection_score": selected["gate"]["score"],
            }
        )

    passing_candidates = [candidate for candidate in candidates if candidate["passed"]]
    selected_candidate = (
        min(passing_candidates, key=lambda candidate: candidate["selection_score"])
        if passing_candidates
        else None
    )
    integration: dict[str, Any] = {"emitted": False}
    if selected_candidate is not None:
        robot_reference = output_root / "selected_khi_g1_reference.pkl"
        object_reference = output_root / "selected_khi_object_reference.pkl"
        robot_receipt = export_khi_robot_reference(
            selected_candidate["selected_trajectory_path"], robot_reference
        )
        object_receipt = export_khi_object_reference(robot_reference, object_reference)
        integration = {
            "emitted": True,
            "robot_reference": robot_receipt,
            "object_reference": object_receipt,
            "pair_validation": validate_khi_replay_pair(robot_reference, object_reference),
        }

    physics_response = interpret_physics_feedback(feedback)
    receipt = {
        "schema_version": 1,
        "kind": "anchor_closed_loop_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "simulation_and_non_actuating_validation_only",
        "physical_robot_commands_sent": False,
        "agent_policy": "bounded_recipe_search_with_hard_quality_gates",
        "state_transitions": [
            "human_video_ingested",
            "fixed_rate_pose_extracted",
            "embodiment_recipes_evaluated",
            "technical_quality_gated",
            "candidate_selected" if selected_candidate else "candidate_rejected",
            "khi_artifacts_emitted" if integration["emitted"] else "khi_export_blocked",
            "physics_feedback_interpreted",
        ],
        "embodiment": {
            "profile_path": str(Path(embodiment_path).resolve()),
            "profile_sha256": _sha256(Path(embodiment_path).resolve()),
            "urdf_path": str(Path(urdf_path).resolve()),
            "urdf_sha256": _sha256(Path(urdf_path).resolve()),
            "motion_profile": motion_profile,
        },
        "candidates": candidates,
        "selected_candidate": selected_candidate,
        "khi_integration": integration,
        "physics_feedback": physics_response,
        "technical_pass": selected_candidate is not None and integration["emitted"],
        "physics_pass": physics_response["physics_qualified"],
        "production_release_ready": False,
    }
    receipt_path = output_root / "anchor_closed_loop_receipt_v1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt
