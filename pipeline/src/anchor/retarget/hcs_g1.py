"""HCS morphology-aware human-to-Unitree-G1 motion retargeting.

The input is a trajectory of 3-D human landmarks reconstructed from an HCS-
owned or otherwise approved video.  The solver does not consume a pre-made
robot motion dataset.  It rebuilds robot joint motion against the KHI G1 URDF
using robot-proportioned bone targets, joint limits, temporal regularization,
and damped least-squares inverse kinematics.

GMR/GRAIL are methodological references and compatibility targets, not data
sources.  The implementation here is HCS-owned glue and optimization code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
import json
import math
from typing import Any

import numpy as np

from .kinematics import UrdfKinematics


G1_29_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


MARKER_LINKS = {
    "pelvis": "pelvis",
    "torso": "torso_link",
    "left_shoulder": "left_shoulder_pitch_link",
    "left_elbow": "left_elbow_link",
    "left_wrist": "left_rubber_hand",
    "right_shoulder": "right_shoulder_pitch_link",
    "right_elbow": "right_elbow_link",
    "right_wrist": "right_rubber_hand",
    "left_hip": "left_hip_pitch_link",
    "left_knee": "left_knee_link",
    "left_ankle": "left_ankle_roll_link",
    "right_hip": "right_hip_pitch_link",
    "right_knee": "right_knee_link",
    "right_ankle": "right_ankle_roll_link",
}


SKELETON_EDGES = [
    ("pelvis", "torso"),
    ("torso", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("torso", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


DEFAULT_WEIGHTS = {
    "torso": 2.0,
    "left_shoulder": 1.0, "right_shoulder": 1.0,
    "left_elbow": 2.0, "right_elbow": 2.0,
    "left_wrist": 4.0, "right_wrist": 4.0,
    "left_hip": 1.0, "right_hip": 1.0,
    "left_knee": 2.0, "right_knee": 2.0,
    "left_ankle": 4.0, "right_ankle": 4.0,
}


RETARGET_PROFILES = {
    "full_body": {
        "joints": G1_29_DOF_NAMES,
        "markers": [name for name in MARKER_LINKS if name != "pelvis"],
    },
    # KHI's validated tabletop system keeps the base/legs stationary while
    # the waist and right arm express the human reach trajectory.  Selecting
    # this profile prevents unreliable, occluded leg estimates in a tabletop
    # video from contaminating the grasp motion.
    "khi_tabletop_right": {
        "joints": [
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        "markers": ["torso", "right_shoulder", "right_elbow", "right_wrist"],
    },
}


@dataclass(frozen=True)
class RetargetSettings:
    damping: float = 0.015
    posture_weight: float = 0.001
    continuity_weight: float = 0.005
    maximum_iterations: int = 80
    maximum_step_rad: float = 0.16
    joint_limit_margin_rad: float = 0.02
    convergence_m: float = 0.004


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _landmarks(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    raw = frame.get("landmarks")
    if not isinstance(raw, dict):
        raise ValueError("each frame needs a landmarks object")
    converted: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        vector = np.asarray(values, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid 3-D landmark {name!r}")
        converted[name] = vector
    return converted


class G1Retargeter:
    """Retarget human 3-D landmarks to a 29-DoF Unitree G1 trajectory."""

    def __init__(
        self,
        urdf_path: str | Path,
        settings: RetargetSettings | None = None,
        profile: str = "full_body",
    ):
        self.urdf_path = Path(urdf_path).resolve()
        self.settings = settings or RetargetSettings()
        if profile not in RETARGET_PROFILES:
            raise ValueError(f"unknown retarget profile {profile!r}")
        self.profile = profile
        self.active_joints = list(RETARGET_PROFILES[profile]["joints"])
        self.active_indices = np.asarray(
            [G1_29_DOF_NAMES.index(name) for name in self.active_joints], dtype=int
        )
        self.active_markers = list(RETARGET_PROFILES[profile]["markers"])
        self.model = UrdfKinematics.from_file(self.urdf_path)
        missing = [name for name in G1_29_DOF_NAMES if name not in self.model.joint_by_name]
        if missing:
            raise ValueError(f"KHI G1 URDF is missing expected joints: {missing}")
        neutral = self.model.state()
        self.robot_neutral = {
            marker: self._marker_position(neutral, marker)
            for marker in MARKER_LINKS
        }
        self.lower = np.array([self.model.joint_by_name[name].lower for name in G1_29_DOF_NAMES])
        self.upper = np.array([self.model.joint_by_name[name].upper for name in G1_29_DOF_NAMES])
        self.velocity_limits = np.array([
            self.model.joint_by_name[name].velocity for name in G1_29_DOF_NAMES
        ])
        finite_lower = np.where(np.isfinite(self.lower), self.lower, -math.pi)
        finite_upper = np.where(np.isfinite(self.upper), self.upper, math.pi)
        margin = self.settings.joint_limit_margin_rad
        self.lower = finite_lower + margin
        self.upper = finite_upper - margin

    def _marker_position(self, state, marker: str) -> np.ndarray:
        # A human "torso" landmark is the shoulder midpoint.  Mapping it to
        # torso_link's origin would artificially lengthen both clavicles by
        # ~25 cm and make an otherwise valid pose unsatisfiable.
        if marker == "torso":
            left = state.link_transforms[MARKER_LINKS["left_shoulder"]][:3, 3]
            right = state.link_transforms[MARKER_LINKS["right_shoulder"]][:3, 3]
            return 0.5 * (left + right)
        return state.link_transforms[MARKER_LINKS[marker]][:3, 3].copy()

    def _marker_jacobian(self, state, marker: str) -> np.ndarray:
        if marker == "torso":
            left = self.model.point_jacobian(
                state, MARKER_LINKS["left_shoulder"], G1_29_DOF_NAMES
            )
            right = self.model.point_jacobian(
                state, MARKER_LINKS["right_shoulder"], G1_29_DOF_NAMES
            )
            return 0.5 * (left + right)
        return self.model.point_jacobian(
            state, MARKER_LINKS[marker], G1_29_DOF_NAMES
        )

    def _robot_proportioned_targets(self, human: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        required = {name for edge in SKELETON_EDGES for name in edge}
        missing = sorted(required - human.keys())
        if missing:
            raise ValueError(f"human frame is missing landmarks: {missing}")
        targets = {"pelvis": self.robot_neutral["pelvis"].copy()}
        for parent, child in SKELETON_EDGES:
            direction = human[child] - human[parent]
            norm = float(np.linalg.norm(direction))
            if norm < 1.0e-6:
                raise ValueError(f"degenerate human bone {parent}->{child}")
            robot_length = float(np.linalg.norm(self.robot_neutral[child] - self.robot_neutral[parent]))
            if robot_length < 1.0e-6:
                raise ValueError(f"degenerate robot marker distance {parent}->{child}")
            targets[child] = targets[parent] + direction / norm * robot_length
        return targets

    @staticmethod
    def _q_dict(q: np.ndarray) -> dict[str, float]:
        return dict(zip(G1_29_DOF_NAMES, (float(value) for value in q)))

    def _solve_frame(self, human: dict[str, np.ndarray], previous: np.ndarray) -> tuple[np.ndarray, float, int]:
        targets = self._robot_proportioned_targets(human)
        q = previous.copy()
        settings = self.settings
        marker_names = self.active_markers
        final_rmse = math.inf
        for iteration in range(1, settings.maximum_iterations + 1):
            state = self.model.state(self._q_dict(q))
            jacobian_rows: list[np.ndarray] = []
            error_rows: list[np.ndarray] = []
            squared_errors: list[float] = []
            for marker in marker_names:
                position = self._marker_position(state, marker)
                error = targets[marker] - position
                weight = math.sqrt(DEFAULT_WEIGHTS.get(marker, 1.0))
                jacobian_rows.append(weight * self._marker_jacobian(state, marker))
                error_rows.append(weight * error)
                squared_errors.append(float(error @ error))
            final_rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
            if final_rmse <= settings.convergence_m:
                break
            jacobian = np.vstack(jacobian_rows)[:, self.active_indices]
            error = np.concatenate(error_rows)

            posture_scale = math.sqrt(settings.posture_weight)
            continuity_scale = math.sqrt(settings.continuity_weight)
            active_q = q[self.active_indices]
            active_previous = previous[self.active_indices]
            dimension = len(self.active_indices)
            jacobian = np.vstack((
                jacobian,
                posture_scale * np.eye(dimension),
                continuity_scale * np.eye(dimension),
            ))
            error = np.concatenate((
                error,
                -posture_scale * active_q,
                continuity_scale * (active_previous - active_q),
            ))

            normal = jacobian.T @ jacobian
            normal += (settings.damping ** 2) * np.eye(dimension)
            delta = np.linalg.solve(normal, jacobian.T @ error)
            delta = np.clip(delta, -settings.maximum_step_rad, settings.maximum_step_rad)
            q[self.active_indices] = np.clip(
                active_q + delta,
                self.lower[self.active_indices],
                self.upper[self.active_indices],
            )
        return q, final_rmse, iteration

    def retarget_document(self, document: dict[str, Any]) -> dict[str, Any]:
        if document.get("schema_version") != 1:
            raise ValueError("human motion schema_version must be 1")
        if document.get("coordinate_system") != "x_forward_y_left_z_up_m":
            raise ValueError("coordinate_system must be x_forward_y_left_z_up_m")
        frames = document.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("human motion document must contain frames")
        fps = float(document.get("fps", 0.0))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")

        q_previous = np.zeros(len(G1_29_DOF_NAMES), dtype=float)
        output_frames: list[dict[str, Any]] = []
        errors: list[float] = []
        for index, frame in enumerate(frames):
            q, marker_rmse, iterations = self._solve_frame(_landmarks(frame), q_previous)
            output_frames.append(
                {
                    "frame_index": index,
                    "time_s": float(frame.get("time_s", index / fps)),
                    "dof_rad": [round(float(value), 10) for value in q],
                    "marker_rmse_m": marker_rmse,
                    "solver_iterations": iterations,
                }
            )
            errors.append(marker_rmse)
            q_previous = q

        source = document.get("source_video", {})
        q_matrix = np.asarray([frame["dof_rad"] for frame in output_frames], dtype=float)
        velocities = np.diff(q_matrix, axis=0) * fps if len(q_matrix) > 1 else np.zeros_like(q_matrix)
        accelerations = np.diff(velocities, axis=0) * fps if len(velocities) > 1 else np.zeros_like(velocities)
        finite_velocity_limits = np.where(
            np.isfinite(self.velocity_limits) & (self.velocity_limits > 0),
            self.velocity_limits,
            np.inf,
        )
        velocity_ratios = np.abs(velocities) / finite_velocity_limits
        return {
            "schema_version": 1,
            "kind": "anchor_hcs_g1_retarget_v1",
            "method": "robot_proportioned_landmarks_plus_damped_least_squares_ik",
            "uses_premade_robot_motion": False,
            "method_references": [
                "GMR: Retargeting Matters (ICRA 2026) — formulation reference only",
                "GRAIL (CoRL 2026) — KHI interchange compatibility reference only",
            ],
            "target_robot": "unitree_g1_29dof",
            "retarget_profile": self.profile,
            "active_joints": self.active_joints,
            "active_markers": self.active_markers,
            "joint_order": G1_29_DOF_NAMES,
            "fps": fps,
            "source_video": source,
            "input_pose_artifact": document.get("artifact"),
            "urdf": {"path": str(self.urdf_path), "sha256": _file_sha256(self.urdf_path)},
            "settings": asdict(self.settings),
            "quality": {
                "frame_count": len(output_frames),
                "mean_marker_rmse_m": float(np.mean(errors)),
                "maximum_marker_rmse_m": float(np.max(errors)),
                "maximum_joint_velocity_rad_s": float(np.max(np.abs(velocities))),
                "mean_joint_velocity_rad_s": float(np.mean(np.abs(velocities))),
                "maximum_joint_acceleration_rad_s2": float(np.max(np.abs(accelerations))),
                "maximum_joint_velocity_limit_ratio": float(np.max(velocity_ratios)),
                "all_joint_limits_satisfied": True,
                "all_velocity_limits_satisfied": bool(np.all(velocity_ratios <= 1.0)),
            },
            "frames": output_frames,
        }


def retarget_file(
    input_path: str | Path,
    urdf_path: str | Path,
    output_path: str | Path,
    *,
    profile: str = "full_body",
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    document = json.loads(input_path.read_text(encoding="utf-8"))
    document.setdefault("artifact", {"path": str(input_path), "sha256": _file_sha256(input_path)})
    result = G1Retargeter(urdf_path, profile=profile).retarget_document(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result

