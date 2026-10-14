"""Profile-driven human-to-humanoid morphology retargeting.

The solver is intentionally robot-agnostic.  A versioned embodiment profile
binds human landmarks to URDF links, defines the robot joint order, and names
the active joint/marker subsets.  Adding a new humanoid therefore requires a
verified URDF and profile, not a fork of the optimization code.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import math
from typing import Any

import numpy as np

from .hcs_g1 import RetargetSettings
from .kinematics import UrdfKinematics


def _sha256(path: Path) -> str:
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


def load_embodiment_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).resolve()
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("kind") != "anchor_embodiment_profile_v1":
        raise ValueError("not an Anchor embodiment profile v1")
    if document.get("status") != "validated":
        raise ValueError(
            f"embodiment profile {document.get('robot_id', '<unknown>')!r} is not validated; "
            "install the vendor URDF and verified joint/link mapping first"
        )
    required = ("robot_id", "joint_order", "marker_links", "skeleton_edges", "profiles")
    missing = [name for name in required if not document.get(name)]
    if missing:
        raise ValueError(f"embodiment profile is missing required fields: {missing}")
    document["_profile_path"] = str(profile_path)
    document["_profile_sha256"] = _sha256(profile_path)
    return document


class MorphologyRetargeter:
    """Retarget body-aligned human landmarks using a data-defined embodiment."""

    def __init__(
        self,
        urdf_path: str | Path,
        embodiment_path: str | Path,
        *,
        profile: str,
        settings: RetargetSettings | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        self.embodiment = load_embodiment_profile(embodiment_path)
        self.settings = settings or RetargetSettings()
        self.model = UrdfKinematics.from_file(self.urdf_path)
        self.joint_order = list(self.embodiment["joint_order"])
        profile_document = self.embodiment["profiles"].get(profile)
        if not isinstance(profile_document, dict):
            raise ValueError(f"unknown embodiment motion profile {profile!r}")
        self.profile = profile
        self.active_joints = list(profile_document["joints"])
        self.active_markers = list(profile_document["markers"])
        self.weights = {
            str(name): float(value)
            for name, value in self.embodiment.get("marker_weights", {}).items()
        }
        self.edges = [tuple(edge) for edge in self.embodiment["skeleton_edges"]]
        self.marker_links = self.embodiment["marker_links"]

        missing_joints = [name for name in self.joint_order if name not in self.model.joint_by_name]
        if missing_joints:
            raise ValueError(f"URDF is missing embodiment joints: {missing_joints}")
        if not set(self.active_joints).issubset(self.joint_order):
            raise ValueError("active joints must be a subset of joint_order")
        self.active_indices = np.asarray(
            [self.joint_order.index(name) for name in self.active_joints], dtype=int
        )
        neutral_state = self.model.state()
        available_links = set(neutral_state.link_transforms)
        missing_links = sorted(
            {
                link
                for marker in self.marker_links.values()
                for link in self._links(marker)
                if link not in available_links
            }
        )
        if missing_links:
            raise ValueError(f"URDF is missing embodiment marker links: {missing_links}")
        required_markers = {name for edge in self.edges for name in edge}
        if not required_markers.issubset(self.marker_links):
            raise ValueError("marker_links does not cover every skeleton edge")
        if not set(self.active_markers).issubset(required_markers):
            raise ValueError("active markers are not in the embodiment skeleton")

        self.robot_neutral = {
            marker: self._marker_position(neutral_state, marker) for marker in self.marker_links
        }
        self.lower = np.asarray(
            [self.model.joint_by_name[name].lower for name in self.joint_order], dtype=float
        )
        self.upper = np.asarray(
            [self.model.joint_by_name[name].upper for name in self.joint_order], dtype=float
        )
        self.velocity_limits = np.asarray(
            [self.model.joint_by_name[name].velocity for name in self.joint_order], dtype=float
        )
        finite_lower = np.where(np.isfinite(self.lower), self.lower, -math.pi)
        finite_upper = np.where(np.isfinite(self.upper), self.upper, math.pi)
        margin = self.settings.joint_limit_margin_rad
        self.lower = finite_lower + margin
        self.upper = finite_upper - margin

    @staticmethod
    def _links(specification: Any) -> list[str]:
        if isinstance(specification, str):
            return [specification]
        if isinstance(specification, dict) and specification.get("mode") == "centroid":
            links = specification.get("links")
            if isinstance(links, list) and links and all(isinstance(link, str) for link in links):
                return links
        raise ValueError(f"invalid marker link specification: {specification!r}")

    def _marker_position(self, state, marker: str) -> np.ndarray:
        links = self._links(self.marker_links[marker])
        points = [state.link_transforms[link][:3, 3] for link in links]
        return np.mean(points, axis=0)

    def _marker_jacobian(self, state, marker: str) -> np.ndarray:
        links = self._links(self.marker_links[marker])
        jacobians = [
            self.model.point_jacobian(state, link, self.joint_order) for link in links
        ]
        return np.mean(jacobians, axis=0)

    def _q_dict(self, q: np.ndarray) -> dict[str, float]:
        return dict(zip(self.joint_order, (float(value) for value in q)))

    def _robot_proportioned_targets(
        self, human: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        required = {name for edge in self.edges for name in edge}
        missing = sorted(required - human.keys())
        if missing:
            raise ValueError(f"human frame is missing landmarks: {missing}")
        root_marker = self.edges[0][0]
        targets = {root_marker: self.robot_neutral[root_marker].copy()}
        for parent, child in self.edges:
            if parent not in targets:
                raise ValueError(f"skeleton_edges are not parent-before-child at {parent}->{child}")
            direction = human[child] - human[parent]
            human_length = float(np.linalg.norm(direction))
            if human_length < 1.0e-6:
                raise ValueError(f"degenerate human bone {parent}->{child}")
            robot_length = float(
                np.linalg.norm(self.robot_neutral[child] - self.robot_neutral[parent])
            )
            if robot_length < 1.0e-6:
                raise ValueError(f"degenerate robot marker distance {parent}->{child}")
            targets[child] = targets[parent] + direction / human_length * robot_length
        return targets

    def _solve_frame(
        self, human: dict[str, np.ndarray], previous: np.ndarray
    ) -> tuple[np.ndarray, float, int]:
        targets = self._robot_proportioned_targets(human)
        q = previous.copy()
        settings = self.settings
        final_rmse = math.inf
        for iteration in range(1, settings.maximum_iterations + 1):
            state = self.model.state(self._q_dict(q))
            jacobian_rows: list[np.ndarray] = []
            error_rows: list[np.ndarray] = []
            squared_errors: list[float] = []
            for marker in self.active_markers:
                error = targets[marker] - self._marker_position(state, marker)
                weight = math.sqrt(self.weights.get(marker, 1.0))
                jacobian_rows.append(weight * self._marker_jacobian(state, marker))
                error_rows.append(weight * error)
                squared_errors.append(float(error @ error))
            final_rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
            if final_rmse <= settings.convergence_m:
                break
            jacobian = np.vstack(jacobian_rows)[:, self.active_indices]
            error = np.concatenate(error_rows)
            active_q = q[self.active_indices]
            active_previous = previous[self.active_indices]
            dimension = len(self.active_indices)
            posture_scale = math.sqrt(settings.posture_weight)
            continuity_scale = math.sqrt(settings.continuity_weight)
            jacobian = np.vstack(
                (
                    jacobian,
                    posture_scale * np.eye(dimension),
                    continuity_scale * np.eye(dimension),
                )
            )
            error = np.concatenate(
                (
                    error,
                    -posture_scale * active_q,
                    continuity_scale * (active_previous - active_q),
                )
            )
            normal = jacobian.T @ jacobian
            normal += (settings.damping**2) * np.eye(dimension)
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

        previous = np.zeros(len(self.joint_order), dtype=float)
        output_frames: list[dict[str, Any]] = []
        errors: list[float] = []
        for index, frame in enumerate(frames):
            q, marker_rmse, iterations = self._solve_frame(_landmarks(frame), previous)
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
            previous = q

        q_matrix = np.asarray([frame["dof_rad"] for frame in output_frames], dtype=float)
        velocities = np.diff(q_matrix, axis=0) * fps if len(q_matrix) > 1 else np.zeros_like(q_matrix)
        accelerations = (
            np.diff(velocities, axis=0) * fps if len(velocities) > 1 else np.zeros_like(velocities)
        )
        finite_velocity_limits = np.where(
            np.isfinite(self.velocity_limits) & (self.velocity_limits > 0),
            self.velocity_limits,
            np.inf,
        )
        velocity_ratios = np.abs(velocities) / finite_velocity_limits
        return {
            "schema_version": 1,
            "kind": "anchor_morphology_retarget_v1",
            "method": "robot_proportioned_landmarks_plus_damped_least_squares_ik",
            "uses_premade_robot_motion": False,
            "target_robot": self.embodiment["robot_id"],
            "embodiment_profile": {
                "path": self.embodiment["_profile_path"],
                "sha256": self.embodiment["_profile_sha256"],
            },
            "retarget_profile": self.profile,
            "active_joints": self.active_joints,
            "active_markers": self.active_markers,
            "joint_order": self.joint_order,
            "fps": fps,
            "source_video": document.get("source_video", {}),
            "input_pose_artifact": document.get("artifact"),
            "urdf": {"path": str(self.urdf_path), "sha256": _sha256(self.urdf_path)},
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


def retarget_embodiment_file(
    input_path: str | Path,
    urdf_path: str | Path,
    embodiment_path: str | Path,
    output_path: str | Path,
    *,
    profile: str,
    settings: RetargetSettings | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    document = json.loads(input_path.read_text(encoding="utf-8"))
    document.setdefault("artifact", {"path": str(input_path), "sha256": _sha256(input_path)})
    result = MorphologyRetargeter(
        urdf_path, embodiment_path, profile=profile, settings=settings
    ).retarget_document(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
