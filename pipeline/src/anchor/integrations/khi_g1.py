"""KHI G1 reference exporter and compatibility validator."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
from typing import Any

import numpy as np

from anchor.retarget.hcs_g1 import G1_29_DOF_NAMES


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_khi_robot_reference(
    trajectory_path: str | Path,
    output_path: str | Path,
    *,
    root_height_m: float = 0.793,
) -> dict[str, Any]:
    """Export Anchor G1 JSON to the joblib layout consumed by KHI Isaac Lab.

    No hand primitive is inferred: KHI's learned RH56 policy controls the hand.
    The zero ``hand_action_right`` field keeps the legacy physics replay loader
    schema-compatible while clearly representing an open hand.
    """
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib is required for KHI reference export") from error

    trajectory_path = Path(trajectory_path).resolve()
    output_path = Path(output_path).resolve()
    document = json.loads(trajectory_path.read_text(encoding="utf-8"))
    supported_kinds = {"anchor_hcs_g1_retarget_v1", "anchor_morphology_retarget_v1"}
    if document.get("kind") not in supported_kinds:
        raise ValueError("input is not a supported Anchor G1 trajectory")
    if document.get("target_robot") not in {None, "unitree_g1_29dof"}:
        raise ValueError("KHI G1 export refuses a non-G1 embodiment")
    if document.get("joint_order") != G1_29_DOF_NAMES:
        raise ValueError("trajectory joint order does not match KHI G1_29_DOF_NAMES")
    if document.get("uses_premade_robot_motion") is not False:
        raise ValueError("lineage must explicitly exclude pre-made robot motion")
    dof = np.asarray([frame["dof_rad"] for frame in document["frames"]], dtype=np.float32)
    if dof.ndim != 2 or dof.shape[1] != 29 or not np.isfinite(dof).all():
        raise ValueError(f"expected finite Nx29 trajectory, got {dof.shape}")
    fps = float(document["fps"])
    frames = dof.shape[0]
    key = "anchor_" + _sha256(trajectory_path)[:16]
    record = {
        key: {
            "dof": dof,
            "hand_action_right": np.zeros(frames, dtype=np.float32),
            "root_trans_offset": np.tile(
                np.array([0.0, 0.0, root_height_m], dtype=np.float32), (frames, 1)
            ),
            "root_rot": np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)
            ),
            "fps": fps,
            "_anchor_lineage": {
                "schema_version": 1,
                "source_trajectory_path": str(trajectory_path),
                "source_trajectory_sha256": _sha256(trajectory_path),
                "source_video": document.get("source_video"),
                "urdf": document.get("urdf"),
                "retarget_profile": document.get("retarget_profile"),
                "uses_premade_robot_motion": False,
                "hand_control": "KHI RH56 policy; not inferred by this exporter",
            },
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(record, output_path, compress=3)
    loaded = joblib.load(output_path)
    loaded_record = loaded[key]
    if np.asarray(loaded_record["dof"]).shape != (frames, 29):
        raise RuntimeError("KHI joblib round-trip changed the trajectory shape")
    return {
        "schema_version": 1,
        "kind": "anchor_khi_g1_export_receipt_v1",
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "motion_key": key,
        "frame_count": frames,
        "joint_count": 29,
        "fps": fps,
        "joblib_round_trip": True,
        "uses_premade_robot_motion": False,
        "source_trajectory_sha256": _sha256(trajectory_path),
    }


def validate_khi_robot_reference(path: str | Path) -> dict[str, Any]:
    import joblib

    path = Path(path).resolve()
    document = joblib.load(path)
    if not isinstance(document, dict) or len(document) != 1:
        raise ValueError("KHI robot reference needs exactly one motion key")
    key = next(iter(document))
    record = document[key]
    dof = np.asarray(record["dof"])
    if dof.ndim != 2 or dof.shape[1] != 29 or not np.isfinite(dof).all():
        raise ValueError("KHI reference dof must be a finite Nx29 array")
    for name, width in (("root_trans_offset", 3), ("root_rot", 4)):
        values = np.asarray(record[name])
        if values.shape != (dof.shape[0], width) or not np.isfinite(values).all():
            raise ValueError(f"invalid {name} shape")
    primitive = np.asarray(record["hand_action_right"])
    if primitive.shape != (dof.shape[0],):
        raise ValueError("invalid hand_action_right shape")
    lineage = record.get("_anchor_lineage", {})
    if lineage.get("uses_premade_robot_motion") is not False:
        raise ValueError("KHI reference lacks Anchor no-pre-made-motion lineage")
    return {
        "valid": True,
        "path": str(path),
        "sha256": _sha256(path),
        "motion_key": key,
        "frame_count": int(dof.shape[0]),
        "joint_count": int(dof.shape[1]),
        "fps": float(record["fps"]),
        "lineage": lineage,
    }


def export_khi_object_reference(
    robot_reference_path: str | Path,
    output_path: str | Path,
    *,
    relative_position_m: tuple[float, float, float] = (0.487, -0.180, -0.089),
) -> dict[str, Any]:
    """Create the static free-cube initialization paired to an Anchor motion.

    The KHI replay requires robot and object joblib files with the same motion
    key.  This companion contains no robot motion and does not attach or drive
    the object; KHI uses its first pose only to initialize the free cube.
    """
    import joblib

    robot_path = Path(robot_reference_path).resolve()
    output_path = Path(output_path).resolve()
    robot = joblib.load(robot_path)
    if not isinstance(robot, dict) or len(robot) != 1:
        raise ValueError("KHI robot reference needs exactly one motion key")
    key = next(iter(robot))
    robot_record = robot[key]
    root = np.asarray(robot_record["root_trans_offset"], dtype=np.float32)
    if root.ndim != 2 or root.shape[1] != 3 or not np.isfinite(root).all():
        raise ValueError("robot root_trans_offset must be a finite Nx3 array")
    relative = np.asarray(relative_position_m, dtype=np.float32)
    if relative.shape != (3,) or not np.isfinite(relative).all():
        raise ValueError("relative_position_m must contain three finite values")
    positions = (root + relative).reshape(root.shape[0], 1, 3)
    rotations = np.tile(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (root.shape[0], 1, 1),
    )
    object_reference = {
        key: {
            "root_pos": positions,
            "root_rot": rotations,
            "fps": float(robot_record["fps"]),
            "_anchor_lineage": {
                "schema_version": 1,
                "robot_reference_path": str(robot_path),
                "robot_reference_sha256": _sha256(robot_path),
                "initialization_only": True,
                "free_object": True,
                "object_pose_commands_after_reset": False,
                "relative_position_m": relative.astype(float).tolist(),
            },
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(object_reference, output_path, compress=3)
    validation = validate_khi_replay_pair(robot_path, output_path)
    return {
        "schema_version": 1,
        "kind": "anchor_khi_object_export_receipt_v1",
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "motion_key": key,
        "frame_count": int(root.shape[0]),
        "relative_position_m": relative.astype(float).tolist(),
        "free_object": True,
        "object_pose_commands_after_reset": False,
        "pair_validation": validation,
    }


def validate_khi_replay_pair(
    robot_reference_path: str | Path,
    object_reference_path: str | Path,
) -> dict[str, Any]:
    """Validate the fields read by KHI ``aprilcube_inspire_grail_replay``."""
    import joblib

    robot_path = Path(robot_reference_path).resolve()
    object_path = Path(object_reference_path).resolve()
    robot = joblib.load(robot_path)
    object_document = joblib.load(object_path)
    if not isinstance(robot, dict) or len(robot) != 1:
        raise ValueError("KHI robot reference needs exactly one motion key")
    key = next(iter(robot))
    if not isinstance(object_document, dict) or key not in object_document:
        raise ValueError(f"motion key {key!r} missing from object reference")
    robot_record = robot[key]
    object_record = object_document[key]
    dof = np.asarray(robot_record["dof"])
    primitive = np.asarray(robot_record["hand_action_right"])
    root = np.asarray(robot_record["root_trans_offset"])
    root_rot = np.asarray(robot_record["root_rot"])
    object_pos = np.asarray(object_record["root_pos"])
    if dof.ndim != 2 or dof.shape[1] != 29:
        raise ValueError("robot dof must be Nx29")
    frame_count = dof.shape[0]
    if primitive.shape != (frame_count,):
        raise ValueError("hand_action_right must contain one value per frame")
    if root.shape != (frame_count, 3) or root_rot.shape != (frame_count, 4):
        raise ValueError("robot root arrays do not match the motion length")
    if object_pos.ndim != 3 or object_pos.shape[0] < 1 or object_pos.shape[1:] != (1, 3):
        raise ValueError("object root_pos must be Nx1x3")
    for values in (dof, primitive, root, root_rot, object_pos):
        if not np.isfinite(values).all():
            raise ValueError("KHI replay pair contains non-finite values")
    relative = object_pos[0, 0] - root[0]
    return {
        "valid": True,
        "robot_reference_path": str(robot_path),
        "robot_reference_sha256": _sha256(robot_path),
        "object_reference_path": str(object_path),
        "object_reference_sha256": _sha256(object_path),
        "motion_key": key,
        "frame_count": int(frame_count),
        "initial_relative_object_position_m": relative.astype(float).tolist(),
        "khi_replay_fields_present": True,
    }

