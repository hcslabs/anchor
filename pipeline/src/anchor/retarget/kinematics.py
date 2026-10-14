"""Small, dependency-light URDF forward kinematics used by Anchor retargeting.

This module intentionally depends only on NumPy and the Python standard
library.  It reads the target robot's own URDF, so the retargeter is bound to
the same joint axes, link lengths, and limits as the KHI deployment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np


def _numbers(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(value) for value in text.split()], dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    return transform


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float


@dataclass
class KinematicState:
    link_transforms: dict[str, np.ndarray]
    joint_positions: dict[str, np.ndarray]
    joint_axes: dict[str, np.ndarray]


class UrdfKinematics:
    """URDF tree with FK and analytic point Jacobians."""

    def __init__(self, joints: list[Joint], root_link: str):
        self.joints = joints
        self.root_link = root_link
        self.joint_by_name = {joint.name: joint for joint in joints}
        self.parent_joint_by_link = {joint.child: joint for joint in joints}
        children: dict[str, list[Joint]] = {}
        for joint in joints:
            children.setdefault(joint.parent, []).append(joint)
        self._children = children
        self.ordered_joints = self._topological_joints()

    @classmethod
    def from_file(cls, path: str | Path) -> "UrdfKinematics":
        root = ET.parse(Path(path)).getroot()
        links = {node.attrib["name"] for node in root.findall("link")}
        joints: list[Joint] = []
        children: set[str] = set()
        for node in root.findall("joint"):
            parent = node.find("parent").attrib["link"]
            child = node.find("child").attrib["link"]
            origin_node = node.find("origin")
            xyz = _numbers(origin_node.get("xyz") if origin_node is not None else None, (0, 0, 0))
            rpy = _numbers(origin_node.get("rpy") if origin_node is not None else None, (0, 0, 0))
            axis_node = node.find("axis")
            axis = _numbers(axis_node.get("xyz") if axis_node is not None else None, (1, 0, 0))
            limit_node = node.find("limit")
            lower = float(limit_node.get("lower", "-inf")) if limit_node is not None else -math.inf
            upper = float(limit_node.get("upper", "inf")) if limit_node is not None else math.inf
            velocity = float(limit_node.get("velocity", "inf")) if limit_node is not None else math.inf
            joints.append(
                Joint(
                    name=node.attrib["name"],
                    kind=node.attrib.get("type", "fixed"),
                    parent=parent,
                    child=child,
                    origin=_origin_matrix(xyz, rpy),
                    axis=axis,
                    lower=lower,
                    upper=upper,
                    velocity=velocity,
                )
            )
            children.add(child)
        roots = sorted(links - children)
        if len(roots) != 1:
            raise ValueError(f"URDF must have one root link, found {roots}")
        return cls(joints, roots[0])

    def _topological_joints(self) -> list[Joint]:
        ordered: list[Joint] = []

        def visit(link: str) -> None:
            for joint in self._children.get(link, []):
                ordered.append(joint)
                visit(joint.child)

        visit(self.root_link)
        if len(ordered) != len(self.joints):
            raise ValueError("URDF contains disconnected or cyclic joints")
        return ordered

    def state(self, q: dict[str, float] | None = None) -> KinematicState:
        q = q or {}
        links = {self.root_link: np.eye(4, dtype=float)}
        positions: dict[str, np.ndarray] = {}
        axes: dict[str, np.ndarray] = {}
        for joint in self.ordered_joints:
            joint_frame = links[joint.parent] @ joint.origin
            positions[joint.name] = joint_frame[:3, 3].copy()
            axes[joint.name] = joint_frame[:3, :3] @ joint.axis
            motion = np.eye(4, dtype=float)
            value = float(q.get(joint.name, 0.0))
            if joint.kind in {"revolute", "continuous"}:
                motion = _axis_angle(joint.axis, value)
            elif joint.kind == "prismatic":
                motion[:3, 3] = joint.axis * value
            links[joint.child] = joint_frame @ motion
        return KinematicState(links, positions, axes)

    def ancestors(self, link: str) -> set[str]:
        names: set[str] = set()
        current = link
        while current != self.root_link:
            joint = self.parent_joint_by_link.get(current)
            if joint is None:
                raise KeyError(f"unknown or disconnected link: {link}")
            names.add(joint.name)
            current = joint.parent
        return names

    def point_jacobian(
        self,
        state: KinematicState,
        link: str,
        active_joints: list[str],
        point_local: np.ndarray | None = None,
    ) -> np.ndarray:
        transform = state.link_transforms[link]
        point = transform[:3, 3] if point_local is None else (transform @ np.r_[point_local, 1.0])[:3]
        ancestors = self.ancestors(link)
        jacobian = np.zeros((3, len(active_joints)), dtype=float)
        for column, name in enumerate(active_joints):
            if name not in ancestors:
                continue
            joint = self.joint_by_name[name]
            axis = state.joint_axes[name]
            if joint.kind in {"revolute", "continuous"}:
                jacobian[:, column] = np.cross(axis, point - state.joint_positions[name])
            elif joint.kind == "prismatic":
                jacobian[:, column] = axis
        return jacobian

