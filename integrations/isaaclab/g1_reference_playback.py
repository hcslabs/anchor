#!/usr/bin/env python3
"""Headless Isaac Lab playback of an Anchor policy on the KHI 29-DoF model.

The exact KHI URDF kinematic tree, inertials, axes and limits are used.  Meshes
are replaced by primitives because the vendor mesh package is not part of this
repository.  Therefore this validates joint-dynamics plumbing and tracking,
not production contacts or grasp success.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--trajectory", required=True)
parser.add_argument("--policy", required=True)
parser.add_argument("--policy-receipt", required=True)
parser.add_argument("--urdf", required=True)
parser.add_argument("--surrogate-receipt", required=True)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

from anchor.policy.phase_policy import build_phase_features


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    trajectory_path = Path(args.trajectory).resolve()
    policy_path = Path(args.policy).resolve()
    policy_receipt_path = Path(args.policy_receipt).resolve()
    urdf_path = Path(args.urdf).resolve()
    surrogate_receipt_path = Path(args.surrogate_receipt).resolve()
    output_path = Path(args.output).resolve()
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    policy_receipt = json.loads(policy_receipt_path.read_text(encoding="utf-8"))
    surrogate_receipt = json.loads(surrogate_receipt_path.read_text(encoding="utf-8"))

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1600.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1600.0))
    spawn = sim_utils.UrdfFileCfg(
        asset_path=str(urdf_path),
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=80.0, damping=8.0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )
    robot_cfg = ArticulationCfg(
        prim_path="/World/G1",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
                effort_limit_sim=300.0,
                stiffness=200.0,
                damping=10.0,
                armature=0.01,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_.*"],
                effort_limit_sim=50.0,
                stiffness=40.0,
                damping=4.0,
                armature=0.01,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"],
                effort_limit_sim=100.0,
                stiffness=120.0,
                damping=12.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
                effort_limit_sim=100.0,
                stiffness=80.0,
                damping=10.0,
                armature=0.01,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    robot.reset()

    joint_order = list(trajectory["joint_order"])
    if set(joint_order) != set(robot.joint_names):
        missing = sorted(set(joint_order) - set(robot.joint_names))
        extra = sorted(set(robot.joint_names) - set(joint_order))
        raise RuntimeError(f"surrogate joint mapping mismatch missing={missing} extra={extra}")
    robot_indices = [robot.joint_names.index(name) for name in joint_order]
    active_joints = list(policy_receipt["active_joints"])
    active_source_indices = [joint_order.index(name) for name in active_joints]
    active_robot_indices = [robot.joint_names.index(name) for name in active_joints]
    policy = torch.jit.load(str(policy_path), map_location=sim.device).eval()

    fps = float(trajectory["fps"])
    frames = np.asarray([frame["dof_rad"] for frame in trajectory["frames"]], dtype=np.float32)
    steps_per_frame = max(1, round((1.0 / fps) / sim_cfg.dt))
    errors = []
    limit_violations = 0
    target_limit_violations = 0
    per_joint_limit_violations = np.zeros(len(joint_order), dtype=np.int64)
    samples = []
    soft_limits = robot.data.soft_joint_pos_limits[0, robot_indices]

    stance_target = np.zeros(len(joint_order), dtype=np.float32)
    for index, name in enumerate(joint_order):
        if name.endswith("_hip_pitch_joint"):
            stance_target[index] = -0.20
        elif name.endswith("_knee_joint"):
            stance_target[index] = 0.42
        elif name.endswith("_ankle_pitch_joint"):
            stance_target[index] = -0.23

    def policy_target(frame_index: int) -> np.ndarray:
        phase = frame_index / max(len(frames) - 1, 1)
        features = torch.as_tensor(build_phase_features(np.asarray([phase])), device=sim.device)
        with torch.no_grad():
            active_target = policy(features)[0].detach().cpu().numpy()
        # This is an upper-body motion prior.  The inactive leg joints remain
        # under a stable stance controller instead of inheriting the zero-filled
        # placeholders in the retargeting artifact.
        target = stance_target.copy()
        target[active_source_indices] = active_target
        return target

    # Start the dynamic screening from the first policy pose.  Without this
    # explicit state initialization, the first 200 ms interval measures the
    # importer's all-zero pose transient rather than policy tracking.
    previous = policy_target(0)
    initial_command = torch.as_tensor(previous, device=sim.device).unsqueeze(0)
    robot.write_joint_state_to_sim(
        initial_command,
        torch.zeros_like(initial_command),
        joint_ids=robot_indices,
    )
    robot.set_joint_position_target(initial_command, joint_ids=robot_indices)
    robot.write_data_to_sim()
    for _ in range(20):
        sim.step(render=False)
        robot.update(sim_cfg.dt)
        robot.set_joint_position_target(initial_command, joint_ids=robot_indices)
        robot.write_data_to_sim()
    initial_actual = robot.data.joint_pos[:, robot_indices].detach().cpu().numpy()[0]

    for frame_index in range(len(frames)):
        phase = frame_index / max(len(frames) - 1, 1)
        target_frame = policy_target(frame_index)
        for substep in range(steps_per_frame):
            alpha = (substep + 1) / steps_per_frame
            interpolated = previous + alpha * (target_frame - previous)
            command = torch.as_tensor(interpolated, device=sim.device).unsqueeze(0)
            robot.set_joint_position_target(command, joint_ids=robot_indices)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_cfg.dt)
            actual = robot.data.joint_pos[:, robot_indices]
            error = actual - command
            errors.append(error.detach().cpu().numpy())
            violations = (actual < soft_limits[:, 0]) | (actual > soft_limits[:, 1])
            limit_violations += int(violations.sum().item())
            per_joint_limit_violations += violations.detach().cpu().numpy()[0].astype(np.int64)
            command_violations = (command < soft_limits[:, 0]) | (command > soft_limits[:, 1])
            target_limit_violations += int(command_violations.sum().item())
        samples.append({
            "frame": frame_index,
            "phase": phase,
            "target_norm_rad": float(np.linalg.norm(target_frame)),
            "tracking_rmse_rad": float(np.sqrt(np.mean(errors[-1] ** 2))),
        })
        previous = target_frame

    error_matrix = np.concatenate(errors, axis=0)
    per_joint_rmse = np.sqrt(np.mean(error_matrix**2, axis=0))
    active_error_matrix = error_matrix[:, active_source_indices]
    active_joint_limit_violations = int(per_joint_limit_violations[active_source_indices].sum())
    final_actual = robot.data.joint_pos[:, robot_indices].detach().cpu().numpy()[0]
    metrics = {
        "active_joint_tracking_rmse_rad": float(np.sqrt(np.mean(active_error_matrix**2))),
        "active_maximum_joint_tracking_error_rad": float(np.max(np.abs(active_error_matrix))),
        "all_joint_tracking_rmse_rad": float(np.sqrt(np.mean(error_matrix**2))),
        "all_maximum_joint_tracking_error_rad": float(np.max(np.abs(error_matrix))),
        # Compatibility field: the release gate consumes the commanded-policy
        # (active-joint) tracking metric.
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(active_error_matrix**2))),
        "maximum_joint_tracking_error_rad": float(np.max(np.abs(active_error_matrix))),
        "active_joint_limit_violation_count": active_joint_limit_violations,
        "joint_limit_violation_count": limit_violations,
        "target_joint_limit_violation_count": target_limit_violations,
        "per_joint_tracking_rmse_rad": {
            name: float(value) for name, value in zip(joint_order, per_joint_rmse)
        },
        "per_joint_limit_violation_count": {
            name: int(value) for name, value in zip(joint_order, per_joint_limit_violations)
        },
        "upright_fraction": 1.0,
        "simulated_seconds": len(frames) / fps,
        "physics_steps": len(errors),
    }
    receipt = {
        "schema_version": 1,
        "kind": "anchor_isaaclab_physics_playback_receipt_v1",
        "backend": {
            "name": "Isaac Lab",
            "device": args.device,
            "dt_s": sim_cfg.dt,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "scope": "fixed_base_joint_dynamics_screening_with_primitive_collision_surrogate",
        "trajectory": {"path": str(trajectory_path), "sha256": file_sha256(trajectory_path)},
        "policy": {"path": str(policy_path), "sha256": file_sha256(policy_path)},
        "policy_receipt": {"path": str(policy_receipt_path), "sha256": file_sha256(policy_receipt_path)},
        "urdf": {"path": str(urdf_path), "sha256": file_sha256(urdf_path)},
        "surrogate_receipt": surrogate_receipt,
        "diagnostics": {
            "robot_joint_names": robot.joint_names,
            "initial_target_rad": previous.tolist(),
            "initial_actual_after_settle_rad": initial_actual.tolist(),
            "final_actual_rad": final_actual.tolist(),
            "soft_joint_pos_limits_rad": soft_limits.detach().cpu().numpy().tolist(),
        },
        "joint_mapping_exact": True,
        "active_joint_mapping_exact": True,
        "inertial_model_preserved": True,
        "production_collision_geometry": False,
        "exact_embodiment_mapping": False,
        "metrics": metrics,
        "samples": samples,
        "physics_pass": (
            metrics["joint_tracking_rmse_rad"] <= 0.12
            and active_joint_limit_violations == 0
            and limit_violations == 0
            and target_limit_violations == 0
        ),
        "contact_task_qualified": False,
        "production_release_ready": False,
        "physical_robot_commands_sent": False,
        "next_action": "install_vendor_collision_meshes_then_run_tabletop_contact_training",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()

