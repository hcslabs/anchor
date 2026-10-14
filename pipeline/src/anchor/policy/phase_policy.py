"""Phase-conditioned imitation policy trained from an Anchor trajectory.

This is the policy warm-start stage, not the final contact policy.  It learns
the motion prior produced by morphology retargeting.  Physics RL may then add a
bounded residual for contact, balance and robustness without losing lineage to
the source human video.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import math
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_phase_features(phase: np.ndarray, harmonics: int = 4) -> np.ndarray:
    """Encode normalized motion phase using a continuous Fourier basis."""

    values = np.asarray(phase, dtype=np.float32).reshape(-1, 1)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("phase must be finite and within [0, 1]")
    columns = [values]
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * math.pi * harmonic * values
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.concatenate(columns, axis=1)


def _load_training_data(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") not in {"anchor_morphology_retarget_v1", "hcs_g1_retarget_v1"}:
        raise ValueError("policy input must be an Anchor retargeted trajectory")
    if document.get("uses_premade_robot_motion") is not False:
        raise ValueError("policy training refuses pre-made robot motion")
    frames = document.get("frames")
    joint_order = list(document.get("joint_order", []))
    active_joints = list(document.get("active_joints", []))
    if not frames or not joint_order or not active_joints:
        raise ValueError("trajectory is missing frames or joint metadata")
    indices = [joint_order.index(name) for name in active_joints]
    target = np.asarray([[frame["dof_rad"][index] for index in indices] for frame in frames], dtype=np.float32)
    phase = np.linspace(0.0, 1.0, len(frames), dtype=np.float32)
    return document, phase, target


def train_phase_policy(
    trajectory_path: str | Path,
    output_directory: str | Path,
    *,
    epochs: int = 1200,
    seed: int = 7,
    hidden_size: int = 128,
    learning_rate: float = 3.0e-3,
    device: str | None = None,
) -> dict[str, Any]:
    """Train and export a deterministic TorchScript motion-prior policy."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install anchor-pipeline[policy] to train a policy") from exc

    trajectory_path = Path(trajectory_path).resolve()
    output_root = Path(output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    document, phase, target_np = _load_training_data(trajectory_path)
    torch.manual_seed(seed)
    np.random.seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    features_np = build_phase_features(phase)
    features = torch.as_tensor(features_np, device=selected_device)
    target = torch.as_tensor(target_np, device=selected_device)
    target_mean = target.mean(dim=0)
    target_scale = target.std(dim=0).clamp_min(1.0e-4)
    normalized_target = (target - target_mean) / target_scale

    network = nn.Sequential(
        nn.Linear(features.shape[1], hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, target.shape[1]),
    ).to(selected_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    final = {}
    for epoch in range(epochs):
        # Small phase and observation perturbations are the first sim-to-real
        # robustness layer; physics randomization is applied downstream.
        jitter = torch.randn_like(features[:, :1]) * 0.002
        noisy_phase = (torch.as_tensor(phase, device=selected_device).reshape(-1, 1) + jitter).clamp(0.0, 1.0)
        noisy_features = torch.as_tensor(
            build_phase_features(noisy_phase.detach().cpu().numpy().reshape(-1)),
            device=selected_device,
        )
        noisy_features = noisy_features + 0.001 * torch.randn_like(noisy_features)
        prediction = network(noisy_features)
        position_loss = torch.mean((prediction - normalized_target) ** 2)
        predicted_q = prediction * target_scale + target_mean
        velocity_loss = torch.mean((torch.diff(predicted_q, dim=0) - torch.diff(target, dim=0)) ** 2)
        acceleration_loss = torch.mean(
            (torch.diff(predicted_q, n=2, dim=0) - torch.diff(target, n=2, dim=0)) ** 2
        )
        loss = position_loss + 0.05 * velocity_loss + 0.01 * acceleration_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "epoch": epoch + 1,
            "loss": float(loss.detach().cpu()),
            "position_loss": float(position_loss.detach().cpu()),
            "velocity_loss": float(velocity_loss.detach().cpu()),
            "acceleration_loss": float(acceleration_loss.detach().cpu()),
        }

    class Policy(nn.Module):
        def __init__(self, model, mean, scale):
            super().__init__()
            self.model = model
            self.register_buffer("mean", mean)
            self.register_buffer("scale", scale)

        def forward(self, phase_features):
            return self.model(phase_features) * self.scale + self.mean

    policy = Policy(network.eval(), target_mean, target_scale).eval()
    scripted = torch.jit.trace(policy, features[:1])
    policy_path = output_root / "anchor_phase_policy.pt"
    scripted.save(str(policy_path))
    with torch.no_grad():
        predicted = policy(features)
        rmse = torch.sqrt(torch.mean((predicted - target) ** 2))
        max_error = torch.max(torch.abs(predicted - target))

    receipt = {
        "schema_version": 1,
        "kind": "anchor_motion_prior_policy_receipt_v1",
        "method": "phase_conditioned_imitation_with_velocity_and_acceleration_regularization",
        "policy_role": "video_motion_prior_warm_start_before_physics_residual_learning",
        "source_trajectory": {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)},
        "source_video": document.get("source_video"),
        "embodiment_profile": document.get("embodiment_profile"),
        "urdf": document.get("urdf"),
        "active_joints": document["active_joints"],
        "fps": float(document["fps"]),
        "frame_count": len(document["frames"]),
        "feature_contract": {"phase": True, "fourier_harmonics": 4, "dimension": features.shape[1]},
        "training": {
            "epochs": epochs,
            "seed": seed,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "device": selected_device,
            "domain_randomization": {"phase_std": 0.002, "feature_noise_std": 0.001},
            "final": final,
        },
        "fit": {"joint_rmse_rad": float(rmse.cpu()), "maximum_joint_error_rad": float(max_error.cpu())},
        "artifact": {"path": str(policy_path), "sha256": _sha256(policy_path), "format": "torchscript"},
        "physical_robot_commands_sent": False,
        "production_release_ready": False,
    }
    receipt_path = output_root / "anchor_motion_prior_policy_receipt_v1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt

