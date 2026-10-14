"""Hash-bound proof capsules for Anchor → KHI G1 lineage."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from anchor.integrations.khi_g1 import validate_khi_replay_pair, validate_khi_robot_reference


KHI_CODE_ARTIFACTS = [
    (
        "khi_g1_urdf",
        "g1_29dof_pc2.urdf",
        "The exact 29-DoF geometry, axes, and limits used by the physical KHI project.",
    ),
    (
        "khi_grail_compatibility_replay",
        "simulation/unitree_sim_isaaclab/tools/aprilcube_inspire_grail_replay.py",
        "KHI physics replay contract used to validate reference layout and free-object behavior.",
    ),
    (
        "khi_training_environment",
        "simulation/unitree_sim_isaaclab/rh56_grasp_rl/env_cfg.py",
        "KHI Isaac Lab RH56 grasp environment consuming a G1 reference.",
    ),
    (
        "khi_reward_and_action_logic",
        "simulation/unitree_sim_isaaclab/rh56_grasp_rl/mdp.py",
        "KHI reference playback, reward, contact, lift, and hold logic.",
    ),
    (
        "khi_ppo_configuration",
        "simulation/unitree_sim_isaaclab/rh56_grasp_rl/agents/rsl_rl_ppo_cfg.py",
        "KHI PPO training configuration.",
    ),
    (
        "khi_runbook",
        "simulation/RUN_ON_VAST.md",
        "KHI simulation and evaluation runbook.",
    ),
]


ANCHOR_CODE_ARTIFACTS = [
    (
        "anchor_pose_extractor",
        "pipeline/src/anchor/enrich/human_pose.py",
        "HCS video decoding, human-pose extraction, body-frame normalization, and provenance code.",
    ),
    (
        "anchor_urdf_kinematics",
        "pipeline/src/anchor/retarget/kinematics.py",
        "HCS URDF FK and analytic point-Jacobian implementation.",
    ),
    (
        "anchor_g1_retargeter",
        "pipeline/src/anchor/retarget/hcs_g1.py",
        "HCS robot-proportioned, joint-limited DLS retargeter and KHI profile.",
    ),
    (
        "anchor_khi_exporter",
        "pipeline/src/anchor/integrations/khi_g1.py",
        "HCS KHI joblib exporter and exact-loader compatibility validator.",
    ),
    (
        "anchor_proof_verifier",
        "pipeline/src/anchor/proof.py",
        "HCS no-copy manifest and lineage verifier.",
    ),
]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(identifier: str, path: Path, role: str, description: str) -> dict[str, Any]:
    path = path.resolve()
    exists = path.is_file()
    return {
        "id": identifier,
        "role": role,
        "description": description,
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else None,
        "copied_into_capsule": False,
    }


def build_khi_proof_manifest(
    *,
    khi_root: str | Path,
    human_video: str | Path,
    human_motion: str | Path,
    g1_trajectory: str | Path,
    khi_reference: str | Path,
    khi_object_reference: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    khi_root = Path(khi_root).resolve()
    anchor_root = Path(__file__).resolve().parents[3]
    artifacts = [
        _artifact(
            "human_video",
            Path(human_video),
            "source",
            "Human demonstration video selected for the KHI tabletop proof.",
        ),
        _artifact(
            "human_motion",
            Path(human_motion),
            "derived",
            "Body-aligned 3-D human landmarks extracted from the selected video.",
        ),
        _artifact(
            "hcs_g1_trajectory",
            Path(g1_trajectory),
            "derived",
            "HCS-owned morphology-aware G1 joint trajectory; no pre-made robot motion.",
        ),
        _artifact(
            "khi_g1_reference",
            Path(khi_reference),
            "integration",
            "Joblib reference accepted by the KHI Isaac Lab reference loader.",
        ),
        _artifact(
            "khi_object_reference",
            Path(khi_object_reference),
            "integration",
            "Key-matched free-cube initialization companion for the KHI replay loader.",
        ),
    ]
    for identifier, relative, description in KHI_CODE_ARTIFACTS:
        artifacts.append(_artifact(identifier, khi_root / relative, "khi_project", description))
    for identifier, relative, description in ANCHOR_CODE_ARTIFACTS:
        artifacts.append(_artifact(identifier, anchor_root / relative, "anchor_code", description))
    manifest = {
        "schema_version": 1,
        "kind": "anchor_khi_human_video_proof_v1",
        "proof_id": "anchor-khi-g1-human-video-retarget-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim": (
            "The KHI-compatible G1 reference was computed from the selected human video "
            "by HCS retargeting code, not copied from a released robot-motion dataset."
        ),
        "disk_policy": {
            "artifact_mode": "hash_references_no_copy",
            "large_source_files_duplicated": False,
        },
        "artifacts": artifacts,
        "edges": [
            {"from": "human_video", "to": "human_motion", "relation": "pose_extracted_from"},
            {"from": "anchor_pose_extractor", "to": "human_motion", "relation": "generated_by"},
            {"from": "human_motion", "to": "hcs_g1_trajectory", "relation": "retargeted_by_hcs"},
            {"from": "anchor_g1_retargeter", "to": "hcs_g1_trajectory", "relation": "generated_by"},
            {"from": "anchor_urdf_kinematics", "to": "hcs_g1_trajectory", "relation": "kinematics_backend"},
            {"from": "khi_g1_urdf", "to": "hcs_g1_trajectory", "relation": "defines_embodiment"},
            {"from": "hcs_g1_trajectory", "to": "khi_g1_reference", "relation": "exported_without_motion_substitution"},
            {"from": "anchor_khi_exporter", "to": "khi_g1_reference", "relation": "generated_by"},
            {"from": "khi_g1_reference", "to": "khi_object_reference", "relation": "shares_motion_key_with"},
            {"from": "anchor_khi_exporter", "to": "khi_object_reference", "relation": "generated_by"},
            {"from": "khi_g1_reference", "to": "khi_training_environment", "relation": "loader_compatible_with"},
        ],
        "scope": {
            "technical_lineage": "must verify locally",
            "physics_simulation": "not run by this lightweight proof",
            "physical_robot_execution": "not authorized by this proof",
            "production_pose_reconstruction": "requires VideoMimic/SMPL-X or equivalent metric pass",
            "source_rights": "must be attested before external release or training use",
        },
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def verify_khi_proof_manifest(path: str | Path, receipt_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "anchor_khi_human_video_proof_v1":
        raise ValueError("not an Anchor KHI proof manifest")
    by_id = {artifact["id"]: artifact for artifact in manifest["artifacts"]}
    artifact_checks = []
    all_hashes_match = True
    for artifact in manifest["artifacts"]:
        artifact_path = Path(artifact["path"])
        exists = artifact_path.is_file()
        actual_hash = _sha256(artifact_path) if exists else None
        matches = exists and actual_hash == artifact.get("sha256")
        all_hashes_match = all_hashes_match and matches
        artifact_checks.append({
            "id": artifact["id"],
            "exists": exists,
            "hash_matches": matches,
            "actual_sha256": actual_hash,
        })

    lineage_checks: dict[str, bool] = {}
    if all(identifier in by_id for identifier in ("human_video", "human_motion", "hcs_g1_trajectory", "khi_g1_reference", "khi_object_reference", "khi_g1_urdf")):
        video_hash = by_id["human_video"]["sha256"]
        human = json.loads(Path(by_id["human_motion"]["path"]).read_text(encoding="utf-8"))
        trajectory = json.loads(Path(by_id["hcs_g1_trajectory"]["path"]).read_text(encoding="utf-8"))
        reference = validate_khi_robot_reference(by_id["khi_g1_reference"]["path"])
        pair = validate_khi_replay_pair(
            by_id["khi_g1_reference"]["path"],
            by_id["khi_object_reference"]["path"],
        )
        lineage_checks = {
            "human_pose_names_source_video": human.get("source_video", {}).get("sha256") == video_hash,
            "trajectory_names_source_video": trajectory.get("source_video", {}).get("sha256") == video_hash,
            "trajectory_names_human_pose": (
                trajectory.get("input_pose_artifact", {}).get("sha256") == by_id["human_motion"]["sha256"]
            ),
            "trajectory_uses_khi_urdf": (
                trajectory.get("urdf", {}).get("sha256") == by_id["khi_g1_urdf"]["sha256"]
            ),
            "trajectory_excludes_premade_robot_motion": trajectory.get("uses_premade_robot_motion") is False,
            "khi_reference_names_trajectory": (
                reference["lineage"].get("source_trajectory_sha256") == by_id["hcs_g1_trajectory"]["sha256"]
            ),
            "khi_reference_joblib_loader_compatible": reference.get("valid") is True,
            "khi_robot_object_motion_keys_match": pair.get("valid") is True,
            "khi_replay_fields_present": pair.get("khi_replay_fields_present") is True,
        }
        screening_only = human.get("quality", {}).get("screening_only") is True
        rights_status = human.get("source_video", {}).get("rights_status")
    else:
        screening_only = True
        rights_status = None

    technical_lineage_complete = all_hashes_match and bool(lineage_checks) and all(lineage_checks.values())
    rights_verified = rights_status in {"t1_verified_capture", "t2_partner_licensed", "verified"}
    receipt = {
        "schema_version": 1,
        "kind": "anchor_khi_proof_verification_v1",
        "manifest_path": str(path),
        "manifest_sha256": _sha256(path),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "artifact_checks": artifact_checks,
        "lineage_checks": lineage_checks,
        "technical_lineage_complete": technical_lineage_complete,
        "human_video_used": technical_lineage_complete,
        "uses_premade_robot_motion": False if technical_lineage_complete else None,
        "screening_pose_only": screening_only,
        "source_rights_verified": rights_verified,
        "khi_physics_simulation_verified": False,
        "physical_robot_verified": False,
        "production_release_ready": (
            technical_lineage_complete and rights_verified and not screening_only
        ),
        "verdict": (
            "TECHNICAL_LINEAGE_PASS_SCREENING_ONLY"
            if technical_lineage_complete
            else "FAIL"
        ),
    }
    if receipt_path is not None:
        output = Path(receipt_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt

