"""Build a primitive-collision URDF while preserving source kinematics.

The surrogate is for dynamics plumbing and joint-tracking screening when the
vendor mesh package cannot be redistributed.  It is never a substitute for
contact qualification with the production collision geometry.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import xml.etree.ElementTree as ET


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_collision_surrogate(source: str | Path, output: str | Path) -> dict:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    root = ET.parse(source_path).getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in list(link.findall(tag)):
                link.remove(element)
        name = link.attrib.get("name", "")
        size = (0.20, 0.14, 0.16) if name in {"pelvis", "torso_link"} else (0.055, 0.055, 0.11)
        for tag in ("visual", "collision"):
            element = ET.SubElement(link, tag)
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "box", {"size": " ".join(str(value) for value in size)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    receipt = {
        "schema_version": 1,
        "kind": "anchor_urdf_collision_surrogate_receipt_v1",
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "kinematics_preserved": True,
        "inertials_preserved": True,
        "production_collision_geometry_preserved": False,
        "allowed_use": "fixed_base_joint_dynamics_and_pipeline_screening_only",
    }
    receipt_path = output_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt

