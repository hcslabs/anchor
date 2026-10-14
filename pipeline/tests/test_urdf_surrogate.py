from pathlib import Path
import xml.etree.ElementTree as ET

from anchor.sim.urdf_surrogate import build_collision_surrogate


def test_surrogate_preserves_joints_and_replaces_meshes(tmp_path: Path):
    source = tmp_path / "source.urdf"
    source.write_text(
        """<robot name="r"><link name="base"><inertial><mass value="1"/></inertial><visual><geometry><mesh filename="x.stl"/></geometry></visual></link><link name="arm"/><joint name="j" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>""",
        encoding="utf-8",
    )
    output = tmp_path / "surrogate.urdf"
    receipt = build_collision_surrogate(source, output)
    root = ET.parse(output).getroot()
    assert [joint.attrib["name"] for joint in root.findall("joint")] == ["j"]
    assert root.findall(".//mesh") == []
    assert len(root.findall(".//collision")) == 2
    assert receipt["kinematics_preserved"] is True
    assert receipt["production_collision_geometry_preserved"] is False

