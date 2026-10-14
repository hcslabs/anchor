from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from anchor.integrations.khi_g1 import (
    export_khi_object_reference,
    export_khi_robot_reference,
    validate_khi_replay_pair,
    validate_khi_robot_reference,
)
from anchor.retarget.hcs_g1 import G1_29_DOF_NAMES


class KhiG1ExportTest(unittest.TestCase):
    def test_export_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.json"
            output = root / "reference.pkl"
            trajectory.write_text(json.dumps({
                "schema_version": 1,
                "kind": "anchor_hcs_g1_retarget_v1",
                "uses_premade_robot_motion": False,
                "target_robot": "unitree_g1_29dof",
                "retarget_profile": "khi_tabletop_right",
                "joint_order": G1_29_DOF_NAMES,
                "fps": 5.0,
                "source_video": {"sha256": "a" * 64},
                "urdf": {"sha256": "b" * 64},
                "frames": [
                    {"dof_rad": [0.0] * 29},
                    {"dof_rad": [0.01] * 29},
                ],
            }), encoding="utf-8")
            receipt = export_khi_robot_reference(trajectory, output)
            self.assertTrue(receipt["joblib_round_trip"])
            validated = validate_khi_robot_reference(output)
            self.assertTrue(validated["valid"])
            self.assertEqual(validated["frame_count"], 2)
            self.assertFalse(validated["lineage"]["uses_premade_robot_motion"])
            object_output = root / "object_reference.pkl"
            object_receipt = export_khi_object_reference(output, object_output)
            self.assertTrue(object_receipt["free_object"])
            pair = validate_khi_replay_pair(output, object_output)
            self.assertTrue(pair["valid"])
            self.assertEqual(pair["motion_key"], receipt["motion_key"])


if __name__ == "__main__":
    unittest.main()
