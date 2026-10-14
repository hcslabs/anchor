from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anchor.retarget.hcs_g1 import G1_29_DOF_NAMES, G1Retargeter, MARKER_LINKS, retarget_file


class HcsG1RetargetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = Path(os.environ.get(
            "ANCHOR_KHI_G1_URDF",
            r"C:\Users\salio\Downloads\G1_COMPLETE_HANDOFF_20260915_01\source\g1-aprilcube-autonomy\g1_29dof_pc2.urdf",
        ))
        if not configured.exists():
            raise unittest.SkipTest("KHI URDF integration link has not been installed")
        cls.urdf = configured

    def _human_document(self) -> dict:
        retargeter = G1Retargeter(self.urdf)
        poses = []
        for step in range(4):
            q = np.zeros(len(G1_29_DOF_NAMES), dtype=float)
            q[G1_29_DOF_NAMES.index("right_shoulder_pitch_joint")] = 0.12 * step
            q[G1_29_DOF_NAMES.index("right_shoulder_roll_joint")] = -0.06 * step
            q[G1_29_DOF_NAMES.index("right_elbow_joint")] = 0.15 * step
            state = retargeter.model.state(dict(zip(G1_29_DOF_NAMES, q)))
            poses.append({
                "time_s": step / 30.0,
                "landmarks": {
                    marker: retargeter._marker_position(state, marker).tolist()
                    for marker in MARKER_LINKS
                },
            })
        return {
            "schema_version": 1,
            "kind": "anchor_human_motion_v1",
            "coordinate_system": "x_forward_y_left_z_up_m",
            "fps": 30.0,
            "source_video": {"content_id": "anc1-test", "rights": "fixture_only"},
            "frames": poses,
        }

    def test_round_trip_robot_proportioned_landmarks(self) -> None:
        result = G1Retargeter(self.urdf).retarget_document(self._human_document())
        self.assertFalse(result["uses_premade_robot_motion"])
        self.assertEqual(result["joint_order"], G1_29_DOF_NAMES)
        self.assertEqual(result["quality"]["frame_count"], 4)
        self.assertLess(result["quality"]["maximum_marker_rmse_m"], 0.012)
        self.assertTrue(result["quality"]["all_joint_limits_satisfied"])

    def test_file_output_is_small_and_bound_to_urdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "human.json"
            output = Path(directory) / "g1.json"
            source.write_text(json.dumps(self._human_document()), encoding="utf-8")
            result = retarget_file(source, self.urdf, output)
            self.assertTrue(output.exists())
            self.assertLess(output.stat().st_size, 200_000)
            self.assertEqual(len(result["urdf"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
