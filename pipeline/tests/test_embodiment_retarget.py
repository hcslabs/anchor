from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import unittest

from anchor.retarget.embodiment import MorphologyRetargeter, load_embodiment_profile
from anchor.retarget.hcs_g1 import G1Retargeter


class EmbodimentRetargetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[2]
        cls.urdf = Path(os.environ.get("ANCHOR_KHI_G1_URDF", ""))
        cls.profile = cls.repo / "integrations" / "khi_g1" / "unitree_g1_29dof.embodiment.json"
        if not cls.urdf.is_file():
            raise unittest.SkipTest("KHI URDF integration link has not been installed")

    def _human_document(self):
        legacy = G1Retargeter(self.urdf)
        frames = []
        for step in range(3):
            q = {name: 0.0 for name in legacy.model.joint_by_name}
            q["right_shoulder_pitch_joint"] = 0.08 * step
            q["right_elbow_joint"] = 0.12 * step
            state = legacy.model.state(q)
            landmarks = {
                marker: legacy._marker_position(state, marker).tolist()
                for marker in legacy.robot_neutral
            }
            frames.append({"time_s": step / 5.0, "landmarks": landmarks})
        return {
            "schema_version": 1,
            "coordinate_system": "x_forward_y_left_z_up_m",
            "fps": 5.0,
            "source_video": {"sha256": "synthetic-test"},
            "frames": frames,
        }

    def test_profile_drives_generic_solver(self) -> None:
        result = MorphologyRetargeter(
            self.urdf, self.profile, profile="khi_tabletop_right"
        ).retarget_document(self._human_document())
        self.assertEqual(result["kind"], "anchor_morphology_retarget_v1")
        self.assertEqual(result["target_robot"], "unitree_g1_29dof")
        self.assertFalse(result["uses_premade_robot_motion"])
        self.assertEqual(result["quality"]["frame_count"], 3)
        self.assertTrue(result["quality"]["all_joint_limits_satisfied"])

    def test_unvalidated_h2_template_refuses_execution(self) -> None:
        template = self.repo / "integrations" / "embodiments" / "unitree_h2.template.json"
        with self.assertRaisesRegex(ValueError, "not validated"):
            load_embodiment_profile(template)
