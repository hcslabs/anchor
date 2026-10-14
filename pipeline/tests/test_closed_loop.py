from __future__ import annotations

import unittest

from anchor.agent.closed_loop import evaluate_quality, interpret_physics_feedback


class ClosedLoopTest(unittest.TestCase):
    def test_gate_passes_clean_technical_candidate(self) -> None:
        human = {"quality": {"detection_ratio": 0.8, "uniform_timeline_preserved": True}}
        trajectory = {
            "uses_premade_robot_motion": False,
            "quality": {
                "frame_count": 20,
                "mean_marker_rmse_m": 0.009,
                "maximum_marker_rmse_m": 0.025,
                "maximum_joint_velocity_limit_ratio": 0.4,
                "all_joint_limits_satisfied": True,
                "all_velocity_limits_satisfied": True,
            },
        }
        self.assertTrue(evaluate_quality(human, trajectory)["passed"])

    def test_feedback_requests_lateral_motion_after_table_contact(self) -> None:
        result = interpret_physics_feedback(
            {
                "kind": "anchor_khi_physics_feedback_v1",
                "table_collision_count": 2,
                "grasp_success_rate": 0.8,
                "minimum_hold_s": 5.0,
            }
        )
        self.assertFalse(result["physics_qualified"])
        self.assertIn("lateral_clearance", result["next_action"])
