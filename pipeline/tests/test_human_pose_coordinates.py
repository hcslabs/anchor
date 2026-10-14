from __future__ import annotations

import unittest

import numpy as np

from anchor.enrich.human_pose import LANDMARK_INDEX, canonical_landmarks


class HumanPoseCoordinatesTest(unittest.TestCase):
    def test_body_frame_is_pelvis_centred_and_right_handed(self) -> None:
        world = np.zeros((33, 3), dtype=float)
        world[LANDMARK_INDEX["left_hip"]] = [0.0, 0.5, 0.0]
        world[LANDMARK_INDEX["right_hip"]] = [0.0, -0.5, 0.0]
        world[LANDMARK_INDEX["left_shoulder"]] = [0.0, 0.7, 1.0]
        world[LANDMARK_INDEX["right_shoulder"]] = [0.0, -0.7, 1.0]
        world[LANDMARK_INDEX["left_elbow"]] = [0.3, 0.8, 0.8]
        world[LANDMARK_INDEX["right_elbow"]] = [0.3, -0.8, 0.8]
        world[LANDMARK_INDEX["left_wrist"]] = [0.6, 0.9, 0.6]
        world[LANDMARK_INDEX["right_wrist"]] = [0.6, -0.9, 0.6]
        world[LANDMARK_INDEX["left_knee"]] = [0.0, 0.4, -0.8]
        world[LANDMARK_INDEX["right_knee"]] = [0.0, -0.4, -0.8]
        world[LANDMARK_INDEX["left_ankle"]] = [0.0, 0.4, -1.6]
        world[LANDMARK_INDEX["right_ankle"]] = [0.0, -0.4, -1.6]
        converted = canonical_landmarks(world)
        np.testing.assert_allclose(converted["pelvis"], [0.0, 0.0, 0.0])
        self.assertGreater(converted["left_wrist"][1], 0.0)
        self.assertGreater(converted["left_wrist"][0], 0.0)
        self.assertGreater(converted["torso"][2], 0.0)


if __name__ == "__main__":
    unittest.main()
