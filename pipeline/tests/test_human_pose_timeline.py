import unittest

import numpy as np

from anchor.enrich.human_pose import interpolate_detection_timeline


class HumanPoseTimelineTest(unittest.TestCase):
    def test_internal_gaps_are_filled_without_time_compression(self):
        def detection(value, visibility):
            return {
                "mean_visibility": visibility,
                "landmarks": {"right_wrist": np.asarray([value, 0.0, 0.0])},
            }

        detections = [None, detection(0.0, 0.6), None, None, detection(3.0, 0.9), None]
        first, last, timeline = interpolate_detection_timeline(detections)

        self.assertEqual((first, last), (1, 4))
        self.assertEqual([item["source_frame_index"] for item in timeline], [1, 2, 3, 4])
        self.assertEqual([item["interpolated"] for item in timeline], [False, True, True, False])
        np.testing.assert_allclose(
            [item["landmarks"]["right_wrist"][0] for item in timeline],
            [0.0, 1.0, 2.0, 3.0],
        )
        np.testing.assert_allclose(
            [item["mean_visibility"] for item in timeline],
            [0.6, 0.7, 0.8, 0.9],
        )


if __name__ == "__main__":
    unittest.main()
