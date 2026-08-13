import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "tools"))

from multiperson_tracking_core_v1 import (  # noqa: E402
    PoseDetection,
    TrackManager,
    bbox_iou,
)


def detection(instance_id, centre_x, centre_y, half_width=0.05, half_height=0.10):
    return PoseDetection(
        instance_id=instance_id,
        centre_x=centre_x,
        centre_y=centre_y,
        bbox_left=centre_x - half_width,
        bbox_top=centre_y - half_height,
        bbox_right=centre_x + half_width,
        bbox_bottom=centre_y + half_height,
    )


class BboxTests(unittest.TestCase):
    def test_iou_identical_boxes_is_one(self):
        item = detection(0, 0.5, 0.5)
        self.assertAlmostEqual(bbox_iou(item, item), 1.0)

    def test_iou_disjoint_boxes_is_zero(self):
        self.assertEqual(
            bbox_iou(detection(0, 0.2, 0.5), detection(1, 0.8, 0.5)),
            0.0,
        )


class TrackManagerTests(unittest.TestCase):
    def test_detection_order_change_does_not_swap_tracks(self):
        manager = TrackManager(max_centre_distance=0.25, max_missed_frames=3)
        first = manager.update(
            0,
            [detection(0, 0.25, 0.5), detection(1, 0.75, 0.5)],
        )
        second = manager.update(
            1,
            [detection(0, 0.74, 0.5), detection(1, 0.26, 0.5)],
        )

        self.assertEqual([row["track_id"] for row in first], [0, 1])
        self.assertEqual([row["track_id"] for row in second], [1, 0])
        self.assertTrue(all(row["association_status"] == "matched" for row in second))

    def test_distant_detection_starts_new_track(self):
        manager = TrackManager(max_centre_distance=0.10, max_missed_frames=3)
        manager.update(0, [detection(0, 0.2, 0.5)])

        result = manager.update(1, [detection(0, 0.8, 0.5)])

        self.assertEqual(result[0]["track_id"], 1)
        self.assertEqual(result[0]["association_status"], "new")

    def test_track_expires_after_registered_gap(self):
        manager = TrackManager(max_centre_distance=0.25, max_missed_frames=2)
        manager.update(0, [detection(0, 0.5, 0.5)])

        self.assertEqual(manager.active_track_count(2), 1)
        self.assertEqual(manager.active_track_count(3), 0)

    def test_new_tracks_have_zero_association_confidence(self):
        manager = TrackManager()
        result = manager.update(0, [detection(0, 0.5, 0.5)])

        self.assertEqual(result[0]["association_confidence"], 0.0)
        self.assertIsNone(result[0]["association_cost"])


if __name__ == "__main__":
    unittest.main()
