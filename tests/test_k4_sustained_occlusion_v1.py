from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "tools") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "tools"))

import freeze_k4_sustained_occlusion_v1 as freezer
import run_k4_root_fault_injection_v1 as root_runner
import run_k4_sustained_occlusion_v1 as runner


def candidates() -> list[dict]:
    return [
        {
            "mapping": {"canonical_joint": "pelvis"},
            "in_image": True,
            "visibility": 0.99,
            "pixel_x": 480,
            "pixel_y": 300,
            "depth_m": 3.4,
        },
        {
            "mapping": {"canonical_joint": "right_shoulder"},
            "in_image": True,
            "visibility": 0.99,
            "pixel_x": 460,
            "pixel_y": 250,
            "depth_m": 3.5,
        },
    ]


def row(point) -> dict:
    if point is None:
        return {"valid": 0, "x_m": "", "y_m": "", "z_m": ""}
    return {
        "valid": 1,
        "x_m": point[0],
        "y_m": point[1],
        "z_m": point[2],
    }


class SustainedOcclusionV1Tests(unittest.TestCase):
    def test_registered_intervals_are_non_overlapping_and_exact(self):
        for duration in freezer.DURATIONS:
            blocks = freezer.intervals(duration)
            frames = freezer.fault_frames(duration)
            self.assertEqual(len(blocks), 3)
            self.assertEqual(len(frames), 3 * duration)
            self.assertEqual(len(set(frames)), len(frames))
            self.assertEqual(blocks[0]["start_frame"], 45)
            self.assertEqual(
                blocks[-1]["end_frame_inclusive"],
                165 + duration - 1,
            )

    def test_landmark_dropout_removes_pixel_support(self):
        scenario = {
            "fault_type": "missing_landmark_impulse",
            "fault_frames": [10, 11],
            "shoulder_depth_offset_m": None,
        }
        values = candidates()
        active, source, injected = root_runner.apply_fault(
            values,
            scenario,
            10,
        )
        self.assertTrue(active)
        self.assertAlmostEqual(source, 3.5)
        self.assertIsNone(injected)
        shoulder = values[1]
        self.assertFalse(shoulder["in_image"])
        self.assertEqual(shoulder["visibility"], 0.0)
        self.assertIsNone(shoulder["depth_m"])

    def test_recovery_requires_three_stable_frames(self):
        by_key = {}
        clean = {}
        for sequence in range(1, 8):
            for joint in root_runner.JOINTS:
                clean[("r", "k4_inferred", sequence, joint)] = row(
                    (0.0, 0.0, 0.0)
                )
                point = (
                    (0.002, 0.0, 0.0)
                    if sequence in {1, 2, 4}
                    else (0.0005, 0.0, 0.0)
                )
                by_key[
                    ("r", "s", "k4_inferred", sequence, joint)
                ] = row(point)
        observed = runner.recovery_after_interval(
            by_key,
            clean,
            "r",
            "s",
            "k4_inferred",
            0,
        )
        self.assertEqual(observed, 5)

    def test_maximum_invalid_streak(self):
        rows = [
            row((0.0, 0.0, 0.0)),
            row(None),
            row(None),
            row((0.0, 0.0, 0.0)),
            row(None),
        ]
        self.assertEqual(runner.max_invalid_streak(rows), 2)


if __name__ == "__main__":
    unittest.main()
