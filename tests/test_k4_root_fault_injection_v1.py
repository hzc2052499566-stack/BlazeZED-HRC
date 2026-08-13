from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "tools") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "tools"))

import freeze_k4_root_fault_injection_v1 as freezer
import run_k4_root_fault_injection_v1 as runner


def candidates(shoulder_depth: float | None = 3.5) -> list[dict]:
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
            "depth_m": shoulder_depth,
        },
    ]


def estimate_row(point) -> dict:
    if point is None:
        return {"valid": 0, "x_m": "", "y_m": "", "z_m": ""}
    return {
        "valid": 1,
        "x_m": point[0],
        "y_m": point[1],
        "z_m": point[2],
    }


class K4RootFaultInjectionV1Tests(unittest.TestCase):
    def test_frozen_scenarios_use_fixed_frames_and_offsets(self):
        by_name = {row["name"]: row for row in freezer.SCENARIOS}
        self.assertEqual(
            by_name["positive_impulse_150mm"]["fault_frames"],
            [60, 120, 180],
        )
        self.assertEqual(
            by_name["positive_impulse_150mm"][
                "shoulder_depth_offset_m"
            ],
            0.150,
        )
        self.assertEqual(
            by_name["negative_impulse_150mm"][
                "shoulder_depth_offset_m"
            ],
            -0.150,
        )
        self.assertEqual(
            by_name["missing_root_impulse"]["fault_type"],
            "missing_depth_impulse",
        )

    def test_positive_fault_is_applied_only_on_registered_frame(self):
        scenario = {
            "fault_type": "additive_depth_impulse",
            "fault_frames": [60],
            "shoulder_depth_offset_m": 0.150,
        }
        rows = candidates()
        active, source, injected = runner.apply_fault(rows, scenario, 60)
        self.assertTrue(active)
        self.assertAlmostEqual(source, 3.5)
        self.assertAlmostEqual(injected, 3.65)
        self.assertAlmostEqual(rows[1]["depth_m"], 3.65)

        rows = candidates()
        active, source, injected = runner.apply_fault(rows, scenario, 59)
        self.assertFalse(active)
        self.assertAlmostEqual(source, 3.5)
        self.assertAlmostEqual(injected, 3.5)

    def test_missing_fault_removes_only_shoulder_depth(self):
        scenario = {
            "fault_type": "missing_depth_impulse",
            "fault_frames": [120],
            "shoulder_depth_offset_m": None,
        }
        rows = candidates()
        active, source, injected = runner.apply_fault(rows, scenario, 120)
        self.assertTrue(active)
        self.assertAlmostEqual(source, 3.5)
        self.assertIsNone(injected)
        self.assertAlmostEqual(rows[0]["depth_m"], 3.4)

    def test_candidate_reconstruction_retains_missing_depth_landmark(self):
        row = {
            "pixel_x": "449",
            "pixel_y": "278",
            "visibility": "0.994",
            "depth_m": "",
        }
        result = runner.candidate_from_raw(
            row,
            {"canonical_joint": "right_elbow"},
        )
        self.assertTrue(result["in_image"])
        self.assertAlmostEqual(result["visibility"], 0.994)
        self.assertIsNone(result["depth_m"])

    def test_recovery_requires_both_joints_within_one_millimetre(self):
        by_key = {}
        clean = {}
        for sequence in range(1, 4):
            for joint in runner.JOINTS:
                clean[("run", "k4_inferred", sequence, joint)] = (
                    estimate_row((0.0, 0.0, 0.0))
                )
                point = (
                    (0.002, 0.0, 0.0)
                    if sequence == 1
                    else (0.0005, 0.0, 0.0)
                )
                by_key[
                    ("run", "fault", "k4_inferred", sequence, joint)
                ] = estimate_row(point)
        observed = runner.recovery_frames(
            by_key,
            clean,
            "run",
            "fault",
            "k4_inferred",
            0,
        )
        self.assertEqual(observed, 2)

    def test_protocol_hash_changes_with_fault_definition(self):
        first = {
            "scenario": {"offset_m": 0.150},
            "frames": [60, 120, 180],
        }
        second = {
            "scenario": {"offset_m": 0.151},
            "frames": [60, 120, 180],
        }
        self.assertNotEqual(
            runner.canonical_payload_sha256(first),
            runner.canonical_payload_sha256(second),
        )


if __name__ == "__main__":
    unittest.main()
