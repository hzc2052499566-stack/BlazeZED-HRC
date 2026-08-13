from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import analyse_view_ac_best_view_fusion_pilot_v1 as analysis  # noqa: E402


class ReliabilityTests(unittest.TestCase):
    @staticmethod
    def row(valid: int, depth: str) -> dict[str, str]:
        return {"valid": str(valid), "depth_m": depth}

    def replay(self, joint_depth: str, shoulder: str = "3.5", pelvis: str = "3.5"):
        raw = {
            (0, "right_elbow"): self.row(1, joint_depth),
            (0, "right_wrist"): self.row(1, joint_depth),
            (0, "right_shoulder"): self.row(1, shoulder),
            (0, "pelvis"): self.row(1, pelvis),
        }
        return raw

    def test_reliability_flags_occluder_plane_depth(self):
        raw = self.replay("1.975")
        self.assertEqual(analysis.raw_reliability(raw, 0, "right_wrist"), "occluded_suspected")

    def test_reliability_accepts_body_depth(self):
        raw = self.replay("3.6")
        self.assertEqual(analysis.raw_reliability(raw, 0, "right_wrist"), "reliable")

    def test_reliability_marks_missing_depth(self):
        raw = self.replay("")
        self.assertEqual(analysis.raw_reliability(raw, 0, "right_wrist"), "no_measured_depth")

    def test_reliability_marks_missing_torso_reference(self):
        raw = self.replay("3.6", shoulder="")
        raw[(0, "right_shoulder")]["valid"] = "0"
        self.assertEqual(analysis.raw_reliability(raw, 0, "right_wrist"), "torso_reference_unavailable")


class SelectorTests(unittest.TestCase):
    def test_score_is_lexicographic_and_tie_prefers_a(self):
        self.assertEqual(analysis.select_reliability_view((2, 2, 0.8), (2, 2, 0.7), True, True), "a")
        self.assertEqual(analysis.select_reliability_view((1, 2, 0.99), (2, 1, 0.1), True, True), "c")
        self.assertEqual(analysis.select_reliability_view((2, 2, 0.8), (2, 2, 0.8), True, True), "a")

    def test_incomplete_arm_falls_back_to_complete_view(self):
        self.assertEqual(analysis.select_reliability_view((9, 9, 1.0), (0, 0, 0.0), False, True), "c")
        self.assertEqual(analysis.select_reliability_view((0, 0, 0.0), (9, 9, 1.0), True, False), "a")
        self.assertEqual(analysis.select_reliability_view((0, 0, 0.0), (0, 0, 0.0), False, False), "")


class CalibrationTests(unittest.TestCase):
    @staticmethod
    def gt_row(world: np.ndarray, camera: np.ndarray) -> dict[str, str]:
        return {
            "world_x_m": str(world[0]), "world_y_m": str(world[1]), "world_z_m": str(world[2]),
            "gt_x_m": str(camera[0]), "gt_y_m": str(camera[1]), "gt_z_m": str(camera[2]),
        }

    def test_gt_calibration_recovers_common_frame(self):
        world = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        camera_a = world + np.asarray([2.0, -1.0, 0.5])
        camera_c = world + np.asarray([-3.0, 0.25, 1.5])
        keys = [(index, "j") for index in range(4)]
        gt_a = {key: self.gt_row(w, a) for key, w, a in zip(keys, world, camera_a)}
        gt_c = {key: self.gt_row(w, c) for key, w, c in zip(keys, world, camera_c)}
        calibration = analysis.build_calibration(gt_a, gt_c, "a", "c", "camera_a", "camera_c")
        transformed = analysis.transform_c(camera_c[2], calibration)
        self.assertTrue(np.allclose(transformed, camera_a[2], atol=1e-12))
        self.assertLess(calibration["maximum_cross_view_fit_residual_m"], 1e-12)


if __name__ == "__main__":
    unittest.main()
