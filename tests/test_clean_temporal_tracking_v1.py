from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "tools") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "tools"))

import analyse_clean_temporal_tracking_v1 as temporal


class CleanTemporalMetricTests(unittest.TestCase):
    def test_static_jitter_uses_three_axis_population_standard_deviation(self):
        points = {}
        for frame, x in enumerate((0.0, 0.001, 0.002)):
            for joint in temporal.PRIMARY_JOINTS:
                points[(frame, joint)] = (x, 0.0, 0.0)
        result = temporal.static_jitter_mm(points, (0, 1, 2))
        expected = math.sqrt(2.0 / 3.0)
        self.assertAlmostEqual(
            result["right_arm_two_joint_mean_mm"], expected, places=9
        )

    def test_step_residual_subtracts_true_motion(self):
        truth = {}
        perfect = {}
        biased = {}
        for frame in range(3):
            for joint in temporal.PRIMARY_JOINTS:
                truth[(frame, joint)] = (frame * 0.01, 0.0, 0.0)
                perfect[(frame, joint)] = (frame * 0.01 + 0.1, 0.0, 0.0)
                biased[(frame, joint)] = (frame * 0.012 + 0.1, 0.0, 0.0)
        perfect_residuals = temporal.temporal_step_residuals_mm(
            perfect, truth, [1, 2]
        )
        self.assertTrue(all(abs(value) < 1e-9 for value in perfect_residuals))
        residuals = temporal.temporal_step_residuals_mm(biased, truth, [1, 2])
        self.assertTrue(all(abs(value - 2.0) < 1e-9 for value in residuals))

    def test_positive_best_lag_means_estimate_lags_ground_truth(self):
        truth = {}
        estimate = {}
        for start, end in temporal.DYNAMIC_SEGMENTS:
            for frame in range(start, end + 1):
                for offset, joint in enumerate(temporal.PRIMARY_JOINTS):
                    truth[(frame, joint)] = (
                        math.sin((frame - start) / 11.0 + offset),
                        math.cos((frame - start) / 17.0 + offset),
                        0.0,
                    )
            for frame in range(start + 2, end + 1):
                for joint in temporal.PRIMARY_JOINTS:
                    estimate[(frame, joint)] = truth[(frame - 2, joint)]
        result = temporal.best_alignment_lag(estimate, truth)
        self.assertEqual(result["lag_frames"], 2)
        self.assertAlmostEqual(result["lag_ms"], 1000.0 / 30.0)


if __name__ == "__main__":
    unittest.main()
