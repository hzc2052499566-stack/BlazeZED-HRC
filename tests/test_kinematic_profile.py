from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import calibrate_kinematic_profile as calibration


class KinematicProfileEstimatorTests(unittest.TestCase):
    def test_weighted_median_respects_weight(self):
        result = calibration.weighted_median(
            [0.2, 0.3, 0.4],
            [1.0, 10.0, 1.0],
        )
        self.assertEqual(result, 0.3)

    def test_weighted_median_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            calibration.weighted_median([0.2, 0.3], [1.0, -1.0])

    def test_huber_location_resists_large_outlier(self):
        values = np.asarray([0.25] * 20 + [0.60], dtype=np.float64)
        result = calibration.robust_location(
            values,
            np.ones_like(values),
            huber_delta=1.5,
            scale_floor_m=0.005,
            maximum_iterations=50,
            tolerance_m=1e-9,
        )
        self.assertLess(abs(result["location_m"] - 0.25), 0.002)

    def test_cv_is_none_for_single_sample(self):
        self.assertIsNone(calibration.coefficient_of_variation([0.25]))

    def test_depth_method_allows_live_sampler_method_set(self):
        config = {
            "allowed_depth_sampling_methods": [
                "adaptive_median_window",
                "forearm_directional_tracking",
            ]
        }
        self.assertTrue(
            calibration.depth_sampling_method_allowed(
                "forearm_directional_tracking",
                config,
            )
        )
        self.assertFalse(
            calibration.depth_sampling_method_allowed(
                "median_7x7",
                config,
            )
        )

    def test_depth_method_preserves_single_offline_contract(self):
        config = {"required_depth_sampling_method": "median_7x7"}
        self.assertTrue(
            calibration.depth_sampling_method_allowed(
                "median_7x7",
                config,
            )
        )
        self.assertFalse(
            calibration.depth_sampling_method_allowed(
                "adaptive_median_window",
                config,
            )
        )

    def test_stratified_bootstrap_is_deterministic(self):
        values = {
            "run_a": [0.24, 0.25, 0.26],
            "run_b": [0.25, 0.26, 0.27],
        }
        first = calibration.stratified_bootstrap_ci(values, 100, 123)
        second = calibration.stratified_bootstrap_ci(values, 100, 123)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
