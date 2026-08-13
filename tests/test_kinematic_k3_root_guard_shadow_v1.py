from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "tools") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "tools"))

import evaluate_kinematic_k3_root_guard_shadow_v1 as guard


class K3RootGuardShadowTests(unittest.TestCase):
    def test_first_valid_root_is_accepted(self):
        result = guard.guarded_root_depth(
            3.4,
            None,
            3.3,
            None,
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["depth_m"], 3.4)
        self.assertFalse(result["clipped"])

    def test_pelvis_motion_is_added_before_rate_limit(self):
        result = guard.guarded_root_depth(
            3.43,
            3.4,
            3.32,
            3.30,
        )
        self.assertAlmostEqual(result["prediction_m"], 3.42)
        self.assertAlmostEqual(result["depth_m"], 3.43)
        self.assertFalse(result["clipped"])

    def test_positive_root_outlier_is_limited_to_frozen_gate(self):
        result = guard.guarded_root_depth(
            3.60,
            3.40,
            3.30,
            3.30,
        )
        self.assertAlmostEqual(result["depth_m"], 3.46)
        self.assertTrue(result["clipped"])

    def test_negative_root_outlier_is_limited_to_frozen_gate(self):
        result = guard.guarded_root_depth(
            3.10,
            3.40,
            3.30,
            3.30,
        )
        self.assertAlmostEqual(result["depth_m"], 3.34)
        self.assertTrue(result["clipped"])

    def test_missing_root_uses_pelvis_compensated_prediction(self):
        result = guard.guarded_root_depth(
            None,
            3.40,
            3.32,
            3.30,
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["depth_m"], 3.42)
        self.assertTrue(result["clipped"])


if __name__ == "__main__":
    unittest.main()
