from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import depth_sampling_core as sampling


class DepthSamplingCoreTests(unittest.TestCase):
    def test_point_rejects_invalid_centre(self):
        depth = np.full((9, 9), 3.0, dtype=np.float32)
        depth[4, 4] = np.nan
        result = sampling.point_depth(depth, 4, 4)
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_reason"], "invalid_centre_pixel")

    def test_median_recovers_from_invalid_centre(self):
        depth = np.full((9, 9), 3.0, dtype=np.float32)
        depth[4, 4] = np.nan
        result = sampling.median_window_depth(
            depth,
            4,
            4,
            1,
            method="median_3x3",
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["depth_m"], 3.0)
        self.assertEqual(result["valid_pixel_count"], 8)

    def test_adaptive_expands_after_empty_inner_window(self):
        depth = np.full((21, 21), np.nan, dtype=np.float32)
        depth[4:17, 4:17] = 2.5
        depth[7:14, 7:14] = np.nan
        result = sampling.adaptive_median_depth(depth, 10, 10)
        self.assertTrue(result["valid"])
        self.assertEqual(result["window_radius_px"], 5)
        self.assertAlmostEqual(result["depth_m"], 2.5)

    def test_limb_corridor_prefers_kinematic_surface(self):
        depth = np.full((50, 50), 4.0, dtype=np.float32)
        depth[24:27, 20:31] = 3.0
        result = sampling.clustered_depth(
            depth,
            20,
            25,
            search_radius_px=20,
            cluster_gap_m=0.03,
            spatial_band_px=12.0,
            minimum_cluster_size=5,
            preferred_direction_xy=(15.0, 0.0),
            kinematic_reference_m=3.0,
            method="limb_aware_wrist_cluster",
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["directional_constraint_used"])
        self.assertAlmostEqual(result["depth_m"], 3.0)

    def test_all_invalid_returns_explicit_reason(self):
        depth = np.full((11, 11), np.nan, dtype=np.float32)
        result = sampling.adaptive_median_depth(depth, 5, 5)
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_reason"], "insufficient_valid_pixels")


if __name__ == "__main__":
    unittest.main()
