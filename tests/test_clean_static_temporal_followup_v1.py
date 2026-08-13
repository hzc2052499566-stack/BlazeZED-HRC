from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyse_clean_static_temporal_followup_v1 as followup


class CleanStaticTemporalFollowupTests(unittest.TestCase):
    def test_vector_jitter_combines_axis_population_sd(self):
        points = [(0.0, 0.0, 0.0), (0.002, 0.0, 0.0)]
        self.assertAlmostEqual(followup.vector_jitter(points, 1000.0), 1.0)

    def test_vector_jitter_returns_none_for_one_point(self):
        self.assertIsNone(followup.vector_jitter([(1.0, 2.0)], 1.0))

    def test_distance_mm(self):
        self.assertAlmostEqual(
            followup.distance_mm((0.0, 0.0, 0.0), (0.003, 0.004, 0.0)),
            5.0,
        )


if __name__ == "__main__":
    unittest.main()
