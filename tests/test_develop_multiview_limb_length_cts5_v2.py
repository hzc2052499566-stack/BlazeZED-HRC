"""Contract tests for CTS5 calibrated trimmed scalar fusion."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import develop_multiview_limb_length_cts5_v2 as tool  # noqa: E402


class WeightedMedianTests(unittest.TestCase):
    def test_weighted_median_respects_cell_mass(self):
        self.assertEqual(tool.weighted_median([1, 2, 100], [0.2, 0.4, 0.4]), 2.0)

    def test_negative_weight_is_rejected(self):
        with self.assertRaises(tool.CTS5DevelopmentError):
            tool.weighted_median([1, 2], [1, -1])


class TrimRuleTests(unittest.TestCase):
    def test_complete_five_discards_both_extremes(self):
        self.assertAlmostEqual(tool.trimmed_scalar([100, 10, 12, 11, -50]), 11.0)

    def test_four_discards_both_extremes_and_averages_two(self):
        self.assertAlmostEqual(tool.trimmed_scalar([100, 10, 12, -50]), 11.0)

    def test_three_uses_median(self):
        self.assertEqual(tool.trimmed_scalar([100, 10, 12]), 12.0)

    def test_fewer_than_three_abstains(self):
        self.assertIsNone(tool.trimmed_scalar([10, 12]))


class OffsetTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for repeat, shift in (("r1", 0.0), ("r2", 10.0)):
            for phase, count in (("active", 1), ("inactive", 3)):
                for bone in tool.BONES:
                    for index in range(count):
                        row = {
                            "repeat": repeat,
                            "phase": phase,
                            "bone": bone,
                            "gt_length_mm": 100.0,
                        }
                        # Active and inactive receive equal total cell mass even
                        # though inactive contains three times as many rows.
                        error = shift + (0.0 if phase == "active" else 20.0 + index)
                        for view in tool.VIEWS:
                            row[f"{view}_length_mm"] = 100.0 + error
                        rows.append(row)
        return rows

    def test_balanced_offsets_require_all_capture_phase_bone_cells(self):
        rows = [row for row in self.make_rows()
                if not (row["repeat"] == "r2" and row["phase"] == "active"
                        and row["bone"] == "right_forearm")]
        with self.assertRaises(tool.CTS5DevelopmentError):
            tool.fit_balanced_offsets(rows, ["r1", "r2"])

    def test_balanced_offsets_are_finite_for_all_view_bone_pairs(self):
        offsets = tool.fit_balanced_offsets(self.make_rows(), ["r1", "r2"])
        self.assertEqual(set(offsets), set(tool.BONES))
        for bone in tool.BONES:
            self.assertEqual(set(offsets[bone]), set(tool.VIEWS))
            self.assertTrue(all(math.isfinite(value) for value in offsets[bone].values()))

    def test_cts5_subtracts_per_view_offsets_before_trimming(self):
        offsets = {bone: {view: float(index) for index, view in enumerate(tool.VIEWS)}
                   for bone in tool.BONES}
        lengths = {view: 200.0 + index for index, view in enumerate(tool.VIEWS)}
        estimate, count = tool.cts5_estimate(lengths, "right_forearm", offsets)
        self.assertEqual(count, 5)
        self.assertEqual(estimate, 200.0)

    def test_missing_rule_counts_only_available_values(self):
        offsets = {bone: {view: 0.0 for view in tool.VIEWS} for bone in tool.BONES}
        lengths = {view: 100.0 for view in tool.VIEWS}
        lengths["p010"] = None
        estimate, count = tool.cts5_estimate(lengths, "right_forearm", offsets)
        self.assertEqual(count, 4)
        self.assertEqual(estimate, 100.0)


if __name__ == "__main__":
    unittest.main()
