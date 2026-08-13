"""Unit tests for the pre-lock relock patch, method version v2."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kinematic_prelock_relock_v2 as relock  # noqa: E402


def rejected(depth=None, candidate=3.3448):
    return {
        "depth_m": depth,
        "temporal_rejected": True,
        "rejected_candidate_depth_m": candidate,
    }


class PrelockDiscriminatorTests(unittest.TestCase):
    def test_equal_reference_and_previous_is_prelock(self):
        self.assertTrue(relock._is_prelock_reference(3.2699, 3.2699))

    def test_distinct_values_are_not_prelock(self):
        """After lock the caller passes a real measurement, not the prior."""
        self.assertFalse(relock._is_prelock_reference(3.3397, 3.2797))

    def test_missing_values_are_not_prelock(self):
        self.assertFalse(relock._is_prelock_reference(None, 3.2699))
        self.assertFalse(relock._is_prelock_reference(3.2699, None))
        self.assertFalse(relock._is_prelock_reference(None, None))


class RelockResultTests(unittest.TestCase):
    def test_prelock_rejection_is_accepted_at_the_rejected_candidate(self):
        out = relock.relock_result(rejected(), prelock=True)
        self.assertEqual(out["depth_m"], 3.3448)
        self.assertFalse(out["temporal_rejected"])
        self.assertTrue(out["prelock_relock_applied"])

    def test_post_lock_rejection_is_left_alone(self):
        out = relock.relock_result(rejected(), prelock=False)
        self.assertIsNone(out["depth_m"])
        self.assertTrue(out["temporal_rejected"])
        self.assertNotIn("prelock_relock_applied", out)

    def test_accepted_result_is_untouched(self):
        source = {
            "depth_m": 3.3443,
            "temporal_rejected": False,
            "rejected_candidate_depth_m": None,
        }
        self.assertIs(relock.relock_result(source, prelock=True), source)

    def test_rejection_without_a_candidate_is_left_alone(self):
        out = relock.relock_result(
            rejected(candidate=None), prelock=True
        )
        self.assertIsNone(out["depth_m"])

    def test_original_result_is_not_mutated(self):
        source = rejected()
        relock.relock_result(source, prelock=True)
        self.assertIsNone(source["depth_m"])
        self.assertTrue(source["temporal_rejected"])


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType("fake_pipeline")
        self.module.__file__ = str(TOOLS / "kinematic_prelock_relock_v2.py")
        self.calls = []

        def clustered_valid_depth_m(
            depth,
            pixel_x,
            pixel_y,
            search_radius_px=12,
            previous_depth_m=None,
            kinematic_reference_m=None,
        ):
            self.calls.append((previous_depth_m, kinematic_reference_m))
            return rejected()

        self.module.clustered_valid_depth_m = clustered_valid_depth_m

    def test_patch_rescues_the_prelock_case(self):
        relock.apply(self.module)
        out = self.module.clustered_valid_depth_m(
            None, 451, 275,
            previous_depth_m=3.2699,
            kinematic_reference_m=3.2699,
        )
        self.assertEqual(out["depth_m"], 3.3448)

    def test_patch_leaves_the_tracking_case_alone(self):
        relock.apply(self.module)
        out = self.module.clustered_valid_depth_m(
            None, 451, 275,
            previous_depth_m=3.3397,
            kinematic_reference_m=3.2797,
        )
        self.assertIsNone(out["depth_m"])

    def test_patch_binds_positional_arguments(self):
        relock.apply(self.module)
        out = self.module.clustered_valid_depth_m(
            None, 451, 275, 12, 3.2699, 3.2699
        )
        self.assertEqual(out["depth_m"], 3.3448)

    def test_double_apply_is_refused(self):
        relock.apply(self.module)
        with self.assertRaises(RuntimeError):
            relock.apply(self.module)

    def test_revert_restores_the_original(self):
        original = self.module.clustered_valid_depth_m
        relock.apply(self.module)
        relock.revert(self.module)
        self.assertIs(self.module.clustered_valid_depth_m, original)

    def test_revert_without_apply_is_refused(self):
        with self.assertRaises(RuntimeError):
            relock.revert(self.module)

    def test_provenance_records_both_module_hashes(self):
        provenance = relock.apply(self.module)
        self.assertEqual(
            provenance["method_version"], relock.METHOD_VERSION
        )
        self.assertEqual(len(provenance["patched_module_sha256"]), 64)
        self.assertEqual(len(provenance["patch_module_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
