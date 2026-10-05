from __future__ import annotations

import unittest

from tools import fs_cts5_known_occluder_surface_rejection_v1 as target


class KnownOccluderSurfacePrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = {"eye_world_operational_m": [0.0, 0.0, 0.0], "aim_world_operational_m": [1.0, 0.0, 0.0]}
        self.obb = {
            "centre_world_operational_m": [3.0, 0.0, 0.0],
            "axes_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "half_extents_operational_m": [0.05, 1.0, 1.0],
        }

    @staticmethod
    def row(x: float, *, valid: int = 1):
        return {
            "valid": str(valid), "counts_as_measured_valid": "1",
            "method": "raw_measured", "provenance": "raw_measured",
            "x_m": str(x), "y_m": "0", "z_m": "0",
        }

    def test_active_surface_match_is_rejected_with_new_provenance(self) -> None:
        result = target.classify_raw_joint(self.row(3.0), self.camera, [self.obb], active=True)
        self.assertTrue(result["occluder_rejected"])
        self.assertEqual(result["output_provenance"], "occluder_rejected")
        self.assertAlmostEqual(result["surface_depth_residual_m"], 0.05, places=12)

    def test_inactive_is_identity(self) -> None:
        result = target.classify_raw_joint(self.row(3.0), self.camera, [self.obb], active=False)
        self.assertFalse(result["occluder_rejected"])
        self.assertEqual(result["output_provenance"], "raw_measured")

    def test_surface_mismatch_remains_raw_measured(self) -> None:
        result = target.classify_raw_joint(self.row(3.2), self.camera, [self.obb], active=True)
        self.assertFalse(result["occluder_rejected"])
        self.assertEqual(result["output_provenance"], "raw_measured")

    def test_invalid_is_not_relabelled(self) -> None:
        result = target.classify_raw_joint(self.row(3.0, valid=0), self.camera, [self.obb], active=True)
        self.assertFalse(result["occluder_rejected"])
        self.assertEqual(result["output_provenance"], "raw_invalid")

    def test_filtered_bone_requires_two_nonrejected_raw_endpoints(self) -> None:
        first = self.row(3.0)
        second = self.row(3.3)
        keep = target.classify_raw_joint(first, self.camera, [self.obb], active=False)
        reject = target.classify_raw_joint(first, self.camera, [self.obb], active=True)
        other = target.classify_raw_joint(second, self.camera, [self.obb], active=True)
        self.assertTrue(target.filtered_bone_available(first, second, keep, other))
        self.assertFalse(target.filtered_bone_available(first, second, reject, other))


if __name__ == "__main__":
    unittest.main()
