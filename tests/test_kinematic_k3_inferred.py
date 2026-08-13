from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "tools") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "tools"))

import kinematic_k3_inferred as k3


PROFILE = json.loads(
    (
        WORKSPACE
        / "configs"
        / "subject_profiles"
        / "female_police_live_visual_lite_roi065_wristv5_v2.json"
    ).read_text(encoding="utf-8")
)
CONFIG = json.loads(
    (
        WORKSPACE / "configs" / "kinematic_constraints_live_v2.json"
    ).read_text(encoding="utf-8")
)


def candidate(joint: str, x: int, y: int, depth=None) -> dict:
    return {
        "mapping": {"canonical_joint": joint},
        "pixel_x": x,
        "pixel_y": y,
        "visibility": 1.0,
        "in_image": True,
        "depth_m": depth,
    }


class KinematicK3InferredTests(unittest.TestCase):
    INTRINSICS = (370.8, 370.8, 480.0, 300.0)

    def test_inference_is_separate_and_preserves_profile_bones(self):
        candidates = [
            candidate("right_shoulder", 470, 260, 3.45),
            candidate("right_elbow", 450, 270),
            candidate("right_wrist", 430, 280),
        ]
        results = k3.infer_right_arm(
            candidates,
            PROFILE,
            CONFIG,
            self.INTRINSICS,
            elbow_branch="far",
            wrist_branch="far",
        )
        self.assertEqual([row["joint"] for row in results], [
            "right_elbow",
            "right_wrist",
        ])
        self.assertTrue(all(row["valid"] for row in results))
        self.assertTrue(
            all(
                row["provenance"] == "inferred_ray_bone_k3"
                for row in results
            )
        )
        self.assertLess(results[0]["bone_residual_m"], 1e-8)
        self.assertLess(
            results[1]["bone_residual_m"],
            k3.profile_tolerance_m(
                PROFILE,
                CONFIG,
                "right_forearm",
            ),
        )

    def test_small_ray_miss_uses_profile_bounded_soft_constraint(self):
        parent_depth = 3.4
        parent_pixel = (480, 300)
        fx, fy, cx, cy = self.INTRINSICS
        parent_point = k3.reconstruct_from_depth(
            parent_depth,
            *parent_pixel,
            fx,
            fy,
            cx,
            cy,
        )
        result = k3.infer_child(
            "right_wrist",
            candidate("right_wrist", 508, 300),
            parent_depth,
            parent_pixel,
            parent_point,
            "inferred_right_elbow",
            0.25038310820497534,
            0.020030648656398028,
            "far",
            self.INTRINSICS,
        )
        self.assertTrue(result["valid"])
        self.assertGreaterEqual(result["constraint_relaxation_m"], 0.0)
        self.assertLessEqual(
            result["constraint_relaxation_m"],
            0.020030648656398028,
        )

    def test_missing_parent_emits_invalid_inferred_row(self):
        result = k3.infer_child(
            "right_wrist",
            candidate("right_wrist", 430, 280),
            None,
            None,
            None,
            "inferred_right_elbow",
            0.25,
            0.02,
            "far",
            self.INTRINSICS,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_reason"], "parent_unavailable")


if __name__ == "__main__":
    unittest.main()
