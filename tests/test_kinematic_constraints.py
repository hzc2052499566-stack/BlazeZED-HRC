from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
TOOLS_DIR = WORKSPACE / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import kinematic_constraints as constraints
import replay_kinematic_constraints as replay


def profile_for_two_bones(enabled=True):
    bones = [
        {
            "name": "right_upper_arm",
            "joint_a": "right_shoulder",
            "joint_b": "right_elbow",
            "reference_length_m": 0.25,
            "robust_scale_m": 0.005,
            "constraint_enabled": enabled,
        },
        {
            "name": "right_forearm",
            "joint_a": "right_elbow",
            "joint_b": "right_wrist",
            "reference_length_m": 0.25,
            "robust_scale_m": 0.005,
            "constraint_enabled": enabled,
        },
    ]
    while len(bones) < 8:
        index = len(bones)
        bones.append(
            {
                "name": "unused_{}".format(index),
                "joint_a": "unused_a_{}".format(index),
                "joint_b": "unused_b_{}".format(index),
                "reference_length_m": 0.3,
                "robust_scale_m": 0.005,
                "constraint_enabled": False,
            }
        )
    return {"overall_quality_status": "passed", "bones": bones}


class KinematicConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (WORKSPACE / "configs" / "kinematic_constraints.json").read_text(
                encoding="utf-8"
            )
        )
        cls.guarded_config = json.loads(
            (
                WORKSPACE
                / "configs"
                / "kinematic_constraints_v2.json"
            ).read_text(encoding="utf-8")
        )

    def setUp(self):
        self.points = {
            "right_shoulder": np.asarray([0.0, 0.0, 0.0]),
            "right_elbow": np.asarray([0.40, 0.0, 0.0]),
            "right_wrist": np.asarray([0.70, 0.0, 0.0]),
        }
        self.confidence = {joint: 99.0 for joint in self.points}
        self.valid = {joint: True for joint in self.points}

    def test_k0_is_exact_and_does_not_mutate_inputs(self):
        original = copy.deepcopy(self.points)
        result = constraints.apply_frame(
            self.points,
            self.confidence,
            self.valid,
            profile_for_two_bones(),
            self.config,
            "k0_passthrough",
        )
        for joint in self.points:
            np.testing.assert_array_equal(result["points"][joint], original[joint])
            np.testing.assert_array_equal(self.points[joint], original[joint])
        self.assertEqual(result["valid"], self.valid)

    def test_k1_rejects_distal_endpoint_of_violating_bone(self):
        result = constraints.apply_frame(
            self.points,
            self.confidence,
            self.valid,
            profile_for_two_bones(),
            self.config,
            "k1_length_gate",
        )
        self.assertFalse(result["valid"]["right_elbow"])
        self.assertFalse(result["valid"]["right_wrist"])
        self.assertTrue(result["valid"]["right_shoulder"])
        self.assertEqual(
            result["rejected_joints"],
            {"right_elbow", "right_wrist"},
        )

    def test_k2_reduces_total_length_error_and_respects_cap(self):
        before = abs(0.40 - 0.25) + abs(0.30 - 0.25)
        result = constraints.apply_frame(
            self.points,
            self.confidence,
            self.valid,
            profile_for_two_bones(),
            self.config,
            "k2_soft_projection",
        )
        after = (
            abs(
                constraints.bone_length(
                    result["points"], "right_shoulder", "right_elbow"
                )
                - 0.25
            )
            + abs(
                constraints.bone_length(
                    result["points"], "right_elbow", "right_wrist"
                )
                - 0.25
            )
        )
        self.assertLess(after, before)
        for joint, point in result["points"].items():
            displacement = np.linalg.norm(point - self.points[joint])
            self.assertLessEqual(displacement, 0.0400000001)
        self.assertEqual(result["valid"], self.valid)

    def test_low_confidence_endpoint_moves_more(self):
        confidence = dict(self.confidence)
        confidence["right_elbow"] = 20.0
        result = constraints.apply_frame(
            self.points,
            confidence,
            self.valid,
            profile_for_two_bones(),
            self.config,
            "k2_soft_projection",
        )
        shoulder_move = np.linalg.norm(
            result["points"]["right_shoulder"]
            - self.points["right_shoulder"]
        )
        elbow_move = np.linalg.norm(
            result["points"]["right_elbow"] - self.points["right_elbow"]
        )
        self.assertGreater(elbow_move, shoulder_move)

    def test_disabled_bones_do_not_modify_coordinates(self):
        result = constraints.apply_frame(
            self.points,
            self.confidence,
            self.valid,
            profile_for_two_bones(enabled=False),
            self.config,
            "k2_soft_projection",
        )
        for joint in self.points:
            np.testing.assert_array_equal(
                result["points"][joint],
                self.points[joint],
            )
        self.assertEqual(result["adjusted_joints"], set())

    def test_missing_joint_is_never_recovered(self):
        valid = dict(self.valid)
        valid["right_wrist"] = False
        points = dict(self.points)
        points.pop("right_wrist")
        result = constraints.apply_frame(
            points,
            self.confidence,
            valid,
            profile_for_two_bones(),
            self.config,
            "k2_soft_projection",
        )
        self.assertNotIn("right_wrist", result["points"])
        self.assertFalse(result["valid"]["right_wrist"])

    def test_guarded_projection_skips_out_of_domain_bones(self):
        result = constraints.apply_frame(
            self.points,
            self.confidence,
            self.valid,
            profile_for_two_bones(),
            self.guarded_config,
            "k2_guarded_projection",
        )
        for joint in self.points:
            np.testing.assert_array_equal(
                result["points"][joint],
                self.points[joint],
            )

    def test_guarded_projection_skips_connected_chain(self):
        points = {
            "right_shoulder": np.asarray([0.0, 0.0, 0.0]),
            "right_elbow": np.asarray([0.265, 0.0, 0.0]),
            "right_wrist": np.asarray([0.665, 0.0, 0.0]),
        }
        result = constraints.apply_frame(
            points,
            self.confidence,
            self.valid,
            profile_for_two_bones(),
            self.guarded_config,
            "k2_guarded_projection",
        )
        for joint in points:
            np.testing.assert_array_equal(result["points"][joint], points[joint])

    def test_guarded_projection_adjusts_inside_gate(self):
        points = {
            "right_shoulder": np.asarray([0.0, 0.0, 0.0]),
            "right_elbow": np.asarray([0.265, 0.0, 0.0]),
            "right_wrist": np.asarray([0.530, 0.0, 0.0]),
        }
        result = constraints.apply_frame(
            points,
            self.confidence,
            self.valid,
            profile_for_two_bones(),
            self.guarded_config,
            "k2_guarded_projection",
        )
        self.assertTrue(result["adjusted_joints"])
        self.assertLess(
            constraints.bone_length(
                result["points"], "right_shoulder", "right_elbow"
            ),
            0.265,
        )

    def test_guarded_variant_writes_projected_coordinates(self):
        self.assertTrue(
            replay.writes_projected_coordinates("k2_guarded_projection")
        )
        self.assertTrue(replay.writes_projected_coordinates("k2_soft_projection"))
        self.assertFalse(replay.writes_projected_coordinates("k0_passthrough"))


if __name__ == "__main__":
    unittest.main()
