from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import ur10e_kinematics_v1 as kin  # noqa: E402


# Independently measured from the asset by the scene-asset measurement step.
MEASURED_BBOX_MIN = (-0.095, -0.095, 0.0)
MEASURED_BBOX_MAX = (1.231, 0.291, 0.275)


class ChainTests(unittest.TestCase):
    def test_the_chain_matches_the_asset_the_inspector_read(self):
        self.assertEqual(len(kin.JOINT_CHAIN), 6)
        self.assertEqual(kin.LINK_ORDER[0], "base_link")
        self.assertEqual(kin.LINK_ORDER[-1], "wrist_3_link")
        self.assertEqual(len(kin.LINK_ORDER), 7)

    def test_every_link_gets_a_transform(self):
        transforms = kin.link_transforms(kin.zero_pose())
        self.assertEqual(set(transforms), set(kin.LINK_ORDER))

    def test_base_link_is_the_identity(self):
        matrix = kin.link_transforms(kin.zero_pose())["base_link"]
        self.assertEqual(matrix, kin.identity4())

    def test_a_missing_angle_raises(self):
        angles = kin.zero_pose()
        del angles["elbow_joint"]
        with self.assertRaises(kin.KinematicsError):
            kin.link_transforms(angles)

    def test_an_unknown_joint_raises(self):
        angles = kin.zero_pose()
        angles["gripper_joint"] = 0.0
        with self.assertRaises(kin.KinematicsError):
            kin.link_transforms(angles)

    def test_the_assets_own_limits_are_enforced(self):
        angles = kin.zero_pose()
        angles["elbow_joint"] = 200.0
        with self.assertRaises(kin.KinematicsError):
            kin.link_transforms(angles)


class MeasuredAgreementTests(unittest.TestCase):
    """Cross-check the computed chain against an independently measured bound.

    The asset measurement step recorded the UR10e's bounding box by a completely
    separate code path.  At the zero pose the wrist sits at the far corner of
    that box, so agreement there means the chain composition is right.
    """

    def test_zero_pose_wrist_lands_at_the_measured_corner(self):
        wrist = kin.link_origins(kin.zero_pose())["wrist_3_link"]
        self.assertAlmostEqual(wrist[1], MEASURED_BBOX_MAX[1], delta=0.005)
        # x falls short by the wrist link's own mesh, which the bbox includes.
        self.assertLess(wrist[0], MEASURED_BBOX_MAX[0])
        self.assertGreater(wrist[0], MEASURED_BBOX_MAX[0] - 0.10)

    def test_zero_pose_stays_inside_the_measured_height(self):
        origins = kin.link_origins(kin.zero_pose())
        for link, position in origins.items():
            self.assertLessEqual(position[2], MEASURED_BBOX_MAX[2] + 0.01, link)

    def test_the_default_pose_is_too_low_for_a_chest_height_arm(self):
        # This is why the occlusion pilot measured 0% arm occlusion.
        self.assertLess(kin.reach_height(kin.zero_pose()), 0.3)


class PoseTests(unittest.TestCase):
    def raised(self, lift=-90.0, elbow=-45.0):
        angles = kin.zero_pose()
        angles["shoulder_lift_joint"] = lift
        angles["elbow_joint"] = elbow
        return angles

    def test_lifting_the_shoulder_raises_the_wrist_to_chest_height(self):
        self.assertGreater(kin.reach_height(self.raised()), 1.2)

    def test_a_vertical_pose_stacks_the_links(self):
        origins = kin.link_origins(self.raised(elbow=0.0))
        self.assertAlmostEqual(origins["forearm_link"][0], 0.0, places=6)
        self.assertAlmostEqual(origins["forearm_link"][1], 0.0, places=6)
        self.assertGreater(origins["wrist_1_link"][2], origins["forearm_link"][2])

    def test_bending_the_elbow_moves_the_wrist_off_the_column(self):
        straight = kin.link_origins(self.raised(elbow=0.0))["wrist_1_link"]
        bent = kin.link_origins(self.raised(elbow=-45.0))["wrist_1_link"]
        self.assertGreater(abs(bent[0] - straight[0]), 0.3)

    def test_upper_arm_length_matches_the_asset(self):
        origins = kin.link_origins(kin.zero_pose())
        span = (
            (origins["forearm_link"][0] - origins["upper_arm_link"][0]) ** 2
            + (origins["forearm_link"][1] - origins["upper_arm_link"][1]) ** 2
            + (origins["forearm_link"][2] - origins["upper_arm_link"][2]) ** 2
        ) ** 0.5
        self.assertAlmostEqual(span, 0.6127, places=4)

    def test_panning_rotates_the_whole_arm_about_the_base(self):
        angles = self.raised()
        straight = kin.link_origins(angles)["wrist_1_link"]
        angles["shoulder_pan_joint"] = 90.0
        panned = kin.link_origins(angles)["wrist_1_link"]
        self.assertAlmostEqual(straight[2], panned[2], places=6)
        radius_before = (straight[0] ** 2 + straight[1] ** 2) ** 0.5
        radius_after = (panned[0] ** 2 + panned[1] ** 2) ** 0.5
        self.assertAlmostEqual(radius_before, radius_after, places=6)


class MathTests(unittest.TestCase):
    def test_quaternion_identity(self):
        self.assertEqual(kin.quaternion_matrix((1.0, 0.0, 0.0, 0.0)), kin.identity4())

    def test_quaternion_ninety_about_x(self):
        matrix = kin.quaternion_matrix((0.70710677, 0.70710677, 0.0, 0.0))
        point = kin.transform_point((0.0, 1.0, 0.0), matrix)
        self.assertAlmostEqual(point[1], 0.0, places=6)
        self.assertAlmostEqual(point[2], 1.0, places=6)

    def test_zero_quaternion_raises(self):
        with self.assertRaises(kin.KinematicsError):
            kin.quaternion_matrix((0.0, 0.0, 0.0, 0.0))

    def test_rotation_z_is_a_rotation(self):
        matrix = kin.rotation_z(90.0)
        point = kin.transform_point((1.0, 0.0, 0.0), matrix)
        self.assertAlmostEqual(point[0], 0.0, places=9)
        self.assertAlmostEqual(point[1], 1.0, places=9)


class HygieneTests(unittest.TestCase):
    def test_module_has_no_isaac_dependency(self):
        source = (TOOLS / "ur10e_kinematics_v1.py").read_text(encoding="utf-8")
        for forbidden in ("import omni", "from pxr", "import carb"):
            self.assertNotIn(forbidden, source)

    def test_the_flat_hierarchy_finding_is_recorded(self):
        source = (TOOLS / "ur10e_kinematics_v1.py").read_text(encoding="utf-8")
        self.assertIn("flat siblings", source)


if __name__ == "__main__":
    unittest.main()
