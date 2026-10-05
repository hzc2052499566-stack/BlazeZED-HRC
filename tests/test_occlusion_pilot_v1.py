from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import occlusion_pilot_v1 as pilot  # noqa: E402
from fs_cts5_camera_bank_v1 import JOINTS  # noqa: E402


INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
WIDTH = 640
HEIGHT = 480


def flat_depth(value):
    return [[value] * WIDTH for _ in range(HEIGHT)]


def standing_positions(x=0.0):
    """A simple upright pose, all joints in front of a camera on -Y."""
    return {
        "left_shoulder": (x + 0.18, 0.0, 1.45),
        "right_shoulder": (x - 0.18, 0.0, 1.45),
        "left_elbow": (x + 0.30, 0.0, 1.20),
        "right_elbow": (x - 0.30, 0.0, 1.20),
        "left_wrist": (x + 0.35, 0.0, 0.98),
        "right_wrist": (x - 0.35, 0.0, 0.98),
        "left_hip": (x + 0.10, 0.0, 0.90),
        "right_hip": (x - 0.10, 0.0, 0.90),
        "left_knee": (x + 0.10, 0.0, 0.48),
        "right_knee": (x - 0.10, 0.0, 0.48),
        "left_ankle": (x + 0.10, 0.0, 0.06),
        "right_ankle": (x - 0.10, 0.0, 0.06),
    }


def view(depth, eye=(0.0, -3.0, 1.2), aim=(0.0, 0.0, 1.2)):
    return {"eye": eye, "aim": aim, "intrinsics": INTRINSICS, "depth": depth}


class RingTests(unittest.TestCase):
    def test_ring_covers_every_azimuth_and_elevation(self):
        poses = pilot.ring_camera_poses((0.0, 0.0, 1.0), 3.5, (0, 90, 180, 270), (-10, 0, 20))
        self.assertEqual(len(poses), 12)
        names = {pose["name"] for pose in poses}
        self.assertEqual(len(names), 12)

    def test_eyes_sit_at_the_requested_radius(self):
        aim = (1.0, -2.0, 1.1)
        poses = pilot.ring_camera_poses(aim, 3.5, (0, 45), (0,))
        for pose in poses:
            distance = sum(
                (pose["eye"][axis] - aim[axis]) ** 2 for axis in range(3)
            ) ** 0.5
            self.assertAlmostEqual(distance, 3.5, places=9)

    def test_elevation_raises_the_eye(self):
        low = pilot.ring_camera_poses((0, 0, 1.0), 3.0, (0,), (-10,))[0]
        high = pilot.ring_camera_poses((0, 0, 1.0), 3.0, (0,), (30,))[0]
        self.assertLess(low["eye"][2], high["eye"][2])

    def test_a_zero_radius_raises(self):
        with self.assertRaises(pilot.OcclusionPilotError):
            pilot.ring_camera_poses((0, 0, 0), 0.0, (0,), (0,))


class DepthSamplingTests(unittest.TestCase):
    def test_in_frame_sample_returns_the_value(self):
        depth = flat_depth(2.5)
        self.assertAlmostEqual(pilot.sample_depth(depth, 100.4, 200.6), 2.5, places=9)

    def test_out_of_frame_returns_none(self):
        depth = flat_depth(2.5)
        self.assertIsNone(pilot.sample_depth(depth, -5.0, 100.0))
        self.assertIsNone(pilot.sample_depth(depth, 100.0, HEIGHT + 3.0))

    def test_non_finite_or_zero_depth_returns_none(self):
        depth = flat_depth(0.0)
        self.assertIsNone(pilot.sample_depth(depth, 10.0, 10.0))
        depth = flat_depth(float("inf"))
        self.assertIsNone(pilot.sample_depth(depth, 10.0, 10.0))

    def test_empty_image_raises(self):
        with self.assertRaises(pilot.OcclusionPilotError):
            pilot.sample_depth([], 1.0, 1.0)


class ClassificationTests(unittest.TestCase):
    def test_a_clear_view_reads_visible(self):
        result = pilot.classify_endpoint(
            (0.0, 0.0, 1.2), (0.0, -3.0, 1.2), (0.0, 0.0, 1.2), INTRINSICS, flat_depth(3.0)
        )
        self.assertEqual(result["state"], "visible")

    def test_something_nearer_reads_occluded(self):
        result = pilot.classify_endpoint(
            (0.0, 0.0, 1.2), (0.0, -3.0, 1.2), (0.0, 0.0, 1.2), INTRINSICS, flat_depth(1.5)
        )
        self.assertEqual(result["state"], "occluded")
        self.assertAlmostEqual(result["depth_gap_m"], 1.5, places=6)

    def test_the_margin_absorbs_skin_depth(self):
        # A surface 4 cm nearer than the pivot is the body itself, not an occluder.
        result = pilot.classify_endpoint(
            (0.0, 0.0, 1.2), (0.0, -3.0, 1.2), (0.0, 0.0, 1.2), INTRINSICS, flat_depth(2.96)
        )
        self.assertEqual(result["state"], "visible")

    def test_a_point_behind_the_camera_is_reported(self):
        result = pilot.classify_endpoint(
            (0.0, -5.0, 1.2), (0.0, -3.0, 1.2), (0.0, 0.0, 1.2), INTRINSICS, flat_depth(3.0)
        )
        self.assertEqual(result["state"], "behind_camera")

    def test_a_point_outside_the_image_is_reported(self):
        result = pilot.classify_endpoint(
            (9.0, 0.0, 1.2), (0.0, -3.0, 1.2), (0.0, 0.0, 1.2), INTRINSICS, flat_depth(3.0)
        )
        self.assertEqual(result["state"], "out_of_frame")


class FrameTests(unittest.TestCase):
    def test_all_clear_gives_every_bone_every_view(self):
        views = [view(flat_depth(10.0)) for _ in range(5)]
        result = pilot.evaluate_frame(standing_positions(), views)
        for bone, entry in result["bones"].items():
            self.assertEqual(entry["measured_view_count"], 5, bone)
            self.assertTrue(entry["meets_minimum"], bone)

    def test_occluding_three_of_five_views_drops_below_the_minimum(self):
        views = [view(flat_depth(10.0)) for _ in range(2)] + [
            view(flat_depth(0.5)) for _ in range(3)
        ]
        result = pilot.evaluate_frame(standing_positions(), views)
        for bone, entry in result["bones"].items():
            self.assertEqual(entry["measured_view_count"], 2, bone)
            self.assertFalse(entry["meets_minimum"], bone)

    def test_a_missing_joint_raises(self):
        positions = standing_positions()
        del positions["right_wrist"]
        with self.assertRaises(pilot.OcclusionPilotError):
            pilot.evaluate_frame(positions, [view(flat_depth(10.0))])

    def test_no_views_raises(self):
        with self.assertRaises(pilot.OcclusionPilotError):
            pilot.evaluate_frame(standing_positions(), [])

    def test_every_mapped_joint_is_classified(self):
        result = pilot.evaluate_frame(standing_positions(), [view(flat_depth(10.0))])
        self.assertEqual(set(result["endpoints"]), set(JOINTS))


class SummaryTests(unittest.TestCase):
    def frames(self, depth_value, count=4):
        views = [view(flat_depth(depth_value)) for _ in range(5)]
        return [pilot.evaluate_frame(standing_positions(), views) for _ in range(count)]

    def test_clear_frames_report_no_occlusion(self):
        summary = pilot.summarise(self.frames(10.0), ["a", "b", "c", "d", "e"])
        self.assertAlmostEqual(summary["leg_occlusion_rate"], 0.0, places=9)
        self.assertAlmostEqual(summary["target_arm_occlusion_rate"], 0.0, places=9)

    def test_fully_occluded_frames_report_full_rates(self):
        summary = pilot.summarise(self.frames(0.5), ["a", "b", "c", "d", "e"])
        self.assertAlmostEqual(summary["leg_occlusion_rate"], 1.0, places=9)
        self.assertAlmostEqual(summary["target_arm_occlusion_rate"], 1.0, places=9)

    def test_shortfall_counts_frames_below_the_three_view_minimum(self):
        summary = pilot.summarise(self.frames(0.5), ["a", "b", "c", "d", "e"])
        for bone, entry in summary["bone_view_shortfall"].items():
            self.assertEqual(entry["below_minimum"], entry["frames"], bone)

    def test_summary_states_it_is_a_pilot(self):
        summary = pilot.summarise(self.frames(10.0), ["a", "b", "c", "d", "e"])
        self.assertIn("Pilot viewpoints only", summary["note"])
        self.assertIn("step 4", summary["note"])
        self.assertFalse(summary["formal_capture_authorized"])

    def test_no_frames_raises(self):
        with self.assertRaises(pilot.OcclusionPilotError):
            pilot.summarise([], ["a"])


class DepthScaleTests(unittest.TestCase):
    """attempt_01 read 100% occlusion everywhere, including idle frames.

    Rendered depth arrives in the stage's declared units (metersPerUnit = 0.01)
    while joint depths are computed in operational metres, so every comparison
    was 0.035 against 3.5 and every endpoint looked occluded.
    """

    def test_matched_scales_are_plausible(self):
        result = pilot.depth_consistency(
            [{"rendered_depth_m": 3.45, "joint_depth_m": 3.5}] * 5
        )
        self.assertTrue(result["plausible"])
        self.assertEqual(result["diagnosis"], "")

    def test_a_hundredfold_mismatch_is_diagnosed_not_reported_as_occlusion(self):
        result = pilot.depth_consistency(
            [{"rendered_depth_m": 0.0345, "joint_depth_m": 3.5}] * 5
        )
        self.assertFalse(result["plausible"])
        self.assertIn("unit scale mismatch", result["diagnosis"])

    def test_the_inverse_mismatch_is_caught_too(self):
        result = pilot.depth_consistency(
            [{"rendered_depth_m": 345.0, "joint_depth_m": 3.5}] * 5
        )
        self.assertFalse(result["plausible"])

    def test_genuine_occlusion_stays_plausible(self):
        # Half the samples occluded by 20 cm: still the same scale.
        samples = [{"rendered_depth_m": 3.3, "joint_depth_m": 3.5}] * 5
        samples += [{"rendered_depth_m": 3.48, "joint_depth_m": 3.5}] * 5
        self.assertTrue(pilot.depth_consistency(samples)["plausible"])

    def test_no_samples_raises(self):
        with self.assertRaises(pilot.OcclusionPilotError):
            pilot.depth_consistency([{"state": "behind_camera"}])


class DifferentialTests(unittest.TestCase):
    """attempt_02 counted the character's own skin as occlusion.

    A joint pivot sits 6-8 cm inside the body, so a fixed margin against the
    pivot cannot separate skin from props.  The same view is rendered with the
    props hidden and the difference is what counts.
    """

    def classify(self, with_props, without_props):
        return pilot.classify_endpoint_differential(
            (0.0, 0.0, 1.2),
            (0.0, -3.0, 1.2),
            (0.0, 0.0, 1.2),
            INTRINSICS,
            flat_depth(with_props),
            flat_depth(without_props),
        )

    def test_skin_in_front_of_the_pivot_is_not_occlusion(self):
        result = self.classify(2.94, 2.94)
        self.assertEqual(result["state"], "visible")
        self.assertAlmostEqual(result["prop_depth_gap_m"], 0.0, places=9)
        self.assertAlmostEqual(result["self_depth_gap_m"], 0.06, places=6)

    def test_a_prop_in_front_reads_occluded(self):
        result = self.classify(2.40, 2.94)
        self.assertEqual(result["state"], "occluded")
        self.assertAlmostEqual(result["prop_depth_gap_m"], 0.54, places=6)

    def test_a_prop_just_inside_the_margin_does_not_count(self):
        result = self.classify(2.93, 2.94)
        self.assertEqual(result["state"], "visible")

    def test_frames_use_the_differential_path_when_a_baseline_is_present(self):
        views = [
            {
                "eye": (0.0, -3.0, 1.2),
                "aim": (0.0, 0.0, 1.2),
                "intrinsics": INTRINSICS,
                "depth": flat_depth(2.9),
                "depth_clear": flat_depth(2.9),
            }
            for _ in range(4)
        ]
        result = pilot.evaluate_frame(standing_positions(), views)
        for states in result["endpoints"].values():
            for state in states:
                self.assertIn(state["state"], ("visible", "out_of_frame"))

    def test_the_prop_margin_is_far_below_body_thickness(self):
        self.assertLess(pilot.PROP_DEPTH_MARGIN_M, 0.05)


class SharedRingTests(unittest.TestCase):
    """The search and the render must score the same cameras.

    attempt_04/05 measured 0% arm occlusion while the search predicted four
    blocked views, because the search aimed its ring at the forearm midpoint
    and the pilot aimed its own at the mid-body: same azimuths, different eyes.
    """

    def test_aim_is_mid_body_not_the_arm(self):
        positions = standing_positions()
        aim = pilot.ring_aim_point(positions)
        self.assertAlmostEqual(aim[0], 0.0, places=6)
        self.assertAlmostEqual(aim[1], 0.0, places=6)
        self.assertAlmostEqual(aim[2], (0.90 + 1.45) / 2.0, places=6)

    def test_the_standard_ring_has_one_definition(self):
        ring = pilot.standard_ring(standing_positions())
        self.assertEqual(
            len(ring), len(pilot.RING_AZIMUTHS_DEG) * len(pilot.RING_ELEVATIONS_DEG)
        )
        for pose in ring:
            distance = sum(
                (pose["eye"][axis] - pose["aim"][axis]) ** 2 for axis in range(3)
            ) ** 0.5
            self.assertAlmostEqual(distance, pilot.RING_RADIUS_M, places=6)

    def test_every_pose_shares_the_same_aim(self):
        ring = pilot.standard_ring(standing_positions())
        aims = {tuple(pose["aim"]) for pose in ring}
        self.assertEqual(len(aims), 1)

    def test_a_missing_joint_raises_rather_than_guessing_an_aim(self):
        positions = standing_positions()
        del positions["right_shoulder"]
        with self.assertRaises(pilot.OcclusionPilotError):
            pilot.ring_aim_point(positions)


class AvailabilityTests(unittest.TestCase):
    """Availability is GT-free: in frame, no prop, and not behind the body."""

    def test_a_clear_endpoint_is_available(self):
        self.assertTrue(
            pilot.endpoint_available({"state": "visible", "self_depth_gap_m": 0.07})
        )

    def test_a_prop_occluded_endpoint_is_not(self):
        self.assertFalse(
            pilot.endpoint_available({"state": "occluded", "self_depth_gap_m": 0.07})
        )

    def test_an_out_of_frame_endpoint_is_not(self):
        self.assertFalse(pilot.endpoint_available({"state": "out_of_frame"}))

    def test_a_self_occluded_endpoint_is_not(self):
        # Another body part 0.6 m in front of the pivot, not the limb's own skin.
        self.assertFalse(
            pilot.endpoint_available({"state": "visible", "self_depth_gap_m": 0.6})
        )

    def test_skin_depth_still_counts_as_available(self):
        for gap in (0.0, 0.06, 0.08, 0.24):
            self.assertTrue(
                pilot.endpoint_available({"state": "visible", "self_depth_gap_m": gap}),
                gap,
            )

    def test_a_missing_baseline_does_not_invent_self_occlusion(self):
        self.assertTrue(pilot.endpoint_available({"state": "visible"}))


class HygieneTests(unittest.TestCase):
    def test_module_has_no_isaac_dependency(self):
        source = (TOOLS / "occlusion_pilot_v1.py").read_text(encoding="utf-8")
        for forbidden in ("import omni", "from pxr", "import carb"):
            self.assertNotIn(forbidden, source)

    def test_it_reuses_the_validated_camera_maths(self):
        source = (TOOLS / "occlusion_pilot_v1.py").read_text(encoding="utf-8")
        self.assertIn("from fs_cts5_camera_bank_v1 import", source)

    def test_the_minimum_matches_the_five_view_policy(self):
        self.assertEqual(pilot.MIN_MEASURED_VIEWS, 3)


if __name__ == "__main__":
    unittest.main()
