"""Unit tests for the adaptive camera positioning objective.

Every case here is synthetic and deterministic.  The reproductions of the
frozen measured outcomes (view A blocked, the 0.12 m stereo baseline clearing
the elbow but not the wrist, view C clear, and the 37.7 / 41.9 px separations)
belong to the scan driver, which records them as validity gates in its own
report rather than hiding them in the test suite.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import adaptive_camera_objective_v1 as objective  # noqa: E402


FX = FY = 370.8000035762786
CX, CY = 480.0, 300.0
WIDTH, HEIGHT = 960, 600

# Recorded in camera_extrinsics_operational_v1.json for the measured-first
# formal capture; embedded as constants so the test needs no data files.
VIEW_A_EYE = (4.280518825024692, 0.9069533930182956, 1.4980182539216949)
VIEW_A_ROTATION = (
    (-0.7328786014196136, 0.6803594311694603, 0.0),
    (-0.6803594311694604, -0.7328786014196139, -2.220446049250313e-16),
    (2.220446049250313e-16, -2.220446049250313e-16, 1.0000000000000004),
)
PELVIS_FRAME_0 = (1.7154437200560442, -1.4743046160748159, 1.4980182539216955)


def camera_at(eye, target):
    return objective.LookAtCamera.look_at(eye, target, FX, FY, CX, CY, WIDTH, HEIGHT)


def unit_box(centre=(0.0, 0.0, 0.0), half=(1.0, 1.0, 1.0)):
    return objective.OrientedBox(centre, np.eye(3), half)


class CameraTests(unittest.TestCase):
    def test_look_at_basis_is_orthonormal_and_right_handed(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        np.testing.assert_allclose(camera.basis @ camera.basis.T, np.eye(3), atol=1e-12)
        forward, left, up = camera.basis
        np.testing.assert_allclose(np.cross(forward, left), up, atol=1e-12)

    def test_look_at_reproduces_the_recorded_view_a_rotation(self):
        target = (PELVIS_FRAME_0[0], PELVIS_FRAME_0[1], VIEW_A_EYE[2])
        camera = camera_at(VIEW_A_EYE, target)
        # from_extrinsics transposes, so basis rows must match rotation columns.
        np.testing.assert_allclose(camera.basis, np.array(VIEW_A_ROTATION).T, atol=1e-12)

    def test_from_extrinsics_matches_look_at(self):
        target = (PELVIS_FRAME_0[0], PELVIS_FRAME_0[1], VIEW_A_EYE[2])
        built = objective.LookAtCamera.from_extrinsics(
            VIEW_A_ROTATION, VIEW_A_EYE, FX, FY, CX, CY, WIDTH, HEIGHT)
        np.testing.assert_allclose(built.basis, camera_at(VIEW_A_EYE, target).basis, atol=1e-12)

    def test_pelvis_projects_to_the_principal_point_when_aimed_at_it(self):
        target = (PELVIS_FRAME_0[0], PELVIS_FRAME_0[1], VIEW_A_EYE[2])
        pixel, forward = camera_at(VIEW_A_EYE, target).project(PELVIS_FRAME_0)
        np.testing.assert_allclose(pixel, [CX, CY], atol=1e-6)
        self.assertAlmostEqual(forward, 3.5, places=6)

    def test_projection_inverts_the_frozen_back_projection(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        point = (0.4, -0.3, 1.9)
        pixel, forward = camera.project(point)
        # agents.md 3.1: X = depth, Y = -(u - cx) * depth / fx, Z = -(v - cy) * depth / fy
        recovered_camera = np.array([
            forward,
            -(pixel[0] - CX) * forward / FX,
            -(pixel[1] - CY) * forward / FY,
        ])
        np.testing.assert_allclose(recovered_camera, camera.camera_point(point), atol=1e-9)

    def test_point_behind_the_camera_does_not_project(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        self.assertIsNone(camera.project((5.0, 0.0, 1.5)))

    def test_roi_window_is_centred_and_inclusive_at_the_edge(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        half_w = WIDTH * objective.ROI_SCALE / 2.0
        self.assertTrue(camera.inside_roi((CX + half_w, CY)))
        self.assertFalse(camera.inside_roi((CX + half_w + 1e-6, CY)))
        self.assertFalse(camera.inside_roi((CX, CY + HEIGHT * objective.ROI_SCALE / 2.0 + 1e-6)))

    def test_degenerate_look_at_is_rejected(self):
        with self.assertRaises(objective.ObjectiveError):
            camera_at((0.0, 0.0, 0.0), (0.0, 0.0, 2.0))
        with self.assertRaises(objective.ObjectiveError):
            camera_at((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


class SeparationTests(unittest.TestCase):
    def test_separation_is_the_perpendicular_pixel_distance(self):
        value = objective.torso_separation_px((0.0, 0.0), (0.0, 100.0), (30.0, 40.0))
        self.assertAlmostEqual(value, 30.0, places=9)

    def test_separation_ignores_position_along_the_axis(self):
        near = objective.torso_separation_px((0.0, 0.0), (0.0, 100.0), (12.0, 5.0))
        far = objective.torso_separation_px((0.0, 0.0), (0.0, 100.0), (12.0, 95.0))
        self.assertAlmostEqual(near, far, places=9)

    def test_degenerate_torso_axis_is_rejected(self):
        with self.assertRaises(objective.ObjectiveError):
            objective.torso_separation_px((5.0, 5.0), (5.0, 5.0), (1.0, 1.0))


class OrientedBoxTests(unittest.TestCase):
    def test_usd_matrix_recovers_half_extents_and_axes(self):
        angle = math.radians(30.0)
        rotation = np.array([
            [math.cos(angle), math.sin(angle), 0.0],
            [-math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        half = np.array([0.086, 0.103, 0.025])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation * half[:, None]
        matrix[3, :3] = [2.9, -0.6, 1.6]
        box = objective.OrientedBox.from_usd_matrix(matrix)
        np.testing.assert_allclose(box.half_extents, half, atol=1e-12)
        np.testing.assert_allclose(box.axes, rotation, atol=1e-12)
        np.testing.assert_allclose(box.centre, [2.9, -0.6, 1.6], atol=1e-12)

    def test_degenerate_usd_matrix_is_rejected(self):
        matrix = np.eye(4)
        matrix[:3, :3] = 0.0
        with self.assertRaises(objective.ObjectiveError):
            objective.OrientedBox.from_usd_matrix(matrix)

    def test_segment_through_the_box_has_zero_clearance(self):
        self.assertAlmostEqual(unit_box().clearance_m((-5.0, 0.0, 0.0), (5.0, 0.0, 0.0)), 0.0, places=9)

    def test_segment_beside_a_face_reports_the_face_offset(self):
        value = unit_box().clearance_m((-5.0, 3.0, 0.0), (5.0, 3.0, 0.0))
        self.assertAlmostEqual(value, 2.0, places=6)

    def test_segment_past_a_corner_reports_the_corner_distance(self):
        value = unit_box().clearance_m((-5.0, 4.0, 4.0), (5.0, 4.0, 4.0))
        self.assertAlmostEqual(value, math.hypot(3.0, 3.0), places=6)

    def test_clearance_uses_the_segment_not_the_infinite_line(self):
        # The line through these points would graze the box, but the segment
        # stops well short of it.
        value = unit_box().clearance_m((-5.0, 0.0, 0.0), (-3.0, 0.0, 0.0))
        self.assertAlmostEqual(value, 2.0, places=6)

    def test_grazing_a_face_from_outside_stays_non_negative(self):
        value = unit_box().clearance_m((-5.0, 1.0, 0.0), (5.0, 1.0, 0.0))
        self.assertGreaterEqual(value, 0.0)
        self.assertAlmostEqual(value, 0.0, places=6)


class DepthProxyTests(unittest.TestCase):
    def test_empty_proxy_never_blocks(self):
        proxy = objective.DepthSurfaceProxy(np.zeros((0, 3)), np.zeros(0))
        self.assertEqual(len(proxy), 0)
        self.assertTrue(math.isinf(proxy.clearance_m((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))))

    def test_proxy_clearance_subtracts_the_point_radius(self):
        proxy = objective.DepthSurfaceProxy(np.array([[0.0, 2.0, 0.0]]), np.array([0.5]))
        value = proxy.clearance_m((-5.0, 0.0, 0.0), (5.0, 0.0, 0.0))
        self.assertAlmostEqual(value, 1.5, places=9)

    def test_proxy_clearance_is_clamped_at_zero(self):
        proxy = objective.DepthSurfaceProxy(np.array([[0.0, 0.1, 0.0]]), np.array([0.5]))
        self.assertAlmostEqual(proxy.clearance_m((-5.0, 0.0, 0.0), (5.0, 0.0, 0.0)), 0.0, places=9)

    def test_mismatched_points_and_radii_are_rejected(self):
        with self.assertRaises(objective.ObjectiveError):
            objective.DepthSurfaceProxy(np.zeros((3, 3)), np.zeros(2))

    def _depth_map_with_two_patches(self):
        depth = np.full((HEIGHT, WIDTH), np.inf, dtype=float)
        depth[295:306, 475:486] = 2.0            # patch containing the principal point
        depth[400:406, 600:606] = 2.0            # a disconnected second patch
        return depth

    def test_flood_fill_keeps_only_the_connected_seed_patch(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        proxy = objective.build_depth_surface_proxy(
            self._depth_map_with_two_patches(), camera, [(CX, CY)], torso_reference_depth_m=3.5)
        self.assertEqual(len(proxy), 11 * 11)

    def test_flood_fill_back_projects_to_the_observed_depth(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        proxy = objective.build_depth_surface_proxy(
            self._depth_map_with_two_patches(), camera, [(CX, CY)], torso_reference_depth_m=3.5)
        expected = np.asarray(camera.eye) + 2.0 * camera.basis[0]
        self.assertTrue(np.isclose(proxy.points, expected, atol=1e-9).all(axis=1).any())
        np.testing.assert_allclose(proxy.radii, 2.0 / FX / 2.0, atol=1e-12)

    def test_surface_within_the_frozen_margin_is_not_a_proxy(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        depth = np.full((HEIGHT, WIDTH), np.inf, dtype=float)
        # 3.5 - 3.2 = 0.3 m, inside OCCLUSION_DEPTH_MARGIN_M, so not an occluder.
        depth[295:306, 475:486] = 3.2
        proxy = objective.build_depth_surface_proxy(
            depth, camera, [(CX, CY)], torso_reference_depth_m=3.5)
        self.assertEqual(len(proxy), 0)

    def test_unseeded_proxy_keeps_every_near_pixel(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        proxy = objective.build_depth_surface_proxy(
            self._depth_map_with_two_patches(), camera, None, torso_reference_depth_m=3.5)
        self.assertEqual(len(proxy), 11 * 11 + 6 * 6)

    def test_unseeded_proxy_still_respects_the_frozen_margin(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        depth = np.full((HEIGHT, WIDTH), np.inf, dtype=float)
        depth[295:306, 475:486] = 3.2
        proxy = objective.build_depth_surface_proxy(
            depth, camera, None, torso_reference_depth_m=3.5)
        self.assertEqual(len(proxy), 0)

    def test_seed_outside_the_near_surface_yields_nothing(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        proxy = objective.build_depth_surface_proxy(
            self._depth_map_with_two_patches(), camera, [(10.0, 10.0)], torso_reference_depth_m=3.5)
        self.assertEqual(len(proxy), 0)

    def test_non_2d_depth_map_is_rejected(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        with self.assertRaises(objective.ObjectiveError):
            objective.build_depth_surface_proxy(
                np.zeros((4, 4, 3)), camera, [(CX, CY)], torso_reference_depth_m=3.5)


class EvaluateTests(unittest.TestCase):
    """A synthetic subject standing 3.0 m from the origin, arm to one side."""

    def setUp(self):
        self.joints = {
            "right_shoulder": (0.0, 0.15, 1.45),
            "pelvis": (0.0, 0.0, 1.00),
            "right_elbow": (0.0, 0.45, 1.30),
            "right_wrist": (0.0, 0.70, 1.20),
        }
        self.camera = camera_at((3.0, 0.0, 1.30), (0.0, 0.0, 1.30))

    def test_clear_viewpoint_is_feasible_and_scored_on_the_wrist(self):
        result = objective.evaluate_candidate(self.camera, self.joints)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["reason"], "feasible")
        self.assertAlmostEqual(result["score"], result["wrist_separation_px"], places=12)
        self.assertGreater(result["wrist_separation_px"], objective.MIN_WRIST_SEPARATION_PX)
        self.assertAlmostEqual(result["pelvis_range_m"], math.hypot(3.0, 0.30), places=9)

    def test_missing_registered_joint_is_a_contract_error(self):
        broken = dict(self.joints)
        broken.pop("pelvis")
        with self.assertRaises(objective.ObjectiveError):
            objective.evaluate_candidate(self.camera, broken)

    def test_unavailable_upstream_estimate_is_reported_not_raised(self):
        broken = dict(self.joints)
        broken["right_wrist"] = None
        result = objective.evaluate_candidate(self.camera, broken)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "upstream_estimate_unavailable")
        self.assertEqual(result["score"], -math.inf)

    def test_joint_outside_the_roi_window_is_rejected(self):
        far = dict(self.joints)
        far["right_wrist"] = (0.0, 3.0, 1.20)
        result = objective.evaluate_candidate(self.camera, far)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "outside_roi")

    def test_pelvis_range_is_recorded_but_never_gated(self):
        # A camera well outside the installation envelope is still scored per
        # frame; F2 is a registration-time check, not a per-frame one.
        camera = camera_at((6.0, 0.0, 1.30), (0.0, 0.0, 1.30))
        result = objective.evaluate_candidate(camera, self.joints)
        self.assertGreater(result["pelvis_range_m"], objective.DISTANCE_ENVELOPE_M[1])
        self.assertNotEqual(result["reason"], "outside_distance_envelope")

    def test_registered_arc_radius_is_inside_the_envelope(self):
        self.assertTrue(objective.radius_within_envelope(3.50))
        self.assertTrue(objective.radius_within_envelope(2.00))

    def test_radius_outside_the_envelope_is_rejected(self):
        self.assertFalse(objective.radius_within_envelope(3.75))
        self.assertFalse(objective.radius_within_envelope(1.50))

    def test_occluding_box_on_the_line_of_sight_is_rejected(self):
        box = objective.OrientedBox((1.5, 0.35, 1.25), np.eye(3), (0.03, 0.20, 0.20))
        result = objective.evaluate_candidate(self.camera, self.joints, occluder=box)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "occluded")
        self.assertEqual(result["clearance_m"]["right_wrist"], 0.0)

    def test_box_off_the_line_of_sight_leaves_the_candidate_feasible(self):
        box = objective.OrientedBox((1.5, -2.0, 1.25), np.eye(3), (0.03, 0.20, 0.20))
        result = objective.evaluate_candidate(self.camera, self.joints, occluder=box)
        self.assertTrue(result["feasible"])
        self.assertGreater(result["clearance_m"]["right_wrist"], 0.0)

    def test_viewpoint_along_the_arm_is_rejected_as_self_occluded(self):
        # Looking down the arm collapses the wrist onto the torso axis.
        camera = camera_at((0.0, 3.0, 1.30), (0.0, 0.0, 1.30))
        result = objective.evaluate_candidate(camera, self.joints)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "self_occluded")
        self.assertLessEqual(result["wrist_separation_px"], objective.MIN_WRIST_SEPARATION_PX)

    def test_record_only_geometry_survives_rejection(self):
        camera = camera_at((0.0, 3.0, 1.30), (0.0, 0.0, 1.30))
        result = objective.evaluate_candidate(camera, self.joints, reference_camera=self.camera)
        self.assertFalse(result["feasible"])
        self.assertIsNotNone(result["elbow_separation_px"])
        self.assertIsNotNone(result["baseline_m"])
        self.assertIsNotNone(result["triangulation_angle_deg"])

    def test_triangulation_angle_is_recorded_not_scored(self):
        other = camera_at((0.0, 3.0, 1.30), (0.0, 0.0, 1.30))
        result = objective.evaluate_candidate(self.camera, self.joints, reference_camera=other)
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["score"], result["wrist_separation_px"], places=12)

    def test_triangulation_angle_of_perpendicular_rays(self):
        angle = objective.triangulation_angle_deg((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(angle, 90.0, places=9)


class MeasuredHoldTests(unittest.TestCase):
    def setUp(self):
        self.joints = {
            "right_shoulder": (0.0, 0.15, 1.45),
            "pelvis": (0.0, 0.0, 1.00),
            "right_elbow": (0.0, 0.45, 1.30),
            "right_wrist": (0.0, 0.70, 1.20),
        }
        self.measured = {name: objective.MEASURED_SOURCE for name in objective.ARM_JOINTS}
        self.inferred = {name: "k4_fallback_inferred" for name in objective.ARM_JOINTS}

    def test_measured_frame_passes_through_unheld(self):
        state = objective.MeasuredGeometryHold().update(self.joints, self.measured)
        self.assertTrue(state["available"])
        self.assertFalse(state["held"])
        self.assertEqual(state["age_frames"], 0)
        self.assertEqual(state["joints"]["right_wrist"], self.joints["right_wrist"])

    def test_inference_before_any_measurement_is_unavailable(self):
        state = objective.MeasuredGeometryHold().update(self.joints, self.inferred)
        self.assertFalse(state["available"])
        self.assertIsNone(state["joints"])
        self.assertEqual(state["reason"], "no_measured_frame_yet")

    def test_inference_after_a_measurement_reuses_it_and_ages(self):
        hold = objective.MeasuredGeometryHold()
        hold.update(self.joints, self.measured)
        moved = dict(self.joints)
        moved["right_wrist"] = (0.0, 9.9, 9.9)
        first = hold.update(moved, self.inferred)
        second = hold.update(moved, self.inferred)
        self.assertTrue(first["held"] and second["held"])
        self.assertEqual((first["age_frames"], second["age_frames"]), (1, 2))
        # The held geometry is the measured one, not the inferred update.
        self.assertEqual(second["joints"]["right_wrist"], self.joints["right_wrist"])
        self.assertEqual(second["reason"], "held_inferred")

    def test_a_single_inferred_arm_joint_triggers_the_hold(self):
        hold = objective.MeasuredGeometryHold()
        hold.update(self.joints, self.measured)
        mixed = dict(self.measured)
        mixed["right_elbow"] = "k4_fallback_inferred"
        self.assertTrue(hold.update(self.joints, mixed)["held"])

    def test_missing_torso_joint_triggers_the_hold(self):
        hold = objective.MeasuredGeometryHold()
        hold.update(self.joints, self.measured)
        broken = dict(self.joints)
        broken["pelvis"] = None
        state = hold.update(broken, self.measured)
        self.assertTrue(state["held"])
        self.assertEqual(state["reason"], "held_incomplete")

    def test_a_later_measurement_clears_the_hold(self):
        hold = objective.MeasuredGeometryHold()
        hold.update(self.joints, self.measured)
        hold.update(self.joints, self.inferred)
        moved = dict(self.joints)
        moved["right_wrist"] = (0.0, 0.72, 1.18)
        state = hold.update(moved, self.measured)
        self.assertFalse(state["held"])
        self.assertEqual(state["age_frames"], 0)
        self.assertEqual(state["joints"]["right_wrist"], moved["right_wrist"])


class PolylineTests(unittest.TestCase):
    def test_chain_is_sampled_at_one_pixel_spacing(self):
        samples = objective.rasterise_polyline([(0.0, 0.0), (10.0, 0.0)])
        self.assertEqual(len(samples), 11)
        np.testing.assert_allclose(samples[0], [0.0, 0.0])
        np.testing.assert_allclose(samples[-1], [10.0, 0.0])

    def test_multi_segment_chain_covers_every_vertex(self):
        samples = objective.rasterise_polyline([(0.0, 0.0), (0.0, 4.0), (3.0, 4.0)])
        as_tuples = {tuple(np.round(point, 6)) for point in samples}
        for vertex in ((0.0, 0.0), (0.0, 4.0), (3.0, 4.0)):
            self.assertIn(vertex, as_tuples)

    def test_zero_length_segment_does_not_divide_by_zero(self):
        samples = objective.rasterise_polyline([(5.0, 5.0), (5.0, 5.0)])
        self.assertEqual(len(samples), 2)

    def test_a_single_pixel_chain_is_returned_as_is(self):
        self.assertEqual(len(objective.rasterise_polyline([(5.0, 5.0)])), 1)
        self.assertEqual(objective.rasterise_polyline([]), [])

    def test_chain_seeds_reach_a_patch_the_endpoints_miss(self):
        camera = camera_at((3.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        depth = np.full((HEIGHT, WIDTH), np.inf, dtype=float)
        depth[295:306, 475:486] = 2.0
        endpoints = [(400.0, 300.0), (560.0, 300.0)]      # both off the patch
        self.assertEqual(len(objective.build_depth_surface_proxy(
            depth, camera, endpoints, torso_reference_depth_m=3.5)), 0)
        chain = objective.rasterise_polyline(endpoints)   # crosses the patch
        self.assertEqual(len(objective.build_depth_surface_proxy(
            depth, camera, chain, torso_reference_depth_m=3.5)), 11 * 11)


class RankingTests(unittest.TestCase):
    def test_ranking_is_by_descending_score_and_drops_infeasible(self):
        evaluations = {
            "b": {"feasible": True, "score": 30.0},
            "a": {"feasible": True, "score": 45.0},
            "c": {"feasible": False, "score": -math.inf},
        }
        self.assertEqual(objective.rank_candidates(evaluations), ["a", "b"])
        self.assertEqual(objective.select_best(evaluations), "a")

    def test_ties_break_deterministically_on_candidate_id(self):
        evaluations = {
            "m20": {"feasible": True, "score": 40.0},
            "m10": {"feasible": True, "score": 40.0},
        }
        self.assertEqual(objective.rank_candidates(evaluations), ["m10", "m20"])

    def test_no_feasible_candidate_selects_nothing(self):
        evaluations = {"a": {"feasible": False, "score": -math.inf}}
        self.assertEqual(objective.rank_candidates(evaluations), [])
        self.assertIsNone(objective.select_best(evaluations))


if __name__ == "__main__":
    unittest.main()
