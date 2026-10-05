from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import common_camera_bank_v1 as bank  # noqa: E402
from fs_cts5_camera_bank_v1 import BONES, JOINTS  # noqa: E402


AIM = (1.9, -1.9, 1.1)
RADIUS = 3.5


def layout():
    """Nine candidates: three elevated in the G1 sector, six spread around."""
    return [
        {"name": "v1", "azimuth_deg": 0.0, "elevation_deg": 15.0, "role": "g1_sector_elevated"},
        {"name": "v2", "azimuth_deg": 60.0, "elevation_deg": 15.0, "role": "g1_sector_elevated"},
        {"name": "v3", "azimuth_deg": 120.0, "elevation_deg": 15.0, "role": "g1_sector_elevated"},
        {"name": "v4", "azimuth_deg": 0.0, "elevation_deg": -5.0, "role": "low_control"},
        {"name": "v5", "azimuth_deg": 60.0, "elevation_deg": -5.0, "role": "low_control"},
        {"name": "v6", "azimuth_deg": 180.0, "elevation_deg": 5.0, "role": "rear"},
        {"name": "v7", "azimuth_deg": 240.0, "elevation_deg": 5.0, "role": "rear"},
        {"name": "v8", "azimuth_deg": 300.0, "elevation_deg": 5.0, "role": "front_side"},
        {"name": "v9", "azimuth_deg": 300.0, "elevation_deg": 20.0, "role": "front_side_elevated"},
    ]


def views():
    return bank.build_candidate_bank(AIM, RADIUS, layout())


def availability(visible_views, characters=("F01", "F02", "M01", "M02"), scenarios=("G0", "G1"), frames=(0, 1)):
    """Every listed view measures every joint; the rest measure nothing."""
    names = [entry["name"] for entry in layout()]
    return {
        character: {
            scenario: {
                frame: {
                    name: {joint: name in visible_views for joint in JOINTS}
                    for name in names
                }
                for frame in frames
            }
            for scenario in scenarios
        }
        for character in characters
    }


class BankTests(unittest.TestCase):
    def test_bank_holds_exactly_nine_named_candidates(self):
        result = views()
        self.assertEqual(len(result), bank.CANDIDATE_COUNT)
        self.assertEqual(len({view["name"] for view in result}), 9)

    def test_every_eye_sits_at_the_bank_radius(self):
        for view in views():
            distance = sum(
                (view["eye"][axis] - AIM[axis]) ** 2 for axis in range(3)
            ) ** 0.5
            self.assertAlmostEqual(distance, RADIUS, places=9)

    def test_each_candidate_records_why_it_is_in_the_bank(self):
        for view in views():
            self.assertTrue(view["role"])

    def test_a_bank_of_the_wrong_size_is_rejected(self):
        with self.assertRaises(bank.CameraBankError):
            bank.build_candidate_bank(AIM, RADIUS, layout()[:8])

    def test_duplicate_names_are_rejected(self):
        broken = layout()
        broken[1]["name"] = "v1"
        with self.assertRaises(bank.CameraBankError):
            bank.build_candidate_bank(AIM, RADIUS, broken)


class G1CapabilityTests(unittest.TestCase):
    """The pilot measured leg occlusion only from elevated az 0-120 views."""

    def test_elevated_sector_views_are_g1_capable(self):
        self.assertTrue(bank.g1_capable({"azimuth_deg": 60.0, "elevation_deg": 15.0}))

    def test_low_views_in_the_sector_are_not(self):
        self.assertFalse(bank.g1_capable({"azimuth_deg": 60.0, "elevation_deg": -5.0}))

    def test_elevated_views_outside_the_sector_are_not(self):
        self.assertFalse(bank.g1_capable({"azimuth_deg": 240.0, "elevation_deg": 20.0}))

    def test_a_bank_without_enough_such_views_cannot_produce_g1(self):
        flat = [dict(entry, elevation_deg=-5.0) for entry in layout()]
        result = bank.check_bank_can_produce_g1(
            bank.build_candidate_bank(AIM, RADIUS, flat)
        )
        self.assertFalse(result["pass"])
        self.assertIn("occlusion pilot", result["basis"])

    def test_the_reference_bank_can_produce_g1(self):
        result = bank.check_bank_can_produce_g1(views())
        self.assertTrue(result["pass"])
        self.assertGreaterEqual(len(result["g1_capable_views"]), 2)


class GtFreeTests(unittest.TestCase):
    """6.6.5 forbids GT error and per-character optima in this selection."""

    def test_allowed_evidence_passes(self):
        bank.assert_gt_free(bank.ALLOWED_EVIDENCE)

    def test_gt_error_is_refused(self):
        with self.assertRaises(bank.CameraBankError):
            bank.assert_gt_free(("endpoint_availability", "gt_error_p95"))

    def test_ground_truth_by_any_name_is_refused(self):
        with self.assertRaises(bank.CameraBankError):
            bank.assert_gt_free(("ground_truth_mae",))

    def test_per_character_optimum_is_refused(self):
        with self.assertRaises(bank.CameraBankError):
            bank.assert_gt_free(("per_character_optimum",))

    def test_unknown_evidence_is_refused_rather_than_ignored(self):
        with self.assertRaises(bank.CameraBankError):
            bank.assert_gt_free(("vibes",))

    def test_the_selection_records_that_no_gt_was_read(self):
        result = bank.select_common_five(views(), availability(("v1", "v2", "v3", "v6", "v7")))
        self.assertFalse(result["gt_error_used"])
        self.assertIn("no GT error was read", result["note"])


class CellCountTests(unittest.TestCase):
    def test_all_views_measuring_gives_the_full_count(self):
        names = [entry["name"] for entry in layout()]
        counts = bank.cell_measured_counts(availability(names), names[:5])
        self.assertEqual(counts["min_measured_views"], 5)
        self.assertTrue(counts["meets_minimum"])

    def test_a_cell_below_three_is_located(self):
        counts = bank.cell_measured_counts(
            availability(("v1", "v2")), ("v1", "v2", "v3", "v4", "v5")
        )
        self.assertEqual(counts["min_measured_views"], 2)
        self.assertFalse(counts["meets_minimum"])
        self.assertEqual(len(counts["worst_cell"]), 4)

    def test_every_bone_is_a_target_cell(self):
        names = [entry["name"] for entry in layout()]
        counts = bank.cell_measured_counts(availability(names), names[:5])
        # 4 characters x 2 scenarios x 2 frames x 8 bones
        self.assertEqual(counts["cell_count"], 4 * 2 * 2 * len(BONES))

    def test_missing_availability_raises_rather_than_assuming(self):
        record = availability(("v1",))
        del record["F01"]["G0"][0]["v3"]
        with self.assertRaises(bank.CameraBankError):
            bank.cell_measured_counts(record, ("v1", "v2", "v3", "v4", "v5"))

    def test_an_empty_subset_raises(self):
        with self.assertRaises(bank.CameraBankError):
            bank.cell_measured_counts(availability(("v1",)), ())


class SelectionTests(unittest.TestCase):
    def test_one_subset_serves_every_character(self):
        result = bank.select_common_five(
            views(), availability(("v1", "v2", "v3", "v6", "v7"))
        )
        self.assertTrue(result["pass"])
        self.assertEqual(len(result["selected"]["subset"]), 5)
        self.assertEqual(result["subset_count"], 126)

    def test_selection_prefers_the_larger_margin(self):
        names = [entry["name"] for entry in layout()]
        result = bank.select_common_five(views(), availability(names))
        self.assertEqual(result["selected"]["min_measured_views"], 5)

    def test_an_impossible_bank_fails_instead_of_relaxing_the_minimum(self):
        result = bank.select_common_five(views(), availability(("v1", "v2")))
        self.assertFalse(result["pass"])
        self.assertIsNone(result["selected"])
        self.assertIn("revising the candidate bank", result["failure"])

    def test_selection_is_deterministic(self):
        record = availability(("v1", "v2", "v3", "v6", "v7"))
        first = bank.select_common_five(views(), record)
        second = bank.select_common_five(views(), record)
        self.assertEqual(first["selected"]["subset"], second["selected"]["subset"])

    def test_a_character_that_disagrees_drags_the_common_subset_down(self):
        # M02 sees only two views; the shared subset must reflect that.
        record = availability(("v1", "v2", "v3", "v6", "v7"))
        for scenario in record["M02"].values():
            for frame in scenario.values():
                for name in ("v3", "v6", "v7"):
                    frame[name] = {joint: False for joint in JOINTS}
        result = bank.select_common_five(views(), record)
        self.assertFalse(result["pass"])

    def test_g1_capability_breaks_ties_before_geometry(self):
        names = [entry["name"] for entry in layout()]
        result = bank.select_common_five(views(), availability(names))
        self.assertGreaterEqual(result["selected"]["g1_capable_count"], 1)


class ScenarioTargetTests(unittest.TestCase):
    """6.6.5's 'target cell' reading, settled 2026-08-19.

    A scenario's target cells are the bones that scenario is about.  The
    strictest alternative was measured infeasible on the real bank: its best
    five-view subset left a cell with one measured view, because a forearm at
    reach onset is self-occluded from all but one or two viewpoints.
    """

    def test_the_map_covers_every_scenario(self):
        self.assertEqual(set(bank.SCENARIO_TARGET_BONES), {"G0", "G1", "G2"})

    def test_g1_targets_the_legs_and_g2_the_right_arm(self):
        self.assertEqual(set(bank.SCENARIO_TARGET_BONES["G1"]), set(bank.LEG_BONES))
        self.assertEqual(
            set(bank.SCENARIO_TARGET_BONES["G2"]), {"right_upper_arm", "right_forearm"}
        )

    def test_g0_carries_all_eight_bones(self):
        self.assertEqual(len(bank.SCENARIO_TARGET_BONES["G0"]), len(BONES))

    def test_counting_honours_the_scenario_targets(self):
        # An arm-only failure must not sink G1, which is a leg scenario.
        record = availability(("v1", "v2", "v3", "v6", "v7"))
        for scenario in record["F01"].values():
            for frame in scenario.values():
                for name in frame:
                    for joint in ("left_elbow", "left_wrist"):
                        frame[name][joint] = False
        strict = bank.cell_measured_counts(record, ("v1", "v2", "v3", "v6", "v7"))
        scoped = bank.cell_measured_counts(
            record, ("v1", "v2", "v3", "v6", "v7"),
            {"G0": bank.LEG_BONES, "G1": bank.LEG_BONES},
        )
        self.assertLess(strict["min_measured_views"], scoped["min_measured_views"])

    def test_the_selection_records_which_reading_it_used(self):
        result = bank.select_common_five(
            views(), availability(("v1", "v2", "v3", "v6", "v7"))
        )
        self.assertEqual(set(result["scenario_targets"]), {"G0", "G1", "G2"})

    def test_the_strict_reading_is_still_available(self):
        record = availability(("v1", "v2", "v3", "v6", "v7"))
        result = bank.select_common_five(views(), record, scenario_targets=None)
        self.assertIsNone(result["scenario_targets"])


class BoundaryTests(unittest.TestCase):
    def test_module_declares_its_status(self):
        self.assertIs(bank.FORMAL_CAPTURE_AUTHORIZED, False)
        self.assertEqual(bank.MIN_MEASURED_VIEWS, 3)
        self.assertEqual(bank.SELECTED_COUNT, 5)

    def test_module_has_no_isaac_dependency(self):
        source = (TOOLS / "common_camera_bank_v1.py").read_text(encoding="utf-8")
        for forbidden in ("import omni", "from pxr", "import carb"):
            self.assertNotIn(forbidden, source)


class SurveyBuildTests(unittest.TestCase):
    """6.6.19 revises the bank, which means surveying more than nine candidates."""

    def test_build_views_accepts_a_survey_larger_than_the_bank(self):
        survey = layout() + [
            {"name": "v10", "azimuth_deg": 30.0, "elevation_deg": 0.0, "role": "infill"},
            {"name": "v11", "azimuth_deg": 210.0, "elevation_deg": 10.0, "role": "infill"},
        ]
        views = bank.build_views(AIM, RADIUS, survey)
        self.assertEqual(len(views), 11)
        self.assertEqual([view["name"] for view in views][-2:], ["v10", "v11"])

    def test_the_nine_view_contract_still_binds_the_frozen_bank(self):
        survey = layout() + [
            {"name": "v10", "azimuth_deg": 30.0, "elevation_deg": 0.0, "role": "infill"},
        ]
        with self.assertRaises(bank.CameraBankError):
            bank.build_candidate_bank(AIM, RADIUS, survey)

    def test_build_views_still_rejects_duplicates_and_bad_radius(self):
        duplicated = layout() + [dict(layout()[0])]
        with self.assertRaises(bank.CameraBankError):
            bank.build_views(AIM, RADIUS, duplicated)
        with self.assertRaises(bank.CameraBankError):
            bank.build_views(AIM, 0.0, layout())

    def test_survey_geometry_matches_the_nine_view_bank_for_shared_names(self):
        """A surveyed candidate must land where the frozen bank would put it."""
        frozen = {view["name"]: view for view in bank.build_candidate_bank(AIM, RADIUS, layout())}
        surveyed = {
            view["name"]: view
            for view in bank.build_views(
                AIM,
                RADIUS,
                layout()
                + [{"name": "v10", "azimuth_deg": 30.0, "elevation_deg": 0.0, "role": "infill"}],
            )
        }
        for name, view in frozen.items():
            self.assertEqual(view["eye"], surveyed[name]["eye"])
            self.assertEqual(view["aim"], surveyed[name]["aim"])


if __name__ == "__main__":
    unittest.main()
