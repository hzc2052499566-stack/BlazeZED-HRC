from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import common_bank_rule_lock_v1 as lock  # noqa: E402


SCENE_HASHES = {name: "0" * 64 for name in lock.CHARACTERS}
SELECTOR_HASHES = {"selector": "1" * 64, "tests": "2" * 64}


class AngleTableTests(unittest.TestCase):
    def setUp(self):
        self.table = lock.pairwise_angle_table()

    def test_table_is_square_symmetric_and_zero_on_the_diagonal(self):
        self.assertEqual(sorted(self.table), sorted(lock.SURVEY_VIEWS))
        for first in self.table:
            self.assertEqual(self.table[first][first], 0)
            for second in self.table:
                self.assertEqual(self.table[first][second], self.table[second][first])

    def test_every_entry_is_an_integer_nanodegree(self):
        for row in self.table.values():
            for value in row.values():
                self.assertIsInstance(value, int)

    def test_known_angles_land_where_geometry_says(self):
        degree = lock.NANODEGREES_PER_DEGREE
        self.assertEqual(self.table["az000_el00"]["az180_el00"], 180 * degree)
        self.assertEqual(self.table["az000_el00"]["az030_el00"], 30 * degree)
        self.assertEqual(self.table["az000_el00"]["az000_el15"], 15 * degree)
        self.assertEqual(self.table["az300_el10"]["az300_el20"], 10 * degree)

    def test_the_table_is_reproducible(self):
        self.assertEqual(self.table, lock.pairwise_angle_table())

    def test_distinct_views_are_never_coincident(self):
        """A zero off-diagonal angle would mean two candidates are one camera."""
        for first in self.table:
            for second in self.table:
                if first != second:
                    self.assertGreater(self.table[first][second], 0)

    def test_min_pairwise_angle_needs_two_views(self):
        with self.assertRaises(lock.RuleLockError):
            lock.min_pairwise_angle(["az000_el00"], self.table)


class RecoverableEventTests(unittest.TestCase):
    def build(self, occluded_from, occluded_to, window=(75, 150)):
        measurable = [1] * lock.FRAME_COUNT
        table = [0] * lock.FRAME_COUNT
        for frame in range(occluded_from, occluded_to + 1):
            measurable[frame] = 0
            table[frame] = 1
        return measurable, table, window

    def test_a_clean_occluded_clean_event_is_found(self):
        measurable, table, window = self.build(85, 125)
        event = lock.recoverable_event(measurable, table, window)
        self.assertTrue(event["found"])
        self.assertEqual(event["occluded"], [85, 125])
        self.assertEqual(event["clear_before"], [80, 84])
        self.assertEqual(event["clear_after"], [126, 130])

    def test_an_occlusion_shorter_than_thirty_frames_is_not_an_event(self):
        measurable, table, window = self.build(85, 105)
        self.assertFalse(lock.recoverable_event(measurable, table, window)["found"])

    def test_a_constant_occlusion_has_no_recovery_to_measure(self):
        """The elevated views are blocked for the whole capture; that is not an
        event, and 6.6.20's shape is what says so."""
        measurable = [0] * lock.FRAME_COUNT
        table = [1] * lock.FRAME_COUNT
        self.assertFalse(lock.recoverable_event(measurable, table, (75, 150))["found"])

    def test_flanks_must_fall_inside_the_march_window(self):
        # Occluded from the very start of the window: no room for five clear
        # frames before it without leaving the window.
        measurable, table, window = self.build(75, 120)
        self.assertFalse(lock.recoverable_event(measurable, table, window)["found"])
        measurable, table, window = self.build(100, 148)
        self.assertFalse(lock.recoverable_event(measurable, table, window)["found"])

    def test_a_gap_in_the_clear_flank_breaks_the_event(self):
        measurable, table, window = self.build(85, 125)
        measurable[82] = 0
        self.assertFalse(lock.recoverable_event(measurable, table, window)["found"])

    def test_an_event_outside_the_window_does_not_count(self):
        measurable, table, _ = self.build(10, 60)
        self.assertFalse(lock.recoverable_event(measurable, table, (75, 150))["found"])

    def test_window_must_lie_inside_the_motion(self):
        measurable, table, _ = self.build(85, 125)
        with self.assertRaises(lock.RuleLockError):
            lock.recoverable_event(measurable, table, (75, lock.FRAME_COUNT))


class BatchContractTests(unittest.TestCase):
    def test_batches_partition_the_survey_exactly_once_plus_the_anchor(self):
        report = lock.check_batches()
        covered = []
        for views in report["batches"].values():
            covered.extend(view for view in views if view != lock.SURVEY_ANCHOR)
        self.assertEqual(
            sorted(covered),
            sorted(view for view in lock.SURVEY_VIEWS if view != lock.SURVEY_ANCHOR),
        )
        self.assertEqual(report["anchor"], lock.SURVEY_ANCHOR)

    def test_no_batch_exceeds_the_render_ceiling(self):
        """About nine products per step is the reliable ceiling; the batches sit
        under it so a dropped frame can never look like unavailability."""
        for count in lock.check_batches()["render_products_per_step"].values():
            self.assertLessEqual(count, 9)

    def test_every_batch_carries_the_anchor(self):
        broken = (("batch_1", ("az000_el15",)), ("batch_2", (lock.SURVEY_ANCHOR,)))
        with self.assertRaises(lock.RuleLockError):
            lock.check_batches(broken)

    def test_a_view_rendered_twice_is_rejected(self):
        doubled = tuple(
            (label, views + ("az180_el10",)) if label == "batch_3" else (label, views)
            for label, views in lock.BATCHES
        )
        with self.assertRaises(lock.RuleLockError):
            lock.check_batches(doubled)


class LayoutTests(unittest.TestCase):
    def test_the_survey_holds_eighteen_uniquely_named_candidates(self):
        self.assertEqual(len(lock.SURVEY_LAYOUT), lock.SURVEY_COUNT)
        self.assertEqual(len(set(lock.SURVEY_VIEWS)), lock.SURVEY_COUNT)

    def test_the_layout_hash_moves_when_a_camera_moves(self):
        nudged = [dict(entry) for entry in lock.SURVEY_LAYOUT]
        nudged[0]["elevation_deg"] = 16.0
        self.assertNotEqual(lock.layout_sha256(), lock.layout_sha256(nudged))

    def test_the_layout_hash_ignores_ordering_but_not_content(self):
        reordered = list(reversed([dict(entry) for entry in lock.SURVEY_LAYOUT]))
        self.assertNotEqual(lock.layout_sha256(), lock.layout_sha256(reordered))


class RuleLockDocumentTests(unittest.TestCase):
    def setUp(self):
        self.document = lock.rule_lock_document(SELECTOR_HASHES, SCENE_HASHES)

    def test_the_pair_space_is_the_number_6_6_20_fixes(self):
        self.assertEqual(self.document["ordering"]["pair_space"], 6126120)
        self.assertTrue(self.document["ordering"]["exhaustive_required"])

    def test_the_freeze_keeps_the_floor_and_the_margin_apart(self):
        gates = self.document["hard_gates"]
        self.assertEqual(gates["deployment_floor"], 3)
        self.assertEqual(gates["required_measured_views"], 4)

    def test_g2_is_recorded_as_diagnostic_and_never_as_a_selection_scenario(self):
        unit = self.document["selection_unit"]
        self.assertEqual(unit["selection_scenarios"], ["G0", "G1"])
        self.assertEqual(unit["diagnostic_scenarios"], ["G2"])
        self.assertNotIn("G2", unit["target_bones"])

    def test_the_document_carries_the_frozen_angle_table(self):
        self.assertEqual(
            self.document["pairwise_angle_nanodegrees"], lock.pairwise_angle_table()
        )
        self.assertTrue(
            self.document["angle_quantisation"][
                "recomputation_at_selection_time_forbidden"
            ]
        )

    def test_the_document_refuses_a_layout_that_is_not_the_survey(self):
        with self.assertRaises(lock.RuleLockError):
            lock.rule_lock_document(
                SELECTOR_HASHES, SCENE_HASHES, list(lock.SURVEY_LAYOUT)[:17]
            )

    def test_the_document_refuses_incomplete_scene_hashes(self):
        with self.assertRaises(lock.RuleLockError):
            lock.rule_lock_document(SELECTOR_HASHES, {"F01": "0" * 64})

    def test_nothing_is_authorised_by_writing_the_rule_down(self):
        self.assertFalse(self.document["formal_capture_authorized"])
        self.assertFalse(self.document["gt_error_used"])

    def test_the_statement_boundary_is_carried_with_the_rule(self):
        self.assertIn("cohort-designed", self.document["statement_boundary"])
        self.assertIn("not held-out", self.document["statement_boundary"])


class ModuleBoundaryTests(unittest.TestCase):
    def test_module_has_no_isaac_dependency(self):
        source = (TOOLS / "common_bank_rule_lock_v1.py").read_text(encoding="utf-8")
        for forbidden in ("import omni", "from pxr", "import carb"):
            self.assertNotIn(forbidden, source)

    def test_the_rule_names_its_specification(self):
        self.assertIn("6.6.20", lock.RULE_SOURCE)


if __name__ == "__main__":
    unittest.main()
