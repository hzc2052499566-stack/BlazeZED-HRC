from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import common_bank_rule_lock_v1 as lock  # noqa: E402
import select_common_bank_18_9_5_v1 as selector  # noqa: E402


CLEAR = "1" * lock.FRAME_COUNT
NEVER = "0" * lock.FRAME_COUNT
# Both stretches sit inside their march window with room for the five clear
# frames the event shape demands on each side.
EVENT_SPANS = {"left": (85, 125), "right": (170, 205)}
LEFT_CARRIERS = ("az180_el10", "az210_el10")
RIGHT_CARRIERS = ("az240_el10", "az270_el10")


def _occlude(span):
    measurable = list(CLEAR)
    table = list(NEVER)
    for frame in range(span[0], span[1] + 1):
        measurable[frame] = "0"
        table[frame] = "1"
    return "".join(measurable), "".join(table)


def build_matrix(carriers=(), blind=()):
    """A matrix where everything is measurable except the events we plant.

    ``carriers`` is a sequence of ``(view, side)``: that view loses the side's
    thigh to the workbench for one contiguous stretch, and regains it, for every
    character.  ``blind`` is a sequence of ``(view, bone)`` the view never
    measures at all.
    """
    views = list(lock.SURVEY_VIEWS)
    characters = list(lock.CHARACTERS)
    measured = {
        view: {
            character: {
                scenario: {bone: CLEAR for bone in lock.TARGET_BONES[scenario]}
                for scenario in lock.SELECTION_SCENARIOS
            }
            for character in characters
        }
        for view in views
    }
    occluded = {
        view: {
            character: {bone: NEVER for bone in lock.TARGET_BONES["G1"]}
            for character in characters
        }
        for view in views
    }
    for view, side in carriers:
        bone = lock.SIDE_BONES[side][0]
        g1, table = _occlude(EVENT_SPANS[side])
        for character in characters:
            measured[view][character]["G1"][bone] = g1
            occluded[view][character][bone] = table
    for view, bone in blind:
        for character in characters:
            measured[view][character]["G0"][bone] = NEVER
            if bone in lock.TARGET_BONES["G1"]:
                measured[view][character]["G1"][bone] = NEVER
    return {
        "views": views,
        "characters": characters,
        "frame_count": lock.FRAME_COUNT,
        "E": measured,
        "T": occluded,
    }


def carriers_for(spare=True):
    plan = [(LEFT_CARRIERS[0], "left"), (RIGHT_CARRIERS[0], "right")]
    if spare:
        plan += [(LEFT_CARRIERS[1], "left"), (RIGHT_CARRIERS[1], "right")]
    return plan


class MatrixValidationTests(unittest.TestCase):
    def test_a_matrix_missing_a_view_is_rejected(self):
        matrix = build_matrix()
        matrix["views"] = matrix["views"][:-1]
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)

    def test_a_matrix_missing_a_character_is_rejected(self):
        matrix = build_matrix()
        matrix["characters"] = ["F01", "F02", "M01"]
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)

    def test_a_short_capture_is_rejected(self):
        matrix = build_matrix()
        matrix["frame_count"] = 121
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)

    def test_a_truncated_bit_string_is_rejected(self):
        matrix = build_matrix()
        matrix["E"]["az000_el00"]["F01"]["G0"]["left_thigh"] = "101"
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)

    def test_a_non_binary_value_is_rejected(self):
        matrix = build_matrix()
        matrix["E"]["az000_el00"]["F01"]["G0"]["left_thigh"] = "2" * lock.FRAME_COUNT
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)

    def test_a_missing_table_array_is_rejected(self):
        matrix = build_matrix()
        del matrix["T"]["az000_el00"]["F01"]["left_thigh"]
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)

    def test_binary_lists_are_accepted_as_well_as_bit_strings(self):
        matrix = build_matrix()
        matrix["E"]["az000_el00"]["F01"]["G0"]["left_thigh"] = [1] * lock.FRAME_COUNT
        data = selector.load_matrix(matrix)
        self.assertEqual(sum(data["E"][("az000_el00", "F01", "G0", "left_thigh")]), 241)


class DerivedQuantityTests(unittest.TestCase):
    def test_blind_pairs_are_found_and_partial_loss_is_not_blindness(self):
        matrix = build_matrix(carriers=carriers_for(), blind=[("az090_el00", "left_forearm")])
        data = selector.load_matrix(matrix)
        self.assertEqual(
            selector.blind_pairs(data), [{"view": "az090_el00", "bone": "left_forearm"}]
        )

    def test_event_views_finds_exactly_the_planted_carriers(self):
        data = selector.load_matrix(build_matrix(carriers=carriers_for()))
        carriers = selector.event_views(data)
        for character in lock.CHARACTERS:
            self.assertEqual(sorted(carriers[character]["left"]), sorted(LEFT_CARRIERS))
            self.assertEqual(
                sorted(carriers[character]["right"]), sorted(RIGHT_CARRIERS)
            )

    def test_clean_counts_are_integer_frame_counts(self):
        data = selector.load_matrix(build_matrix(blind=[("az090_el00", "left_forearm")]))
        counts = selector.clean_frame_counts(data)
        self.assertEqual(counts["az000_el00"]["min_character_bone_frames"], 241)
        self.assertEqual(counts["az090_el00"]["min_character_bone_frames"], 0)
        self.assertEqual(
            counts["az000_el00"]["total_g0_measured_frames"],
            len(lock.CHARACTERS) * len(lock.TARGET_BONES["G0"]) * 241,
        )


class NoEventSelectionTests(unittest.TestCase):
    """Nothing is ever occluded, so no G1 scenario exists to select for."""

    @classmethod
    def setUpClass(cls):
        cls.result = selector.select(build_matrix())

    def test_the_whole_pair_space_is_audited(self):
        self.assertEqual(self.result["pairs_audited"], 6126120)
        self.assertEqual(self.result["pairs_expected"], 6126120)

    def test_no_pair_qualifies_and_the_failure_is_named(self):
        self.assertFalse(self.result["pass"])
        self.assertIsNone(self.result["chosen"])
        self.assertEqual(self.result["failure"], "no_qualifying_pair")
        self.assertEqual(self.result["qualifying_pairs"], 0)

    def test_every_rejection_is_attributed_to_the_event_gate(self):
        failures = self.result["hard_gate_failures"]
        self.assertEqual(failures["f_recoverable_events"], 6126120)
        self.assertEqual(failures["f_measured_views"], 0)


class SingleCarrierSelectionTests(unittest.TestCase):
    """One carrier per side: the reserve can never replace a carrier."""

    @classmethod
    def setUpClass(cls):
        cls.result = selector.select(build_matrix(carriers=carriers_for(spare=False)))

    def test_no_pair_qualifies_because_the_reserve_cannot_cover_f(self):
        self.assertFalse(self.result["pass"])
        self.assertEqual(self.result["qualifying_pairs"], 0)
        failures = self.result["hard_gate_failures"]
        self.assertGreater(failures["r_swap_coverage"], 0)

    def test_the_pair_space_is_still_audited_in_full(self):
        self.assertEqual(self.result["pairs_audited"], 6126120)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = build_matrix(carriers=carriers_for())
        cls.result = selector.select(cls.matrix)

    def test_the_pair_space_is_audited_in_full(self):
        self.assertEqual(self.result["pairs_audited"], 6126120)
        self.assertEqual(self.result["pairs_expected"], 6126120)

    def test_a_pair_is_chosen_and_nothing_is_authorised_by_it(self):
        self.assertTrue(self.result["pass"])
        self.assertIsNone(self.result["failure"])
        self.assertFalse(self.result["formal_capture_authorized"])
        self.assertFalse(self.result["gt_error_used"])

    def test_the_chosen_subset_carries_one_event_view_per_side(self):
        selected = set(self.result["chosen"]["F"])
        self.assertEqual(len(selected & set(LEFT_CARRIERS)), 1)
        self.assertEqual(len(selected & set(RIGHT_CARRIERS)), 1)

    def test_two_carriers_of_a_side_would_break_the_four_view_floor(self):
        """Both carriers of one side occlude the same bone over the same frames,
        so a subset holding both drops that cell to three measured views."""
        data = selector.load_matrix(self.matrix)
        scoring = selector.score_subsets(data, lock.pairwise_angle_table())
        both = tuple(
            sorted(list(LEFT_CARRIERS) + ["az000_el00", "az030_el00", "az060_el00"])
        )
        entry = scoring["subsets"][both]
        self.assertEqual(entry["min_measured_views"], 3)
        self.assertFalse(entry["gate_measured_views"])
        self.assertFalse(entry["passes"])

    def test_the_reserve_can_replace_every_selected_view(self):
        chosen = self.result["chosen"]
        data = selector.load_matrix(self.matrix)
        scoring = selector.score_subsets(data, lock.pairwise_angle_table())
        for dropped in chosen["F"]:
            remainder = [view for view in chosen["F"] if view != dropped]
            self.assertTrue(
                any(
                    scoring["subsets"][tuple(sorted(remainder + [candidate]))]["passes"]
                    for candidate in chosen["R"]
                ),
                "no reserve can replace {}".format(dropped),
            )

    def test_the_bank_is_the_selection_plus_the_reserve(self):
        chosen = self.result["chosen"]
        self.assertEqual(len(chosen["F"]), 5)
        self.assertEqual(len(chosen["R"]), 4)
        self.assertEqual(sorted(chosen["B"]), sorted(chosen["F"] + chosen["R"]))
        self.assertEqual(len(set(chosen["B"])), 9)

    def test_every_score_component_is_an_integer(self):
        for value in self.result["chosen"]["score"]:
            self.assertIsInstance(value, int)

    def test_the_chosen_pair_outranks_the_runner_up(self):
        chosen = self.result["chosen"]
        runner_up = self.result["runner_up"]
        self.assertIsNotNone(runner_up)
        self.assertLessEqual(
            selector._rank_key(chosen), selector._rank_key(runner_up)
        )
        self.assertNotEqual(
            (chosen["F"], chosen["R"]), (runner_up["F"], runner_up["R"])
        )

    def test_the_result_is_reproducible(self):
        again = selector.select(copy.deepcopy(self.matrix))
        self.assertEqual(again["chosen"]["F"], self.result["chosen"]["F"])
        self.assertEqual(again["chosen"]["R"], self.result["chosen"]["R"])
        self.assertEqual(again["chosen"]["score"], self.result["chosen"]["score"])
        self.assertEqual(again["m0_anchor"]["view"], self.result["m0_anchor"]["view"])

    def test_m0_is_inside_the_selection(self):
        self.assertIn(self.result["m0_anchor"]["view"], self.result["chosen"]["F"])
        self.assertTrue(self.result["m0_anchor"]["fixed_across_characters"])

    def test_events_are_reported_per_character_and_side(self):
        events = self.result["chosen"]["events"]
        for character in lock.CHARACTERS:
            self.assertTrue(events[character]["left"])
            self.assertTrue(events[character]["right"])
            self.assertGreaterEqual(
                len(events[character]["distinct_views"]),
                lock.EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER,
            )


class M0AnchorTests(unittest.TestCase):
    def setUp(self):
        self.table = lock.pairwise_angle_table()

    def test_the_worst_character_bone_rate_decides_first(self):
        data = selector.load_matrix(
            build_matrix(blind=[("az000_el00", "left_forearm")])
        )
        clean = selector.clean_frame_counts(data)
        anchor = selector.m0_anchor(
            data, ["az000_el00", "az030_el00", "az180_el10"], self.table, clean
        )
        self.assertNotEqual(anchor["view"], "az000_el00")

    def test_the_historical_view_wins_only_on_the_angle_key(self):
        """With every rate tied, the tie-break is distance from az000_el00 --
        which the historical view itself wins at zero."""
        data = selector.load_matrix(build_matrix())
        clean = selector.clean_frame_counts(data)
        anchor = selector.m0_anchor(
            data, ["az180_el10", "az000_el00", "az240_el10"], self.table, clean
        )
        self.assertEqual(anchor["view"], "az000_el00")
        self.assertEqual(anchor["angle_to_historical_nanodeg"], 0)

    def test_m0_needs_a_selection(self):
        data = selector.load_matrix(build_matrix())
        with self.assertRaises(selector.SelectionError):
            selector.m0_anchor(data, [], self.table, selector.clean_frame_counts(data))


class ModuleBoundaryTests(unittest.TestCase):
    def test_selector_reads_no_ground_truth(self):
        source = (TOOLS / "select_common_bank_18_9_5_v1.py").read_text(encoding="utf-8")
        for forbidden in ("gt_error", "ground_truth", "mpjpe", "per_character_optimum"):
            self.assertNotIn(forbidden, source.lower().replace("gt_error_used", ""))

    def test_selector_has_no_isaac_dependency(self):
        source = (TOOLS / "select_common_bank_18_9_5_v1.py").read_text(encoding="utf-8")
        for forbidden in ("import omni", "from pxr", "import carb"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
