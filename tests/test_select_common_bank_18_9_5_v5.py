from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import common_bank_rule_lock_v5 as lock  # noqa: E402
import select_common_bank_18_9_5_v5 as selector  # noqa: E402


CLEAR = "1" * lock.FRAME_COUNT
NEVER = "0" * lock.FRAME_COUNT
EVENT_SPANS = {"left": (85, 125), "right": (170, 205)}
LEFT_CARRIERS = ("az180_el10", "az210_el10")
RIGHT_CARRIERS = ("az240_el10", "az270_el10")
THIN_VIEW = "az150_el10"
THIN_SPAN = (100, 104)


def _occlude(span):
    measurable = list(CLEAR)
    table = list(NEVER)
    for frame in range(span[0], span[1] + 1):
        measurable[frame] = "0"
        table[frame] = "1"
    return "".join(measurable), "".join(table)


def build_matrix(carriers=(), blind=(), thin=()):
    """Everything measurable except the events planted, plus optional gaps.

    ``thin`` is a sequence of ``(view, bone, (start, end))``: the view simply
    cannot measure that bone over that span in G1, with no table attribution.  It
    exists to push a handful of cells below the deployment floor without making
    them a scenario event.
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
    for view, bone, span in thin:
        flags = list(measured[view][characters[0]]["G1"][bone])
        for frame in range(span[0], span[1] + 1):
            flags[frame] = "0"
        for character in characters:
            measured[view][character]["G1"][bone] = "".join(flags)
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


def full_carriers():
    return [
        (LEFT_CARRIERS[0], "left"),
        (LEFT_CARRIERS[1], "left"),
        (RIGHT_CARRIERS[0], "right"),
        (RIGHT_CARRIERS[1], "right"),
    ]


class MatrixValidationTests(unittest.TestCase):
    def test_a_matrix_on_the_v1_layout_is_rejected(self):
        import common_bank_rule_lock_v1 as v1

        matrix = build_matrix()
        matrix["views"] = list(v1.SURVEY_VIEWS)
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)

    def test_a_short_capture_is_rejected(self):
        matrix = build_matrix()
        matrix["frame_count"] = 120
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)

    def test_a_non_binary_value_is_rejected(self):
        matrix = build_matrix()
        matrix["E"]["az000_el00"]["F01"]["G0"]["left_thigh"] = "2" * lock.FRAME_COUNT
        with self.assertRaises(selector.SelectionError):
            selector.load_matrix(matrix)


class PerCellFloorTests(unittest.TestCase):
    """v5 withdraws v3's coverage gate and restores v2's per-cell floor."""

    def setUp(self):
        self.angle_table = lock.pairwise_angle_table()

    def score(self, matrix, subset):
        data = selector.load_matrix(matrix)
        scoring = selector.score_subsets(data, self.angle_table)
        return scoring["subsets"][tuple(sorted(subset))]

    def test_a_handful_of_thin_cells_is_vetoed_again(self):
        """Under v3 this passed on coverage; v5 puts the veto back."""
        matrix = build_matrix(
            carriers=full_carriers(),
            thin=[(THIN_VIEW, "left_thigh", THIN_SPAN)],
        )
        subset = list(LEFT_CARRIERS) + [THIN_VIEW, "az000_el00", "az030_el00"]
        entry = self.score(matrix, subset)
        self.assertEqual(entry["min_measured_views"], 2)
        self.assertEqual(entry["cells_at_minimum"], 20)
        self.assertFalse(entry["gate_measured_views"])
        self.assertFalse(entry["passes"])

    def test_the_floor_is_applied_per_scenario(self):
        matrix = build_matrix(carriers=full_carriers())
        entry = self.score(
            matrix, list(LEFT_CARRIERS) + list(RIGHT_CARRIERS) + ["az000_el00"]
        )
        for scenario, floor in lock.MEASURED_VIEWS_FLOOR.items():
            self.assertGreaterEqual(
                entry["per_scenario_min_measured_views"][scenario], floor
            )
        self.assertTrue(entry["gate_measured_views"])

    def test_g1_may_sit_on_the_floor_while_g0_may_not(self):
        self.assertEqual(lock.MEASURED_VIEWS_FLOOR["G1"], lock.DEPLOYMENT_FLOOR)
        self.assertGreater(
            lock.MEASURED_VIEWS_FLOOR["G0"], lock.MEASURED_VIEWS_FLOOR["G1"]
        )

    def test_the_selector_no_longer_speaks_of_coverage(self):
        source = (TOOLS / "select_common_bank_18_9_5_v5.py").read_text(encoding="utf-8")
        self.assertNotIn("COVERAGE", source)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = build_matrix(
            carriers=full_carriers(),
            thin=[(THIN_VIEW, "left_thigh", THIN_SPAN)],
        )
        cls.result = selector.select(cls.matrix)

    def test_the_pair_space_is_audited_in_full(self):
        self.assertEqual(self.result["pairs_audited"], 6126120)
        self.assertEqual(self.result["pairs_expected"], 6126120)

    def test_a_pair_is_chosen_under_the_per_cell_floor(self):
        self.assertTrue(self.result["pass"])
        self.assertIsNone(self.result["failure"])
        self.assertEqual(self.result["availability_gate"], "per_cell_floor")

    def test_nothing_is_authorised_and_no_ground_truth_is_read(self):
        self.assertFalse(self.result["formal_capture_authorized"])
        self.assertFalse(self.result["gt_error_used"])

    def test_the_chosen_subset_carries_an_event_view_per_side(self):
        selected = set(self.result["chosen"]["F"])
        self.assertTrue(selected & set(LEFT_CARRIERS))
        self.assertTrue(selected & set(RIGHT_CARRIERS))

    def test_the_chosen_pair_reports_its_per_scenario_minimum(self):
        minima = self.result["chosen"]["per_scenario_min_measured_views"]
        for scenario, floor in lock.MEASURED_VIEWS_FLOOR.items():
            self.assertGreaterEqual(minima[scenario], floor)

    def test_the_ordering_still_prefers_the_best_worst_cell(self):
        """Key 1 is unchanged, so the per-cell minimum is still a preference."""
        self.assertEqual(self.result["ordering_keys"][0], "max_min_measured_views")
        self.assertGreaterEqual(
            self.result["chosen"]["score"][0], lock.DEPLOYMENT_FLOOR
        )

    def test_every_score_component_is_an_integer(self):
        for value in self.result["chosen"]["score"]:
            self.assertIsInstance(value, int)

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

    def test_m0_is_inside_the_selection(self):
        self.assertIn(self.result["m0_anchor"]["view"], self.result["chosen"]["F"])

    def test_the_result_is_reproducible(self):
        again = selector.select(copy.deepcopy(self.matrix))
        self.assertEqual(again["chosen"]["F"], self.result["chosen"]["F"])
        self.assertEqual(again["chosen"]["R"], self.result["chosen"]["R"])
        self.assertEqual(again["chosen"]["score"], self.result["chosen"]["score"])


class NoEventSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = selector.select(build_matrix())

    def test_availability_alone_does_not_make_a_scenario(self):
        self.assertFalse(self.result["pass"])
        self.assertEqual(self.result["qualifying_pairs"], 0)
        self.assertEqual(
            self.result["hard_gate_failures"]["f_recoverable_events"], 6126120
        )
        self.assertEqual(self.result["hard_gate_failures"]["f_measured_views"], 0)

    def test_the_whole_pair_space_is_still_audited(self):
        self.assertEqual(self.result["pairs_audited"], 6126120)


class ModuleBoundaryTests(unittest.TestCase):
    def test_selector_reads_no_ground_truth(self):
        source = (TOOLS / "select_common_bank_18_9_5_v5.py").read_text(encoding="utf-8")
        cleaned = source.lower().replace("gt_error_used", "")
        for forbidden in ("gt_error", "ground_truth", "mpjpe", "per_character_optimum"):
            self.assertNotIn(forbidden, cleaned)

    def test_selector_has_no_isaac_dependency(self):
        source = (TOOLS / "select_common_bank_18_9_5_v5.py").read_text(encoding="utf-8")
        for forbidden in ("import omni", "from pxr", "import carb"):
            self.assertNotIn(forbidden, source)

    def test_the_earlier_selectors_are_left_alone(self):
        """Each earlier selector is hashed into its own frozen lock."""
        v1_source = (TOOLS / "select_common_bank_18_9_5_v1.py").read_text(encoding="utf-8")
        self.assertIn("REQUIRED_MEASURED_VIEWS", v1_source)
        self.assertNotIn("COVERAGE_GATES", v1_source)
        v3_source = (TOOLS / "select_common_bank_18_9_5_v3.py").read_text(encoding="utf-8")
        self.assertIn("COVERAGE_GATES", v3_source)
        self.assertNotIn("provably_mandatory", v3_source)
        v4_source = (TOOLS / "select_common_bank_18_9_5_v4.py").read_text(encoding="utf-8")
        self.assertIn("COVERAGE_GATES", v4_source)
        self.assertIn("provably_mandatory", v4_source)


class MandatoryExemptionTests(unittest.TestCase):
    """The change v4 exists for."""

    @classmethod
    def setUpClass(cls):
        cls.matrix = build_matrix(carriers=full_carriers())
        cls.data = selector.load_matrix(cls.matrix)
        cls.scoring = selector.score_subsets(cls.data, lock.pairwise_angle_table())

    def test_a_sole_carrier_for_a_side_becomes_provably_mandatory(self):
        """One left carrier only, so every qualifying subset must hold it."""
        matrix = build_matrix(
            carriers=[
                (LEFT_CARRIERS[0], "left"),
                (RIGHT_CARRIERS[0], "right"),
                (RIGHT_CARRIERS[1], "right"),
            ]
        )
        data = selector.load_matrix(matrix)
        scoring = selector.score_subsets(data, lock.pairwise_angle_table())
        passing = [e["subset"] for e in scoring["subsets"].values() if e["passes"]]
        self.assertTrue(passing)
        self.assertIn(LEFT_CARRIERS[0], lock.provably_mandatory(passing))

    def test_with_two_carriers_a_side_nothing_is_mandatory(self):
        passing = [e["subset"] for e in self.scoring["subsets"].values() if e["passes"]]
        self.assertTrue(passing)
        self.assertEqual(lock.provably_mandatory(passing), [])

    def test_the_exemption_turns_a_dead_search_into_a_live_one(self):
        """With a sole left carrier, v3's reserve gate can never be met."""
        matrix = build_matrix(
            carriers=[
                (LEFT_CARRIERS[0], "left"),
                (RIGHT_CARRIERS[0], "right"),
                (RIGHT_CARRIERS[1], "right"),
            ]
        )
        import select_common_bank_18_9_5_v3 as v3_selector

        without = v3_selector.select(matrix)
        self.assertEqual(without["qualifying_pairs"], 0)
        self.assertGreater(without["hard_gate_failures"]["r_swap_coverage"], 0)

        with_exemption = selector.select(matrix)
        self.assertGreater(with_exemption["qualifying_pairs"], 0)
        self.assertEqual(
            with_exemption["provably_mandatory_views"], [LEFT_CARRIERS[0]]
        )
        self.assertIn(LEFT_CARRIERS[0], with_exemption["chosen"]["F"])

    def test_every_non_exempt_selected_view_is_still_swappable(self):
        matrix = build_matrix(
            carriers=[
                (LEFT_CARRIERS[0], "left"),
                (RIGHT_CARRIERS[0], "right"),
                (RIGHT_CARRIERS[1], "right"),
            ]
        )
        result = selector.select(matrix)
        chosen = result["chosen"]
        exempt = set(result["provably_mandatory_views"])
        data = selector.load_matrix(matrix)
        scoring = selector.score_subsets(data, lock.pairwise_angle_table())
        for dropped in chosen["F"]:
            if dropped in exempt:
                continue
            remainder = [view for view in chosen["F"] if view != dropped]
            self.assertTrue(
                any(
                    scoring["subsets"][tuple(sorted(remainder + [candidate]))]["passes"]
                    for candidate in chosen["R"]
                ),
                "no reserve can replace non-exempt {}".format(dropped),
            )

    def test_the_exempt_views_are_named_in_the_result(self):
        result = selector.select(self.matrix)
        self.assertIn("provably_mandatory_views", result)
        self.assertEqual(
            result["provably_mandatory_views"], result["reserve_gate_exempts"]
        )


if __name__ == "__main__":
    unittest.main()
