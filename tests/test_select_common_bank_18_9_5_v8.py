from __future__ import annotations

import inspect
import itertools
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import common_bank_rule_lock_v7 as lock  # noqa: E402
import select_common_bank_18_9_5_v8 as selector  # noqa: E402


SMALL_VIEWS = ("A", "B", "C", "D")


def small_angle_table():
    names = SMALL_VIEWS + (lock.HISTORICAL_M0_VIEW,)
    positions = {name: index for index, name in enumerate(names)}
    return {
        first: {
            second: (
                0
                if first == second
                else 10 + 7 * abs(positions[first] - positions[second])
            )
            for second in names
        }
        for first in names
    }


def fake_scoring(repeat_index=0, failed=()):
    """Two-of-four scoring fixture; constants are patched only around searches."""
    failed = {tuple(sorted(item)) for item in failed}
    measured = (5, 3, 4)[repeat_index]
    cells = (2, 7, 5)[repeat_index]
    clean_minimum = (80, 60, 70)[repeat_index]
    subsets = {}
    for combination in itertools.combinations(SMALL_VIEWS, 2):
        passes = combination not in failed
        subsets[combination] = {
            "subset": list(combination),
            "min_measured_views": measured,
            "cells_at_minimum": cells,
            "worst_cell": {"repeat": repeat_index, "F": list(combination)},
            "coverage": {"G0": 1.0, "G1": 1.0},
            "min_view_character_bone_frames": clean_minimum,
            # Deliberately wrong: v8 must use the single geometry table instead.
            "min_pairwise_angle_nanodeg": 999_000 + repeat_index,
            "gate_coverage": passes,
            "gate_recoverable_events": True,
            "gate_distinct_event_views": True,
            "passes": passes,
            "events": {},
        }
    clean = {
        view: {
            "min_character_bone_frames": clean_minimum,
            "total_g0_measured_frames": 1_000 - 10 * repeat_index,
            "g0_cell_frames": 2_000,
        }
        for view in SMALL_VIEWS
    }
    return {"subsets": subsets, "clean": clean, "event_views": {}}


class PairSpaceTests(unittest.TestCase):
    def test_real_pair_space_is_still_exactly_six_million(self):
        self.assertEqual(selector.pair_space_count(), 6_126_120)
        self.assertEqual(
            selector.pair_space_count(
                lock.SURVEY_COUNT, lock.SELECTED_COUNT, lock.RESERVE_COUNT
            ),
            6_126_120,
        )

    def test_small_pair_space_uses_the_same_combinatorial_identity(self):
        self.assertEqual(selector.pair_space_count(4, 2, 1), 12)

    def test_impossible_counts_are_rejected(self):
        with self.assertRaises(selector.SelectionError):
            selector.pair_space_count(4, 3, 2)


class ThreeRepeatInputTests(unittest.TestCase):
    def test_exactly_three_matrices_are_required_before_scoring(self):
        with self.assertRaises(selector.SelectionError):
            selector._prepare_repeats([{}, {}], small_angle_table())
        with self.assertRaises(selector.SelectionError):
            selector._prepare_repeats([{}, {}, {}, {}], small_angle_table())

    def test_a_mapping_cannot_masquerade_as_the_repeat_sequence(self):
        with self.assertRaises(selector.SelectionError):
            selector._prepare_repeats({}, small_angle_table())


class RobustCoreSearchTests(unittest.TestCase):
    def run_small(self, scorings, leaderboard_limit=20):
        old_selected = selector.SELECTED_COUNT
        old_reserve = selector.RESERVE_COUNT
        try:
            selector.SELECTED_COUNT = 2
            selector.RESERVE_COUNT = 1
            return selector._core_select(
                scorings,
                small_angle_table(),
                SMALL_VIEWS,
                leaderboard_limit=leaderboard_limit,
            )
        finally:
            selector.SELECTED_COUNT = old_selected
            selector.RESERVE_COUNT = old_reserve

    def test_every_pair_is_audited_in_the_patched_small_fixture(self):
        result = self.run_small([fake_scoring(0), fake_scoring(1), fake_scoring(2)])
        self.assertEqual(result["pairs_expected"], 12)
        self.assertEqual(result["pairs_audited"], 12)
        self.assertEqual(result["qualifying_pairs"], 12)

    def test_six_data_keys_are_reduced_by_repeat_worst(self):
        result = self.run_small([fake_scoring(0), fake_scoring(1), fake_scoring(2)])
        chosen = result["chosen"]
        self.assertEqual(chosen["score"][0:3], [3, -7, 60])
        self.assertEqual(chosen["score"][4:7], [3, 2, 3])
        self.assertEqual(len(chosen["repeat_data_scores"]), 3)

    def test_geometry_comes_from_one_fixed_table_not_repeat_entries(self):
        angle_table = small_angle_table()
        result = self.run_small([fake_scoring(0), fake_scoring(1), fake_scoring(2)])
        chosen = result["chosen"]
        self.assertEqual(
            chosen["score"][3],
            lock.min_pairwise_angle(chosen["F"], angle_table),
        )
        self.assertEqual(
            chosen["score"][7],
            lock.min_pairwise_angle(chosen["B"], angle_table),
        )
        self.assertNotEqual(chosen["score"][3], 999_000)

    def test_a_single_repeat_f_failure_eliminates_that_f(self):
        result = self.run_small(
            [fake_scoring(0), fake_scoring(1, failed=[("A", "B")]), fake_scoring(2)]
        )
        self.assertFalse(
            any(item["F"] == ["A", "B"] for item in result["leaderboard"])
        )
        self.assertGreaterEqual(
            result["hard_gate_failures"]["f_coverage_not_all_repeats"], 2
        )
        self.assertEqual(result["pairs_audited"], 12)

    def test_a_swap_must_be_legal_in_each_repeat(self):
        result = self.run_small(
            [fake_scoring(0), fake_scoring(1, failed=[("A", "C")]), fake_scoring(2)]
        )
        pairs = {(tuple(item["F"]), tuple(item["R"])) for item in result["leaderboard"]}
        self.assertNotIn((("A", "B"), ("C",)), pairs)
        self.assertIn((("A", "B"), ("D",)), pairs)

    def test_every_numeric_score_component_is_an_integer(self):
        result = self.run_small([fake_scoring(0), fake_scoring(1), fake_scoring(2)])
        self.assertTrue(all(isinstance(value, int) for value in result["chosen"]["score"]))


class WorstCleanM0Tests(unittest.TestCase):
    def test_m0_uses_the_worst_clean_keys_not_a_pooled_or_mean_value(self):
        scorings = [fake_scoring(index) for index in range(3)]
        a_values = ((100, 1_000), (100, 1_000), (10, 100))
        b_values = ((20, 200), (20, 200), (20, 200))
        for index, scoring in enumerate(scorings):
            scoring["clean"]["A"]["min_character_bone_frames"] = a_values[index][0]
            scoring["clean"]["A"]["total_g0_measured_frames"] = a_values[index][1]
            scoring["clean"]["B"]["min_character_bone_frames"] = b_values[index][0]
            scoring["clean"]["B"]["total_g0_measured_frames"] = b_values[index][1]
        result = selector._m0_from_worst_clean(
            ["A", "B"], small_angle_table(), scorings, SMALL_VIEWS
        )
        self.assertEqual(result["view"], "B")
        self.assertEqual(result["min_character_bone_frames"], 20)
        self.assertEqual(result["clean_reduction"], "repeat_wise_worst")


class LoroStabilityTests(unittest.TestCase):
    def setUp(self):
        self.full = {
            "F": ["A", "B"],
            "R": ["C", "D"],
            "B": ["A", "B", "C", "D"],
        }
        self.loro = {
            "F": ["A", "C"],
            "R": ["B", "D"],
            "B": ["A", "B", "C", "D"],
        }
        self.m0 = {"view": "A"}

    def record(self, scorings, loro=None, loro_m0=None):
        return selector._loro_stability_record(
            0,
            self.full,
            self.m0,
            self.loro if loro is None else loro,
            self.m0 if loro_m0 is None else loro_m0,
            scorings,
            SMALL_VIEWS,
        )

    def test_one_view_change_from_full_reserve_passes_when_legal_three_of_three(self):
        record = self.record([fake_scoring(0), fake_scoring(1), fake_scoring(2)])
        self.assertTrue(record["B_exact"])
        self.assertTrue(record["M0_exact"])
        self.assertEqual(record["F_removed"], ["B"])
        self.assertEqual(record["F_added"], ["C"])
        self.assertEqual(
            record["replacement_legal_v7_swap_by_complete_repeat"],
            [True, True, True],
        )
        self.assertTrue(record["pass"])

    def test_replacement_fails_if_one_complete_repeat_says_swap_is_illegal(self):
        record = self.record(
            [fake_scoring(0), fake_scoring(1, failed=[("A", "C")]), fake_scoring(2)]
        )
        self.assertEqual(
            record["replacement_legal_v7_swap_by_complete_repeat"],
            [True, False, True],
        )
        self.assertFalse(record["replacement_legal_v7_swap_3_of_3"])
        self.assertFalse(record["pass"])

    def test_bank_and_m0_are_exact_gates(self):
        bad_bank = dict(self.loro, B=["A", "B", "C", "X"])
        self.assertFalse(
            self.record(
                [fake_scoring(0), fake_scoring(1), fake_scoring(2)], loro=bad_bank
            )["pass"]
        )
        self.assertFalse(
            self.record(
                [fake_scoring(0), fake_scoring(1), fake_scoring(2)],
                loro_m0={"view": "B"},
            )["pass"]
        )

    def test_more_than_one_f_change_is_rejected_before_swap_lookup(self):
        loro = {
            "F": ["C", "D"],
            "R": ["A", "B"],
            "B": ["A", "B", "C", "D"],
        }
        record = self.record(
            [fake_scoring(0), fake_scoring(1), fake_scoring(2)], loro=loro
        )
        self.assertFalse(record["F_changes_at_most_one"])
        self.assertFalse(record["pass"])

    def test_exact_f_is_a_vacuously_legal_zero_change(self):
        record = self.record(
            [fake_scoring(0), fake_scoring(1), fake_scoring(2)], loro=self.full
        )
        self.assertEqual(record["F_removed"], [])
        self.assertEqual(record["F_added"], [])
        self.assertTrue(record["replacement_legal_v7_swap_3_of_3"])
        self.assertTrue(record["pass"])


class LocoBoundaryTests(unittest.TestCase):
    def test_loco_is_absent_from_the_decision_function(self):
        self.assertEqual(
            list(inspect.signature(selector._final_pass).parameters),
            ["full_candidate", "loro_records"],
        )
        self.assertTrue(
            selector._final_pass(
                {"F": ["A"]}, [{"pass": True}, {"pass": True}, {"pass": True}]
            )
        )

    def test_drop_character_filters_both_binary_maps_without_copying_arrays(self):
        marker = [1, 0, 1]
        data = {
            "views": ["A"],
            "characters": list(lock.CHARACTERS),
            "frame_count": 3,
            "E": {("A", character, "G0", "bone"): marker for character in lock.CHARACTERS},
            "T": {("A", character, "bone"): marker for character in lock.CHARACTERS},
        }
        omitted = lock.CHARACTERS[0]
        reduced = selector._drop_character(data, omitted)
        self.assertEqual(len(reduced["characters"]), 3)
        self.assertNotIn(omitted, reduced["characters"])
        self.assertTrue(all(key[1] != omitted for key in reduced["E"]))
        self.assertTrue(all(key[1] != omitted for key in reduced["T"]))
        self.assertIs(next(iter(reduced["E"].values())), marker)


class SelectorOrchestrationTests(unittest.TestCase):
    def test_full_three_loro_and_report_only_loco_are_wired_to_the_decision(self):
        scorings = [fake_scoring(index) for index in range(3)]
        prepared = [
            {
                "index": index,
                "label": "R{}".format(index + 1),
                "data": {
                    "views": list(SMALL_VIEWS),
                    "characters": ["F01", "F02"],
                    "frame_count": 1,
                    "E": {},
                    "T": {},
                },
                "scoring": scoring,
            }
            for index, scoring in enumerate(scorings)
        ]
        with (
            mock.patch.object(selector, "SELECTED_COUNT", 2),
            mock.patch.object(selector, "RESERVE_COUNT", 1),
            mock.patch.object(selector, "CHARACTERS", ("F01",)),
            mock.patch.object(
                selector, "pairwise_angle_table", return_value=small_angle_table()
            ),
            mock.patch.object(selector, "_prepare_repeats", return_value=prepared),
            mock.patch.object(selector.v7, "score_subsets", return_value=fake_scoring(0)),
        ):
            result = selector.select([{}, {}, {}], leaderboard_limit=3)
        self.assertTrue(result["pass"])
        self.assertEqual(result["pairs_audited"], 12)
        self.assertEqual(result["pairs_expected"], 12)
        self.assertEqual(len(result["loro"]), 3)
        self.assertTrue(all(record["pass"] for record in result["loro"]))
        self.assertEqual(len(result["loco"]), 1)
        self.assertTrue(result["loco"][0]["report_only"])
        self.assertEqual(result["selected_view_ids"], result["chosen"]["F"])
        self.assertFalse(result["formal_capture_authorized"])


class ModuleBoundaryTests(unittest.TestCase):
    def test_selector_contains_no_target_error_or_per_character_optimisation(self):
        source = (TOOLS / "select_common_bank_18_9_5_v8.py").read_text(encoding="utf-8")
        lowered = source.lower()
        for forbidden in ("gt_error", "ground_truth", "mpjpe", "per_character_optimum"):
            self.assertNotIn(forbidden, lowered)

    def test_selector_has_no_simulator_or_array_library_dependency(self):
        source = (TOOLS / "select_common_bank_18_9_5_v8.py").read_text(encoding="utf-8")
        for forbidden in ("import omni", "from pxr", "import carb", "import numpy"):
            self.assertNotIn(forbidden, source)

    def test_v7_scoring_is_reused_instead_of_reimplemented(self):
        source = (TOOLS / "select_common_bank_18_9_5_v8.py").read_text(encoding="utf-8")
        self.assertIn("v7.load_matrix", source)
        self.assertIn("v7.score_subsets", source)
        self.assertIn("v7._swap_targets", source)


if __name__ == "__main__":
    unittest.main()
