from __future__ import annotations

import inspect
import itertools
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import common_bank_successor_v2_probe_selection as selector  # noqa: E402


def event(
    side="left",
    clear_before=5,
    occluded=20,
    clear_after=5,
    found=None,
    bone=None,
):
    if found is None:
        found = clear_before >= 5 and occluded >= 20 and clear_after >= 5
    return {
        "found": found,
        "bone": selector.SIDE_BONES[side][0] if bone is None else bone,
        "clear_before_frames": clear_before,
        "occluded_frames": occluded,
        "clear_after_frames": clear_after,
    }


def empty_repeat():
    return {
        view_id: {
            character: {side: event(side, 0, 0, 0) for side in selector.SIDES}
            for character in selector.CHARACTERS
        }
        for view_id in selector.PROBE_VIEW_IDS
    }


def angle_table(view_ids, default=10):
    return {
        first: {
            second: 0 if first == second else default
            for second in view_ids
        }
        for first in view_ids
    }


class LayoutTests(unittest.TestCase):
    def test_grid_is_exactly_24_unique_views_on_the_registered_fans(self):
        self.assertEqual(len(selector.PROBE_LAYOUT), 24)
        self.assertEqual(len(set(selector.PROBE_VIEW_IDS)), 24)
        self.assertEqual(
            {entry["azimuth_deg"] for entry in selector.PROBE_LAYOUT},
            {345.0, 0.0, 15.0, 45.0, 60.0, 75.0},
        )
        self.assertEqual(selector.RADIUS_M, 3.5)

    def test_angle_table_is_symmetric_integer_and_zero_on_diagonal(self):
        table = selector.pairwise_angle_table()
        for first in selector.PROBE_VIEW_IDS:
            self.assertEqual(table[first][first], 0)
            for second in selector.PROBE_VIEW_IDS:
                self.assertIsInstance(table[first][second], int)
                self.assertEqual(table[first][second], table[second][first])


class EventTests(unittest.TestCase):
    def test_exact_5_20_5_event_has_zero_slack(self):
        self.assertEqual(selector.side_summary_slacks(event(), "left"), (0, 0))

    def test_short_clear_or_occluded_run_is_not_eligible(self):
        self.assertIsNone(selector.side_summary_slacks(event(clear_before=4), "left"))
        self.assertIsNone(selector.side_summary_slacks(event(occluded=19), "left"))
        self.assertIsNone(selector.side_summary_slacks(event(clear_after=4), "left"))

    def test_positive_below_threshold_and_counts_beyond_the_window_are_rejected(self):
        broken = event(clear_before=4, found=True)
        with self.assertRaises(selector.ProbeSelectionError):
            selector.side_summary_slacks(broken, "left")
        with self.assertRaises(selector.ProbeSelectionError):
            selector.side_summary_slacks(event("left", 30, 30, 30), "left")

    def test_negative_record_can_retain_near_miss_counts_from_replay(self):
        self.assertIsNone(
            selector.side_summary_slacks(event("left", 15, 30, 15, found=False), "left")
        )

    def test_bone_must_match_the_side(self):
        wrong = event()
        wrong["bone"] = "right_thigh"
        with self.assertRaises(selector.ProbeSelectionError):
            selector.side_summary_slacks(wrong, "left")

    def test_replay_side_winner_uses_the_frozen_five_field_order(self):
        thigh, shank = selector.SIDE_BONES["left"]
        clear_wins = selector.choose_side_summary(
            [
                event("left", 6, 20, 6, bone=thigh),
                event("left", 5, 50, 5, bone=shank),
            ],
            "left",
        )
        self.assertEqual(clear_wins["bone"], thigh)
        total_clear_wins = selector.choose_side_summary(
            [
                event("left", 7, 30, 9, bone=thigh),
                event("left", 7, 30, 8, bone=shank),
            ],
            "left",
        )
        self.assertEqual(total_clear_wins["bone"], thigh)

    def test_side_summary_schema_is_closed_and_typed(self):
        missing = event()
        del missing["occluded_frames"]
        with self.assertRaises(selector.ProbeSelectionError):
            selector.side_summary_slacks(missing, "left")
        wrong_type = event()
        wrong_type["found"] = 1
        with self.assertRaises(selector.ProbeSelectionError):
            selector.side_summary_slacks(wrong_type, "left")
        extra = event()
        extra["gt_error"] = 0.0
        with self.assertRaises(selector.ProbeSelectionError):
            selector.side_summary_slacks(extra, "left")


class RobustCarrierTests(unittest.TestCase):
    def test_view_is_robust_only_for_same_stratum_in_both_repeats(self):
        repeats = {"rep_01": empty_repeat(), "rep_02": empty_repeat()}
        view = selector.PROBE_VIEW_IDS[0]
        repeats["rep_01"][view]["F01"]["left"] = event("left", 8, 30, 9)
        repeats["rep_02"][view]["F01"]["left"] = event("left", 7, 28, 6)
        repeats["rep_01"][view]["F01"]["right"] = event("right")
        result = selector.robust_carrier_profiles(repeats)
        self.assertEqual(result["profiles"][("F01", "left")][view], (1, 8))
        self.assertNotIn(view, result["profiles"][("F01", "right")])

    def test_exactly_two_repeat_ids_and_all_views_are_required(self):
        repeats = {"rep_01": empty_repeat()}
        with self.assertRaises(selector.ProbeSelectionError):
            selector.robust_carrier_profiles(repeats)
        repeats = {"rep_01": empty_repeat(), "rep_02": empty_repeat()}
        del repeats["rep_01"][selector.PROBE_VIEW_IDS[-1]]
        with self.assertRaises(selector.ProbeSelectionError):
            selector.robust_carrier_profiles(repeats)

    def test_closed_schema_rejects_result_fields(self):
        repeats = {"rep_01": empty_repeat(), "rep_02": empty_repeat()}
        view = selector.PROBE_VIEW_IDS[0]
        repeats["rep_01"][view]["F01"]["left"]["gt_error"] = 0.0
        with self.assertRaises(selector.ProbeSelectionError):
            selector.robust_carrier_profiles(repeats)

    def test_public_selector_exhausts_24_choose_12_and_fails_with_null_ids(self):
        result = selector.select(
            {"rep_01": empty_repeat(), "rep_02": empty_repeat()}
        )
        self.assertFalse(result["pass"])
        self.assertIsNone(result["selected_view_ids"])
        self.assertEqual(result["subsets_audited"], math.comb(24, 12))
        self.assertEqual(result["qualifying_subsets"], 0)


class ExhaustiveSearchTests(unittest.TestCase):
    def search(self, profiles, view_ids, selected=3, minimum=3, table=None):
        return selector._search_profiles(
            profiles,
            view_ids,
            angle_table(view_ids) if table is None else table,
            selected,
            minimum,
        )

    def test_every_small_combination_is_audited(self):
        views = ("A", "B", "C", "D", "E")
        profiles = {"s": {view: (0, 0) for view in views}}
        result = self.search(profiles, views)
        self.assertEqual(result["subsets_audited"], math.comb(5, 3))
        self.assertEqual(result["subsets_expected"], math.comb(5, 3))
        self.assertEqual(result["qualifying_subsets"], math.comb(5, 3))

    def test_no_eligible_subset_fails_closed_with_null_ids(self):
        views = ("A", "B", "C", "D")
        result = self.search({"s": {"A": (0, 0), "B": (0, 0)}}, views)
        self.assertIsNone(result["selected_view_ids"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["qualifying_subsets"], 0)

    def test_clear_slack_is_first_ordering_key(self):
        views = ("A", "B", "C", "D")
        profiles = {
            "s": {"A": (9, 0), "B": (8, 0), "C": (7, 0), "D": (1, 99)}
        }
        self.assertEqual(self.search(profiles, views)["selected_view_ids"], ["A", "B", "C"])

    def test_occluded_slack_is_second_ordering_key(self):
        views = ("A", "B", "C", "D")
        profiles = {
            "s": {"A": (5, 9), "B": (5, 8), "C": (5, 7), "D": (5, 1)}
        }
        self.assertEqual(self.search(profiles, views)["selected_view_ids"], ["A", "B", "C"])

    def test_minimum_carrier_count_precedes_total_incidence(self):
        views = ("A", "B", "C", "D", "E", "F")
        profiles = {
            "s1": {view: (0, 0) for view in ("A", "B", "C", "E")},
            "s2": {view: (0, 0) for view in ("A", "B", "C", "D", "F")},
            "s3": {view: (0, 0) for view in ("A", "B", "C", "D", "F")},
            "s4": {view: (0, 0) for view in ("A", "B", "C", "D", "F")},
        }
        result = self.search(profiles, views, selected=5)
        self.assertEqual(result["selected_view_ids"], ["A", "B", "C", "D", "E"])
        self.assertEqual(result["score"][2], 4)
        self.assertEqual(result["score"][3], 16)

    def test_total_incidence_breaks_equal_minimum_count(self):
        views = ("A", "B", "C", "D", "E")
        profiles = {
            "s1": {view: (0, 0) for view in ("A", "B", "C")},
            "s2": {view: (0, 0) for view in ("A", "B", "C", "E")},
        }
        result = self.search(profiles, views, selected=4)
        self.assertEqual(result["selected_view_ids"], ["A", "B", "C", "E"])
        self.assertEqual(result["score"][2], 3)
        self.assertEqual(result["score"][3], 7)

    def test_angle_precedes_lexical_tie_break(self):
        views = ("A", "B", "C", "D")
        profiles = {"s": {view: (0, 0) for view in views}}
        table = angle_table(views, default=10)
        table["A"]["B"] = table["B"]["A"] = 5
        for first, second in (("A", "D"), ("B", "D"), ("C", "D")):
            table[first][second] = table[second][first] = 20
        self.assertEqual(
            self.search(profiles, views, table=table)["selected_view_ids"],
            ["A", "C", "D"],
        )

    def test_lexicographically_smallest_ids_win_a_complete_tie(self):
        views = ("A", "B", "C", "D")
        profiles = {"s": {view: (0, 0) for view in views}}
        self.assertEqual(self.search(profiles, views)["selected_view_ids"], ["A", "B", "C"])

    def test_mask_implementation_matches_an_independent_brute_force_reference(self):
        views = ("A", "B", "C", "D", "E", "F", "G")
        profiles = {
            "s1": {view: (index % 4, (index * 3) % 7) for index, view in enumerate(views)},
            "s2": {
                view: ((index + 2) % 5, (index * 2 + 1) % 6)
                for index, view in enumerate(views)
                if view != "G"
            },
            "s3": {
                view: ((index * 3) % 6, index % 3)
                for index, view in enumerate(views)
                if view not in ("A", "F")
            },
        }
        table = angle_table(views)
        for first_index, first in enumerate(views):
            for second_index, second in enumerate(views):
                if first != second:
                    table[first][second] = abs(first_index - second_index) * 11 + 3

        qualifying = []
        for combination in itertools.combinations(views, 4):
            values = {
                stratum: [profile[view] for view in combination if view in profile]
                for stratum, profile in profiles.items()
            }
            if any(len(items) < 3 for items in values.values()):
                continue
            counts = [len(items) for items in values.values()]
            score = (
                min(sorted((item[0] for item in items), reverse=True)[2] for items in values.values()),
                min(sorted((item[1] for item in items), reverse=True)[2] for items in values.values()),
                min(counts),
                sum(counts),
                min(table[first][second] for first, second in itertools.combinations(combination, 2)),
            )
            qualifying.append((score, combination))
        reference_score = max(item[0] for item in qualifying)
        reference_ids = min(item[1] for item in qualifying if item[0] == reference_score)
        result = self.search(profiles, views, selected=4, table=table)
        self.assertEqual(result["qualifying_subsets"], len(qualifying))
        self.assertEqual(result["score"], list(reference_score))
        self.assertEqual(result["selected_view_ids"], list(reference_ids))


class IsolationTests(unittest.TestCase):
    def test_module_has_no_file_or_old_selector_dependency(self):
        source = inspect.getsource(selector)
        self.assertNotIn("select_common_bank_18_9_5", source)
        self.assertNotIn("open(", source)
        self.assertNotIn("write_text", source)
        self.assertNotIn("write_bytes", source)
        imports = {
            line.strip()
            for line in source.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        }
        self.assertEqual(
            imports,
            {
                "from __future__ import annotations",
                "import math",
                "from decimal import Decimal, ROUND_HALF_EVEN",
                "from typing import Mapping, Sequence",
            },
        )


if __name__ == "__main__":
    unittest.main()
