from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import common_bank_randomized_block_schedule_v1 as schedule_module  # noqa: E402


class InterfaceAndProvenanceTests(unittest.TestCase):
    def test_public_interfaces_return_dictionaries(self):
        schedule = schedule_module.build_schedule()
        report = schedule_module.validate_schedule(schedule)
        self.assertIsInstance(schedule, dict)
        self.assertIsInstance(report, dict)
        self.assertTrue(report["pass"], report["errors"])
        self.assertTrue(report["valid"], report["errors"])

    def test_build_is_deterministic_and_json_serialisable(self):
        first = schedule_module.build_schedule()
        second = schedule_module.build_schedule()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(json.loads(json.dumps(first)), first)

    def test_build_performs_no_file_io(self):
        with mock.patch("builtins.open", side_effect=AssertionError("unexpected I/O")):
            schedule = schedule_module.build_schedule()
        self.assertFalse(schedule["construction_reads_experimental_results"])
        self.assertFalse(schedule["construction_uses_file_io"])

    def test_seed_and_payload_hash_are_self_consistent(self):
        schedule = schedule_module.build_schedule()
        self.assertEqual(
            schedule["seed_sha256"],
            hashlib.sha256(schedule_module.SEED_TEXT.encode("utf-8")).hexdigest(),
        )
        report = schedule_module.validate_schedule(schedule)
        self.assertEqual(report["schedule_sha256"], report["calculated_schedule_sha256"])
        self.assertEqual(
            schedule["schedule_sha256"],
            "3947750b234113fd1ff16e90aaa7bae548945f0ca3082c33d07e0b2d25984cca",
        )


class FactorialAndProductStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = schedule_module.build_schedule()

    def test_twelve_sessions_are_four_roles_by_three_repeats(self):
        sessions = self.schedule["sessions"]
        self.assertEqual(len(sessions), 12)
        self.assertEqual(
            Counter((item["character"], item["repeat"]) for item in sessions),
            Counter(
                (character, repeat)
                for character in schedule_module.CHARACTERS
                for repeat in schedule_module.REPEATS
            ),
        )
        self.assertTrue(
            all(item["repeat_index"] == item["repeat"] for item in sessions)
        )

    def test_every_session_has_six_full_batches_and_eighteen_unique_primaries(self):
        expected_views = set(schedule_module.CANDIDATE_VIEWS)
        for session in self.schedule["sessions"]:
            self.assertEqual(len(session["batches"]), 6)
            primaries = []
            for batch in session["batches"]:
                self.assertEqual(len(batch["products"]), 4)
                self.assertEqual(
                    {product["creation_slot"] for product in batch["products"]},
                    {0, 1, 2, 3},
                )
                batch_primaries = [
                    product
                    for product in batch["products"]
                    if product["occurrence_kind"] == "primary"
                ]
                self.assertEqual(len(batch_primaries), 3)
                primaries.extend(product["view_id"] for product in batch_primaries)
            self.assertEqual(len(primaries), 18)
            self.assertEqual(set(primaries), expected_views)

    def test_bridge_A_fills_five_batches_and_B_fills_A_primary_batch(self):
        bridge_a = self.schedule["bridge_A"]
        bridge_b = self.schedule["bridge_B"]
        self.assertNotEqual(bridge_a, bridge_b)
        _, bridge_b_elevation = schedule_module._view_geometry(bridge_b)
        self.assertEqual(bridge_b_elevation, 0)
        self.assertGreaterEqual(
            schedule_module._azimuth_separation_deg(bridge_a, bridge_b),
            schedule_module.BRIDGE_B_MIN_AZIMUTH_SEPARATION_DEG,
        )
        self.assertIn(bridge_b, self.schedule["bridge_B_geometry_eligible"])
        for session in self.schedule["sessions"]:
            primary_batches = {}
            fillers = {}
            for batch in session["batches"]:
                for product in batch["products"]:
                    if product["occurrence_kind"] == "primary":
                        primary_batches[product["view_id"]] = batch["batch_id"]
                    else:
                        fillers[batch["batch_id"]] = product
            self.assertNotEqual(primary_batches[bridge_a], primary_batches[bridge_b])
            self.assertEqual(
                Counter(product["occurrence_kind"] for product in fillers.values()),
                Counter({"bridge_A": 5, "bridge_B": 1}),
            )
            for batch_id, filler in fillers.items():
                if batch_id == primary_batches[bridge_a]:
                    self.assertEqual(filler["occurrence_kind"], "bridge_B")
                    self.assertEqual(filler["view_id"], bridge_b)
                else:
                    self.assertEqual(filler["occurrence_kind"], "bridge_A")
                    self.assertEqual(filler["view_id"], bridge_a)
                self.assertFalse(filler["science_input"])

    def test_all_occurrence_ids_are_globally_unique(self):
        ids = [
            product["occurrence_id"]
            for session in self.schedule["sessions"]
            for batch in session["batches"]
            for product in batch["products"]
        ]
        self.assertEqual(len(ids), 288)
        self.assertEqual(len(set(ids)), 288)


class ExactBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = schedule_module.build_schedule()

    def _primary_rows(self):
        for session in self.schedule["sessions"]:
            for batch in session["batches"]:
                for product in batch["products"]:
                    if product["occurrence_kind"] == "primary":
                        yield session, batch, product

    def test_each_view_primary_uses_each_batch_exactly_twice(self):
        counts = defaultdict(Counter)
        for _, batch, product in self._primary_rows():
            counts[product["view_id"]][batch["batch_id"]] += 1
        expected = Counter({label: 2 for label in schedule_module.BATCH_LABELS})
        for view in schedule_module.CANDIDATE_VIEWS:
            self.assertEqual(counts[view], expected)

    def test_same_character_repeats_use_three_batches_and_three_slots(self):
        batches = defaultdict(set)
        slots = defaultdict(set)
        for session, batch, product in self._primary_rows():
            key = (session["character"], product["view_id"])
            batches[key].add(batch["batch_id"])
            slots[key].add(product["creation_slot"])
        for character in schedule_module.CHARACTERS:
            for view in schedule_module.CANDIDATE_VIEWS:
                self.assertEqual(len(batches[(character, view)]), 3)
                self.assertEqual(len(slots[(character, view)]), 3)
                self.assertTrue(slots[(character, view)].issubset({0, 1, 2, 3}))

    def test_each_primary_view_uses_all_four_creation_slots_three_times(self):
        counts = defaultdict(Counter)
        for _, _, product in self._primary_rows():
            counts[product["view_id"]][product["creation_slot"]] += 1
        expected = Counter({slot: 3 for slot in range(4)})
        for view in schedule_module.CANDIDATE_VIEWS:
            self.assertEqual(counts[view], expected)

    def test_primary_filler_and_both_bridges_are_role_slot_balanced(self):
        primary = defaultdict(Counter)
        filler = defaultdict(Counter)
        bridge_a = defaultdict(Counter)
        bridge_b = defaultdict(Counter)
        for session in self.schedule["sessions"]:
            character = session["character"]
            for batch in session["batches"]:
                for product in batch["products"]:
                    target = (
                        primary
                        if product["occurrence_kind"] == "primary"
                        else filler
                    )
                    target[character][product["creation_slot"]] += 1
                    if product["occurrence_kind"] == "bridge_A":
                        bridge_a[character][product["creation_slot"]] += 1
                    elif product["occurrence_kind"] == "bridge_B":
                        bridge_b[character][product["creation_slot"]] += 1
        for character in schedule_module.CHARACTERS:
            self.assertEqual(sorted(primary[character].values()), [13, 13, 14, 14])
            self.assertEqual(sorted(filler[character].values()), [4, 4, 5, 5])
            self.assertEqual(sorted(bridge_a[character].values()), [3, 4, 4, 4])
            self.assertEqual(sorted(bridge_b[character].values()), [1, 1, 1])

    def test_bridge_slots_are_exactly_balanced_globally_and_by_batch(self):
        all_filler = Counter()
        bridge_a = Counter()
        bridge_b = Counter()
        by_batch = defaultdict(Counter)
        for session in self.schedule["sessions"]:
            for batch in session["batches"]:
                filler = next(
                    product
                    for product in batch["products"]
                    if product["occurrence_kind"] != "primary"
                )
                slot = filler["creation_slot"]
                all_filler[slot] += 1
                by_batch[batch["batch_id"]][slot] += 1
                if filler["occurrence_kind"] == "bridge_A":
                    bridge_a[slot] += 1
                else:
                    bridge_b[slot] += 1
        self.assertEqual(all_filler, Counter({slot: 18 for slot in range(4)}))
        self.assertEqual(bridge_a, Counter({slot: 15 for slot in range(4)}))
        self.assertEqual(bridge_b, Counter({slot: 3 for slot in range(4)}))
        for batch in schedule_module.BATCH_LABELS:
            self.assertEqual(by_batch[batch], Counter({slot: 3 for slot in range(4)}))

    def test_each_batch_label_uses_each_execution_position_twice(self):
        counts = defaultdict(Counter)
        for session in self.schedule["sessions"]:
            for batch in session["batches"]:
                counts[batch["batch_id"]][batch["execution_position"]] += 1
        expected = Counter({position: 2 for position in range(1, 7)})
        for label in schedule_module.BATCH_LABELS:
            self.assertEqual(counts[label], expected)

    def test_each_view_also_uses_each_execution_position_twice(self):
        counts = defaultdict(Counter)
        for _, batch, product in self._primary_rows():
            counts[product["view_id"]][batch["execution_position"]] += 1
        expected = Counter({position: 2 for position in range(1, 7)})
        for view in schedule_module.CANDIDATE_VIEWS:
            self.assertEqual(counts[view], expected)

    def test_all_scenario_orders_occur_twice(self):
        counts = Counter(
            tuple(session["scenario_order"]) for session in self.schedule["sessions"]
        )
        expected_orders = set(itertools.permutations(schedule_module.SCENARIOS))
        self.assertEqual(set(counts), expected_orders)
        self.assertTrue(all(counts[order] == 2 for order in expected_orders))

    def test_role_session_order_is_counterbalanced(self):
        position_counts = Counter(
            session["role_session_position"] for session in self.schedule["sessions"]
        )
        self.assertEqual(position_counts, Counter({1: 3, 2: 3, 3: 3, 4: 3}))
        for character in schedule_module.CHARACTERS:
            positions = {
                session["role_session_position"]
                for session in self.schedule["sessions"]
                if session["character"] == character
            }
            self.assertEqual(len(positions), 3)


class ValidatorRejectionTests(unittest.TestCase):
    def setUp(self):
        self.schedule = schedule_module.build_schedule()

    def assertRejected(self, schedule):
        report = schedule_module.validate_schedule(schedule)
        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"])

    def test_non_mapping_is_rejected_without_raising(self):
        report = schedule_module.validate_schedule([])
        self.assertFalse(report["pass"])
        self.assertFalse(report["valid"])

    def test_malformed_nested_values_are_rejected_without_raising(self):
        damaged = copy.deepcopy(self.schedule)
        damaged["sessions"][0]["scenario_order"] = [["not-hashable"]]
        self.assertRejected(damaged)

    def test_infinite_indices_and_slots_are_rejected_without_raising(self):
        paths = (
            ("global_execution_index",),
            ("batches", 0, "execution_position"),
            ("batches", 0, "products", 0, "creation_slot"),
        )
        for path in paths:
            with self.subTest(path=path):
                damaged = copy.deepcopy(self.schedule)
                target = damaged["sessions"][0]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = float("inf")
                self.assertRejected(damaged)

    def test_changed_seed_is_rejected(self):
        damaged = copy.deepcopy(self.schedule)
        damaged["seed_text"] += "-changed"
        self.assertRejected(damaged)

    def test_changed_primary_identity_is_rejected(self):
        damaged = copy.deepcopy(self.schedule)
        first_batch = damaged["sessions"][0]["batches"][0]
        primaries = [
            product
            for product in first_batch["products"]
            if product["occurrence_kind"] == "primary"
        ]
        primaries[0]["view_id"] = primaries[1]["view_id"]
        self.assertRejected(damaged)

    def test_changed_creation_slot_is_rejected(self):
        damaged = copy.deepcopy(self.schedule)
        products = damaged["sessions"][0]["batches"][0]["products"]
        products[0]["creation_slot"] = products[1]["creation_slot"]
        self.assertRejected(damaged)

    def test_changed_bridge_is_rejected(self):
        damaged = copy.deepcopy(self.schedule)
        session = damaged["sessions"][0]
        filler = next(
            product
            for batch in session["batches"]
            for product in batch["products"]
            if product["occurrence_kind"] == "bridge_B"
        )
        filler["view_id"] = damaged["bridge_A"]
        self.assertRejected(damaged)

    def test_changed_scenario_order_is_rejected(self):
        damaged = copy.deepcopy(self.schedule)
        damaged["sessions"][0]["scenario_order"] = ["G0", "G0", "W"]
        self.assertRejected(damaged)

    def test_forged_payload_hash_alone_is_rejected(self):
        damaged = copy.deepcopy(self.schedule)
        damaged["schedule_sha256"] = "0" * 64
        self.assertRejected(damaged)


if __name__ == "__main__":
    unittest.main()
