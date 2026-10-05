import unittest
from collections import Counter
from tools.common_bank_successor_v4_schedule import *

class ScheduleTests(unittest.TestCase):
    def test_library_and_balance(self):
        self.assertEqual(24, len(CANDIDATE_IDS)); self.assertEqual(24, len(set(CANDIDATE_IDS)))
        for character in CHARACTERS:
            slots = {view: [] for view in CANDIDATE_IDS}
            for repeat in REPEATS:
                schedule = session_schedule(character, repeat)
                self.assertEqual(8, len(schedule))
                for batch in schedule:
                    self.assertEqual(BRIDGE_ID, batch["view_ids"][0])
                    for slot, view in enumerate(batch["view_ids"][1:]): slots[view].append(slot)
            self.assertTrue(all(sorted(value) == [0,1,2] for value in slots.values()))
    def test_twelve_sessions_deterministic(self):
        self.assertEqual(full_schedule(), full_schedule()); self.assertEqual(12, len(full_schedule()))
