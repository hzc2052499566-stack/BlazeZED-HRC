import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import multiview_conditional_view_c_v1 as conditional


def arm(measured_count=2, complete=True):
    joints = {
        "right_elbow": {"point_a": (1.0, 2.0, 3.0), "source": "k2_reliable_measured", "reliability": "reliable", "visibility": 0.9},
        "right_wrist": {"point_a": (1.1, 2.1, 3.1), "source": "k2_reliable_measured", "reliability": "reliable", "visibility": 0.8},
    }
    return {"joints": joints, "score": (measured_count, int(complete), 0.8), "complete": complete}


class ConditionalViewCTest(unittest.TestCase):
    def test_does_not_trigger_when_both_a_joints_are_reliably_measured(self):
        self.assertFalse(conditional.activation_decision(arm())["trigger_view_c"])

    def test_triggers_for_partial_measured_arm(self):
        result = conditional.activation_decision(arm(measured_count=1))
        self.assertTrue(result["trigger_view_c"])
        self.assertEqual(result["reason"], "view_a_partial_reliable_measured_arm")

    def test_triggers_for_incomplete_a(self):
        self.assertTrue(conditional.activation_decision(arm(2, False))["trigger_view_c"])

    def test_untriggered_output_is_view_a(self):
        result = conditional.select_conditionally(arm(), None)
        self.assertEqual(result["selected_view"], "a")
        self.assertEqual(result["joints"]["right_wrist"]["point_a"], (1.1, 2.1, 3.1))

    def test_trigger_requires_view_c(self):
        with self.assertRaises(ValueError):
            conditional.select_conditionally(arm(measured_count=1), None)


if __name__ == "__main__":
    unittest.main()
