from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import analyse_view_ac_measured_first_exploratory_v2 as v2  # noqa: E402


def row(valid="1", depth="3.6", x="1", visibility="0.9"):
    return {"valid": valid, "depth_m": depth, "x_m": x, "y_m": "2", "z_m": "3", "visibility": visibility}


class MeasuredFirstTests(unittest.TestCase):
    def replay(self, status_depth="3.6", k2_valid="1", k4_valid="1"):
        raw = {
            (0, "right_elbow"): row(depth=status_depth), (0, "right_wrist"): row(depth=status_depth),
            (0, "right_shoulder"): row(depth="3.5"), (0, "pelvis"): row(depth="3.5"),
        }
        k2 = {(0, joint): row(valid=k2_valid) for joint in ("right_elbow", "right_wrist")}
        k4 = {(0, joint): row(valid=k4_valid) for joint in ("right_elbow", "right_wrist")}
        return {"raw_measured": raw, "k2_guarded": k2, "k4_inferred": k4}

    def test_reliable_measured_is_preferred(self):
        point, source, status, _ = v2.measured_first_candidate(self.replay(), 0, "right_wrist")
        self.assertIsNotNone(point)
        self.assertEqual(source, "k2_reliable_measured")
        self.assertEqual(status, "reliable")

    def test_occluder_depth_uses_k4_fallback(self):
        point, source, status, _ = v2.measured_first_candidate(self.replay(status_depth="1.975"), 0, "right_wrist")
        self.assertIsNotNone(point)
        self.assertEqual(source, "k4_fallback_inferred")
        self.assertEqual(status, "occluded_suspected")

    def test_missing_k2_uses_k4_fallback(self):
        point, source, _, _ = v2.measured_first_candidate(self.replay(k2_valid="0"), 0, "right_wrist")
        self.assertIsNotNone(point)
        self.assertEqual(source, "k4_fallback_inferred")

    def test_arm_score_prioritises_measured_count(self):
        candidate = (None, "unavailable", "no_measured_depth", 0.9)
        measured = (None, "k2_reliable_measured", "reliable", 0.8)
        inferred = (None, "k4_fallback_inferred", "no_measured_depth", 0.99)
        self.assertGreater(v2.arm_score({"e": measured, "w": measured}), v2.arm_score({"e": inferred, "w": candidate}))


if __name__ == "__main__":
    unittest.main()
