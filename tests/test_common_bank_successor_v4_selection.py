import unittest
from tools.common_bank_successor_v4_selection import *

def fixture(passing):
    summary={"found":passing,"bone":"left_thigh","clear_before_frames":5 if passing else 0,"occluded_frames":20 if passing else 0,"clear_after_frames":5 if passing else 0}
    def side_summary(side):
        value=dict(summary); value["bone"]=f"{side}_thigh"; return value
    return {f"r{i}":{v:{c:{s:side_summary(s) for s in SIDES} for c in CHARACTERS} for v in CANDIDATE_IDS} for i in range(1,4)}

class SelectionTests(unittest.TestCase):
    def test_robust_all_three(self):
        data=fixture(True); data["r2"][CANDIDATE_IDS[0]]["F01"]["left"]={"found":False,"bone":"left_thigh","clear_before_frames":0,"occluded_frames":0,"clear_after_frames":0}
        self.assertNotIn(CANDIDATE_IDS[0], robust_carrier_profiles(data)[("F01","left")])
    def test_zero_is_terminal_and_full_audit(self):
        result=select(fixture(False)); self.assertFalse(result["pass"]); self.assertEqual(EXPECTED_SUBSETS,result["subsets_audited"])
    def test_malformed_flag_rejected(self):
        data=fixture(True); data["r1"][CANDIDATE_IDS[0]]["F01"]["left"]["gt_error"]=0
        with self.assertRaises(SelectionError): robust_carrier_profiles(data)
    def test_deterministic_bone_choice_and_slacks(self):
        values=[{"found":True,"bone":"left_thigh","clear_before_frames":6,"occluded_frames":20,"clear_after_frames":6},{"found":True,"bone":"left_shank","clear_before_frames":7,"occluded_frames":20,"clear_after_frames":7}]
        self.assertEqual("left_shank",choose_side_summary(values,"left")["bone"])
        self.assertEqual((2,0),side_summary_slacks(values[1],"left"))
