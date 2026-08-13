from __future__ import annotations

import sys,unittest
from pathlib import Path

TOOLS=Path(__file__).resolve().parents[1]/"tools"
if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
import multiview_best_view_measured_first_v1 as m  # noqa:E402


class DeploymentMethodTests(unittest.TestCase):
    def test_depth_reliability(self):
        self.assertEqual(m.depth_reliability(None,(3.5,3.5)),"no_measured_depth")
        self.assertEqual(m.depth_reliability(1.975,(3.5,3.5)),"occluded_suspected")
        self.assertEqual(m.depth_reliability(3.6,(3.5,3.5)),"reliable")

    def test_measured_first_and_fallback(self):
        self.assertEqual(m.measured_first((1,2,3),(4,5,6),"reliable")[1],"k2_reliable_measured")
        self.assertEqual(m.measured_first((1,2,3),(4,5,6),"occluded_suspected")[0],(4.0,5.0,6.0))

    def test_arm_selection_and_tie(self):
        def arm(measured,visibility):
            source="k2_reliable_measured" if measured else "k4_fallback_inferred"
            joints={j:{"point_a":(1,2,3),"source":source,"reliability":"reliable","visibility":visibility} for j in m.ARM_JOINTS}
            return {"joints":joints,"score":(2 if measured else 0,1,visibility),"complete":True}
        self.assertEqual(m.select_arm(arm(False,.99),arm(True,.5))["selected_view"],"c")
        self.assertEqual(m.select_arm(arm(True,.9),arm(True,.9))["selected_view"],"a")

    def test_transform(self):
        t={"rotation_3x3":[[1,0,0],[0,1,0],[0,0,1]],"translation_m":[1,2,3]}
        self.assertEqual(m.transform((2,3,4),t),(3.0,5.0,7.0))


if __name__=="__main__":unittest.main()
