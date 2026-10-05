from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_common_bank_availability_v1 as analysis  # noqa: E402
from fs_cts5_camera_bank_v1 import BONES, JOINTS  # noqa: E402


VIEWS = [
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
]
LEGS = ("left_thigh", "right_thigh", "left_shank", "right_shank")


def joints_for(measured_bones):
    """Endpoints available exactly for the bones listed."""
    available = {joint: False for joint in JOINTS}
    for bone in measured_bones:
        for endpoint in BONES[bone]:
            available[endpoint] = True
    return available


def matrix(spec, characters=("F01",), frames=("0",)):
    """``spec[scenario][view]`` -> the bones that view measures, for every cell."""
    return {
        character: {
            scenario: {
                frame: {view: joints_for(spec[scenario][view]) for view in VIEWS}
                for frame in frames
            }
            for scenario in spec
        }
        for character in characters
    }


def record_for(availability):
    return {
        "attempt": "attempt_test",
        "classification": "excluded_engineering_asset_not_formal_data",
        "formal_capture_authorized": False,
        "bank": [{"name": view} for view in VIEWS],
        "availability": availability,
        "selection": {"evidence_used": ["endpoint_availability"]},
    }


def all_bones_spec():
    everything = tuple(BONES)
    return {
        "G0": {view: everything for view in VIEWS},
        "G1": {view: everything for view in VIEWS},
    }


class BoneMeasuredTests(unittest.TestCase):
    def test_needs_both_endpoints(self):
        spec = all_bones_spec()
        spec["G0"]["v1"] = ("left_thigh",)
        availability = matrix(spec)
        self.assertTrue(
            analysis.bone_measured(availability, "F01", "G0", "0", "v1", "left_thigh")
        )
        # left_shank shares left_knee with left_thigh but its ankle is missing.
        self.assertFalse(
            analysis.bone_measured(availability, "F01", "G0", "0", "v1", "left_shank")
        )

    def test_missing_cell_is_an_error_not_a_false(self):
        availability = matrix(all_bones_spec())
        with self.assertRaises(analysis.BankAnalysisError):
            analysis.bone_measured(availability, "F01", "G0", "99", "v1", "left_thigh")


class ViewProfileTests(unittest.TestCase):
    def test_a_bone_never_measured_in_the_clean_render_is_blind(self):
        spec = all_bones_spec()
        spec["G0"]["v1"] = LEGS
        profiles = analysis.view_profiles(matrix(spec), VIEWS)
        self.assertEqual(
            sorted(profiles["v1"]["blind_bones_clean"]),
            sorted(("left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm")),
        )
        self.assertEqual(profiles["v2"]["blind_bones_clean"], [])

    def test_a_bone_lost_on_some_frames_is_partial_not_blind(self):
        spec = all_bones_spec()
        availability = matrix(spec, frames=("0", "30"))
        availability["F01"]["G0"]["30"]["v1"] = joints_for(LEGS)
        profiles = analysis.view_profiles(availability, VIEWS)
        self.assertEqual(profiles["v1"]["blind_bones_clean"], [])
        self.assertEqual(
            profiles["v1"]["partial_bones_clean"]["left_forearm"],
            {"measured": 1, "total": 2},
        )


class PropOcclusionTests(unittest.TestCase):
    def test_counts_only_what_the_props_took(self):
        spec = all_bones_spec()
        spec["G1"]["v1"] = ("right_thigh", "right_shank")
        strength = analysis.prop_occlusion_strength(matrix(spec), VIEWS, "G1")
        self.assertEqual(strength["v1"]["occluded_cells"], 2)
        self.assertEqual(strength["v1"]["target_cells"], len(LEGS))
        self.assertEqual(strength["v2"]["occluded_cells"], 0)

    def test_self_occlusion_is_not_billed_to_the_props(self):
        """A bone the view could not measure with the props hidden is not a loss."""
        spec = all_bones_spec()
        spec["G0"]["v1"] = ("right_thigh", "right_shank")
        spec["G1"]["v1"] = ("right_thigh", "right_shank")
        strength = analysis.prop_occlusion_strength(matrix(spec), VIEWS, "G1")
        self.assertEqual(strength["v1"]["occluded_cells"], 0)


class CellCensusTests(unittest.TestCase):
    def test_histograms_and_hardest_cell(self):
        spec = all_bones_spec()
        for view in ("v1", "v2", "v3", "v4", "v5", "v6"):
            spec["G1"][view] = ("left_thigh", "left_shank", "right_thigh")
        census = analysis.cell_census(matrix(spec), VIEWS, "G1")
        self.assertEqual(census["cells"], len(LEGS))
        self.assertEqual(census["measurable_views_histogram"], {3: 1, 9: 3})
        self.assertEqual(census["prop_occluded_views_histogram"], {0: 3, 6: 1})
        hardest = census["hardest_cells"][0]
        self.assertEqual(hardest["bone"], "right_shank")
        self.assertEqual(hardest["measuring"], ["v7", "v8", "v9"])


class SubsetMarginTests(unittest.TestCase):
    def test_forced_views_when_only_three_views_can_see_a_cell(self):
        spec = all_bones_spec()
        for view in ("v1", "v2", "v3", "v4", "v5", "v6"):
            spec["G1"][view] = ("left_thigh", "left_shank", "right_thigh")
        margins = analysis.subset_margins(matrix(spec), VIEWS)
        self.assertEqual(margins["forced_views"], ["v7", "v8", "v9"])
        self.assertEqual(margins["free_slots"], 2)
        self.assertEqual(margins["best_margin"], 0)
        self.assertTrue(margins["feasible_count"] < margins["subset_count"])
        for entry in margins["feasible_subsets"]:
            self.assertTrue({"v7", "v8", "v9"}.issubset(set(entry["subset"])))

    def test_an_unconstrained_bank_has_no_forced_view_and_margin_two(self):
        margins = analysis.subset_margins(matrix(all_bones_spec()), VIEWS)
        self.assertEqual(margins["forced_views"], [])
        self.assertEqual(margins["feasible_count"], margins["subset_count"])
        self.assertEqual(margins["best_margin"], 2)

    def test_near_miss_subsets_are_one_view_short(self):
        spec = all_bones_spec()
        for view in ("v1", "v2", "v3", "v4", "v5", "v6", "v7"):
            spec["G1"][view] = ("left_thigh", "left_shank", "right_thigh")
        margins = analysis.subset_margins(matrix(spec), VIEWS)
        self.assertEqual(margins["feasible_count"], 0)
        self.assertEqual(margins["forced_views"], [])
        self.assertIsNone(margins["free_slots"])
        self.assertTrue(margins["near_miss_count"] > 0)
        for entry in margins["near_miss"]:
            self.assertEqual(entry["min_measured_views"], 2)


class DiagnoseTests(unittest.TestCase):
    def test_rejects_ground_truth_evidence(self):
        record = record_for(matrix(all_bones_spec()))
        record["selection"]["evidence_used"] = ["endpoint_availability", "gt_error"]
        with self.assertRaises(Exception):
            analysis.diagnose(record)

    def test_rejects_a_record_without_a_matrix(self):
        record = record_for({})
        with self.assertRaises(analysis.BankAnalysisError):
            analysis.diagnose(record)

    def test_reports_scenarios_views_and_frames(self):
        diagnosis = analysis.diagnose(record_for(matrix(all_bones_spec())))
        self.assertEqual(diagnosis["scenarios"], ["G0", "G1"])
        self.assertEqual(diagnosis["views"], VIEWS)
        self.assertEqual(diagnosis["sampled_frames"]["G1"], ["0"])
        self.assertFalse(diagnosis["formal_capture_authorized"])
        self.assertFalse(diagnosis["gt_error_used"])
        self.assertNotIn("G2", diagnosis["cell_census"])


class CliTests(unittest.TestCase):
    def test_records_are_write_once(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "availability.json"
            source.write_text(
                json.dumps(record_for(matrix(all_bones_spec()))), encoding="utf-8"
            )
            target = Path(folder) / "out" / "diagnosis.json"
            self.assertEqual(analysis.main([str(source), str(target)]), 0)
            written = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(written["source_record_sha256"], analysis._sha256_file(source))
            with self.assertRaises(analysis.BankAnalysisError):
                analysis.main([str(source), str(target)])


if __name__ == "__main__":
    unittest.main()
