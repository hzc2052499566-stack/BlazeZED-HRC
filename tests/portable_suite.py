"""Run the audited, hardware-free unit-test allowlist.

This is intentionally not test discovery. The full test tree includes ZED SDK,
Isaac Sim, local-asset, and frozen-provenance checks that are unsuitable for a
hosted GitHub runner.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIRECTORY = REPOSITORY_ROOT / "tests"

# Keep this list explicit so a newly added hardware test cannot silently enter CI.
PORTABLE_TEST_MODULES = (
    "test_adaptive_camera_objective_v1",
    "test_body38_capture_protocol_v1",
    "test_build_view_ac_engineering_pair_v1",
    "test_clean_static_temporal_followup_v1",
    "test_clean_temporal_tracking_v1",
    "test_compare_dynamic_kinematic_gt_v1",
    "test_depth_sampling_ablation",
    "test_develop_multiview_limb_length_cts5_v2",
    "test_k4_root_fault_injection_v1",
    "test_k4_sustained_occlusion_v1",
    "test_kinematic_constraints",
    "test_kinematic_k3_inferred",
    "test_kinematic_k3_root_guard_shadow_v1",
    "test_kinematic_profile",
    "test_multiperson_tracking_core_v1",
    "test_multiview_best_view_measured_first_v1",
    "test_multiview_conditional_view_c_v1",
    "test_prelock_relock_v2",
    "test_replay_view_ac_common_frame_engineering_v1",
    "test_timecode_marker_v1",
    "test_view_ac_best_view_fusion_pilot_v1",
    "test_view_ac_measured_first_exploratory_v2",
    "test_view_ac_stream_protocol_v1",
)


def build_suite() -> unittest.TestSuite:
    """Load only the audited portable modules."""

    # Direct script execution places tests/, but not necessarily the repository
    # root, on sys.path. Add both deterministically for test and tools imports.
    for path in (REPOSITORY_ROOT, TESTS_DIRECTORY):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module_name in PORTABLE_TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(module_name))
    return suite


def main() -> int:
    print(
        f"Running {len(PORTABLE_TEST_MODULES)} explicitly allowlisted portable "
        "test modules."
    )
    result = unittest.TextTestRunner(verbosity=2).run(build_suite())
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
