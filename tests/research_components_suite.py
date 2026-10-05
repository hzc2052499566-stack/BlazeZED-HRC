"""Run the public supplemental research-component tests with synthetic inputs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "test_analyze_common_bank_availability_v1",
    "test_common_bank_randomized_block_schedule_v1",
    "test_common_bank_rule_lock_v1",
    "test_common_bank_successor_v2_probe_selection",
    "test_common_bank_successor_v4_schedule",
    "test_common_bank_successor_v4_selection",
    "test_common_camera_bank_v1",
    "test_fs_cts5_known_occluder_surface_rejection_v1",
    "test_occlusion_pilot_v1",
    "test_select_common_bank_18_9_5_v1",
    "test_select_common_bank_18_9_5_v3",
    "test_select_common_bank_18_9_5_v4",
    "test_select_common_bank_18_9_5_v5",
    "test_select_common_bank_18_9_5_v6",
    "test_select_common_bank_18_9_5_v7",
    "test_select_common_bank_18_9_5_v8",
    "test_ur10e_kinematics_v1",
)


def main() -> int:
    for directory in (ROOT, ROOT / "tools", ROOT / "tests"):
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
    suite = unittest.TestSuite()
    for module in MODULES:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    print(f"Running {len(MODULES)} supplemental synthetic-fixture test modules.")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
