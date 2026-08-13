from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import replay_view_ac_common_frame_engineering_v1 as replay  # noqa: E402


def raw_row(
    *,
    sequence: int = 0,
    joint: str = "right_wrist",
    valid: bool = True,
    point: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> dict[str, str]:
    return {
        "run_id": "source_run",
        "sequence_index": str(sequence),
        "cycle_index": "0",
        "animation_frame_code": str(sequence),
        "usd_time_code": str(float(sequence)),
        "method": "raw_measured",
        "canonical_joint": joint,
        "valid": "1" if valid else "0",
        "x_m": str(point[0]) if valid else "",
        "y_m": str(point[1]) if valid else "",
        "z_m": str(point[2]) if valid else "",
        "counts_as_measured_valid": "1",
        "provenance": "measured_dynamic_arm_v8_raw",
        "depth_m": str(point[0]) if valid else "",
        "pixel_x": "450",
        "pixel_y": "276",
        "visibility": "0.99",
        "invalid_reason": "" if valid else "depth_unavailable",
        "warmup_excluded": "0",
    }


class CandidateTransformTests(unittest.TestCase):
    def test_a_is_passthrough_and_c_uses_registered_transform(self):
        rows = replay.build_candidates(
            pair_id="pair_01",
            view_a_rows={(0, "right_wrist"): raw_row()},
            view_c_rows={(0, "right_wrist"): raw_row()},
            rotation_a_from_c=np.asarray(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            translation_a_from_c=np.asarray([10.0, 20.0, 30.0]),
        )
        self.assertEqual(len(rows), 2)
        a, c = rows
        self.assertEqual(a["view_id"], "a")
        self.assertEqual(
            [a["common_x_m"], a["common_y_m"], a["common_z_m"]],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(c["view_id"], "c")
        self.assertEqual(
            [c["common_x_m"], c["common_y_m"], c["common_z_m"]],
            [8.0, 21.0, 33.0],
        )
        self.assertEqual(
            c["output_class"], "excluded_engineering_measured_c"
        )
        self.assertEqual(
            c["claim_eligibility"], "excluded_engineering_replay_only"
        )
        self.assertEqual(c["eligible_for_formal"], 0)
        self.assertEqual(c["same_capture_session"], 0)
        self.assertEqual(
            c["calibration_source"],
            "gt_correspondence_engineering_only",
        )
        self.assertEqual(
            c["source_provenance"], "measured_dynamic_arm_v8_raw"
        )

    def test_invalid_candidate_stays_invalid_without_coordinates(self):
        invalid = raw_row(valid=False)
        rows = replay.build_candidates(
            pair_id="pair_01",
            view_a_rows={(0, "right_wrist"): invalid},
            view_c_rows={(0, "right_wrist"): invalid},
            rotation_a_from_c=np.eye(3),
            translation_a_from_c=np.zeros(3),
        )
        for row in rows:
            self.assertEqual(row["valid"], 0)
            self.assertEqual(row["common_x_m"], "")
            self.assertEqual(row["invalid_reason"], "depth_unavailable")

    def test_key_mismatch_is_rejected(self):
        with self.assertRaises(replay.EngineeringReplayError):
            replay.build_candidates(
                pair_id="pair_01",
                view_a_rows={(0, "right_wrist"): raw_row()},
                view_c_rows={(0, "right_elbow"): raw_row(
                    joint="right_elbow"
                )},
                rotation_a_from_c=np.eye(3),
                translation_a_from_c=np.zeros(3),
            )

    def test_time_metadata_mismatch_is_rejected(self):
        c = raw_row()
        c["usd_time_code"] = "1.0"
        with self.assertRaises(replay.EngineeringReplayError):
            replay.build_candidates(
                pair_id="pair_01",
                view_a_rows={(0, "right_wrist"): raw_row()},
                view_c_rows={(0, "right_wrist"): c},
                rotation_a_from_c=np.eye(3),
                translation_a_from_c=np.zeros(3),
            )


class CalibrationContractTests(unittest.TestCase):
    def make_calibration(
        self,
        root: Path,
        *,
        eligible_for_formal: bool = False,
    ) -> tuple[Path, Path]:
        sync = root / "sync_manifest.json"
        sync.write_text("{}\n", encoding="utf-8")
        sync_hash = hashlib.sha256(sync.read_bytes()).hexdigest()
        calibration = root / "calibration.json"
        calibration.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "claim_eligibility": (
                        "excluded_engineering_replay_only"
                    ),
                    "calibration_source": (
                        "gt_correspondence_engineering_only"
                    ),
                    "eligible_for_formal": eligible_for_formal,
                    "uses_ground_truth": True,
                    "sync_manifest": {"sha256": sync_hash},
                    "relative_transforms": {
                        "a_from_c": {
                            "rotation_3x3": np.eye(3).tolist(),
                            "translation_m": [1.0, 2.0, 3.0],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return calibration, sync

    def test_valid_engineering_calibration_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration, sync = self.make_calibration(Path(directory))
            payload, rotation, translation = replay.validate_calibration(
                calibration, sync
            )
            self.assertTrue(payload["uses_ground_truth"])
            np.testing.assert_allclose(rotation, np.eye(3))
            np.testing.assert_allclose(translation, [1.0, 2.0, 3.0])

    def test_formal_eligible_gt_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration, sync = self.make_calibration(
                Path(directory), eligible_for_formal=True
            )
            with self.assertRaises(replay.EngineeringReplayError):
                replay.validate_calibration(calibration, sync)

    def test_sync_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, sync = self.make_calibration(root)
            sync.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaises(replay.EngineeringReplayError):
                replay.validate_calibration(calibration, sync)


if __name__ == "__main__":
    unittest.main()
