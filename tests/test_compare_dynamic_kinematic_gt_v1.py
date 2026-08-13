from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


WORKSPACE = Path(__file__).resolve().parents[1]
TOOLS_DIR = WORKSPACE / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import compare_dynamic_kinematic_gt_v1 as comparator


GT_FIELDS = (
    "sequence_index",
    "cycle_index",
    "animation_frame_code",
    "usd_time_code",
    "canonical_joint",
    "gt_x_m",
    "gt_y_m",
    "gt_z_m",
    "gt_semantics",
)
ESTIMATE_FIELDS = (
    "run_id",
    "sequence_index",
    "cycle_index",
    "animation_frame_code",
    "usd_time_code",
    "method",
    "canonical_joint",
    "valid",
    "x_m",
    "y_m",
    "z_m",
    "counts_as_measured_valid",
    "provenance",
    "warmup_excluded",
)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gt_row(
    sequence: int,
    joint: str,
    point: tuple[float, float, float],
    *,
    semantics: str = comparator.GT_SEMANTICS,
) -> dict:
    return {
        "sequence_index": sequence,
        "cycle_index": 0,
        "animation_frame_code": sequence * 6,
        "usd_time_code": sequence * 6,
        "canonical_joint": joint,
        "gt_x_m": point[0],
        "gt_y_m": point[1],
        "gt_z_m": point[2],
        "gt_semantics": semantics,
    }


def estimate_row(
    sequence: int,
    method: str,
    joint: str,
    point: tuple[float, float, float] | None,
    *,
    warmup: int = 0,
    provenance: str | None = None,
) -> dict:
    measured = method not in {"k3_inferred", "k4_inferred"}
    return {
        "run_id": "dynamic_replay_test",
        "sequence_index": sequence,
        "cycle_index": 0,
        "animation_frame_code": sequence * 6,
        "usd_time_code": sequence * 6,
        "method": method,
        "canonical_joint": joint,
        "valid": int(point is not None),
        "x_m": "" if point is None else point[0],
        "y_m": "" if point is None else point[1],
        "z_m": "" if point is None else point[2],
        "counts_as_measured_valid": int(measured),
        "provenance": provenance or method,
        "warmup_excluded": warmup,
    }


class DynamicKinematicGTComparatorTests(unittest.TestCase):
    def paths(self, temporary: str) -> tuple[Path, Path]:
        root = Path(temporary)
        return root / "dynamic_gt.csv", root / "estimates.csv"

    def base_gt(self) -> list[dict]:
        rows = []
        for sequence in (0, 1):
            for joint, point in {
                "right_shoulder": (0.0, 0.0, 0.0),
                "right_elbow": (1.0, 0.0, 0.0),
                "right_wrist": (2.0, 0.0, 0.0),
            }.items():
                rows.append(gt_row(sequence, joint, point))
        return rows

    def test_measured_method_reports_exact_errors_coverage_and_bones(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = []
            frame_points = {
                0: {
                    "right_shoulder": (0.0, 0.0, 0.0),
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                },
                1: {
                    "right_shoulder": (0.01, 0.0, 0.0),
                    "right_elbow": (1.02, 0.0, 0.0),
                    "right_wrist": None,
                },
            }
            for sequence, points in frame_points.items():
                for joint, point in points.items():
                    estimates.append(
                        estimate_row(
                            sequence,
                            "raw_measured",
                            joint,
                            point,
                        )
                    )
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            report = comparator.compare(
                gt_path,
                estimate_path,
                "raw_measured",
            )

        self.assertEqual(
            report["reported_joints"],
            ["right_shoulder", "right_elbow", "right_wrist"],
        )
        self.assertEqual(report["coverage"]["valid_joint_sample_count"], 5)
        self.assertAlmostEqual(report["coverage"]["coverage_rate"], 5 / 6)
        self.assertAlmostEqual(
            report["joint_metrics"]["right_shoulder"][
                "mean_3d_error_mm"
            ],
            5.0,
        )
        self.assertAlmostEqual(
            report["joint_metrics"]["right_shoulder"][
                "rmse_3d_error_mm"
            ],
            10.0 / (2.0**0.5),
        )
        upper = report["bone_metrics"]["right_upper_arm"]
        self.assertEqual(upper["valid_frame_count"], 2)
        self.assertAlmostEqual(upper["mae_mm"], 5.0)
        forearm = report["bone_metrics"]["right_forearm"]
        self.assertEqual(forearm["valid_frame_count"], 1)
        self.assertAlmostEqual(forearm["mae_mm"], 0.0)
        primary = report["primary_metric"]
        self.assertEqual(
            primary["id"],
            "right_arm_two_joint_mean_position_error_mm",
        )
        self.assertEqual(primary["expected_sequence_count"], 2)
        self.assertEqual(primary["valid_sequence_count"], 1)
        self.assertAlmostEqual(primary["coverage_rate"], 0.5)
        self.assertAlmostEqual(primary["mean_3d_error_mm"], 0.0)
        self.assertEqual(len(primary["per_sequence"]), 2)
        self.assertTrue(primary["per_sequence"][0]["valid"])
        self.assertFalse(primary["per_sequence"][1]["valid"])
        self.assertIsNone(
            primary["per_sequence"][1][
                "two_joint_mean_position_error_mm"
            ]
        )

    def test_k4_reports_only_two_inferred_joints_and_forearm(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = []
            for sequence in (0, 1):
                for joint, point in {
                    "right_shoulder": (0.0, 0.0, 0.0),
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items():
                    estimates.append(
                        estimate_row(
                            sequence,
                            "raw_measured",
                            joint,
                            point,
                        )
                    )
                for joint, point in {
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items():
                    estimates.append(
                        estimate_row(
                            sequence,
                            "k4_inferred",
                            joint,
                            point,
                            provenance="inferred_ray_bone_k4",
                        )
                    )
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            report = comparator.compare(
                gt_path,
                estimate_path,
                "k4_inferred",
            )

        self.assertEqual(
            report["reported_joints"],
            ["right_elbow", "right_wrist"],
        )
        self.assertEqual(report["reported_bones"], ["right_forearm"])
        self.assertNotIn("right_shoulder", report["joint_metrics"])
        self.assertEqual(report["coverage"]["coverage_rate"], 1.0)

    def test_k3_is_an_inferred_two_joint_comparator_peer_of_k4(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = [
                estimate_row(
                    sequence,
                    "k3_inferred",
                    joint,
                    point,
                    provenance="inferred_ray_bone_k3",
                )
                for sequence in (0, 1)
                for joint, point in {
                    "right_elbow": (1.01, 0.0, 0.0),
                    "right_wrist": (2.03, 0.0, 0.0),
                }.items()
            ]
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            report = comparator.compare(
                gt_path,
                estimate_path,
                "k3_inferred",
            )

        self.assertEqual(
            report["reported_joints"],
            ["right_elbow", "right_wrist"],
        )
        self.assertEqual(report["reported_bones"], ["right_forearm"])
        self.assertEqual(report["primary_metric"]["valid_sequence_count"], 2)
        self.assertAlmostEqual(
            report["primary_metric"]["mean_3d_error_mm"],
            20.0,
        )
        for sample in report["primary_metric"]["per_sequence"]:
            self.assertAlmostEqual(
                sample["two_joint_mean_position_error_mm"],
                20.0,
            )

    def test_measured_method_selects_registered_arm_subset_from_full_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            gt_rows = self.base_gt()
            for sequence in (0, 1):
                gt_rows.append(
                    gt_row(sequence, "left_ankle", (0.0, 0.0, -1.0))
                )
            write_csv(gt_path, GT_FIELDS, gt_rows)
            estimates = []
            for sequence in (0, 1):
                for joint, point in {
                    "right_shoulder": (0.0, 0.0, 0.0),
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                    "left_ankle": (0.0, 0.0, -1.0),
                }.items():
                    estimates.append(
                        estimate_row(
                            sequence,
                            "raw_measured",
                            joint,
                            point,
                        )
                    )
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            report = comparator.compare(
                gt_path,
                estimate_path,
                "raw_measured",
            )

        self.assertEqual(
            report["ignored_nonreporting_estimate_joints"],
            ["left_ankle"],
        )
        self.assertEqual(
            report["coverage"]["expected_joint_sample_count"],
            6,
        )
        self.assertEqual(report["coverage"]["coverage_rate"], 1.0)
        for sample in report["primary_metric"]["per_sequence"]:
            self.assertAlmostEqual(
                sample["two_joint_mean_position_error_mm"],
                0.0,
            )

    def test_default_excludes_warmup_from_both_key_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = []
            for sequence in (0, 1):
                for joint, point in {
                    "right_shoulder": (0.0, 0.0, 0.0),
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items():
                    estimates.append(
                        estimate_row(
                            sequence,
                            "k2_guarded",
                            joint,
                            point,
                            warmup=int(sequence == 0),
                        )
                    )
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            report = comparator.compare(
                gt_path,
                estimate_path,
                "k2_guarded",
            )
            included = comparator.compare(
                gt_path,
                estimate_path,
                "k2_guarded",
                include_warmup=True,
            )

        self.assertEqual(
            report["warmup_filter"]["excluded_sequence_indices"],
            [0],
        )
        self.assertEqual(
            report["warmup_filter"]["excluded_estimate_key_count"],
            3,
        )
        self.assertEqual(
            report["warmup_filter"]["excluded_ground_truth_key_count"],
            3,
        )
        self.assertEqual(
            report["warmup_filter"]["excluded_joint_pair_count"],
            3,
        )
        self.assertEqual(report["coverage"]["expected_joint_sample_count"], 3)
        self.assertEqual(included["coverage"]["expected_joint_sample_count"], 6)

    def test_refuses_time_mean_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            gt_rows = self.base_gt()
            gt_rows[0]["gt_semantics"] = "time_mean"
            write_csv(gt_path, GT_FIELDS, gt_rows)
            estimates = [
                estimate_row(
                    sequence,
                    "k4_inferred",
                    joint,
                    point,
                    provenance="inferred",
                )
                for sequence in (0, 1)
                for joint, point in {
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items()
            ]
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            with self.assertRaisesRegex(
                comparator.ComparisonContractError,
                "time-mean GT",
            ):
                comparator.compare(
                    gt_path,
                    estimate_path,
                    "k4_inferred",
                )

    def test_refuses_duplicate_ground_truth_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            gt_rows = self.base_gt()
            gt_rows.append(dict(gt_rows[0]))
            write_csv(gt_path, GT_FIELDS, gt_rows)
            estimates = [
                estimate_row(
                    sequence,
                    "k4_inferred",
                    joint,
                    point,
                    provenance="inferred",
                )
                for sequence in (0, 1)
                for joint, point in {
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items()
            ]
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            with self.assertRaisesRegex(
                comparator.ComparisonContractError,
                "Duplicate GT key",
            ):
                comparator.compare(
                    gt_path,
                    estimate_path,
                    "k4_inferred",
                )

    def test_refuses_duplicate_estimate_key_and_key_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = [
                estimate_row(
                    sequence,
                    "k4_inferred",
                    joint,
                    point,
                    provenance="inferred",
                )
                for sequence in (0, 1)
                for joint, point in {
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items()
            ]
            estimates.append(dict(estimates[0]))
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            with self.assertRaisesRegex(
                comparator.ComparisonContractError,
                "Duplicate estimate key",
            ):
                comparator.compare(
                    gt_path,
                    estimate_path,
                    "k4_inferred",
                )

            estimates = estimates[:-2]
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            with self.assertRaisesRegex(
                comparator.ComparisonContractError,
                "rectangular/exact",
            ):
                comparator.compare(
                    gt_path,
                    estimate_path,
                    "k4_inferred",
                )

    def test_refuses_an_entire_missing_estimate_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = [
                estimate_row(
                    0,
                    "k4_inferred",
                    joint,
                    point,
                    provenance="inferred",
                )
                for joint, point in {
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items()
            ]
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            with self.assertRaisesRegex(
                comparator.ComparisonContractError,
                "exact sequence-key mismatch",
            ):
                comparator.compare(
                    gt_path,
                    estimate_path,
                    "k4_inferred",
                )

    def test_refuses_frame_metadata_mismatch_and_wrong_k4_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = [
                estimate_row(
                    sequence,
                    "k4_inferred",
                    joint,
                    point,
                    provenance="inferred",
                )
                for sequence in (0, 1)
                for joint, point in {
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items()
            ]
            estimates[0]["counts_as_measured_valid"] = 1
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            with self.assertRaisesRegex(
                comparator.ComparisonContractError,
                "counts_as_measured_valid",
            ):
                comparator.compare(
                    gt_path,
                    estimate_path,
                    "k4_inferred",
                )

            estimates[0]["counts_as_measured_valid"] = 0
            estimates[0]["usd_time_code"] = 99
            estimates[1]["usd_time_code"] = 99
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            with self.assertRaisesRegex(
                comparator.ComparisonContractError,
                "frame-key metadata mismatch",
            ):
                comparator.compare(
                    gt_path,
                    estimate_path,
                    "k4_inferred",
                )

    def test_cli_writes_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            output = Path(temporary) / "comparison.json"
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = [
                estimate_row(
                    sequence,
                    "k4_inferred",
                    joint,
                    point,
                    provenance="inferred",
                )
                for sequence in (0, 1)
                for joint, point in {
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items()
            ]
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            argv = [
                "compare_dynamic_kinematic_gt_v1.py",
                "--ground-truth-csv",
                str(gt_path),
                "--estimate-csv",
                str(estimate_path),
                "--method",
                "k4_inferred",
                "--output-json",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                comparator.os,
                "replace",
                wraps=comparator.os.replace,
            ) as replace:
                return_code = comparator.main()
            report = json.loads(output.read_text(encoding="utf-8"))
            expected_ground_truth_sha256 = hashlib.sha256(
                gt_path.read_bytes()
            ).hexdigest()
            expected_estimate_sha256 = hashlib.sha256(
                estimate_path.read_bytes()
            ).hexdigest()
            temporary_output, committed_output = replace.call_args.args

        self.assertEqual(return_code, 0)
        self.assertEqual(report["method"], "k4_inferred")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            report["ground_truth_csv_sha256"],
            expected_ground_truth_sha256,
        )
        self.assertEqual(
            report["estimate_csv_sha256"],
            expected_estimate_sha256,
        )
        comparator_path = Path(comparator.__file__).resolve()
        self.assertEqual(report["comparator_path"], str(comparator_path))
        self.assertEqual(
            report["comparator_sha256"],
            hashlib.sha256(comparator_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            Path(temporary_output).parent.resolve(),
            output.parent.resolve(),
        )
        self.assertEqual(Path(committed_output), output.resolve())
        self.assertFalse(Path(temporary_output).exists())

    def test_cli_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            gt_path, estimate_path = self.paths(temporary)
            output = Path(temporary) / "comparison.json"
            write_csv(gt_path, GT_FIELDS, self.base_gt())
            estimates = [
                estimate_row(
                    sequence,
                    "k3_inferred",
                    joint,
                    point,
                    provenance="inferred",
                )
                for sequence in (0, 1)
                for joint, point in {
                    "right_elbow": (1.0, 0.0, 0.0),
                    "right_wrist": (2.0, 0.0, 0.0),
                }.items()
            ]
            write_csv(estimate_path, ESTIMATE_FIELDS, estimates)
            output.write_text("do not replace\n", encoding="utf-8")
            argv = [
                "compare_dynamic_kinematic_gt_v1.py",
                "--ground-truth-csv",
                str(gt_path),
                "--estimate-csv",
                str(estimate_path),
                "--method",
                "k3_inferred",
                "--output-json",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                comparator.os,
                "replace",
                wraps=comparator.os.replace,
            ) as replace:
                return_code = comparator.main()

            self.assertEqual(return_code, 2)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "do not replace\n",
            )
            replace.assert_not_called()
            self.assertFalse(
                any(
                    path.name.startswith(f".{output.name}.")
                    for path in Path(temporary).iterdir()
                )
            )


if __name__ == "__main__":
    unittest.main()
