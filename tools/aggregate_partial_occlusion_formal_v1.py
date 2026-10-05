"""Aggregate the three partial-occlusion formal repeats.

Run-level mean +/- sample SD is the reporting unit; frames are never treated
as independent repeats.  This tool also computes the occluder-surface capture
diagnostic that the v2 Pilot could only derive by hand, and evaluates the
pre-registered gates recorded in the frozen protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    ROOT
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "dynamic_partial_occlusion_formal_v1"
)
PROTOCOL_PATH = EXPERIMENT_ROOT / "partial_occlusion_formal_protocol.json"
SUMMARY_JSON = EXPERIMENT_ROOT / "formal_summary.json"
SUMMARY_CSV = EXPERIMENT_ROOT / "formal_run_summary.csv"
REPORT_NAME = "partial_occlusion_formal_report.json"
REPEAT_COUNT = 3
METHODS = ("raw_measured", "k2_guarded", "k3_inferred", "k4_inferred")
PHASES = ("active", "inactive")
ENDPOINT_JOINTS = ("right_elbow", "right_wrist")
JOINTS = ("right_shoulder", "right_elbow", "right_wrist")

GATE_NAMES = (
    "all_repeats_captured_240_frames",
    "all_repeats_manipulation_checks_passed",
    "all_repeats_detection_rate_unity",
    "all_repeats_gt_not_read_during_estimation",
    "all_repeats_k3_k4_active_coverage_unity",
    "all_repeats_raw_active_coverage_below_inactive",
    "all_repeats_raw_active_error_above_inactive",
    "all_repeats_inferred_active_error_below_raw_active",
    "all_repeats_inferred_recovery_at_most_raw_recovery",
    "all_repeats_k4_identical_to_k3",
    "all_repeats_leaked_depth_captured_by_occluder",
)


class AggregationError(RuntimeError):
    """Raised when the formal aggregation contract is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(values: list[float]) -> dict:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "sample_sd": None}
    return {
        "n": len(clean),
        "mean": float(statistics.fmean(clean)),
        "sample_sd": (
            float(statistics.stdev(clean)) if len(clean) > 1 else 0.0
        ),
        "values": clean,
    }


def repeat_dir(index: int) -> Path:
    return EXPERIMENT_ROOT / "unit_corrected" / "rep_{:02d}".format(index)


def occluder_diagnostic(directory: Path, scene: dict, active: set) -> dict:
    """Measure how measured depth behaves inside the occlusion windows.

    A leaked sample is called occluder-captured when its estimated forward
    depth is closer to the registered occluder front plane than to its own
    ground-truth depth.  The rule uses only registered geometry, so it is not
    a threshold tuned after seeing the data.
    """
    plane = abs(float(scene["occluder_plane_z_camera_local"]))
    front_plane_m = plane - float(scene["half_depth"])
    gt = {
        (int(row["sequence_index"]), row["canonical_joint"]): row
        for row in read_csv(directory / "ground_truth_joints.csv")
    }
    estimates = read_csv(
        directory / "dynamic_k4_replay" / "dynamic_kinematic_estimates.csv"
    )
    total = 0
    leaked: list[dict[str, float]] = []
    for row in estimates:
        if row["method"] != "raw_measured":
            continue
        if row["canonical_joint"] not in ENDPOINT_JOINTS:
            continue
        sequence = int(row["sequence_index"])
        if sequence not in active:
            continue
        total += 1
        if str(row.get("valid", "")).strip() != "1":
            continue
        forward = float(row["x_m"])
        if not math.isfinite(forward):
            continue
        gt_row = gt[(sequence, row["canonical_joint"])]
        gt_forward = float(gt_row["gt_x_m"])
        leaked.append(
            {
                "forward_m": forward,
                "gt_forward_m": gt_forward,
                "error_mm": 1000.0
                * math.dist(
                    (
                        forward,
                        float(row["y_m"]),
                        float(row["z_m"]),
                    ),
                    (
                        gt_forward,
                        float(gt_row["gt_y_m"]),
                        float(gt_row["gt_z_m"]),
                    ),
                ),
            }
        )
    # Occluder-captured means the estimate sits closer to the registered
    # occluder front plane than to its own ground-truth depth. The rule uses
    # only registered geometry, so no threshold is tuned after the fact.
    captured = [
        item
        for item in leaked
        if abs(item["forward_m"] - front_plane_m)
        < abs(item["forward_m"] - item["gt_forward_m"])
    ]
    return {
        "occluder_front_plane_m": front_plane_m,
        "active_joint_samples": total,
        "gate_rejected_samples": total - len(leaked),
        "gate_rejection_rate": (
            (total - len(leaked)) / total if total else None
        ),
        "leaked_samples": len(leaked),
        "occluder_captured_samples": len(captured),
        "occluder_captured_rate_of_leaked": (
            len(captured) / len(leaked) if leaked else None
        ),
        "leaked_min_forward_depth_m": (
            min(item["forward_m"] for item in leaked) if leaked else None
        ),
        "leaked_mean_forward_depth_m": (
            statistics.fmean([item["forward_m"] for item in leaked])
            if leaked
            else None
        ),
        "leaked_mean_error_mm": (
            statistics.fmean([item["error_mm"] for item in leaked])
            if leaked
            else None
        ),
    }


def collect_repeat(index: int, scene: dict) -> dict:
    directory = repeat_dir(index)
    report_path = directory / REPORT_NAME
    if not report_path.is_file():
        raise AggregationError(
            "Missing formal repeat report: {}".format(report_path)
        )
    report = read_json(report_path)
    if report.get("status") != "complete":
        raise AggregationError("Repeat {} is not complete.".format(index))
    if int(report.get("repeat_index", -1)) != index:
        raise AggregationError(
            "Repeat {} report records a different repeat index.".format(index)
        )
    replay_state = read_json(
        directory
        / "dynamic_k4_replay"
        / "dynamic_kinematic_replay_state.json"
    )
    capture_state = read_json(
        EXPERIMENT_ROOT
        / "raw"
        / "rep_{:02d}".format(index)
        / "capture_state.json"
    )
    active = set()
    for block in report["occlusion_intervals"]:
        active.update(
            range(
                int(block["start_frame"]),
                int(block["end_frame_inclusive"]) + 1,
            )
        )
    return {
        "repeat_index": index,
        "report": report,
        "report_sha256": sha256_file(report_path),
        "capture_complete": (
            capture_state.get("status") == "complete"
            and int(capture_state.get("completed_sequence_count", -1)) == 240
            and int(capture_state.get("ground_truth_row_count", -1)) == 3600
        ),
        "gt_read_during_estimation": bool(
            replay_state.get("gt_read_during_estimation", True)
        ),
        "occluder_diagnostic": occluder_diagnostic(directory, scene, active),
    }


def evaluate_gates(repeats: list[dict]) -> dict:
    reports = [item["report"] for item in repeats]

    def metric(report, method, phase, key):
        return report["method_metrics"][method][phase][key]

    gates = {
        "all_repeats_captured_240_frames": all(
            item["capture_complete"] for item in repeats
        ),
        "all_repeats_manipulation_checks_passed": all(
            report["manipulation_check"]["checks"]["all_passed"]
            for report in reports
        ),
        "all_repeats_detection_rate_unity": all(
            report["body_detection_rate"][phase] == 1.0
            for report in reports
            for phase in PHASES
        ),
        "all_repeats_gt_not_read_during_estimation": all(
            not item["gt_read_during_estimation"] for item in repeats
        ),
        "all_repeats_k3_k4_active_coverage_unity": all(
            metric(report, method, "active", "joint_sample_coverage") == 1.0
            for report in reports
            for method in ("k3_inferred", "k4_inferred")
        ),
        "all_repeats_raw_active_coverage_below_inactive": all(
            metric(report, "raw_measured", "active", "joint_sample_coverage")
            < metric(
                report, "raw_measured", "inactive", "joint_sample_coverage"
            )
            for report in reports
        ),
        "all_repeats_raw_active_error_above_inactive": all(
            metric(
                report,
                "raw_measured",
                "active",
                "two_joint_position_error_mean_mm",
            )
            > metric(
                report,
                "raw_measured",
                "inactive",
                "two_joint_position_error_mean_mm",
            )
            for report in reports
        ),
        "all_repeats_inferred_active_error_below_raw_active": all(
            metric(
                report,
                "k3_inferred",
                "active",
                "two_joint_position_error_mean_mm",
            )
            < metric(
                report,
                "raw_measured",
                "active",
                "two_joint_position_error_mean_mm",
            )
            for report in reports
        ),
        "all_repeats_inferred_recovery_at_most_raw_recovery": all(
            all(
                inferred is not None
                and raw is not None
                and inferred <= raw
                for inferred, raw in zip(
                    report["method_metrics"]["k3_inferred"][
                        "recovery_frames_after_intervals"
                    ],
                    report["method_metrics"]["raw_measured"][
                        "recovery_frames_after_intervals"
                    ],
                )
            )
            for report in reports
        ),
        "all_repeats_k4_identical_to_k3": all(
            report["method_metrics"]["k4_inferred"]
            == report["method_metrics"]["k3_inferred"]
            for report in reports
        ),
        "all_repeats_leaked_depth_captured_by_occluder": all(
            item["occluder_diagnostic"]["occluder_captured_rate_of_leaked"]
            is not None
            and item["occluder_diagnostic"][
                "occluder_captured_rate_of_leaked"
            ]
            > 0.5
            for item in repeats
        ),
    }
    if set(gates) != set(GATE_NAMES):
        raise AggregationError("Gate name set drifted from the registry.")
    gates["all_passed"] = all(gates.values())
    return gates


def build_summary(repeats: list[dict], protocol: dict) -> dict:
    reports = [item["report"] for item in repeats]
    methods: dict[str, dict] = {}
    for method in METHODS:
        methods[method] = {}
        for phase in PHASES:
            methods[method][phase] = {
                key: summarise(
                    [
                        report["method_metrics"][method][phase][key]
                        for report in reports
                    ]
                )
                for key in (
                    "joint_sample_coverage",
                    "two_joint_position_error_mean_mm",
                    "two_joint_position_error_p95_mm",
                )
            }
        methods[method]["recovery_frames_after_intervals"] = [
            summarise(
                [
                    report["method_metrics"][method][
                        "recovery_frames_after_intervals"
                    ][index]
                    for report in reports
                ]
            )
            for index in range(
                len(reports[0]["method_metrics"][method][
                    "recovery_frames_after_intervals"
                ])
            )
        ]

    manipulation = {
        joint: {
            phase: summarise(
                [
                    report["manipulation_check"][joint][phase][
                        "mean_patch_green_fraction"
                    ]
                    for report in reports
                ]
            )
            for phase in PHASES
        }
        for joint in ENDPOINT_JOINTS
    }
    landmarks = {
        joint: {
            phase: {
                key: summarise(
                    [
                        report["landmark_metrics"][joint][phase][key]
                        for report in reports
                    ]
                )
                for key in ("eligible_rate", "in_image_rate", "visibility_mean")
            }
            for phase in PHASES
        }
        for joint in JOINTS
    }
    diagnostic = {
        key: summarise(
            [item["occluder_diagnostic"][key] for item in repeats]
        )
        for key in (
            "gate_rejection_rate",
            "occluder_captured_rate_of_leaked",
            "leaked_min_forward_depth_m",
            "leaked_mean_forward_depth_m",
            "leaked_mean_error_mm",
        )
    }
    diagnostic["occluder_front_plane_m"] = repeats[0][
        "occluder_diagnostic"
    ]["occluder_front_plane_m"]

    return {
        "schema_version": 1,
        "status": "complete",
        "purpose": "dynamic_partial_occlusion_formal_v1",
        "claim_eligibility": "formal_three_run",
        "protocol_sha256": protocol["protocol_sha256"],
        "repeat_count": len(repeats),
        "repeat_reports": {
            "rep_{:02d}".format(item["repeat_index"]): item["report_sha256"]
            for item in repeats
        },
        "statistical_unit": "independent_run",
        "dispersion": "sample_sd_over_runs_ddof_1",
        "manipulation_check_green_fraction": manipulation,
        "body_detection_rate": {
            phase: summarise(
                [report["body_detection_rate"][phase] for report in reports]
            )
            for phase in PHASES
        },
        "landmark_metrics": landmarks,
        "method_metrics": methods,
        "occluder_depth_diagnostic": diagnostic,
        "preregistered_gates": evaluate_gates(repeats),
        "interpretation_rules": [
            "Run-level mean +/- sample SD over three runs is the reporting "
            "unit; frames are not independent repeats.",
            "K3/K4 coverage is inferred coverage, not measured-depth "
            "validity.",
            "The endpoint is right elbow plus right wrist position error, "
            "not MPJPE.",
            "Raw active mean error is conditional on passing the depth "
            "validity gates and must be reported together with the gate "
            "rejection rate, never as a coverage-matched comparison.",
            "A rendered occluder is not a physical occlusion of a real "
            "human.",
        ],
    }


def write_csv(repeats: list[dict]) -> None:
    rows = []
    for item in repeats:
        report = item["report"]
        for method in METHODS:
            for phase in PHASES:
                block = report["method_metrics"][method][phase]
                rows.append(
                    {
                        "repeat_index": item["repeat_index"],
                        "method": method,
                        "phase": phase,
                        "joint_sample_coverage": block[
                            "joint_sample_coverage"
                        ],
                        "two_joint_position_error_mean_mm": block[
                            "two_joint_position_error_mean_mm"
                        ],
                        "two_joint_position_error_p95_mm": block[
                            "two_joint_position_error_p95_mm"
                        ],
                        "recovery_interval_1_frames": report[
                            "method_metrics"
                        ][method]["recovery_frames_after_intervals"][0],
                        "recovery_interval_2_frames": report[
                            "method_metrics"
                        ][method]["recovery_frames_after_intervals"][1],
                    }
                )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=REPEAT_COUNT)
    args = parser.parse_args()

    if not PROTOCOL_PATH.is_file():
        raise AggregationError(
            "Freeze the formal protocol first: {}".format(PROTOCOL_PATH)
        )
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("status") != "frozen":
        raise AggregationError("Formal protocol is not frozen.")
    if SUMMARY_JSON.exists():
        raise AggregationError(
            "Refusing to overwrite formal summary: {}".format(SUMMARY_JSON)
        )
    scene = read_json(Path(protocol["registered_scene"]["scene_manifest"]))

    repeats = [
        collect_repeat(index, scene)
        for index in range(1, args.repeats + 1)
    ]
    summary = build_summary(repeats, protocol)
    atomic_write_json(SUMMARY_JSON, summary)
    write_csv(repeats)

    gates = summary["preregistered_gates"]
    print("Formal summary: {}".format(SUMMARY_JSON))
    print("Run summary CSV: {}".format(SUMMARY_CSV))
    for name in GATE_NAMES:
        print("  {:58s} {}".format(name, gates[name]))
    print("  {:58s} {}".format("ALL PASSED", gates["all_passed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
