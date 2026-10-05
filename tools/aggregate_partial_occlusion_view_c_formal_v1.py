"""Aggregate the three view C partial-occlusion formal repeats.

Run-level mean +/- sample SD over runs is the reporting unit.

Every pre-registered gate here is either a validity check or a **within-view
relational** claim. None encodes a threshold chosen after reading the pilot.
Cross-view absolute comparison against view A is deliberately absent: the two
datasets express ground truth in different camera frames, so only the
occluded-versus-clean contrast inside view C is sound.
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
    / "output/experiments/kinematic_constraints_ablation"
    / "dynamic_partial_occlusion_view_c_formal_v1"
)
PROTOCOL_PATH = EXPERIMENT_ROOT / "view_c_formal_protocol.json"
SUMMARY_JSON = EXPERIMENT_ROOT / "view_c_formal_summary.json"
SUMMARY_CSV = EXPERIMENT_ROOT / "view_c_formal_run_summary.csv"
REPORT_NAME = "partial_occlusion_view_c_report.json"
REPEAT_COUNT = 3
METHODS = ("raw_measured", "k2_guarded", "k3_inferred", "k4_inferred")
PHASES = ("active", "inactive")
ENDPOINT_JOINTS = ("right_elbow", "right_wrist")
TORSO_REFERENCE_JOINTS = ("right_shoulder", "pelvis")
COLLAPSE_MARGIN_M = 0.50

GATE_NAMES = (
    "all_repeats_captured_240_frames",
    "all_repeats_secondary_view_unoccluded",
    "all_repeats_secondary_view_resolvable",
    "all_repeats_detection_rate_unity",
    "all_repeats_gt_not_read_during_estimation",
    "all_repeats_measured_coverage_not_reduced_by_occlusion",
    "all_repeats_measured_error_not_inflated_by_occlusion",
    "all_repeats_measured_beats_inferred_under_occlusion",
    "all_repeats_k4_identical_to_k3",
    "all_repeats_measured_recovery_at_most_one_frame",
    "all_repeats_no_depth_collapse_under_occlusion",
)


class AggregationError(RuntimeError):
    """Raised when the view C aggregation contract is incomplete."""


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


def summarise(values: list) -> dict:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "sample_sd": None, "values": []}
    return {
        "n": len(clean),
        "mean": float(statistics.fmean(clean)),
        "sample_sd": float(statistics.stdev(clean)) if len(clean) > 1 else 0.0,
        "values": clean,
    }


def repeat_dir(index: int) -> Path:
    return EXPERIMENT_ROOT / "unit_corrected" / "rep_{:02d}".format(index)


def depth_collapse_count(directory: Path, active: set) -> dict:
    """How many endpoint samples show the registered depth-collapse signature.

    Uses the same 0.50 m rule registered by occlusion_reliability_flag_v1. In
    view A this fires on essentially every measured sample that survives the
    validity gates inside the occlusion windows. In view C it should not fire
    at all, because the occluder is not between the camera and the arm.
    """
    rows = read_csv(
        directory / "dynamic_k4_replay" / "dynamic_kinematic_estimates.csv"
    )
    depth = {}
    for row in rows:
        if row["method"] != "raw_measured":
            continue
        key = (int(row["sequence_index"]), row["canonical_joint"])
        valid = str(row.get("valid", "")).strip() == "1"
        depth[key] = float(row["x_m"]) if valid else None

    total = 0
    valid_count = 0
    collapsed = 0
    for frame in sorted(active):
        reference = [
            depth.get((frame, joint)) for joint in TORSO_REFERENCE_JOINTS
        ]
        reference = [v for v in reference if v is not None]
        torso = statistics.fmean(reference) if reference else None
        for joint in ENDPOINT_JOINTS:
            total += 1
            value = depth.get((frame, joint))
            if value is None or not math.isfinite(value):
                continue
            valid_count += 1
            if torso is not None and torso - value > COLLAPSE_MARGIN_M:
                collapsed += 1
    return {
        "active_joint_samples": total,
        "measured_valid_samples": valid_count,
        "depth_collapsed_samples": collapsed,
        "depth_collapse_rate_of_valid": (
            collapsed / valid_count if valid_count else None
        ),
    }


def collect_repeat(index: int) -> dict:
    directory = repeat_dir(index)
    report_path = directory / REPORT_NAME
    if not report_path.is_file():
        raise AggregationError("Missing repeat report: {}".format(report_path))
    report = read_json(report_path)
    if report.get("status") != "complete":
        raise AggregationError("Repeat {} is not complete.".format(index))
    replay_state = read_json(
        directory / "dynamic_k4_replay"
        / "dynamic_kinematic_replay_state.json"
    )
    capture_state = read_json(
        EXPERIMENT_ROOT / "raw" / "rep_{:02d}".format(index)
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
        "depth_collapse": depth_collapse_count(directory, active),
    }


def evaluate_gates(repeats: list[dict]) -> dict:
    reports = [item["report"] for item in repeats]

    def metric(report, method, phase, key):
        return report["method_metrics"][method][phase][key]

    gates = {
        "all_repeats_captured_240_frames": all(
            item["capture_complete"] for item in repeats
        ),
        "all_repeats_secondary_view_unoccluded": all(
            report["secondary_view_unoccluded_check"]["checks"]["all_passed"]
            for report in reports
        ),
        "all_repeats_secondary_view_resolvable": all(
            report["secondary_view_resolvable_check"]["checks"]["all_passed"]
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
        "all_repeats_measured_coverage_not_reduced_by_occlusion": all(
            metric(r, "raw_measured", "active", "joint_sample_coverage")
            >= metric(r, "raw_measured", "inactive", "joint_sample_coverage")
            for r in reports
        ),
        "all_repeats_measured_error_not_inflated_by_occlusion": all(
            metric(
                r, "raw_measured", "active",
                "two_joint_position_error_mean_mm",
            )
            <= metric(
                r, "raw_measured", "inactive",
                "two_joint_position_error_mean_mm",
            )
            for r in reports
        ),
        "all_repeats_measured_beats_inferred_under_occlusion": all(
            metric(
                r, "raw_measured", "active",
                "two_joint_position_error_mean_mm",
            )
            < metric(
                r, "k3_inferred", "active",
                "two_joint_position_error_mean_mm",
            )
            for r in reports
        ),
        "all_repeats_k4_identical_to_k3": all(
            r["method_metrics"]["k4_inferred"]
            == r["method_metrics"]["k3_inferred"]
            for r in reports
        ),
        "all_repeats_measured_recovery_at_most_one_frame": all(
            value is not None and value <= 1
            for r in reports
            for value in r["method_metrics"]["raw_measured"][
                "recovery_frames_after_intervals"
            ]
        ),
        "all_repeats_no_depth_collapse_under_occlusion": all(
            item["depth_collapse"]["depth_collapsed_samples"] == 0
            for item in repeats
        ),
    }
    if set(gates) != set(GATE_NAMES):
        raise AggregationError("Gate name set drifted from the registry.")
    gates["all_passed"] = all(gates.values())
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=REPEAT_COUNT)
    args = parser.parse_args()

    if not PROTOCOL_PATH.is_file():
        raise AggregationError(
            "Freeze the protocol first: {}".format(PROTOCOL_PATH)
        )
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("status") != "frozen":
        raise AggregationError("View C formal protocol is not frozen.")
    if SUMMARY_JSON.exists():
        raise AggregationError(
            "Refusing to overwrite: {}".format(SUMMARY_JSON)
        )

    repeats = [
        collect_repeat(i) for i in range(1, args.repeats + 1)
    ]
    reports = [item["report"] for item in repeats]

    methods = {}
    for method in METHODS:
        methods[method] = {}
        for phase in PHASES:
            methods[method][phase] = {
                key: summarise(
                    [
                        r["method_metrics"][method][phase][key]
                        for r in reports
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
                    r["method_metrics"][method][
                        "recovery_frames_after_intervals"
                    ][i]
                    for r in reports
                ]
            )
            for i in range(2)
        ]

    summary = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "dynamic_partial_occlusion_view_c_formal_v1",
        "claim_eligibility": "formal_three_run",
        "protocol_sha256": protocol["protocol_sha256"],
        "repeat_count": len(repeats),
        "repeat_reports": {
            "rep_{:02d}".format(i["repeat_index"]): i["report_sha256"]
            for i in repeats
        },
        "statistical_unit": "independent_run",
        "dispersion": "sample_sd_over_runs_ddof_1",
        "landmark_metrics": {
            joint: {
                phase: {
                    key: summarise(
                        [
                            r["landmark_metrics"][joint][phase][key]
                            for r in reports
                        ]
                    )
                    for key in ("eligible_rate", "visibility_mean")
                }
                for phase in PHASES
            }
            for joint in ("right_shoulder",) + ENDPOINT_JOINTS
        },
        "body_detection_rate": {
            phase: summarise(
                [r["body_detection_rate"][phase] for r in reports]
            )
            for phase in PHASES
        },
        "method_metrics": methods,
        "depth_collapse_diagnostic": {
            key: summarise(
                [item["depth_collapse"][key] for item in repeats]
            )
            for key in (
                "measured_valid_samples",
                "depth_collapsed_samples",
                "depth_collapse_rate_of_valid",
            )
        },
        "preregistered_gates": evaluate_gates(repeats),
        "interpretation_rules": [
            "Run-level mean +/- sample SD over three runs is the reporting "
            "unit; frames are not independent repeats.",
            "View C ground truth is in view C's own camera frame. Absolute "
            "errors are NOT comparable to view A; only the occluded versus "
            "non-occluded contrast inside view C is sound.",
            "The endpoint is right elbow plus right wrist position error, "
            "not MPJPE.",
            "K3/K4 branch selection is calibrated for view A geometry and is "
            "expected to transfer poorly here; their numbers are reported, "
            "not defended.",
            "A rendered occluder is not a physical occlusion of a real "
            "human.",
        ],
    }

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(SUMMARY_JSON), flags)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")

    rows = []
    for item in repeats:
        for method in METHODS:
            for phase in PHASES:
                block = item["report"]["method_metrics"][method][phase]
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
                    }
                )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gates = summary["preregistered_gates"]
    print("Summary: {}".format(SUMMARY_JSON))
    for name in GATE_NAMES:
        print("  {:56s} {}".format(name, gates[name]))
    print("  {:56s} {}".format("ALL PASSED", gates["all_passed"]))
    raw = methods["raw_measured"]
    print()
    print("raw measured, occluded window : {:.2f} +/- {:.2f} mm, "
          "coverage {:.3f}".format(
              raw["active"]["two_joint_position_error_mean_mm"]["mean"],
              raw["active"]["two_joint_position_error_mean_mm"]["sample_sd"],
              raw["active"]["joint_sample_coverage"]["mean"],
          ))
    print("raw measured, clean frames    : {:.2f} +/- {:.2f} mm, "
          "coverage {:.3f}".format(
              raw["inactive"]["two_joint_position_error_mean_mm"]["mean"],
              raw["inactive"]["two_joint_position_error_mean_mm"]["sample_sd"],
              raw["inactive"]["joint_sample_coverage"]["mean"],
          ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
