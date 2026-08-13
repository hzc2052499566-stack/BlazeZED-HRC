"""Run the frozen sustained K4 root-loss sensitivity experiment."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_k4_root_fault_injection_v1 as base


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "k4_sustained_occlusion_sensitivity_v1"
    / "sustained_occlusion_protocol.json"
)
ESTIMATES_NAME = "sustained_occlusion_estimates.csv"
RUN_SUMMARY_NAME = "sustained_occlusion_run_summary.csv"
SUMMARY_NAME = "sustained_occlusion_summary.json"
STATE_NAME = "sustained_occlusion_analysis_state.json"
ALGORITHM_VERSION = "k4_sustained_occlusion_sensitivity_v1_20260729"
RECOVERY_THRESHOLD_M = 0.001
RECOVERY_STABLE_FRAMES = 3
RECOVERY_SEARCH_FRAMES = 30
RUN_FIELDS = (
    "run_id",
    "scenario",
    "method",
    "duration_frames",
    "active_sample_count",
    "active_joint_sample_coverage",
    "active_elbow_coverage",
    "active_wrist_coverage",
    "active_clean_displacement_mean_mm",
    "active_clean_displacement_p95_mm",
    "active_clean_displacement_max_mm",
    "active_gt_error_mean_mm",
    "active_root_prediction_frame_count",
    "maximum_consecutive_invalid_frames",
    "recovery_frames_mean",
    "recovery_frames_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def sample_sd(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def percentile(values: list[float], q: float) -> float | None:
    return base.percentile(values, q)


def max_invalid_streak(rows: list[dict]) -> int:
    longest = 0
    current = 0
    for row in rows:
        if base.row_point(row) is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def recovery_after_interval(
    by_key: dict,
    clean_by_key: dict,
    run_id: str,
    scenario: str,
    method: str,
    end_frame: int,
) -> int | None:
    for offset in range(1, RECOVERY_SEARCH_FRAMES + 1):
        first = end_frame + offset
        last = first + RECOVERY_STABLE_FRAMES - 1
        if last >= 240:
            break
        stable = True
        for sequence in range(first, last + 1):
            for joint in base.JOINTS:
                altered = base.row_point(
                    by_key[(run_id, scenario, method, sequence, joint)]
                )
                clean = base.row_point(
                    clean_by_key[(run_id, method, sequence, joint)]
                )
                if (
                    altered is None
                    or clean is None
                    or float(np.linalg.norm(altered - clean))
                    > RECOVERY_THRESHOLD_M
                ):
                    stable = False
                    break
            if not stable:
                break
        if stable:
            return offset
    return None


def analyse(protocol: dict, estimates: list[dict]) -> tuple[list[dict], dict]:
    # This call is deliberately after the estimates artifact has been written.
    ground_truth = base.load_ground_truth(protocol)
    by_key = {}
    clean_by_key = {}
    for row in estimates:
        key = (
            row["run_id"],
            row["scenario"],
            row["method"],
            int(row["sequence_index"]),
            row["canonical_joint"],
        )
        if key in by_key:
            raise base.ContractError(f"Duplicate estimate key: {key}")
        by_key[key] = row
        if row["scenario"] == "clean_control":
            clean_by_key[
                (
                    row["run_id"],
                    row["method"],
                    int(row["sequence_index"]),
                    row["canonical_joint"],
                )
            ] = row

    run_rows = []
    for run_id in protocol["run_ids"]:
        for scenario in protocol["scenarios"]:
            name = scenario["name"]
            active_frames = {int(value) for value in scenario["fault_frames"]}
            for method in base.METHODS:
                active_rows = [
                    by_key[(run_id, name, method, sequence, joint)]
                    for sequence in sorted(active_frames)
                    for joint in base.JOINTS
                ]
                valid_active = [
                    row for row in active_rows if base.row_point(row) is not None
                ]
                displacements = []
                gt_errors = []
                for row in valid_active:
                    sequence = int(row["sequence_index"])
                    joint = row["canonical_joint"]
                    point = base.row_point(row)
                    clean = base.row_point(
                        clean_by_key[(run_id, method, sequence, joint)]
                    )
                    if clean is not None:
                        displacements.append(
                            float(np.linalg.norm(point - clean)) * 1000.0
                        )
                    gt_errors.append(
                        float(
                            np.linalg.norm(
                                point
                                - ground_truth[(run_id, sequence, joint)]
                            )
                        )
                        * 1000.0
                    )
                recoveries = [
                    recovery_after_interval(
                        by_key,
                        clean_by_key,
                        run_id,
                        name,
                        method,
                        int(block["end_frame_inclusive"]),
                    )
                    for block in scenario["fault_intervals"]
                ]
                observed_recovery = [
                    value for value in recoveries if value is not None
                ]
                per_joint_coverage = {}
                per_joint_streak = {}
                for joint in base.JOINTS:
                    rows = [
                        by_key[(run_id, name, method, sequence, joint)]
                        for sequence in sorted(active_frames)
                    ]
                    per_joint_coverage[joint] = (
                        sum(base.row_point(row) is not None for row in rows)
                        / len(rows)
                        if rows
                        else 1.0
                    )
                    per_joint_streak[joint] = max_invalid_streak(rows)
                root_prediction_count = (
                    len(
                        {
                            int(row["sequence_index"])
                            for row in active_rows
                            if row["shoulder_root_reason"]
                            == "missing_root_predicted_from_pelvis"
                        }
                    )
                    if method == "k4_inferred"
                    else 0
                )
                run_rows.append(
                    {
                        "run_id": run_id,
                        "scenario": name,
                        "method": method,
                        "duration_frames": int(
                            scenario["duration_frames"]
                        ),
                        "active_sample_count": len(active_rows),
                        "active_joint_sample_coverage": (
                            len(valid_active) / len(active_rows)
                            if active_rows
                            else 1.0
                        ),
                        "active_elbow_coverage": per_joint_coverage[
                            "right_elbow"
                        ],
                        "active_wrist_coverage": per_joint_coverage[
                            "right_wrist"
                        ],
                        "active_clean_displacement_mean_mm": (
                            mean(displacements)
                        ),
                        "active_clean_displacement_p95_mm": (
                            percentile(displacements, 95.0)
                        ),
                        "active_clean_displacement_max_mm": (
                            max(displacements) if displacements else None
                        ),
                        "active_gt_error_mean_mm": mean(gt_errors),
                        "active_root_prediction_frame_count": (
                            root_prediction_count
                        ),
                        "maximum_consecutive_invalid_frames": max(
                            per_joint_streak.values()
                        ),
                        "recovery_frames_mean": mean(observed_recovery),
                        "recovery_frames_max": (
                            max(observed_recovery)
                            if observed_recovery
                            else None
                        ),
                    }
                )

    scenario_summary = {}
    for scenario in protocol["scenarios"]:
        name = scenario["name"]
        methods = {}
        for method in base.METHODS:
            rows = [
                row
                for row in run_rows
                if row["scenario"] == name and row["method"] == method
            ]
            metrics = {}
            for field in RUN_FIELDS:
                if field in {"run_id", "scenario", "method"}:
                    continue
                values = [
                    float(row[field])
                    for row in rows
                    if row[field] is not None
                ]
                metrics[field] = {
                    "run_values": values,
                    "mean": mean(values),
                    "sample_sd": sample_sd(values),
                }
            methods[method] = metrics
        scenario_summary[name] = {
            "fault_type": scenario["fault_type"],
            "duration_frames": scenario["duration_frames"],
            "fault_intervals": scenario["fault_intervals"],
            "methods": methods,
            "k4_minus_k3_active_coverage": (
                methods["k4_inferred"]["active_joint_sample_coverage"][
                    "mean"
                ]
                - methods["k3_inferred"]["active_joint_sample_coverage"][
                    "mean"
                ]
            ),
        }

    checks = {}
    limits = protocol["pre_registered_checks"]
    depth_names = [
        scenario["name"]
        for scenario in protocol["scenarios"]
        if scenario["fault_type"] == "missing_depth_impulse"
    ]
    checks["depth_loss_k4_coverage"] = all(
        scenario_summary[name]["methods"]["k4_inferred"][
            "active_joint_sample_coverage"
        ]["mean"]
        >= float(limits["depth_loss_k4_minimum_active_coverage"])
        for name in depth_names
    )
    checks["depth_loss_k4_displacement"] = all(
        (
            scenario_summary[name]["methods"]["k4_inferred"][
                "active_clean_displacement_max_mm"
            ]["mean"]
            is not None
            and scenario_summary[name]["methods"]["k4_inferred"][
                "active_clean_displacement_max_mm"
            ]["mean"]
            <= float(
                limits["depth_loss_k4_maximum_clean_displacement_mm"]
            )
        )
        for name in depth_names
    )
    checks["depth_loss_k4_recovery"] = all(
        (
            scenario_summary[name]["methods"]["k4_inferred"][
                "recovery_frames_max"
            ]["mean"]
            is not None
            and scenario_summary[name]["methods"]["k4_inferred"][
                "recovery_frames_max"
            ]["mean"]
            <= float(limits["depth_loss_k4_maximum_recovery_frames"])
        )
        for name in depth_names
    )
    landmark_name = next(
        scenario["name"]
        for scenario in protocol["scenarios"]
        if scenario["fault_type"] == "missing_landmark_impulse"
    )
    checks["landmark_loss_expected_boundary"] = all(
        np.isclose(
            scenario_summary[landmark_name]["methods"][method][
                "active_joint_sample_coverage"
            ]["mean"],
            float(
                limits[
                    f"landmark_loss_expected_{method[:2]}_active_coverage"
                ]
            ),
            rtol=0.0,
            atol=1e-12,
        )
        for method in base.METHODS
    )
    checks["all_passed"] = all(checks.values())
    return run_rows, {
        "scenarios": scenario_summary,
        "pre_registered_checks": checks,
    }


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = base.validate_protocol(protocol_path)
    if protocol.get("protocol_name") != (
        "k4_sustained_occlusion_sensitivity_v1"
    ):
        raise base.ContractError("Unexpected sustained-occlusion protocol.")
    output_root = protocol_path.parent
    estimates_path = output_root / ESTIMATES_NAME
    run_summary_path = output_root / RUN_SUMMARY_NAME
    summary_path = output_root / SUMMARY_NAME
    state_path = output_root / STATE_NAME
    collisions = [
        path
        for path in (
            estimates_path,
            run_summary_path,
            summary_path,
            state_path,
        )
        if path.exists()
    ]
    if collisions:
        raise base.ContractError(
            "Refusing to overwrite sustained-occlusion output(s): "
            + ", ".join(str(path) for path in collisions)
        )

    state = {
        "schema_version": 1,
        "status": "running",
        "algorithm_version": ALGORITHM_VERSION,
        "protocol": str(protocol_path),
        "protocol_file_sha256": base.sha256_file(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "gt_read_during_estimation": False,
        "started_wall_time_ns": time.time_ns(),
    }
    base.write_json_new(state_path, state)
    try:
        estimates, reproduction = base.run_estimation(protocol)
        base.write_csv(estimates_path, estimates, base.ESTIMATE_FIELDS)
        state.update(
            {
                "status": "estimation_complete",
                "estimate_row_count": len(estimates),
                "estimates_sha256": base.sha256_file(estimates_path),
                "clean_reproduction": reproduction,
                "estimation_completed_wall_time_ns": time.time_ns(),
            }
        )
        base.write_json_replace(state_path, state)
        run_rows, analysis = analyse(protocol, estimates)
        base.write_csv(run_summary_path, run_rows, RUN_FIELDS)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "purpose": "k4_sustained_occlusion_sensitivity_v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "run_ids": protocol["run_ids"],
            "scenario_count": len(protocol["scenarios"]),
            "estimate_row_count": len(estimates),
            "clean_reproduction": reproduction,
            **analysis,
        }
        base.write_json_new(summary_path, summary)
        state.update(
            {
                "status": "complete",
                "run_summary_sha256": base.sha256_file(run_summary_path),
                "summary_sha256": base.sha256_file(summary_path),
                "pre_registered_checks": analysis[
                    "pre_registered_checks"
                ],
                "completed_wall_time_ns": time.time_ns(),
            }
        )
        base.write_json_replace(state_path, state)
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_wall_time_ns": time.time_ns(),
            }
        )
        base.write_json_replace(state_path, state)
        raise

    print(f"Complete: {summary_path}")
    print(
        "Checks passed: "
        f"{summary['pre_registered_checks']['all_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
