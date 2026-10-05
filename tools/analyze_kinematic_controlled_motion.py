"""Analyse raw versus online-K2g trajectories for controlled arm motion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


TRACKED_JOINTS = (
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def stats(values: list[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "median": statistics.median(finite),
        "p95": percentile(finite, 95.0),
        "p99": percentile(finite, 99.0),
        "max": max(finite),
    }


def load_rows(
    tracking_path: Path,
    warmup_frames: int,
) -> tuple[dict[int, dict[str, dict]], int]:
    frames: dict[int, dict[str, dict]] = {}
    steady_frame_indices = set()
    with tracking_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if not str(row.get("tracker", "")).startswith(
                "blazepose_zed_depth"
            ):
                continue
            frame_index = int(row["frame_index"])
            if frame_index < warmup_frames:
                continue
            steady_frame_indices.add(frame_index)
            frames.setdefault(frame_index, {})[
                row["canonical_joint"]
            ] = row
    return frames, len(steady_frame_indices)


def point_from_row(row: dict, mode: str) -> np.ndarray | None:
    valid_key = "raw_valid" if mode == "raw" else "valid"
    prefix = "raw_" if mode == "raw" else ""
    if int(row.get(valid_key, 0)) != 1:
        return None
    try:
        point = np.asarray(
            [
                float(row[prefix + "x_m"]),
                float(row[prefix + "y_m"]),
                float(row[prefix + "z_m"]),
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return point if np.all(np.isfinite(point)) else None


def trajectory_metrics(
    frames: dict[int, dict[str, dict]],
    steady_frame_count: int,
    joint: str,
    mode: str,
) -> dict:
    samples = []
    for frame_index, joints in sorted(frames.items()):
        row = joints.get(joint)
        point = point_from_row(row, mode) if row is not None else None
        if point is None:
            continue
        samples.append(
            (
                frame_index,
                int(row["zed_timestamp_ns"]),
                point,
            )
        )
    points = [sample[2] for sample in samples]
    if points:
        matrix = np.stack(points)
        ranges_m = np.ptp(matrix, axis=0)
        excursion_m = float(np.linalg.norm(ranges_m))
    else:
        ranges_m = np.asarray([0.0, 0.0, 0.0])
        excursion_m = None

    steps_mm = []
    speed_mps = []
    velocity_by_frame = {}
    for previous, current in zip(samples, samples[1:]):
        if current[0] - previous[0] != 1:
            continue
        dt_s = (current[1] - previous[1]) / 1_000_000_000.0
        if not 0.0 < dt_s <= 0.25:
            continue
        delta = current[2] - previous[2]
        step_m = float(np.linalg.norm(delta))
        steps_mm.append(step_m * 1000.0)
        velocity = delta / dt_s
        velocity_by_frame[current[0]] = (current[1], velocity)
        speed_mps.append(float(np.linalg.norm(velocity)))

    acceleration_mps2 = []
    acceleration_by_frame = {}
    velocity_items = sorted(velocity_by_frame.items())
    for previous, current in zip(velocity_items, velocity_items[1:]):
        if current[0] - previous[0] != 1:
            continue
        dt_s = (current[1][0] - previous[1][0]) / 1_000_000_000.0
        if not 0.0 < dt_s <= 0.25:
            continue
        acceleration = (current[1][1] - previous[1][1]) / dt_s
        acceleration_by_frame[current[0]] = (
            current[1][0],
            acceleration,
        )
        acceleration_mps2.append(float(np.linalg.norm(acceleration)))

    jerk_mps3 = []
    acceleration_items = sorted(acceleration_by_frame.items())
    for previous, current in zip(
        acceleration_items,
        acceleration_items[1:],
    ):
        if current[0] - previous[0] != 1:
            continue
        dt_s = (current[1][0] - previous[1][0]) / 1_000_000_000.0
        if not 0.0 < dt_s <= 0.25:
            continue
        jerk = (current[1][1] - previous[1][1]) / dt_s
        jerk_mps3.append(float(np.linalg.norm(jerk)))

    return {
        "sample_count": len(samples),
        "valid_rate": (
            len(samples) / steady_frame_count
            if steady_frame_count
            else None
        ),
        "axis_range_m": {
            "x_forward": float(ranges_m[0]),
            "y_left": float(ranges_m[1]),
            "z_up": float(ranges_m[2]),
        },
        "bounding_box_excursion_m": excursion_m,
        "frame_step_mm": stats(steps_mm),
        "speed_mps": stats(speed_mps),
        "acceleration_mps2": stats(acceleration_mps2),
        "jerk_mps3": stats(jerk_mps3),
    }


def count_low_pose_episodes(
    frames: dict[int, dict[str, dict]],
    joint: str,
    minimum_frames: int = 3,
) -> dict:
    samples = []
    for frame_index, joints in sorted(frames.items()):
        row = joints.get(joint)
        point = point_from_row(row, "raw") if row is not None else None
        if point is not None:
            samples.append((frame_index, float(point[2])))
    if not samples:
        return {
            "count": 0,
            "threshold_z_m": None,
            "minimum_episode_frames": minimum_frames,
        }
    z_values = [sample[1] for sample in samples]
    threshold = min(z_values) + 0.25 * (max(z_values) - min(z_values))
    episodes = []
    current = []
    previous_frame = None
    for frame_index, z_m in samples:
        low = z_m <= threshold
        contiguous = (
            previous_frame is not None
            and frame_index - previous_frame == 1
        )
        if low and (not current or contiguous):
            current.append(frame_index)
        elif low:
            if len(current) >= minimum_frames:
                episodes.append(current)
            current = [frame_index]
        else:
            if len(current) >= minimum_frames:
                episodes.append(current)
            current = []
        previous_frame = frame_index
    if len(current) >= minimum_frames:
        episodes.append(current)
    return {
        "count": len(episodes),
        "threshold_z_m": threshold,
        "minimum_episode_frames": minimum_frames,
        "episode_lengths_frames": [len(episode) for episode in episodes],
    }


def bone_consistency(
    frames: dict[int, dict[str, dict]],
    profile: dict,
) -> dict:
    output = {}
    pooled_raw_errors = []
    pooled_final_errors = []
    regressions = []
    validity_sample_mismatches = 0
    for bone in profile["bones"]:
        reference_m = float(bone["reference_length_m"])
        raw_lengths = []
        final_lengths = []
        for joints in frames.values():
            row_a = joints.get(bone["joint_a"])
            row_b = joints.get(bone["joint_b"])
            if row_a is None or row_b is None:
                continue
            raw_a = point_from_row(row_a, "raw")
            raw_b = point_from_row(row_b, "raw")
            final_a = point_from_row(row_a, "final")
            final_b = point_from_row(row_b, "final")
            if (raw_a is None) != (final_a is None):
                validity_sample_mismatches += 1
            if (raw_b is None) != (final_b is None):
                validity_sample_mismatches += 1
            if raw_a is not None and raw_b is not None:
                raw_lengths.append(float(np.linalg.norm(raw_b - raw_a)))
            if final_a is not None and final_b is not None:
                final_lengths.append(
                    float(np.linalg.norm(final_b - final_a))
                )
        raw_errors_mm = [
            abs(length - reference_m) * 1000.0
            for length in raw_lengths
        ]
        final_errors_mm = [
            abs(length - reference_m) * 1000.0
            for length in final_lengths
        ]
        raw_mae_mm = (
            statistics.fmean(raw_errors_mm) if raw_errors_mm else None
        )
        final_mae_mm = (
            statistics.fmean(final_errors_mm) if final_errors_mm else None
        )
        delta_mae_mm = (
            final_mae_mm - raw_mae_mm
            if raw_mae_mm is not None and final_mae_mm is not None
            else None
        )
        if delta_mae_mm is not None:
            regressions.append(delta_mae_mm)
        pooled_raw_errors.extend(raw_errors_mm)
        pooled_final_errors.extend(final_errors_mm)
        output[bone["name"]] = {
            "reference_length_m": reference_m,
            "raw_sample_count": len(raw_lengths),
            "final_sample_count": len(final_lengths),
            "raw_mae_mm": raw_mae_mm,
            "final_mae_mm": final_mae_mm,
            "delta_mae_mm": delta_mae_mm,
            "raw_length_sd_mm": (
                statistics.stdev(raw_lengths) * 1000.0
                if len(raw_lengths) > 1
                else None
            ),
            "final_length_sd_mm": (
                statistics.stdev(final_lengths) * 1000.0
                if len(final_lengths) > 1
                else None
            ),
        }
    raw_pooled_mae = (
        statistics.fmean(pooled_raw_errors)
        if pooled_raw_errors
        else None
    )
    final_pooled_mae = (
        statistics.fmean(pooled_final_errors)
        if pooled_final_errors
        else None
    )
    return {
        "bones": output,
        "pooled_sample_count": len(pooled_raw_errors),
        "raw_pooled_mae_mm": raw_pooled_mae,
        "final_pooled_mae_mm": final_pooled_mae,
        "delta_pooled_mae_mm": (
            final_pooled_mae - raw_pooled_mae
            if raw_pooled_mae is not None
            and final_pooled_mae is not None
            else None
        ),
        "maximum_per_bone_mae_regression_mm": (
            max(regressions) if regressions else None
        ),
        "raw_final_validity_sample_mismatch_count": (
            validity_sample_mismatches
        ),
    }


def analyze(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "live_performance_summary.json"
    tracking_path = run_dir / "tracking_joints.csv"
    if not summary_path.is_file() or not tracking_path.is_file():
        raise FileNotFoundError(
            "Missing live summary or tracking CSV in " + str(run_dir)
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    configuration = summary["configuration"]
    warmup_frames = int(configuration["warmup_frames"])
    profile_path = Path(configuration["kinematic_profile_path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    frames, steady_frame_count = load_rows(
        tracking_path,
        warmup_frames,
    )
    trajectories = {}
    for joint in TRACKED_JOINTS:
        trajectories[joint] = {
            mode: trajectory_metrics(
                frames,
                steady_frame_count,
                joint,
                mode,
            )
            for mode in ("raw", "final")
        }
    consistency = bone_consistency(frames, profile)
    low_pose_episodes = count_low_pose_episodes(
        frames,
        "right_wrist",
    )
    online = summary["online_kinematic_constraints"]
    algorithm_checks = {
        "raw_final_validity_preserved": (
            consistency[
                "raw_final_validity_sample_mismatch_count"
            ]
            == 0
            and online["all_frames"]["validity_preserved"] is True
        ),
        "zero_rejected_joints": (
            int(online["all_frames"]["rejected_joint_count"]) == 0
        ),
        "maximum_correction_at_most_40mm": (
            float(online["maximum_observed_joint_correction_mm"])
            <= 40.000001
        ),
        "kinematic_processing_p95_at_most_3ms": (
            float(online["steady_state"]["processing_ms"]["p95"])
            <= 3.0
        ),
        "pooled_profile_mae_not_increased": (
            consistency["delta_pooled_mae_mm"] is not None
            and consistency["delta_pooled_mae_mm"] <= 1e-9
        ),
        "maximum_per_bone_mae_regression_at_most_1mm": (
            consistency["maximum_per_bone_mae_regression_mm"]
            is not None
            and consistency[
                "maximum_per_bone_mae_regression_mm"
            ]
            <= 1.0
        ),
    }
    motion_checks = {
        "steady_frames_at_least_800": steady_frame_count >= 800,
        "right_wrist_excursion_at_least_0p25m": (
            trajectories["right_wrist"]["raw"][
                "bounding_box_excursion_m"
            ]
            is not None
            and trajectories["right_wrist"]["raw"][
                "bounding_box_excursion_m"
            ]
            >= 0.25
        ),
        "right_elbow_excursion_at_least_0p15m": (
            trajectories["right_elbow"]["raw"][
                "bounding_box_excursion_m"
            ]
            is not None
            and trajectories["right_elbow"]["raw"][
                "bounding_box_excursion_m"
            ]
            >= 0.15
        ),
        "at_least_four_low_pose_episodes": (
            low_pose_episodes["count"] >= 4
        ),
    }
    return {
        "schema_version": 1,
        "purpose": (
            "controlled_arm_motion_raw_vs_online_k2g_trajectory_analysis"
        ),
        "run_dir": str(run_dir),
        "warmup_frames": warmup_frames,
        "steady_frame_count": steady_frame_count,
        "algorithm_status": (
            "passed" if all(algorithm_checks.values()) else "failed"
        ),
        "motion_coverage_status": (
            "passed" if all(motion_checks.values()) else "failed"
        ),
        "algorithm_checks": algorithm_checks,
        "motion_coverage_checks": motion_checks,
        "low_pose_episodes": low_pose_episodes,
        "joint_trajectories": trajectories,
        "bone_profile_consistency": consistency,
        "online_kinematic_constraints": online,
        "interpretation_restrictions": [
            (
                "Raw and final trajectories share the same live observations, "
                "so their difference measures constraint behaviour."
            ),
            (
                "There is no synchronized Isaac skeleton ground truth; do not "
                "claim MPJPE, anatomical accuracy or sensor-to-motion lag."
            ),
            (
                "Latency metrics describe the host and consumer processing "
                "path, not photon-to-joint latency."
            ),
        ],
    }


def main() -> int:
    args = parse_args()
    report = analyze(args.run_dir)
    output_path = (
        args.output_json
        or args.run_dir.resolve() / "controlled_motion_analysis.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "algorithm_status": report["algorithm_status"],
            "motion_coverage_status": report["motion_coverage_status"],
            "steady_frame_count": report["steady_frame_count"],
            "output": str(output_path),
        },
        indent=2,
    ))
    return (
        0
        if report["algorithm_status"] == "passed"
        and report["motion_coverage_status"] == "passed"
        else 10
    )


if __name__ == "__main__":
    raise SystemExit(main())
