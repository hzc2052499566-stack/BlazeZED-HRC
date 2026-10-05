"""Robust motion-coverage analysis for densely baked controlled Pilot V3."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

import analyze_kinematic_controlled_motion as base


def contiguous_episode_count(
    samples: list[tuple[int, float]],
    predicate,
    minimum_frames: int,
) -> tuple[int, list[int]]:
    lengths = []
    current_length = 0
    previous_frame = None
    for frame_index, value in samples:
        contiguous = (
            previous_frame is not None
            and frame_index - previous_frame == 1
        )
        if predicate(value):
            if current_length and not contiguous:
                if current_length >= minimum_frames:
                    lengths.append(current_length)
                current_length = 0
            current_length += 1
        else:
            if current_length >= minimum_frames:
                lengths.append(current_length)
            current_length = 0
        previous_frame = frame_index
    if current_length >= minimum_frames:
        lengths.append(current_length)
    return len(lengths), lengths


def pixel_motion_report(
    tracking_path: Path,
    warmup_frames: int,
) -> dict:
    samples = []
    with tracking_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["frame_index"]) >= warmup_frames
                and row["canonical_joint"] == "right_wrist"
                and str(row["tracker"]).startswith(
                    "blazepose_zed_depth"
                )
            ):
                samples.append(
                    (
                        int(row["frame_index"]),
                        float(row["pixel_x"]),
                        float(row["pixel_y"]),
                    )
                )
    samples.sort()
    if len(samples) < 5:
        return {
            "sample_count": len(samples),
            "selected_axis": None,
            "robust_range_px": None,
            "low_episode_count": 0,
            "high_episode_count": 0,
        }
    matrix = np.asarray(
        [[sample[1], sample[2]] for sample in samples],
        dtype=np.float64,
    )
    # Five-frame rolling median suppresses isolated BlazePose pixel spikes.
    smoothed = np.stack(
        [
            np.median(
                matrix[
                    max(0, index - 2):
                    min(len(matrix), index + 3)
                ],
                axis=0,
            )
            for index in range(len(matrix))
        ]
    )
    percentiles = np.percentile(smoothed, [10.0, 90.0], axis=0)
    robust_ranges = percentiles[1] - percentiles[0]
    axis_index = int(np.argmax(robust_ranges))
    axis_name = "pixel_x" if axis_index == 0 else "pixel_y"
    low = float(percentiles[0, axis_index])
    high = float(percentiles[1, axis_index])
    midpoint = (low + high) / 2.0
    axis_samples = [
        (sample[0], float(smoothed[index, axis_index]))
        for index, sample in enumerate(samples)
    ]
    low_count, low_lengths = contiguous_episode_count(
        axis_samples,
        lambda value: value <= midpoint,
        minimum_frames=5,
    )
    high_count, high_lengths = contiguous_episode_count(
        axis_samples,
        lambda value: value > midpoint,
        minimum_frames=5,
    )
    return {
        "sample_count": len(samples),
        "selected_axis": axis_name,
        "robust_p10_px": low,
        "robust_p90_px": high,
        "robust_range_px": float(robust_ranges[axis_index]),
        "midpoint_px": midpoint,
        "low_episode_count": low_count,
        "high_episode_count": high_count,
        "low_episode_lengths_frames": low_lengths,
        "high_episode_lengths_frames": high_lengths,
        "complete_episode_pair_count": min(low_count, high_count),
    }


def robust_joint_excursion(
    frames: dict[int, dict[str, dict]],
    joint: str,
) -> dict:
    points = []
    for joints in frames.values():
        row = joints.get(joint)
        point = base.point_from_row(row, "raw") if row else None
        if point is not None:
            points.append(point)
    if not points:
        return {
            "sample_count": 0,
            "robust_axis_range_m": None,
            "robust_bounding_box_excursion_m": None,
        }
    matrix = np.stack(points)
    lower, upper = np.percentile(matrix, [5.0, 95.0], axis=0)
    ranges = upper - lower
    return {
        "sample_count": len(points),
        "robust_axis_range_m": {
            "x_forward": float(ranges[0]),
            "y_left": float(ranges[1]),
            "z_up": float(ranges[2]),
        },
        "robust_bounding_box_excursion_m": float(
            np.linalg.norm(ranges)
        ),
    }


def depth_jump_report(
    frames: dict[int, dict[str, dict]],
    joint: str,
) -> dict:
    samples = []
    for frame_index, joints in sorted(frames.items()):
        row = joints.get(joint)
        point = base.point_from_row(row, "raw") if row else None
        if point is not None:
            samples.append((frame_index, point))
    steps_mm = []
    for previous, current in zip(samples, samples[1:]):
        if current[0] - previous[0] != 1:
            continue
        steps_mm.append(
            abs(float(current[1][0] - previous[1][0])) * 1000.0
        )
    return {
        "consecutive_step_count": len(steps_mm),
        "forward_depth_step_mm": base.stats(steps_mm),
        "over_50mm_count": sum(step > 50.0 for step in steps_mm),
        "over_100mm_count": sum(step > 100.0 for step in steps_mm),
    }


def analyze(run_dir: Path) -> dict:
    report = base.analyze(run_dir)
    run_dir = run_dir.resolve()
    tracking_path = run_dir / "tracking_joints.csv"
    frames, steady_frame_count = base.load_rows(
        tracking_path,
        int(report["warmup_frames"]),
    )
    pixel_motion = pixel_motion_report(
        tracking_path,
        int(report["warmup_frames"]),
    )
    robust_excursions = {
        joint: robust_joint_excursion(frames, joint)
        for joint in ("right_elbow", "right_wrist")
    }
    depth_jumps = {
        joint: depth_jump_report(frames, joint)
        for joint in ("right_elbow", "right_wrist")
    }
    motion_checks = {
        "steady_frames_at_least_1000": steady_frame_count >= 1000,
        "right_wrist_2d_robust_range_at_least_20px": (
            pixel_motion.get("robust_range_px") is not None
            and pixel_motion["robust_range_px"] >= 20.0
        ),
        "at_least_four_complete_2d_motion_cycles": (
            pixel_motion.get("complete_episode_pair_count", 0) >= 4
        ),
        "right_wrist_robust_excursion_at_least_0p20m": (
            robust_excursions["right_wrist"][
                "robust_bounding_box_excursion_m"
            ]
            is not None
            and robust_excursions["right_wrist"][
                "robust_bounding_box_excursion_m"
            ]
            >= 0.20
        ),
        "right_elbow_robust_excursion_at_least_0p10m": (
            robust_excursions["right_elbow"][
                "robust_bounding_box_excursion_m"
            ]
            is not None
            and robust_excursions["right_elbow"][
                "robust_bounding_box_excursion_m"
            ]
            >= 0.10
        ),
        "right_elbow_zero_depth_steps_over_100mm": (
            depth_jumps["right_elbow"]["over_100mm_count"] == 0
        ),
        "right_wrist_zero_depth_steps_over_100mm": (
            depth_jumps["right_wrist"]["over_100mm_count"] == 0
        ),
    }
    report["purpose"] = (
        "controlled_arm_motion_v3_robust_raw_vs_online_k2g_analysis"
    )
    report["motion_coverage_checks"] = motion_checks
    report["motion_coverage_status"] = (
        "passed" if all(motion_checks.values()) else "failed"
    )
    report["pixel_motion_coverage"] = pixel_motion
    report["robust_joint_excursions"] = robust_excursions
    report["joint_depth_continuity"] = depth_jumps
    report["low_pose_episodes"] = {
        "deprecated_for_v3": True,
        "reason": (
            "V1/V2 3-D min/max threshold was corrupted by rare depth "
            "surface outliers; V3 uses robust 2-D cycle episodes."
        ),
    }
    return report
