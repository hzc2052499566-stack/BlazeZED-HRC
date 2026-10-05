"""Timing and geometry gates for the human-executable FS-CTS5 motion v2."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


FPS = 60
END_FRAME = 240
FRAME_COUNT = END_FRAME + 1
MOTION_TAG = "fs_cts5_full_body_reach_alternating_low_march_v2"
CONTROL_KEYS = ("reach", "left_march", "right_march")


class MotionGeometryError(RuntimeError):
    """Raised when a baked motion violates a human-executable geometry gate."""


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _ramp(frame: int, start: int, end: int) -> float:
    if end <= start:
        raise ValueError("Ramp end must be after start.")
    return smoothstep((int(frame) - start) / float(end - start))


def _pulse(frame: int, rise_start: int, rise_end: int,
           hold_end: int, fall_end: int) -> float:
    if frame < rise_start:
        return 0.0
    if frame <= rise_end:
        return _ramp(frame, rise_start, rise_end)
    if frame < hold_end:
        return 1.0
    if frame <= fall_end:
        return 1.0 - _ramp(frame, hold_end, fall_end)
    return 0.0


def controls_for_frame(frame: int) -> dict[str, float]:
    """Return a balanced, non-overlapping low-march schedule for frame 0..240."""
    frame = int(frame)
    if frame < 0 or frame > END_FRAME:
        raise ValueError(f"Frame outside 0..{END_FRAME}: {frame}")

    if frame < 20:
        reach = 0.0
    elif frame <= 60:
        reach = _ramp(frame, 20, 60)
    elif frame < 210:
        reach = 1.0
    else:
        reach = 1.0 - _ramp(frame, 210, END_FRAME)

    left_march = _pulse(frame, 75, 105, 120, 150)
    right_march = _pulse(frame, 155, 185, 200, 230)
    if left_march > 0.0 and right_march > 0.0:
        raise RuntimeError("Left and right march controls must never overlap.")
    return {
        "reach": reach,
        "left_march": left_march,
        "right_march": right_march,
    }


def phase_for_frame(frame: int) -> str:
    controls = controls_for_frame(frame)
    if controls["left_march"] > 0.0:
        return "left_low_march"
    if controls["right_march"] > 0.0:
        return "right_low_march"
    if frame < 20:
        return "neutral_baseline"
    if controls["reach"] > 0.0:
        return "bilateral_reach_balance"
    return "neutral_return"


def schedule_summary() -> dict[str, object]:
    samples = [controls_for_frame(frame) for frame in range(FRAME_COUNT)]
    simultaneous = sum(
        sample["left_march"] > 0.0 and sample["right_march"] > 0.0
        for sample in samples
    )
    return {
        "fps": FPS,
        "end_frame": END_FRAME,
        "frame_count": FRAME_COUNT,
        "motion_tag": MOTION_TAG,
        "loop_closes": samples[0] == samples[-1],
        "max_reach": max(sample["reach"] for sample in samples),
        "max_left_march": max(sample["left_march"] for sample in samples),
        "max_right_march": max(sample["right_march"] for sample in samples),
        "simultaneous_leg_active_frames": simultaneous,
    }


def _vec_sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[i]) - float(right[i]) for i in range(3))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[i]) * float(right[i]) for i in range(3))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 1.0e-12:
        raise MotionGeometryError("Cannot normalise a zero-length geometry vector.")
    return tuple(float(value) / length for value in vector)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return _norm(_vec_sub(left, right))


def _angle_deg(first: Sequence[float], vertex: Sequence[float],
               third: Sequence[float]) -> float:
    a = _unit(_vec_sub(first, vertex))
    b = _unit(_vec_sub(third, vertex))
    cosine = max(-1.0, min(1.0, _dot(a, b)))
    return math.degrees(math.acos(cosine))


BONES = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_thigh": ("right_hip", "right_knee"),
    "right_shank": ("right_knee", "right_ankle"),
}


def validate_motion_geometry(
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
    controls_by_frame: Mapping[int, Mapping[str, float]],
) -> dict[str, float | int | bool]:
    """Validate support, clearance, separation and joint-angle geometry.

    Inputs are skeleton-space joint pivots for all 241 frames.  The gates are
    scale-normalised by measured limb lengths, so the same checks can be
    repeated after fresh-load packaging without assuming scene-unit metadata.
    """
    expected_frames = set(range(FRAME_COUNT))
    if set(positions_by_frame) != expected_frames:
        raise MotionGeometryError("Geometry must contain every frame 0..240.")
    if set(controls_by_frame) != expected_frames:
        raise MotionGeometryError("Controls must contain every frame 0..240.")
    required_joints = {joint for endpoints in BONES.values() for joint in endpoints}
    for frame, positions in positions_by_frame.items():
        missing = required_joints - set(positions)
        if missing:
            raise MotionGeometryError(f"Frame {frame} is missing joints: {sorted(missing)}")

    baseline = positions_by_frame[0]
    leg_lengths = {
        side: (
            _distance(baseline[f"{side}_hip"], baseline[f"{side}_knee"])
            + _distance(baseline[f"{side}_knee"], baseline[f"{side}_ankle"])
        )
        for side in ("left", "right")
    }
    mean_leg_length = sum(leg_lengths.values()) / 2.0
    if mean_leg_length <= 1.0e-9:
        raise MotionGeometryError("Baseline leg length is zero.")

    lateral_axis = _unit(_vec_sub(baseline["left_ankle"], baseline["right_ankle"]))
    baseline_ankle_separation = _dot(
        _vec_sub(baseline["left_ankle"], baseline["right_ankle"]), lateral_axis
    )
    baseline_knee_separation = _dot(
        _vec_sub(baseline["left_knee"], baseline["right_knee"]), lateral_axis
    )
    if baseline_ankle_separation <= 1.0e-9 or baseline_knee_separation <= 1.0e-9:
        raise MotionGeometryError("Baseline left/right ordering is degenerate.")

    min_ankle_ratio = float("inf")
    min_knee_ratio = float("inf")
    max_support_drift_ratio = 0.0
    min_knee_angle = 180.0
    peak_knee_angles = []
    peak_clearance_ratios = []
    min_ankle_below_knee_ratio = float("inf")
    max_bone_relative_change = 0.0
    simultaneous = 0

    baseline_lengths = {
        bone: _distance(baseline[start], baseline[end])
        for bone, (start, end) in BONES.items()
    }
    for frame in range(FRAME_COUNT):
        positions = positions_by_frame[frame]
        controls = controls_by_frame[frame]
        left_active = float(controls["left_march"]) > 1.0e-12
        right_active = float(controls["right_march"]) > 1.0e-12
        simultaneous += int(left_active and right_active)

        ankle_separation = _dot(
            _vec_sub(positions["left_ankle"], positions["right_ankle"]), lateral_axis
        )
        knee_separation = _dot(
            _vec_sub(positions["left_knee"], positions["right_knee"]), lateral_axis
        )
        min_ankle_ratio = min(min_ankle_ratio, ankle_separation / baseline_ankle_separation)
        min_knee_ratio = min(min_knee_ratio, knee_separation / baseline_knee_separation)

        for side in ("left", "right"):
            knee_angle = _angle_deg(
                positions[f"{side}_hip"],
                positions[f"{side}_knee"],
                positions[f"{side}_ankle"],
            )
            min_knee_angle = min(min_knee_angle, knee_angle)
            shank_length = baseline_lengths[f"{side}_shank"]
            below_ratio = (
                float(positions[f"{side}_knee"][2])
                - float(positions[f"{side}_ankle"][2])
            ) / shank_length
            min_ankle_below_knee_ratio = min(min_ankle_below_knee_ratio, below_ratio)

        if left_active:
            drift = _distance(positions["right_ankle"], baseline["right_ankle"])
            max_support_drift_ratio = max(max_support_drift_ratio, drift / leg_lengths["right"])
        if right_active:
            drift = _distance(positions["left_ankle"], baseline["left_ankle"])
            max_support_drift_ratio = max(max_support_drift_ratio, drift / leg_lengths["left"])

        for side, key in (("left", "left_march"), ("right", "right_march")):
            if float(controls[key]) >= 0.999:
                peak_knee_angles.append(_angle_deg(
                    positions[f"{side}_hip"],
                    positions[f"{side}_knee"],
                    positions[f"{side}_ankle"],
                ))
                clearance = (
                    float(positions[f"{side}_ankle"][2])
                    - float(baseline[f"{side}_ankle"][2])
                ) / leg_lengths[side]
                peak_clearance_ratios.append(clearance)

        for bone, (start, end) in BONES.items():
            observed = _distance(positions[start], positions[end])
            reference = baseline_lengths[bone]
            max_bone_relative_change = max(
                max_bone_relative_change, abs(observed - reference) / reference
            )

    if simultaneous != 0:
        raise MotionGeometryError("Left and right march overlap in baked geometry.")
    if min_knee_ratio < 0.60:
        raise MotionGeometryError(f"Knees cross or collapse inward: ratio={min_knee_ratio:.4f}")
    if min_ankle_ratio < 0.60:
        raise MotionGeometryError(f"Ankles cross or collapse inward: ratio={min_ankle_ratio:.4f}")
    if max_support_drift_ratio > 0.01:
        raise MotionGeometryError(
            f"Support ankle drift exceeds 1% leg length: {max_support_drift_ratio:.5f}"
        )
    if min_knee_angle < 120.0:
        raise MotionGeometryError(f"Knee flexion is too deep: {min_knee_angle:.2f} deg")
    if not peak_knee_angles or min(peak_knee_angles) < 120.0 or max(peak_knee_angles) > 175.0:
        raise MotionGeometryError("Peak low-march knee angle is outside 120..175 deg.")
    if not peak_clearance_ratios or min(peak_clearance_ratios) < 0.02:
        raise MotionGeometryError("Swing ankle clearance is below 2% leg length.")
    if max(peak_clearance_ratios) > 0.20:
        raise MotionGeometryError("Swing ankle clearance exceeds low-march range.")
    if min_ankle_below_knee_ratio < 0.45:
        raise MotionGeometryError("An ankle rises too close to or above its knee.")
    if max_bone_relative_change > 1.0e-6:
        raise MotionGeometryError(
            f"Rigid segment length changed: relative={max_bone_relative_change:.3e}"
        )

    return {
        "human_executable_geometry_pass": True,
        "simultaneous_leg_active_frames": simultaneous,
        "min_knee_separation_ratio": min_knee_ratio,
        "min_ankle_separation_ratio": min_ankle_ratio,
        "max_support_ankle_drift_leg_ratio": max_support_drift_ratio,
        "minimum_knee_angle_deg": min_knee_angle,
        "peak_knee_angle_min_deg": min(peak_knee_angles),
        "peak_knee_angle_max_deg": max(peak_knee_angles),
        "peak_swing_ankle_clearance_leg_ratio_min": min(peak_clearance_ratios),
        "peak_swing_ankle_clearance_leg_ratio_max": max(peak_clearance_ratios),
        "min_ankle_below_knee_shank_ratio": min_ankle_below_knee_ratio,
        "max_bone_length_relative_change": max_bone_relative_change,
    }
