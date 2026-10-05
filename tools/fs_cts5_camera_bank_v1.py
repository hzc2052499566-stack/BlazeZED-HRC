"""Pure-Python planning and projection gates for the FS-CTS5 camera bank.

This module deliberately has no Isaac Sim or ``pxr`` dependency.  The Isaac
preparer imports it, while unit tests exercise the camera placement and all
240 capture-step projection gates in a normal Python interpreter.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


WIDTH = 960
HEIGHT = 600
CAPTURE_FRAME_COUNT = 240
CAPTURE_FRAMES = tuple(range(CAPTURE_FRAME_COUNT))

# A compact engineering bank: the centre entry reuses the reference-camera
# eye, while the remaining entries change azimuth, elevation, or both.  The
# eventual five-view selection is deliberately left to the GT-free pilot.
AZIMUTH_OFFSETS_DEG = (-45.0, 0.0, 30.0)
ELEVATION_OFFSETS_DEG = (-12.0, 0.0, 12.0)
MAX_CAMERA_COUNT = 9

# Frozen scene geometry inherited from the pre-capture camera-placement bank.
# These constants were fixed independently of the FS-CTS5 v2 error outcomes;
# the 240-frame motion is used only to reject unusable projections.
FROZEN_AIM_WORLD_OPERATIONAL_M = (
    1.7154437200560442,
    -1.4743046160748159,
    1.4980182539216955,
)
FROZEN_REFERENCE_RADIUS_OPERATIONAL_M = 3.5
REFERENCE_RADIUS_TOLERANCE_M = 0.01

JOINTS = (
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
)
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

MIN_CAMERA_RADIUS = 2.0
MAX_CAMERA_RADIUS = 6.0
MIN_POSITIVE_DEPTH = 0.10
MIN_VIEWPORT_MARGIN_PX = 2.0
CENTRAL_ROI_FRACTION = 0.65
ROI_LEFT = (1.0 - CENTRAL_ROI_FRACTION) * WIDTH / 2.0
ROI_RIGHT = WIDTH - ROI_LEFT
ROI_TOP = (1.0 - CENTRAL_ROI_FRACTION) * HEIGHT / 2.0
ROI_BOTTOM = HEIGHT - ROI_TOP
MIN_ROI_MARGIN_PX = 10.0
MIN_PROJECTED_BONE_LENGTH_PX = 8.0


class CameraBankError(RuntimeError):
    """Raised when a candidate bank cannot satisfy its engineering gates."""


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 1.0e-12:
        raise CameraBankError("Cannot normalise a zero-length vector.")
    return tuple(float(value) / length for value in vector)


def camera_basis(eye: Sequence[float], aim: Sequence[float]):
    """Return USD camera right/up/forward axes for a world-up look-at pose."""
    forward = _unit(_sub(aim, eye))
    world_up = (0.0, 0.0, 1.0)
    right = (
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    )
    right = _unit(right)
    up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    return right, up, forward


def project_from_eye(
    eye: Sequence[float],
    aim: Sequence[float],
    point: Sequence[float],
    intrinsics: Mapping[str, float],
):
    """Project one world point, returning ``(u, v, positive_depth)`` or None."""
    right, up, forward = camera_basis(eye, aim)
    delta = _sub(point, eye)
    depth = _dot(delta, forward)
    if depth <= 1.0e-12:
        return None
    across = _dot(delta, right)
    vertical = _dot(delta, up)
    return (
        float(intrinsics["cx"]) + float(intrinsics["fx"]) * across / depth,
        float(intrinsics["cy"]) - float(intrinsics["fy"]) * vertical / depth,
        depth,
    )


def candidate_id(azimuth_offset_deg: float, elevation_offset_deg: float) -> str:
    """Stable, USD-path-safe id for a candidate pose."""
    def token(prefix: str, value: float) -> str:
        rounded = int(round(abs(float(value))))
        sign = "m" if value < 0 else "p"
        return f"{prefix}{sign}{rounded:03d}"

    return token("az", azimuth_offset_deg) + "_" + token("el", elevation_offset_deg)


def mapped_joint_bbox_centre(
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
) -> tuple[float, float, float]:
    """Centre of the complete 240-frame mapped-joint envelope."""
    _validate_position_contract(positions_by_frame)
    values = [
        positions_by_frame[frame][joint]
        for frame in CAPTURE_FRAMES
        for joint in JOINTS
    ]
    return tuple(
        (min(float(point[axis]) for point in values)
         + max(float(point[axis]) for point in values)) / 2.0
        for axis in range(3)
    )


def plan_candidate_bank(
    reference_eye: Sequence[float],
    target: Sequence[float] = FROZEN_AIM_WORLD_OPERATIONAL_M,
    *,
    bank_scope: str = "/World/FsCts5CandidateCameraBankV1",
) -> dict:
    """Place the 3x3 azimuth/elevation bank on the reference-eye sphere."""
    reference_vector = _sub(reference_eye, target)
    observed_reference_radius = _norm(reference_vector)
    if observed_reference_radius < MIN_CAMERA_RADIUS or observed_reference_radius > MAX_CAMERA_RADIUS:
        raise CameraBankError(
            f"Reference camera radius {observed_reference_radius:.3f} is outside "
            f"{MIN_CAMERA_RADIUS:.1f}..{MAX_CAMERA_RADIUS:.1f} operational metres."
        )
    if abs(observed_reference_radius - FROZEN_REFERENCE_RADIUS_OPERATIONAL_M) > (
        REFERENCE_RADIUS_TOLERANCE_M
    ):
        raise CameraBankError(
            "Reference eye disagrees with the frozen 3.50 m camera sphere: "
            f"observed {observed_reference_radius:.6f} m."
        )
    radius = FROZEN_REFERENCE_RADIUS_OPERATIONAL_M
    horizontal = math.hypot(reference_vector[0], reference_vector[1])
    if horizontal <= 1.0e-9:
        raise CameraBankError("Reference camera is vertically aligned with the target.")
    base_azimuth = math.atan2(reference_vector[1], reference_vector[0])
    base_elevation = math.atan2(reference_vector[2], horizontal)

    candidates = []
    for elevation_offset in ELEVATION_OFFSETS_DEG:
        elevation = base_elevation + math.radians(elevation_offset)
        if abs(math.degrees(elevation)) >= 75.0:
            raise CameraBankError("A candidate elevation is too close to world up/down.")
        horizontal_radius = radius * math.cos(elevation)
        for azimuth_offset in AZIMUTH_OFFSETS_DEG:
            azimuth = base_azimuth + math.radians(azimuth_offset)
            identifier = candidate_id(azimuth_offset, elevation_offset)
            eye = (
                float(target[0]) + horizontal_radius * math.cos(azimuth),
                float(target[1]) + horizontal_radius * math.sin(azimuth),
                float(target[2]) + radius * math.sin(elevation),
            )
            candidates.append({
                "candidate_id": identifier,
                "prim_path": f"{bank_scope}/Camera_{identifier}",
                "azimuth_offset_deg": float(azimuth_offset),
                "elevation_offset_deg": float(elevation_offset),
                "absolute_azimuth_deg": math.degrees(azimuth),
                "absolute_elevation_deg": math.degrees(elevation),
                "eye_world_operational_m": list(eye),
                "aim_world_operational_m": [float(value) for value in target],
            })

    if len(candidates) > MAX_CAMERA_COUNT:
        raise CameraBankError("Candidate count exceeds the nine-camera engineering cap.")
    if len({entry["candidate_id"] for entry in candidates}) != len(candidates):
        raise CameraBankError("Candidate ids are not unique.")
    return {
        "bank_scope": bank_scope,
        "camera_count": len(candidates),
        "reference_eye_world_operational_m": [float(value) for value in reference_eye],
        "target_world_operational_m": [float(value) for value in target],
        "placement_target_source": "frozen_scene_constant_not_motion_error_optimised",
        "reference_radius_operational_m": radius,
        "observed_reference_radius_operational_m": observed_reference_radius,
        "reference_absolute_azimuth_deg": math.degrees(base_azimuth),
        "reference_absolute_elevation_deg": math.degrees(base_elevation),
        "azimuth_offsets_deg": list(AZIMUTH_OFFSETS_DEG),
        "elevation_offsets_deg": list(ELEVATION_OFFSETS_DEG),
        "candidates": candidates,
    }


def _validate_position_contract(
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
) -> None:
    if set(positions_by_frame) != set(CAPTURE_FRAMES):
        raise CameraBankError("Projection input must contain exactly frames 0..239.")
    for frame in CAPTURE_FRAMES:
        missing = set(JOINTS) - set(positions_by_frame[frame])
        if missing:
            raise CameraBankError(
                f"Frame {frame} is missing mapped joints: {sorted(missing)}"
            )
        for joint in JOINTS:
            point = positions_by_frame[frame][joint]
            if len(point) != 3 or not all(math.isfinite(float(value)) for value in point):
                raise CameraBankError(f"Frame {frame} joint {joint} is not finite 3D.")


def validate_projection_preflight(
    plan: Mapping[str, object],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
    intrinsics: Mapping[str, float],
) -> dict:
    """Gate all 9 x 240 camera-frame projections before authoring an overlay."""
    _validate_position_contract(positions_by_frame)
    for key in ("fx", "fy", "cx", "cy", "width", "height"):
        if key not in intrinsics or not math.isfinite(float(intrinsics[key])):
            raise CameraBankError("Incomplete or non-finite camera intrinsics: " + key)
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    if width != WIDTH or height != HEIGHT:
        raise CameraBankError(f"Expected {WIDTH}x{HEIGHT}, got {width}x{height}.")
    if float(intrinsics["fx"]) <= 0.0 or float(intrinsics["fy"]) <= 0.0:
        raise CameraBankError("Focal lengths must be positive.")

    candidates = list(plan["candidates"])
    if not candidates or len(candidates) > MAX_CAMERA_COUNT:
        raise CameraBankError("Candidate bank must contain between one and nine cameras.")
    if len({float(item["azimuth_offset_deg"]) for item in candidates}) < 2:
        raise CameraBankError("Candidate bank lacks azimuth diversity.")
    if len({float(item["elevation_offset_deg"]) for item in candidates}) < 2:
        raise CameraBankError("Candidate bank lacks elevation diversity.")

    expected_joint_samples = CAPTURE_FRAME_COUNT * len(JOINTS)
    report = {
        "projection_geometry_pass": True,
        "capture_frame_start": 0,
        "capture_frame_end": CAPTURE_FRAME_COUNT - 1,
        "capture_frame_count": CAPTURE_FRAME_COUNT,
        "loop_endpoint_frame_240_excluded_from_capture_denominator": True,
        "joint_count": len(JOINTS),
        "bone_count": len(BONES),
        "camera_count": len(candidates),
        "motion_envelope_bbox_centre_world_operational_m": list(
            mapped_joint_bbox_centre(positions_by_frame)
        ),
        "motion_envelope_used_for_projection_gate_only_not_camera_placement": True,
        "expected_joint_samples_per_camera": expected_joint_samples,
        "gates": {
            "min_positive_depth_operational_m": MIN_POSITIVE_DEPTH,
            "min_viewport_margin_px": MIN_VIEWPORT_MARGIN_PX,
            "central_roi_fraction": CENTRAL_ROI_FRACTION,
            "central_roi_bounds_px": [ROI_LEFT, ROI_TOP, ROI_RIGHT, ROI_BOTTOM],
            "min_central_roi_margin_px": MIN_ROI_MARGIN_PX,
            "min_projected_bone_length_px": MIN_PROJECTED_BONE_LENGTH_PX,
        },
        "per_camera": {},
    }
    for camera in candidates:
        eye = camera["eye_world_operational_m"]
        aim = camera["aim_world_operational_m"]
        minimum_depth = float("inf")
        minimum_margin = float("inf")
        minimum_roi_margin = float("inf")
        minimum_bone_length = float("inf")
        projected_samples = 0
        for frame in CAPTURE_FRAMES:
            pixels = {}
            for joint in JOINTS:
                projected = project_from_eye(
                    eye, aim, positions_by_frame[frame][joint], intrinsics
                )
                if projected is None:
                    raise CameraBankError(
                        f"{camera['candidate_id']} frame {frame} joint {joint} is behind camera."
                    )
                u, v, depth = projected
                margin = min(u, (width - 1.0) - u, v, (height - 1.0) - v)
                roi_margin = min(
                    u - ROI_LEFT,
                    ROI_RIGHT - u,
                    v - ROI_TOP,
                    ROI_BOTTOM - v,
                )
                minimum_depth = min(minimum_depth, depth)
                minimum_margin = min(minimum_margin, margin)
                minimum_roi_margin = min(minimum_roi_margin, roi_margin)
                pixels[joint] = (u, v)
                projected_samples += 1
            for start, end in BONES.values():
                length = math.hypot(
                    pixels[start][0] - pixels[end][0],
                    pixels[start][1] - pixels[end][1],
                )
                minimum_bone_length = min(minimum_bone_length, length)

        if projected_samples != expected_joint_samples:
            raise CameraBankError("Projection sample denominator is incomplete.")
        if minimum_depth < MIN_POSITIVE_DEPTH:
            raise CameraBankError(
                f"{camera['candidate_id']} depth floor failed: {minimum_depth:.3f}."
            )
        if minimum_margin < MIN_VIEWPORT_MARGIN_PX:
            raise CameraBankError(
                f"{camera['candidate_id']} leaves the viewport: margin {minimum_margin:.2f}px."
            )
        if minimum_roi_margin < MIN_ROI_MARGIN_PX:
            raise CameraBankError(
                f"{camera['candidate_id']} leaves the frozen central ROI 0.65: "
                f"margin {minimum_roi_margin:.2f}px."
            )
        if minimum_bone_length < MIN_PROJECTED_BONE_LENGTH_PX:
            raise CameraBankError(
                f"{camera['candidate_id']} undersamples a bone: {minimum_bone_length:.2f}px."
            )
        report["per_camera"][camera["candidate_id"]] = {
            "projected_joint_samples": projected_samples,
            "all_mapped_joints_in_frame": True,
            "all_mapped_joints_inside_central_roi_0_65": True,
            "minimum_positive_depth_operational_m": minimum_depth,
            "minimum_viewport_margin_px": minimum_margin,
            "minimum_central_roi_margin_px": minimum_roi_margin,
            "minimum_projected_bone_length_px": minimum_bone_length,
        }
    return report
