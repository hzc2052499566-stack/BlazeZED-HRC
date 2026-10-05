"""Pure-Python planner for the FS-CTS5 camera-bank v2 elevation repair.

The v1 clean plausibility audit isolated the failure to the three ``+12``
degree candidates.  The six candidates at ``-12`` and ``0`` degrees are kept
bit-for-bit in pose space.  The replacement elevation is the pre-declared
binary midpoint between the passing ``0`` degree level and failing ``+12``
degree level: ``+6`` degrees.  No GT position or limb-length error is an input
to this planner.

This module deliberately has no Isaac Sim or ``pxr`` dependency.  The Isaac
preparer supplies the immutable v1 candidate plan and unit tests exercise all
240-frame projection gates in a normal Python interpreter.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import fs_cts5_camera_bank_v1 as v1


WIDTH = v1.WIDTH
HEIGHT = v1.HEIGHT
CAPTURE_FRAME_COUNT = v1.CAPTURE_FRAME_COUNT
CAPTURE_FRAMES = v1.CAPTURE_FRAMES
JOINTS = v1.JOINTS
BONES = v1.BONES
FROZEN_AIM_WORLD_OPERATIONAL_M = v1.FROZEN_AIM_WORLD_OPERATIONAL_M
FROZEN_REFERENCE_RADIUS_OPERATIONAL_M = v1.FROZEN_REFERENCE_RADIUS_OPERATIONAL_M
REFERENCE_RADIUS_TOLERANCE_M = v1.REFERENCE_RADIUS_TOLERANCE_M

AZIMUTH_OFFSETS_DEG = (-45.0, 0.0, 30.0)
ELEVATION_OFFSETS_DEG = (-12.0, 0.0, 6.0)
RETAINED_ELEVATION_OFFSETS_DEG = (-12.0, 0.0)
REJECTED_ELEVATION_OFFSET_DEG = 12.0
PASSING_UPPER_BOUND_ELEVATION_OFFSET_DEG = 0.0
REPAIR_ELEVATION_OFFSET_DEG = 6.0
MAX_CAMERA_COUNT = 9
DEFAULT_BANK_SCOPE = "/World/FsCts5CandidateCameraBankV2"

RETAINED_VIEW_IDS = tuple(
    v1.candidate_id(azimuth, elevation)
    for elevation in RETAINED_ELEVATION_OFFSETS_DEG
    for azimuth in AZIMUTH_OFFSETS_DEG
)
REPLACEMENT_VIEW_IDS = tuple(
    v1.candidate_id(azimuth, REPAIR_ELEVATION_OFFSET_DEG)
    for azimuth in AZIMUTH_OFFSETS_DEG
)
REJECTED_V1_VIEW_IDS = tuple(
    v1.candidate_id(azimuth, REJECTED_ELEVATION_OFFSET_DEG)
    for azimuth in AZIMUTH_OFFSETS_DEG
)
ANCHOR_VIEW_ID = "azp000_elp000"


class CameraBankV2Error(RuntimeError):
    """Raised when the v2 repair would drift from its objective contract."""


def _as_finite_triplet(value: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise CameraBankV2Error(f"{label} must contain exactly three values.")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise CameraBankV2Error(f"{label} contains NaN or Inf.")
    return result


def _candidate_map(plan: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    candidates = list(plan.get("candidates", []))
    result = {str(item.get("candidate_id", "")): item for item in candidates}
    if len(candidates) != 9 or len(result) != 9 or "" in result:
        raise CameraBankV2Error(f"{label} must contain nine unique candidate ids.")
    return result


def validate_v1_source_plan(v1_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable v1 factorial plan used for retained poses."""
    if int(v1_plan.get("camera_count", -1)) != 9:
        raise CameraBankV2Error("The v1 source plan does not contain nine cameras.")
    if tuple(float(value) for value in v1_plan.get("azimuth_offsets_deg", [])) != (
        -45.0, 0.0, 30.0
    ):
        raise CameraBankV2Error("The v1 source azimuth levels changed.")
    if tuple(float(value) for value in v1_plan.get("elevation_offsets_deg", [])) != (
        -12.0, 0.0, 12.0
    ):
        raise CameraBankV2Error("The v1 source elevation levels changed.")
    source = _candidate_map(v1_plan, "v1 source plan")
    expected_ids = set(RETAINED_VIEW_IDS) | set(REJECTED_V1_VIEW_IDS)
    if set(source) != expected_ids:
        raise CameraBankV2Error("The v1 source candidate-id set changed.")
    if ANCHOR_VIEW_ID not in source:
        raise CameraBankV2Error("The required centre anchor is absent from v1.")
    for candidate_id, candidate in source.items():
        _as_finite_triplet(
            candidate.get("eye_world_operational_m", []),
            f"v1 {candidate_id} eye",
        )
        _as_finite_triplet(
            candidate.get("aim_world_operational_m", []),
            f"v1 {candidate_id} aim",
        )
    return {
        "v1_source_plan_pass": True,
        "camera_count": 9,
        "retained_view_ids": list(RETAINED_VIEW_IDS),
        "rejected_view_ids": list(REJECTED_V1_VIEW_IDS),
        "anchor_view_id": ANCHOR_VIEW_ID,
    }


def _validate_reference_against_v1(
    reference_eye: Sequence[float],
    target: Sequence[float],
    v1_plan: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    eye = _as_finite_triplet(reference_eye, "reference eye")
    aim = _as_finite_triplet(target, "target")
    source_eye = _as_finite_triplet(
        v1_plan.get("reference_eye_world_operational_m", []),
        "v1 reference eye",
    )
    source_target = _as_finite_triplet(
        v1_plan.get("target_world_operational_m", []),
        "v1 target",
    )
    if eye != source_eye:
        raise CameraBankV2Error(
            "The active reference eye differs from the immutable v1 source plan."
        )
    if aim != source_target or aim != FROZEN_AIM_WORLD_OPERATIONAL_M:
        raise CameraBankV2Error(
            "The active/frozen aim differs from the immutable v1 source plan."
        )
    radius = math.dist(eye, aim)
    if abs(radius - FROZEN_REFERENCE_RADIUS_OPERATIONAL_M) > REFERENCE_RADIUS_TOLERANCE_M:
        raise CameraBankV2Error("The reference eye left the frozen 3.50 m sphere.")
    return eye, aim


def _replacement_candidate(
    *,
    reference_eye: Sequence[float],
    target: Sequence[float],
    azimuth_offset_deg: float,
    bank_scope: str,
) -> dict[str, Any]:
    vector = tuple(float(reference_eye[index]) - float(target[index]) for index in range(3))
    horizontal = math.hypot(vector[0], vector[1])
    if horizontal <= 1.0e-9:
        raise CameraBankV2Error("Reference camera is vertically aligned with the aim.")
    base_azimuth = math.atan2(vector[1], vector[0])
    base_elevation = math.atan2(vector[2], horizontal)
    azimuth = base_azimuth + math.radians(float(azimuth_offset_deg))
    elevation = base_elevation + math.radians(REPAIR_ELEVATION_OFFSET_DEG)
    radius = FROZEN_REFERENCE_RADIUS_OPERATIONAL_M
    horizontal_radius = radius * math.cos(elevation)
    identifier = v1.candidate_id(azimuth_offset_deg, REPAIR_ELEVATION_OFFSET_DEG)
    eye = (
        float(target[0]) + horizontal_radius * math.cos(azimuth),
        float(target[1]) + horizontal_radius * math.sin(azimuth),
        float(target[2]) + radius * math.sin(elevation),
    )
    return {
        "candidate_id": identifier,
        "prim_path": f"{bank_scope}/Camera_{identifier}",
        "azimuth_offset_deg": float(azimuth_offset_deg),
        "elevation_offset_deg": REPAIR_ELEVATION_OFFSET_DEG,
        "absolute_azimuth_deg": math.degrees(azimuth),
        "absolute_elevation_deg": math.degrees(elevation),
        "eye_world_operational_m": list(eye),
        "aim_world_operational_m": [float(value) for value in target],
        "v2_pose_role": "new_binary_midpoint_repair_pose",
    }


def plan_candidate_bank_v2(
    reference_eye: Sequence[float],
    v1_plan: Mapping[str, Any],
    target: Sequence[float] = FROZEN_AIM_WORLD_OPERATIONAL_M,
    *,
    bank_scope: str = DEFAULT_BANK_SCOPE,
) -> dict[str, Any]:
    """Build the repaired 3x3 bank while preserving six v1 poses exactly."""
    validate_v1_source_plan(v1_plan)
    eye, aim = _validate_reference_against_v1(reference_eye, target, v1_plan)
    if not bank_scope.startswith("/World/") or bank_scope.endswith("/"):
        raise CameraBankV2Error("The v2 bank scope must be one absolute /World path.")
    midpoint = (
        PASSING_UPPER_BOUND_ELEVATION_OFFSET_DEG
        + REJECTED_ELEVATION_OFFSET_DEG
    ) / 2.0
    if midpoint != REPAIR_ELEVATION_OFFSET_DEG:
        raise CameraBankV2Error("The objective binary-midpoint repair rule drifted.")

    source = _candidate_map(v1_plan, "v1 source plan")
    candidates: list[dict[str, Any]] = []
    for elevation in ELEVATION_OFFSETS_DEG:
        for azimuth in AZIMUTH_OFFSETS_DEG:
            identifier = v1.candidate_id(azimuth, elevation)
            if elevation in RETAINED_ELEVATION_OFFSETS_DEG:
                retained = dict(source[identifier])
                retained["prim_path"] = f"{bank_scope}/Camera_{identifier}"
                retained["v2_pose_role"] = "retained_v1_pose_exact"
                candidates.append(retained)
            else:
                candidates.append(_replacement_candidate(
                    reference_eye=eye,
                    target=aim,
                    azimuth_offset_deg=azimuth,
                    bank_scope=bank_scope,
                ))

    plan = {
        "bank_scope": bank_scope,
        "camera_count": len(candidates),
        "reference_eye_world_operational_m": list(eye),
        "target_world_operational_m": list(aim),
        "placement_target_source": "frozen_v1_scene_constant_not_GT_error_optimised",
        "reference_radius_operational_m": FROZEN_REFERENCE_RADIUS_OPERATIONAL_M,
        "observed_reference_radius_operational_m": math.dist(eye, aim),
        "reference_absolute_azimuth_deg": float(v1_plan["reference_absolute_azimuth_deg"]),
        "reference_absolute_elevation_deg": float(v1_plan["reference_absolute_elevation_deg"]),
        "azimuth_offsets_deg": list(AZIMUTH_OFFSETS_DEG),
        "elevation_offsets_deg": list(ELEVATION_OFFSETS_DEG),
        "repair_rule": {
            "method": "objective_binary_midpoint_of_passing_0_and_failing_plus_12",
            "passing_elevation_offset_deg": PASSING_UPPER_BOUND_ELEVATION_OFFSET_DEG,
            "failing_elevation_offset_deg": REJECTED_ELEVATION_OFFSET_DEG,
            "replacement_elevation_offset_deg": REPAIR_ELEVATION_OFFSET_DEG,
            "uses_GT_position_or_limb_length_error": False,
            "uses_post_repair_tracking_outcomes": False,
        },
        "retained_view_ids": list(RETAINED_VIEW_IDS),
        "replacement_view_ids": list(REPLACEMENT_VIEW_IDS),
        "rejected_v1_view_ids": list(REJECTED_V1_VIEW_IDS),
        "required_anchor_view_id": ANCHOR_VIEW_ID,
        "candidates": candidates,
    }
    if len(candidates) != MAX_CAMERA_COUNT or len(_candidate_map(plan, "v2 plan")) != 9:
        raise CameraBankV2Error("The repaired bank is not nine unique cameras.")
    validate_retained_pose_identity(plan, v1_plan)
    return plan


def validate_retained_pose_identity(
    v2_plan: Mapping[str, Any],
    v1_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact numeric identity for all six retained camera poses."""
    source = _candidate_map(v1_plan, "v1 source plan")
    repaired = _candidate_map(v2_plan, "v2 plan")
    pose_fields = (
        "azimuth_offset_deg",
        "elevation_offset_deg",
        "absolute_azimuth_deg",
        "absolute_elevation_deg",
        "eye_world_operational_m",
        "aim_world_operational_m",
    )
    mismatches: list[str] = []
    scalar_values_checked = 0
    for identifier in RETAINED_VIEW_IDS:
        for field in pose_fields:
            if repaired[identifier].get(field) != source[identifier].get(field):
                mismatches.append(f"{identifier}:{field}")
            value = source[identifier].get(field)
            scalar_values_checked += len(value) if isinstance(value, list) else 1
        if repaired[identifier].get("prim_path") == source[identifier].get("prim_path"):
            raise CameraBankV2Error(
                f"Retained view {identifier} was not moved to the independent v2 scope."
            )
    if mismatches:
        raise CameraBankV2Error(
            "A retained v2 pose differs from v1: " + ", ".join(mismatches)
        )
    if ANCHOR_VIEW_ID not in RETAINED_VIEW_IDS:
        raise CameraBankV2Error("The required anchor is not one of the retained views.")
    return {
        "retained_pose_identity_pass": True,
        "comparison_operator": "exact_JSON_numeric_value_equality_no_tolerance",
        "retained_view_count": len(RETAINED_VIEW_IDS),
        "retained_view_ids": list(RETAINED_VIEW_IDS),
        "pose_fields_compared": list(pose_fields),
        "scalar_values_checked": scalar_values_checked,
        "mismatches": [],
        "anchor_retained": True,
        "new_scope_is_distinct": True,
    }


def validate_projection_preflight(
    plan: Mapping[str, Any],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
    intrinsics: Mapping[str, float],
) -> dict[str, Any]:
    """Run the unchanged v1 240-frame ROI and eight-bone projection gates."""
    if tuple(float(value) for value in plan.get("elevation_offsets_deg", [])) != (
        -12.0, 0.0, 6.0
    ):
        raise CameraBankV2Error("Projection input is not the registered v2 elevation repair.")
    try:
        report = v1.validate_projection_preflight(plan, positions_by_frame, intrinsics)
    except v1.CameraBankError as error:
        raise CameraBankV2Error(str(error)) from error
    if set(report.get("per_camera", {})) != (
        set(RETAINED_VIEW_IDS) | set(REPLACEMENT_VIEW_IDS)
    ):
        raise CameraBankV2Error("Projection report does not cover the repaired nine-view bank.")
    report = dict(report)
    report["camera_bank_version"] = 2
    report["repair_elevation_offset_deg"] = REPAIR_ELEVATION_OFFSET_DEG
    report["retained_six_pose_identity_checked_separately"] = True
    report["uses_GT_error_for_camera_placement"] = False
    return report


# Re-export the pure geometry functions used by the Isaac preparer.
camera_basis = v1.camera_basis
project_from_eye = v1.project_from_eye
mapped_joint_bbox_centre = v1.mapped_joint_bbox_centre

