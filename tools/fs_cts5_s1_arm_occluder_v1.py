"""Pure geometry planner for the FS-CTS5 S1 arm-occluder pilot.

This module is deliberately independent of Isaac Sim and ``pxr``.  It uses
clean USD-skeleton world coordinates only to author and validate projection
geometry; it never consumes a prediction, a limb-length error, or formal
data.  The future scientific five-view subset remains unselected.

The overlap denominator is intentionally explicit: one overlap *cell* is one
``view x bone x active frame`` combination, and its overlap is the number of
occluded samples divided by exactly 33 samples on that 3-D bone.  Counts of
clear or intended-occluded *views* are then computed from those cells, not
from the 33 ray samples.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from fs_cts5_camera_bank_v2 import (
    ANCHOR_VIEW_ID,
    AZIMUTH_OFFSETS_DEG,
    BONES,
    CAPTURE_FRAMES,
    ELEVATION_OFFSETS_DEG,
    JOINTS,
    camera_basis,
)
from fs_cts5_full_body_motion_v2 import (  # exact active-phase binding
    MOTION_TAG,
    controls_for_frame,
    phase_for_frame,
)


ACTIVE_RANGES = ((90, 119), (170, 199))
ACTIVE_FRAMES = tuple(
    frame
    for start, end in ACTIVE_RANGES
    for frame in range(start, end + 1)
)
TARGET_BONES = ("right_upper_arm", "right_forearm")
NEGATIVE_CONTROL_BONES = (
    "left_upper_arm",
    "left_forearm",
    "left_thigh",
    "left_shank",
    "right_thigh",
    "right_shank",
)
SAMPLED_BONES = TARGET_BONES + NEGATIVE_CONTROL_BONES
RIGHT_ARM_JOINTS = ("right_shoulder", "right_elbow", "right_wrist")

BONE_SAMPLE_COUNT = 33
MIN_FIRST_SURFACE_FORWARD_LEAD_M = 0.10
OCCLUDED_THRESHOLD = 0.80
CLEAR_THRESHOLD = 0.05
MIN_POOL_CLEAR_VIEW_COUNT = 3
MIN_POOL_INTENDED_OCCLUDED_VIEW_COUNT = 1
MAX_NEGATIVE_CONTROL_MEAN_OVERLAP = 0.05
MAX_NEGATIVE_CONTROL_CELL_FRACTION_ABOVE_CLEAR = 0.05

PLANE_OFFSETS_TOWARD_ANCHOR_M = (0.35, 0.45, 0.55)
HORIZONTAL_MARGINS_M = (0.04, 0.06, 0.08)
VERTICAL_MARGINS_M = (0.05, 0.07, 0.09)
PANEL_THICKNESSES_M = (0.05,)

EXPECTED_CAMERA_COUNT = 9
EXPECTED_ACTIVE_FRAME_COUNT = 60
EXPECTED_TARGET_CELL_COUNT = 9 * 2 * 60
EXPECTED_CONTROL_CELL_COUNT = 9 * 6 * 60
EXPECTED_ANCHOR_TARGET_CELL_COUNT = 2 * 60
EXPECTED_SUBSET_COUNT = math.comb(8, 4)
EXPECTED_VIEW_LAYOUT = (
    ("azm045_elm012", -45.0, -12.0),
    ("azp000_elm012", 0.0, -12.0),
    ("azp030_elm012", 30.0, -12.0),
    ("azm045_elp000", -45.0, 0.0),
    ("azp000_elp000", 0.0, 0.0),
    ("azp030_elp000", 30.0, 0.0),
    ("azm045_elp006", -45.0, 6.0),
    ("azp000_elp006", 0.0, 6.0),
    ("azp030_elp006", 30.0, 6.0),
)
EXPECTED_ACTIVE_PHASE_COUNTS = {"left_low_march": 30, "right_low_march": 30}
ACTIVE_RIGHT_ARM_ARTICULATION_ROLE = (
    "held_at_bilateral_reach_pose_reach_control_equals_1.0"
)


class S1GeometryError(RuntimeError):
    """Raised if the S1 geometry contract cannot be satisfied."""


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[index]) + float(right[index]) for index in range(3))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))


def _scale(vector: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(value) * float(scalar) for value in vector)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 1.0e-12:
        raise S1GeometryError("Cannot normalise a zero-length vector.")
    return tuple(float(value) / length for value in vector)


def _finite_triplet(value: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise S1GeometryError(f"{label} must have three values.")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise S1GeometryError(f"{label} contains NaN or Inf.")
    return result


def is_active_frame(frame: int) -> bool:
    """Return whether the shared S1 panel must be rendered at ``frame``."""
    return any(start <= int(frame) <= end for start, end in ACTIVE_RANGES)


def validate_inputs(
    camera_plan: Mapping[str, Any],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    """Validate the complete camera-bank-v2 and 240-frame skeleton inputs."""
    candidates = list(camera_plan.get("candidates", []))
    ids = [str(entry.get("candidate_id", "")) for entry in candidates]
    if len(candidates) != EXPECTED_CAMERA_COUNT or len(set(ids)) != EXPECTED_CAMERA_COUNT:
        raise S1GeometryError("S1 requires nine unique camera-bank-v2 candidates.")
    expected_ids = [item[0] for item in EXPECTED_VIEW_LAYOUT]
    if ids != expected_ids:
        raise S1GeometryError(
            "The camera-bank-v2 candidate ids/order changed; expected the exact "
            "registered nine-view layout."
        )
    if tuple(float(value) for value in camera_plan.get("azimuth_offsets_deg", [])) != (
        tuple(float(value) for value in AZIMUTH_OFFSETS_DEG)
    ):
        raise S1GeometryError("The camera-bank-v2 azimuth levels changed.")
    if tuple(float(value) for value in camera_plan.get("elevation_offsets_deg", [])) != (
        tuple(float(value) for value in ELEVATION_OFFSETS_DEG)
    ):
        raise S1GeometryError("The camera-bank-v2 elevation levels changed.")
    for entry, (expected_id, expected_azimuth, expected_elevation) in zip(
        candidates, EXPECTED_VIEW_LAYOUT
    ):
        if str(entry.get("candidate_id")) != expected_id:
            raise S1GeometryError("The registered candidate-id mapping changed.")
        if float(entry.get("azimuth_offset_deg", math.nan)) != expected_azimuth:
            raise S1GeometryError(f"{expected_id} has the wrong azimuth offset.")
        if float(entry.get("elevation_offset_deg", math.nan)) != expected_elevation:
            raise S1GeometryError(f"{expected_id} has the wrong elevation offset.")
        _finite_triplet(entry.get("eye_world_operational_m", []), expected_id + " eye")
        _finite_triplet(entry.get("aim_world_operational_m", []), expected_id + " aim")

    phase_counts: dict[str, int] = {}
    for frame in ACTIVE_FRAMES:
        phase = phase_for_frame(frame)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        controls = controls_for_frame(frame)
        if abs(float(controls["reach"]) - 1.0) > 1.0e-12:
            raise S1GeometryError(
                "An S1 active frame is not held at the v2 bilateral-reach pose."
            )
    if phase_counts != EXPECTED_ACTIVE_PHASE_COUNTS:
        raise S1GeometryError(
            "S1 active ranges no longer bind to 30 left-low-march and 30 "
            "right-low-march v2 frames."
        )

    if set(positions_by_frame) != set(CAPTURE_FRAMES):
        raise S1GeometryError("Skeleton geometry must contain exactly frames 0..239.")
    for frame in CAPTURE_FRAMES:
        missing = set(JOINTS) - set(positions_by_frame[frame])
        if missing:
            raise S1GeometryError(f"Frame {frame} is missing joints: {sorted(missing)}")
        for joint in JOINTS:
            _finite_triplet(positions_by_frame[frame][joint], f"frame {frame} {joint}")
    return {
        "input_contract_pass": True,
        "camera_count": len(candidates),
        "capture_frame_count": len(CAPTURE_FRAMES),
        "active_frame_count": len(ACTIVE_FRAMES),
        "active_phase_counts": phase_counts,
        "source_motion_tag": MOTION_TAG,
        "right_arm_articulation_role": ACTIVE_RIGHT_ARM_ARTICULATION_ROLE,
        "anchor_view_id": ANCHOR_VIEW_ID,
    }


def panel_for_frame(
    joints: Mapping[str, Sequence[float]],
    anchor_camera: Mapping[str, Any],
    *,
    plane_offset_toward_anchor_m: float,
    horizontal_margin_m: float,
    vertical_margin_m: float,
    panel_thickness_m: float,
) -> dict[str, Any]:
    """Construct one anchor-facing dynamic world-space OBB for one frame."""
    eye = _finite_triplet(anchor_camera["eye_world_operational_m"], "anchor eye")
    aim = _finite_triplet(anchor_camera["aim_world_operational_m"], "anchor aim")
    right, up, forward = camera_basis(eye, aim)
    arm_points = [_finite_triplet(joints[name], name) for name in RIGHT_ARM_JOINTS]
    depths = [_dot(_sub(point, eye), forward) for point in arm_points]
    if min(depths) <= (
        float(plane_offset_toward_anchor_m)
        + MIN_FIRST_SURFACE_FORWARD_LEAD_M
    ):
        raise S1GeometryError("The requested S1 plane is not safely in front of the arm.")
    plane_depth = min(depths) - float(plane_offset_toward_anchor_m)
    plane_origin = _add(eye, _scale(forward, plane_depth))

    coordinates: list[tuple[float, float]] = []
    for point, depth in zip(arm_points, depths):
        intersection = _add(eye, _scale(_sub(point, eye), plane_depth / depth))
        delta = _sub(intersection, plane_origin)
        coordinates.append((_dot(delta, right), _dot(delta, up)))
    minimum_right = min(value[0] for value in coordinates)
    maximum_right = max(value[0] for value in coordinates)
    minimum_up = min(value[1] for value in coordinates)
    maximum_up = max(value[1] for value in coordinates)
    centre_right = 0.5 * (minimum_right + maximum_right)
    centre_up = 0.5 * (minimum_up + maximum_up)
    centre = _add(
        _add(plane_origin, _scale(right, centre_right)),
        _scale(up, centre_up),
    )
    half_extents = (
        0.5 * (maximum_right - minimum_right) + float(horizontal_margin_m),
        0.5 * (maximum_up - minimum_up) + float(vertical_margin_m),
        0.5 * float(panel_thickness_m),
    )
    if min(half_extents) <= 0.0:
        raise S1GeometryError("S1 panel half extents must be positive.")
    return {
        "centre_world_operational_m": list(centre),
        "axes_world": [list(right), list(up), list(forward)],
        "half_extents_operational_m": list(half_extents),
        "plane_depth_from_anchor_m": plane_depth,
        "arm_forward_depth_range_m": [min(depths), max(depths)],
        "volume_m3": 8.0 * half_extents[0] * half_extents[1] * half_extents[2],
    }


def ray_obb_first_hit_distance(
    origin: Sequence[float],
    target: Sequence[float],
    obb: Mapping[str, Any],
) -> float | None:
    """Return the first world-distance hit on an OBB before ``target``."""
    ray = _sub(target, origin)
    target_distance = _norm(ray)
    if target_distance <= 1.0e-12:
        raise S1GeometryError("Ray origin and target coincide.")
    direction = _unit(ray)
    centre = _finite_triplet(obb["centre_world_operational_m"], "OBB centre")
    axes = [_finite_triplet(axis, "OBB axis") for axis in obb["axes_world"]]
    half_extents = tuple(float(value) for value in obb["half_extents_operational_m"])
    t_near = 0.0
    t_far = target_distance
    relative = _sub(origin, centre)
    for axis, half_extent in zip(axes, half_extents):
        projected_origin = _dot(relative, axis)
        projected_direction = _dot(direction, axis)
        if abs(projected_direction) <= 1.0e-12:
            if projected_origin < -half_extent or projected_origin > half_extent:
                return None
            continue
        first = (-half_extent - projected_origin) / projected_direction
        second = (half_extent - projected_origin) / projected_direction
        if first > second:
            first, second = second, first
        t_near = max(t_near, first)
        t_far = min(t_far, second)
        if t_near > t_far:
            return None
    if t_far < 0.0 or t_near > target_distance:
        return None
    return max(0.0, t_near)


def bone_overlap_fraction(
    eye: Sequence[float],
    camera_forward: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
    obb: Mapping[str, Any],
) -> tuple[float, float | None, float | None]:
    """Return overlap plus range/forward lead diagnostics.

    A ray sample counts as occluded only when the OBB first hit is at least
    0.10 m closer in that *view's camera-forward depth*.  Euclidean range lead
    is retained only as a diagnostic and cannot make a sample pass.
    """
    origin = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(camera_forward, dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    if forward.shape != (3,) or forward_norm <= 1.0e-12:
        raise S1GeometryError("Camera forward must be one nonzero 3-vector.")
    forward = forward / forward_norm
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    fractions = np.linspace(0.0, 1.0, BONE_SAMPLE_COUNT, dtype=np.float64)
    points = first[None, :] + fractions[:, None] * (second - first)[None, :]
    rays = points - origin[None, :]
    target_distances = np.linalg.norm(rays, axis=1)
    if np.any(target_distances <= 1.0e-12):
        raise S1GeometryError("A sampled bone point coincides with the camera eye.")
    directions = rays / target_distances[:, None]
    forward_cosines = directions @ forward
    if np.any(forward_cosines <= 1.0e-12):
        raise S1GeometryError("A sampled bone point is not in camera-forward space.")
    centre = np.asarray(obb["centre_world_operational_m"], dtype=np.float64)
    axes = np.asarray(obb["axes_world"], dtype=np.float64)
    half_extents = np.asarray(obb["half_extents_operational_m"], dtype=np.float64)
    if axes.shape != (3, 3) or half_extents.shape != (3,):
        raise S1GeometryError("The OBB axis/extent shape changed.")

    t_near = np.zeros(BONE_SAMPLE_COUNT, dtype=np.float64)
    t_far = target_distances.copy()
    valid = np.ones(BONE_SAMPLE_COUNT, dtype=bool)
    relative = origin - centre
    for axis, half_extent in zip(axes, half_extents):
        projected_origin = float(np.dot(relative, axis))
        projected_direction = directions @ axis
        nonparallel = np.abs(projected_direction) > 1.0e-12
        if projected_origin < -half_extent or projected_origin > half_extent:
            valid &= nonparallel
        first_hit = np.empty(BONE_SAMPLE_COUNT, dtype=np.float64)
        second_hit = np.empty(BONE_SAMPLE_COUNT, dtype=np.float64)
        first_hit.fill(-np.inf)
        second_hit.fill(np.inf)
        first_hit[nonparallel] = (
            -half_extent - projected_origin
        ) / projected_direction[nonparallel]
        second_hit[nonparallel] = (
            half_extent - projected_origin
        ) / projected_direction[nonparallel]
        lower = np.minimum(first_hit, second_hit)
        upper = np.maximum(first_hit, second_hit)
        t_near = np.maximum(t_near, lower)
        t_far = np.minimum(t_far, upper)
        valid &= t_near <= t_far
    valid &= t_far >= 0.0
    valid &= t_near <= target_distances
    first_hits = np.maximum(0.0, t_near)
    range_leads = target_distances - first_hits
    forward_leads = range_leads * forward_cosines
    counted = valid & (
        forward_leads + 1.0e-12 >= MIN_FIRST_SURFACE_FORWARD_LEAD_M
    )
    count = int(np.count_nonzero(counted))
    hit_count = int(np.count_nonzero(valid))
    minimum_range_lead = float(np.min(range_leads[valid])) if hit_count else None
    minimum_forward_lead = float(np.min(forward_leads[valid])) if hit_count else None
    return (
        count / float(BONE_SAMPLE_COUNT),
        minimum_range_lead,
        minimum_forward_lead,
    )


def overlap_class(overlap: float) -> str:
    if float(overlap) + 1.0e-12 >= OCCLUDED_THRESHOLD:
        return "intended_occluded"
    if float(overlap) <= CLEAR_THRESHOLD + 1.0e-12:
        return "clear"
    return "transitional"


def _subset_geometry_audit(
    candidates: Sequence[Mapping[str, Any]],
    target_cell_classes: Mapping[tuple[str, int, str], str],
    control_cell_overlaps: Mapping[tuple[str, int, str], float],
) -> dict[str, Any]:
    """Audit all 70 fixed subsets across every target cell and both bursts."""
    by_id = {str(entry["candidate_id"]): entry for entry in candidates}
    other_ids = sorted(set(by_id) - {ANCHOR_VIEW_ID})
    feasible: list[tuple[str, ...]] = []
    rows: list[dict[str, Any]] = []
    for subset_index, others in enumerate(itertools.combinations(other_ids, 4)):
        subset = tuple(sorted((ANCHOR_VIEW_ID,) + others))
        azimuths = {float(by_id[view]["azimuth_offset_deg"]) for view in subset}
        elevations = {float(by_id[view]["elevation_offset_deg"]) for view in subset}
        azimuth_diversity_pass = azimuths == set(
            float(value) for value in AZIMUTH_OFFSETS_DEG
        )
        elevation_diversity_pass = elevations == set(
            float(value) for value in ELEVATION_OFFSETS_DEG
        )
        clear_failure_count = 0
        occluded_failure_count = 0
        for frame in ACTIVE_FRAMES:
            for bone in TARGET_BONES:
                classes = [target_cell_classes[(view, frame, bone)] for view in subset]
                clear_failure_count += int(classes.count("clear") < 3)
                occluded_failure_count += int(
                    classes.count("intended_occluded") < 1
                )
        burst_audit: dict[str, dict[str, Any]] = {}
        for start, end in ACTIVE_RANGES:
            burst = f"{start}_{end}"
            values = [
                control_cell_overlaps[(view, frame, bone)]
                for view in subset
                for frame in range(start, end + 1)
                for bone in NEGATIVE_CONTROL_BONES
            ]
            above = sum(value > CLEAR_THRESHOLD + 1.0e-12 for value in values)
            mean_overlap = sum(values) / len(values)
            fraction_above = above / len(values)
            burst_audit[burst] = {
                "cell_denominator": len(values),
                "mean_overlap": mean_overlap,
                "cell_fraction_above_0_05": fraction_above,
                "mean_overlap_gate_pass": (
                    mean_overlap
                    <= MAX_NEGATIVE_CONTROL_MEAN_OVERLAP + 1.0e-12
                ),
                "fraction_above_0_05_gate_pass": (
                    fraction_above
                    <= MAX_NEGATIVE_CONTROL_CELL_FRACTION_ABOVE_CLEAR + 1.0e-12
                ),
            }
        failure_reasons: list[str] = []
        if not azimuth_diversity_pass:
            failure_reasons.append("missing_registered_azimuth_level")
        if not elevation_diversity_pass:
            failure_reasons.append("missing_registered_elevation_level")
        if clear_failure_count:
            failure_reasons.append("target_cell_clear_view_count_below_3")
        if occluded_failure_count:
            failure_reasons.append(
                "target_cell_intended_occluded_view_count_below_1"
            )
        for burst, report in burst_audit.items():
            if not report["mean_overlap_gate_pass"]:
                failure_reasons.append(
                    f"negative_control_{burst}_mean_overlap_above_0_05"
                )
            if not report["fraction_above_0_05_gate_pass"]:
                failure_reasons.append(
                    f"negative_control_{burst}_fraction_above_0_05_above_0_05"
                )
        passed = not failure_reasons
        row = {
            "registered_subset_index": subset_index,
            "view_ids": list(subset),
            "same_fixed_subset_used_for_all_target_cells_and_both_bursts": True,
            "target_bone_active_frame_cell_denominator": len(TARGET_BONES)
            * len(ACTIVE_FRAMES),
            "target_clear_view_count_below_3_cell_count": clear_failure_count,
            "target_intended_occluded_view_count_below_1_cell_count": (
                occluded_failure_count
            ),
            "azimuth_diversity_pass": azimuth_diversity_pass,
            "elevation_diversity_pass": elevation_diversity_pass,
            "negative_control_by_burst": burst_audit,
            "subset_pass": passed,
            "failure_reasons": failure_reasons,
            "selected": False,
        }
        rows.append(row)
        if passed:
            feasible.append(subset)
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return {
        "all_anchor_containing_five_view_subsets_enumerated": len(rows),
        "expected_anchor_containing_five_view_subsets": EXPECTED_SUBSET_COUNT,
        "geometry_only_feasible_subset_count": len(feasible),
        "negative_control_cell_denominator_per_subset_per_burst": 5 * 6 * 30,
        "complete_subset_audit_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "subset_rows": rows,
        "same_fixed_subset_required_across_all_cells_and_bursts": True,
        "per_cell_subset_switching_allowed": False,
        "selected_view_ids": None,
        "interpretation": (
            "A nonzero count proves only that S1 geometry leaves at least one "
            "fixed diverse five-view option across every target cell and both "
            "bursts. It neither chooses nor ranks the final five."
        ),
    }


def evaluate_geometry(
    camera_plan: Mapping[str, Any],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
    parameters: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate all registered S1 overlap denominators for one grid entry."""
    validate_inputs(camera_plan, positions_by_frame)
    candidates = list(camera_plan["candidates"])
    by_id = {str(entry["candidate_id"]): entry for entry in candidates}
    anchor = by_id[ANCHOR_VIEW_ID]
    panels = {
        frame: panel_for_frame(
            positions_by_frame[frame],
            anchor,
            plane_offset_toward_anchor_m=float(parameters["plane_offset_toward_anchor_m"]),
            horizontal_margin_m=float(parameters["horizontal_margin_m"]),
            vertical_margin_m=float(parameters["vertical_margin_m"]),
            panel_thickness_m=float(parameters["panel_thickness_m"]),
        )
        for frame in CAPTURE_FRAMES
    }

    rows: list[dict[str, Any]] = []
    target_classes: dict[tuple[str, int, str], str] = {}
    control_cell_overlaps: dict[tuple[str, int, str], float] = {}
    target_overlaps: list[float] = []
    anchor_target_overlaps: list[float] = []
    control_overlaps: list[float] = []
    by_burst_controls: dict[str, list[float]] = {"90_119": [], "170_199": []}
    minimum_clear_count = EXPECTED_CAMERA_COUNT
    minimum_occluded_count = EXPECTED_CAMERA_COUNT

    for frame in ACTIVE_FRAMES:
        burst = "90_119" if frame <= 119 else "170_199"
        for entry in candidates:
            view_id = str(entry["candidate_id"])
            eye = entry["eye_world_operational_m"]
            _, _, camera_forward = camera_basis(
                eye, entry["aim_world_operational_m"]
            )
            for bone in SAMPLED_BONES:
                start_name, end_name = BONES[bone]
                overlap, minimum_range_lead, minimum_forward_lead = bone_overlap_fraction(
                    eye,
                    camera_forward,
                    positions_by_frame[frame][start_name],
                    positions_by_frame[frame][end_name],
                    panels[frame],
                )
                classification = overlap_class(overlap)
                role = "target" if bone in TARGET_BONES else "negative_control"
                rows.append({
                    "view_id": view_id,
                    "sequence_index": frame,
                    "burst": burst,
                    "bone": bone,
                    "role": role,
                    "occluded_sample_count": int(round(overlap * BONE_SAMPLE_COUNT)),
                    "bone_sample_denominator": BONE_SAMPLE_COUNT,
                    "overlap_fraction": overlap,
                    "overlap_class": classification,
                    "minimum_first_surface_range_lead_m_diagnostic": minimum_range_lead,
                    "minimum_first_surface_camera_forward_lead_m": minimum_forward_lead,
                    "occlusion_lead_gate_uses": "camera_forward_lead_only",
                })
                if role == "target":
                    target_overlaps.append(overlap)
                    target_classes[(view_id, frame, bone)] = classification
                    if view_id == ANCHOR_VIEW_ID:
                        anchor_target_overlaps.append(overlap)
                else:
                    control_overlaps.append(overlap)
                    by_burst_controls[burst].append(overlap)
                    control_cell_overlaps[(view_id, frame, bone)] = overlap

        for bone in TARGET_BONES:
            classes = [target_classes[(str(entry["candidate_id"]), frame, bone)] for entry in candidates]
            minimum_clear_count = min(minimum_clear_count, classes.count("clear"))
            minimum_occluded_count = min(
                minimum_occluded_count, classes.count("intended_occluded")
            )

    if len(target_overlaps) != EXPECTED_TARGET_CELL_COUNT:
        raise S1GeometryError("Target overlap denominator drifted.")
    if len(anchor_target_overlaps) != EXPECTED_ANCHOR_TARGET_CELL_COUNT:
        raise S1GeometryError("Anchor target overlap denominator drifted.")
    if len(control_overlaps) != EXPECTED_CONTROL_CELL_COUNT:
        raise S1GeometryError("Negative-control overlap denominator drifted.")
    transitional_count = sum(overlap_class(value) == "transitional" for value in target_overlaps)
    control_above = sum(value > CLEAR_THRESHOLD + 1.0e-12 for value in control_overlaps)
    burst_reports = {}
    for burst, values in by_burst_controls.items():
        above = sum(value > CLEAR_THRESHOLD + 1.0e-12 for value in values)
        burst_reports[burst] = {
            "cell_denominator": len(values),
            "mean_overlap": sum(values) / len(values),
            "cell_fraction_above_0_05": above / len(values),
        }
    subset_audit = _subset_geometry_audit(
        candidates, target_classes, control_cell_overlaps
    )
    gates = {
        "anchor_all_120_target_cells_overlap_at_least_0_80": (
            min(anchor_target_overlaps) + 1.0e-12 >= OCCLUDED_THRESHOLD
        ),
        "each_of_120_target_bone_frames_has_at_least_3_clear_pool_views": (
            minimum_clear_count >= MIN_POOL_CLEAR_VIEW_COUNT
        ),
        "each_of_120_target_bone_frames_has_at_least_1_intended_occluded_pool_view": (
            minimum_occluded_count >= MIN_POOL_INTENDED_OCCLUDED_VIEW_COUNT
        ),
        "at_least_one_geometry_only_diverse_five_view_subset_remains": (
            subset_audit["geometry_only_feasible_subset_count"] >= 1
        ),
    }
    volumes = [float(panels[frame]["volume_m3"]) for frame in ACTIVE_FRAMES]
    summary = {
        "geometry_pass": all(gates.values()),
        "parameters": {key: float(value) for key, value in parameters.items()},
        "overlap_denominators": {
            "samples_per_view_bone_active_frame_cell": BONE_SAMPLE_COUNT,
            "target_view_bone_active_frame_cells": len(target_overlaps),
            "anchor_target_bone_active_frame_cells": len(anchor_target_overlaps),
            "negative_control_view_bone_active_frame_cells": len(control_overlaps),
            "target_bone_active_frame_cells_for_view_count_gates": 2 * 60,
            "negative_control_cells_per_burst": 9 * 6 * 30,
        },
        "anchor_minimum_target_overlap": min(anchor_target_overlaps),
        "candidate_pool_minimum_clear_view_count_per_target_bone_frame": minimum_clear_count,
        "candidate_pool_minimum_intended_occluded_view_count_per_target_bone_frame": minimum_occluded_count,
        "target_transitional_cell_count": transitional_count,
        "target_transitional_cell_fraction": transitional_count / len(target_overlaps),
        "target_transitional_fraction_is_report_only_not_a_pool_wide_gate": True,
        "negative_control_mean_overlap": sum(control_overlaps) / len(control_overlaps),
        "negative_control_cell_count_above_0_05": control_above,
        "negative_control_cell_fraction_above_0_05": control_above / len(control_overlaps),
        "negative_control_by_burst": burst_reports,
        "pool_wide_negative_control_summaries_are_report_only": True,
        "active_panel_volume_m3": {
            "mean": sum(volumes) / len(volumes),
            "max": max(volumes),
            "min": min(volumes),
        },
        "geometry_only_subset_audit": subset_audit,
        "gates": gates,
        "uses_clean_skeleton_for_projection_geometry": True,
        "uses_GT_position_or_limb_length_error": False,
        "selected_view_ids": None,
    }
    return {"summary": summary, "panels_by_frame": panels, "overlap_rows": rows}


def plan_s1_geometry(
    camera_plan: Mapping[str, Any],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    """Enumerate the frozen S1 grid and choose minimum mean opaque volume."""
    validate_inputs(camera_plan, positions_by_frame)
    candidates = list(camera_plan["candidates"])
    anchor = next(
        entry for entry in candidates if entry["candidate_id"] == ANCHOR_VIEW_ID
    )
    volume_order: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    registered_index = 0
    for offset in PLANE_OFFSETS_TOWARD_ANCHOR_M:
        for horizontal in HORIZONTAL_MARGINS_M:
            for vertical in VERTICAL_MARGINS_M:
                for thickness in PANEL_THICKNESSES_M:
                    parameters = {
                        "plane_offset_toward_anchor_m": offset,
                        "horizontal_margin_m": horizontal,
                        "vertical_margin_m": vertical,
                        "panel_thickness_m": thickness,
                    }
                    volumes = [
                        panel_for_frame(
                            positions_by_frame[frame],
                            anchor,
                            plane_offset_toward_anchor_m=offset,
                            horizontal_margin_m=horizontal,
                            vertical_margin_m=vertical,
                            panel_thickness_m=thickness,
                        )["volume_m3"]
                        for frame in ACTIVE_FRAMES
                    ]
                    volume_order.append({
                        "registered_grid_index": registered_index,
                        "parameters": parameters,
                        "mean_opaque_volume_m3": sum(volumes) / len(volumes),
                    })
                    registered_index += 1
    if registered_index != 27:
        raise S1GeometryError("The registered S1 grid no longer contains 27 entries.")
    volume_order.sort(key=lambda item: (
        item["mean_opaque_volume_m3"],
        item["parameters"]["plane_offset_toward_anchor_m"],
        item["parameters"]["horizontal_margin_m"],
        item["parameters"]["vertical_margin_m"],
        item["parameters"]["panel_thickness_m"],
    ))
    chosen: dict[str, Any] | None = None
    passing_count = 0
    for candidate in volume_order:
        evaluation = evaluate_geometry(
            camera_plan, positions_by_frame, candidate["parameters"]
        )
        evaluation["registered_grid_index"] = candidate["registered_grid_index"]
        summary = evaluation["summary"]
        audit.append({
            "registered_grid_index": candidate["registered_grid_index"],
            "parameters": summary["parameters"],
            "mean_opaque_volume_m3": candidate["mean_opaque_volume_m3"],
            "geometry_pass": summary["geometry_pass"],
            "gates": dict(summary["gates"]),
            "geometry_only_feasible_subset_count": summary[
                "geometry_only_subset_audit"
            ]["geometry_only_feasible_subset_count"],
            "failed_gates": [
                name for name, passed in summary["gates"].items() if not passed
            ],
        })
        if summary["geometry_pass"]:
            passing_count += 1
            if chosen is None:
                chosen = evaluation
    if chosen is None:
        failures = sorted({gate for row in audit for gate in row["failed_gates"]})
        raise S1GeometryError(
            "No S1 geometry passes the frozen analytic gates. Do not author or "
            "capture; issue a new draft. Failed gate set: " + ", ".join(failures)
        )
    chosen["grid_audit"] = audit
    chosen["registered_grid_entry_count"] = len(volume_order)
    chosen["gate_evaluated_grid_entry_count"] = len(audit)
    chosen["passing_grid_entry_count"] = passing_count
    chosen["unevaluated_larger_volume_grid_entry_count"] = 0
    chosen["selection_rule"] = (
        "Fully gate-evaluate all 27 entries. Choose the passing entry with "
        "minimum active-frame mean opaque volume, then numeric parameter tuple. "
        "The audit retains every entry's parameters, mean volume and gates. No "
        "prediction or GT error is an input."
    )
    return chosen
