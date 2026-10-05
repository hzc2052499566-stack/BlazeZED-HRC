"""Pure geometry planner for the FS-CTS5 S2 recoverable-table pilot.

The planner uses only the deterministic FS-CTS5 USD skeleton, the frozen
camera-bank-v2 geometry, and the completed S1 engineering subset record.  It
does not consume RGB-D predictions, limb-length error, or formal data.

One geometry consists of two world-space OBBs: a horizontal table top and a
narrow, camera-facing modesty panel.  Both are visible only in the two
registered 30-frame bursts.  Every overlap cell is one
``view x bone x active frame`` and always contains exactly 33 samples.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

import fs_cts5_s1_arm_occluder_v1 as s1
from fs_cts5_camera_bank_v2 import (
    ANCHOR_VIEW_ID,
    AZIMUTH_OFFSETS_DEG,
    BONES,
    CAPTURE_FRAMES,
    ELEVATION_OFFSETS_DEG,
    JOINTS,
)


ACTIVE_RANGES = s1.ACTIVE_RANGES
ACTIVE_FRAMES = s1.ACTIVE_FRAMES
TARGET_BONES = ("left_thigh", "left_shank", "right_thigh", "right_shank")
NEGATIVE_CONTROL_BONES = (
    "left_upper_arm",
    "left_forearm",
    "right_upper_arm",
    "right_forearm",
)
SAMPLED_BONES = TARGET_BONES + NEGATIVE_CONTROL_BONES

BONE_SAMPLE_COUNT = 33
MIN_FIRST_SURFACE_FORWARD_LEAD_M = 0.10
OCCLUDED_THRESHOLD = 0.80
CLEAR_THRESHOLD = 0.05
MIN_POOL_CLEAR_VIEW_COUNT = 3
MIN_POOL_INTENDED_OCCLUDED_VIEW_COUNT = 1
MAX_NEGATIVE_CONTROL_MEAN_OVERLAP = 0.05
MAX_NEGATIVE_CONTROL_CELL_FRACTION_ABOVE_CLEAR = 0.05

CENTRE_OFFSETS_TOWARD_ANCHOR_M = (0.45, 0.60, 0.75)
TOP_WIDTHS_M = (0.65, 0.80, 0.95)
TOP_DEPTHS_M = (0.18, 0.30, 0.45)
TOP_THICKNESSES_M = (0.06,)
KNEE_TO_HIP_FRACTIONS = (0.80, 0.95, 1.05)
PANEL_WIDTH_FRACTIONS = (0.80, 1.00)
ANKLE_TO_KNEE_FRACTIONS = (0.15, 0.30, 0.45)
PANEL_THICKNESSES_M = (0.05,)

EXPECTED_GRID_COUNT = 486
EXPECTED_SUBSET_COUNT = math.comb(8, 4)
EXPECTED_CAMERA_COUNT = 9
EXPECTED_ACTIVE_FRAME_COUNT = 60
EXPECTED_TARGET_CELL_COUNT = 9 * 4 * 60
EXPECTED_CONTROL_CELL_COUNT = 9 * 4 * 60
EXPECTED_ANCHOR_TARGET_CELL_COUNT = 4 * 60
EXPECTED_PRIOR_S1_SUBSET_INDEX = 29
EXPECTED_PRIOR_S1_SUBSET = (
    "azm045_elm012",
    "azp000_elm012",
    "azp000_elp000",
    "azp030_elm012",
    "azp030_elp006",
)
EXPECTED_S1_RESULT_STATUS = "PASS_excluded_engineering_FS_CTS5_S1_v4_pilot_complete"


class S2GeometryError(RuntimeError):
    """Raised when the registered S2 geometry contract cannot be satisfied."""


def _triplet(value: Sequence[float], label: str) -> np.ndarray:
    if len(value) != 3:
        raise S2GeometryError(f"{label} must contain three values.")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise S2GeometryError(f"{label} contains NaN or Inf.")
    return result


def _unit(value: Sequence[float], label: str) -> np.ndarray:
    result = _triplet(value, label)
    length = float(np.linalg.norm(result))
    if length <= 1.0e-12:
        raise S2GeometryError(f"{label} cannot be zero length.")
    return result / length


def is_active_frame(frame: int) -> bool:
    return any(start <= int(frame) <= end for start, end in ACTIVE_RANGES)


def validate_s1_result(s1_result: Mapping[str, Any]) -> dict[str, Any]:
    if s1_result.get("status") != EXPECTED_S1_RESULT_STATUS:
        raise S2GeometryError("S2 requires the completed S1 engineering result record.")
    if s1_result.get("attempt_consumed") is not True:
        raise S2GeometryError("The bound S1 attempt is not recorded as consumed.")
    if s1_result.get("formal_capture_authorized") is not False:
        raise S2GeometryError("The S1 result must remain excluded engineering evidence.")
    if s1_result.get("selected_view_ids") is not None:
        raise S2GeometryError("S1 must not have selected a final five-view bank.")
    subset = s1_result.get("S1_feasible_subset", {})
    if int(subset.get("registered_subset_index", -1)) != EXPECTED_PRIOR_S1_SUBSET_INDEX:
        raise S2GeometryError("The sole S1-feasible subset index changed.")
    if tuple(subset.get("view_ids", ())) != EXPECTED_PRIOR_S1_SUBSET:
        raise S2GeometryError("The sole S1-feasible subset membership changed.")
    if subset.get("S1_stage_subset_pass") is not True:
        raise S2GeometryError("The bound S1 subset no longer passes S1.")
    return {
        "S1_result_contract_pass": True,
        "S1_feasible_subset_index": EXPECTED_PRIOR_S1_SUBSET_INDEX,
        "S1_feasible_subset_view_ids": list(EXPECTED_PRIOR_S1_SUBSET),
        "S1_final_five_selected": False,
    }


def validate_inputs(
    camera_plan: Mapping[str, Any],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
    s1_result: Mapping[str, Any],
) -> dict[str, Any]:
    base = s1.validate_inputs(camera_plan, positions_by_frame)
    if tuple(float(v) for v in camera_plan.get("azimuth_offsets_deg", ())) != tuple(
        float(v) for v in AZIMUTH_OFFSETS_DEG
    ):
        raise S2GeometryError("The S2 azimuth bank changed.")
    if tuple(float(v) for v in camera_plan.get("elevation_offsets_deg", ())) != tuple(
        float(v) for v in ELEVATION_OFFSETS_DEG
    ):
        raise S2GeometryError("The S2 elevation bank changed.")
    return {**base, **validate_s1_result(s1_result), "S2_target_bones": list(TARGET_BONES)}


def _frame0_anatomy(
    joints: Mapping[str, Sequence[float]], anchor_camera: Mapping[str, Any]
) -> dict[str, Any]:
    left_hip = _triplet(joints["left_hip"], "left hip")
    right_hip = _triplet(joints["right_hip"], "right hip")
    pelvis = 0.5 * (left_hip + right_hip)
    anchor_eye = _triplet(anchor_camera["eye_world_operational_m"], "anchor eye")
    forward = anchor_eye - pelvis
    forward[2] = 0.0
    forward = _unit(forward, "pelvis-to-anchor horizontal axis")
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    lateral = _unit(np.cross(up, forward), "table lateral axis")
    knee_z = float(
        np.median(
            [
                _triplet(joints["left_knee"], "left knee")[2],
                _triplet(joints["right_knee"], "right knee")[2],
            ]
        )
    )
    hip_z = float(np.median([left_hip[2], right_hip[2]]))
    ankle_z = float(
        np.median(
            [
                _triplet(joints["left_ankle"], "left ankle")[2],
                _triplet(joints["right_ankle"], "right ankle")[2],
            ]
        )
    )
    if not ankle_z < knee_z < hip_z:
        raise S2GeometryError("Frame-0 ankle/knee/hip heights are not anatomical.")
    return {
        "pelvis_proxy_world_operational_m": pelvis,
        "forward_world": forward,
        "lateral_world": lateral,
        "up_world": up,
        "baseline_ankle_z": ankle_z,
        "baseline_knee_z": knee_z,
        "baseline_hip_z": hip_z,
    }


def table_geometry(
    joints: Mapping[str, Sequence[float]],
    anchor_camera: Mapping[str, Any],
    parameters: Mapping[str, float],
) -> dict[str, Any]:
    anatomy = _frame0_anatomy(joints, anchor_camera)
    pelvis = anatomy["pelvis_proxy_world_operational_m"]
    forward = anatomy["forward_world"]
    lateral = anatomy["lateral_world"]
    up = anatomy["up_world"]

    centre_offset = float(parameters["centre_offset_toward_anchor_m"])
    top_width = float(parameters["top_width_m"])
    top_depth = float(parameters["top_depth_m"])
    top_thickness = float(parameters["top_thickness_m"])
    knee_to_hip = float(parameters["knee_to_hip_fraction"])
    panel_width_fraction = float(parameters["front_panel_width_fraction_of_top"])
    ankle_to_knee = float(parameters["ankle_to_knee_fraction"])
    panel_thickness = float(parameters["front_panel_thickness_m"])

    top_surface_z = anatomy["baseline_knee_z"] + knee_to_hip * (
        anatomy["baseline_hip_z"] - anatomy["baseline_knee_z"]
    )
    top_underside_z = top_surface_z - top_thickness
    panel_lower_z = anatomy["baseline_ankle_z"] + ankle_to_knee * (
        anatomy["baseline_knee_z"] - anatomy["baseline_ankle_z"]
    )
    panel_height = top_underside_z - panel_lower_z
    if panel_height <= 0.05:
        raise S2GeometryError("The registered S2 panel collapsed vertically.")

    horizontal_centre = pelvis.copy()
    horizontal_centre[2] = 0.0
    horizontal_centre = horizontal_centre + centre_offset * forward
    top_centre = horizontal_centre.copy()
    top_centre[2] = top_surface_z - 0.5 * top_thickness
    panel_centre = horizontal_centre + (0.5 * top_depth - 0.5 * panel_thickness) * forward
    panel_centre[2] = 0.5 * (panel_lower_z + top_underside_z)
    axes = [lateral.tolist(), forward.tolist(), up.tolist()]
    top = {
        "geometry_role": "table_top",
        "centre_world_operational_m": top_centre.tolist(),
        "axes_world": axes,
        "half_extents_operational_m": [0.5 * top_width, 0.5 * top_depth, 0.5 * top_thickness],
    }
    panel_width = top_width * panel_width_fraction
    panel = {
        "geometry_role": "front_modesty_panel",
        "centre_world_operational_m": panel_centre.tolist(),
        "axes_world": axes,
        "half_extents_operational_m": [0.5 * panel_width, 0.5 * panel_thickness, 0.5 * panel_height],
    }
    volume = top_width * top_depth * top_thickness + panel_width * panel_thickness * panel_height
    return {
        "table_top": top,
        "front_modesty_panel": panel,
        "opaque_volume_m3": volume,
        "derived_heights": {
            "baseline_ankle_z": anatomy["baseline_ankle_z"],
            "baseline_knee_z": anatomy["baseline_knee_z"],
            "baseline_hip_z": anatomy["baseline_hip_z"],
            "top_surface_z": top_surface_z,
            "top_underside_z": top_underside_z,
            "panel_lower_z": panel_lower_z,
            "panel_height_m": panel_height,
        },
        "reference_axes": {
            "pelvis_proxy_world_operational_m": pelvis.tolist(),
            "forward_world": forward.tolist(),
            "lateral_world": lateral.tolist(),
            "up_world": up.tolist(),
        },
    }


def ray_multi_obb_first_hit_distance(
    ray_origin: Sequence[float],
    target: Sequence[float],
    obbs: Sequence[Mapping[str, Any]],
) -> float | None:
    hits = [s1.ray_obb_first_hit_distance(ray_origin, target, obb) for obb in obbs]
    finite = [float(hit) for hit in hits if hit is not None]
    return min(finite) if finite else None


def bone_overlap_fraction(
    camera_eye: Sequence[float],
    camera_forward: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
    obbs: Sequence[Mapping[str, Any]],
) -> tuple[float, float | None, float | None]:
    eye = _triplet(camera_eye, "camera eye")
    forward = _unit(camera_forward, "camera forward")
    start_point = _triplet(start, "bone start")
    end_point = _triplet(end, "bone end")
    occluded = 0
    range_leads: list[float] = []
    forward_leads: list[float] = []
    for sample_index in range(BONE_SAMPLE_COUNT):
        lam = sample_index / (BONE_SAMPLE_COUNT - 1)
        point = start_point + lam * (end_point - start_point)
        ray = point - eye
        target_range = float(np.linalg.norm(ray))
        hit = ray_multi_obb_first_hit_distance(eye, point, obbs)
        if hit is None or target_range <= 1.0e-12:
            continue
        direction = ray / target_range
        range_lead = target_range - hit
        forward_lead = range_lead * float(np.dot(direction, forward))
        range_leads.append(range_lead)
        forward_leads.append(forward_lead)
        if forward_lead + 1.0e-12 >= MIN_FIRST_SURFACE_FORWARD_LEAD_M:
            occluded += 1
    return (
        occluded / BONE_SAMPLE_COUNT,
        min(range_leads) if range_leads else None,
        min(forward_leads) if forward_leads else None,
    )


def overlap_class(overlap: float) -> str:
    if float(overlap) + 1.0e-12 >= OCCLUDED_THRESHOLD:
        return "intended_occluded"
    if float(overlap) <= CLEAR_THRESHOLD + 1.0e-12:
        return "clear"
    return "transitional"


def _sample_tensor(
    camera_plan: Mapping[str, Any],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
) -> dict[str, np.ndarray]:
    candidates = list(camera_plan["candidates"])
    lambdas = np.linspace(0.0, 1.0, BONE_SAMPLE_COUNT, dtype=np.float64)
    points = np.empty(
        (len(candidates), len(ACTIVE_FRAMES), len(SAMPLED_BONES), BONE_SAMPLE_COUNT, 3),
        dtype=np.float64,
    )
    eyes = np.empty_like(points)
    forwards = np.empty_like(points)
    for view_index, camera in enumerate(candidates):
        eye = _triplet(camera["eye_world_operational_m"], "camera eye")
        aim = _triplet(camera["aim_world_operational_m"], "camera aim")
        forward = _unit(aim - eye, "camera forward")
        for frame_index, frame in enumerate(ACTIVE_FRAMES):
            for bone_index, bone in enumerate(SAMPLED_BONES):
                start_name, end_name = BONES[bone]
                start = _triplet(positions_by_frame[frame][start_name], start_name)
                end = _triplet(positions_by_frame[frame][end_name], end_name)
                points[view_index, frame_index, bone_index] = (
                    start[None, :] + lambdas[:, None] * (end - start)[None, :]
                )
                eyes[view_index, frame_index, bone_index] = eye
                forwards[view_index, frame_index, bone_index] = forward
    rays = points - eyes
    lengths = np.linalg.norm(rays, axis=-1)
    if np.any(lengths <= 1.0e-12):
        raise S2GeometryError("A camera eye coincides with a sampled skeleton point.")
    directions = rays / lengths[..., None]
    forward_cosines = np.sum(directions * forwards, axis=-1)
    return {
        "points": points,
        "eyes": eyes,
        "directions": directions,
        "lengths": lengths,
        "forward_cosines": forward_cosines,
    }


def _batch_first_hits(samples: Mapping[str, np.ndarray], obb: Mapping[str, Any]) -> np.ndarray:
    shape = samples["lengths"].shape
    eyes = samples["eyes"].reshape((-1, 3))
    directions = samples["directions"].reshape((-1, 3))
    lengths = samples["lengths"].reshape(-1)
    centre = _triplet(obb["centre_world_operational_m"], "OBB centre")
    axes = np.asarray(obb["axes_world"], dtype=np.float64)
    extents = np.asarray(obb["half_extents_operational_m"], dtype=np.float64)
    if axes.shape != (3, 3) or extents.shape != (3,):
        raise S2GeometryError("The S2 OBB shape is invalid.")
    if not np.all(np.isfinite(axes)) or not np.all(np.isfinite(extents)) or np.any(extents <= 0):
        raise S2GeometryError("The S2 OBB contains invalid axes or extents.")
    local_origin = (eyes - centre) @ axes.T
    local_direction = directions @ axes.T
    t_near = np.zeros(len(lengths), dtype=np.float64)
    t_far = lengths.copy()
    valid = np.ones(len(lengths), dtype=bool)
    for axis_index in range(3):
        origin = local_origin[:, axis_index]
        direction = local_direction[:, axis_index]
        extent = float(extents[axis_index])
        parallel = np.abs(direction) <= 1.0e-12
        valid &= ~(parallel & (np.abs(origin) > extent + 1.0e-12))
        safe_direction = np.where(parallel, 1.0, direction)
        first = (-extent - origin) / safe_direction
        second = (extent - origin) / safe_direction
        low = np.where(parallel, -np.inf, np.minimum(first, second))
        high = np.where(parallel, np.inf, np.maximum(first, second))
        t_near = np.maximum(t_near, low)
        t_far = np.minimum(t_far, high)
    valid &= t_far + 1.0e-12 >= t_near
    valid &= t_near <= lengths + 1.0e-12
    return np.where(valid, np.maximum(t_near, 0.0), np.nan).reshape(shape)


def _evaluate_hits(
    samples: Mapping[str, np.ndarray], geometry: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    hits = [
        _batch_first_hits(samples, geometry["table_top"]),
        _batch_first_hits(samples, geometry["front_modesty_panel"]),
    ]
    stacked = np.stack(hits, axis=0)
    finite = np.isfinite(stacked)
    nearest = np.min(np.where(finite, stacked, np.inf), axis=0)
    nearest = np.where(np.any(finite, axis=0), nearest, np.nan)
    range_lead = samples["lengths"] - nearest
    forward_lead = range_lead * samples["forward_cosines"]
    occluded = np.isfinite(nearest) & (
        forward_lead + 1.0e-12 >= MIN_FIRST_SURFACE_FORWARD_LEAD_M
    )
    overlaps = np.mean(occluded, axis=-1)
    return {
        "occluded": occluded,
        "overlaps": overlaps,
        "range_lead": range_lead,
        "forward_lead": forward_lead,
    }


def _subset_audit(
    candidates: Sequence[Mapping[str, Any]], overlaps: np.ndarray
) -> dict[str, Any]:
    by_id = {str(item["candidate_id"]): item for item in candidates}
    index_by_id = {str(item["candidate_id"]): index for index, item in enumerate(candidates)}
    other_ids = sorted(set(by_id) - {ANCHOR_VIEW_ID})
    rows: list[dict[str, Any]] = []
    joint_feasible: list[tuple[str, ...]] = []
    for subset_index, others in enumerate(itertools.combinations(other_ids, 4)):
        subset = tuple(sorted((ANCHOR_VIEW_ID,) + others))
        indices = [index_by_id[view_id] for view_id in subset]
        azimuths = {float(by_id[view]["azimuth_offset_deg"]) for view in subset}
        elevations = {float(by_id[view]["elevation_offset_deg"]) for view in subset}
        azimuth_pass = azimuths == set(float(value) for value in AZIMUTH_OFFSETS_DEG)
        elevation_pass = elevations == set(float(value) for value in ELEVATION_OFFSETS_DEG)
        target = overlaps[np.asarray(indices), :, : len(TARGET_BONES)]
        clear_failures = int(np.sum(np.sum(target <= CLEAR_THRESHOLD + 1.0e-12, axis=0) < 3))
        occluded_failures = int(
            np.sum(np.sum(target + 1.0e-12 >= OCCLUDED_THRESHOLD, axis=0) < 1)
        )
        burst_audit: dict[str, dict[str, Any]] = {}
        for start, end in ACTIVE_RANGES:
            first = ACTIVE_FRAMES.index(start)
            last = ACTIVE_FRAMES.index(end) + 1
            values = overlaps[np.asarray(indices), first:last, len(TARGET_BONES) :].reshape(-1)
            mean_overlap = float(np.mean(values))
            fraction_above = float(np.mean(values > CLEAR_THRESHOLD + 1.0e-12))
            burst_audit[f"{start}_{end}"] = {
                "cell_denominator": int(values.size),
                "mean_overlap": mean_overlap,
                "cell_fraction_above_0_05": fraction_above,
                "mean_overlap_gate_pass": mean_overlap <= MAX_NEGATIVE_CONTROL_MEAN_OVERLAP + 1.0e-12,
                "fraction_above_0_05_gate_pass": fraction_above <= MAX_NEGATIVE_CONTROL_CELL_FRACTION_ABOVE_CLEAR + 1.0e-12,
            }
        prior_s1_pass = subset_index == EXPECTED_PRIOR_S1_SUBSET_INDEX and subset == tuple(
            sorted(EXPECTED_PRIOR_S1_SUBSET)
        )
        reasons: list[str] = []
        if not prior_s1_pass:
            reasons.append("prior_S1_stage_not_feasible_for_this_subset")
        if not azimuth_pass:
            reasons.append("missing_registered_azimuth_level")
        if not elevation_pass:
            reasons.append("missing_registered_elevation_level")
        if occluded_failures:
            reasons.append("S2_target_cell_intended_occluded_view_count_below_1")
        for burst, report in burst_audit.items():
            if not report["mean_overlap_gate_pass"]:
                reasons.append(f"S2_negative_control_{burst}_mean_overlap_above_0_05")
            if not report["fraction_above_0_05_gate_pass"]:
                reasons.append(f"S2_negative_control_{burst}_fraction_above_0_05")
        passed = not reasons
        row = {
            "registered_subset_index": subset_index,
            "view_ids": list(subset),
            "prior_S1_stage_subset_pass": prior_s1_pass,
            "same_fixed_subset_used_for_all_S1_and_S2_cells_and_bursts": True,
            "S2_target_bone_active_frame_cell_denominator": len(TARGET_BONES) * len(ACTIVE_FRAMES),
            "S2_target_clear_view_count_below_3_cell_count": clear_failures,
            "S2_target_clear_view_count_within_subset_is_report_only": True,
            "S2_target_intended_occluded_view_count_below_1_cell_count": occluded_failures,
            "azimuth_diversity_pass": azimuth_pass,
            "elevation_diversity_pass": elevation_pass,
            "negative_control_by_burst": burst_audit,
            "joint_S1_S2_geometry_subset_pass": passed,
            "failure_reasons": reasons,
            "selected": False,
        }
        rows.append(row)
        if passed:
            joint_feasible.append(subset)
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return {
        "all_anchor_containing_five_view_subsets_enumerated": len(rows),
        "expected_anchor_containing_five_view_subsets": EXPECTED_SUBSET_COUNT,
        "prior_S1_feasible_subset_count": 1,
        "joint_S1_S2_geometry_feasible_subset_count": len(joint_feasible),
        "joint_S1_S2_geometry_feasible_view_ids": [list(value) for value in joint_feasible],
        "complete_subset_audit_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "subset_rows": rows,
        "selected_view_ids": None,
        "future_S3_same_subset_required": True,
    }


def _prior_s1_subset_geometry_pass(
    candidates: Sequence[Mapping[str, Any]], overlaps: np.ndarray
) -> bool:
    """Fast grid gate for the only subset that survived S1."""
    index_by_id = {
        str(item["candidate_id"]): index for index, item in enumerate(candidates)
    }
    indices = np.asarray([index_by_id[view_id] for view_id in EXPECTED_PRIOR_S1_SUBSET])
    target = overlaps[indices, :, : len(TARGET_BONES)]
    if np.any(np.sum(target + 1.0e-12 >= OCCLUDED_THRESHOLD, axis=0) < 1):
        return False
    for start, end in ACTIVE_RANGES:
        first = ACTIVE_FRAMES.index(start)
        last = ACTIVE_FRAMES.index(end) + 1
        values = overlaps[indices, first:last, len(TARGET_BONES) :].reshape(-1)
        if float(np.mean(values)) > MAX_NEGATIVE_CONTROL_MEAN_OVERLAP + 1.0e-12:
            return False
        if float(np.mean(values > CLEAR_THRESHOLD + 1.0e-12)) > (
            MAX_NEGATIVE_CONTROL_CELL_FRACTION_ABOVE_CLEAR + 1.0e-12
        ):
            return False
    return True


def _parameters() -> list[dict[str, float]]:
    rows = []
    for values in itertools.product(
        CENTRE_OFFSETS_TOWARD_ANCHOR_M,
        TOP_WIDTHS_M,
        TOP_DEPTHS_M,
        TOP_THICKNESSES_M,
        KNEE_TO_HIP_FRACTIONS,
        PANEL_WIDTH_FRACTIONS,
        ANKLE_TO_KNEE_FRACTIONS,
        PANEL_THICKNESSES_M,
    ):
        rows.append(
            dict(
                zip(
                    (
                        "centre_offset_toward_anchor_m",
                        "top_width_m",
                        "top_depth_m",
                        "top_thickness_m",
                        "knee_to_hip_fraction",
                        "front_panel_width_fraction_of_top",
                        "ankle_to_knee_fraction",
                        "front_panel_thickness_m",
                    ),
                    values,
                )
            )
        )
    if len(rows) != EXPECTED_GRID_COUNT:
        raise S2GeometryError("The registered S2 grid cardinality changed.")
    return rows


def _summary_for_hits(
    candidates: Sequence[Mapping[str, Any]],
    hits: Mapping[str, np.ndarray],
    *,
    complete_subset_audit: bool,
) -> dict[str, Any]:
    overlaps = hits["overlaps"]
    anchor_index = next(
        index for index, item in enumerate(candidates) if item["candidate_id"] == ANCHOR_VIEW_ID
    )
    target = overlaps[:, :, : len(TARGET_BONES)]
    anchor_minimum = float(np.min(target[anchor_index]))
    clear_counts = np.sum(target <= CLEAR_THRESHOLD + 1.0e-12, axis=0)
    occluded_counts = np.sum(target + 1.0e-12 >= OCCLUDED_THRESHOLD, axis=0)
    prior_subset_pass = _prior_s1_subset_geometry_pass(candidates, overlaps)
    subset = _subset_audit(candidates, overlaps) if complete_subset_audit else None
    joint_feasible_count = (
        int(subset["joint_S1_S2_geometry_feasible_subset_count"])
        if subset is not None
        else int(prior_subset_pass)
    )
    gates = {
        "anchor_target_overlap_min_0_80": anchor_minimum + 1.0e-12 >= OCCLUDED_THRESHOLD,
        "candidate_pool_clear_view_count_min_3": int(np.min(clear_counts)) >= 3,
        "candidate_pool_intended_occluded_view_count_min_1": int(np.min(occluded_counts)) >= 1,
        "prior_S1_fixed_subset_continues_through_S2_geometry": (
            joint_feasible_count >= 1
        ),
    }
    return {
        "geometry_pass": all(gates.values()),
        "gates": gates,
        "anchor_minimum_target_overlap": anchor_minimum,
        "candidate_pool_minimum_clear_view_count_per_target_bone_frame": int(np.min(clear_counts)),
        "candidate_pool_minimum_intended_occluded_view_count_per_target_bone_frame": int(np.min(occluded_counts)),
        "joint_S1_S2_geometry_feasible_subset_count": joint_feasible_count,
        "joint_S1_S2_subset_audit": subset,
    }


def _overlap_rows(
    camera_plan: Mapping[str, Any],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    obbs = (geometry["table_top"], geometry["front_modesty_panel"])
    for camera in camera_plan["candidates"]:
        eye = _triplet(camera["eye_world_operational_m"], "camera eye")
        aim = _triplet(camera["aim_world_operational_m"], "camera aim")
        forward = _unit(aim - eye, "camera forward")
        for frame in ACTIVE_FRAMES:
            burst = next(f"{start}_{end}" for start, end in ACTIVE_RANGES if start <= frame <= end)
            for bone in SAMPLED_BONES:
                start_name, end_name = BONES[bone]
                overlap, range_lead, forward_lead = bone_overlap_fraction(
                    eye,
                    forward,
                    positions_by_frame[frame][start_name],
                    positions_by_frame[frame][end_name],
                    obbs,
                )
                rows.append(
                    {
                        "view_id": str(camera["candidate_id"]),
                        "sequence_index": frame,
                        "burst": burst,
                        "bone": bone,
                        "role": "target" if bone in TARGET_BONES else "negative_control",
                        "occluded_sample_count": int(round(overlap * BONE_SAMPLE_COUNT)),
                        "bone_sample_denominator": BONE_SAMPLE_COUNT,
                        "overlap_fraction": overlap,
                        "overlap_class": overlap_class(overlap),
                        "minimum_first_surface_range_lead_m_diagnostic": range_lead,
                        "minimum_first_surface_camera_forward_lead_m": forward_lead,
                        "occlusion_lead_gate_uses": "camera_forward_lead_only",
                    }
                )
    return rows


def plan_s2_geometry(
    camera_plan: Mapping[str, Any],
    positions_by_frame: Mapping[int, Mapping[str, Sequence[float]]],
    s1_result: Mapping[str, Any],
) -> dict[str, Any]:
    input_contract = validate_inputs(camera_plan, positions_by_frame, s1_result)
    candidates = list(camera_plan["candidates"])
    anchor = next(item for item in candidates if item["candidate_id"] == ANCHOR_VIEW_ID)
    samples = _sample_tensor(camera_plan, positions_by_frame)
    grid_audit: list[dict[str, Any]] = []
    passing: list[tuple[float, tuple[float, ...], int, dict[str, Any], dict[str, Any]]] = []
    parameter_names = tuple(_parameters()[0])
    for grid_index, parameters in enumerate(_parameters()):
        geometry = table_geometry(positions_by_frame[0], anchor, parameters)
        hits = _evaluate_hits(samples, geometry)
        summary = _summary_for_hits(candidates, hits, complete_subset_audit=False)
        parameter_tuple = tuple(float(parameters[name]) for name in parameter_names)
        row = {
            "registered_grid_index": grid_index,
            "parameters": dict(parameters),
            "opaque_volume_m3": float(geometry["opaque_volume_m3"]),
            "gates": summary["gates"],
            "geometry_pass": summary["geometry_pass"],
            "anchor_minimum_target_overlap": summary["anchor_minimum_target_overlap"],
            "candidate_pool_minimum_clear_view_count_per_target_bone_frame": summary[
                "candidate_pool_minimum_clear_view_count_per_target_bone_frame"
            ],
            "candidate_pool_minimum_intended_occluded_view_count_per_target_bone_frame": summary[
                "candidate_pool_minimum_intended_occluded_view_count_per_target_bone_frame"
            ],
            "joint_S1_S2_geometry_feasible_subset_count": summary[
                "joint_S1_S2_geometry_feasible_subset_count"
            ],
            "selected": False,
        }
        grid_audit.append(row)
        if summary["geometry_pass"]:
            passing.append(
                (
                    float(geometry["opaque_volume_m3"]),
                    parameter_tuple,
                    grid_index,
                    geometry,
                    summary,
                )
            )
    if not passing:
        raise S2GeometryError(
            "No registered S2 geometry keeps the sole S1-feasible subset recoverable. "
            "Do not capture or relax gates in place."
        )
    _volume, _tuple, selected_index, geometry, _fast_summary = min(
        passing, key=lambda item: (item[0], item[1])
    )
    summary = _summary_for_hits(
        candidates, _evaluate_hits(samples, geometry), complete_subset_audit=True
    )
    grid_audit[selected_index]["selected"] = True
    overlap_rows = _overlap_rows(camera_plan, positions_by_frame, geometry)
    return {
        "schema_version": 1,
        "status": "PASS_excluded_engineering_S2_analytic_geometry",
        "classification": "excluded_engineering_FS_CTS5_S2_projection_geometry",
        "registered_grid_index": selected_index,
        "registered_grid_entry_count": EXPECTED_GRID_COUNT,
        "gate_evaluated_grid_entry_count": EXPECTED_GRID_COUNT,
        "selected_parameters": grid_audit[selected_index]["parameters"],
        "geometry": geometry,
        "summary": {
            **summary,
            "opaque_volume_m3": float(geometry["opaque_volume_m3"]),
            "overlap_denominators": {
                "samples_per_view_bone_active_frame_cell": BONE_SAMPLE_COUNT,
                "target_view_bone_active_frame_cells": EXPECTED_TARGET_CELL_COUNT,
                "anchor_target_bone_active_frame_cells": EXPECTED_ANCHOR_TARGET_CELL_COUNT,
                "negative_control_view_bone_active_frame_cells": EXPECTED_CONTROL_CELL_COUNT,
            },
            "selected_view_ids": None,
            "uses_GT_position_or_limb_length_error": False,
            "formal_capture_authorized": False,
        },
        "grid_audit": grid_audit,
        "overlap_rows": overlap_rows,
        "input_contract": input_contract,
        "selection_rule": "minimum opaque volume, then registered parameter tuple lexicographically",
        "scientific_final_five_selected": False,
        "selected_view_ids": None,
        "formal_capture_authorized": False,
        "S2_capture_authorized": False,
    }
