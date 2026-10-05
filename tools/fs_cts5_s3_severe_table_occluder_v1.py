"""Pure projection-geometry planner for the FS-CTS5 S3 severe-table pilot.

S3 reuses the already selected S2 table top and placement.  It replaces the
front modesty panel with a registered wider/lower panel and, when requested,
adds symmetric side wings.  Geometry is chosen without opening tracking
predictions or any position/limb-length error result.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np

import fs_cts5_s2_table_occluder_v1 as s2


ACTIVE_RANGES = s2.ACTIVE_RANGES
ACTIVE_FRAMES = s2.ACTIVE_FRAMES
TARGET_BONES = s2.TARGET_BONES
NEGATIVE_CONTROL_BONES = s2.NEGATIVE_CONTROL_BONES
SAMPLED_BONES = s2.SAMPLED_BONES
BONE_SAMPLE_COUNT = s2.BONE_SAMPLE_COUNT
OCCLUDED_THRESHOLD = s2.OCCLUDED_THRESHOLD
CLEAR_THRESHOLD = s2.CLEAR_THRESHOLD
MIN_FIRST_SURFACE_FORWARD_LEAD_M = s2.MIN_FIRST_SURFACE_FORWARD_LEAD_M

FIXED_SUBSET_INDEX = s2.EXPECTED_PRIOR_S1_SUBSET_INDEX
FIXED_SUBSET = s2.EXPECTED_PRIOR_S1_SUBSET
MIN_FIXED_SUBSET_INTENDED_OCCLUDED_VIEWS = 3
MAX_NEGATIVE_CONTROL_MEAN_OVERLAP = 0.05
MAX_NEGATIVE_CONTROL_CELL_FRACTION_ABOVE_CLEAR = 0.05

FRONT_PANEL_WIDTHS_M = (1.60, 2.00, 2.40)
PANEL_LOWER_OFFSETS_BELOW_ANKLE_M = (0.00, 0.10)
SYMMETRIC_SIDE_WING_DEPTHS_M = (0.00, 0.35, 0.70)
PANEL_THICKNESSES_M = (0.05,)
EXPECTED_GRID_COUNT = 18
EXPECTED_SUBSET_COUNT = 70
EXPECTED_OVERLAP_ROW_COUNT = 9 * 60 * 8
EXPECTED_S2_GRID_INDEX = 170
EXPECTED_S2_STATUS = "PASS_excluded_engineering_S2_analytic_geometry"


class S3GeometryError(RuntimeError):
    """Raised when the frozen S3 projection contract cannot be satisfied."""


def _copy_obb(obb: Mapping[str, Any], *, role: str | None = None) -> dict[str, Any]:
    return {
        "geometry_role": role or str(obb["geometry_role"]),
        "centre_world_operational_m": [float(v) for v in obb["centre_world_operational_m"]],
        "axes_world": [[float(v) for v in row] for row in obb["axes_world"]],
        "half_extents_operational_m": [float(v) for v in obb["half_extents_operational_m"]],
    }


def validate_s2_geometry_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("status") != EXPECTED_S2_STATUS:
        raise S3GeometryError("S3 requires the frozen passing S2 analytic geometry record.")
    if int(record.get("registered_grid_index", -1)) != EXPECTED_S2_GRID_INDEX:
        raise S3GeometryError("The registered S2 grid index changed.")
    if record.get("formal_capture_authorized") is not False:
        raise S3GeometryError("The S2 geometry must remain excluded engineering evidence.")
    if record.get("selected_view_ids") is not None:
        raise S3GeometryError("S2 must not select the final five-view bank.")
    subset_rows = (
        record.get("summary", {})
        .get("joint_S1_S2_subset_audit", {})
        .get("subset_rows", [])
    )
    if len(subset_rows) != EXPECTED_SUBSET_COUNT:
        raise S3GeometryError("The S2 70-subset audit is missing.")
    fixed = subset_rows[FIXED_SUBSET_INDEX]
    if tuple(fixed.get("view_ids", ())) != tuple(sorted(FIXED_SUBSET)):
        raise S3GeometryError("The fixed S1/S2 subset membership changed.")
    if fixed.get("joint_S1_S2_geometry_subset_pass") is not True:
        raise S3GeometryError("The fixed subset no longer passes S1/S2 geometry.")
    geometry = record.get("geometry", {})
    for role in ("table_top", "front_modesty_panel"):
        if role not in geometry:
            raise S3GeometryError(f"The S2 base geometry is missing {role}.")
        s2._triplet(geometry[role]["centre_world_operational_m"], role)
    return {
        "S2_geometry_record_pass": True,
        "S2_registered_grid_index": EXPECTED_S2_GRID_INDEX,
        "fixed_subset_index": FIXED_SUBSET_INDEX,
        "fixed_subset_view_ids": list(FIXED_SUBSET),
    }


def parameters() -> list[dict[str, float]]:
    rows = [
        {
            "total_front_panel_width_m": width,
            "panel_lower_offset_below_baseline_ankle_m": lower,
            "symmetric_side_wing_depth_m": wing,
            "panel_thickness_m": thickness,
        }
        for width, lower, wing, thickness in itertools.product(
            FRONT_PANEL_WIDTHS_M,
            PANEL_LOWER_OFFSETS_BELOW_ANKLE_M,
            SYMMETRIC_SIDE_WING_DEPTHS_M,
            PANEL_THICKNESSES_M,
        )
    ]
    if len(rows) != EXPECTED_GRID_COUNT:
        raise S3GeometryError("The registered S3 grid cardinality changed.")
    return rows


def severe_geometry(
    s2_record: Mapping[str, Any], values: Mapping[str, float]
) -> dict[str, Any]:
    validate_s2_geometry_record(s2_record)
    base = s2_record["geometry"]
    top = _copy_obb(base["table_top"])
    old_panel = base["front_modesty_panel"]
    axes = np.asarray(old_panel["axes_world"], dtype=np.float64)
    if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3), atol=1.0e-10):
        raise S3GeometryError("The S2 reference axes are not orthonormal.")
    lateral, forward, up = axes
    old_centre = np.asarray(old_panel["centre_world_operational_m"], dtype=np.float64)
    old_half = np.asarray(old_panel["half_extents_operational_m"], dtype=np.float64)
    old_lower_z = float(old_centre[2] - old_half[2])
    baseline_ankle_z = float(base["derived_heights"]["baseline_ankle_z"])
    top_underside_z = float(base["derived_heights"]["top_underside_z"])

    width = float(values["total_front_panel_width_m"])
    lower_offset = float(values["panel_lower_offset_below_baseline_ankle_m"])
    wing_depth = float(values["symmetric_side_wing_depth_m"])
    thickness = float(values["panel_thickness_m"])
    lower_z = baseline_ankle_z - lower_offset
    height = top_underside_z - lower_z
    if height <= 0.05:
        raise S3GeometryError("The S3 severe panel collapsed vertically.")
    centre = old_centre.copy()
    centre[2] = 0.5 * (lower_z + top_underside_z)
    front = {
        "geometry_role": "wide_lower_front_modesty_panel",
        "centre_world_operational_m": centre.tolist(),
        "axes_world": axes.tolist(),
        "half_extents_operational_m": [0.5 * width, 0.5 * thickness, 0.5 * height],
    }
    obbs: list[dict[str, Any]] = [top, front]
    wings: list[dict[str, Any]] = []
    if wing_depth > 0.0:
        for side, sign in (("left", -1.0), ("right", 1.0)):
            wing_centre = centre.copy()
            wing_centre += sign * (0.5 * width - 0.5 * thickness) * lateral
            wing_centre -= (0.5 * wing_depth - 0.5 * thickness) * forward
            wing = {
                "geometry_role": f"{side}_side_wing",
                "centre_world_operational_m": wing_centre.tolist(),
                "axes_world": axes.tolist(),
                "half_extents_operational_m": [0.5 * thickness, 0.5 * wing_depth, 0.5 * height],
            }
            wings.append(wing)
            obbs.append(wing)
    volume = (
        np.prod(np.asarray(top["half_extents_operational_m"]) * 2.0)
        + width * thickness * height
        + 2.0 * thickness * wing_depth * height
    )
    return {
        "table_top": top,
        "wide_lower_front_modesty_panel": front,
        "side_wings": wings,
        "all_obbs": obbs,
        "opaque_volume_m3": float(volume),
        "derived_heights": {
            "baseline_ankle_z": baseline_ankle_z,
            "old_S2_panel_lower_z": old_lower_z,
            "S3_panel_lower_z": lower_z,
            "top_underside_z": top_underside_z,
            "S3_panel_height_m": height,
        },
        "reference_axes": base["reference_axes"],
        "base_S2_table_top_reused_exactly": top == _copy_obb(base["table_top"]),
    }


def _evaluate(samples: Mapping[str, np.ndarray], geometry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    hit_arrays = [s2._batch_first_hits(samples, obb) for obb in geometry["all_obbs"]]
    stacked = np.stack(hit_arrays, axis=0)
    finite = np.isfinite(stacked)
    nearest = np.min(np.where(finite, stacked, np.inf), axis=0)
    nearest = np.where(np.any(finite, axis=0), nearest, np.nan)
    range_lead = samples["lengths"] - nearest
    forward_lead = range_lead * samples["forward_cosines"]
    occluded = np.isfinite(nearest) & (
        forward_lead + 1.0e-12 >= MIN_FIRST_SURFACE_FORWARD_LEAD_M
    )
    return {
        "overlaps": np.mean(occluded, axis=-1),
        "range_lead": range_lead,
        "forward_lead": forward_lead,
    }


def _fixed_subset_summary(candidates: Sequence[Mapping[str, Any]], overlaps: np.ndarray) -> dict[str, Any]:
    index = {str(row["candidate_id"]): i for i, row in enumerate(candidates)}
    indices = np.asarray([index[view] for view in FIXED_SUBSET])
    target = overlaps[indices, :, : len(TARGET_BONES)]
    intended_counts = np.sum(target + 1.0e-12 >= OCCLUDED_THRESHOLD, axis=0)
    intended_failures = int(np.sum(intended_counts < MIN_FIXED_SUBSET_INTENDED_OCCLUDED_VIEWS))
    burst_rows: dict[str, Any] = {}
    controls_pass = True
    for start, end in ACTIVE_RANGES:
        first = ACTIVE_FRAMES.index(start)
        last = ACTIVE_FRAMES.index(end) + 1
        values = overlaps[indices, first:last, len(TARGET_BONES) :].reshape(-1)
        mean = float(np.mean(values))
        fraction = float(np.mean(values > CLEAR_THRESHOLD + 1.0e-12))
        row = {
            "cell_denominator": int(values.size),
            "mean_overlap": mean,
            "cell_fraction_above_0_05": fraction,
            "mean_overlap_gate_pass": mean <= MAX_NEGATIVE_CONTROL_MEAN_OVERLAP + 1.0e-12,
            "fraction_above_0_05_gate_pass": fraction <= MAX_NEGATIVE_CONTROL_CELL_FRACTION_ABOVE_CLEAR + 1.0e-12,
        }
        controls_pass &= row["mean_overlap_gate_pass"] and row["fraction_above_0_05_gate_pass"]
        burst_rows[f"{start}_{end}"] = row
    anchor_index = index[s2.ANCHOR_VIEW_ID]
    anchor_min = float(np.min(overlaps[anchor_index, :, : len(TARGET_BONES)]))
    gates = {
        "fixed_subset_target_intended_occluded_view_count_min_3": intended_failures == 0,
        "fixed_subset_negative_controls_by_burst": bool(controls_pass),
        "anchor_target_whole_bone_overlap_min_0_80": anchor_min + 1.0e-12 >= OCCLUDED_THRESHOLD,
    }
    return {
        "geometry_pass": all(gates.values()),
        "gates": gates,
        "fixed_subset_index": FIXED_SUBSET_INDEX,
        "fixed_subset_view_ids": list(FIXED_SUBSET),
        "target_cell_denominator": len(TARGET_BONES) * len(ACTIVE_FRAMES),
        "target_cell_intended_occluded_count_below_3": intended_failures,
        "minimum_intended_occluded_view_count": int(np.min(intended_counts)),
        "maximum_intended_occluded_view_count": int(np.max(intended_counts)),
        "anchor_minimum_target_overlap": anchor_min,
        "negative_control_by_burst": burst_rows,
        "raw_available_view_count_not_evaluated_at_geometry_stage": True,
    }


def _overlap_rows(
    camera_plan: Mapping[str, Any],
    positions: Mapping[int, Mapping[str, Sequence[float]]],
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for camera in camera_plan["candidates"]:
        eye = s2._triplet(camera["eye_world_operational_m"], "camera eye")
        aim = s2._triplet(camera["aim_world_operational_m"], "camera aim")
        forward = s2._unit(aim - eye, "camera forward")
        for frame in ACTIVE_FRAMES:
            burst = next(f"{a}_{b}" for a, b in ACTIVE_RANGES if a <= frame <= b)
            for bone in SAMPLED_BONES:
                start_name, end_name = s2.BONES[bone]
                overlap, range_lead, forward_lead = s2.bone_overlap_fraction(
                    eye, forward, positions[frame][start_name], positions[frame][end_name], geometry["all_obbs"]
                )
                rows.append({
                    "view_id": str(camera["candidate_id"]),
                    "sequence_index": frame,
                    "burst": burst,
                    "bone": bone,
                    "role": "target" if bone in TARGET_BONES else "negative_control",
                    "occluded_sample_count": int(round(overlap * BONE_SAMPLE_COUNT)),
                    "bone_sample_denominator": BONE_SAMPLE_COUNT,
                    "overlap_fraction": overlap,
                    "overlap_class": s2.overlap_class(overlap),
                    "minimum_first_surface_range_lead_m_diagnostic": range_lead,
                    "minimum_first_surface_camera_forward_lead_m": forward_lead,
                    "occlusion_lead_gate_uses": "camera_forward_lead_only",
                })
    return rows


def plan_s3_geometry(
    camera_plan: Mapping[str, Any],
    positions: Mapping[int, Mapping[str, Sequence[float]]],
    s2_record: Mapping[str, Any],
) -> dict[str, Any]:
    base_contract = s2.validate_inputs(
        camera_plan,
        positions,
        # validate_inputs only needs the S1 result; S3 binds the S2 record separately.
        {"status": s2.EXPECTED_S1_RESULT_STATUS, "attempt_consumed": True,
         "formal_capture_authorized": False, "selected_view_ids": None,
         "S1_feasible_subset": {"registered_subset_index": FIXED_SUBSET_INDEX,
          "view_ids": list(FIXED_SUBSET), "S1_stage_subset_pass": True}},
    )
    s2_contract = validate_s2_geometry_record(s2_record)
    samples = s2._sample_tensor(camera_plan, positions)
    audit: list[dict[str, Any]] = []
    passing: list[tuple[float, tuple[float, ...], int, dict[str, Any], dict[str, Any]]] = []
    names = tuple(parameters()[0])
    for grid_index, values in enumerate(parameters()):
        geometry = severe_geometry(s2_record, values)
        summary = _fixed_subset_summary(camera_plan["candidates"], _evaluate(samples, geometry)["overlaps"])
        row = {
            "registered_grid_index": grid_index,
            "parameters": dict(values),
            "opaque_volume_m3": geometry["opaque_volume_m3"],
            "geometry_pass": summary["geometry_pass"],
            "gates": summary["gates"],
            "minimum_intended_occluded_view_count": summary["minimum_intended_occluded_view_count"],
            "target_cell_intended_occluded_count_below_3": summary["target_cell_intended_occluded_count_below_3"],
            "selected": False,
        }
        audit.append(row)
        if summary["geometry_pass"]:
            passing.append((geometry["opaque_volume_m3"], tuple(values[n] for n in names), grid_index, geometry, summary))
    if not passing:
        raise S3GeometryError("No registered S3 geometry makes fixed subset 29 geometrically severe while preserving arm controls.")
    _volume, _tuple, chosen, geometry, summary = min(passing, key=lambda item: (item[0], item[1]))
    audit[chosen]["selected"] = True
    rows = _overlap_rows(camera_plan, positions, geometry)
    if len(rows) != EXPECTED_OVERLAP_ROW_COUNT:
        raise S3GeometryError("The S3 overlap denominator changed.")
    return {
        "schema_version": 1,
        "status": "PASS_excluded_engineering_S3_analytic_geometry",
        "classification": "excluded_engineering_FS_CTS5_S3_projection_geometry",
        "registered_grid_index": chosen,
        "registered_grid_entry_count": EXPECTED_GRID_COUNT,
        "gate_evaluated_grid_entry_count": EXPECTED_GRID_COUNT,
        "selected_parameters": audit[chosen]["parameters"],
        "geometry": geometry,
        "summary": {**summary, "selected_view_ids": None, "formal_capture_authorized": False},
        "grid_audit": audit,
        "overlap_rows": rows,
        "input_contract": {**base_contract, **s2_contract},
        "selection_rule": "minimum opaque volume, then registered parameter tuple lexicographically",
        "scientific_final_five_selected": False,
        "selected_view_ids": None,
        "formal_capture_authorized": False,
        "S3_capture_authorized": False,
    }
