"""Analyse registered View A/C best-view and fusion methods (pilot only).

All estimator methods consume paired v2 K4 replay rows.  The deployable
best-view rule is arm-level and GT-free: among views with a complete K4 right
arm, compare lexicographically (reliable raw-depth joint count, measured raw-
depth joint count, minimum BlazePose visibility), with View A as a frozen tie
break.  A raw joint is ``occluded_suspected`` when torso_reference_depth minus
joint_depth exceeds 0.50 m; missing raw joint depth is not reliable.

The common-frame transform is fitted from evaluation Skeleton GT and is
explicitly engineering-only.  The oracle selector also reads GT and is only an
upper bound.  No result from this one-capture pilot is formal evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

import build_view_ac_engineering_pair_v1 as geometry
import replay_view_ac_common_frame_engineering_v1 as replay_check


EXPECTED_FRAMES = 240
EXPECTED_JOINTS = 15
ARM_JOINTS = ("right_elbow", "right_wrist")
ACTIVE_FRAMES = frozenset(range(60, 90)) | frozenset(range(150, 180))
RAW_REFERENCE_JOINTS = ("right_shoulder", "pelvis")
OCCLUSION_DEPTH_MARGIN_M = 0.50
CLASSIFICATION = "excluded_engineering_same_capture_best_view_fusion_pilot"
CLAIM_ELIGIBILITY = "excluded_engineering_pilot_only"
CALIBRATION_SOURCE = "gt_correspondence_engineering_only"
METHODS = (
    "view_a_k4",
    "view_c_k4",
    "mean_fusion_available",
    "visibility_weighted_fusion_available",
    "reliability_best_view",
    "oracle_best_view_arm",
)

CANDIDATE_FIELDS = [
    "sequence_index",
    "animation_frame_code",
    "usd_time_code",
    "phase",
    "canonical_joint",
    "method",
    "valid",
    "common_x_m",
    "common_y_m",
    "common_z_m",
    "selected_view",
    "weight_a",
    "weight_c",
    "view_a_valid",
    "view_c_valid",
    "view_a_reliability",
    "view_c_reliability",
    "view_a_visibility",
    "view_c_visibility",
    "selection_score_a",
    "selection_score_c",
    "provenance",
    "classification",
    "claim_eligibility",
    "eligible_for_formal",
    "calibration_source",
]


class AnalysisError(RuntimeError):
    """Raised when the registered pilot-analysis contract is violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisError("JSON root must be an object: {}".format(path))
    return payload


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnalysisError("CSV has no header: {}".format(path))
        return list(reader)


def finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError("{} is not numeric: {!r}".format(label, value)) from error
    if not math.isfinite(result):
        raise AnalysisError("{} is not finite: {!r}".format(label, value))
    return result


def integer(value: object, label: str) -> int:
    number = finite(value, label)
    result = int(number)
    if number != float(result):
        raise AnalysisError("{} is not integral: {!r}".format(label, value))
    return result


def flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(
        ".{}.{}.{}.tmp".format(path.name, os.getpid(), time.time_ns())
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(
        ".{}.{}.{}.tmp".format(path.name, os.getpid(), time.time_ns())
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=CANDIDATE_FIELDS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_ground_truth(
    directory: Path, expected_view_id: str
) -> tuple[
    str,
    str,
    dict[tuple[int, str], dict[str, str]],
    dict[int, dict[str, str]],
]:
    state = read_json(directory / "unit_correction_state.json")
    if state.get("status") != "complete" or float(state.get("correction_factor", 0)) != 100.0:
        raise AnalysisError("Unit correction is incomplete: {}".format(directory))
    outputs = state.get("output_sha256", {})
    for name in (
        "rgbd_manifest.csv",
        "ground_truth_joints.csv",
        "ground_truth_metadata.json",
        "source_frame_content_hashes.csv",
        "source_dynamic_rgbd_gt_protocol.json",
    ):
        path = directory / name
        if not path.is_file() or outputs.get(name) != sha256_file(path):
            raise AnalysisError("Unit-corrected artifact mismatch: {}".format(path))
    manifest_rows = read_csv(directory / "rgbd_manifest.csv")
    gt_rows = read_csv(directory / "ground_truth_joints.csv")
    if len(manifest_rows) != EXPECTED_FRAMES or len(gt_rows) != EXPECTED_FRAMES * EXPECTED_JOINTS:
        raise AnalysisError("Unit-corrected row count mismatch: {}".format(directory))
    manifest: dict[int, dict[str, str]] = {}
    run_id = ""
    camera_prim = ""
    for expected_sequence, row in enumerate(manifest_rows):
        sequence = integer(row.get("sequence_index"), "manifest.sequence_index")
        if sequence != expected_sequence or finite(row.get("usd_time_code"), "usd_time_code") != float(sequence):
            raise AnalysisError("Manifest exact-time contract failed.")
        if not run_id:
            run_id = str(row.get("run_id", ""))
            camera_prim = str(row.get("camera_prim", ""))
        if row.get("run_id") != run_id or row.get("camera_prim") != camera_prim:
            raise AnalysisError("Manifest view identity changed.")
        manifest[sequence] = row
    expected_camera = {
        "a": "/World/ZED_X_01/base_link/ZED_X/CameraLeft",
        "c": "/World/HRCViewCCamera",
    }[expected_view_id]
    if camera_prim != expected_camera:
        raise AnalysisError("Unexpected camera for View {}.".format(expected_view_id))
    keyed: dict[tuple[int, str], dict[str, str]] = {}
    joints_by_frame: dict[int, set[str]] = {}
    for row in gt_rows:
        sequence = integer(row.get("sequence_index"), "gt.sequence_index")
        joint = str(row.get("canonical_joint", ""))
        key = (sequence, joint)
        if row.get("run_id") != run_id or not joint or key in keyed:
            raise AnalysisError("Invalid GT identity/key: {}".format(key))
        keyed[key] = row
        joints_by_frame.setdefault(sequence, set()).add(joint)
    if sorted(joints_by_frame) != list(range(EXPECTED_FRAMES)) or any(
        len(joints) != EXPECTED_JOINTS for joints in joints_by_frame.values()
    ):
        raise AnalysisError("GT joint coverage is incomplete.")
    return run_id, camera_prim, keyed, manifest


def build_calibration(
    gt_a: dict[tuple[int, str], dict[str, str]],
    gt_c: dict[tuple[int, str], dict[str, str]],
    run_a: str,
    run_c: str,
    camera_a: str,
    camera_c: str,
) -> dict[str, Any]:
    if set(gt_a) != set(gt_c):
        raise AnalysisError("A/C GT key sets differ.")
    ordered = sorted(gt_a)
    world_a = np.asarray(
        [[finite(gt_a[key][field], field) for field in ("world_x_m", "world_y_m", "world_z_m")] for key in ordered]
    )
    world_c = np.asarray(
        [[finite(gt_c[key][field], field) for field in ("world_x_m", "world_y_m", "world_z_m")] for key in ordered]
    )
    if not np.array_equal(world_a, world_c):
        raise AnalysisError("A/C same-capture operational-world GT differs.")
    camera_points_a = np.asarray(
        [[finite(gt_a[key][field], field) for field in ("gt_x_m", "gt_y_m", "gt_z_m")] for key in ordered]
    )
    camera_points_c = np.asarray(
        [[finite(gt_c[key][field], field) for field in ("gt_x_m", "gt_y_m", "gt_z_m")] for key in ordered]
    )
    world_from_a = geometry.fit_rigid_transform(
        camera_points_a, world_a, source_frame="view_a_camera_zed", target_frame="operational_world"
    )
    world_from_c = geometry.fit_rigid_transform(
        camera_points_c, world_a, source_frame="view_c_camera_zed", target_frame="operational_world"
    )
    a_from_c = geometry.relative_transform(
        world_from_c,
        world_from_a,
        source_frame="view_c_camera_zed",
        target_frame="view_a_camera_zed",
    )
    residual = geometry.cross_view_residual_summary(
        a_from_c, camera_points_c, camera_points_a
    )
    _, origin_a = geometry.transform_arrays(world_from_a)
    _, origin_c = geometry.transform_arrays(world_from_c)
    return {
        "schema_version": 1,
        "status": "complete",
        "classification": CLASSIFICATION,
        "claim_eligibility": CLAIM_ELIGIBILITY,
        "eligible_for_formal": False,
        "uses_ground_truth": True,
        "calibration_source": CALIBRATION_SOURCE,
        "same_capture_session": True,
        "correspondence_count": len(ordered),
        "views": {
            "a": {"run_id": run_a, "camera_prim": camera_a, "world_from_camera": world_from_a},
            "c": {"run_id": run_c, "camera_prim": camera_c, "world_from_camera": world_from_c},
        },
        "relative_transforms": {"a_from_c": a_from_c},
        "baseline_m": float(np.linalg.norm(origin_c - origin_a)),
        "maximum_cross_view_fit_residual_m": residual["maximum_residual_m"],
        "warning": "GT-derived engineering calibration; prohibited for formal/deployable claims.",
    }


def load_replay(
    replay_dir: Path, expected_run_id: str
) -> dict[str, dict[tuple[int, str], dict[str, str]]]:
    validated = replay_check.validate_replay_directory(replay_dir.resolve(), "pilot")
    state = validated["state"]
    if state.get("manifest_qc", {}).get("run_id") != expected_run_id:
        raise AnalysisError("Replay run_id does not match corrected source.")
    rows = read_csv(validated["estimates_path"])
    selected: dict[str, dict[tuple[int, str], dict[str, str]]] = {
        "raw_measured": {},
        "k4_inferred": {},
    }
    for row in rows:
        method = str(row.get("method", ""))
        if method not in selected:
            continue
        sequence = integer(row.get("sequence_index"), "estimate.sequence_index")
        joint = str(row.get("canonical_joint", ""))
        key = (sequence, joint)
        if row.get("run_id") != expected_run_id or key in selected[method]:
            raise AnalysisError("Replay identity/duplicate key failure.")
        selected[method][key] = row
    if len(selected["raw_measured"]) != EXPECTED_FRAMES * EXPECTED_JOINTS:
        raise AnalysisError("Replay raw row count is not 3600.")
    if len(selected["k4_inferred"]) != EXPECTED_FRAMES * len(ARM_JOINTS):
        raise AnalysisError("Replay K4 right-arm row count is not 480.")
    return selected


def row_point(row: dict[str, str]) -> np.ndarray | None:
    if not flag(row.get("valid")):
        return None
    return np.asarray([finite(row[field], field) for field in ("x_m", "y_m", "z_m")], dtype=float)


def visibility(row: dict[str, str]) -> float:
    value = str(row.get("visibility", "")).strip()
    if not value:
        return 0.0
    return min(1.0, max(0.0, finite(value, "visibility")))


def raw_reliability(
    raw: dict[tuple[int, str], dict[str, str]], sequence: int, joint: str
) -> str:
    joint_row = raw[(sequence, joint)]
    if not flag(joint_row.get("valid")) or not str(joint_row.get("depth_m", "")).strip():
        return "no_measured_depth"
    reference_depths = []
    for reference_joint in RAW_REFERENCE_JOINTS:
        row = raw[(sequence, reference_joint)]
        if flag(row.get("valid")) and str(row.get("depth_m", "")).strip():
            reference_depths.append(finite(row["depth_m"], "reference depth"))
    if len(reference_depths) != len(RAW_REFERENCE_JOINTS):
        return "torso_reference_unavailable"
    torso_reference = statistics.fmean(reference_depths)
    joint_depth = finite(joint_row["depth_m"], "joint depth")
    if torso_reference - joint_depth > OCCLUSION_DEPTH_MARGIN_M:
        return "occluded_suspected"
    return "reliable"


def arm_score(
    replay: dict[str, dict[tuple[int, str], dict[str, str]]], sequence: int
) -> tuple[int, int, float]:
    statuses = [raw_reliability(replay["raw_measured"], sequence, joint) for joint in ARM_JOINTS]
    reliable_count = sum(status == "reliable" for status in statuses)
    measured_count = sum(status != "no_measured_depth" for status in statuses)
    minimum_visibility = min(
        visibility(replay["k4_inferred"][(sequence, joint)]) for joint in ARM_JOINTS
    )
    return reliable_count, measured_count, minimum_visibility


def select_reliability_view(
    score_a: tuple[int, int, float],
    score_c: tuple[int, int, float],
    arm_complete_a: bool,
    arm_complete_c: bool,
) -> str:
    """Apply the frozen GT-free arm-level selector; exact ties choose A."""
    if arm_complete_a and arm_complete_c:
        return "c" if score_c > score_a else "a"
    if arm_complete_a:
        return "a"
    if arm_complete_c:
        return "c"
    return ""


def validate_analysis_protocol(
    protocol: dict[str, Any],
    protocol_path: Path,
    *,
    view_a_dir: Path,
    view_c_dir: Path,
    view_a_replay: Path,
    view_c_replay: Path,
    output_dir: Path,
) -> None:
    if (
        protocol.get("status") != "frozen"
        or protocol.get("classification") != CLASSIFICATION
        or protocol.get("claim_eligibility") != CLAIM_ELIGIBILITY
        or bool(protocol.get("eligible_for_formal"))
    ):
        raise AnalysisError("Analysis protocol is not the frozen excluded pilot protocol.")
    design = protocol.get("registered_design", {})
    expected_design = {
        "frame_count": EXPECTED_FRAMES,
        "active_frame_ranges_inclusive": [[60, 89], [150, 179]],
        "arm_joints": list(ARM_JOINTS),
        "methods": list(METHODS),
        "occlusion_depth_margin_m": OCCLUSION_DEPTH_MARGIN_M,
        "selector_score_lexicographic": [
            "reliable_raw_depth_joint_count",
            "measured_raw_depth_joint_count",
            "minimum_blazepose_visibility",
        ],
        "selector_tie_break": "view_a",
        "selector_scope": "complete_right_arm_frame",
        "calibration_source": CALIBRATION_SOURCE,
        "model_complexity": 0,
        "roi_scale": 0.65,
        "warmup_frames": 0,
        "k4_replay_patch": "prelock_relock_v2",
        "repeat_count": 1,
    }
    if design != expected_design:
        raise AnalysisError("Frozen analysis design does not match this implementation.")
    registered = protocol.get("registered_sha256", {})
    if registered.get("analysis_tool") != sha256_file(Path(__file__).resolve()):
        raise AnalysisError("Analysis tool hash differs from the frozen protocol.")
    paths = protocol.get("registered_paths", {})
    observed_paths = {
        "analysis_protocol": protocol_path,
        "view_a_dir": view_a_dir,
        "view_c_dir": view_c_dir,
        "view_a_replay": view_a_replay,
        "view_c_replay": view_c_replay,
        "analysis_output": output_dir,
    }
    for label, observed in observed_paths.items():
        registered_path = paths.get(label)
        if registered_path is None or Path(str(registered_path)).resolve() != observed.resolve():
            raise AnalysisError("Registered path mismatch for {}.".format(label))
    for label in ("capture_validation", "view_a_unit_state", "view_c_unit_state"):
        registered_path = Path(str(paths.get(label, "")))
        if not registered_path.is_file() or registered.get(label) != sha256_file(registered_path):
            raise AnalysisError("Registered input hash mismatch for {}.".format(label))


def transform_c(point: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    transform = calibration["relative_transforms"]["a_from_c"]
    return geometry.apply_transform(transform, point.reshape(1, 3))[0]


def candidate_payload(
    *, method: str, sequence: int, joint: str, point: np.ndarray | None,
    selected_view: str, weight_a: float, weight_c: float,
    point_a: np.ndarray | None, point_c: np.ndarray | None,
    status_a: str, status_c: str, visibility_a: float, visibility_c: float,
    score_a: tuple[int, int, float], score_c: tuple[int, int, float], provenance: str,
) -> dict[str, Any]:
    return {
        "sequence_index": sequence,
        "animation_frame_code": sequence,
        "usd_time_code": sequence,
        "phase": "active" if sequence in ACTIVE_FRAMES else "inactive",
        "canonical_joint": joint,
        "method": method,
        "valid": int(point is not None),
        "common_x_m": float(point[0]) if point is not None else "",
        "common_y_m": float(point[1]) if point is not None else "",
        "common_z_m": float(point[2]) if point is not None else "",
        "selected_view": selected_view,
        "weight_a": weight_a,
        "weight_c": weight_c,
        "view_a_valid": int(point_a is not None),
        "view_c_valid": int(point_c is not None),
        "view_a_reliability": status_a,
        "view_c_reliability": status_c,
        "view_a_visibility": visibility_a,
        "view_c_visibility": visibility_c,
        "selection_score_a": json.dumps(score_a, separators=(",", ":")),
        "selection_score_c": json.dumps(score_c, separators=(",", ":")),
        "provenance": provenance,
        "classification": CLASSIFICATION,
        "claim_eligibility": CLAIM_ELIGIBILITY,
        "eligible_for_formal": 0,
        "calibration_source": CALIBRATION_SOURCE,
    }


def build_candidates(
    replay_a: dict[str, dict[tuple[int, str], dict[str, str]]],
    replay_c: dict[str, dict[tuple[int, str], dict[str, str]]],
    gt_a: dict[tuple[int, str], dict[str, str]],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sequence in range(EXPECTED_FRAMES):
        score_a = arm_score(replay_a, sequence)
        score_c = arm_score(replay_c, sequence)
        arm_complete_a = all(row_point(replay_a["k4_inferred"][(sequence, joint)]) is not None for joint in ARM_JOINTS)
        arm_complete_c = all(row_point(replay_c["k4_inferred"][(sequence, joint)]) is not None for joint in ARM_JOINTS)
        reliability_selected = select_reliability_view(
            score_a, score_c, arm_complete_a, arm_complete_c
        )
        if arm_complete_a and arm_complete_c:
            errors = {}
            for view_id, replay in (("a", replay_a), ("c", replay_c)):
                joint_errors = []
                for joint in ARM_JOINTS:
                    point = row_point(replay["k4_inferred"][(sequence, joint)])
                    if view_id == "c" and point is not None:
                        point = transform_c(point, calibration)
                    target = np.asarray([finite(gt_a[(sequence, joint)][field], field) for field in ("gt_x_m", "gt_y_m", "gt_z_m")])
                    joint_errors.append(float(np.linalg.norm(point - target)))
                errors[view_id] = statistics.fmean(joint_errors)
            oracle_selected = "c" if errors["c"] < errors["a"] else "a"
        elif arm_complete_a:
            oracle_selected = "a"
        elif arm_complete_c:
            oracle_selected = "c"
        else:
            oracle_selected = ""

        for joint in ARM_JOINTS:
            row_a = replay_a["k4_inferred"][(sequence, joint)]
            row_c = replay_c["k4_inferred"][(sequence, joint)]
            point_a = row_point(row_a)
            point_c_camera = row_point(row_c)
            point_c = transform_c(point_c_camera, calibration) if point_c_camera is not None else None
            vis_a, vis_c = visibility(row_a), visibility(row_c)
            status_a = raw_reliability(replay_a["raw_measured"], sequence, joint)
            status_c = raw_reliability(replay_c["raw_measured"], sequence, joint)

            method_values: dict[str, tuple[np.ndarray | None, str, float, float, str]] = {
                "view_a_k4": (point_a, "a" if point_a is not None else "", 1.0 if point_a is not None else 0.0, 0.0, "single_view_a_k4"),
                "view_c_k4": (point_c, "c" if point_c is not None else "", 0.0, 1.0 if point_c is not None else 0.0, "single_view_c_k4"),
            }
            if point_a is not None and point_c is not None:
                mean_point = (point_a + point_c) / 2.0
                total_visibility = vis_a + vis_c
                if total_visibility > 0.0:
                    weight_a, weight_c = vis_a / total_visibility, vis_c / total_visibility
                else:
                    weight_a = weight_c = 0.5
                visibility_point = weight_a * point_a + weight_c * point_c
            elif point_a is not None:
                mean_point = visibility_point = point_a
                weight_a, weight_c = 1.0, 0.0
            elif point_c is not None:
                mean_point = visibility_point = point_c
                weight_a, weight_c = 0.0, 1.0
            else:
                mean_point = visibility_point = None
                weight_a = weight_c = 0.0
            method_values["mean_fusion_available"] = (
                mean_point, "a+c" if point_a is not None and point_c is not None else ("a" if point_a is not None else ("c" if point_c is not None else "")),
                0.5 if point_a is not None and point_c is not None else weight_a,
                0.5 if point_a is not None and point_c is not None else weight_c,
                "available_mean_fusion_k4",
            )
            method_values["visibility_weighted_fusion_available"] = (
                visibility_point, "a+c" if point_a is not None and point_c is not None else ("a" if point_a is not None else ("c" if point_c is not None else "")),
                weight_a, weight_c, "visibility_weighted_available_fusion_k4",
            )
            selected_point = point_a if reliability_selected == "a" else (point_c if reliability_selected == "c" else None)
            method_values["reliability_best_view"] = (
                selected_point, reliability_selected,
                1.0 if reliability_selected == "a" else 0.0,
                1.0 if reliability_selected == "c" else 0.0,
                "gt_free_arm_level_reliability_selector",
            )
            oracle_point = point_a if oracle_selected == "a" else (point_c if oracle_selected == "c" else None)
            method_values["oracle_best_view_arm"] = (
                oracle_point, oracle_selected,
                1.0 if oracle_selected == "a" else 0.0,
                1.0 if oracle_selected == "c" else 0.0,
                "evaluation_gt_oracle_arm_selector",
            )
            for method in METHODS:
                point, selected, wa, wc, provenance = method_values[method]
                output.append(candidate_payload(
                    method=method, sequence=sequence, joint=joint, point=point,
                    selected_view=selected, weight_a=wa, weight_c=wc,
                    point_a=point_a, point_c=point_c, status_a=status_a, status_c=status_c,
                    visibility_a=vis_a, visibility_c=vis_c, score_a=score_a, score_c=score_c,
                    provenance=provenance,
                ))
    return output


def phase_metrics(
    rows: list[dict[str, Any]], gt_a: dict[tuple[int, str], dict[str, str]], method: str, active: bool
) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method and ((row["sequence_index"] in ACTIVE_FRAMES) == active)]
    errors = []
    complete_frames = 0
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        by_frame.setdefault(int(row["sequence_index"]), []).append(row)
        if int(row["valid"]) == 1:
            point = np.asarray([float(row[field]) for field in ("common_x_m", "common_y_m", "common_z_m")])
            target_row = gt_a[(int(row["sequence_index"]), str(row["canonical_joint"]))]
            target = np.asarray([finite(target_row[field], field) for field in ("gt_x_m", "gt_y_m", "gt_z_m")])
            errors.append(float(np.linalg.norm(point - target)) * 1000.0)
    complete_frames = sum(all(int(row["valid"]) == 1 for row in frame_rows) for frame_rows in by_frame.values())
    view_counts = {label: sum(row["selected_view"] == label for row in selected) for label in ("a", "c", "a+c", "")}
    return {
        "expected_joint_samples": len(selected),
        "valid_joint_samples": len(errors),
        "coverage": len(errors) / len(selected),
        "complete_arm_frame_coverage": complete_frames / len(by_frame),
        "conditional_mean_error_mm": statistics.fmean(errors) if errors else None,
        "conditional_p95_error_mm": float(np.percentile(errors, 95.0)) if errors else None,
        "selected_view_joint_counts": view_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-a-dir", type=Path, required=True)
    parser.add_argument("--view-c-dir", type=Path, required=True)
    parser.add_argument("--view-a-replay", type=Path, required=True)
    parser.add_argument("--view-c-replay", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise AnalysisError("Refusing to overwrite analysis output: {}".format(output_dir))
    protocol_path = args.analysis_protocol.resolve()
    view_a_dir = args.view_a_dir.resolve()
    view_c_dir = args.view_c_dir.resolve()
    view_a_replay = args.view_a_replay.resolve()
    view_c_replay = args.view_c_replay.resolve()
    protocol = read_json(protocol_path)
    validate_analysis_protocol(
        protocol,
        protocol_path,
        view_a_dir=view_a_dir,
        view_c_dir=view_c_dir,
        view_a_replay=view_a_replay,
        view_c_replay=view_c_replay,
        output_dir=output_dir,
    )
    run_a, camera_a, gt_a, _ = load_ground_truth(view_a_dir, "a")
    run_c, camera_c, gt_c, _ = load_ground_truth(view_c_dir, "c")
    calibration = build_calibration(gt_a, gt_c, run_a, run_c, camera_a, camera_c)
    replay_a = load_replay(view_a_replay, run_a)
    replay_c = load_replay(view_c_replay, run_c)
    candidates = build_candidates(replay_a, replay_c, gt_a, calibration)
    expected_rows = EXPECTED_FRAMES * len(ARM_JOINTS) * len(METHODS)
    if len(candidates) != expected_rows:
        raise AnalysisError("Candidate row count mismatch.")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "classification": CLASSIFICATION,
        "claim_eligibility": CLAIM_ELIGIBILITY,
        "eligible_for_formal": False,
        "same_capture_session": True,
        "statistical_unit": "single_engineering_pilot_capture",
        "repeat_count": 1,
        "calibration_source": CALIBRATION_SOURCE,
        "uses_ground_truth_for_calibration": True,
        "occlusion_depth_margin_m": OCCLUSION_DEPTH_MARGIN_M,
        "methods": list(METHODS),
        "primary_endpoint_name": "right_arm_two_joint_mean_position_error_mm",
        "calibration": {
            "baseline_m": calibration["baseline_m"],
            "maximum_cross_view_fit_residual_m": calibration["maximum_cross_view_fit_residual_m"],
        },
        "metrics": {
            method: {
                "active": phase_metrics(candidates, gt_a, method, True),
                "inactive": phase_metrics(candidates, gt_a, method, False),
            }
            for method in METHODS
        },
        "interpretation_restrictions": [
            "Excluded one-capture engineering pilot; no repeatability claim.",
            "GT-derived common-frame calibration is not deployable or formal-eligible.",
            "Oracle best view is evaluation-only and not an estimator.",
            "All method-performance numbers are diagnostic until a later formal protocol and independent repeats.",
        ],
        "inputs": {
            "analysis_protocol": str(args.analysis_protocol.resolve()),
            "analysis_protocol_sha256": sha256_file(args.analysis_protocol.resolve()),
            "view_a_dir": str(args.view_a_dir.resolve()),
            "view_c_dir": str(args.view_c_dir.resolve()),
            "view_a_replay": str(args.view_a_replay.resolve()),
            "view_c_replay": str(args.view_c_replay.resolve()),
        },
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    calibration_path = output_dir / "gt_derived_engineering_calibration.json"
    candidates_path = output_dir / "view_ac_best_view_fusion_candidates.csv"
    summary_path = output_dir / "view_ac_best_view_fusion_summary.json"
    write_json_exclusive(calibration_path, calibration)
    write_csv_exclusive(candidates_path, candidates)
    summary["outputs"] = {
        "calibration_sha256": sha256_file(calibration_path),
        "candidates_sha256": sha256_file(candidates_path),
    }
    write_json_exclusive(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
