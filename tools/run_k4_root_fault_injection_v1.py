"""Run the frozen post-sampling K4 shoulder-root fault-injection study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import kinematic_k3_inferred as k3
import kinematic_k4_inferred as k4


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "k4_root_fault_injection_v1"
    / "fault_injection_protocol.json"
)
ESTIMATES_NAME = "fault_injection_estimates.csv"
RUN_SUMMARY_NAME = "fault_injection_run_summary.csv"
SUMMARY_NAME = "fault_injection_summary.json"
STATE_NAME = "fault_injection_analysis_state.json"
ALGORITHM_VERSION = "k4_root_fault_injection_v1_20260729"
JOINTS = ("right_elbow", "right_wrist")
METHODS = ("k3_inferred", "k4_inferred")
RECOVERY_THRESHOLD_M = 0.001
RECOVERY_SEARCH_FRAMES = 10
ESTIMATE_FIELDS = (
    "run_id",
    "scenario",
    "sequence_index",
    "animation_frame_code",
    "usd_time_code",
    "fault_active",
    "method",
    "canonical_joint",
    "valid",
    "x_m",
    "y_m",
    "z_m",
    "provenance",
    "invalid_reason",
    "root_branch",
    "source_shoulder_depth_m",
    "injected_shoulder_depth_m",
    "guarded_shoulder_depth_m",
    "shoulder_root_clipped",
    "shoulder_root_reason",
)
RUN_SUMMARY_FIELDS = (
    "run_id",
    "scenario",
    "method",
    "joint_sample_coverage",
    "fault_joint_sample_coverage",
    "clean_displacement_mean_mm",
    "clean_displacement_p95_mm",
    "clean_displacement_max_mm",
    "fault_clean_displacement_mean_mm",
    "fault_clean_displacement_p95_mm",
    "fault_clean_displacement_max_mm",
    "gt_error_mean_mm",
    "fault_gt_error_mean_mm",
    "root_clip_frame_count",
    "recovery_frames_mean",
    "recovery_frames_max",
)


class ContractError(RuntimeError):
    """Raised when a frozen input or output contract is violated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ContractError(f"CSV input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ContractError(f"CSV input is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def write_json_replace(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def resolve_workspace_path(value: str) -> Path:
    path = (WORKSPACE / value).resolve()
    try:
        path.relative_to(WORKSPACE.resolve())
    except ValueError as error:
        raise ContractError(
            f"Frozen path escapes the workspace: {value}"
        ) from error
    return path


def validate_protocol(path: Path) -> dict:
    if not path.is_file():
        raise ContractError(f"Frozen protocol does not exist: {path}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise ContractError("Fault-injection protocol is not frozen.")
    expected = protocol.get("protocol_sha256")
    unhashed = dict(protocol)
    unhashed.pop("protocol_sha256", None)
    observed = canonical_payload_sha256(unhashed)
    if expected != observed:
        raise ContractError(
            f"Protocol integrity mismatch: {expected} versus {observed}"
        )
    if protocol.get("methods") != list(METHODS):
        raise ContractError("Frozen method inventory differs from runner.")
    if protocol.get("reporting_joints") != list(JOINTS):
        raise ContractError("Frozen reporting joints differ from runner.")
    if not math.isclose(
        float(protocol["k4_shoulder_rate_limit_m"]),
        k4.DEFAULT_SHOULDER_RATE_LIMIT_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError("Frozen K4 shoulder rate limit differs.")
    return protocol


def verify_frozen_file(entry: dict) -> Path:
    path = resolve_workspace_path(entry["path"])
    observed = sha256_file(path)
    if observed != entry["sha256"]:
        raise ContractError(
            f"Frozen input hash mismatch for {path}: "
            f"{entry['sha256']} versus {observed}"
        )
    return path


def optional_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"Expected finite value; observed {value!r}.")
    return number


def point_from_result(result: dict) -> np.ndarray | None:
    if not result["valid"] or result.get("point") is None:
        return None
    return np.asarray(result["point"], dtype=np.float64)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def sample_sd_or_none(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def candidate_from_raw(row: dict, mapping: dict) -> dict:
    pixel_x = str(row.get("pixel_x", "")).strip()
    pixel_y = str(row.get("pixel_y", "")).strip()
    visibility = optional_float(row.get("visibility")) or 0.0
    return {
        "mapping": mapping,
        "in_image": pixel_x != "" and pixel_y != "",
        "visibility": visibility,
        "pixel_x": int(pixel_x) if pixel_x != "" else 0,
        "pixel_y": int(pixel_y) if pixel_y != "" else 0,
        "depth_m": optional_float(row.get("depth_m")),
    }


def apply_fault(
    candidates: list[dict],
    scenario: dict,
    sequence_index: int,
) -> tuple[bool, float | None, float | None]:
    shoulder = next(
        candidate
        for candidate in candidates
        if candidate["mapping"]["canonical_joint"] == "right_shoulder"
    )
    source_depth = shoulder.get("depth_m")
    fault_active = sequence_index in {
        int(value) for value in scenario["fault_frames"]
    }
    if not fault_active:
        return False, source_depth, source_depth
    if scenario["fault_type"] == "additive_depth_impulse":
        if source_depth is None:
            raise ContractError(
                "Cannot add a frozen shoulder-depth impulse to missing depth."
            )
        shoulder["depth_m"] = float(source_depth) + float(
            scenario["shoulder_depth_offset_m"]
        )
    elif scenario["fault_type"] == "missing_depth_impulse":
        shoulder["depth_m"] = None
    elif scenario["fault_type"] == "missing_landmark_impulse":
        shoulder["in_image"] = False
        shoulder["visibility"] = 0.0
        shoulder["depth_m"] = None
    else:
        raise ContractError(
            f"Unsupported frozen fault type: {scenario['fault_type']}"
        )
    return True, source_depth, shoulder.get("depth_m")


def build_estimate_row(
    run_id: str,
    scenario_name: str,
    manifest: dict,
    fault_active: bool,
    method: str,
    result: dict,
    source_shoulder_depth_m: float | None,
    injected_shoulder_depth_m: float | None,
    root: dict | None,
) -> dict:
    point = point_from_result(result)
    valid = point is not None
    return {
        "run_id": run_id,
        "scenario": scenario_name,
        "sequence_index": int(manifest["sequence_index"]),
        "animation_frame_code": int(manifest["animation_frame_code"]),
        "usd_time_code": float(manifest["usd_time_code"]),
        "fault_active": int(fault_active),
        "method": method,
        "canonical_joint": result["joint"],
        "valid": int(valid),
        "x_m": float(point[0]) if valid else "",
        "y_m": float(point[1]) if valid else "",
        "z_m": float(point[2]) if valid else "",
        "provenance": result["provenance"],
        "invalid_reason": result["invalid_reason"],
        "root_branch": result["root_branch"],
        "source_shoulder_depth_m": (
            source_shoulder_depth_m
            if source_shoulder_depth_m is not None
            else ""
        ),
        "injected_shoulder_depth_m": (
            injected_shoulder_depth_m
            if injected_shoulder_depth_m is not None
            else ""
        ),
        "guarded_shoulder_depth_m": (
            root["guarded_shoulder_depth_m"]
            if root is not None
            and root["guarded_shoulder_depth_m"] is not None
            else ""
        ),
        "shoulder_root_clipped": (
            int(root["shoulder_root_clipped"])
            if root is not None
            else ""
        ),
        "shoulder_root_reason": (
            root["shoulder_root_reason"] if root is not None else ""
        ),
    }


def load_run_inputs(
    protocol: dict,
    run_id: str,
    mapping_rows: list[dict],
) -> tuple[list[dict], dict[int, list[dict]], dict]:
    frozen = protocol["frozen_inputs"]["runs"][run_id]
    manifest_path = verify_frozen_file(frozen["manifest"])
    source_path = verify_frozen_file(frozen["source_estimates"])
    verify_frozen_file(frozen["landmark_cache_state"])
    verify_frozen_file(frozen["source_replay_state"])
    manifest_rows = read_csv(manifest_path)
    if [int(row["sequence_index"]) for row in manifest_rows] != list(
        range(240)
    ):
        raise ContractError(f"{run_id} manifest must contain sequence 0..239.")
    source_rows = read_csv(source_path)
    raw_by_sequence: dict[int, list[dict]] = defaultdict(list)
    original_inferred = {}
    for row in source_rows:
        sequence_index = int(row["sequence_index"])
        if row["method"] == "raw_measured":
            raw_by_sequence[sequence_index].append(row)
        elif row["method"] in METHODS:
            key = (
                row["method"],
                sequence_index,
                row["canonical_joint"],
            )
            if key in original_inferred:
                raise ContractError(f"Duplicate original estimate key: {key}")
            original_inferred[key] = row
    expected_joints = {
        row["canonical_joint"] for row in mapping_rows
    }
    for sequence_index in range(240):
        observed = {
            row["canonical_joint"]
            for row in raw_by_sequence[sequence_index]
        }
        if observed != expected_joints:
            raise ContractError(
                f"{run_id} sequence {sequence_index} raw joint set differs."
            )
    return manifest_rows, raw_by_sequence, original_inferred


def run_estimation(protocol: dict) -> tuple[list[dict], dict]:
    profile_path = verify_frozen_file(
        protocol["frozen_inputs"]["profile"]
    )
    config_path = verify_frozen_file(protocol["frozen_inputs"]["config"])
    mapping_path = verify_frozen_file(
        protocol["frozen_inputs"]["joint_mapping"]
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mapping_rows = read_csv(mapping_path)
    mapping_by_joint = {
        row["canonical_joint"]: row for row in mapping_rows
    }
    if len(mapping_by_joint) != 15:
        raise ContractError("Expected exactly 15 unique joint mappings.")

    estimate_rows: list[dict] = []
    clean_reproduction_max_difference_m = 0.0
    clean_reproduction_validity_mismatch_count = 0
    applied_fault_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for run_id in protocol["run_ids"]:
        manifest_rows, raw_by_sequence, original = load_run_inputs(
            protocol,
            run_id,
            mapping_rows,
        )
        for scenario in protocol["scenarios"]:
            root_guard = k4.ShoulderRootGuard(
                rate_limit_m=float(
                    protocol["k4_shoulder_rate_limit_m"]
                )
            )
            for manifest in manifest_rows:
                sequence_index = int(manifest["sequence_index"])
                candidates = [
                    candidate_from_raw(
                        row,
                        mapping_by_joint[row["canonical_joint"]],
                    )
                    for row in raw_by_sequence[sequence_index]
                ]
                fault_active, source_depth, injected_depth = apply_fault(
                    candidates,
                    scenario,
                    sequence_index,
                )
                if fault_active:
                    applied_fault_counts[run_id][scenario["name"]] += 1
                intrinsics = (
                    float(manifest["fx"]),
                    float(manifest["fy"]),
                    float(manifest["cx"]),
                    float(manifest["cy"]),
                )
                k3_results = k3.infer_right_arm(
                    candidates,
                    profile,
                    config,
                    intrinsics,
                    elbow_branch="far",
                    wrist_branch="far",
                )
                k4_results, root = k4.infer_right_arm(
                    candidates,
                    profile,
                    config,
                    intrinsics,
                    root_guard,
                    elbow_branch="far",
                    wrist_branch="far",
                )
                for method, results, metadata in (
                    ("k3_inferred", k3_results, None),
                    ("k4_inferred", k4_results, root),
                ):
                    for result in results:
                        row = build_estimate_row(
                            run_id,
                            scenario["name"],
                            manifest,
                            fault_active,
                            method,
                            result,
                            source_depth,
                            injected_depth,
                            metadata,
                        )
                        estimate_rows.append(row)
                        if scenario["name"] != "clean_control":
                            continue
                        key = (
                            method,
                            sequence_index,
                            result["joint"],
                        )
                        source = original.get(key)
                        if source is None:
                            raise ContractError(
                                f"Missing completed clean estimate: {key}"
                            )
                        source_valid = str(source["valid"]).strip() == "1"
                        current_valid = bool(row["valid"])
                        if source_valid != current_valid:
                            clean_reproduction_validity_mismatch_count += 1
                        elif current_valid:
                            source_point = np.asarray(
                                [
                                    float(source["x_m"]),
                                    float(source["y_m"]),
                                    float(source["z_m"]),
                                ],
                                dtype=np.float64,
                            )
                            current_point = np.asarray(
                                [row["x_m"], row["y_m"], row["z_m"]],
                                dtype=np.float64,
                            )
                            clean_reproduction_max_difference_m = max(
                                clean_reproduction_max_difference_m,
                                float(
                                    np.linalg.norm(
                                        current_point - source_point
                                    )
                                ),
                            )
    expected_rows = (
        len(protocol["run_ids"])
        * len(protocol["scenarios"])
        * 240
        * len(METHODS)
        * len(JOINTS)
    )
    if len(estimate_rows) != expected_rows:
        raise ContractError(
            f"Estimate row count mismatch: {len(estimate_rows)} versus "
            f"{expected_rows}."
        )
    for run_id in protocol["run_ids"]:
        for scenario in protocol["scenarios"]:
            expected = (
                0
                if scenario["name"] == "clean_control"
                else len(scenario["fault_frames"])
            )
            observed = applied_fault_counts[run_id][scenario["name"]]
            if observed != expected:
                raise ContractError(
                    f"{run_id}/{scenario['name']} applied {observed} "
                    f"faults; expected {expected}."
                )
    reproduction = {
        "validity_mismatch_count": (
            clean_reproduction_validity_mismatch_count
        ),
        "maximum_xyz_difference_m": (
            clean_reproduction_max_difference_m
        ),
        "passed": (
            clean_reproduction_validity_mismatch_count == 0
            and clean_reproduction_max_difference_m <= 1e-12
        ),
    }
    if not reproduction["passed"]:
        raise ContractError(
            "Clean replay did not exactly reproduce completed K3/K4 output: "
            f"{reproduction}"
        )
    return estimate_rows, reproduction


def row_point(row: dict) -> np.ndarray | None:
    if not bool(int(row["valid"])):
        return None
    return np.asarray(
        [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
        dtype=np.float64,
    )


def recovery_frames(
    by_key: dict,
    clean_by_key: dict,
    run_id: str,
    scenario: str,
    method: str,
    fault_frame: int,
) -> int | None:
    for offset in range(1, RECOVERY_SEARCH_FRAMES + 1):
        sequence = fault_frame + offset
        if sequence >= 240:
            break
        distances = []
        for joint in JOINTS:
            fault_point = row_point(
                by_key[(run_id, scenario, method, sequence, joint)]
            )
            clean_point = row_point(
                clean_by_key[(run_id, method, sequence, joint)]
            )
            if fault_point is None or clean_point is None:
                distances = []
                break
            distances.append(
                float(np.linalg.norm(fault_point - clean_point))
            )
        if distances and max(distances) <= RECOVERY_THRESHOLD_M:
            return offset
    return None


def load_ground_truth(protocol: dict) -> dict:
    """Load GT only after estimation artifacts have been written."""
    ground_truth = {}
    for run_id in protocol["run_ids"]:
        entry = protocol["frozen_inputs"]["runs"][run_id]["ground_truth"]
        path = verify_frozen_file(entry)
        for row in read_csv(path):
            if row["canonical_joint"] not in JOINTS:
                continue
            key = (
                run_id,
                int(row["sequence_index"]),
                row["canonical_joint"],
            )
            if key in ground_truth:
                raise ContractError(f"Duplicate GT key: {key}")
            if row.get("gt_semantics") != "synchronized_per_frame":
                raise ContractError(f"Unsynchronized GT semantics: {key}")
            ground_truth[key] = np.asarray(
                [
                    float(row["gt_x_m"]),
                    float(row["gt_y_m"]),
                    float(row["gt_z_m"]),
                ],
                dtype=np.float64,
            )
    expected = len(protocol["run_ids"]) * 240 * len(JOINTS)
    if len(ground_truth) != expected:
        raise ContractError(
            f"GT key count mismatch: {len(ground_truth)} versus {expected}."
        )
    return ground_truth


def analyse(
    protocol: dict,
    estimate_rows: list[dict],
) -> tuple[list[dict], dict]:
    ground_truth = load_ground_truth(protocol)
    by_key = {}
    clean_by_key = {}
    roots = {}
    for row in estimate_rows:
        key = (
            row["run_id"],
            row["scenario"],
            row["method"],
            int(row["sequence_index"]),
            row["canonical_joint"],
        )
        if key in by_key:
            raise ContractError(f"Duplicate fault estimate key: {key}")
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
        if row["method"] == "k4_inferred":
            root_key = (
                row["run_id"],
                row["scenario"],
                int(row["sequence_index"]),
            )
            roots[root_key] = {
                "clipped": bool(int(row["shoulder_root_clipped"])),
                "reason": row["shoulder_root_reason"],
            }

    run_rows = []
    paired_run_metrics = {}
    scenario_by_name = {
        scenario["name"]: scenario for scenario in protocol["scenarios"]
    }
    for run_id in protocol["run_ids"]:
        for scenario_name, scenario in scenario_by_name.items():
            fault_frames = {int(value) for value in scenario["fault_frames"]}
            for method in METHODS:
                rows = [
                    by_key[
                        (
                            run_id,
                            scenario_name,
                            method,
                            sequence,
                            joint,
                        )
                    ]
                    for sequence in range(240)
                    for joint in JOINTS
                ]
                valid_rows = [row for row in rows if row_point(row) is not None]
                fault_rows = [
                    row
                    for row in rows
                    if int(row["sequence_index"]) in fault_frames
                ]
                valid_fault_rows = [
                    row for row in fault_rows if row_point(row) is not None
                ]
                displacements = []
                fault_displacements = []
                gt_errors = []
                fault_gt_errors = []
                for row in valid_rows:
                    sequence = int(row["sequence_index"])
                    joint = row["canonical_joint"]
                    point = row_point(row)
                    clean = row_point(
                        clean_by_key[(run_id, method, sequence, joint)]
                    )
                    if clean is not None:
                        displacement = float(np.linalg.norm(point - clean))
                        displacements.append(displacement)
                        if sequence in fault_frames:
                            fault_displacements.append(displacement)
                    gt_error = float(
                        np.linalg.norm(
                            point - ground_truth[(run_id, sequence, joint)]
                        )
                    )
                    gt_errors.append(gt_error)
                    if sequence in fault_frames:
                        fault_gt_errors.append(gt_error)
                recoveries = [
                    recovery_frames(
                        by_key,
                        clean_by_key,
                        run_id,
                        scenario_name,
                        method,
                        frame,
                    )
                    for frame in sorted(fault_frames)
                ]
                observed_recoveries = [
                    value for value in recoveries if value is not None
                ]
                root_clip_count = (
                    sum(
                        roots[(run_id, scenario_name, sequence)]["clipped"]
                        for sequence in range(240)
                    )
                    if method == "k4_inferred"
                    else 0
                )
                run_rows.append(
                    {
                        "run_id": run_id,
                        "scenario": scenario_name,
                        "method": method,
                        "joint_sample_coverage": (
                            len(valid_rows) / len(rows)
                        ),
                        "fault_joint_sample_coverage": (
                            len(valid_fault_rows) / len(fault_rows)
                            if fault_rows
                            else 1.0
                        ),
                        "clean_displacement_mean_mm": (
                            mean_or_none(displacements) * 1000.0
                            if displacements
                            else None
                        ),
                        "clean_displacement_p95_mm": (
                            percentile(displacements, 95.0) * 1000.0
                            if displacements
                            else None
                        ),
                        "clean_displacement_max_mm": (
                            max(displacements) * 1000.0
                            if displacements
                            else None
                        ),
                        "fault_clean_displacement_mean_mm": (
                            mean_or_none(fault_displacements) * 1000.0
                            if fault_displacements
                            else None
                        ),
                        "fault_clean_displacement_p95_mm": (
                            percentile(fault_displacements, 95.0) * 1000.0
                            if fault_displacements
                            else None
                        ),
                        "fault_clean_displacement_max_mm": (
                            max(fault_displacements) * 1000.0
                            if fault_displacements
                            else None
                        ),
                        "gt_error_mean_mm": (
                            mean_or_none(gt_errors) * 1000.0
                        ),
                        "fault_gt_error_mean_mm": (
                            mean_or_none(fault_gt_errors) * 1000.0
                            if fault_gt_errors
                            else None
                        ),
                        "root_clip_frame_count": root_clip_count,
                        "recovery_frames_mean": (
                            mean_or_none(observed_recoveries)
                            if observed_recoveries
                            else None
                        ),
                        "recovery_frames_max": (
                            max(observed_recoveries)
                            if observed_recoveries
                            else None
                        ),
                    }
                )

            paired_gt_deltas = []
            for sequence in sorted(fault_frames):
                method_errors = {}
                for method in METHODS:
                    errors = []
                    for joint in JOINTS:
                        point = row_point(
                            by_key[
                                (
                                    run_id,
                                    scenario_name,
                                    method,
                                    sequence,
                                    joint,
                                )
                            ]
                        )
                        if point is None:
                            errors = []
                            break
                        errors.append(
                            float(
                                np.linalg.norm(
                                    point
                                    - ground_truth[
                                        (run_id, sequence, joint)
                                    ]
                                )
                            )
                        )
                    if errors:
                        method_errors[method] = statistics.fmean(errors)
                if set(method_errors) == set(METHODS):
                    paired_gt_deltas.append(
                        (
                            method_errors["k4_inferred"]
                            - method_errors["k3_inferred"]
                        )
                        * 1000.0
                    )
            method_rows = {
                row["method"]: row
                for row in run_rows
                if row["run_id"] == run_id
                and row["scenario"] == scenario_name
            }
            k3_displacement = method_rows["k3_inferred"][
                "fault_clean_displacement_mean_mm"
            ]
            k4_displacement = method_rows["k4_inferred"][
                "fault_clean_displacement_mean_mm"
            ]
            attenuation = (
                1.0 - (k4_displacement / k3_displacement)
                if k3_displacement is not None
                and k4_displacement is not None
                and k3_displacement > 0.0
                else None
            )
            paired_run_metrics[(run_id, scenario_name)] = {
                "paired_fault_frame_count": len(paired_gt_deltas),
                "k4_minus_k3_fault_gt_error_mean_mm": (
                    mean_or_none(paired_gt_deltas)
                ),
                "k4_transmission_attenuation_fraction": attenuation,
            }

    scenario_summary = {}
    for scenario in protocol["scenarios"]:
        name = scenario["name"]
        method_summary = {}
        for method in METHODS:
            selected = [
                row
                for row in run_rows
                if row["scenario"] == name and row["method"] == method
            ]
            metrics = {}
            for field in RUN_SUMMARY_FIELDS:
                if field in {"run_id", "scenario", "method"}:
                    continue
                values = [
                    float(row[field])
                    for row in selected
                    if row[field] is not None
                ]
                metrics[field] = {
                    "run_values": values,
                    "mean": mean_or_none(values),
                    "sample_sd": sample_sd_or_none(values),
                }
            method_summary[method] = metrics
        paired = [
            paired_run_metrics[(run_id, name)]
            for run_id in protocol["run_ids"]
        ]
        paired_summary = {}
        for field in (
            "paired_fault_frame_count",
            "k4_minus_k3_fault_gt_error_mean_mm",
            "k4_transmission_attenuation_fraction",
        ):
            values = [
                float(row[field])
                for row in paired
                if row[field] is not None
            ]
            paired_summary[field] = {
                "run_values": values,
                "mean": mean_or_none(values),
                "sample_sd": sample_sd_or_none(values),
            }
        scenario_summary[name] = {
            "fault_type": scenario["fault_type"],
            "fault_frames": scenario["fault_frames"],
            "methods": method_summary,
            "paired_k4_vs_k3": paired_summary,
        }

    positive = scenario_summary["positive_impulse_150mm"]
    negative = scenario_summary["negative_impulse_150mm"]
    missing = scenario_summary["missing_root_impulse"]
    clean = scenario_summary["clean_control"]
    mechanical_checks = {
        "clean_k4_clip_count_is_zero": (
            clean["methods"]["k4_inferred"]["root_clip_frame_count"][
                "mean"
            ]
            == 0.0
        ),
        "positive_k4_attenuates_mean_displacement": (
            positive["paired_k4_vs_k3"][
                "k4_transmission_attenuation_fraction"
            ]["mean"]
            is not None
            and positive["paired_k4_vs_k3"][
                "k4_transmission_attenuation_fraction"
            ]["mean"]
            > 0.0
        ),
        "negative_k4_attenuates_mean_displacement": (
            negative["paired_k4_vs_k3"][
                "k4_transmission_attenuation_fraction"
            ]["mean"]
            is not None
            and negative["paired_k4_vs_k3"][
                "k4_transmission_attenuation_fraction"
            ]["mean"]
            > 0.0
        ),
        "missing_k4_fault_coverage_exceeds_k3": (
            missing["methods"]["k4_inferred"][
                "fault_joint_sample_coverage"
            ]["mean"]
            > missing["methods"]["k3_inferred"][
                "fault_joint_sample_coverage"
            ]["mean"]
        ),
    }
    mechanical_checks["all_passed"] = all(mechanical_checks.values())
    return run_rows, {
        "scenarios": scenario_summary,
        "mechanical_checks": mechanical_checks,
        "paired_run_metrics": [
            {
                "run_id": run_id,
                "scenario": scenario,
                **metrics,
            }
            for (run_id, scenario), metrics in paired_run_metrics.items()
        ],
    }


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(protocol_path)
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
        raise ContractError(
            "Refusing to overwrite fault-injection output(s): "
            + ", ".join(str(path) for path in collisions)
        )

    state = {
        "schema_version": 1,
        "status": "running",
        "algorithm_version": ALGORITHM_VERSION,
        "protocol": str(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "gt_read_during_estimation": False,
        "started_wall_time_ns": time.time_ns(),
    }
    write_json_new(state_path, state)
    try:
        estimates, reproduction = run_estimation(protocol)
        write_csv(estimates_path, estimates, ESTIMATE_FIELDS)
        state.update(
            {
                "status": "estimation_complete",
                "estimate_row_count": len(estimates),
                "estimates_sha256": sha256_file(estimates_path),
                "clean_reproduction": reproduction,
                "estimation_completed_wall_time_ns": time.time_ns(),
            }
        )
        write_json_replace(state_path, state)

        run_rows, analysis = analyse(protocol, estimates)
        write_csv(run_summary_path, run_rows, RUN_SUMMARY_FIELDS)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "algorithm_version": ALGORITHM_VERSION,
            "protocol_sha256": protocol["protocol_sha256"],
            "run_ids": protocol["run_ids"],
            "scenario_count": len(protocol["scenarios"]),
            "fault_frames_per_noncontrol_scenario": 3,
            "estimate_row_count": len(estimates),
            "gt_read_during_estimation": False,
            "clean_reproduction": reproduction,
            **analysis,
            "artifacts": {
                "estimates": str(estimates_path),
                "estimates_sha256": sha256_file(estimates_path),
                "run_summary": str(run_summary_path),
                "run_summary_sha256": sha256_file(run_summary_path),
            },
        }
        summary["summary_payload_sha256"] = canonical_payload_sha256(
            summary
        )
        write_json_new(summary_path, summary)
        state.update(
            {
                "status": "complete",
                "run_summary_sha256": sha256_file(run_summary_path),
                "summary_sha256": sha256_file(summary_path),
                "mechanical_checks": analysis["mechanical_checks"],
                "completed_wall_time_ns": time.time_ns(),
            }
        )
        write_json_replace(state_path, state)
        print(json.dumps(summary["mechanical_checks"], indent=2))
        print(f"Fault-injection summary: {summary_path}")
        return 0
    except BaseException as error:
        state.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_wall_time_ns": time.time_ns(),
            }
        )
        write_json_replace(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
