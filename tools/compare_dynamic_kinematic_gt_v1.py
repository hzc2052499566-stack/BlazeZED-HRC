"""Strict per-frame dynamic-GT comparison for right-arm kinematics.

The comparator deliberately does not support a time-averaged ground truth.
Ground-truth rows must declare ``gt_semantics=synchronized_per_frame`` and
are joined one-to-one with one selected estimate method by
``(sequence_index, canonical_joint)``.  No timestamp-nearest matching,
interpolation, frame averaging, or coordinate transform is performed.

Ground-truth CSV schema
-----------------------
Required columns::

    sequence_index,cycle_index,animation_frame_code,usd_time_code,
    canonical_joint,gt_x_m,gt_y_m,gt_z_m,gt_semantics

``gt_semantics`` must be exactly ``synchronized_per_frame`` on every row.

Unified estimate CSV schema
---------------------------
Required columns::

    run_id,sequence_index,cycle_index,animation_frame_code,usd_time_code,
    method,canonical_joint,valid,x_m,y_m,z_m,
    counts_as_measured_valid,provenance,warmup_excluded

Supported methods and their fixed reporting sets are:

* ``raw_measured``: right shoulder, elbow, and wrist;
* ``k2_guarded``: right shoulder, elbow, and wrist;
* ``k3_inferred``: right elbow and wrist only;
* ``k4_inferred``: right elbow and wrist only.

By default, sequence indices marked ``warmup_excluded=1`` by the selected
estimate method are removed from both inputs before the exact key-set check.
Use ``--include-warmup`` only when warm-up samples are intentionally part of
the registered evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


SCHEMA_VERSION = 1
GT_SEMANTICS = "synchronized_per_frame"
FRAME_FIELDS = (
    "cycle_index",
    "animation_frame_code",
    "usd_time_code",
)
GROUND_TRUTH_REQUIRED_FIELDS = frozenset(
    {
        "sequence_index",
        *FRAME_FIELDS,
        "canonical_joint",
        "gt_x_m",
        "gt_y_m",
        "gt_z_m",
        "gt_semantics",
    }
)
ESTIMATE_REQUIRED_FIELDS = frozenset(
    {
        "run_id",
        "sequence_index",
        *FRAME_FIELDS,
        "method",
        "canonical_joint",
        "valid",
        "x_m",
        "y_m",
        "z_m",
        "counts_as_measured_valid",
        "provenance",
        "warmup_excluded",
    }
)
METHOD_JOINTS = {
    "raw_measured": (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ),
    "k2_guarded": (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ),
    "k3_inferred": (
        "right_elbow",
        "right_wrist",
    ),
    "k4_inferred": (
        "right_elbow",
        "right_wrist",
    ),
}
METHOD_COUNTS_AS_MEASURED_VALID = {
    "raw_measured": True,
    "k2_guarded": True,
    "k3_inferred": False,
    "k4_inferred": False,
}
METHOD_BONES = {
    "raw_measured": {
        "right_upper_arm": ("right_shoulder", "right_elbow"),
        "right_forearm": ("right_elbow", "right_wrist"),
    },
    "k2_guarded": {
        "right_upper_arm": ("right_shoulder", "right_elbow"),
        "right_forearm": ("right_elbow", "right_wrist"),
    },
    "k3_inferred": {
        "right_forearm": ("right_elbow", "right_wrist"),
    },
    "k4_inferred": {
        "right_forearm": ("right_elbow", "right_wrist"),
    },
}


class ComparisonContractError(ValueError):
    """Raised when an input violates the exact dynamic comparison contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic_new(path: Path, payload: dict) -> None:
    """Atomically create a JSON report without replacing an existing path."""
    path = path.resolve()
    if path.exists():
        raise ComparisonContractError(
            f"Refusing to overwrite existing output JSON: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ComparisonContractError(
                f"Refusing to overwrite existing output JSON: {path}"
            )
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def read_csv(path: Path, required_fields: frozenset[str]) -> list[dict]:
    if not path.is_file():
        raise ComparisonContractError(f"Input CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required_fields - fields)
        if missing:
            raise ComparisonContractError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ComparisonContractError(f"Input CSV has no data rows: {path}")
    return rows


def parse_nonnegative_int(value: object, label: str) -> int:
    text = str(value).strip()
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise ComparisonContractError(
            f"{label} must be an integer; observed {value!r}."
        ) from exc
    if number < 0:
        raise ComparisonContractError(
            f"{label} must not be negative; observed {number}."
        )
    return number


def parse_decimal(value: object, label: str) -> Decimal:
    text = str(value).strip()
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ComparisonContractError(
            f"{label} must be a finite number; observed {value!r}."
        ) from exc
    if not number.is_finite():
        raise ComparisonContractError(
            f"{label} must be finite; observed {value!r}."
        )
    return number


def parse_finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonContractError(
            f"{label} must be a finite number; observed {value!r}."
        ) from exc
    if not math.isfinite(number):
        raise ComparisonContractError(
            f"{label} must be finite; observed {value!r}."
        )
    return number


def parse_binary(value: object, label: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise ComparisonContractError(
        f"{label} must be one of 0, 1, false, true; observed {value!r}."
    )


def frame_metadata(row: dict, source: str, row_number: int) -> tuple:
    return (
        parse_nonnegative_int(
            row["cycle_index"],
            f"{source} row {row_number} cycle_index",
        ),
        parse_decimal(
            row["animation_frame_code"],
            f"{source} row {row_number} animation_frame_code",
        ),
        parse_decimal(
            row["usd_time_code"],
            f"{source} row {row_number} usd_time_code",
        ),
    )


def point_distance(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
) -> float:
    return math.sqrt(
        sum(
            (coordinate_a - coordinate_b) ** 2
            for coordinate_a, coordinate_b in zip(point_a, point_b)
        )
    )


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def error_summary(errors_m: list[float]) -> dict:
    if not errors_m:
        return {
            "valid_error_count": 0,
            "mean_3d_error_mm": None,
            "rmse_3d_error_mm": None,
            "median_3d_error_mm": None,
            "p95_3d_error_mm": None,
            "maximum_3d_error_mm": None,
        }
    return {
        "valid_error_count": len(errors_m),
        "mean_3d_error_mm": (
            sum(errors_m) / len(errors_m) * 1000.0
        ),
        "rmse_3d_error_mm": (
            math.sqrt(
                sum(error**2 for error in errors_m) / len(errors_m)
            )
            * 1000.0
        ),
        "median_3d_error_mm": percentile(errors_m, 0.5) * 1000.0,
        "p95_3d_error_mm": percentile(errors_m, 0.95) * 1000.0,
        "maximum_3d_error_mm": max(errors_m) * 1000.0,
    }


def bone_error_summary(
    signed_errors_m: list[float],
    expected_frame_count: int,
) -> dict:
    absolute = [abs(value) for value in signed_errors_m]
    if not signed_errors_m:
        return {
            "expected_frame_count": expected_frame_count,
            "valid_frame_count": 0,
            "coverage_rate": 0.0 if expected_frame_count else None,
            "bias_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_absolute_error_mm": None,
            "maximum_absolute_error_mm": None,
        }
    return {
        "expected_frame_count": expected_frame_count,
        "valid_frame_count": len(signed_errors_m),
        "coverage_rate": (
            len(signed_errors_m) / expected_frame_count
            if expected_frame_count
            else None
        ),
        "bias_mm": sum(signed_errors_m) / len(signed_errors_m) * 1000.0,
        "mae_mm": sum(absolute) / len(absolute) * 1000.0,
        "rmse_mm": (
            math.sqrt(
                sum(value**2 for value in signed_errors_m)
                / len(signed_errors_m)
            )
            * 1000.0
        ),
        "p95_absolute_error_mm": percentile(absolute, 0.95) * 1000.0,
        "maximum_absolute_error_mm": max(absolute) * 1000.0,
    }


def validate_ground_truth(
    rows: list[dict],
) -> tuple[dict[tuple[int, str], dict], dict[int, tuple], set[str]]:
    indexed: dict[tuple[int, str], dict] = {}
    metadata_by_sequence: dict[int, tuple] = {}
    joints = set()
    semantics = set()
    for row_number, row in enumerate(rows, start=2):
        sequence_index = parse_nonnegative_int(
            row["sequence_index"],
            f"GT row {row_number} sequence_index",
        )
        joint = str(row["canonical_joint"]).strip()
        if not joint:
            raise ComparisonContractError(
                f"GT row {row_number} canonical_joint is blank."
            )
        semantic = str(row["gt_semantics"]).strip()
        semantics.add(semantic)
        if semantic != GT_SEMANTICS:
            raise ComparisonContractError(
                "Dynamic comparison refuses aggregated/time-mean GT: "
                f"GT row {row_number} has gt_semantics={semantic!r}; "
                f"required {GT_SEMANTICS!r}."
            )
        metadata = frame_metadata(row, "GT", row_number)
        existing_metadata = metadata_by_sequence.setdefault(
            sequence_index,
            metadata,
        )
        if existing_metadata != metadata:
            raise ComparisonContractError(
                "GT has inconsistent frame metadata within sequence_index "
                f"{sequence_index}."
            )
        key = (sequence_index, joint)
        if key in indexed:
            raise ComparisonContractError(
                f"Duplicate GT key {key!r}; one row per frame/joint is required."
            )
        indexed[key] = {
            "point": (
                parse_finite_float(
                    row["gt_x_m"],
                    f"GT row {row_number} gt_x_m",
                ),
                parse_finite_float(
                    row["gt_y_m"],
                    f"GT row {row_number} gt_y_m",
                ),
                parse_finite_float(
                    row["gt_z_m"],
                    f"GT row {row_number} gt_z_m",
                ),
            ),
            "metadata": metadata,
        }
        joints.add(joint)
    return indexed, metadata_by_sequence, semantics


def validate_estimates(
    rows: list[dict],
    selected_method: str,
) -> tuple[
    dict[tuple[int, str], dict],
    dict[int, tuple],
    dict[int, bool],
    set[str],
    set[str],
]:
    indexed: dict[tuple[int, str], dict] = {}
    metadata_by_sequence: dict[int, tuple] = {}
    warmup_by_sequence: dict[int, bool] = {}
    methods_present = set()
    run_ids = set()
    all_seen_keys = set()
    expected_measured_semantics = METHOD_COUNTS_AS_MEASURED_VALID[
        selected_method
    ]
    for row_number, row in enumerate(rows, start=2):
        method = str(row["method"]).strip()
        if not method:
            raise ComparisonContractError(
                f"Estimate row {row_number} method is blank."
            )
        methods_present.add(method)
        sequence_index = parse_nonnegative_int(
            row["sequence_index"],
            f"Estimate row {row_number} sequence_index",
        )
        joint = str(row["canonical_joint"]).strip()
        if not joint:
            raise ComparisonContractError(
                f"Estimate row {row_number} canonical_joint is blank."
            )
        global_key = (method, sequence_index, joint)
        if global_key in all_seen_keys:
            raise ComparisonContractError(
                f"Duplicate estimate key {global_key!r}; one row per "
                "method/frame/joint is required."
            )
        all_seen_keys.add(global_key)
        if method != selected_method:
            continue

        metadata = frame_metadata(row, "Estimate", row_number)
        existing_metadata = metadata_by_sequence.setdefault(
            sequence_index,
            metadata,
        )
        if existing_metadata != metadata:
            raise ComparisonContractError(
                "Estimate has inconsistent frame metadata within "
                f"sequence_index {sequence_index} for method "
                f"{selected_method!r}."
            )
        warmup = parse_binary(
            row["warmup_excluded"],
            f"Estimate row {row_number} warmup_excluded",
        )
        existing_warmup = warmup_by_sequence.setdefault(
            sequence_index,
            warmup,
        )
        if existing_warmup != warmup:
            raise ComparisonContractError(
                "Estimate has inconsistent warmup_excluded values within "
                f"sequence_index {sequence_index}."
            )
        counts_as_measured = parse_binary(
            row["counts_as_measured_valid"],
            (
                f"Estimate row {row_number} "
                "counts_as_measured_valid"
            ),
        )
        if counts_as_measured != expected_measured_semantics:
            raise ComparisonContractError(
                f"Method {selected_method!r} requires "
                "counts_as_measured_valid="
                f"{int(expected_measured_semantics)}, but estimate row "
                f"{row_number} has {int(counts_as_measured)}."
            )
        provenance = str(row["provenance"]).strip()
        if not provenance:
            raise ComparisonContractError(
                f"Estimate row {row_number} provenance is blank."
            )
        run_id = str(row["run_id"]).strip()
        if not run_id:
            raise ComparisonContractError(
                f"Estimate row {row_number} run_id is blank."
            )
        run_ids.add(run_id)
        valid = parse_binary(
            row["valid"],
            f"Estimate row {row_number} valid",
        )
        point = None
        coordinate_values = (row["x_m"], row["y_m"], row["z_m"])
        if valid:
            point = tuple(
                parse_finite_float(
                    value,
                    f"Estimate row {row_number} {field}",
                )
                for field, value in zip(
                    ("x_m", "y_m", "z_m"),
                    coordinate_values,
                )
            )
        elif any(str(value).strip() for value in coordinate_values):
            # Invalid rows may retain diagnostics, but any supplied
            # coordinates must still be finite and complete.
            if not all(str(value).strip() for value in coordinate_values):
                raise ComparisonContractError(
                    f"Estimate row {row_number} supplies only part of an "
                    "invalid-row coordinate."
                )
            tuple(
                parse_finite_float(
                    value,
                    f"Estimate row {row_number} {field}",
                )
                for field, value in zip(
                    ("x_m", "y_m", "z_m"),
                    coordinate_values,
                )
            )
        key = (sequence_index, joint)
        indexed[key] = {
            "valid": valid,
            "point": point,
            "metadata": metadata,
            "provenance": provenance,
        }
    if selected_method not in methods_present:
        raise ComparisonContractError(
            f"Estimate CSV contains no rows for method {selected_method!r}."
        )
    if len(run_ids) != 1:
        raise ComparisonContractError(
            f"Selected method must contain exactly one run_id; observed "
            f"{sorted(run_ids)!r}."
        )
    return (
        indexed,
        metadata_by_sequence,
        warmup_by_sequence,
        methods_present,
        run_ids,
    )


def require_rectangular_joint_set(
    indexed_keys: set[tuple[int, str]],
    sequences: set[int],
    selected_joints: tuple[str, ...],
    label: str,
) -> None:
    expected = {
        (sequence_index, joint)
        for sequence_index in sequences
        for joint in selected_joints
    }
    missing = sorted(expected - indexed_keys)
    extra = sorted(indexed_keys - expected)
    if missing or extra:
        raise ComparisonContractError(
            f"{label} selected key set is not rectangular/exact; "
            f"missing={missing[:8]!r}, extra={extra[:8]!r}."
        )


def compare(
    ground_truth_csv: Path,
    estimate_csv: Path,
    method: str,
    include_warmup: bool = False,
) -> dict:
    if method not in METHOD_JOINTS:
        raise ComparisonContractError(
            f"Unsupported method {method!r}; choose one of "
            f"{sorted(METHOD_JOINTS)!r}."
        )
    ground_truth_path = ground_truth_csv.resolve()
    estimate_path = estimate_csv.resolve()
    comparator_path = Path(__file__).resolve()
    gt_rows = read_csv(
        ground_truth_path,
        GROUND_TRUTH_REQUIRED_FIELDS,
    )
    estimate_rows = read_csv(
        estimate_path,
        ESTIMATE_REQUIRED_FIELDS,
    )
    ground_truth_sha256 = sha256_file(ground_truth_path)
    estimate_sha256 = sha256_file(estimate_path)
    comparator_sha256 = sha256_file(comparator_path)
    gt_index, gt_metadata, gt_semantics = validate_ground_truth(gt_rows)
    (
        estimate_index,
        estimate_metadata,
        warmup_by_sequence,
        methods_present,
        run_ids,
    ) = validate_estimates(estimate_rows, method)

    selected_joints = METHOD_JOINTS[method]
    unexpected_estimate_joints = sorted(
        {
            joint
            for _, joint in estimate_index
            if joint not in selected_joints
        }
    )
    if unexpected_estimate_joints and method in {
        "k3_inferred",
        "k4_inferred",
    }:
        raise ComparisonContractError(
            f"Method {method!r} may contain only {selected_joints!r}; "
            f"observed unexpected joint(s) "
            f"{unexpected_estimate_joints!r}."
        )
    # The replay intentionally writes the complete 15-joint measured tables
    # once.  Raw/K2 dynamic right-arm reports select their registered
    # shoulder/elbow/wrist subset without treating the remaining measured
    # joints as an error.  Inferred K3/K4 outputs remain strictly two-joint.
    estimate_index = {
        key: value
        for key, value in estimate_index.items()
        if key[1] in selected_joints
    }

    estimate_sequences = set(estimate_metadata)
    require_rectangular_joint_set(
        set(estimate_index),
        estimate_sequences,
        selected_joints,
        "Estimate",
    )
    selected_gt_all = {
        key: value
        for key, value in gt_index.items()
        if key[1] in selected_joints
    }
    gt_sequences = {
        sequence_index
        for sequence_index, _ in selected_gt_all
    }
    require_rectangular_joint_set(
        set(selected_gt_all),
        gt_sequences,
        selected_joints,
        "Ground truth",
    )
    if gt_sequences != estimate_sequences:
        missing_estimate_sequences = sorted(
            gt_sequences - estimate_sequences
        )
        extra_estimate_sequences = sorted(
            estimate_sequences - gt_sequences
        )
        raise ComparisonContractError(
            "GT/estimate exact sequence-key mismatch before warm-up "
            "filtering; missing_estimate_sequences="
            f"{missing_estimate_sequences[:8]!r}, "
            "extra_estimate_sequences="
            f"{extra_estimate_sequences[:8]!r}."
        )
    excluded_sequences = (
        {
            sequence_index
            for sequence_index, excluded in warmup_by_sequence.items()
            if excluded
        }
        if not include_warmup
        else set()
    )
    included_sequences = estimate_sequences - excluded_sequences
    if not included_sequences:
        raise ComparisonContractError(
            "Warm-up filtering removed every estimate sequence."
        )

    selected_gt_index = {
        key: value
        for key, value in selected_gt_all.items()
        if key[0] in included_sequences
    }
    selected_estimate_index = {
        key: value
        for key, value in estimate_index.items()
        if key[0] in included_sequences
    }
    require_rectangular_joint_set(
        set(selected_gt_index),
        included_sequences,
        selected_joints,
        "Ground truth",
    )
    if set(selected_gt_index) != set(selected_estimate_index):
        missing = sorted(set(selected_gt_index) - set(selected_estimate_index))
        extra = sorted(set(selected_estimate_index) - set(selected_gt_index))
        raise ComparisonContractError(
            "GT/estimate exact join-key mismatch after warm-up filtering; "
            f"missing_estimate={missing[:8]!r}, "
            f"extra_estimate={extra[:8]!r}."
        )
    for sequence_index in sorted(included_sequences):
        if sequence_index not in gt_metadata:
            raise ComparisonContractError(
                f"GT has no frame metadata for sequence_index "
                f"{sequence_index}."
            )
        if gt_metadata[sequence_index] != estimate_metadata[sequence_index]:
            raise ComparisonContractError(
                "GT/estimate frame-key metadata mismatch at "
                f"sequence_index {sequence_index}: "
                f"GT={gt_metadata[sequence_index]!r}, "
                f"estimate={estimate_metadata[sequence_index]!r}."
            )

    errors_by_joint: dict[str, list[float]] = defaultdict(list)
    all_errors = []
    valid_points_by_sequence: dict[int, dict[str, tuple]] = defaultdict(dict)
    gt_points_by_sequence: dict[int, dict[str, tuple]] = defaultdict(dict)
    invalid_count_by_joint = Counter()
    provenance_by_joint: dict[str, set[str]] = defaultdict(set)
    error_by_sequence_joint: dict[tuple[int, str], float] = {}
    for key in sorted(selected_gt_index):
        sequence_index, joint = key
        gt_point = selected_gt_index[key]["point"]
        estimate = selected_estimate_index[key]
        gt_points_by_sequence[sequence_index][joint] = gt_point
        provenance_by_joint[joint].add(estimate["provenance"])
        if not estimate["valid"]:
            invalid_count_by_joint[joint] += 1
            continue
        estimate_point = estimate["point"]
        error_m = point_distance(estimate_point, gt_point)
        errors_by_joint[joint].append(error_m)
        all_errors.append(error_m)
        error_by_sequence_joint[(sequence_index, joint)] = error_m
        valid_points_by_sequence[sequence_index][joint] = estimate_point

    joint_metrics = {}
    for joint in selected_joints:
        expected_count = len(included_sequences)
        valid_count = len(errors_by_joint[joint])
        joint_metrics[joint] = {
            "expected_sample_count": expected_count,
            "valid_sample_count": valid_count,
            "invalid_sample_count": invalid_count_by_joint[joint],
            "coverage_rate": (
                valid_count / expected_count if expected_count else None
            ),
            "provenance_values": sorted(provenance_by_joint[joint]),
            **error_summary(errors_by_joint[joint]),
        }

    bone_metrics = {}
    for bone, (parent, child) in METHOD_BONES[method].items():
        signed_errors = []
        for sequence_index in sorted(included_sequences):
            estimate_points = valid_points_by_sequence[sequence_index]
            if parent not in estimate_points or child not in estimate_points:
                continue
            gt_points = gt_points_by_sequence[sequence_index]
            gt_length_m = point_distance(
                gt_points[parent],
                gt_points[child],
            )
            estimate_length_m = point_distance(
                estimate_points[parent],
                estimate_points[child],
            )
            signed_errors.append(estimate_length_m - gt_length_m)
        bone_metrics[bone] = {
            "parent_joint": parent,
            "child_joint": child,
            **bone_error_summary(
                signed_errors,
                len(included_sequences),
            ),
        }

    primary_errors_m = []
    primary_samples = []
    primary_joints = ("right_elbow", "right_wrist")
    for sequence_index in sorted(included_sequences):
        metadata = estimate_metadata[sequence_index]
        joint_errors_m = {
            joint: error_by_sequence_joint.get((sequence_index, joint))
            for joint in primary_joints
        }
        valid = all(
            joint_errors_m[joint] is not None for joint in primary_joints
        )
        primary_error_m = (
            sum(joint_errors_m.values()) / len(primary_joints)
            if valid
            else None
        )
        if primary_error_m is not None:
            primary_errors_m.append(primary_error_m)
        primary_samples.append(
            {
                "sequence_index": sequence_index,
                "cycle_index": metadata[0],
                "animation_frame_code": str(metadata[1]),
                "usd_time_code": str(metadata[2]),
                "valid": valid,
                "right_elbow_error_mm": (
                    joint_errors_m["right_elbow"] * 1000.0
                    if joint_errors_m["right_elbow"] is not None
                    else None
                ),
                "right_wrist_error_mm": (
                    joint_errors_m["right_wrist"] * 1000.0
                    if joint_errors_m["right_wrist"] is not None
                    else None
                ),
                "two_joint_mean_position_error_mm": (
                    primary_error_m * 1000.0
                    if primary_error_m is not None
                    else None
                ),
            }
        )

    expected_pair_count = len(included_sequences) * len(selected_joints)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "purpose": "synchronized_dynamic_kinematic_gt_comparison_v1",
        "ground_truth_csv": str(ground_truth_path),
        "ground_truth_csv_sha256": ground_truth_sha256,
        "estimate_csv": str(estimate_path),
        "estimate_csv_sha256": estimate_sha256,
        "comparator_path": str(comparator_path),
        "comparator_sha256": comparator_sha256,
        "method": method,
        "run_id": next(iter(run_ids)),
        "reported_joints": list(selected_joints),
        "reported_bones": list(METHOD_BONES[method]),
        "ignored_nonreporting_estimate_joints": (
            unexpected_estimate_joints
        ),
        "methods_present_in_estimate_csv": sorted(methods_present),
        "ground_truth_semantics": sorted(gt_semantics),
        "join_contract": {
            "primary_key": ["sequence_index", "canonical_joint"],
            "frame_metadata_must_match": list(FRAME_FIELDS),
            "join_type": "exact_one_to_one",
            "timestamp_nearest_matching_used": False,
            "interpolation_used": False,
            "ground_truth_averaging_used": False,
            "coordinate_transform_used": False,
        },
        "warmup_filter": {
            "warmup_included": include_warmup,
            "excluded_sequence_count": len(excluded_sequences),
            "excluded_sequence_indices": sorted(excluded_sequences),
            "excluded_estimate_key_count": (
                len(excluded_sequences) * len(selected_joints)
            ),
            "excluded_ground_truth_key_count": (
                len(excluded_sequences) * len(selected_joints)
            ),
            "excluded_joint_pair_count": (
                len(excluded_sequences) * len(selected_joints)
            ),
            "included_sequence_count": len(included_sequences),
        },
        "coverage": {
            "expected_joint_sample_count": expected_pair_count,
            "valid_joint_sample_count": len(all_errors),
            "invalid_joint_sample_count": (
                expected_pair_count - len(all_errors)
            ),
            "coverage_rate": (
                len(all_errors) / expected_pair_count
                if expected_pair_count
                else None
            ),
        },
        "primary_metric": {
            "id": "right_arm_two_joint_mean_position_error_mm",
            "definition": (
                "For each included sequence with both predictions valid, "
                "average the right-elbow and right-wrist Euclidean 3D "
                "errors, then summarize those per-sequence averages."
            ),
            "expected_sequence_count": len(included_sequences),
            "valid_sequence_count": len(primary_errors_m),
            "invalid_sequence_count": (
                len(included_sequences) - len(primary_errors_m)
            ),
            "coverage_rate": (
                len(primary_errors_m) / len(included_sequences)
                if included_sequences
                else None
            ),
            **error_summary(primary_errors_m),
            "per_sequence": primary_samples,
        },
        "overall_3d_error": error_summary(all_errors),
        "joint_metrics": joint_metrics,
        "bone_metrics": bone_metrics,
        "interpretation": (
            "Errors are valid only if both CSVs already use the same "
            "camera coordinate system, metric unit, synchronized sequence "
            "definition, and canonical-joint definitions. K3/K4 rows "
            "remain inferred and must not be relabelled as measured depth."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare synchronized per-frame dynamic GT with one unified "
            "right-arm estimate method."
        )
    )
    parser.add_argument("--ground-truth-csv", type=Path, required=True)
    parser.add_argument("--estimate-csv", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_JOINTS),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help=(
            "Include rows marked warmup_excluded=1. By default those "
            "sequence indices are removed from both inputs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_json.resolve()
    try:
        if output.exists():
            raise ComparisonContractError(
                f"Refusing to overwrite existing output JSON: {output}"
            )
        report = compare(
            args.ground_truth_csv,
            args.estimate_csv,
            args.method,
            include_warmup=args.include_warmup,
        )
        write_json_atomic_new(output, report)
    except (ComparisonContractError, OSError, csv.Error) as exc:
        print(f"Dynamic GT comparison refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["coverage"], indent=2))
    print("Dynamic GT comparison report:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
