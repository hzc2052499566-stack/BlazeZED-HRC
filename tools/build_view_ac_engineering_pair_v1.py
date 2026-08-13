"""Pair preserved View A/C captures and derive engineering-only extrinsics.

This tool is deliberately *not* a formal calibration path.  It exact-pairs
two independently captured, unit-corrected single-view datasets by their
frozen USD frame keys, verifies that their operational-world Skeleton GT is
identical, and then uses the GT world/camera correspondences to fit rigid
camera-to-world transforms.  The result exists only to unblock common-frame
replay development before a genuine dual-render-product capture is made.

Ground truth is read by design, so every output carries explicit labels that
make it ineligible for formal estimation or accuracy evidence.  Inputs are
never modified and the requested output directory must not already exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
EXPECTED_FRAME_COUNT = 240
EXPECTED_JOINT_COUNT = 15
EXPECTED_GT_ROW_COUNT = EXPECTED_FRAME_COUNT * EXPECTED_JOINT_COUNT
EXPECTED_TIME_CODES_PER_SECOND = Decimal("60")
EXPECTED_UNIT_CORRECTION_FACTOR = Decimal("100")
TIMELINE_TIME_TOLERANCE_S = Decimal("1e-15")
RIGID_FIT_MAX_RESIDUAL_M = 1.0e-8
RIGID_ORTHOGONALITY_TOLERANCE = 1.0e-10

SYNC_MANIFEST_NAME = "sync_manifest.json"
CALIBRATION_NAME = "calibration.json"

CLAIM_ELIGIBILITY = "excluded_engineering_replay_only"
CALIBRATION_SOURCE = "gt_correspondence_engineering_only"
SOURCE_CONTENT_HASH_SCHEMA = (
    "dynamic_rgbd_gt_v1_canonical_content_sha256_v1"
)
STATE_OUTPUT_FILES = (
    "rgbd_manifest.csv",
    "ground_truth_joints.csv",
    "ground_truth_metadata.json",
    "source_frame_content_hashes.csv",
    "source_dynamic_rgbd_gt_protocol.json",
)
SOURCE_CONTENT_REQUIRED_FIELDS = frozenset(
    {
        "sample_index",
        "sequence_index",
        "cycle_index",
        "animation_frame_code",
        "usd_time_code",
        "rgb_file",
        "depth_file",
        "camera_params_file",
        "rgb_content_sha256",
        "depth_content_sha256",
        "camera_params_content_sha256",
        "gt_coordinates_sha256",
        "frame_content_sha256",
        "hash_schema",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "run_id",
        "sample_index",
        "sequence_index",
        "cycle_index",
        "animation_frame_code",
        "usd_time_code",
        "timeline_time_s",
        "wall_time_ns",
        "camera_prim",
        "rgb_file",
        "depth_file",
        "camera_params_file",
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
        "depth_scale_to_m",
        "camera_model",
        "protocol_sha256",
        "source_depth_scale_to_m",
        "operational_unit_correction_factor",
        "unit_correction_protocol_sha256",
    }
)

GT_REQUIRED_FIELDS = frozenset(
    {
        "run_id",
        "sample_index",
        "sequence_index",
        "cycle_index",
        "animation_frame_code",
        "usd_time_code",
        "timeline_time_s",
        "canonical_joint",
        "world_x_m",
        "world_y_m",
        "world_z_m",
        "gt_x_m",
        "gt_y_m",
        "gt_z_m",
        "gt_semantics",
        "operational_unit_correction_factor",
        "unit_correction_protocol_sha256",
    }
)


class EngineeringPairingError(RuntimeError):
    """Raised when an input violates the engineering pairing contract."""


@dataclass(frozen=True)
class GroundTruthPoint:
    sequence_index: int
    canonical_joint: str
    frame_metadata: tuple[int, Decimal, Decimal, Decimal]
    world_decimal: tuple[Decimal, Decimal, Decimal]
    world: np.ndarray
    camera: np.ndarray


@dataclass(frozen=True)
class ViewDataset:
    label: str
    directory: Path
    run_id: str
    camera_prim: str
    manifest_rows: tuple[dict[str, str], ...]
    manifest_by_sequence: dict[int, dict[str, str]]
    gt_by_key: dict[tuple[int, str], GroundTruthPoint]
    canonical_joints: tuple[str, ...]
    source_sha256: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EngineeringPairingError(
            f"Could not read valid JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise EngineeringPairingError(f"JSON root must be an object: {path}")
    return value


def read_csv_required(
    path: Path,
    required_fields: frozenset[str],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise EngineeringPairingError(f"Required CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required_fields - fields)
        if missing:
            raise EngineeringPairingError(
                f"{path} is missing column(s): {', '.join(missing)}"
            )
        return list(reader)


def parse_int(value: object, label: str) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise EngineeringPairingError(
            f"{label} must be an integer; observed {value!r}."
        ) from error
    return number


def parse_decimal(value: object, label: str) -> Decimal:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as error:
        raise EngineeringPairingError(
            f"{label} must be a finite decimal; observed {value!r}."
        ) from error
    if not number.is_finite():
        raise EngineeringPairingError(
            f"{label} must be finite; observed {value!r}."
        )
    return number


def parse_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EngineeringPairingError(
            f"{label} must be a finite float; observed {value!r}."
        ) from error
    if not math.isfinite(number):
        raise EngineeringPairingError(
            f"{label} must be finite; observed {value!r}."
        )
    return number


def exact_frame_metadata(
    row: dict[str, str],
    source: str,
) -> tuple[int, Decimal, Decimal, Decimal]:
    return (
        parse_int(row["cycle_index"], f"{source} cycle_index"),
        parse_decimal(
            row["animation_frame_code"],
            f"{source} animation_frame_code",
        ),
        parse_decimal(row["usd_time_code"], f"{source} usd_time_code"),
        parse_decimal(row["timeline_time_s"], f"{source} timeline_time_s"),
    )


def validate_unit_metadata(directory: Path, label: str) -> dict[str, str]:
    metadata_path = directory / "ground_truth_metadata.json"
    state_path = directory / "unit_correction_state.json"
    metadata = read_json(metadata_path)
    state = read_json(state_path)
    if metadata.get("status") != "complete_derived_unit_corrected":
        raise EngineeringPairingError(
            f"View {label} metadata is not a complete unit-corrected dataset."
        )
    if metadata.get("unit") != "operational_metre":
        raise EngineeringPairingError(
            f"View {label} metadata does not declare operational_metre."
        )
    if parse_decimal(
        metadata.get("operational_metres_per_scene_unit"),
        f"View {label} operational_metres_per_scene_unit",
    ) != Decimal("1"):
        raise EngineeringPairingError(
            f"View {label} operational metres-per-unit must equal 1."
        )
    if parse_decimal(
        metadata.get("operational_unit_correction_factor"),
        f"View {label} metadata correction factor",
    ) != EXPECTED_UNIT_CORRECTION_FACTOR:
        raise EngineeringPairingError(
            f"View {label} unit correction factor must equal 100."
        )
    if state.get("status") != "complete":
        raise EngineeringPairingError(
            f"View {label} unit-correction state is not complete."
        )
    if parse_decimal(
        state.get("correction_factor"),
        f"View {label} state correction factor",
    ) != EXPECTED_UNIT_CORRECTION_FACTOR:
        raise EngineeringPairingError(
            f"View {label} state correction factor must equal 100."
        )
    if parse_int(
        state.get("source_sample_count"),
        f"View {label} source_sample_count",
    ) != EXPECTED_FRAME_COUNT:
        raise EngineeringPairingError(
            f"View {label} state must report 240 source samples."
        )
    if parse_int(
        state.get("source_gt_row_count"),
        f"View {label} source_gt_row_count",
    ) != EXPECTED_GT_ROW_COUNT:
        raise EngineeringPairingError(
            f"View {label} state must report 3,600 GT rows."
        )
    if parse_int(
        state.get("hard_linked_rgb_depth_file_count"),
        f"View {label} hard_linked_rgb_depth_file_count",
    ) != EXPECTED_FRAME_COUNT * 2:
        raise EngineeringPairingError(
            f"View {label} state must report 480 linked RGB/depth files."
        )
    if parse_int(
        state.get("rewritten_camera_params_file_count"),
        f"View {label} rewritten_camera_params_file_count",
    ) != EXPECTED_FRAME_COUNT:
        raise EngineeringPairingError(
            f"View {label} state must report 240 camera-parameter files."
        )

    registered_outputs = state.get("output_sha256")
    if not isinstance(registered_outputs, dict):
        raise EngineeringPairingError(
            f"View {label} unit state has no output_sha256 object."
        )
    missing_outputs = sorted(
        set(STATE_OUTPUT_FILES).difference(registered_outputs)
    )
    if missing_outputs:
        raise EngineeringPairingError(
            f"View {label} unit state is missing registered output hash(es): "
            + ", ".join(missing_outputs)
        )
    source_hashes = {
        "unit_correction_state.json": sha256_file(state_path),
    }
    for name in STATE_OUTPUT_FILES:
        path = directory / name
        if not path.is_file():
            raise EngineeringPairingError(
                f"View {label} registered output is missing: {path}"
            )
        registered_hash = str(registered_outputs.get(name, "")).lower()
        if not SHA256_PATTERN.fullmatch(registered_hash):
            raise EngineeringPairingError(
                f"View {label} registered hash for {name} is invalid."
            )
        observed_hash = sha256_file(path)
        if observed_hash != registered_hash:
            raise EngineeringPairingError(
                f"View {label} registered output hash mismatch for {name}."
            )
        source_hashes[name] = observed_hash
    return source_hashes


def validate_manifest_file_references(
    directory: Path,
    label: str,
    manifest_rows: list[dict[str, str]],
) -> None:
    """Require every manifest payload reference to exist inside rgbd_frames."""
    frame_directory = directory / "rgbd_frames"
    if not frame_directory.is_dir():
        raise EngineeringPairingError(
            f"View {label} rgbd_frames directory is missing: {frame_directory}"
        )
    for field in ("rgb_file", "depth_file", "camera_params_file"):
        names = [str(row[field]).strip() for row in manifest_rows]
        if len(set(names)) != EXPECTED_FRAME_COUNT:
            raise EngineeringPairingError(
                f"View {label} manifest {field} values must be 240 unique names."
            )
        for sequence, name in enumerate(names):
            relative = Path(name)
            if (
                not name
                or relative.is_absolute()
                or len(relative.parts) != 1
                or relative.name != name
            ):
                raise EngineeringPairingError(
                    f"View {label} frame {sequence} {field} must be a safe "
                    f"single filename; observed {name!r}."
                )
            referenced = frame_directory / name
            if not referenced.is_file():
                raise EngineeringPairingError(
                    f"View {label} manifest references missing {field}: "
                    f"{referenced}"
                )


def require_sha256(value: object, label: str) -> str:
    digest = str(value).strip().lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise EngineeringPairingError(
            f"{label} must be a lowercase 64-character SHA-256 digest."
        )
    return digest


def validate_preserved_source_audit(
    directory: Path,
    label: str,
    manifest_by_sequence: dict[int, dict[str, str]],
    run_id: str,
    camera_prim: str,
) -> None:
    """Validate the preserved raw-capture protocol and per-frame hash index.

    The unit-corrected camera-parameter JSON files contain extra unit fields,
    so their whole-file hashes cannot equal the raw canonical CameraParams
    digests.  We therefore verify the preserved index's own registered file
    hash, exact frame/file linkage, schema, and digest syntax here.  RGB and
    depth payload existence is checked separately through the manifest.
    """
    content_path = directory / "source_frame_content_hashes.csv"
    content_rows = read_csv_required(
        content_path,
        SOURCE_CONTENT_REQUIRED_FIELDS,
    )
    if len(content_rows) != EXPECTED_FRAME_COUNT:
        raise EngineeringPairingError(
            f"View {label} source frame-content index must contain 240 rows."
        )
    for row_number, row in enumerate(content_rows, start=2):
        source = f"View {label} source content row {row_number}"
        sequence = parse_int(row["sequence_index"], f"{source} sequence_index")
        sample = parse_int(row["sample_index"], f"{source} sample_index")
        if sequence != row_number - 2 or sample != sequence:
            raise EngineeringPairingError(
                f"View {label} source content keys must be ordered 0..239."
            )
        manifest = manifest_by_sequence[sequence]
        for field in (
            "cycle_index",
            "animation_frame_code",
            "usd_time_code",
            "rgb_file",
            "depth_file",
            "camera_params_file",
        ):
            if str(row[field]).strip() != str(manifest[field]).strip():
                raise EngineeringPairingError(
                    f"View {label} source content/manifest mismatch at "
                    f"sequence {sequence}, field {field}."
                )
        if row["hash_schema"].strip() != SOURCE_CONTENT_HASH_SCHEMA:
            raise EngineeringPairingError(
                f"{source} uses an unexpected canonical hash schema."
            )
        for field in (
            "rgb_content_sha256",
            "depth_content_sha256",
            "camera_params_content_sha256",
            "gt_coordinates_sha256",
            "frame_content_sha256",
        ):
            require_sha256(row[field], f"{source} {field}")

    protocol_path = directory / "source_dynamic_rgbd_gt_protocol.json"
    protocol = read_json(protocol_path)
    expected_protocol_hash = sha256_file(protocol_path)
    manifest_protocol_hashes = {
        require_sha256(
            row["protocol_sha256"],
            f"View {label} manifest protocol_sha256",
        )
        for row in manifest_by_sequence.values()
    }
    if manifest_protocol_hashes != {expected_protocol_hash}:
        raise EngineeringPairingError(
            f"View {label} manifest protocol hash does not identify the "
            "preserved source protocol."
        )
    expected_protocol_values = {
        "purpose": "deterministic_dynamic_rgbd_and_skeleton_gt",
        "run_id": run_id,
        "camera_prim_path": camera_prim,
        "time_codes_per_second": 60,
        "frames_per_cycle": EXPECTED_FRAME_COUNT,
        "cycle_count": 1,
        "sequence_count": EXPECTED_FRAME_COUNT,
        "excluded_closure_time_code": EXPECTED_FRAME_COUNT,
    }
    for field, expected in expected_protocol_values.items():
        observed = protocol.get(field)
        if observed != expected:
            raise EngineeringPairingError(
                f"View {label} preserved source protocol {field} mismatch: "
                f"expected {expected!r}, observed {observed!r}."
            )
    first = protocol.get("first_sequence")
    last = protocol.get("last_sequence")
    if not isinstance(first, dict) or not isinstance(last, dict):
        raise EngineeringPairingError(
            f"View {label} source protocol lacks first/last sequence records."
        )
    expected_first = {
        "sample_index": 0,
        "sequence_index": 0,
        "cycle_index": 0,
        "animation_frame_code": 0,
        "usd_time_code": 0,
    }
    expected_last = {
        "sample_index": 239,
        "sequence_index": 239,
        "cycle_index": 0,
        "animation_frame_code": 239,
        "usd_time_code": 239,
    }
    if first != expected_first or last != expected_last:
        raise EngineeringPairingError(
            f"View {label} source protocol first/last sequence mismatch."
        )


def validate_manifest(
    rows: list[dict[str, str]],
    label: str,
) -> tuple[str, str, dict[int, dict[str, str]]]:
    if len(rows) != EXPECTED_FRAME_COUNT:
        raise EngineeringPairingError(
            f"View {label} manifest must contain exactly 240 rows; "
            f"observed {len(rows)}."
        )
    indexed: dict[int, dict[str, str]] = {}
    run_ids: set[str] = set()
    camera_prims: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        source = f"View {label} manifest row {row_number}"
        sequence = parse_int(row["sequence_index"], f"{source} sequence_index")
        sample = parse_int(row["sample_index"], f"{source} sample_index")
        if sequence in indexed:
            raise EngineeringPairingError(
                f"View {label} manifest has duplicate sequence {sequence}."
            )
        if sequence != row_number - 2 or sample != sequence:
            raise EngineeringPairingError(
                f"View {label} sample/sequence indices must be ordered 0..239."
            )
        cycle, animation, usd_time, timeline = exact_frame_metadata(row, source)
        expected = Decimal(sequence)
        if cycle != 0 or animation != expected or usd_time != expected:
            raise EngineeringPairingError(
                f"View {label} frame {sequence} must have cycle=0 and exact "
                "animation/USD time code equal to sequence_index."
            )
        expected_timeline = expected / EXPECTED_TIME_CODES_PER_SECOND
        if abs(timeline - expected_timeline) > TIMELINE_TIME_TOLERANCE_S:
            raise EngineeringPairingError(
                f"View {label} frame {sequence} timeline_time_s is not "
                "consistent with usd_time_code / 60."
            )
        if parse_decimal(
            row["operational_unit_correction_factor"],
            f"{source} correction factor",
        ) != EXPECTED_UNIT_CORRECTION_FACTOR:
            raise EngineeringPairingError(
                f"View {label} manifest correction factor is not 100."
            )
        if parse_decimal(
            row["depth_scale_to_m"], f"{source} depth_scale_to_m"
        ) != EXPECTED_UNIT_CORRECTION_FACTOR:
            raise EngineeringPairingError(
                f"View {label} manifest depth scale is not 100."
            )
        for field in (
            "run_id",
            "camera_prim",
            "rgb_file",
            "depth_file",
            "camera_params_file",
            "protocol_sha256",
            "unit_correction_protocol_sha256",
        ):
            if not str(row[field]).strip():
                raise EngineeringPairingError(f"{source} {field} is blank.")
        run_ids.add(row["run_id"].strip())
        camera_prims.add(row["camera_prim"].strip())
        indexed[sequence] = row
    if set(indexed) != set(range(EXPECTED_FRAME_COUNT)):
        raise EngineeringPairingError(
            f"View {label} manifest sequence set is not exactly 0..239."
        )
    if len(run_ids) != 1 or len(camera_prims) != 1:
        raise EngineeringPairingError(
            f"View {label} manifest must contain one run_id and camera_prim."
        )
    return next(iter(run_ids)), next(iter(camera_prims)), indexed


def validate_ground_truth(
    rows: list[dict[str, str]],
    label: str,
    manifest_by_sequence: dict[int, dict[str, str]],
    expected_run_id: str,
) -> tuple[dict[tuple[int, str], GroundTruthPoint], tuple[str, ...]]:
    if len(rows) != EXPECTED_GT_ROW_COUNT:
        raise EngineeringPairingError(
            f"View {label} GT must contain exactly 3,600 rows; "
            f"observed {len(rows)}."
        )
    indexed: dict[tuple[int, str], GroundTruthPoint] = {}
    joints_by_sequence: dict[int, list[str]] = {
        sequence: [] for sequence in range(EXPECTED_FRAME_COUNT)
    }
    for row_number, row in enumerate(rows, start=2):
        source = f"View {label} GT row {row_number}"
        if row["run_id"].strip() != expected_run_id:
            raise EngineeringPairingError(
                f"{source} run_id differs from the manifest."
            )
        sequence = parse_int(row["sequence_index"], f"{source} sequence_index")
        sample = parse_int(row["sample_index"], f"{source} sample_index")
        if sequence not in manifest_by_sequence or sample != sequence:
            raise EngineeringPairingError(
                f"{source} sample/sequence key is outside 0..239."
            )
        joint = row["canonical_joint"].strip()
        if not joint:
            raise EngineeringPairingError(f"{source} canonical_joint is blank.")
        key = (sequence, joint)
        if key in indexed:
            raise EngineeringPairingError(
                f"View {label} GT has duplicate key {key!r}."
            )
        if row["gt_semantics"].strip() != "synchronized_per_frame":
            raise EngineeringPairingError(
                f"{source} must declare synchronized_per_frame GT."
            )
        if parse_decimal(
            row["operational_unit_correction_factor"],
            f"{source} correction factor",
        ) != EXPECTED_UNIT_CORRECTION_FACTOR:
            raise EngineeringPairingError(
                f"{source} correction factor is not 100."
            )
        metadata = exact_frame_metadata(row, source)
        manifest_metadata = exact_frame_metadata(
            manifest_by_sequence[sequence],
            f"View {label} manifest frame {sequence}",
        )
        if metadata != manifest_metadata:
            raise EngineeringPairingError(
                f"View {label} GT/manifest metadata mismatch at {key!r}."
            )
        world_decimal = tuple(
            parse_decimal(row[field], f"{source} {field}")
            for field in ("world_x_m", "world_y_m", "world_z_m")
        )
        camera_values = tuple(
            parse_float(row[field], f"{source} {field}")
            for field in ("gt_x_m", "gt_y_m", "gt_z_m")
        )
        point = GroundTruthPoint(
            sequence_index=sequence,
            canonical_joint=joint,
            frame_metadata=metadata,
            world_decimal=world_decimal,
            world=np.asarray([float(value) for value in world_decimal]),
            camera=np.asarray(camera_values, dtype=np.float64),
        )
        indexed[key] = point
        joints_by_sequence[sequence].append(joint)
    first_joints = tuple(joints_by_sequence[0])
    if len(first_joints) != EXPECTED_JOINT_COUNT or len(set(first_joints)) != (
        EXPECTED_JOINT_COUNT
    ):
        raise EngineeringPairingError(
            f"View {label} frame 0 must contain 15 unique joints."
        )
    expected_joint_set = set(first_joints)
    for sequence, joints in joints_by_sequence.items():
        if len(joints) != EXPECTED_JOINT_COUNT or set(joints) != expected_joint_set:
            raise EngineeringPairingError(
                f"View {label} frame {sequence} joint set differs from frame 0."
            )
    return indexed, first_joints


def load_view(directory: Path, label: str) -> ViewDataset:
    directory = directory.resolve()
    if not directory.is_dir():
        raise EngineeringPairingError(
            f"View {label} directory does not exist: {directory}"
        )
    source_hashes = validate_unit_metadata(directory, label)
    manifest_path = directory / "rgbd_manifest.csv"
    gt_path = directory / "ground_truth_joints.csv"
    manifest_rows = read_csv_required(manifest_path, MANIFEST_REQUIRED_FIELDS)
    gt_rows = read_csv_required(gt_path, GT_REQUIRED_FIELDS)
    run_id, camera_prim, manifest_by_sequence = validate_manifest(
        manifest_rows, label
    )
    validate_manifest_file_references(directory, label, manifest_rows)
    validate_preserved_source_audit(
        directory,
        label,
        manifest_by_sequence,
        run_id,
        camera_prim,
    )
    gt_by_key, canonical_joints = validate_ground_truth(
        gt_rows,
        label,
        manifest_by_sequence,
        run_id,
    )
    return ViewDataset(
        label=label,
        directory=directory,
        run_id=run_id,
        camera_prim=camera_prim,
        manifest_rows=tuple(manifest_rows),
        manifest_by_sequence=manifest_by_sequence,
        gt_by_key=gt_by_key,
        canonical_joints=canonical_joints,
        source_sha256=source_hashes,
    )


def exact_pair_keys(
    view_a: ViewDataset,
    view_c: ViewDataset,
) -> list[tuple[int, str]]:
    keys_a = set(view_a.gt_by_key)
    keys_c = set(view_c.gt_by_key)
    if keys_a != keys_c:
        raise EngineeringPairingError(
            "View A/C GT key sets differ; exact pairing is impossible."
        )
    if set(view_a.canonical_joints) != set(view_c.canonical_joints):
        raise EngineeringPairingError("View A/C canonical joint sets differ.")
    ordered_keys: list[tuple[int, str]] = []
    for sequence in range(EXPECTED_FRAME_COUNT):
        row_a = view_a.manifest_by_sequence[sequence]
        row_c = view_c.manifest_by_sequence[sequence]
        metadata_a = exact_frame_metadata(row_a, f"View A frame {sequence}")
        metadata_c = exact_frame_metadata(row_c, f"View C frame {sequence}")
        if metadata_a != metadata_c:
            raise EngineeringPairingError(
                f"View A/C exact time-code mismatch at sequence {sequence}."
            )
        for joint in view_a.canonical_joints:
            key = (sequence, joint)
            point_a = view_a.gt_by_key[key]
            point_c = view_c.gt_by_key[key]
            if point_a.frame_metadata != point_c.frame_metadata:
                raise EngineeringPairingError(
                    f"View A/C GT frame metadata mismatch at {key!r}."
                )
            if point_a.world_decimal != point_c.world_decimal:
                raise EngineeringPairingError(
                    f"View A/C operational-world GT mismatch at {key!r}."
                )
            ordered_keys.append(key)
    if len(ordered_keys) != EXPECTED_GT_ROW_COUNT:
        raise EngineeringPairingError("Exact paired GT row count is not 3,600.")
    return ordered_keys


def fit_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_frame: str,
    target_frame: str,
) -> dict[str, Any]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise EngineeringPairingError("Rigid-fit point arrays must both be Nx3.")
    if source.shape[0] < 4:
        raise EngineeringPairingError("Rigid fit requires at least four points.")
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    if np.linalg.matrix_rank(source_centered, tol=1.0e-12) < 3:
        raise EngineeringPairingError(
            f"Rigid-fit correspondences in {source_frame} are rank deficient."
        )
    covariance = source_centered.T @ target_centered
    u_matrix, singular_values, v_transpose = np.linalg.svd(covariance)
    rotation = v_transpose.T @ u_matrix.T
    reflection_corrected = False
    if np.linalg.det(rotation) < 0.0:
        v_transpose[-1, :] *= -1.0
        rotation = v_transpose.T @ u_matrix.T
        reflection_corrected = True
    translation = target_centroid - rotation @ source_centroid
    predicted = (rotation @ source.T).T + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(rotation))
    maximum_residual = float(np.max(residuals))
    if maximum_residual > RIGID_FIT_MAX_RESIDUAL_M:
        raise EngineeringPairingError(
            f"Rigid fit {target_frame}_from_{source_frame} maximum residual "
            f"{maximum_residual:.3e} m exceeds "
            f"{RIGID_FIT_MAX_RESIDUAL_M:.1e} m."
        )
    if (
        orthogonality_error > RIGID_ORTHOGONALITY_TOLERANCE
        or abs(determinant - 1.0) > RIGID_ORTHOGONALITY_TOLERANCE
    ):
        raise EngineeringPairingError(
            f"Rigid fit {target_frame}_from_{source_frame} is not a proper "
            "rotation."
        )
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :3] = rotation
    homogeneous[:3, 3] = translation
    return {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "formula": "p_target = R_target_from_source @ p_source + t_target_from_source_m",
        "rotation_3x3": rotation.tolist(),
        "translation_m": translation.tolist(),
        "homogeneous_4x4": homogeneous.tolist(),
        "fit": {
            "correspondence_count": int(source.shape[0]),
            "rms_residual_m": float(np.sqrt(np.mean(residuals**2))),
            "maximum_residual_m": maximum_residual,
            "mean_residual_m": float(np.mean(residuals)),
            "rotation_determinant": determinant,
            "rotation_orthogonality_error_fro": orthogonality_error,
            "source_centered_rank": int(
                np.linalg.matrix_rank(source_centered, tol=1.0e-12)
            ),
            "covariance_singular_values": singular_values.tolist(),
            "reflection_corrected": reflection_corrected,
        },
    }


def transform_arrays(transform: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(transform["rotation_3x3"], dtype=np.float64),
        np.asarray(transform["translation_m"], dtype=np.float64),
    )


def relative_transform(
    world_from_source: dict[str, Any],
    world_from_target: dict[str, Any],
    *,
    source_frame: str,
    target_frame: str,
) -> dict[str, Any]:
    rotation_world_source, translation_world_source = transform_arrays(
        world_from_source
    )
    rotation_world_target, translation_world_target = transform_arrays(
        world_from_target
    )
    rotation_target_source = rotation_world_target.T @ rotation_world_source
    translation_target_source = rotation_world_target.T @ (
        translation_world_source - translation_world_target
    )
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :3] = rotation_target_source
    homogeneous[:3, 3] = translation_target_source
    orthogonality_error = float(
        np.linalg.norm(
            rotation_target_source.T @ rotation_target_source - np.eye(3),
            ord="fro",
        )
    )
    return {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "formula": "p_target = R_target_from_source @ p_source + t_target_from_source_m",
        "rotation_3x3": rotation_target_source.tolist(),
        "translation_m": translation_target_source.tolist(),
        "homogeneous_4x4": homogeneous.tolist(),
        "fit": {
            "derived_from_camera_to_world_fits": True,
            "rotation_determinant": float(np.linalg.det(rotation_target_source)),
            "rotation_orthogonality_error_fro": orthogonality_error,
        },
    }


def apply_transform(transform: dict[str, Any], points: np.ndarray) -> np.ndarray:
    rotation, translation = transform_arrays(transform)
    source = np.asarray(points, dtype=np.float64)
    return (rotation @ source.T).T + translation


def cross_view_residual_summary(
    transform: dict[str, Any],
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> dict[str, Any]:
    residuals = np.linalg.norm(
        apply_transform(transform, source_points) - target_points,
        axis=1,
    )
    maximum = float(np.max(residuals))
    if maximum > RIGID_FIT_MAX_RESIDUAL_M:
        raise EngineeringPairingError(
            f"Cross-view transform maximum residual {maximum:.3e} m exceeds "
            f"{RIGID_FIT_MAX_RESIDUAL_M:.1e} m."
        )
    return {
        "correspondence_count": int(len(residuals)),
        "rms_residual_m": float(np.sqrt(np.mean(residuals**2))),
        "mean_residual_m": float(np.mean(residuals)),
        "maximum_residual_m": maximum,
    }


def json_number(value: str) -> int | float:
    decimal = parse_decimal(value, "sync-manifest numeric value")
    integral = decimal.to_integral_value()
    return int(integral) if decimal == integral else float(decimal)


def build_sync_manifest(
    view_a: ViewDataset,
    view_c: ViewDataset,
) -> dict[str, Any]:
    pairs = []
    for sequence in range(EXPECTED_FRAME_COUNT):
        row_a = view_a.manifest_by_sequence[sequence]
        row_c = view_c.manifest_by_sequence[sequence]
        pairs.append(
            {
                "sync_index": sequence,
                "sequence_index": sequence,
                "cycle_index": 0,
                "animation_frame_code": sequence,
                "usd_time_code": sequence,
                "timeline_time_s": json_number(row_a["timeline_time_s"]),
                "view_a": {
                    "run_id": view_a.run_id,
                    "sample_index": parse_int(
                        row_a["sample_index"], "View A sample_index"
                    ),
                    "camera_prim": view_a.camera_prim,
                    "wall_time_ns": parse_int(
                        row_a["wall_time_ns"], "View A wall_time_ns"
                    ),
                    "rgb_file": row_a["rgb_file"],
                    "depth_file": row_a["depth_file"],
                    "camera_params_file": row_a["camera_params_file"],
                },
                "view_c": {
                    "run_id": view_c.run_id,
                    "sample_index": parse_int(
                        row_c["sample_index"], "View C sample_index"
                    ),
                    "camera_prim": view_c.camera_prim,
                    "wall_time_ns": parse_int(
                        row_c["wall_time_ns"], "View C wall_time_ns"
                    ),
                    "rgb_file": row_c["rgb_file"],
                    "depth_file": row_c["depth_file"],
                    "camera_params_file": row_c["camera_params_file"],
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "purpose": "view_a_c_exact_timecode_engineering_pair_v1",
        "claim_eligibility": CLAIM_ELIGIBILITY,
        "calibration_source": CALIBRATION_SOURCE,
        "eligible_for_formal": False,
        "uses_ground_truth": True,
        "pairing_semantics": (
            "exact_offline_usd_timecode_pairing_across_independent_"
            "single_view_captures"
        ),
        "synchronization_contract": {
            "same_capture": False,
            "same_render_step": False,
            "wall_time_used_for_pairing": False,
            "timestamp_nearest_matching_used": False,
            "interpolation_used": False,
            "primary_key": ["sync_index"],
            "validated_equal_fields": [
                "sequence_index",
                "cycle_index",
                "animation_frame_code",
                "usd_time_code",
                "timeline_time_s",
            ],
            "warning": (
                "This is deterministic time-code pairing, not evidence of "
                "simultaneous acquisition or camera synchronization."
            ),
        },
        "frame_count": EXPECTED_FRAME_COUNT,
        "gt_joint_count_per_frame": EXPECTED_JOINT_COUNT,
        "exact_world_gt_match": True,
        "views": {
            "a": {
                "directory": str(view_a.directory),
                "run_id": view_a.run_id,
                "camera_prim": view_a.camera_prim,
                "source_sha256": view_a.source_sha256,
            },
            "c": {
                "directory": str(view_c.directory),
                "run_id": view_c.run_id,
                "camera_prim": view_c.camera_prim,
                "source_sha256": view_c.source_sha256,
            },
        },
        "pairs": pairs,
    }


def build_calibration(
    view_a: ViewDataset,
    view_c: ViewDataset,
    ordered_keys: list[tuple[int, str]],
    sync_manifest_path: Path,
) -> dict[str, Any]:
    world_points = np.stack(
        [view_a.gt_by_key[key].world for key in ordered_keys]
    )
    camera_a_points = np.stack(
        [view_a.gt_by_key[key].camera for key in ordered_keys]
    )
    camera_c_points = np.stack(
        [view_c.gt_by_key[key].camera for key in ordered_keys]
    )
    world_from_a = fit_rigid_transform(
        camera_a_points,
        world_points,
        source_frame="view_a_camera_zed",
        target_frame="operational_world",
    )
    world_from_c = fit_rigid_transform(
        camera_c_points,
        world_points,
        source_frame="view_c_camera_zed",
        target_frame="operational_world",
    )
    a_from_c = relative_transform(
        world_from_c,
        world_from_a,
        source_frame="view_c_camera_zed",
        target_frame="view_a_camera_zed",
    )
    c_from_a = relative_transform(
        world_from_a,
        world_from_c,
        source_frame="view_a_camera_zed",
        target_frame="view_c_camera_zed",
    )
    c_to_a_fit = cross_view_residual_summary(
        a_from_c, camera_c_points, camera_a_points
    )
    a_to_c_fit = cross_view_residual_summary(
        c_from_a, camera_a_points, camera_c_points
    )
    _, origin_a = transform_arrays(world_from_a)
    _, origin_c = transform_arrays(world_from_c)
    baseline_m = float(np.linalg.norm(origin_c - origin_a))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "purpose": "view_a_c_gt_correspondence_engineering_calibration_v1",
        "claim_eligibility": CLAIM_ELIGIBILITY,
        "calibration_source": CALIBRATION_SOURCE,
        "eligible_for_formal": False,
        "uses_ground_truth": True,
        "generated_wall_time_ns": time.time_ns(),
        "coordinate_contract": {
            "vector_convention": "column_vector",
            "matrix_storage": "row_major_nested_arrays",
            "transform_formula": (
                "p_target = R_target_from_source @ p_source + "
                "t_target_from_source_m"
            ),
            "homogeneous_formula": "p_target_h = T_target_from_source @ p_source_h",
            "camera_coordinate_system": "ZED RIGHT_HANDED_Z_UP_X_FWD",
            "camera_axes": {"x": "forward", "y": "left", "z": "up"},
            "world_coordinate_system": "operational USD world",
            "unit": "operational_metre",
            "homogeneous_last_row": [0.0, 0.0, 0.0, 1.0],
        },
        "views": {
            "a": {
                "run_id": view_a.run_id,
                "camera_prim": view_a.camera_prim,
                "world_from_camera": world_from_a,
            },
            "c": {
                "run_id": view_c.run_id,
                "camera_prim": view_c.camera_prim,
                "world_from_camera": world_from_c,
            },
        },
        "relative_transforms": {
            "a_from_c": a_from_c,
            "c_from_a": c_from_a,
        },
        "baseline_m": baseline_m,
        "cross_view_fit": {
            "a_from_c": c_to_a_fit,
            "c_from_a": a_to_c_fit,
        },
        "correspondence_contract": {
            "correspondence_count_per_view": len(ordered_keys),
            "frame_count": EXPECTED_FRAME_COUNT,
            "joint_count_per_frame": EXPECTED_JOINT_COUNT,
            "exact_world_gt_match_required": True,
            "gt_fields_used": [
                "world_x_m",
                "world_y_m",
                "world_z_m",
                "gt_x_m",
                "gt_y_m",
                "gt_z_m",
            ],
        },
        "source_sha256": {
            "tool": sha256_file(Path(__file__).resolve()),
            "view_a": view_a.source_sha256,
            "view_c": view_c.source_sha256,
        },
        "sync_manifest": {
            "path": str(sync_manifest_path.resolve()),
            "sha256": sha256_file(sync_manifest_path),
        },
        "warnings": [
            "This calibration reads Isaac Skeleton ground truth and must not "
            "be used by a formal estimator or formal accuracy evaluation.",
            "The paired inputs are independent single-view captures, not a "
            "shared render step or synchronized live-camera acquisition.",
            "Replace this artifact with calibration derived independently of "
            "evaluation GT before formal multiview experiments.",
        ],
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def is_within(path: Path, possible_parent: Path) -> bool:
    try:
        path.relative_to(possible_parent)
    except ValueError:
        return False
    return True


def build_pair(
    view_a_dir: Path,
    view_c_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    view_a_path = view_a_dir.resolve()
    view_c_path = view_c_dir.resolve()
    output_path = output_dir.resolve()
    if view_a_path == view_c_path:
        raise EngineeringPairingError("View A and View C inputs must differ.")
    if output_path.exists():
        raise EngineeringPairingError(
            f"Refusing to overwrite existing output directory: {output_path}"
        )
    if is_within(output_path, view_a_path) or is_within(output_path, view_c_path):
        raise EngineeringPairingError(
            "Output directory must not be inside either preserved input."
        )

    view_a = load_view(view_a_path, "A")
    view_c = load_view(view_c_path, "C")
    if view_a.camera_prim == view_c.camera_prim:
        raise EngineeringPairingError(
            "View A and View C must declare different camera prims."
        )
    ordered_keys = exact_pair_keys(view_a, view_c)
    sync_manifest = build_sync_manifest(view_a, view_c)

    output_path.mkdir(parents=True, exist_ok=False)
    sync_path = output_path / SYNC_MANIFEST_NAME
    calibration_path = output_path / CALIBRATION_NAME
    write_json_exclusive(sync_path, sync_manifest)
    calibration = build_calibration(
        view_a,
        view_c,
        ordered_keys,
        sync_path,
    )
    write_json_exclusive(calibration_path, calibration)
    return {
        "status": "complete",
        "claim_eligibility": CLAIM_ELIGIBILITY,
        "calibration_source": CALIBRATION_SOURCE,
        "eligible_for_formal": False,
        "output_dir": str(output_path),
        "sync_manifest": str(sync_path),
        "sync_manifest_sha256": sha256_file(sync_path),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "frame_count": EXPECTED_FRAME_COUNT,
        "correspondence_count": len(ordered_keys),
        "baseline_m": calibration["baseline_m"],
        "maximum_cross_view_residual_m": max(
            calibration["cross_view_fit"][direction]["maximum_residual_m"]
            for direction in ("a_from_c", "c_from_a")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact-pair View A/C unit-corrected captures and derive an "
            "explicitly engineering-only GT-correspondence calibration."
        )
    )
    parser.add_argument("--view-a-dir", type=Path, required=True)
    parser.add_argument("--view-c-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_pair(
            args.view_a_dir,
            args.view_c_dir,
            args.output_dir,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
