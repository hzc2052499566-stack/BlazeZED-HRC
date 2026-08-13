"""Build common-frame View A/C candidates for engineering smoke tests.

This tool deliberately supports only the excluded cross-capture engineering
pair.  It reads completed View A and View C *v2* replay directories plus the
GT-derived engineering calibration written by
``build_view_ac_engineering_pair_v1.py``.  It never accepts or reads a ground
truth file.  The resulting candidates are useful for checking exact joins,
axis conventions, provenance, and coordinate transforms, but are ineligible
for accuracy or synchronisation claims.

All output coordinates use the fixed View A ZED camera frame:
``+X forward, +Y left, +Z up`` in operational metres.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


ESTIMATES_NAME = "dynamic_kinematic_estimates.csv"
STATE_NAME = "dynamic_kinematic_replay_state.json"
PATCH_NAME = "prelock_relock_patch.json"
OUTPUT_NAME = "view_ac_common_frame_candidates.csv"
OUTPUT_STATE_NAME = "view_ac_common_frame_state.json"
EXPECTED_METHOD = "raw_measured"
EXPECTED_FRAME_COUNT = 240
EXPECTED_JOINT_COUNT = 15
EXPECTED_ROW_COUNT = EXPECTED_FRAME_COUNT * EXPECTED_JOINT_COUNT
EXPECTED_PATCH_METHOD = "prelock_relock_v2_20260801"
CLASSIFICATION = "excluded_engineering_common_frame_smoke"
CLAIM_ELIGIBILITY = "excluded_engineering_replay_only"
ALGORITHM_VERSION = (
    "view_ac_common_frame_engineering_v1_20260802_lineage_hardened"
)
REPLAY_IDENTITY_FIELDS = (
    "profile_sha256",
    "config_sha256",
    "joint_mapping_sha256",
    "replay_source_sha256",
    "pipeline_module_sha256",
    "k3_module_sha256",
    "k4_module_sha256",
)

CANDIDATE_FIELDS = [
    "engineering_pair_id",
    "sync_index",
    "sequence_index",
    "cycle_index",
    "animation_frame_code",
    "usd_time_code",
    "view_id",
    "source_run_id",
    "source_method",
    "canonical_joint",
    "classification",
    "claim_eligibility",
    "eligible_for_formal",
    "calibration_source",
    "source_pair_type",
    "same_capture_session",
    "valid",
    "camera_x_m",
    "camera_y_m",
    "camera_z_m",
    "common_x_m",
    "common_y_m",
    "common_z_m",
    "common_coordinate_frame",
    "counts_as_measured_valid",
    "output_class",
    "source_provenance",
    "depth_m",
    "pixel_x",
    "pixel_y",
    "visibility",
    "invalid_reason",
    "warmup_excluded",
]


class EngineeringReplayError(RuntimeError):
    """Raised when the engineering-only replay contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EngineeringReplayError(
            "Could not read valid JSON: {}".format(path)
        ) from error
    if not isinstance(payload, dict):
        raise EngineeringReplayError(
            "JSON root must be an object: {}".format(path)
        )
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise EngineeringReplayError(
                    "CSV has no header: {}".format(path)
                )
            return list(reader)
    except OSError as error:
        raise EngineeringReplayError(
            "Could not read CSV: {}".format(path)
        ) from error


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(
        ".{}.{}.{}.tmp".format(path.name, os.getpid(), time.time_ns())
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    temporary = path.with_name(
        ".{}.{}.{}.tmp".format(path.name, os.getpid(), time.time_ns())
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def bool_field(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EngineeringReplayError(
            "{} must be numeric; observed {!r}.".format(label, value)
        ) from error
    if not math.isfinite(result):
        raise EngineeringReplayError(
            "{} must be finite; observed {!r}.".format(label, value)
        )
    return result


def integer_field(value: object, label: str) -> int:
    number = finite_float(value, label)
    integer = int(number)
    if number != float(integer):
        raise EngineeringReplayError(
            "{} must be an integer; observed {!r}.".format(label, value)
        )
    return integer


def validate_replay_directory(path: Path, view_id: str) -> dict[str, Any]:
    state_path = path / STATE_NAME
    estimates_path = path / ESTIMATES_NAME
    patch_path = path / PATCH_NAME
    for required in (state_path, estimates_path, patch_path):
        if not required.is_file():
            raise EngineeringReplayError(
                "View {} replay is missing: {}".format(view_id, required)
            )

    state = read_json(state_path)
    patch = read_json(patch_path)
    if state.get("status") != "complete":
        raise EngineeringReplayError(
            "View {} replay is not complete.".format(view_id)
        )
    if int(state.get("estimate_row_count", -1)) != 8400:
        raise EngineeringReplayError(
            "View {} replay must contain 8400 estimate rows.".format(
                view_id
            )
        )
    if int(state.get("frame_row_count", -1)) != EXPECTED_FRAME_COUNT:
        raise EngineeringReplayError(
            "View {} replay must contain 240 frame rows.".format(view_id)
        )
    if int(state.get("cache_warmup_frames", -1)) != 0:
        raise EngineeringReplayError(
            "View {} replay did not use zero warm-up.".format(view_id)
        )
    if not bool_field(state.get("zero_warmup_required")):
        raise EngineeringReplayError(
            "View {} replay did not enforce zero warm-up.".format(view_id)
        )
    if bool_field(state.get("gt_read_during_estimation")):
        raise EngineeringReplayError(
            "View {} replay reports GT access during estimation.".format(
                view_id
            )
        )
    if patch.get("method_version") != EXPECTED_PATCH_METHOD:
        raise EngineeringReplayError(
            "View {} replay does not have the required v2 patch.".format(
                view_id
            )
        )
    if patch.get("patched_module_sha256") != state.get(
        "pipeline_module_sha256"
    ):
        raise EngineeringReplayError(
            "View {} patch/pipeline provenance does not match.".format(
                view_id
            )
        )
    for path_field, hash_field in (
        ("profile_path", "profile_sha256"),
        ("config_path", "config_sha256"),
    ):
        registered_path = Path(str(state.get(path_field, "")))
        if not registered_path.is_file():
            raise EngineeringReplayError(
                "View {} registered {} is missing: {}".format(
                    view_id, path_field, registered_path
                )
            )
        if sha256_file(registered_path) != str(state.get(hash_field, "")):
            raise EngineeringReplayError(
                "View {} registered {} hash does not match.".format(
                    view_id, path_field
                )
            )
    expected_hash = str(state.get("estimates_sha256", ""))
    observed_hash = sha256_file(estimates_path)
    if expected_hash != observed_hash:
        raise EngineeringReplayError(
            "View {} estimates hash does not match replay state.".format(
                view_id
            )
        )
    return {
        "state": state,
        "patch": patch,
        "state_path": state_path,
        "estimates_path": estimates_path,
        "patch_path": patch_path,
        "estimates_sha256": observed_hash,
    }


def validate_calibration(
    calibration_path: Path,
    sync_manifest_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    calibration = read_json(calibration_path)
    if calibration.get("status") != "complete":
        raise EngineeringReplayError("Engineering calibration is incomplete.")
    if calibration.get("claim_eligibility") != CLAIM_ELIGIBILITY:
        raise EngineeringReplayError(
            "Calibration is not labelled as excluded engineering replay."
        )
    if calibration.get("calibration_source") != (
        "gt_correspondence_engineering_only"
    ):
        raise EngineeringReplayError(
            "This tool only accepts the explicit GT-derived engineering "
            "calibration."
        )
    if bool_field(calibration.get("eligible_for_formal")):
        raise EngineeringReplayError(
            "GT-derived engineering calibration cannot be formal-eligible."
        )
    if not bool_field(calibration.get("uses_ground_truth")):
        raise EngineeringReplayError(
            "Engineering calibration must disclose GT use."
        )

    sync_block = calibration.get("sync_manifest", {})
    if not isinstance(sync_block, dict):
        raise EngineeringReplayError("Calibration sync_manifest is invalid.")
    registered_hash = str(sync_block.get("sha256", ""))
    if registered_hash != sha256_file(sync_manifest_path):
        raise EngineeringReplayError(
            "Sync manifest hash does not match calibration."
        )

    try:
        transform = calibration["relative_transforms"]["a_from_c"]
        rotation = np.asarray(transform["rotation_3x3"], dtype=float)
        translation = np.asarray(transform["translation_m"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise EngineeringReplayError(
            "Calibration lacks relative_transforms.a_from_c."
        ) from error
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise EngineeringReplayError(
            "A-from-C transform must contain a 3x3 rotation and 3-vector."
        )
    if not np.all(np.isfinite(rotation)) or not np.all(
        np.isfinite(translation)
    ):
        raise EngineeringReplayError("A-from-C transform is not finite.")
    orthogonality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    )
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if orthogonality_error > 1e-6 or determinant_error > 1e-6:
        raise EngineeringReplayError(
            "A-from-C rotation is not a proper rigid rotation."
        )
    return calibration, rotation, translation


def normalized_path(path: object) -> str:
    """Return a case-normalized absolute path for lineage comparisons."""

    return os.path.normcase(str(Path(str(path)).resolve()))


def validate_source_binding(
    calibration: dict[str, Any],
    sync_manifest: dict[str, Any],
    replay_a: dict[str, Any],
    replay_c: dict[str, Any],
) -> dict[str, str]:
    """Bind both replay inputs to the captures registered by the pair."""

    if sync_manifest.get("status") != "complete":
        raise EngineeringReplayError("Engineering sync manifest is incomplete.")
    if sync_manifest.get("claim_eligibility") != CLAIM_ELIGIBILITY:
        raise EngineeringReplayError(
            "Sync manifest is not labelled as excluded engineering replay."
        )
    if bool_field(sync_manifest.get("eligible_for_formal")):
        raise EngineeringReplayError(
            "Cross-capture sync manifest cannot be formal-eligible."
        )
    if not bool_field(sync_manifest.get("uses_ground_truth")):
        raise EngineeringReplayError("Sync manifest must disclose GT use.")
    contract = sync_manifest.get("synchronization_contract", {})
    if not isinstance(contract, dict):
        raise EngineeringReplayError("Sync contract is invalid.")
    if bool_field(contract.get("same_capture")) or bool_field(
        contract.get("same_render_step")
    ):
        raise EngineeringReplayError(
            "This tool only accepts independent cross-capture pairs."
        )

    pairs = sync_manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_FRAME_COUNT:
        raise EngineeringReplayError(
            "Sync manifest must contain exactly 240 pairs."
        )
    sync_indices = [
        integer_field(pair.get("sync_index"), "pair.sync_index")
        for pair in pairs
        if isinstance(pair, dict)
    ]
    if len(sync_indices) != EXPECTED_FRAME_COUNT or sync_indices != list(
        range(EXPECTED_FRAME_COUNT)
    ):
        raise EngineeringReplayError(
            "Sync manifest pair indices must be ordered exactly 0..239."
        )

    sync_views = sync_manifest.get("views", {})
    calibration_views = calibration.get("views", {})
    if not isinstance(sync_views, dict) or not isinstance(
        calibration_views, dict
    ):
        raise EngineeringReplayError("Pair view lineage blocks are invalid.")

    run_ids: dict[str, str] = {}
    for view_id, replay in (("a", replay_a), ("c", replay_c)):
        sync_view = sync_views.get(view_id, {})
        calibration_view = calibration_views.get(view_id, {})
        if not isinstance(sync_view, dict) or not isinstance(
            calibration_view, dict
        ):
            raise EngineeringReplayError(
                "View {} lineage block is missing.".format(view_id)
            )
        run_id = str(sync_view.get("run_id", ""))
        camera_prim = str(sync_view.get("camera_prim", ""))
        if not run_id or not camera_prim:
            raise EngineeringReplayError(
                "View {} pair lineage lacks run/camera identity.".format(
                    view_id
                )
            )
        if str(calibration_view.get("run_id", "")) != run_id or str(
            calibration_view.get("camera_prim", "")
        ) != camera_prim:
            raise EngineeringReplayError(
                "View {} calibration and sync identities differ.".format(
                    view_id
                )
            )

        state = replay["state"]
        manifest_qc = state.get("manifest_qc", {})
        if not isinstance(manifest_qc, dict) or str(
            manifest_qc.get("run_id", "")
        ) != run_id:
            raise EngineeringReplayError(
                "View {} replay run_id is not bound to the pair.".format(
                    view_id
                )
            )
        source_directory = Path(str(sync_view.get("directory", ""))).resolve()
        if normalized_path(state.get("source_experiment_dir", "")) != (
            normalized_path(source_directory)
        ):
            raise EngineeringReplayError(
                "View {} replay source directory is not bound to the pair."
                .format(view_id)
            )
        source_hashes = sync_view.get("source_sha256", {})
        if not isinstance(source_hashes, dict):
            raise EngineeringReplayError(
                "View {} source hash block is invalid.".format(view_id)
            )
        registered_manifest_hash = str(
            source_hashes.get("rgbd_manifest.csv", "")
        )
        if str(state.get("source_manifest_sha256", "")) != (
            registered_manifest_hash
        ):
            raise EngineeringReplayError(
                "View {} replay manifest hash is not bound to the pair."
                .format(view_id)
            )
        manifest_path = source_directory / "rgbd_manifest.csv"
        if not manifest_path.is_file() or sha256_file(manifest_path) != (
            registered_manifest_hash
        ):
            raise EngineeringReplayError(
                "View {} registered source manifest is missing or changed."
                .format(view_id)
            )
        manifest_rows = read_csv(manifest_path)
        if len(manifest_rows) != EXPECTED_FRAME_COUNT:
            raise EngineeringReplayError(
                "View {} source manifest must contain 240 rows.".format(
                    view_id
                )
            )
        manifest_by_sequence: dict[int, dict[str, str]] = {}
        for row in manifest_rows:
            sequence = integer_field(
                row.get("sequence_index"),
                "{}.manifest.sequence_index".format(view_id),
            )
            if sequence in manifest_by_sequence:
                raise EngineeringReplayError(
                    "View {} source manifest has duplicate sequence {}."
                    .format(view_id, sequence)
                )
            if str(row.get("run_id", "")) != run_id or str(
                row.get("camera_prim", "")
            ) != camera_prim:
                raise EngineeringReplayError(
                    "View {} source manifest identity changed.".format(
                        view_id
                    )
                )
            manifest_by_sequence[sequence] = row
        if sorted(manifest_by_sequence) != list(
            range(EXPECTED_FRAME_COUNT)
        ):
            raise EngineeringReplayError(
                "View {} source manifest keys are not exactly 0..239."
                .format(view_id)
            )

        pair_key = "view_{}".format(view_id)
        for pair in pairs:
            sequence = integer_field(
                pair.get("sequence_index"), "pair.sequence_index"
            )
            if sequence != integer_field(
                pair.get("sync_index"), "pair.sync_index"
            ):
                raise EngineeringReplayError(
                    "Pair sequence and sync indices differ."
                )
            manifest_row = manifest_by_sequence[sequence]
            pair_view = pair.get(pair_key, {})
            if not isinstance(pair_view, dict):
                raise EngineeringReplayError(
                    "Pair view metadata is invalid for view {}.".format(
                        view_id
                    )
                )
            for field in (
                "run_id",
                "camera_prim",
                "rgb_file",
                "depth_file",
                "camera_params_file",
            ):
                if str(pair_view.get(field, "")) != str(
                    manifest_row.get(field, "")
                ):
                    raise EngineeringReplayError(
                        "View {} pair/source field {} differs at sequence {}."
                        .format(view_id, field, sequence)
                    )
        run_ids[view_id] = run_id
    return run_ids


def raw_rows(
    replay: dict[str, Any],
    view_id: str,
    expected_run_id: str | None = None,
) -> dict[tuple[int, str], dict[str, str]]:
    selected = [
        row
        for row in read_csv(replay["estimates_path"])
        if row.get("method") == EXPECTED_METHOD
    ]
    if len(selected) != EXPECTED_ROW_COUNT:
        raise EngineeringReplayError(
            "View {} must contain exactly {} raw rows; observed {}.".format(
                view_id, EXPECTED_ROW_COUNT, len(selected)
            )
        )
    keyed: dict[tuple[int, str], dict[str, str]] = {}
    frames: dict[int, set[str]] = {}
    for row in selected:
        if expected_run_id is not None and str(row.get("run_id", "")) != (
            expected_run_id
        ):
            raise EngineeringReplayError(
                "View {} raw row run_id is not bound to the pair.".format(
                    view_id
                )
            )
        sequence = integer_field(
            row.get("sequence_index"),
            "{}.sequence_index".format(view_id),
        )
        joint = str(row.get("canonical_joint", "")).strip()
        if not joint:
            raise EngineeringReplayError(
                "View {} contains an empty canonical joint.".format(view_id)
            )
        key = (sequence, joint)
        if key in keyed:
            raise EngineeringReplayError(
                "View {} has duplicate key {}.".format(view_id, key)
            )
        keyed[key] = row
        frames.setdefault(sequence, set()).add(joint)
        if integer_field(row.get("cycle_index"), "cycle_index") != 0:
            raise EngineeringReplayError("Only cycle 0 is supported.")
        frame_code = integer_field(
            row.get("animation_frame_code"), "animation_frame_code"
        )
        usd_time = finite_float(row.get("usd_time_code"), "usd_time_code")
        if frame_code != sequence or usd_time != float(sequence):
            raise EngineeringReplayError(
                "View {} key/time contract failed at sequence {}.".format(
                    view_id, sequence
                )
            )
        if bool_field(row.get("warmup_excluded")):
            raise EngineeringReplayError(
                "View {} contains a warm-up-excluded raw row.".format(
                    view_id
                )
            )
        if str(row.get("counts_as_measured_valid", "")).strip() != "1":
            raise EngineeringReplayError(
                "View {} raw row violates measured-method provenance."
                .format(view_id)
            )
        provenance = str(row.get("provenance", "")).strip()
        if not provenance.startswith("measured_"):
            raise EngineeringReplayError(
                "View {} raw row has non-measured provenance.".format(
                    view_id
                )
            )
        valid_text = str(row.get("valid", "")).strip()
        if valid_text not in {"0", "1"}:
            raise EngineeringReplayError(
                "View {} raw validity must be exactly 0 or 1.".format(
                    view_id
                )
            )
        if valid_text == "1":
            for field in ("x_m", "y_m", "z_m", "depth_m"):
                finite_float(
                    row.get(field),
                    "{}.{}".format(view_id, field),
                )
        else:
            for field in ("x_m", "y_m", "z_m", "depth_m"):
                if str(row.get(field, "")).strip():
                    raise EngineeringReplayError(
                        "View {} invalid raw row has a populated {}."
                        .format(view_id, field)
                    )
    if sorted(frames) != list(range(EXPECTED_FRAME_COUNT)):
        raise EngineeringReplayError(
            "View {} frame keys are not exactly 0..239.".format(view_id)
        )
    if any(len(joints) != EXPECTED_JOINT_COUNT for joints in frames.values()):
        raise EngineeringReplayError(
            "View {} does not have 15 raw joints per frame.".format(view_id)
        )
    joint_sets = list(frames.values())
    if any(joints != joint_sets[0] for joints in joint_sets[1:]):
        raise EngineeringReplayError(
            "View {} raw joint set changes between frames.".format(view_id)
        )
    return keyed


def transform_point(
    point: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return rotation @ point + translation


def candidate_row(
    *,
    pair_id: str,
    view_id: str,
    row: dict[str, str],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    valid = bool_field(row.get("valid"))
    camera = None
    common = None
    if valid:
        camera = np.asarray(
            [
                finite_float(row.get("x_m"), "x_m"),
                finite_float(row.get("y_m"), "y_m"),
                finite_float(row.get("z_m"), "z_m"),
            ],
            dtype=float,
        )
        common = transform_point(camera, rotation, translation)
    output_class = "excluded_engineering_measured_{}".format(view_id)
    return {
        "engineering_pair_id": pair_id,
        "sync_index": integer_field(row["sequence_index"], "sync_index"),
        "sequence_index": integer_field(
            row["sequence_index"], "sequence_index"
        ),
        "cycle_index": integer_field(row["cycle_index"], "cycle_index"),
        "animation_frame_code": integer_field(
            row["animation_frame_code"], "animation_frame_code"
        ),
        "usd_time_code": finite_float(
            row["usd_time_code"], "usd_time_code"
        ),
        "view_id": view_id,
        "source_run_id": row.get("run_id", ""),
        "source_method": EXPECTED_METHOD,
        "canonical_joint": row["canonical_joint"],
        "classification": CLASSIFICATION,
        "claim_eligibility": CLAIM_ELIGIBILITY,
        "eligible_for_formal": 0,
        "calibration_source": "gt_correspondence_engineering_only",
        "source_pair_type": "same_timecode_cross_capture",
        "same_capture_session": 0,
        "valid": int(valid),
        "camera_x_m": float(camera[0]) if camera is not None else "",
        "camera_y_m": float(camera[1]) if camera is not None else "",
        "camera_z_m": float(camera[2]) if camera is not None else "",
        "common_x_m": float(common[0]) if common is not None else "",
        "common_y_m": float(common[1]) if common is not None else "",
        "common_z_m": float(common[2]) if common is not None else "",
        "common_coordinate_frame": "view_a_zed_camera_operational_metre",
        "counts_as_measured_valid": row.get(
            "counts_as_measured_valid", ""
        ),
        "output_class": output_class,
        "source_provenance": row.get("provenance", ""),
        "depth_m": row.get("depth_m", ""),
        "pixel_x": row.get("pixel_x", ""),
        "pixel_y": row.get("pixel_y", ""),
        "visibility": row.get("visibility", ""),
        "invalid_reason": row.get("invalid_reason", ""),
        "warmup_excluded": row.get("warmup_excluded", ""),
    }


def build_candidates(
    *,
    pair_id: str,
    view_a_rows: dict[tuple[int, str], dict[str, str]],
    view_c_rows: dict[tuple[int, str], dict[str, str]],
    rotation_a_from_c: np.ndarray,
    translation_a_from_c: np.ndarray,
) -> list[dict[str, Any]]:
    if set(view_a_rows) != set(view_c_rows):
        only_a = sorted(set(view_a_rows).difference(view_c_rows))[:5]
        only_c = sorted(set(view_c_rows).difference(view_a_rows))[:5]
        raise EngineeringReplayError(
            "A/C raw keys differ; only A {}, only C {}.".format(
                only_a, only_c
            )
        )
    identity = np.eye(3, dtype=float)
    zero = np.zeros(3, dtype=float)
    output: list[dict[str, Any]] = []
    for key in sorted(view_a_rows):
        row_a = view_a_rows[key]
        row_c = view_c_rows[key]
        for field in (
            "cycle_index",
            "animation_frame_code",
            "usd_time_code",
            "canonical_joint",
        ):
            if str(row_a.get(field, "")) != str(row_c.get(field, "")):
                raise EngineeringReplayError(
                    "A/C exact-key metadata differs for {} field {}.".format(
                        key, field
                    )
                )
        output.append(
            candidate_row(
                pair_id=pair_id,
                view_id="a",
                row=row_a,
                rotation=identity,
                translation=zero,
            )
        )
        output.append(
            candidate_row(
                pair_id=pair_id,
                view_id="c",
                row=row_c,
                rotation=rotation_a_from_c,
                translation=translation_a_from_c,
            )
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--view-a-replay-dir", type=Path, required=True)
    parser.add_argument("--view-c-replay-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--sync-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise EngineeringReplayError(
            "Refusing to overwrite existing output: {}".format(output_dir)
        )
    replay_a = validate_replay_directory(
        args.view_a_replay_dir.resolve(), "a"
    )
    replay_c = validate_replay_directory(
        args.view_c_replay_dir.resolve(), "c"
    )
    identity_mismatches = [
        field
        for field in REPLAY_IDENTITY_FIELDS
        if replay_a["state"].get(field) != replay_c["state"].get(field)
    ]
    if identity_mismatches:
        raise EngineeringReplayError(
            "A/C replay method identities differ: {}.".format(
                ", ".join(identity_mismatches)
            )
        )
    patch_hash_a = sha256_file(replay_a["patch_path"])
    patch_hash_c = sha256_file(replay_c["patch_path"])
    if patch_hash_a != patch_hash_c:
        raise EngineeringReplayError(
            "A/C replay patch artifacts are not byte-identical."
        )
    calibration, rotation, translation = validate_calibration(
        args.calibration.resolve(), args.sync_manifest.resolve()
    )
    sync_manifest = read_json(args.sync_manifest.resolve())
    source_run_ids = validate_source_binding(
        calibration,
        sync_manifest,
        replay_a,
        replay_c,
    )
    rows_a = raw_rows(replay_a, "a", source_run_ids["a"])
    rows_c = raw_rows(replay_c, "c", source_run_ids["c"])
    candidates = build_candidates(
        pair_id=args.pair_id,
        view_a_rows=rows_a,
        view_c_rows=rows_c,
        rotation_a_from_c=rotation,
        translation_a_from_c=translation,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    candidates_path = output_dir / OUTPUT_NAME
    state_path = output_dir / OUTPUT_STATE_NAME
    write_csv_atomic(candidates_path, candidates, CANDIDATE_FIELDS)
    state = {
        "schema_version": 1,
        "status": "complete",
        "algorithm_version": ALGORITHM_VERSION,
        "classification": CLASSIFICATION,
        "claim_eligibility": CLAIM_ELIGIBILITY,
        "evidence_eligible": False,
        "source_pair_type": "same_timecode_cross_capture",
        "same_capture_session": False,
        "pair_id": args.pair_id,
        "common_coordinate_frame": (
            "view_a_zed_camera_operational_metre"
        ),
        "calibration_source": calibration["calibration_source"],
        "uses_gt_derived_engineering_calibration": True,
        "formal_use_prohibited": True,
        "gt_access_audit": {
            "audit_type": "static_input_surface_contract",
            "ground_truth_path_argument_supported": False,
            "ground_truth_file_access_traced": False,
            "direct_gt_file_read_by_candidate_replay": False,
            "uses_gt_derived_calibration": True,
            "deployable_estimator_claim_allowed": False,
        },
        "gate_results": {
            "both_v2_replays_complete": True,
            "replay_method_hashes_identical": True,
            "replay_patch_hashes_identical": True,
            "source_lineage_bound": True,
            "zero_warmup_enforced": True,
            "source_replays_report_no_gt_access": True,
            "raw_key_sets_exact": True,
            "row_provenance_verified": True,
            "frame_keys_exact_0_to_239": True,
            "fifteen_raw_joints_per_frame": True,
            "a_from_c_rotation_is_rigid": True,
            "engineering_exclusion_labels_present": True,
        },
        "failed_gates": [],
        "candidate_row_count": len(candidates),
        "candidate_row_count_by_view": {
            "a": sum(row["view_id"] == "a" for row in candidates),
            "c": sum(row["view_id"] == "c" for row in candidates),
        },
        "valid_raw_candidate_count_by_view": {
            "a": sum(
                row["view_id"] == "a" and int(row["valid"]) == 1
                for row in candidates
            ),
            "c": sum(
                row["view_id"] == "c" and int(row["valid"]) == 1
                for row in candidates
            ),
        },
        "inputs": {
            "view_a_replay_dir": str(args.view_a_replay_dir.resolve()),
            "view_c_replay_dir": str(args.view_c_replay_dir.resolve()),
            "view_a_run_id": source_run_ids["a"],
            "view_c_run_id": source_run_ids["c"],
            "view_a_replay_state_sha256": sha256_file(
                replay_a["state_path"]
            ),
            "view_c_replay_state_sha256": sha256_file(
                replay_c["state_path"]
            ),
            "view_a_estimates_sha256": replay_a["estimates_sha256"],
            "view_c_estimates_sha256": replay_c["estimates_sha256"],
            "view_a_source_manifest_sha256": replay_a["state"].get(
                "source_manifest_sha256", ""
            ),
            "view_c_source_manifest_sha256": replay_c["state"].get(
                "source_manifest_sha256", ""
            ),
            "view_a_patch_sha256": patch_hash_a,
            "view_c_patch_sha256": patch_hash_c,
            "calibration": str(args.calibration.resolve()),
            "calibration_sha256": sha256_file(args.calibration.resolve()),
            "sync_manifest": str(args.sync_manifest.resolve()),
            "sync_manifest_sha256": sha256_file(
                args.sync_manifest.resolve()
            ),
        },
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            "candidates_csv": str(candidates_path),
            "candidates_sha256": sha256_file(candidates_path),
        },
        "interpretation": (
            "Plumbing smoke only. A and C are independent captures, and "
            "the transform was fitted from Skeleton GT. These outputs may "
            "validate joins, units, axes, and provenance, but no accuracy, "
            "synchronisation, selection, or fusion claim."
        ),
    }
    write_json_atomic(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
