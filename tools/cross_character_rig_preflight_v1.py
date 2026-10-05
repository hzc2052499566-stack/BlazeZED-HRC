"""Pure-Python asset/rig compatibility gates for the cross-character supplement.

This module has no Isaac Sim or ``pxr`` dependency.  The two Isaac Script
Editor scripts import it to evaluate what they measured, while unit tests
exercise every gate in a normal Python interpreter.

Scope is deliberately narrow: it decides whether a candidate character asset
carries a rig this project can already map and measure, and whether a cohort of
four characters is genuinely distinct and spans a body-size range.  It does not
retarget motion, does not touch cameras, and does not authorise any capture.
See AGENTS.md 6.6.2 for the surrounding contract.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


PREFLIGHT_TAG = "fs_cts5_cross_character_rig_preflight_v1"
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False

# Unit convention.  AGENTS.md 4.4 records that the controlled scene is used as
# "one stage unit = one operational metre" even though the USD layer declares
# metersPerUnit = 0.01.  Candidate assets are measured under the same
# convention, and PLAUSIBLE_* gates below exist so that a candidate whose units
# disagree fails loudly instead of silently producing nonsense lengths.
UNIT_CONVENTION = "one_stage_unit_equals_one_operational_metre_per_agents_md_4_4"

# The 15 canonical joints of configs/joint_mapping.csv, keyed by the last path
# component of the UsdSkel joint token.
CANONICAL_JOINT_SUFFIXES = {
    "pelvis": "Pelvis",
    "neck": "NeckTwist01",
    "nose": "FacialBone",
    "left_shoulder": "L_Upperarm",
    "right_shoulder": "R_Upperarm",
    "left_elbow": "L_Forearm",
    "right_elbow": "R_Forearm",
    "left_wrist": "L_Hand",
    "right_wrist": "R_Hand",
    "left_hip": "L_Thigh",
    "right_hip": "R_Thigh",
    "left_knee": "L_Calf",
    "right_knee": "R_Calf",
    "left_ankle": "L_Foot",
    "right_ankle": "R_Foot",
}
CANONICAL_JOINTS = tuple(sorted(CANONICAL_JOINT_SUFFIXES))
# The 13 core joints of the joint contract; neck and nose stay excluded.
CORE_JOINTS = tuple(
    joint for joint in CANONICAL_JOINTS if joint not in ("neck", "nose")
)

# The eight mapped major limb segments of FS-CTS5.  "FS" is the full mapped
# major-limb extension of CTS5, never a literal full skeleton.
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
BONE_NAMES = tuple(sorted(BONES))
MIRROR_BONES = (
    ("left_upper_arm", "right_upper_arm"),
    ("left_forearm", "right_forearm"),
    ("left_thigh", "right_thigh"),
    ("left_shank", "right_shank"),
)

# Chains that must hold as ancestor order inside the skeleton token tree.
# Intermediate twist joints are allowed, reversed or detached chains are not.
REQUIRED_ANCESTOR_CHAINS = (
    ("pelvis", "left_hip", "left_knee", "left_ankle"),
    ("pelvis", "right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
)

# Draft engineering thresholds.  They are recorded before any candidate is
# measured and belong to a draft protocol, not to a frozen formal contract.
PLAUSIBLE_ARM_BONE_RANGE_M = (0.18, 0.42)
PLAUSIBLE_LEG_BONE_RANGE_M = (0.28, 0.60)
PLAUSIBLE_SIZE_PROXY_RANGE_M = (1.20, 2.20)
MAX_LEFT_RIGHT_ASYMMETRY_RATIO = 0.05
UP_AXIS_DOMINANCE_RATIO = 2.0
MIN_ANKLE_TO_NOSE_SPAN_M = 0.8
MAX_STAGE_VS_ASSET_BONE_DIFF_M = 1.0e-4
MIN_MESH_POINT_COUNT = 1000
SCALE_PROPORTIONALITY_TOLERANCE = 0.02
MIN_COHORT_SIZE_SPAN_M = 0.10
MIN_PAIRWISE_SIZE_SEPARATION_M = 0.02
REQUIRED_COHORT_KEYS = ("F01", "F02", "M01", "M02")
INCUMBENT_KEY = "F01"

ARM_BONES = (
    "left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm",
)
LEG_BONES = ("left_thigh", "left_shank", "right_thigh", "right_shank")

# Gates this module deliberately does not cover; they belong to later steps of
# the AGENTS.md 6.6.7 execution order and must not be reported as passed here.
DEFERRED_GATES = (
    "motion_retarget_to_frozen_bilateral_reach_and_low_march",
    "bone_length_invariance_over_241_frames",
    "ground_contact_and_support_foot_geometry",
    "camera_roi_and_projection_coverage",
    "rgb_depth_sanity_and_render_identity",
    "gt_free_blazepose_endpoint_availability",
    "common_five_view_feasibility_across_four_characters",
)

REQUIRED_MEASUREMENT_FIELDS = (
    "character_key",
    "source_kind",
    "source_identity",
    "skeleton_path",
    "joint_token_count",
    "joint_tokens",
    "rest_joint_positions",
    "mesh_point_count",
    "mesh_signature_sha256",
    "mesh_shape_signature_sha256",
    "size_proxy_operational_m",
    "size_proxy_axis_index",
    "declared_meters_per_unit",
)


class RigPreflightError(RuntimeError):
    """Raised when a measurement record is unusable rather than merely failing."""


def _tail(token: str) -> str:
    return str(token).replace("\\", "/").rstrip("/").split("/")[-1]


def resolve_joint_indices(joint_tokens: Sequence[str]) -> dict:
    """Map every canonical joint onto exactly one skeleton joint token.

    Matching uses last-path-component equality, which is stricter than the
    legacy ``endswith`` resolver used by the 20-bundle camera-bank preparer and
    agrees with it on the incumbent rig.
    """
    tokens = [str(token) for token in joint_tokens]
    if not tokens:
        raise RigPreflightError("Skeleton exposes no joint tokens.")
    resolved: dict = {}
    ambiguous: dict = {}
    missing = []
    for joint in CANONICAL_JOINTS:
        suffix = CANONICAL_JOINT_SUFFIXES[joint]
        matches = [
            index for index, token in enumerate(tokens) if _tail(token) == suffix
        ]
        if len(matches) == 1:
            resolved[joint] = matches[0]
        elif not matches:
            missing.append(joint)
        else:
            ambiguous[joint] = [tokens[index] for index in matches]
    return {
        "resolved": resolved,
        "missing": missing,
        "ambiguous": ambiguous,
        "complete": not missing and not ambiguous,
    }


def _ancestor_chain_ok(tokens: Sequence[str], indices: Mapping[str, int]) -> list:
    failures = []
    for chain in REQUIRED_ANCESTOR_CHAINS:
        if any(joint not in indices for joint in chain):
            failures.append(
                "chain {} not fully resolved".format("->".join(chain))
            )
            continue
        paths = [
            str(tokens[indices[joint]]).replace("\\", "/").strip("/")
            for joint in chain
        ]
        for parent, child, parent_joint, child_joint in zip(
            paths, paths[1:], chain, chain[1:]
        ):
            if not child.startswith(parent + "/"):
                failures.append(
                    "{} is not an ancestor of {} ({!r} vs {!r})".format(
                        parent_joint, child_joint, parent, child
                    )
                )
    return failures


def build_parent_indices(joint_tokens: Sequence[str]) -> list:
    """Parent index per joint token, or -1 for a root, from the token paths."""
    tokens = [str(token).replace("\\", "/").strip("/") for token in joint_tokens]
    lookup = {token: index for index, token in enumerate(tokens)}
    if len(lookup) != len(tokens):
        raise RigPreflightError("Skeleton joint tokens are not unique.")
    parents = []
    for token in tokens:
        parent = -1
        parts = token.split("/")
        for cut in range(len(parts) - 1, 0, -1):
            candidate = "/".join(parts[:cut])
            if candidate in lookup:
                parent = lookup[candidate]
                break
        parents.append(parent)
    return parents


def matrix_multiply(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list:
    """4x4 row-major multiply following the USD row-vector convention."""
    if len(left) != 4 or len(right) != 4:
        raise RigPreflightError("Expected 4x4 matrices.")
    product = []
    for row in range(4):
        out_row = []
        for column in range(4):
            total = 0.0
            for index in range(4):
                total += float(left[row][index]) * float(right[index][column])
            out_row.append(total)
        product.append(out_row)
    return product


def compose_skeleton_space_positions(
    local_matrices: Sequence[Sequence[Sequence[float]]],
    parents: Sequence[int],
) -> list:
    """Compose joint-local 4x4 transforms into skeleton-space positions."""
    if len(local_matrices) != len(parents):
        raise RigPreflightError("Transform count does not match joint count.")
    resolved: list = [None] * len(parents)
    for index in range(len(parents)):
        chain = []
        cursor = index
        guard = 0
        while cursor != -1 and resolved[cursor] is None:
            chain.append(cursor)
            cursor = int(parents[cursor])
            guard += 1
            if guard > len(parents):
                raise RigPreflightError("Cycle detected in skeleton parents.")
        for joint in reversed(chain):
            parent = int(parents[joint])
            local = local_matrices[joint]
            if parent == -1:
                resolved[joint] = [list(row) for row in local]
            else:
                resolved[joint] = matrix_multiply(local, resolved[parent])
    return [(matrix[3][0], matrix[3][1], matrix[3][2]) for matrix in resolved]


def distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 3 or len(right) != 3:
        raise RigPreflightError("Joint positions must be 3D.")
    total = 0.0
    for index in range(3):
        delta = float(left[index]) - float(right[index])
        total += delta * delta
    return math.sqrt(total)


def bone_lengths(positions: Mapping[str, Sequence[float]]) -> dict:
    """Compute the eight mapped major-limb lengths from rest-pose positions."""
    lengths = {}
    for bone in BONE_NAMES:
        proximal, distal = BONES[bone]
        if proximal not in positions or distal not in positions:
            raise RigPreflightError(
                "Missing endpoint for {}: needs {} and {}.".format(
                    bone, proximal, distal
                )
            )
        length = distance(positions[proximal], positions[distal])
        if not math.isfinite(length):
            raise RigPreflightError("Non-finite length for {}.".format(bone))
        lengths[bone] = length
    return lengths


def joint_span_z(positions: Mapping[str, Sequence[float]]) -> float:
    values = [float(position[2]) for position in positions.values()]
    if not values:
        raise RigPreflightError("No joint positions to span.")
    return max(values) - min(values)


def infer_up_axis_index(positions: Mapping[str, Sequence[float]]) -> int:
    """Infer the character's own up axis from its rest pose.

    Stage up-axis metadata describes the *containing* stage, not the space a
    character is measured in: attempt_01 measured F01 inside a Y-up lab stage
    while its rest pose and untransformed bounds were Z-up, so the "height"
    proxy actually captured the A-pose arm span.  The ankle-to-nose direction is
    a property of the rig itself and stays correct in both measurement paths.
    """
    required = ("nose", "left_ankle", "right_ankle")
    for joint in required:
        if joint not in positions:
            raise RigPreflightError(
                "Cannot infer an up axis without {}.".format(joint)
            )
    nose = [float(value) for value in positions["nose"]]
    ankle = [
        (float(positions["left_ankle"][axis]) + float(positions["right_ankle"][axis])) / 2.0
        for axis in range(3)
    ]
    delta = [abs(nose[axis] - ankle[axis]) for axis in range(3)]
    order = sorted(range(3), key=lambda axis: delta[axis], reverse=True)
    dominant, runner_up = order[0], order[1]
    if delta[dominant] < MIN_ANKLE_TO_NOSE_SPAN_M:
        raise RigPreflightError(
            "Rest pose spans only {:.4f} along its longest axis; no upright "
            "character to measure.".format(delta[dominant])
        )
    if delta[dominant] < UP_AXIS_DOMINANCE_RATIO * max(delta[runner_up], 1.0e-9):
        raise RigPreflightError(
            "Rest pose has no dominant up axis (deltas {}); measure the character "
            "in an upright rest pose before gating it.".format(
                [round(value, 6) for value in delta]
            )
        )
    return dominant


def _in_range(value: float, bounds: Sequence[float]) -> bool:
    return bool(bounds[0] <= float(value) <= bounds[1])


def evaluate_character(record: Mapping) -> dict:
    """Evaluate one measured candidate against the per-character gates."""
    for field in REQUIRED_MEASUREMENT_FIELDS:
        if field not in record:
            raise RigPreflightError(
                "Measurement record is missing {!r}.".format(field)
            )

    failures = []
    tokens = [str(token) for token in record["joint_tokens"]]
    if len(tokens) != int(record["joint_token_count"]):
        failures.append("joint_token_count disagrees with joint_tokens length")

    mapping = resolve_joint_indices(tokens)
    if mapping["missing"]:
        failures.append(
            "unmapped canonical joints: {}".format(",".join(mapping["missing"]))
        )
    for joint, matches in sorted(mapping["ambiguous"].items()):
        failures.append(
            "ambiguous canonical joint {} matched {}".format(joint, matches)
        )
    if mapping["complete"]:
        failures.extend(_ancestor_chain_ok(tokens, mapping["resolved"]))

    positions = {
        joint: tuple(float(value) for value in position)
        for joint, position in record["rest_joint_positions"].items()
    }
    for joint in CANONICAL_JOINTS:
        if joint not in positions:
            failures.append("no rest position for {}".format(joint))
    for joint, position in sorted(positions.items()):
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            failures.append("invalid rest position for {}".format(joint))

    lengths = None
    asymmetry = {}
    if not failures:
        lengths = bone_lengths(positions)
        for bone, length in sorted(lengths.items()):
            bounds = (
                PLAUSIBLE_ARM_BONE_RANGE_M
                if bone in ARM_BONES
                else PLAUSIBLE_LEG_BONE_RANGE_M
            )
            if not _in_range(length, bounds):
                failures.append(
                    "{} length {:.4f} m outside plausible {}".format(
                        bone, length, bounds
                    )
                )
        for left, right in MIRROR_BONES:
            mean = (lengths[left] + lengths[right]) / 2.0
            ratio = abs(lengths[left] - lengths[right]) / mean if mean > 0 else 1.0
            asymmetry[left.replace("left_", "")] = ratio
            if ratio > MAX_LEFT_RIGHT_ASYMMETRY_RATIO:
                failures.append(
                    "{}/{} asymmetry {:.4f} exceeds {:.4f}".format(
                        left, right, ratio, MAX_LEFT_RIGHT_ASYMMETRY_RATIO
                    )
                )

    if int(record["mesh_point_count"]) < MIN_MESH_POINT_COUNT:
        failures.append(
            "mesh_point_count {} below {}".format(
                record["mesh_point_count"], MIN_MESH_POINT_COUNT
            )
        )
    for field in ("mesh_signature_sha256", "mesh_shape_signature_sha256"):
        if not str(record[field] or "").strip():
            failures.append("{} is empty".format(field))

    rest_pose_up_axis = None
    if lengths is not None:
        try:
            rest_pose_up_axis = infer_up_axis_index(positions)
        except RigPreflightError as error:
            failures.append(str(error))
    recorded_axis = int(record["size_proxy_axis_index"])
    if recorded_axis not in (0, 1, 2):
        failures.append("size_proxy_axis_index {} is not an axis".format(recorded_axis))
    elif rest_pose_up_axis is not None and recorded_axis != rest_pose_up_axis:
        failures.append(
            "size proxy measured along axis {} but the rest pose stands up along "
            "axis {}".format(recorded_axis, rest_pose_up_axis)
        )

    size_proxy = float(record["size_proxy_operational_m"])
    if not math.isfinite(size_proxy) or not _in_range(
        size_proxy, PLAUSIBLE_SIZE_PROXY_RANGE_M
    ):
        failures.append(
            "size proxy {:.4f} m outside plausible {}".format(
                size_proxy, PLAUSIBLE_SIZE_PROXY_RANGE_M
            )
        )

    return {
        "character_key": str(record["character_key"]),
        "source_kind": str(record["source_kind"]),
        "source_identity": str(record["source_identity"]),
        "skeleton_path": str(record["skeleton_path"]),
        "joint_token_count": int(record["joint_token_count"]),
        "mapped_joint_count": len(mapping["resolved"]),
        "core_joint_count": len(CORE_JOINTS),
        "bone_lengths_operational_m": lengths,
        "left_right_asymmetry_ratio": asymmetry,
        "mapped_joint_span_z_operational_m": (
            joint_span_z(positions) if positions else None
        ),
        "rest_pose_up_axis_index": rest_pose_up_axis,
        "size_proxy_axis_index": recorded_axis,
        "size_proxy_operational_m": size_proxy,
        "declared_meters_per_unit": float(record["declared_meters_per_unit"]),
        "unit_convention": UNIT_CONVENTION,
        "failures": failures,
        "pass": not failures,
    }


def validate_discovery_record(payload: Mapping) -> list:
    """Check a discovery record is usable as the source of a shortlist.

    Deliberately not a hash-equality check against the current gate module: a
    discovery record is a folder listing whose validity does not depend on the
    measurement logic, so re-listing the asset server every time this module
    changes would be pure ceremony.  The hashes it carries stay in the record
    for audit and are copied into the measurement record.
    """
    problems = []
    if payload.get("record") != "character_asset_discovery_v1":
        problems.append("not a character_asset_discovery_v1 record")
    if payload.get("formal_capture_authorized") is not False:
        problems.append("discovery did not keep formal_capture_authorized false")
    if payload.get("gt_error_used_for_selection") is not False:
        problems.append("discovery did not keep gt_error_used_for_selection false")
    if payload.get("stage_modified") is not False:
        problems.append("discovery reports a modified stage")
    characters = payload.get("characters") or []
    resolved = [
        entry for entry in characters
        if entry.get("resolved") and str(entry.get("asset_url", "")).strip()
    ]
    if not resolved:
        problems.append("discovery record holds no resolved asset_url")
    if payload.get("character_count") != len(characters):
        problems.append("character_count disagrees with the recorded character list")
    return problems


def proportionality_spread(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Relative spread of per-bone length ratios between two characters.

    A value at or below SCALE_PROPORTIONALITY_TOLERANCE means the second rig is
    essentially the first one under a single uniform scale, which AGENTS.md
    6.6.2 forbids as a fake extra subject.
    """
    ratios = []
    for bone in BONE_NAMES:
        if bone not in left or bone not in right:
            raise RigPreflightError("Cannot compare incomplete bone vectors.")
        denominator = float(right[bone])
        if denominator <= 0.0:
            raise RigPreflightError("Non-positive bone length in comparison.")
        ratios.append(float(left[bone]) / denominator)
    mean = sum(ratios) / len(ratios)
    if mean <= 0.0:
        raise RigPreflightError("Non-positive mean ratio in comparison.")
    return (max(ratios) - min(ratios)) / mean


def evaluate_cohort(records: Sequence[Mapping]) -> dict:
    """Evaluate the four-character cohort: per-character gates plus distinctness."""
    per_character = [evaluate_character(record) for record in records]
    by_key = {entry["character_key"]: entry for entry in per_character}
    failures = []

    if len(by_key) != len(per_character):
        failures.append("duplicate character_key values in cohort")
    missing_keys = [key for key in REQUIRED_COHORT_KEYS if key not in by_key]
    if missing_keys:
        failures.append("cohort missing {}".format(",".join(missing_keys)))
    extra_keys = [key for key in sorted(by_key) if key not in REQUIRED_COHORT_KEYS]
    if extra_keys:
        failures.append("cohort has unexpected keys {}".format(",".join(extra_keys)))
    failed_characters = [
        entry["character_key"] for entry in per_character if not entry["pass"]
    ]
    if failed_characters:
        failures.append(
            "per-character gates failed for {}".format(",".join(sorted(failed_characters)))
        )

    for field in (
        "source_identity",
        "mesh_signature_sha256",
        "mesh_shape_signature_sha256",
    ):
        seen = {}
        for record in records:
            value = str(record.get(field, "")).strip().lower()
            seen.setdefault(value, []).append(str(record.get("character_key")))
        for value, keys in sorted(seen.items()):
            if len(keys) > 1:
                failures.append(
                    "{} shared by {} (not independent rigs/meshes)".format(
                        field, ",".join(sorted(keys))
                    )
                )

    pairwise = []
    complete = [
        entry for entry in per_character if entry["bone_lengths_operational_m"]
    ]
    for index, left in enumerate(complete):
        for right in complete[index + 1:]:
            spread = proportionality_spread(
                left["bone_lengths_operational_m"],
                right["bone_lengths_operational_m"],
            )
            size_gap = abs(
                left["size_proxy_operational_m"] - right["size_proxy_operational_m"]
            )
            pairwise.append(
                {
                    "pair": [left["character_key"], right["character_key"]],
                    "bone_ratio_spread": spread,
                    "size_proxy_gap_operational_m": size_gap,
                }
            )
            if spread <= SCALE_PROPORTIONALITY_TOLERANCE:
                failures.append(
                    "{} and {} differ by ~uniform scale only (spread {:.4f} <= {:.4f})".format(
                        left["character_key"],
                        right["character_key"],
                        spread,
                        SCALE_PROPORTIONALITY_TOLERANCE,
                    )
                )
            if size_gap < MIN_PAIRWISE_SIZE_SEPARATION_M:
                failures.append(
                    "{} and {} size proxies within {:.4f} m (< {:.4f})".format(
                        left["character_key"],
                        right["character_key"],
                        size_gap,
                        MIN_PAIRWISE_SIZE_SEPARATION_M,
                    )
                )

    strata = {}
    size_span = None
    if complete:
        ordered = sorted(complete, key=lambda entry: entry["size_proxy_operational_m"])
        size_span = (
            ordered[-1]["size_proxy_operational_m"]
            - ordered[0]["size_proxy_operational_m"]
        )
        if size_span < MIN_COHORT_SIZE_SPAN_M:
            failures.append(
                "cohort size span {:.4f} m below {:.4f}".format(
                    size_span, MIN_COHORT_SIZE_SPAN_M
                )
            )
        for position, entry in enumerate(ordered):
            if position == 0:
                label = "small"
            elif position == len(ordered) - 1:
                label = "large"
            else:
                label = "medium"
            strata[entry["character_key"]] = label
        if INCUMBENT_KEY in by_key:
            incumbent = by_key[INCUMBENT_KEY]["size_proxy_operational_m"]
            smaller = [
                entry["character_key"]
                for entry in complete
                if entry["size_proxy_operational_m"] < incumbent
            ]
            larger = [
                entry["character_key"]
                for entry in complete
                if entry["size_proxy_operational_m"] > incumbent
            ]
            if not smaller or not larger:
                failures.append(
                    "cohort does not bracket {} in body size (smaller={}, larger={})".format(
                        INCUMBENT_KEY, sorted(smaller), sorted(larger)
                    )
                )

    return {
        "preflight_tag": PREFLIGHT_TAG,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "unit_convention": UNIT_CONVENTION,
        "character_count": len(per_character),
        "characters": per_character,
        "pairwise": pairwise,
        "size_strata_descriptive_only": strata,
        "cohort_size_span_operational_m": size_span,
        "deferred_gates": list(DEFERRED_GATES),
        "failures": failures,
        "overall_pass": not failures,
        "gender_inference_permitted": False,
        "note": (
            "Descriptive simulated-character cohort only; four rigs cannot support "
            "gender, real-population, or multi-person claims."
        ),
    }
