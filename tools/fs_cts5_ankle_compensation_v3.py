"""Ankle compensation for the FS-CTS5 full-body motion (v3, pure Python).

The frozen v2 motion keys hip and knee only.  Its geometry gates are relative
and ankle-based, so they pass while the *foot* rotates toe-down with the shank:
the F01 ground probe measured a swing sole 53 mm **below** the stance plane and
never once above it.  See AGENTS.md 6.6.9.

v3 does not fork v2.  The schedule, the reach/march controls and the existing
geometry contract are re-exported unchanged; this module only computes the extra
ankle rotation that cancels the accumulated hip and knee rotation, so the foot
keeps its rest world orientation while the leg moves.

Convention follows USD: row vectors, ``child_world = child_local * parent_world``.
Rotations are 3x3 row-major; a rotation's inverse is its transpose.
"""

from __future__ import annotations

import math
from typing import Sequence

from fs_cts5_full_body_motion_v2 import (  # noqa: F401  (re-exported schedule)
    CONTROL_KEYS,
    END_FRAME,
    FPS,
    FRAME_COUNT,
    controls_for_frame,
    phase_for_frame,
    schedule_summary,
)


MOTION_TAG = "fs_cts5_full_body_reach_alternating_low_march_ankle_compensated_v3"
BASE_MOTION_TAG = "fs_cts5_full_body_reach_alternating_low_march_v2"
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False

ORTHONORMAL_TOLERANCE = 1.0e-6
# The residual tolerance is set by consequence, not by arithmetic taste: a
# residual foot rotation of r degrees lifts or sinks the toe of a ~0.25 m foot by
# about 0.25 * sin(r).  The ground-contact penetration gate is 5 mm, so 0.2 deg
# (~0.9 mm at the toe) stays an order of magnitude clear of it.  The first F01
# run measured 0.0456 deg, i.e. ~0.2 mm, which is why an earlier 1e-4 deg
# tolerance was a self-imposed obstacle rather than a real one.
RESIDUAL_ROTATION_TOLERANCE_DEG = 0.2
FOOT_LENGTH_FOR_RESIDUAL_M = 0.25
SIDES = ("left", "right")


class AnkleCompensationError(RuntimeError):
    """Raised when a rotation input is not a usable rotation matrix."""


def mat3_multiply(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list:
    return [
        [
            sum(float(left[row][index]) * float(right[index][column]) for index in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


def mat3_transpose(matrix: Sequence[Sequence[float]]) -> list:
    return [[float(matrix[column][row]) for column in range(3)] for row in range(3)]


def mat3_identity() -> list:
    return [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]


def is_rotation(matrix: Sequence[Sequence[float]], tolerance: float = ORTHONORMAL_TOLERANCE) -> bool:
    product = mat3_multiply(matrix, mat3_transpose(matrix))
    identity = mat3_identity()
    return all(
        abs(product[row][column] - identity[row][column]) <= tolerance
        for row in range(3)
        for column in range(3)
    )


def orthonormality_defect(matrix: Sequence[Sequence[float]]) -> float:
    """Largest element of |M M^T - I|, the error transpose-as-inverse inherits."""
    product = mat3_multiply(matrix, mat3_transpose(matrix))
    identity = mat3_identity()
    return max(
        abs(product[row][column] - identity[row][column])
        for row in range(3)
        for column in range(3)
    )


def residual_toe_displacement_m(residual_deg: float) -> float:
    """What a residual foot rotation costs at the toe, in metres."""
    return FOOT_LENGTH_FOR_RESIDUAL_M * abs(math.sin(math.radians(float(residual_deg))))


def require_rotation(matrix, name: str) -> list:
    rotation = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    if not is_rotation(rotation):
        raise AnkleCompensationError("{} is not an orthonormal rotation.".format(name))
    return rotation


def rotation_angle_deg(matrix: Sequence[Sequence[float]]) -> float:
    """Rotation magnitude of a 3x3 rotation, in degrees."""
    trace = sum(float(matrix[index][index]) for index in range(3))
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def axis_angle_matrix(axis: Sequence[float], degrees: float) -> list:
    """Right-handed rotation about `axis` for row vectors."""
    length = math.sqrt(sum(float(value) ** 2 for value in axis))
    if length <= 1.0e-12:
        raise AnkleCompensationError("Rotation axis has zero length.")
    x, y, z = (float(value) / length for value in axis)
    angle = math.radians(float(degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return [
        [cosine + x * x * complement, x * y * complement + z * sine, x * z * complement - y * sine],
        [y * x * complement - z * sine, cosine + y * y * complement, y * z * complement + x * sine],
        [z * x * complement + y * sine, z * y * complement - x * sine, cosine + z * z * complement],
    ]


def extra_rotation(animated_local, rest_local) -> list:
    """Recover the extra local rotation E from ``animated = E * rest``."""
    animated = require_rotation(animated_local, "animated local rotation")
    rest = require_rotation(rest_local, "rest local rotation")
    return mat3_multiply(animated, mat3_transpose(rest))


def foot_world_rotation(rest_chain, extras) -> list:
    """World rotation of the foot for a thigh->calf->foot chain.

    ``rest_chain`` holds the rest local rotations keyed ``thigh``/``calf``/``foot``;
    ``extras`` holds the extra local rotations applied at the same joints.  The
    pelvis-and-above part of the chain is constant, so it cancels out of every
    comparison this module makes and is deliberately not required here.
    """
    thigh = mat3_multiply(extras.get("thigh", mat3_identity()), rest_chain["thigh"])
    calf = mat3_multiply(extras.get("calf", mat3_identity()), rest_chain["calf"])
    foot = mat3_multiply(extras.get("foot", mat3_identity()), rest_chain["foot"])
    return mat3_multiply(foot, mat3_multiply(calf, thigh))


def compensating_ankle_rotation(rest_chain, hip_extra, knee_extra) -> list:
    """Extra ankle rotation that restores the foot's rest world orientation.

    Solving ``E_ankle * L_foot * E_knee * L_calf * E_hip * L_thigh
    == L_foot * L_calf * L_thigh`` gives
    ``E_ankle = (L_foot L_calf L_thigh) (L_foot E_knee L_calf E_hip L_thigh)^-1``.
    """
    for key in ("thigh", "calf", "foot"):
        if key not in rest_chain:
            raise AnkleCompensationError("rest_chain is missing {!r}.".format(key))
    rest = {key: require_rotation(rest_chain[key], "rest " + key) for key in rest_chain}
    hip = require_rotation(hip_extra, "hip extra rotation")
    knee = require_rotation(knee_extra, "knee extra rotation")

    target = mat3_multiply(rest["foot"], mat3_multiply(rest["calf"], rest["thigh"]))
    moved = foot_world_rotation(rest, {"thigh": hip, "calf": knee})
    return mat3_multiply(target, mat3_transpose(moved))


def residual_foot_rotation_deg(rest_chain, hip_extra, knee_extra, ankle_extra) -> float:
    """How far the foot still deviates from its rest world orientation."""
    rest = {key: require_rotation(rest_chain[key], "rest " + key) for key in rest_chain}
    target = mat3_multiply(rest["foot"], mat3_multiply(rest["calf"], rest["thigh"]))
    moved = foot_world_rotation(
        rest,
        {
            "thigh": require_rotation(hip_extra, "hip extra rotation"),
            "calf": require_rotation(knee_extra, "knee extra rotation"),
            "foot": require_rotation(ankle_extra, "ankle extra rotation"),
        },
    )
    return rotation_angle_deg(mat3_multiply(moved, mat3_transpose(target)))


def extend_animation_joints(joints: Sequence[str], added: Sequence[str]) -> dict:
    """Plan an extension of a SkelAnimation joint list.

    The v2 animation drives eight joints and does not list the feet at all, so
    v3 has to widen ``joints`` and every per-joint array with it.  Misaligning
    those arrays would silently drive the wrong joint, so the layout is computed
    once here and every array is built against it.
    """
    existing = [str(token) for token in joints]
    if len(set(existing)) != len(existing):
        raise AnkleCompensationError("The source animation lists a joint twice.")
    new_tokens = [str(token) for token in added]
    if len(set(new_tokens)) != len(new_tokens):
        raise AnkleCompensationError("The added joint list repeats a token.")
    clash = [token for token in new_tokens if token in existing]
    if clash:
        raise AnkleCompensationError(
            "The source animation already drives {}.".format(clash)
        )
    return {
        "joints": existing + new_tokens,
        "kept_count": len(existing),
        "added_count": len(new_tokens),
        "added_indices": {
            token: len(existing) + offset for offset, token in enumerate(new_tokens)
        },
    }


def extend_per_joint_array(values: Sequence, additions: Sequence, layout) -> list:
    """Append per-joint values in the layout's order, preserving the originals."""
    kept = list(values)
    if len(kept) != layout["kept_count"]:
        raise AnkleCompensationError(
            "Expected {} source values, received {}.".format(
                layout["kept_count"], len(kept)
            )
        )
    extra = list(additions)
    if len(extra) != layout["added_count"]:
        raise AnkleCompensationError(
            "Expected {} added values, received {}.".format(
                layout["added_count"], len(extra)
            )
        )
    return kept + extra


def verify_extension_preserved(original: Sequence, extended: Sequence, layout) -> bool:
    """The original per-joint values must survive an extension untouched."""
    if len(extended) != layout["kept_count"] + layout["added_count"]:
        return False
    return list(extended)[: layout["kept_count"]] == list(original)


def compensate_series(rest_chains, hip_extras_by_frame, knee_extras_by_frame) -> dict:
    """Compute ankle compensation for every frame of both legs.

    ``rest_chains`` is keyed by side; the extras are ``{side: {frame: matrix}}``.
    Returns the ankle extras plus the worst residual, which the caller must gate
    before writing any layer.
    """
    expected = set(range(FRAME_COUNT))
    ankle_extras = {}
    worst_residual_deg = 0.0
    worst_frame = None
    worst_side = None
    worst_input_defect = 0.0
    for side in SIDES:
        if side not in rest_chains:
            raise AnkleCompensationError("No rest chain for the {} leg.".format(side))
        hips = hip_extras_by_frame[side]
        knees = knee_extras_by_frame[side]
        if set(hips) != expected or set(knees) != expected:
            raise AnkleCompensationError(
                "The {} leg must supply hip and knee rotations for frames 0..{}.".format(
                    side, END_FRAME
                )
            )
        ankle_extras[side] = {}
        for frame in range(FRAME_COUNT):
            worst_input_defect = max(
                worst_input_defect,
                orthonormality_defect(hips[frame]),
                orthonormality_defect(knees[frame]),
            )
            ankle = compensating_ankle_rotation(
                rest_chains[side], hips[frame], knees[frame]
            )
            ankle_extras[side][frame] = ankle
            residual = residual_foot_rotation_deg(
                rest_chains[side], hips[frame], knees[frame], ankle
            )
            if residual > worst_residual_deg:
                worst_residual_deg = residual
                worst_frame = frame
                worst_side = side

    failures = []
    if worst_residual_deg > RESIDUAL_ROTATION_TOLERANCE_DEG:
        failures.append(
            "foot orientation still deviates {:.6f} deg at {} frame {} ({:.4f} mm "
            "at the toe, limit {:.4f} deg)".format(
                worst_residual_deg,
                worst_side,
                worst_frame,
                residual_toe_displacement_m(worst_residual_deg) * 1000.0,
                RESIDUAL_ROTATION_TOLERANCE_DEG,
            )
        )

    return {
        "motion_tag": MOTION_TAG,
        "base_motion_tag": BASE_MOTION_TAG,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "frame_count": FRAME_COUNT,
        "ankle_extras": ankle_extras,
        "worst_residual_rotation_deg": worst_residual_deg,
        "worst_residual_toe_displacement_m": residual_toe_displacement_m(worst_residual_deg),
        "worst_residual_frame": worst_frame,
        "worst_residual_side": worst_side,
        "worst_input_orthonormality_defect": worst_input_defect,
        "residual_tolerance_deg": RESIDUAL_ROTATION_TOLERANCE_DEG,
        "failures": failures,
        "pass": not failures,
        "note": (
            "Keeps the foot at its rest world orientation. It does not by itself "
            "lift the foot: swing clearance above the stance plane stays a "
            "ground-contact gate."
        ),
    }
