"""Forward kinematics for the UR10e occluder (pure Python).

The Isaac UR10e keeps its seven links as **flat siblings** under ``/ur10e``,
connected by physics joints rather than nested transforms, so posing it by
rotating one link prim moves nothing below that link.  Posing therefore needs an
explicit chain, which is what this module provides: it composes each link's
transform from the joint frames the asset itself declares.

Chain and frames were read from the asset (see the ur10e_rig record), not
assumed from a datasheet.  Convention follows USD: row vectors, 4x4 row-major,
``child = local * parent``, quaternions ``(w, x, y, z)``.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


KINEMATICS_TAG = "fs_cts5_ur10e_kinematics_v1"

# joint name, parent link, child link, localPos0, localRot0 (w, x, y, z)
# Every revolute joint turns about its own local Z, and every localRot1 in the
# asset is identity, so the child frame is parent * T0 * Rz(theta).
JOINT_CHAIN = (
    (
        "shoulder_pan_joint", "base_link", "shoulder_link",
        (0.0, 0.0, 0.1807), (-4.371139e-8, 0.0, 0.0, 1.0),
    ),
    (
        "shoulder_lift_joint", "shoulder_link", "upper_arm_link",
        (0.0, 0.0, 0.0), (0.70710677, 0.70710677, 0.0, 0.0),
    ),
    (
        "elbow_joint", "upper_arm_link", "forearm_link",
        (-0.6127, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0),
    ),
    (
        "wrist_1_joint", "forearm_link", "wrist_1_link",
        (-0.57155, 0.0, 0.17415), (1.0, 0.0, 0.0, 0.0),
    ),
    (
        "wrist_2_joint", "wrist_1_link", "wrist_2_link",
        (0.0, -0.11985, -2.4581646e-11), (0.70710677, 0.70710677, 0.0, 0.0),
    ),
    (
        "wrist_3_joint", "wrist_2_link", "wrist_3_link",
        (0.0, 0.11655, -2.3904805e-11), (0.70710677, -0.70710677, -6.181724e-8, 0.0),
    ),
)
BASE_LINK = "base_link"
LINK_ORDER = (BASE_LINK,) + tuple(entry[2] for entry in JOINT_CHAIN)
JOINT_NAMES = tuple(entry[0] for entry in JOINT_CHAIN)
# The asset's own limits, in degrees.
JOINT_LIMITS_DEG = {
    "shoulder_pan_joint": (-360.0, 360.0),
    "shoulder_lift_joint": (-360.0, 360.0),
    "elbow_joint": (-180.0, 180.0),
    "wrist_1_joint": (-360.0, 360.0),
    "wrist_2_joint": (-360.0, 360.0),
    "wrist_3_joint": (-360.0, 360.0),
}


class KinematicsError(RuntimeError):
    """Raised when a pose request cannot be satisfied by this chain."""


def identity4() -> list:
    return [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]


def matrix_multiply(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list:
    return [
        [
            sum(float(left[row][index]) * float(right[index][column]) for index in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def quaternion_matrix(quaternion: Sequence[float]) -> list:
    """USD ``(w, x, y, z)`` to a row-vector 4x4."""
    w, x, y, z = [float(value) for value in quaternion]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise KinematicsError("Zero-length quaternion.")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    matrix = identity4()
    matrix[0][0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[0][1] = 2.0 * (x * y + z * w)
    matrix[0][2] = 2.0 * (x * z - y * w)
    matrix[1][0] = 2.0 * (x * y - z * w)
    matrix[1][1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[1][2] = 2.0 * (y * z + x * w)
    matrix[2][0] = 2.0 * (x * z + y * w)
    matrix[2][1] = 2.0 * (y * z - x * w)
    matrix[2][2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def translation_matrix(offset: Sequence[float]) -> list:
    matrix = identity4()
    for axis in range(3):
        matrix[3][axis] = float(offset[axis])
    return matrix


def rotation_z(degrees: float) -> list:
    angle = math.radians(float(degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix = identity4()
    matrix[0][0] = cosine
    matrix[0][1] = sine
    matrix[1][0] = -sine
    matrix[1][1] = cosine
    return matrix


def link_transforms(angles_deg: Mapping[str, float]) -> dict:
    """Each link's transform relative to the robot root, for a joint pose."""
    missing = [name for name in JOINT_NAMES if name not in angles_deg]
    if missing:
        raise KinematicsError("No angle given for {}.".format(sorted(missing)))
    for name, angle in angles_deg.items():
        if name not in JOINT_LIMITS_DEG:
            raise KinematicsError("Unknown joint {!r}.".format(name))
        lower, upper = JOINT_LIMITS_DEG[name]
        if not lower <= float(angle) <= upper:
            raise KinematicsError(
                "{} angle {} is outside the asset's limits {}..{}.".format(
                    name, angle, lower, upper
                )
            )

    transforms = {BASE_LINK: identity4()}
    for name, parent, child, offset, quaternion in JOINT_CHAIN:
        joint_frame = matrix_multiply(
            quaternion_matrix(quaternion), translation_matrix(offset)
        )
        transforms[child] = matrix_multiply(
            matrix_multiply(rotation_z(angles_deg[name]), joint_frame),
            transforms[parent],
        )
    return transforms


def link_origins(angles_deg: Mapping[str, float]) -> dict:
    return {
        link: (matrix[3][0], matrix[3][1], matrix[3][2])
        for link, matrix in link_transforms(angles_deg).items()
    }


def zero_pose() -> dict:
    return {name: 0.0 for name in JOINT_NAMES}


def reach_height(angles_deg: Mapping[str, float]) -> float:
    """Height of the wrist above the robot's base, in metres."""
    return link_origins(angles_deg)["wrist_3_link"][2]


def transform_point(point: Sequence[float], matrix: Sequence[Sequence[float]]) -> tuple:
    return tuple(
        float(point[0]) * matrix[0][axis]
        + float(point[1]) * matrix[1][axis]
        + float(point[2]) * matrix[2][axis]
        + matrix[3][axis]
        for axis in range(3)
    )
