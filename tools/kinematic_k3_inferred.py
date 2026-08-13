"""Explicitly inferred right-arm ray/bone completion.

This module never changes measured joint validity. Its outputs must remain in
an inferred namespace with provenance attached.
"""

from __future__ import annotations

import math

import numpy as np


ALGORITHM_VERSION = "right_arm_ray_bone_k3_inferred_v1_20260729"
MINIMUM_VISIBILITY = 0.35


def reconstruct_from_depth(
    depth_m: float,
    pixel_x: int,
    pixel_y: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    return np.asarray(
        [
            depth_m,
            -(pixel_x - cx) * depth_m / fx,
            -(pixel_y - cy) * depth_m / fy,
        ],
        dtype=np.float64,
    )


def ray_bone_roots_m(
    child_pixel_xy: tuple[int, int],
    parent_pixel_xy: tuple[int, int],
    parent_depth_m: float | None,
    reference_length_m: float,
    intrinsics: tuple[float, float, float, float],
) -> list[float]:
    if (
        parent_depth_m is None
        or not math.isfinite(float(parent_depth_m))
    ):
        return []
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    child_ray = reconstruct_from_depth(
        1.0,
        int(child_pixel_xy[0]),
        int(child_pixel_xy[1]),
        fx,
        fy,
        cx,
        cy,
    )
    parent_ray = reconstruct_from_depth(
        1.0,
        int(parent_pixel_xy[0]),
        int(parent_pixel_xy[1]),
        fx,
        fy,
        cx,
        cy,
    )
    parent_depth_m = float(parent_depth_m)
    a = float(np.dot(child_ray, child_ray))
    b = -2.0 * parent_depth_m * float(
        np.dot(child_ray, parent_ray)
    )
    c = (
        parent_depth_m**2 * float(np.dot(parent_ray, parent_ray))
        - float(reference_length_m) ** 2
    )
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    root_term = math.sqrt(max(0.0, discriminant))
    return sorted(
        root
        for root in (
            (-b - root_term) / (2.0 * a),
            (-b + root_term) / (2.0 * a),
        )
        if math.isfinite(root) and root > 0.05
    )


def minimum_ray_distance_to_parent_m(
    child_pixel_xy: tuple[int, int],
    parent_point: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> float:
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    ray = reconstruct_from_depth(
        1.0,
        int(child_pixel_xy[0]),
        int(child_pixel_xy[1]),
        fx,
        fy,
        cx,
        cy,
    )
    parent = np.asarray(parent_point, dtype=np.float64)
    depth_m = max(
        0.05,
        float(np.dot(ray, parent) / np.dot(ray, ray)),
    )
    return float(np.linalg.norm(depth_m * ray - parent))


def profile_bone(profile: dict, name: str) -> dict:
    for bone in profile["bones"]:
        if bone["name"] == name:
            return bone
    raise KeyError(name)


def profile_tolerance_m(
    profile: dict,
    config: dict,
    bone_name: str,
) -> float:
    bone = profile_bone(profile, bone_name)
    gate = config["variants"]["k1_length_gate"]
    return max(
        float(gate["minimum_absolute_tolerance_m"]),
        float(gate["minimum_relative_tolerance"])
        * float(bone["reference_length_m"]),
        float(gate["profile_scale_multiplier"])
        * float(bone["robust_scale_m"]),
    )


def invalid_result(
    joint: str,
    branch: str,
    bone_reference_m: float,
    reason: str,
    parent_provenance: str,
) -> dict:
    return {
        "joint": joint,
        "valid": False,
        "provenance": "inferred_ray_bone_k3",
        "parent_provenance": parent_provenance,
        "root_branch": branch,
        "point": None,
        "depth_m": None,
        "bone_reference_m": float(bone_reference_m),
        "bone_target_m": None,
        "constraint_relaxation_m": None,
        "bone_residual_m": None,
        "invalid_reason": reason,
    }


def infer_child(
    joint: str,
    child: dict | None,
    parent_depth_m: float | None,
    parent_pixel_xy: tuple[int, int] | None,
    parent_point: np.ndarray | None,
    parent_provenance: str,
    reference_length_m: float,
    tolerance_m: float,
    branch: str,
    intrinsics: tuple[float, float, float, float],
) -> dict:
    if branch not in {"near", "far"}:
        raise ValueError("K3 root branch must be near or far.")
    if child is None:
        return invalid_result(
            joint,
            branch,
            reference_length_m,
            "missing_2d_landmark",
            parent_provenance,
        )
    if (
        not child.get("in_image")
        or float(child.get("visibility", 0.0))
        < MINIMUM_VISIBILITY
    ):
        return invalid_result(
            joint,
            branch,
            reference_length_m,
            "unreliable_2d_landmark",
            parent_provenance,
        )
    if (
        parent_depth_m is None
        or parent_pixel_xy is None
        or parent_point is None
    ):
        return invalid_result(
            joint,
            branch,
            reference_length_m,
            "parent_unavailable",
            parent_provenance,
        )

    child_pixel_xy = (
        int(child["pixel_x"]),
        int(child["pixel_y"]),
    )
    roots = ray_bone_roots_m(
        child_pixel_xy,
        parent_pixel_xy,
        parent_depth_m,
        reference_length_m,
        intrinsics,
    )
    target_length_m = float(reference_length_m)
    relaxation_m = 0.0
    if not roots:
        minimum_length_m = minimum_ray_distance_to_parent_m(
            child_pixel_xy,
            parent_point,
            intrinsics,
        )
        required_relaxation_m = max(
            0.0,
            minimum_length_m - float(reference_length_m),
        )
        if required_relaxation_m > float(tolerance_m):
            return invalid_result(
                joint,
                branch,
                reference_length_m,
                "ray_misses_profile_bone_tolerance",
                parent_provenance,
            )
        relaxation_m = required_relaxation_m + 1e-6
        target_length_m = float(reference_length_m) + relaxation_m
        roots = ray_bone_roots_m(
            child_pixel_xy,
            parent_pixel_xy,
            parent_depth_m,
            target_length_m,
            intrinsics,
        )
    if not roots:
        return invalid_result(
            joint,
            branch,
            reference_length_m,
            "no_positive_ray_bone_root",
            parent_provenance,
        )

    depth_m = float(roots[0] if branch == "near" else roots[-1])
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    point = reconstruct_from_depth(
        depth_m,
        child_pixel_xy[0],
        child_pixel_xy[1],
        fx,
        fy,
        cx,
        cy,
    )
    return {
        "joint": joint,
        "valid": True,
        "provenance": "inferred_ray_bone_k3",
        "parent_provenance": parent_provenance,
        "root_branch": branch,
        "point": point,
        "depth_m": depth_m,
        "bone_reference_m": float(reference_length_m),
        "bone_target_m": target_length_m,
        "constraint_relaxation_m": relaxation_m,
        "bone_residual_m": abs(
            float(np.linalg.norm(point - parent_point))
            - float(reference_length_m)
        ),
        "invalid_reason": "",
    }


def infer_right_arm(
    candidates: list[dict],
    profile: dict,
    config: dict,
    intrinsics: tuple[float, float, float, float],
    elbow_branch: str = "far",
    wrist_branch: str = "far",
) -> list[dict]:
    by_joint = {
        candidate["mapping"]["canonical_joint"]: candidate
        for candidate in candidates
    }
    shoulder = by_joint.get("right_shoulder")
    elbow = by_joint.get("right_elbow")
    wrist = by_joint.get("right_wrist")
    upper_arm = profile_bone(profile, "right_upper_arm")
    forearm = profile_bone(profile, "right_forearm")
    upper_reference_m = float(upper_arm["reference_length_m"])
    forearm_reference_m = float(forearm["reference_length_m"])

    shoulder_depth_m = None
    shoulder_pixel_xy = None
    shoulder_point = None
    if (
        shoulder is not None
        and shoulder.get("in_image")
        and float(shoulder.get("visibility", 0.0))
        >= MINIMUM_VISIBILITY
    ):
        raw_depth = shoulder.get("depth_m")
        if raw_depth is not None and math.isfinite(float(raw_depth)):
            shoulder_depth_m = float(raw_depth)
            shoulder_pixel_xy = (
                int(shoulder["pixel_x"]),
                int(shoulder["pixel_y"]),
            )
            fx, fy, cx, cy = (
                float(value) for value in intrinsics
            )
            shoulder_point = reconstruct_from_depth(
                shoulder_depth_m,
                shoulder_pixel_xy[0],
                shoulder_pixel_xy[1],
                fx,
                fy,
                cx,
                cy,
            )

    elbow_result = infer_child(
        "right_elbow",
        elbow,
        shoulder_depth_m,
        shoulder_pixel_xy,
        shoulder_point,
        "measured_right_shoulder",
        upper_reference_m,
        profile_tolerance_m(
            profile,
            config,
            "right_upper_arm",
        ),
        elbow_branch,
        intrinsics,
    )
    elbow_pixel_xy = (
        (
            int(elbow["pixel_x"]),
            int(elbow["pixel_y"]),
        )
        if elbow is not None
        else None
    )
    wrist_result = infer_child(
        "right_wrist",
        wrist,
        elbow_result["depth_m"],
        elbow_pixel_xy,
        elbow_result["point"],
        "inferred_right_elbow",
        forearm_reference_m,
        profile_tolerance_m(
            profile,
            config,
            "right_forearm",
        ),
        wrist_branch,
        intrinsics,
    )
    return [elbow_result, wrist_result]
