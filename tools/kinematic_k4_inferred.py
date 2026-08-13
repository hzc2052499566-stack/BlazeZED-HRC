"""K4 inferred right-arm completion with a guarded shoulder root.

K4 remains an explicitly inferred output.  It never changes measured joint
validity or the measured tracking CSV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kinematic_k3_inferred as k3


ALGORITHM_VERSION = "right_arm_ray_bone_k4_root_guard_v1_20260729"
PROVENANCE = "inferred_ray_bone_k4_root_guard"
DEFAULT_SHOULDER_RATE_LIMIT_M = 0.060


def optional_depth(candidate: dict | None) -> float | None:
    if candidate is None:
        return None
    value = candidate.get("depth_m")
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def guarded_root_depth(
    raw_depth_m: float | None,
    previous_guarded_depth_m: float | None,
    pelvis_depth_m: float | None,
    previous_pelvis_depth_m: float | None,
    rate_limit_m: float = DEFAULT_SHOULDER_RATE_LIMIT_M,
) -> dict:
    if float(rate_limit_m) <= 0.0:
        raise ValueError("K4 shoulder rate limit must be positive.")
    if previous_guarded_depth_m is None:
        return {
            "valid": raw_depth_m is not None,
            "depth_m": raw_depth_m,
            "prediction_m": raw_depth_m,
            "clipped": False,
            "raw_prediction_residual_m": 0.0,
            "reason": "" if raw_depth_m is not None else "no_initial_root",
        }
    pelvis_delta_m = (
        float(pelvis_depth_m) - float(previous_pelvis_depth_m)
        if pelvis_depth_m is not None
        and previous_pelvis_depth_m is not None
        else 0.0
    )
    prediction_m = float(previous_guarded_depth_m) + pelvis_delta_m
    if raw_depth_m is None:
        return {
            "valid": True,
            "depth_m": prediction_m,
            "prediction_m": prediction_m,
            "clipped": True,
            "raw_prediction_residual_m": None,
            "reason": "missing_root_predicted_from_pelvis",
        }
    residual_m = float(raw_depth_m) - prediction_m
    clipped_residual_m = float(
        np.clip(residual_m, -float(rate_limit_m), float(rate_limit_m))
    )
    clipped = not math.isclose(
        clipped_residual_m,
        residual_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    return {
        "valid": True,
        "depth_m": prediction_m + clipped_residual_m,
        "prediction_m": prediction_m,
        "clipped": clipped,
        "raw_prediction_residual_m": residual_m,
        "reason": "root_rate_limited" if clipped else "",
    }


@dataclass
class ShoulderRootGuard:
    rate_limit_m: float = DEFAULT_SHOULDER_RATE_LIMIT_M
    previous_guarded_depth_m: float | None = None
    previous_pelvis_depth_m: float | None = None

    def update(
        self,
        raw_depth_m: float | None,
        pelvis_depth_m: float | None,
    ) -> dict:
        result = guarded_root_depth(
            raw_depth_m,
            self.previous_guarded_depth_m,
            pelvis_depth_m,
            self.previous_pelvis_depth_m,
            self.rate_limit_m,
        )
        if result["valid"]:
            self.previous_guarded_depth_m = float(result["depth_m"])
        if pelvis_depth_m is not None:
            self.previous_pelvis_depth_m = float(pelvis_depth_m)
        return result


def infer_right_arm(
    candidates: list[dict],
    profile: dict,
    config: dict,
    intrinsics: tuple[float, float, float, float],
    root_guard: ShoulderRootGuard,
    elbow_branch: str = "far",
    wrist_branch: str = "far",
) -> tuple[list[dict], dict]:
    by_joint = {
        candidate["mapping"]["canonical_joint"]: candidate
        for candidate in candidates
    }
    shoulder = by_joint.get("right_shoulder")
    pelvis = by_joint.get("pelvis")
    elbow = by_joint.get("right_elbow")
    wrist = by_joint.get("right_wrist")

    raw_shoulder_depth_m = optional_depth(shoulder)
    pelvis_depth_m = optional_depth(pelvis)
    root = root_guard.update(raw_shoulder_depth_m, pelvis_depth_m)
    shoulder_pixel_xy = None
    shoulder_point = None
    if (
        shoulder is not None
        and shoulder.get("in_image")
        and float(shoulder.get("visibility", 0.0))
        >= k3.MINIMUM_VISIBILITY
        and root["valid"]
    ):
        shoulder_pixel_xy = (
            int(shoulder["pixel_x"]),
            int(shoulder["pixel_y"]),
        )
        shoulder_point = k3.reconstruct_from_depth(
            float(root["depth_m"]),
            shoulder_pixel_xy[0],
            shoulder_pixel_xy[1],
            *intrinsics,
        )

    upper_arm = k3.profile_bone(profile, "right_upper_arm")
    forearm = k3.profile_bone(profile, "right_forearm")
    upper_reference_m = float(upper_arm["reference_length_m"])
    forearm_reference_m = float(forearm["reference_length_m"])
    elbow_result = k3.infer_child(
        "right_elbow",
        elbow,
        float(root["depth_m"]) if shoulder_point is not None else None,
        shoulder_pixel_xy,
        shoulder_point,
        "inferred_guarded_right_shoulder",
        upper_reference_m,
        k3.profile_tolerance_m(profile, config, "right_upper_arm"),
        elbow_branch,
        intrinsics,
    )
    elbow_pixel_xy = (
        (int(elbow["pixel_x"]), int(elbow["pixel_y"]))
        if elbow is not None
        else None
    )
    wrist_result = k3.infer_child(
        "right_wrist",
        wrist,
        elbow_result["depth_m"],
        elbow_pixel_xy,
        elbow_result["point"],
        "inferred_right_elbow_k4",
        forearm_reference_m,
        k3.profile_tolerance_m(profile, config, "right_forearm"),
        wrist_branch,
        intrinsics,
    )
    for result in (elbow_result, wrist_result):
        result["provenance"] = PROVENANCE
    root_metadata = {
        "raw_shoulder_depth_m": raw_shoulder_depth_m,
        "pelvis_depth_m": pelvis_depth_m,
        "guarded_shoulder_depth_m": (
            float(root["depth_m"]) if root["valid"] else None
        ),
        "shoulder_prediction_m": root["prediction_m"],
        "shoulder_raw_prediction_residual_m": (
            root["raw_prediction_residual_m"]
        ),
        "shoulder_root_clipped": bool(root["clipped"]),
        "shoulder_root_reason": root["reason"],
        "shoulder_rate_limit_m": float(root_guard.rate_limit_m),
    }
    return [elbow_result, wrist_result], root_metadata
