"""Pure depth-sampling strategies for the paired offline ablation.

The functions in this module intentionally do not depend on MediaPipe or the
ZED SDK.  They operate on an already aligned metric depth image and cached
BlazePose landmark pixels so every ablation condition can use identical 2D
inputs.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


MIN_DEPTH_M = 0.05
MAX_DEPTH_M = 50.0
MIN_VISIBILITY = 0.35

SUPPORTED_METHODS = (
    "point_1x1",
    "median_7x7",
    "adaptive_median_7_11_15",
    "limb_aware_cluster",
)

WRIST_TO_ELBOW = {
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
}
WRIST_TO_SHOULDER = {
    "left_wrist": "left_shoulder",
    "right_wrist": "right_shoulder",
}


def empty_result(method: str, invalid_reason: str = "no_valid_depth") -> dict:
    return {
        "depth_m": None,
        "valid": False,
        "sampling_method": method,
        "window_radius_px": None,
        "window_size_px": None,
        "valid_pixel_count": 0,
        "fallback_distance_px": None,
        "cluster_count": 0,
        "selected_cluster_size": 0,
        "directional_constraint_used": False,
        "kinematic_reference_m": None,
        "invalid_reason": invalid_reason,
    }


def valid_depth_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (
        np.isfinite(values)
        & (values > MIN_DEPTH_M)
        & (values < MAX_DEPTH_M)
    )


def point_depth(depth_m: np.ndarray, pixel_x: int, pixel_y: int) -> dict:
    result = empty_result("point_1x1")
    height, width = depth_m.shape[:2]
    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        result["invalid_reason"] = "landmark_out_of_image"
        return result
    value = float(depth_m[pixel_y, pixel_x])
    if not bool(valid_depth_mask(np.asarray([value]))[0]):
        result["invalid_reason"] = "invalid_centre_pixel"
        return result
    result.update(
        {
            "depth_m": value,
            "valid": True,
            "window_radius_px": 0,
            "window_size_px": 1,
            "valid_pixel_count": 1,
            "invalid_reason": "",
        }
    )
    return result


def median_window_depth(
    depth_m: np.ndarray,
    pixel_x: int,
    pixel_y: int,
    radius_px: int,
    *,
    method: str = "median_window",
    min_valid_pixels: int = 1,
) -> dict:
    result = empty_result(method)
    height, width = depth_m.shape[:2]
    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        result["invalid_reason"] = "landmark_out_of_image"
        return result
    radius_px = int(radius_px)
    x0 = max(0, pixel_x - radius_px)
    x1 = min(width, pixel_x + radius_px + 1)
    y0 = max(0, pixel_y - radius_px)
    y1 = min(height, pixel_y + radius_px + 1)
    values = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    valid = valid_depth_mask(values)
    valid_count = int(np.count_nonzero(valid))
    result.update(
        {
            "window_radius_px": radius_px,
            "window_size_px": 2 * radius_px + 1,
            "valid_pixel_count": valid_count,
        }
    )
    if valid_count < int(min_valid_pixels):
        result["invalid_reason"] = "insufficient_valid_pixels"
        return result
    result.update(
        {
            "depth_m": float(np.median(values[valid])),
            "valid": True,
            "invalid_reason": "",
        }
    )
    return result


def adaptive_median_depth(
    depth_m: np.ndarray,
    pixel_x: int,
    pixel_y: int,
    radii_px: Iterable[int] = (3, 5, 7),
    *,
    min_valid_pixels: int = 1,
) -> dict:
    last_result = empty_result("adaptive_median_7_11_15")
    for radius_px in radii_px:
        result = median_window_depth(
            depth_m,
            pixel_x,
            pixel_y,
            int(radius_px),
            method="adaptive_median_7_11_15",
            min_valid_pixels=min_valid_pixels,
        )
        last_result = result
        if result["valid"]:
            return result
    return last_result


def kinematic_wrist_reference(
    wrist: dict,
    elbow: dict | None,
    shoulder: dict | None,
    elbow_depth_m: float | None,
    shoulder_depth_m: float | None,
    max_elbow_depth_offset_m: float = 0.35,
) -> float | None:
    if elbow is None or shoulder is None:
        return None
    supports = (wrist, elbow, shoulder)
    if any(
        not bool(support.get("in_image"))
        or float(support.get("visibility", 0.0)) < MIN_VISIBILITY
        for support in supports
    ):
        return None
    if elbow_depth_m is None or shoulder_depth_m is None:
        return None
    if not (
        math.isfinite(float(elbow_depth_m))
        and math.isfinite(float(shoulder_depth_m))
    ):
        return None
    upper_arm_px = math.hypot(
        float(elbow["pixel_x"]) - float(shoulder["pixel_x"]),
        float(elbow["pixel_y"]) - float(shoulder["pixel_y"]),
    )
    forearm_px = math.hypot(
        float(wrist["pixel_x"]) - float(elbow["pixel_x"]),
        float(wrist["pixel_y"]) - float(elbow["pixel_y"]),
    )
    if upper_arm_px < 8.0 or forearm_px < 8.0:
        return None
    segment_ratio = max(0.5, min(1.5, forearm_px / upper_arm_px))
    extrapolated_m = float(elbow_depth_m) + segment_ratio * (
        float(elbow_depth_m) - float(shoulder_depth_m)
    )
    return max(
        float(elbow_depth_m) - max_elbow_depth_offset_m,
        min(
            float(elbow_depth_m) + max_elbow_depth_offset_m,
            extrapolated_m,
        ),
    )


def clustered_depth(
    depth_m: np.ndarray,
    pixel_x: int,
    pixel_y: int,
    *,
    search_radius_px: int,
    cluster_gap_m: float,
    spatial_band_px: float,
    minimum_cluster_size: int = 1,
    preferred_direction_xy: tuple[float, float] | None = None,
    directional_half_width_px: float = 4.0,
    directional_backward_tolerance_px: float = 1.0,
    kinematic_reference_m: float | None = None,
    method: str = "clustered_spatial_fallback",
) -> dict:
    result = empty_result(method)
    height, width = depth_m.shape[:2]
    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        result["invalid_reason"] = "landmark_out_of_image"
        return result
    if search_radius_px <= 0:
        result["invalid_reason"] = "fallback_disabled"
        return result

    x0 = max(0, pixel_x - int(search_radius_px))
    x1 = min(width, pixel_x + int(search_radius_px) + 1)
    y0 = max(0, pixel_y - int(search_radius_px))
    y1 = min(height, pixel_y + int(search_radius_px) + 1)
    window = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64)
    valid_y, valid_x = np.nonzero(valid_depth_mask(window))
    if valid_x.size == 0:
        result["invalid_reason"] = "no_valid_depth_in_search"
        return result

    full_x = valid_x + x0
    full_y = valid_y + y0
    distances_px = np.sqrt(
        (full_x - pixel_x) ** 2 + (full_y - pixel_y) ** 2
    )
    radial_mask = distances_px <= float(search_radius_px)
    valid_x = valid_x[radial_mask]
    valid_y = valid_y[radial_mask]
    full_x = full_x[radial_mask]
    full_y = full_y[radial_mask]
    distances_px = distances_px[radial_mask]
    if distances_px.size == 0:
        result["invalid_reason"] = "no_valid_depth_in_search"
        return result

    if preferred_direction_xy is not None:
        direction = np.asarray(preferred_direction_xy, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 8.0:
            result["invalid_reason"] = "limb_direction_too_short"
            return result
        direction /= direction_norm
        relative_x = full_x - pixel_x
        relative_y = full_y - pixel_y
        axial_px = relative_x * direction[0] + relative_y * direction[1]
        perpendicular_px = np.abs(
            relative_x * direction[1] - relative_y * direction[0]
        )
        directional_mask = (
            (axial_px >= -float(directional_backward_tolerance_px))
            & (
                axial_px
                <= min(float(search_radius_px), 0.55 * direction_norm, 16.0)
            )
            & (perpendicular_px <= float(directional_half_width_px))
        )
        if not np.any(directional_mask):
            result["invalid_reason"] = "no_depth_in_limb_corridor"
            return result
        valid_x = valid_x[directional_mask]
        valid_y = valid_y[directional_mask]
        distances_px = distances_px[directional_mask]
        result["directional_constraint_used"] = True

    nearest_distance_px = float(np.min(distances_px))
    spatial_mask = distances_px <= nearest_distance_px + float(spatial_band_px)
    selected_depths = window[valid_y[spatial_mask], valid_x[spatial_mask]]
    selected_distances = distances_px[spatial_mask]
    if selected_depths.size == 0:
        result["invalid_reason"] = "no_depth_in_spatial_band"
        return result

    order = np.argsort(selected_depths, kind="stable")
    ordered_depths = selected_depths[order]
    ordered_distances = selected_distances[order]
    split_points = np.flatnonzero(
        np.diff(ordered_depths) > float(cluster_gap_m)
    ) + 1
    depth_groups = np.split(ordered_depths, split_points)
    distance_groups = np.split(ordered_distances, split_points)
    clusters = []
    for depths, distances in zip(depth_groups, distance_groups):
        if depths.size < int(minimum_cluster_size):
            continue
        minimum_distance = float(np.min(distances))
        clusters.append(
            {
                "depth_m": float(np.median(depths)),
                "size": int(depths.size),
                "minimum_distance_px": minimum_distance,
                "median_distance_px": float(np.median(distances)),
                "weighted_support": float(
                    np.sum(1.0 / (1.0 + distances - minimum_distance))
                ),
            }
        )
    result["cluster_count"] = len(clusters)
    result["fallback_distance_px"] = nearest_distance_px
    result["kinematic_reference_m"] = kinematic_reference_m
    if not clusters:
        result["invalid_reason"] = "no_supported_depth_cluster"
        return result

    if kinematic_reference_m is not None:
        chosen = min(
            clusters,
            key=lambda cluster: (
                abs(cluster["depth_m"] - float(kinematic_reference_m)),
                -cluster["weighted_support"],
                cluster["median_distance_px"],
            ),
        )
    elif preferred_direction_xy is not None:
        chosen = min(
            clusters,
            key=lambda cluster: (
                cluster["depth_m"],
                -cluster["weighted_support"],
                cluster["median_distance_px"],
            ),
        )
    else:
        chosen = max(
            clusters,
            key=lambda cluster: (
                cluster["weighted_support"],
                cluster["size"],
                -cluster["median_distance_px"],
            ),
        )
    result.update(
        {
            "depth_m": chosen["depth_m"],
            "valid": True,
            "valid_pixel_count": chosen["size"],
            "selected_cluster_size": chosen["size"],
            "fallback_distance_px": chosen["minimum_distance_px"],
            "invalid_reason": "",
        }
    )
    return result


def sample_candidates(
    depth_m: np.ndarray,
    candidates: list[dict],
    method: str,
) -> list[dict]:
    """Return one depth result for each cached landmark candidate."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            "Unsupported depth sampling method: {}. Expected one of {}.".format(
                method,
                ", ".join(SUPPORTED_METHODS),
            )
        )

    results: dict[str, dict] = {}
    candidates_by_joint = {
        candidate["canonical_joint"]: candidate for candidate in candidates
    }

    for candidate in candidates:
        joint = candidate["canonical_joint"]
        if not bool(candidate.get("in_image")):
            results[joint] = empty_result(method, "landmark_out_of_image")
            continue
        if method == "point_1x1":
            results[joint] = point_depth(
                depth_m,
                int(candidate["pixel_x"]),
                int(candidate["pixel_y"]),
            )
        elif method == "median_7x7":
            results[joint] = median_window_depth(
                depth_m,
                int(candidate["pixel_x"]),
                int(candidate["pixel_y"]),
                3,
                method="median_7x7",
            )
        else:
            results[joint] = adaptive_median_depth(
                depth_m,
                int(candidate["pixel_x"]),
                int(candidate["pixel_y"]),
            )

    if method != "limb_aware_cluster":
        return [results[candidate["canonical_joint"]] for candidate in candidates]

    for joint, elbow_joint in WRIST_TO_ELBOW.items():
        wrist = candidates_by_joint.get(joint)
        elbow = candidates_by_joint.get(elbow_joint)
        shoulder = candidates_by_joint.get(WRIST_TO_SHOULDER[joint])
        if wrist is None:
            continue
        if (
            not bool(wrist.get("in_image"))
            or float(wrist.get("visibility", 0.0)) < MIN_VISIBILITY
            or elbow is None
            or not bool(elbow.get("in_image"))
            or float(elbow.get("visibility", 0.0)) < MIN_VISIBILITY
        ):
            results[joint] = empty_result(
                "limb_aware_wrist_cluster",
                "wrist_forearm_direction_unavailable",
            )
            continue
        elbow_result = results.get(elbow_joint, {})
        shoulder_result = results.get(WRIST_TO_SHOULDER[joint], {})
        reference_m = kinematic_wrist_reference(
            wrist,
            elbow,
            shoulder,
            elbow_result.get("depth_m"),
            shoulder_result.get("depth_m"),
        )
        preferred_direction_xy = (
            float(elbow["pixel_x"]) - float(wrist["pixel_x"]),
            float(elbow["pixel_y"]) - float(wrist["pixel_y"]),
        )
        results[joint] = clustered_depth(
            depth_m,
            int(wrist["pixel_x"]),
            int(wrist["pixel_y"]),
            search_radius_px=32,
            cluster_gap_m=0.03,
            spatial_band_px=12.0,
            minimum_cluster_size=5,
            preferred_direction_xy=preferred_direction_xy,
            directional_half_width_px=4.0,
            directional_backward_tolerance_px=1.0,
            kinematic_reference_m=reference_m,
            method="limb_aware_wrist_cluster",
        )

    for candidate in candidates:
        joint = candidate["canonical_joint"]
        if joint in WRIST_TO_ELBOW:
            continue
        if results[joint]["valid"]:
            continue
        if not bool(candidate.get("in_image")):
            continue
        results[joint] = clustered_depth(
            depth_m,
            int(candidate["pixel_x"]),
            int(candidate["pixel_y"]),
            search_radius_px=32,
            cluster_gap_m=0.08,
            spatial_band_px=12.0,
            minimum_cluster_size=1,
            method="clustered_spatial_fallback",
        )

    return [results[candidate["canonical_joint"]] for candidate in candidates]

