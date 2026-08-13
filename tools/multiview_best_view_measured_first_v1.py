"""GT-free measured-first View A/C arm selector.

This module contains no file I/O and no ground-truth interface.  Each view
provides reliable K2 measured output when available and otherwise K4 inferred
fallback.  The complete arm is selected lexicographically by measured count,
candidate completeness, and minimum BlazePose visibility; exact ties choose A.
"""

from __future__ import annotations

import math
from typing import Any


ARM_JOINTS = ("right_elbow", "right_wrist")
TORSO_JOINTS = ("right_shoulder", "pelvis")
OCCLUSION_MARGIN_M = 0.50
METHOD_VERSION = "view_ac_measured_first_best_view_v1_20260802"


def finite_point(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    point = tuple(float(component) for component in value)
    if len(point) != 3 or not all(math.isfinite(component) for component in point):
        raise ValueError("Point must contain three finite coordinates.")
    return point


def depth_reliability(joint_depth_m: float | None, torso_depths_m: tuple[float | None, float | None]) -> str:
    if joint_depth_m is None:
        return "no_measured_depth"
    if any(value is None for value in torso_depths_m):
        return "torso_reference_unavailable"
    torso = sum(float(value) for value in torso_depths_m) / 2.0
    if torso - float(joint_depth_m) > OCCLUSION_MARGIN_M:
        return "occluded_suspected"
    return "reliable"


def measured_first(k2_point, k4_point, reliability: str) -> tuple[tuple[float, float, float] | None, str]:
    measured = finite_point(k2_point)
    inferred = finite_point(k4_point)
    if reliability == "reliable" and measured is not None:
        return measured, "k2_reliable_measured"
    if inferred is not None:
        return inferred, "k4_fallback_inferred"
    return None, "unavailable"


def transform(point, transform: dict[str, Any]):
    point = finite_point(point)
    if point is None:
        return None
    rotation = transform["rotation_3x3"]
    translation = transform["translation_m"]
    return tuple(sum(float(rotation[i][j]) * point[j] for j in range(3)) + float(translation[i]) for i in range(3))


def build_view_arm(joints: dict[str, dict[str, Any]], transform_to_a: dict[str, Any] | None = None) -> dict[str, Any]:
    output = {}
    for joint in ARM_JOINTS:
        item = joints[joint]
        point, source = measured_first(item.get("k2_point"), item.get("k4_point"), str(item["reliability"]))
        if point is not None and transform_to_a is not None:
            point = transform(point, transform_to_a)
        output[joint] = {"point_a": point, "source": source, "reliability": str(item["reliability"]), "visibility": float(item["visibility"])}
    measured_count = sum(item["source"] == "k2_reliable_measured" for item in output.values())
    complete = int(all(item["point_a"] is not None for item in output.values()))
    minimum_visibility = min(item["visibility"] for item in output.values())
    return {"joints": output, "score": (measured_count, complete, minimum_visibility), "complete": bool(complete)}


def select_arm(view_a: dict[str, Any], view_c: dict[str, Any]) -> dict[str, Any]:
    if view_a["complete"] and view_c["complete"]:
        selected = "c" if tuple(view_c["score"]) > tuple(view_a["score"]) else "a"
    elif view_a["complete"]:
        selected = "a"
    elif view_c["complete"]:
        selected = "c"
    else:
        selected = ""
    source = view_a if selected == "a" else view_c if selected == "c" else None
    return {
        "selected_view": selected,
        "score_a": tuple(view_a["score"]),
        "score_c": tuple(view_c["score"]),
        "joints": {joint: (source["joints"][joint] if source else {"point_a": None, "source": "unavailable", "reliability": "unavailable", "visibility": 0.0}) for joint in ARM_JOINTS},
        "method_version": METHOD_VERSION,
    }
