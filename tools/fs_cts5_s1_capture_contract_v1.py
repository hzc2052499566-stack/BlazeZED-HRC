"""Pure-Python contract helpers for the FS-CTS5 S1 engineering pilot.

This module intentionally has no Isaac Sim or MediaPipe dependency.  It is
shared by the pre-capture freezer, the RGB/depth manipulation validator and
the GT-blind availability replay.  The final five-view selection is outside
this module: every audit row is emitted with ``selected=False``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "output/isaac_scenes/fs_cts5_20_bundle_v1"

VIEW_LAYOUT = (
    ("azm045_elm012", -45.0, -12.0),
    ("azp000_elm012", 0.0, -12.0),
    ("azp030_elm012", 30.0, -12.0),
    ("azm045_elp000", -45.0, 0.0),
    ("azp000_elp000", 0.0, 0.0),
    ("azp030_elp000", 30.0, 0.0),
    ("azm045_elp006", -45.0, 6.0),
    ("azp000_elp006", 0.0, 6.0),
    ("azp030_elp006", 30.0, 6.0),
)
VIEW_IDS = tuple(item[0] for item in VIEW_LAYOUT)
VIEW_ANGLES = {item[0]: (item[1], item[2]) for item in VIEW_LAYOUT}
ANCHOR_VIEW_ID = "azp000_elp000"
ACTIVE_RANGES = ((90, 119), (170, 199))
ACTIVE_FRAMES = tuple(
    frame for start, end in ACTIVE_RANGES for frame in range(start, end + 1)
)
INACTIVE_FRAMES = tuple(frame for frame in range(240) if frame not in ACTIVE_FRAMES)

TARGET_BONES = {
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "right_forearm": ("right_elbow", "right_wrist"),
}
NEGATIVE_CONTROL_BONES = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_thigh": ("right_hip", "right_knee"),
    "right_shank": ("right_knee", "right_ankle"),
}
ALL_BONES = {**TARGET_BONES, **NEGATIVE_CONTROL_BONES}

BONE_SAMPLE_COUNT = 33
OCCLUDED_OVERLAP_MIN = 0.80
CLEAR_OVERLAP_MAX = 0.05
MIN_FIRST_SURFACE_FORWARD_LEAD_M = 0.10
RAW_DEPTH_TO_OPERATIONAL_SCALE = 100.0
TARGET_RGB_HIT_FRACTION_MIN = 0.80
NEGATIVE_CONTROL_RGB_HIT_FRACTION_MAX = 0.05
INACTIVE_RGB_HIT_FRACTION_MAX = 0.01
DEPTH_SURFACE_ABS_TOLERANCE_M = 0.05
DEPTH_SURFACE_MATCH_FRACTION_MIN = 0.80
JOINT_FORWARD_DEPTH_RANGE_M = (2.0, 5.0)
BONE_LENGTH_RANGE_M = (0.10, 0.80)


class S1CaptureContractError(RuntimeError):
    """Raised when a frozen S1 engineering contract is violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S1CaptureContractError(f"Could not read JSON: {path}") from error
    if not isinstance(payload, dict):
        raise S1CaptureContractError(f"JSON root is not an object: {path}")
    return payload


def active_burst(frame: int) -> str | None:
    for start, end in ACTIVE_RANGES:
        if start <= int(frame) <= end:
            return f"{start}_{end}"
    return None


def enumerate_anchor_subsets() -> list[tuple[str, ...]]:
    others = sorted(set(VIEW_IDS) - {ANCHOR_VIEW_ID})
    subsets = [
        tuple(sorted((ANCHOR_VIEW_ID,) + choice))
        for choice in itertools.combinations(others, 4)
    ]
    if len(subsets) != 70 or len(set(subsets)) != 70:
        raise S1CaptureContractError("Anchor-containing subset denominator is not 70.")
    return subsets


def subset_diversity(view_ids: Sequence[str]) -> dict[str, Any]:
    subset = tuple(view_ids)
    if len(subset) != 5 or len(set(subset)) != 5:
        raise S1CaptureContractError("A future subset must contain five unique views.")
    if ANCHOR_VIEW_ID not in subset or set(subset) - set(VIEW_IDS):
        raise S1CaptureContractError("A subset lacks the anchor or contains an unknown view.")
    azimuths = {VIEW_ANGLES[view][0] for view in subset}
    elevations = {VIEW_ANGLES[view][1] for view in subset}
    return {
        "anchor_pass": True,
        "azimuth_diversity_pass": azimuths == {-45.0, 0.0, 30.0},
        "elevation_diversity_pass": elevations == {-12.0, 0.0, 6.0},
    }


def green_mask(rgb: Any) -> np.ndarray:
    """Return the frozen opaque-green mask used by the v2 occlusion pilot."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise S1CaptureContractError(f"Unexpected RGB array shape: {array.shape}.")
    channels = array[..., :3].astype(np.float32)
    if channels.size and float(np.max(channels)) > 1.5:
        channels /= 255.0
    red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]
    return (
        (green >= 0.30)
        & (green - red >= 0.12)
        & (green - blue >= 0.12)
        & (green >= 1.30 * np.maximum(red, blue))
    )


def project_camera_point(
    point_xyz_m: Sequence[float], intrinsics: Mapping[str, Any]
) -> tuple[float, float]:
    """Project one ZED camera point (+X forward, +Y left, +Z up)."""
    forward, left, up = (float(value) for value in point_xyz_m)
    if not all(math.isfinite(value) for value in (forward, left, up)) or forward <= 0:
        raise S1CaptureContractError("Projection point is not finite camera-forward data.")
    u = float(intrinsics["cx"]) - float(intrinsics["fx"]) * left / forward
    v = float(intrinsics["cy"]) - float(intrinsics["fy"]) * up / forward
    return u, v


def bone_samples(start: Sequence[float], end: Sequence[float]) -> np.ndarray:
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,) or not np.all(np.isfinite([first, second])):
        raise S1CaptureContractError("Bone endpoints must be finite 3-vectors.")
    fractions = np.linspace(0.0, 1.0, BONE_SAMPLE_COUNT, dtype=np.float64)
    return first[None, :] + fractions[:, None] * (second - first)[None, :]


def sample_boolean(mask: np.ndarray, u: float, v: float) -> bool:
    x, y = int(round(float(u))), int(round(float(v)))
    return bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])


def median_depth_3x3_operational(
    raw_depth: Any, u: float, v: float, *, scale: float = RAW_DEPTH_TO_OPERATIONAL_SCALE
) -> float | None:
    array = np.asarray(raw_depth, dtype=np.float64)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise S1CaptureContractError(f"Unexpected depth array shape: {array.shape}.")
    x, y = int(round(float(u))), int(round(float(v)))
    x0, x1 = max(0, x - 1), min(array.shape[1], x + 2)
    y0, y1 = max(0, y - 1), min(array.shape[0], y + 2)
    values = array[y0:y1, x0:x1]
    valid = np.isfinite(values) & (values > 0)
    if not bool(np.any(valid)):
        return None
    return float(np.median(values[valid]) * float(scale))


def raw_joint_available(row: Mapping[str, Any]) -> bool:
    try:
        if int(row["valid"]) != 1 or int(row["counts_as_measured_valid"]) != 1:
            return False
        if str(row["method"]) != "raw_measured":
            return False
        if str(row["provenance"]) not in {"raw_measured", "measured_dynamic_arm_v8_raw"}:
            return False
        point = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        point.shape == (3,)
        and np.all(np.isfinite(point))
        and JOINT_FORWARD_DEPTH_RANGE_M[0] <= point[0] <= JOINT_FORWARD_DEPTH_RANGE_M[1]
    )


def raw_bone_available(start: Mapping[str, Any], end: Mapping[str, Any]) -> bool:
    if not raw_joint_available(start) or not raw_joint_available(end):
        return False
    first = np.asarray([start["x_m"], start["y_m"], start["z_m"]], dtype=np.float64)
    second = np.asarray([end["x_m"], end["y_m"], end["z_m"]], dtype=np.float64)
    length = float(np.linalg.norm(second - first))
    return BONE_LENGTH_RANGE_M[0] <= length <= BONE_LENGTH_RANGE_M[1]


def validate_view_layout(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = [
        (
            str(item.get("candidate_id", "")),
            float(item.get("azimuth_offset_deg", math.nan)),
            float(item.get("elevation_offset_deg", math.nan)),
        )
        for item in candidates
    ]
    if observed != list(VIEW_LAYOUT):
        raise S1CaptureContractError("Camera-bank-v2 nine-view ID/angle mapping drifted.")
    prims = [str(item.get("prim_path", "")) for item in candidates]
    if len(set(prims)) != 9 or any(not value.startswith("/World/") for value in prims):
        raise S1CaptureContractError("Camera-bank-v2 prim identities are not nine unique paths.")
    return {
        "view_count": 9,
        "candidate_view_ids": list(VIEW_IDS),
        "anchor_view_id": ANCHOR_VIEW_ID,
        "mapping_pass": True,
    }


def canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

