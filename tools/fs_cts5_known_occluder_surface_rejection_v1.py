"""Pure GT-free known-occluder surface-rejection primitives for FS-CTS5."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(1, str(TOOLS))

from tools import fs_cts5_camera_bank_v1 as camera_bank
from tools import fs_cts5_s1_arm_occluder_v1 as ray_core
from tools import fs_cts5_s1_capture_contract_v1 as raw_contract


SURFACE_DEPTH_TOLERANCE_M = 0.05
RAY_EXTENT_M = 10.0


class KnownOccluderSurfaceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise KnownOccluderSurfaceError(message)


def camera_point_world_ray(
    camera_point_xyz_m: Sequence[float], candidate: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, float]:
    point = np.asarray(camera_point_xyz_m, dtype=np.float64)
    require(point.shape == (3,) and np.all(np.isfinite(point)), "Camera point is not a finite triplet.")
    require(float(point[0]) > 0.0, "Camera point is not forward-positive.")
    eye = np.asarray(candidate["eye_world_operational_m"], dtype=np.float64)
    require(eye.shape == (3,) and np.all(np.isfinite(eye)), "Camera eye is invalid.")
    right, up, forward = camera_bank.camera_basis(
        candidate["eye_world_operational_m"], candidate["aim_world_operational_m"]
    )
    direction = (
        float(point[0]) * np.asarray(forward, dtype=np.float64)
        - float(point[1]) * np.asarray(right, dtype=np.float64)
        + float(point[2]) * np.asarray(up, dtype=np.float64)
    )
    norm = float(np.linalg.norm(direction))
    require(math.isfinite(norm) and norm > 1.0e-12, "Camera ray has zero length.")
    return eye, direction / norm, float(point[0])


def first_surface_forward_depth_m(
    camera_point_xyz_m: Sequence[float],
    candidate: Mapping[str, Any],
    obbs: Sequence[Mapping[str, Any]],
) -> float | None:
    require(bool(obbs), "At least one registered OBB is required.")
    eye, direction, _ = camera_point_world_ray(camera_point_xyz_m, candidate)
    target = eye + RAY_EXTENT_M * direction
    hits = [ray_core.ray_obb_first_hit_distance(eye, target, obb) for obb in obbs]
    finite = [float(value) for value in hits if value is not None]
    if not finite:
        return None
    _right, _up, forward = camera_bank.camera_basis(
        candidate["eye_world_operational_m"], candidate["aim_world_operational_m"]
    )
    cosine = float(np.dot(direction, np.asarray(forward, dtype=np.float64)))
    require(math.isfinite(cosine) and cosine > 1.0e-12, "Registered ray is not camera-forward.")
    return min(finite) * cosine


def classify_raw_joint(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    obbs: Sequence[Mapping[str, Any]],
    *,
    active: bool,
    tolerance_m: float = SURFACE_DEPTH_TOLERANCE_M,
) -> dict[str, Any]:
    """Classify one immutable raw row without mutating its validity or coordinates."""
    require(abs(float(tolerance_m) - SURFACE_DEPTH_TOLERANCE_M) <= 1.0e-12, "The frozen 50 mm tolerance changed.")
    raw_available = raw_contract.raw_joint_available(row)
    if not raw_available:
        return {
            "raw_joint_available": False,
            "first_surface_forward_depth_m": None,
            "surface_depth_residual_m": None,
            "occluder_rejected": False,
            "output_provenance": "raw_invalid",
        }
    measured = float(row["x_m"])
    if not active:
        return {
            "raw_joint_available": True,
            "first_surface_forward_depth_m": None,
            "surface_depth_residual_m": None,
            "occluder_rejected": False,
            "output_provenance": "raw_measured",
        }
    point = [float(row[name]) for name in ("x_m", "y_m", "z_m")]
    expected = first_surface_forward_depth_m(point, candidate, obbs)
    residual = None if expected is None else abs(measured - expected)
    rejected = residual is not None and residual <= SURFACE_DEPTH_TOLERANCE_M + 1.0e-12
    return {
        "raw_joint_available": True,
        "first_surface_forward_depth_m": expected,
        "surface_depth_residual_m": residual,
        "occluder_rejected": bool(rejected),
        "output_provenance": "occluder_rejected" if rejected else "raw_measured",
    }


def filtered_bone_available(
    start_row: Mapping[str, Any],
    end_row: Mapping[str, Any],
    start_decision: Mapping[str, Any],
    end_decision: Mapping[str, Any],
) -> bool:
    return bool(
        raw_contract.raw_bone_available(start_row, end_row)
        and start_decision.get("occluder_rejected") is False
        and end_decision.get("occluder_rejected") is False
    )
