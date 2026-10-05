"""Analyse the registered Dynamic-GT partial-occlusion engineering Pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = (
    ROOT
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "dynamic_partial_occlusion_v1_engineering_pilot"
    / "unit_corrected"
    / "rep_01"
)
DEFAULT_SCENE_MANIFEST = (
    ROOT
    / "output"
    / "isaac_scenes"
    / "controlled_arm_motion_v6"
    / "controlled_arm_motion_v6_06_partial_occlusion_v1_manifest.json"
)
OUTPUT_NAME = "partial_occlusion_pilot_report.json"
JOINTS = ("right_shoulder", "right_elbow", "right_wrist")
ENDPOINT_JOINTS = ("right_elbow", "right_wrist")
METHODS = (
    "raw_measured",
    "k2_guarded",
    "k3_inferred",
    "k4_inferred",
)
RECOVERY_STABLE_FRAMES = 3
RECOVERY_SEARCH_FRAMES = 30


class AnalysisError(RuntimeError):
    """Raised when the Pilot analysis contract is incomplete."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON root must be an object: {path}")
    return value


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def point(row: dict) -> np.ndarray | None:
    if str(row.get("valid", "")).strip() != "1":
        return None
    values = [float(row[field]) for field in ("x_m", "y_m", "z_m")]
    if not all(math.isfinite(value) for value in values):
        return None
    return np.asarray(values, dtype=np.float64)


def active_frames(scene_manifest: dict) -> set[int]:
    output = set()
    for block in scene_manifest["occlusion_intervals"]:
        start = int(block["start_frame"])
        end = int(block["end_frame_inclusive"])
        output.update(range(start, end + 1))
    if len(output) != 60:
        raise AnalysisError("Registered active frame count must be 60.")
    return output


def green_mask(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise AnalysisError(f"Unexpected RGB array shape: {array.shape}")
    channels = array[..., :3].astype(np.float32)
    if channels.max(initial=0.0) > 1.5:
        channels /= 255.0
    red, green, blue = (
        channels[..., 0],
        channels[..., 1],
        channels[..., 2],
    )
    return (
        (green >= 0.30)
        & (green - red >= 0.12)
        & (green - blue >= 0.12)
        & (green >= 1.30 * np.maximum(red, blue))
    )


def projected_pixel(gt_row: dict, manifest_row: dict) -> tuple[int, int]:
    forward = float(gt_row["gt_x_m"])
    if forward <= 0.0:
        raise AnalysisError("GT point is not in front of CameraLeft.")
    y_left = float(gt_row["gt_y_m"])
    z_up = float(gt_row["gt_z_m"])
    u = float(manifest_row["cx"]) - float(manifest_row["fx"]) * (
        y_left / forward
    )
    v = float(manifest_row["cy"]) - float(manifest_row["fy"]) * (
        z_up / forward
    )
    return int(round(u)), int(round(v))


def patch_green_fraction(mask: np.ndarray, u: int, v: int) -> float:
    radius = 4
    height, width = mask.shape
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def first_stable_recovery(
    estimate_by_key: dict,
    method: str,
    end_frame: int,
) -> int | None:
    for offset in range(1, RECOVERY_SEARCH_FRAMES + 1):
        first = end_frame + offset
        last = first + RECOVERY_STABLE_FRAMES - 1
        if last >= 240:
            break
        if all(
            point(estimate_by_key[(method, frame, joint)]) is not None
            for frame in range(first, last + 1)
            for joint in ENDPOINT_JOINTS
        ):
            return offset
    return None


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def analyse(experiment_dir: Path, scene_manifest_path: Path) -> dict:
    required = {
        "unit_state": experiment_dir / "unit_correction_state.json",
        "cache_state": (
            experiment_dir / "landmark_cache" / "landmark_cache_state.json"
        ),
        "replay_state": (
            experiment_dir
            / "dynamic_k4_replay"
            / "dynamic_kinematic_replay_state.json"
        ),
        "manifest": experiment_dir / "rgbd_manifest.csv",
        "gt": experiment_dir / "ground_truth_joints.csv",
        "landmarks": (
            experiment_dir / "landmark_cache" / "landmarks_2d.csv"
        ),
        "landmark_frames": (
            experiment_dir / "landmark_cache" / "landmark_frames.csv"
        ),
        "estimates": (
            experiment_dir
            / "dynamic_k4_replay"
            / "dynamic_kinematic_estimates.csv"
        ),
        "frames": experiment_dir / "rgbd_frames",
        "scene_manifest": scene_manifest_path,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise AnalysisError("Missing Pilot artifact(s): " + ", ".join(missing))
    for label in ("unit_state", "cache_state", "replay_state"):
        state = read_json(required[label])
        if state.get("status") != "complete":
            raise AnalysisError(f"{label} is not complete.")

    scene = read_json(required["scene_manifest"])
    active = active_frames(scene)
    inactive = set(range(240)) - active
    manifests = {
        int(row["sequence_index"]): row
        for row in read_csv(required["manifest"])
    }
    if set(manifests) != set(range(240)):
        raise AnalysisError("RGB-D manifest sequence set is not 0..239.")
    gt_rows = read_csv(required["gt"])
    gt = {
        (int(row["sequence_index"]), row["canonical_joint"]): row
        for row in gt_rows
    }
    landmark_rows = read_csv(required["landmarks"])
    landmarks = {
        (int(row["frame_index"]), row["canonical_joint"]): row
        for row in landmark_rows
    }
    frame_rows = read_csv(required["landmark_frames"])
    estimates = read_csv(required["estimates"])
    estimate_by_key = {
        (
            row["method"],
            int(row["sequence_index"]),
            row["canonical_joint"],
        ): row
        for row in estimates
    }

    green_at_joint: dict[str, dict[str, list[float]]] = {
        joint: {"active": [], "inactive": []}
        for joint in ENDPOINT_JOINTS
    }
    for sequence, manifest in manifests.items():
        rgb = np.load(required["frames"] / manifest["rgb_file"])
        mask = green_mask(rgb)
        phase = "active" if sequence in active else "inactive"
        for joint in ENDPOINT_JOINTS:
            u, v = projected_pixel(gt[(sequence, joint)], manifest)
            green_at_joint[joint][phase].append(
                patch_green_fraction(mask, u, v)
            )
    manipulation = {
        joint: {
            phase: {
                "mean_patch_green_fraction": mean(values),
                "p05_patch_green_fraction": percentile(values, 5.0),
                "p95_patch_green_fraction": percentile(values, 95.0),
            }
            for phase, values in phases.items()
        }
        for joint, phases in green_at_joint.items()
    }
    manipulation["checks"] = {
        "active_elbow_green_mean_at_least_0p50": (
            manipulation["right_elbow"]["active"][
                "mean_patch_green_fraction"
            ]
            >= 0.50
        ),
        "active_wrist_green_mean_at_least_0p50": (
            manipulation["right_wrist"]["active"][
                "mean_patch_green_fraction"
            ]
            >= 0.50
        ),
        "inactive_elbow_green_mean_below_0p10": (
            manipulation["right_elbow"]["inactive"][
                "mean_patch_green_fraction"
            ]
            < 0.10
        ),
        "inactive_wrist_green_mean_below_0p10": (
            manipulation["right_wrist"]["inactive"][
                "mean_patch_green_fraction"
            ]
            < 0.10
        ),
    }
    manipulation["checks"]["all_passed"] = all(
        manipulation["checks"].values()
    )

    detection = {}
    for phase, frames in (("active", active), ("inactive", inactive)):
        selected = [
            row
            for row in frame_rows
            if int(row["frame_index"]) in frames
        ]
        detection[phase] = mean(
            [float(row["body_detected"]) for row in selected]
        )

    landmark_metrics = {}
    for joint in JOINTS:
        landmark_metrics[joint] = {}
        for phase, frames in (("active", active), ("inactive", inactive)):
            selected = [
                landmarks[(sequence, joint)]
                for sequence in sorted(frames)
            ]
            landmark_metrics[joint][phase] = {
                "eligible_rate": mean(
                    [float(row["eligible"]) for row in selected]
                ),
                "in_image_rate": mean(
                    [float(row["in_image"]) for row in selected]
                ),
                "visibility_mean": mean(
                    [float(row["visibility"]) for row in selected]
                ),
            }

    method_metrics = {}
    for method in METHODS:
        method_metrics[method] = {}
        for phase, frames in (("active", active), ("inactive", inactive)):
            rows = [
                estimate_by_key[(method, sequence, joint)]
                for sequence in sorted(frames)
                for joint in ENDPOINT_JOINTS
            ]
            valid = [row for row in rows if point(row) is not None]
            errors = [
                float(
                    np.linalg.norm(
                        point(row)
                        - np.asarray(
                            [
                                float(
                                    gt[
                                        (
                                            int(row["sequence_index"]),
                                            row["canonical_joint"],
                                        )
                                    ][field]
                                )
                                for field in ("gt_x_m", "gt_y_m", "gt_z_m")
                            ],
                            dtype=np.float64,
                        )
                    )
                )
                * 1000.0
                for row in valid
            ]
            method_metrics[method][phase] = {
                "joint_sample_coverage": len(valid) / len(rows),
                "two_joint_position_error_mean_mm": mean(errors),
                "two_joint_position_error_p95_mm": percentile(errors, 95.0),
            }
        method_metrics[method]["recovery_frames_after_intervals"] = [
            first_stable_recovery(
                estimate_by_key,
                method,
                int(block["end_frame_inclusive"]),
            )
            for block in scene["occlusion_intervals"]
        ]

    return {
        "schema_version": 1,
        "status": "complete",
        "purpose": "dynamic_partial_occlusion_v1_engineering_pilot",
        "claim_eligibility": "excluded_engineering_pilot_only",
        "experiment_dir": str(experiment_dir),
        "scene_manifest": str(scene_manifest_path),
        "registered_active_frame_count": len(active),
        "registered_inactive_frame_count": len(inactive),
        "occlusion_intervals": scene["occlusion_intervals"],
        "manipulation_check": manipulation,
        "body_detection_rate": detection,
        "landmark_metrics": landmark_metrics,
        "method_metrics": method_metrics,
        "interpretation_rules": [
            "The manipulation check must pass before robustness metrics are "
            "interpreted.",
            "This single run is an engineering Pilot, not confirmatory "
            "evidence.",
            "K3/K4 validity is inferred coverage and is not measured-depth "
            "validity.",
            "The endpoint is right elbow plus right wrist position error, "
            "not MPJPE.",
            "Recovery is the first of three consecutive frames with both "
            "endpoint joints valid.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        default=DEFAULT_SCENE_MANIFEST,
    )
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output = experiment_dir / OUTPUT_NAME
    if output.exists():
        raise AnalysisError(f"Refusing to overwrite Pilot report: {output}")
    report = analyse(experiment_dir, args.scene_manifest.resolve())
    atomic_write_json(output, report)
    print(f"Complete: {output}")
    print(
        "Manipulation check passed: "
        f"{report['manipulation_check']['checks']['all_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
