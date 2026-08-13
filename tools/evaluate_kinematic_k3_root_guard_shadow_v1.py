"""Post-hoc K3 root-depth guard diagnostic on completed live Pilots.

This evaluator never rewrites measured rows or the original K3 output.  It
replays the ray/bone inference with a separate, explicitly inferred shoulder
root whose frame-to-frame depth change is limited by the already-frozen
dynamic-arm 60 mm temporal gate.  Pelvis translation is added to the
one-step prediction before the rate limit is applied.

The result is exploratory evidence for a possible successor to K3, not a
confirmatory K3 result and not a 3D-accuracy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import kinematic_k3_inferred as k3
import kinematic_k4_inferred as k4


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "controlled_arm_motion_k3_inferred_v1_pilot"
    / "right_arm_k3_inferred_v1"
    / "d_3p50m"
)
REPEATS = ("rep_01", "rep_02")
RATE_LIMIT_M = 0.060
OUTPUT_CSV_NAME = "k3_root_guard_shadow_joints.csv"
RUN_REPORT_NAME = "k3_root_guard_shadow_report.json"
PAIR_REPORT_NAME = "k3_root_guard_shadow_pair_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    return parser.parse_args()


def optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def is_valid(row: dict | None) -> bool:
    return bool(row is not None and str(row.get("valid", "0")) == "1")


def guarded_root_depth(
    raw_depth_m: float | None,
    previous_guarded_depth_m: float | None,
    pelvis_depth_m: float | None,
    previous_pelvis_depth_m: float | None,
    rate_limit_m: float = RATE_LIMIT_M,
) -> dict:
    return k4.guarded_root_depth(
        raw_depth_m,
        previous_guarded_depth_m,
        pelvis_depth_m,
        previous_pelvis_depth_m,
        rate_limit_m,
    )


def load_rows(path: Path, joint_field: str) -> dict[int, dict[str, dict]]:
    rows: dict[int, dict[str, dict]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(int(row["frame_index"]), {})[
                row[joint_field]
            ] = row
    return rows


def point_step_statistics(
    rows: list[dict],
    joint: str,
    warmup_frames: int,
) -> dict:
    selected = [
        row
        for row in rows
        if row["joint"] == joint
        and row["valid"]
        and row["frame_index"] >= warmup_frames
    ]
    selected.sort(key=lambda row: row["frame_index"])
    transitions = []
    for previous, current in zip(selected, selected[1:]):
        if current["frame_index"] != previous["frame_index"] + 1:
            continue
        step_mm = float(
            np.linalg.norm(current["point"] - previous["point"]) * 1000.0
        )
        transitions.append(
            {
                "from_frame": previous["frame_index"],
                "to_frame": current["frame_index"],
                "step_3d_mm": step_mm,
            }
        )
    values = np.asarray(
        [row["step_3d_mm"] for row in transitions],
        dtype=np.float64,
    )
    return {
        "valid_count": len(selected),
        "step_3d_p95_mm": (
            float(np.percentile(values, 95)) if values.size else None
        ),
        "step_3d_maximum_mm": (
            float(np.max(values)) if values.size else None
        ),
        "steps_over_50mm": int(np.count_nonzero(values > 50.0)),
        "steps_over_100mm": int(np.count_nonzero(values > 100.0)),
        "largest_transitions": sorted(
            transitions,
            key=lambda row: row["step_3d_mm"],
            reverse=True,
        )[:10],
    }


def write_shadow_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "frame_index",
        "zed_timestamp_ns",
        "joint",
        "valid",
        "provenance",
        "parent_provenance",
        "root_branch",
        "x_m",
        "y_m",
        "z_m",
        "depth_m",
        "pixel_x",
        "pixel_y",
        "visibility",
        "bone_reference_m",
        "bone_target_m",
        "constraint_relaxation_mm",
        "bone_residual_mm",
        "invalid_reason",
        "raw_shoulder_depth_m",
        "guarded_shoulder_depth_m",
        "shoulder_prediction_m",
        "shoulder_raw_prediction_residual_mm",
        "shoulder_root_clipped",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in fields}
            )


def analyze_run(run_dir: Path) -> dict:
    summary = json.loads(
        (run_dir / "live_performance_summary.json").read_text(
            encoding="utf-8"
        )
    )
    original_qc = json.loads(
        (run_dir / "k3_inferred_live_pilot_qc.json").read_text(
            encoding="utf-8"
        )
    )
    configuration = summary["configuration"]
    profile = json.loads(
        Path(configuration["kinematic_profile_path"]).read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(
        Path(configuration["kinematic_config_path"]).read_text(
            encoding="utf-8"
        )
    )
    intrinsics_dict = configuration["left_camera_intrinsics"]
    intrinsics = tuple(
        float(intrinsics_dict[name])
        for name in ("fx", "fy", "cx", "cy")
    )
    warmup_frames = int(configuration["warmup_frames"])
    tracking = load_rows(run_dir / "tracking_joints.csv", "canonical_joint")
    original_k3 = load_rows(run_dir / "k3_inferred_joints.csv", "joint")

    upper_arm = k3.profile_bone(profile, "right_upper_arm")
    forearm = k3.profile_bone(profile, "right_forearm")
    upper_reference_m = float(upper_arm["reference_length_m"])
    forearm_reference_m = float(forearm["reference_length_m"])
    upper_tolerance_m = k3.profile_tolerance_m(
        profile, config, "right_upper_arm"
    )
    forearm_tolerance_m = k3.profile_tolerance_m(
        profile, config, "right_forearm"
    )

    previous_guarded_depth_m = None
    previous_pelvis_depth_m = None
    shadow_rows: list[dict] = []
    root_rows: list[dict] = []
    steady_clip_frames: list[int] = []
    for frame_index in sorted(tracking):
        measured = tracking[frame_index]
        original = original_k3[frame_index]
        shoulder = measured["right_shoulder"]
        pelvis = measured["pelvis"]
        raw_shoulder_depth_m = (
            optional_float(shoulder["depth_m"])
            if is_valid(shoulder)
            else None
        )
        pelvis_depth_m = (
            optional_float(pelvis["depth_m"])
            if is_valid(pelvis)
            else None
        )
        root = guarded_root_depth(
            raw_shoulder_depth_m,
            previous_guarded_depth_m,
            pelvis_depth_m,
            previous_pelvis_depth_m,
        )
        if root["valid"]:
            previous_guarded_depth_m = float(root["depth_m"])
        if pelvis_depth_m is not None:
            previous_pelvis_depth_m = pelvis_depth_m
        if root["clipped"] and frame_index >= warmup_frames:
            steady_clip_frames.append(frame_index)

        shoulder_pixel = (
            int(shoulder["pixel_x"]),
            int(shoulder["pixel_y"]),
        )
        shoulder_point = (
            k3.reconstruct_from_depth(
                float(root["depth_m"]),
                shoulder_pixel[0],
                shoulder_pixel[1],
                *intrinsics,
            )
            if root["valid"]
            else None
        )
        if shoulder_point is not None:
            root_rows.append(
                {
                    "frame_index": frame_index,
                    "joint": "guarded_right_shoulder",
                    "valid": True,
                    "point": shoulder_point,
                }
            )

        elbow_source = original["right_elbow"]
        wrist_source = original["right_wrist"]
        elbow_candidate = {
            "in_image": True,
            "visibility": float(elbow_source["visibility"]),
            "pixel_x": int(elbow_source["pixel_x"]),
            "pixel_y": int(elbow_source["pixel_y"]),
        }
        wrist_candidate = {
            "in_image": True,
            "visibility": float(wrist_source["visibility"]),
            "pixel_x": int(wrist_source["pixel_x"]),
            "pixel_y": int(wrist_source["pixel_y"]),
        }
        elbow_result = k3.infer_child(
            "right_elbow",
            elbow_candidate,
            float(root["depth_m"]) if root["valid"] else None,
            shoulder_pixel if root["valid"] else None,
            shoulder_point,
            "inferred_guarded_right_shoulder",
            upper_reference_m,
            upper_tolerance_m,
            "far",
            intrinsics,
        )
        elbow_pixel = (
            elbow_candidate["pixel_x"],
            elbow_candidate["pixel_y"],
        )
        wrist_result = k3.infer_child(
            "right_wrist",
            wrist_candidate,
            elbow_result["depth_m"],
            elbow_pixel,
            elbow_result["point"],
            "inferred_right_elbow_root_guard_shadow",
            forearm_reference_m,
            forearm_tolerance_m,
            "far",
            intrinsics,
        )
        for source, result in (
            (elbow_source, elbow_result),
            (wrist_source, wrist_result),
        ):
            point = result["point"]
            shadow_rows.append(
                {
                    "frame_index": frame_index,
                    "zed_timestamp_ns": int(source["zed_timestamp_ns"]),
                    "joint": result["joint"],
                    "valid": bool(result["valid"]),
                    "provenance": (
                        "inferred_ray_bone_k3_root_guard_shadow"
                    ),
                    "parent_provenance": result["parent_provenance"],
                    "root_branch": result["root_branch"],
                    "point": point,
                    "x_m": point[0] if point is not None else "",
                    "y_m": point[1] if point is not None else "",
                    "z_m": point[2] if point is not None else "",
                    "depth_m": (
                        result["depth_m"]
                        if result["depth_m"] is not None
                        else ""
                    ),
                    "pixel_x": int(source["pixel_x"]),
                    "pixel_y": int(source["pixel_y"]),
                    "visibility": float(source["visibility"]),
                    "bone_reference_m": result["bone_reference_m"],
                    "bone_target_m": (
                        result["bone_target_m"]
                        if result["bone_target_m"] is not None
                        else ""
                    ),
                    "constraint_relaxation_mm": (
                        1000.0 * result["constraint_relaxation_m"]
                        if result["constraint_relaxation_m"] is not None
                        else ""
                    ),
                    "bone_residual_mm": (
                        1000.0 * result["bone_residual_m"]
                        if result["bone_residual_m"] is not None
                        else ""
                    ),
                    "invalid_reason": result["invalid_reason"],
                    "raw_shoulder_depth_m": (
                        raw_shoulder_depth_m
                        if raw_shoulder_depth_m is not None
                        else ""
                    ),
                    "guarded_shoulder_depth_m": (
                        root["depth_m"] if root["valid"] else ""
                    ),
                    "shoulder_prediction_m": (
                        root["prediction_m"]
                        if root["prediction_m"] is not None
                        else ""
                    ),
                    "shoulder_raw_prediction_residual_mm": (
                        1000.0 * root["raw_prediction_residual_m"]
                        if root["raw_prediction_residual_m"] is not None
                        else ""
                    ),
                    "shoulder_root_clipped": int(root["clipped"]),
                }
            )

    steady_frame_count = int(summary["steady_frame_count"])
    root_stats = point_step_statistics(
        root_rows, "guarded_right_shoulder", warmup_frames
    )
    elbow_stats = point_step_statistics(
        shadow_rows, "right_elbow", warmup_frames
    )
    wrist_stats = point_step_statistics(
        shadow_rows, "right_wrist", warmup_frames
    )
    elbow_coverage = elbow_stats["valid_count"] / steady_frame_count
    wrist_coverage = wrist_stats["valid_count"] / steady_frame_count
    feasibility_passed = bool(
        elbow_coverage >= 0.95
        and wrist_coverage >= 0.95
        and elbow_stats["steps_over_100mm"] == 0
        and wrist_stats["steps_over_100mm"] == 0
    )
    residuals = [
        abs(float(row["shoulder_raw_prediction_residual_mm"]))
        for row in shadow_rows[::2]
        if row["frame_index"] >= warmup_frames
        and row["shoulder_raw_prediction_residual_mm"] != ""
    ]
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "post_hoc_k3_root_guard_shadow_diagnostic",
        "run_dir": str(run_dir),
        "output_semantics": {
            "counts_as_measured_valid": False,
            "modifies_original_k3_output": False,
            "synchronised_ground_truth_available": False,
            "accuracy_claim_permitted": False,
            "confirmatory_claim_permitted": False,
        },
        "guard": {
            "type": "pelvis_compensated_shoulder_depth_rate_limit",
            "rate_limit_m_per_processed_frame": RATE_LIMIT_M,
            "threshold_source": (
                "Frozen 0.060 m dynamic-elbow temporal gate; not fitted "
                "to the K3 Pilot outcomes."
            ),
            "steady_clipped_frame_count": len(steady_clip_frames),
            "steady_clipped_frame_indices": steady_clip_frames,
            "absolute_raw_prediction_residual_p95_mm": (
                float(np.percentile(residuals, 95)) if residuals else None
            ),
            "absolute_raw_prediction_residual_maximum_mm": (
                max(residuals) if residuals else None
            ),
        },
        "original_k3": {
            "algorithm_status": original_qc["algorithm_status"],
            "right_elbow": original_qc["metrics"]["right_elbow"],
            "right_wrist": original_qc["metrics"]["right_wrist"],
        },
        "shadow": {
            "guarded_right_shoulder": root_stats,
            "right_elbow": {
                "coverage_rate": elbow_coverage,
                **elbow_stats,
            },
            "right_wrist": {
                "coverage_rate": wrist_coverage,
                **wrist_stats,
            },
        },
        "decision": {
            "exploratory_shadow_feasibility_passed": feasibility_passed,
            "interpretation": (
                "A pass supports pre-registering a separate live successor "
                "Pilot. It does not repair K3 Repeat2 and does not establish "
                "3D accuracy."
            ),
        },
        "outputs": {
            "shadow_rows_csv": str(run_dir / OUTPUT_CSV_NAME),
            "run_report_json": str(run_dir / RUN_REPORT_NAME),
        },
    }
    write_shadow_rows(run_dir / OUTPUT_CSV_NAME, shadow_rows)
    (run_dir / RUN_REPORT_NAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def analyze_pair(base_dir: Path) -> dict:
    base_dir = base_dir.resolve()
    reports = [
        analyze_run(base_dir / repeat)
        for repeat in REPEATS
    ]
    passed = all(
        report["decision"]["exploratory_shadow_feasibility_passed"]
        for report in reports
    )
    pair_report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "post_hoc_k3_root_guard_shadow_pair_diagnostic",
        "base_dir": str(base_dir),
        "repeat_reports": reports,
        "decision": {
            "all_repeat_shadows_passed": passed,
            "pre_register_separate_live_successor_pilot": passed,
            "do_not_relabel_k3_repeat2": True,
            "accuracy_claim_permitted": False,
        },
    }
    (base_dir / PAIR_REPORT_NAME).write_text(
        json.dumps(
            pair_report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return pair_report


def main() -> int:
    args = parse_args()
    report = analyze_pair(args.base_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
