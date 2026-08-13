"""Analyse the registered clean-static temporal follow-up."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "output" / "experiments" / "camera_distance_sweep" / "d_3p50m"
OUTPUT_ROOT = (
    ROOT
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "clean_static_temporal_followup_v1"
)
PROTOCOL = OUTPUT_ROOT / "protocol_lock_v1.json"
SUMMARY = OUTPUT_ROOT / "formal_summary.json"
RUNS = ("rep_01", "rep_02", "rep_03")
CONDITIONS = (
    "T0_frame_independent",
    "T1_tracker_only",
    "T2_tracker_plus_smoothing",
)
PRIMARY_JOINTS = ("right_elbow", "right_wrist")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarise(values: list[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p95": percentile(finite, 0.95),
        "max": max(finite) if finite else None,
    }


def distance_mm(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second))) * 1000.0


def vector_jitter(values: list[tuple[float, ...]], scale: float) -> float | None:
    if len(values) < 2:
        return None
    return math.sqrt(
        sum(statistics.pstdev(point[axis] for point in values) ** 2 for axis in range(len(values[0])))
    ) * scale


def load_truth(source: Path) -> dict[tuple[int, str], tuple[float, float, float]]:
    result = {}
    for row in read_csv(source / "ground_truth_joints.csv"):
        result[(int(row["sample_index"]), row["canonical_joint"])] = (
            float(row["gt_x_m"]),
            float(row["gt_y_m"]),
            float(row["gt_z_m"]),
        )
    return result


def validate_condition(condition_root: Path, condition: str, protocol: dict) -> None:
    cache = read_json(condition_root / "landmark_cache" / "landmark_cache_state.json")
    replay = read_json(condition_root / "replay" / "offline_tracker_state.json")
    expected = protocol["estimation_contract"]["conditions"][condition]
    checks = {
        "cache complete": cache.get("status") == "complete",
        "replay complete": replay.get("status") == "complete",
        "static_image_mode": cache.get("static_image_mode") == expected["static_image_mode"],
        "smooth_landmarks": cache.get("smooth_landmarks") == expected["smooth_landmarks"],
        "model_complexity": cache.get("model_complexity") == 0,
        "roi_scale": cache.get("roi_scale") == 0.65,
        "depth_sampling": replay.get("depth_sampling_method") == "median_7x7",
        "warmup": replay.get("warmup_frames") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{condition_root}: invalid config {failed}")
    performance = read_json(condition_root / "replay" / "performance_summary.json")
    if performance.get("method_contract", {}).get("temporal_state_enabled") is not False:
        raise RuntimeError(f"{condition_root}: depth replay is not stateless")


def analyse_condition(
    condition_root: Path,
    truth: dict[tuple[int, str], tuple[float, float, float]],
    evaluation_frames: list[int],
) -> dict:
    tracking_rows = read_csv(condition_root / "replay" / "tracking_joints.csv")
    landmark_rows = read_csv(condition_root / "landmark_cache" / "landmarks_2d.csv")
    frame_set = set(evaluation_frames)
    points: dict[tuple[int, str], tuple[float, float, float]] = {}
    depths: dict[str, list[float]] = {joint: [] for joint in PRIMARY_JOINTS}
    for row in tracking_rows:
        frame = int(row["frame_index"])
        joint = row["canonical_joint"]
        if frame not in frame_set or joint not in PRIMARY_JOINTS or row["valid"] != "1":
            continue
        point = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
        points[(frame, joint)] = point
        depths[joint].append(float(row["depth_m"]))

    float_pixels: dict[str, list[tuple[float, float]]] = {joint: [] for joint in PRIMARY_JOINTS}
    integer_pixels: dict[str, set[tuple[int, int]]] = {joint: set() for joint in PRIMARY_JOINTS}
    for row in landmark_rows:
        frame = int(row["frame_index"])
        joint = row["canonical_joint"]
        if frame not in frame_set or joint not in PRIMARY_JOINTS or row["eligible"] != "1":
            continue
        float_pixels[joint].append((float(row["pixel_x_float"]), float(row["pixel_y_float"])))
        integer_pixels[joint].add((int(row["pixel_x"]), int(row["pixel_y"])))

    complete_frames = [
        frame
        for frame in evaluation_frames
        if all((frame, joint) in points for joint in PRIMARY_JOINTS)
    ]
    frame_errors = []
    for frame in complete_frames:
        joint_errors = [
            distance_mm(points[(frame, joint)], truth[(frame, joint)])
            for joint in PRIMARY_JOINTS
        ]
        frame_errors.append(statistics.fmean(joint_errors))

    per_joint_3d = {}
    per_joint_2d = {}
    per_joint_depth = {}
    for joint in PRIMARY_JOINTS:
        joint_points = [points[(frame, joint)] for frame in evaluation_frames if (frame, joint) in points]
        per_joint_3d[joint] = vector_jitter(joint_points, 1000.0)
        per_joint_2d[joint] = vector_jitter(float_pixels[joint], 1.0)
        per_joint_depth[joint] = (
            statistics.pstdev(depths[joint]) * 1000.0 if len(depths[joint]) >= 2 else None
        )

    def joint_mean(values: dict[str, float | None]) -> float | None:
        finite = [value for value in values.values() if value is not None]
        return statistics.fmean(finite) if len(finite) == len(PRIMARY_JOINTS) else None

    return {
        "evaluation_frame_count": len(evaluation_frames),
        "right_arm_pair_valid_frame_count": len(complete_frames),
        "right_arm_pair_coverage": len(complete_frames) / len(evaluation_frames),
        "right_arm_two_joint_position_error_mm": summarise(frame_errors),
        "static_3d_jitter_mm": {
            "per_joint": per_joint_3d,
            "right_arm_two_joint_mean": joint_mean(per_joint_3d),
        },
        "static_2d_float_landmark_jitter_px": {
            "per_joint": per_joint_2d,
            "right_arm_two_joint_mean": joint_mean(per_joint_2d),
        },
        "depth_jitter_mm": {
            "per_joint": per_joint_depth,
            "right_arm_two_joint_mean": joint_mean(per_joint_depth),
        },
        "unique_integer_pixel_locations": {
            joint: len(values) for joint, values in integer_pixels.items()
        },
        "provenance": {
            "cache_state_sha256": sha256_file(condition_root / "landmark_cache" / "landmark_cache_state.json"),
            "replay_state_sha256": sha256_file(condition_root / "replay" / "offline_tracker_state.json"),
            "tracking_joints_sha256": sha256_file(condition_root / "replay" / "tracking_joints.csv"),
        },
    }


def make_gate(identifier: str, observed, threshold, passed: bool) -> dict:
    return {"id": identifier, "observed": observed, "threshold": threshold, "passed": bool(passed)}


def main() -> int:
    if SUMMARY.exists():
        raise FileExistsError(f"Refusing to overwrite {SUMMARY}")
    protocol = read_json(PROTOCOL)
    settle = int(protocol["analysis_contract"]["excluded_initial_frames"])
    run_results = []
    for run_id in RUNS:
        source = SOURCE_ROOT / run_id
        manifest = read_csv(source / "rgbd_manifest.csv")
        all_frames = [int(row["sample_index"]) for row in manifest]
        evaluation_frames = all_frames[settle:]
        truth = load_truth(source)
        metrics = {}
        for condition in CONDITIONS:
            condition_root = OUTPUT_ROOT / "runs" / run_id / condition
            validate_condition(condition_root, condition, protocol)
            metrics[condition] = analyse_condition(condition_root, truth, evaluation_frames)
        t0 = metrics["T0_frame_independent"]
        t2 = metrics["T2_tracker_plus_smoothing"]
        jitter0 = t0["static_3d_jitter_mm"]["right_arm_two_joint_mean"]
        jitter2 = t2["static_3d_jitter_mm"]["right_arm_two_joint_mean"]
        run_results.append(
            {
                "run_id": run_id,
                "source_manifest_sha256": sha256_file(source / "rgbd_manifest.csv"),
                "evaluation_frames": [evaluation_frames[0], evaluation_frames[-1]],
                "condition_metrics": metrics,
                "T2_vs_T0": {
                    "static_3d_jitter_ratio": jitter2 / jitter0 if jitter0 else None,
                    "static_3d_jitter_delta_mm": jitter2 - jitter0,
                    "coverage_delta": t2["right_arm_pair_coverage"] - t0["right_arm_pair_coverage"],
                    "mean_position_error_delta_mm": t2["right_arm_two_joint_position_error_mm"]["mean"] - t0["right_arm_two_joint_position_error_mm"]["mean"],
                    "p95_position_error_delta_mm": t2["right_arm_two_joint_position_error_mm"]["p95"] - t0["right_arm_two_joint_position_error_mm"]["p95"],
                },
            }
        )

    thresholds = protocol["replication_contract"]
    contrasts = [run["T2_vs_T0"] for run in run_results]
    jitter_ratios = [item["static_3d_jitter_ratio"] for item in contrasts]
    coverage_deltas = [item["coverage_delta"] for item in contrasts]
    mean_deltas = [item["mean_position_error_delta_mm"] for item in contrasts]
    p95_deltas = [item["p95_position_error_delta_mm"] for item in contrasts]
    frame_counts = [
        run["condition_metrics"]["T0_frame_independent"]["evaluation_frame_count"]
        for run in run_results
    ]
    gates = [
        make_gate("S1_run_count", len(run_results), 3, len(run_results) == 3),
        make_gate("S2_jitter_ratio_each_run", jitter_ratios, thresholds["S2_jitter_ratio_max"], all(value <= thresholds["S2_jitter_ratio_max"] for value in jitter_ratios)),
        make_gate("S3_coverage_delta_each_run", coverage_deltas, thresholds["S3_coverage_delta_min"], all(value >= thresholds["S3_coverage_delta_min"] for value in coverage_deltas)),
        make_gate("S4_mean_error_delta_each_run_mm", mean_deltas, thresholds["S4_mean_error_delta_max_mm"], all(value <= thresholds["S4_mean_error_delta_max_mm"] for value in mean_deltas)),
        make_gate("S5_p95_error_delta_each_run_mm", p95_deltas, thresholds["S5_p95_error_delta_max_mm"], all(value <= thresholds["S5_p95_error_delta_max_mm"] for value in p95_deltas)),
        make_gate("S6_evaluation_frames_each_run", frame_counts, thresholds["S6_min_evaluation_frames"], all(value >= thresholds["S6_min_evaluation_frames"] for value in frame_counts)),
    ]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "classification": "registered_static_generalisation_replay_existing_data",
        "protocol_sha256": sha256_file(PROTOCOL),
        "run_count": len(run_results),
        "runs": run_results,
        "replication_checks": gates,
        "formal_contract_met": all(item["passed"] for item in gates),
        "interpretation_boundary": "640x360 static generalisation evidence; cannot overturn clean_temporal_tracking_v1 or substitute for fresh 960x600 confirmation",
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY}")
    for item in gates:
        print(f"  {item['id']}: {'PASS' if item['passed'] else 'FAIL'} (observed={item['observed']})")
    print("formal_contract_met:", payload["formal_contract_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
