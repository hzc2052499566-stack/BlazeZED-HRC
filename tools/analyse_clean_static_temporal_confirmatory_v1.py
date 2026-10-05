"""Analyse the fresh 960x600 clean-static temporal confirmatory runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyse_clean_static_temporal_followup_v1 as common


EXP_ROOT = ROOT / "output" / "experiments" / "kinematic_constraints_ablation" / "clean_static_temporal_confirmatory_v1"
SOURCE_ROOT = EXP_ROOT / "raw"
OUTPUT_ROOT = EXP_ROOT / "derived"
PROTOCOL = EXP_ROOT / "analysis_protocol_lock_v1.json"
SUMMARY = EXP_ROOT / "formal_summary.json"
RUNS = ("rep_02", "rep_03", "rep_04")
CONDITIONS = (
    "T0_frame_independent",
    "T1_tracker_only",
    "T2_tracker_plus_smoothing",
)
PRIMARY_JOINTS = common.PRIMARY_JOINTS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


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
        "cache warmup": cache.get("warmup_frames") == 0,
        "depth_sampling": replay.get("depth_sampling_method") == "median_7x7",
        "replay warmup": replay.get("warmup_frames") == 0,
    }
    performance = read_json(condition_root / "replay" / "performance_summary.json")
    checks["stateless depth"] = performance.get("processing_variant", {}).get("temporal_state_enabled") is False
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{condition_root}: invalid condition {failed}")


def load_tracking_points(condition_root: Path, frames: set[int]) -> dict[tuple[int, str], tuple[float, float, float]]:
    points = {}
    for row in common.read_csv(condition_root / "replay" / "tracking_joints.csv"):
        frame = int(row["frame_index"])
        joint = row["canonical_joint"]
        if frame in frames and joint in PRIMARY_JOINTS and row["valid"] == "1":
            point = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
            if all(math.isfinite(value) for value in point):
                points[(frame, joint)] = point
    return points


def paired_error_summary(
    first: dict[tuple[int, str], tuple[float, float, float]],
    second: dict[tuple[int, str], tuple[float, float, float]],
    truth: dict[tuple[int, str], tuple[float, float, float]],
    frames: list[int],
) -> dict:
    common_frames = [
        frame for frame in frames
        if all((frame, joint) in first and (frame, joint) in second for joint in PRIMARY_JOINTS)
    ]
    first_errors = []
    second_errors = []
    for frame in common_frames:
        first_errors.append(statistics.fmean(
            common.distance_mm(first[(frame, joint)], truth[(frame, joint)]) for joint in PRIMARY_JOINTS
        ))
        second_errors.append(statistics.fmean(
            common.distance_mm(second[(frame, joint)], truth[(frame, joint)]) for joint in PRIMARY_JOINTS
        ))
    return {
        "common_frame_count": len(common_frames),
        "common_frame_fraction": len(common_frames) / len(frames),
        "T0_mm": common.summarise(first_errors),
        "T2_mm": common.summarise(second_errors),
    }


def max_gt_axis_range(source: Path) -> float:
    rows = common.read_csv(source / "ground_truth_joints.csv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["canonical_joint"], []).append(row)
    maximum = 0.0
    for group in groups.values():
        for axis in ("gt_x_m", "gt_y_m", "gt_z_m"):
            values = [float(row[axis]) for row in group]
            maximum = max(maximum, max(values) - min(values))
    return maximum


def gate(identifier: str, observed, threshold, passed: bool) -> dict:
    return {"id": identifier, "observed": observed, "threshold": threshold, "passed": bool(passed)}


def aggregate(values: list[float]) -> dict:
    return {
        **common.summarise(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> int:
    if SUMMARY.exists():
        raise FileExistsError(f"Refusing to overwrite {SUMMARY}")
    protocol = read_json(PROTOCOL)
    input_manifest = EXP_ROOT / "analysis_input_manifest.json"
    if sha256_file(input_manifest) != protocol["input_manifest_sha256"]:
        raise RuntimeError("Analysis input manifest hash mismatch.")
    settle = int(protocol["analysis_contract"]["excluded_initial_frames"])
    run_results = []
    for run_id in RUNS:
        source = SOURCE_ROOT / run_id
        manifest = common.read_csv(source / "rgbd_manifest.csv")
        all_frames = [int(row["sample_index"]) for row in manifest]
        evaluation_frames = all_frames[settle:]
        truth = common.load_truth(source)
        condition_metrics = {}
        tracking = {}
        for condition in CONDITIONS:
            condition_root = OUTPUT_ROOT / "runs" / run_id / condition
            validate_condition(condition_root, condition, protocol)
            condition_metrics[condition] = common.analyse_condition(
                condition_root, truth, evaluation_frames
            )
            tracking[condition] = load_tracking_points(condition_root, set(evaluation_frames))
        t0 = condition_metrics["T0_frame_independent"]
        t2 = condition_metrics["T2_tracker_plus_smoothing"]
        jitter0 = t0["static_3d_jitter_mm"]["right_arm_two_joint_mean"]
        jitter2 = t2["static_3d_jitter_mm"]["right_arm_two_joint_mean"]
        paired = paired_error_summary(
            tracking["T0_frame_independent"],
            tracking["T2_tracker_plus_smoothing"],
            truth,
            evaluation_frames,
        )
        contrast = {
            "static_3d_jitter_ratio_T2_over_T0": jitter2 / jitter0 if jitter0 else None,
            "static_3d_jitter_delta_T2_minus_T0_mm": jitter2 - jitter0,
            "coverage_delta_T2_minus_T0": t2["right_arm_pair_coverage"] - t0["right_arm_pair_coverage"],
            "common_valid_frame_fraction": paired["common_frame_fraction"],
            "mean_position_error_delta_T2_minus_T0_mm": paired["T2_mm"]["mean"] - paired["T0_mm"]["mean"],
            "p95_position_error_delta_T2_minus_T0_mm": paired["T2_mm"]["p95"] - paired["T0_mm"]["p95"],
        }
        result = {
            "run_id": run_id,
            "raw_frame_count": len(all_frames),
            "evaluation_frame_count": len(evaluation_frames),
            "evaluation_frame_range": [evaluation_frames[0], evaluation_frames[-1]],
            "max_gt_axis_range_m": max_gt_axis_range(source),
            "condition_metrics": condition_metrics,
            "paired_position_error_common_valid": paired,
            "T2_vs_T0": contrast,
            "source_provenance": {
                "manifest_sha256": sha256_file(source / "rgbd_manifest.csv"),
                "ground_truth_sha256": sha256_file(source / "ground_truth_joints.csv"),
            },
        }
        write_exclusive(OUTPUT_ROOT / "runs" / run_id / "per_run_summary.json", result)
        run_results.append(result)

    contrasts = [run["T2_vs_T0"] for run in run_results]
    jitter_ratios = [item["static_3d_jitter_ratio_T2_over_T0"] for item in contrasts]
    coverage_deltas = [item["coverage_delta_T2_minus_T0"] for item in contrasts]
    mean_deltas = [item["mean_position_error_delta_T2_minus_T0_mm"] for item in contrasts]
    p95_deltas = [item["p95_position_error_delta_T2_minus_T0_mm"] for item in contrasts]
    raw_counts = [run["raw_frame_count"] for run in run_results]
    eval_counts = [run["evaluation_frame_count"] for run in run_results]
    gt_ranges = [run["max_gt_axis_range_m"] for run in run_results]
    thresholds = protocol["replication_contract"]
    checks = [
        gate("C1_run_count", len(run_results), thresholds["C1_run_count"], len(run_results) == thresholds["C1_run_count"]),
        gate("C2_raw_frames_each_run", raw_counts, thresholds["C2_min_raw_frames"], all(value >= thresholds["C2_min_raw_frames"] for value in raw_counts)),
        gate("C3_evaluation_frames_each_run", eval_counts, thresholds["C3_min_evaluation_frames"], all(value >= thresholds["C3_min_evaluation_frames"] for value in eval_counts)),
        gate("C4_jitter_ratio_each_run", jitter_ratios, thresholds["C4_jitter_ratio_max"], len(jitter_ratios) == 3 and all(value <= thresholds["C4_jitter_ratio_max"] for value in jitter_ratios)),
        gate("C5_coverage_delta_each_run", coverage_deltas, thresholds["C5_coverage_delta_min"], all(value >= thresholds["C5_coverage_delta_min"] for value in coverage_deltas)),
        gate("C6_mean_error_delta_each_run_mm", mean_deltas, thresholds["C6_mean_error_delta_max_mm"], all(value <= thresholds["C6_mean_error_delta_max_mm"] for value in mean_deltas)),
        gate("C7_p95_error_delta_each_run_mm", p95_deltas, thresholds["C7_p95_error_delta_max_mm"], all(value <= thresholds["C7_p95_error_delta_max_mm"] for value in p95_deltas)),
        gate("C8_GT_stationary_each_run", gt_ranges, thresholds["C8_max_gt_axis_range_m"], all(value <= thresholds["C8_max_gt_axis_range_m"] for value in gt_ranges)),
    ]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "classification": "fresh_three_run_960x600_clean_static_confirmatory",
        "run_set": list(RUNS),
        "validity_excluded_runs": {"rep_01": "89 raw / 79 evaluation frames; below frozen 90 / 80 minima"},
        "capture_protocol_v1_sha256": protocol["capture_protocol_v1_sha256"],
        "replacement_protocol_v2_sha256": protocol["replacement_protocol_v2_sha256"],
        "analysis_protocol_sha256": sha256_file(PROTOCOL),
        "input_manifest_sha256": sha256_file(input_manifest),
        "runs": run_results,
        "run_level_aggregate": {
            "jitter_ratio_T2_over_T0": aggregate(jitter_ratios),
            "coverage_delta_T2_minus_T0": aggregate(coverage_deltas),
            "mean_position_error_delta_T2_minus_T0_mm": aggregate(mean_deltas),
            "p95_position_error_delta_T2_minus_T0_mm": aggregate(p95_deltas),
        },
        "replication_checks": checks,
        "formal_contract_met": all(item["passed"] for item in checks),
        "reporting_boundaries": [
            "The primary endpoint contains only right elbow and right wrist and is not full-body MPJPE.",
            "Jitter is reported only for a registered static pose after the 10-frame settling interval.",
            "Run is the experimental unit; frames are not independent replicates.",
            "This new confirmatory version does not rewrite clean_temporal_tracking_v1 formal_contract_met=false.",
            "BODY_38 is not evaluated."
        ],
    }
    write_exclusive(SUMMARY, payload)
    print(f"wrote {SUMMARY}")
    for item in checks:
        print(f"  {item['id']}: {'PASS' if item['passed'] else 'FAIL'} (observed={item['observed']})")
    print("formal_contract_met:", payload["formal_contract_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
