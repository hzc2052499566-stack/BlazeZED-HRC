"""Analyse the frozen clean temporal tracking three-condition replay.

The experiment isolates MediaPipe's video tracker and landmark smoothing while
holding RGB, rendered metric depth, intrinsics, depth sampling and GT fixed.
Dynamic motion is evaluated with a GT-referenced frame-step residual; the word
``jitter`` is reserved for the two registered stationary holds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "clean_temporal_tracking_v1"
)
SOURCE_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "dynamic_k4_gt_confirmatory_v1"
    / "unit_corrected"
)
PROTOCOL_PATH = OUTPUT_ROOT / "protocol_lock_v1.json"
SUMMARY_PATH = OUTPUT_ROOT / "formal_summary.json"

RUNS = ("rep_01", "rep_02", "rep_03")
SOURCE_RUNS = {
    "rep_01": "confirmatory_rep_01",
    "rep_02": "confirmatory_rep_02",
    "rep_03": "confirmatory_rep_03",
}
CONDITIONS = (
    "T0_frame_independent",
    "T1_tracker_only",
    "T2_tracker_plus_smoothing",
)
PRIMARY_JOINTS = ("right_elbow", "right_wrist")
FRAME_COUNT = 240

# Conservative phase windows leave keyframe boundaries out of phase-specific
# position summaries. Dynamic edges include the transition into motion.
HOLD_LOW = tuple(range(0, 30))
HOLD_HIGH = tuple(range(121, 150))
DYNAMIC_EDGE_ENDS = tuple(range(31, 120)) + tuple(range(151, 240))
DYNAMIC_SEGMENTS = ((31, 119), (151, 239))
LAG_SEARCH = tuple(range(-6, 7))


class TemporalTrackingError(RuntimeError):
    """Raised when a frozen temporal-tracking contract is violated."""


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
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: dict, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise TemporalTrackingError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarise(values: list[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": percentile(finite, 0.95),
        "max": max(finite),
    }


def run_summary(values: list[float]) -> dict:
    result = summarise(values)
    result["sample_sd"] = (
        statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
    )
    result["sd_over_mean"] = (
        result["sample_sd"] / result["mean"]
        if result["mean"] not in (None, 0.0)
        else None
    )
    result["values"] = list(values)
    return result


def point(row: dict[str, str]) -> tuple[float, float, float]:
    return float(row["x_m"]), float(row["y_m"]), float(row["z_m"])


def gt_point(row: dict[str, str]) -> tuple[float, float, float]:
    return float(row["gt_x_m"]), float(row["gt_y_m"]), float(row["gt_z_m"])


def distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def subtract(
    current: tuple[float, float, float], previous: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(current, previous))


def vector_mean(
    values: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    return tuple(statistics.fmean(value[axis] for value in values) for axis in range(3))


def static_jitter_mm(
    points: dict[tuple[int, str], tuple[float, float, float]],
    frames: tuple[int, ...],
) -> dict:
    per_joint: dict[str, float | None] = {}
    for joint in PRIMARY_JOINTS:
        values = [points[(frame, joint)] for frame in frames if (frame, joint) in points]
        if len(values) != len(frames):
            per_joint[joint] = None
            continue
        axis_std = [statistics.pstdev(value[axis] for value in values) for axis in range(3)]
        per_joint[joint] = math.sqrt(sum(value**2 for value in axis_std)) * 1000.0
    valid = [value for value in per_joint.values() if value is not None]
    return {
        "frame_count": len(frames),
        "complete_joint_count": len(valid),
        "per_joint_mm": per_joint,
        "right_arm_two_joint_mean_mm": (
            statistics.fmean(valid) if len(valid) == len(PRIMARY_JOINTS) else None
        ),
    }


def right_arm_frame_errors_mm(
    points: dict[tuple[int, str], tuple[float, float, float]],
    truth: dict[tuple[int, str], tuple[float, float, float]],
    frames: list[int],
) -> list[float]:
    errors: list[float] = []
    for frame in frames:
        joint_errors = []
        for joint in PRIMARY_JOINTS:
            key = (frame, joint)
            if key not in points or key not in truth:
                break
            joint_errors.append(distance(points[key], truth[key]) * 1000.0)
        if len(joint_errors) == len(PRIMARY_JOINTS):
            errors.append(statistics.fmean(joint_errors))
    return errors


def temporal_step_residuals_mm(
    points: dict[tuple[int, str], tuple[float, float, float]],
    truth: dict[tuple[int, str], tuple[float, float, float]],
    edge_ends: list[int],
) -> list[float]:
    residuals: list[float] = []
    for current in edge_ends:
        previous = current - 1
        joint_residuals = []
        for joint in PRIMARY_JOINTS:
            current_key = (current, joint)
            previous_key = (previous, joint)
            if (
                current_key not in points
                or previous_key not in points
                or current_key not in truth
                or previous_key not in truth
            ):
                break
            estimate_step = subtract(points[current_key], points[previous_key])
            truth_step = subtract(truth[current_key], truth[previous_key])
            joint_residuals.append(distance(estimate_step, truth_step) * 1000.0)
        if len(joint_residuals) == len(PRIMARY_JOINTS):
            residuals.append(statistics.fmean(joint_residuals))
    return residuals


def best_alignment_lag(
    points: dict[tuple[int, str], tuple[float, float, float]],
    truth: dict[tuple[int, str], tuple[float, float, float]],
) -> dict:
    candidates = []
    for lag in LAG_SEARCH:
        squared_errors: list[float] = []
        comparison_count = 0
        for start, end in DYNAMIC_SEGMENTS:
            for joint in PRIMARY_JOINTS:
                pairs = []
                for frame in range(start, end + 1):
                    truth_frame = frame - lag
                    if not start <= truth_frame <= end:
                        continue
                    estimate_key = (frame, joint)
                    truth_key = (truth_frame, joint)
                    if estimate_key in points and truth_key in truth:
                        pairs.append((points[estimate_key], truth[truth_key]))
                if len(pairs) < 3:
                    continue
                estimate_mean = vector_mean([pair[0] for pair in pairs])
                truth_mean = vector_mean([pair[1] for pair in pairs])
                for estimate, expected in pairs:
                    estimate_centered = subtract(estimate, estimate_mean)
                    truth_centered = subtract(expected, truth_mean)
                    squared_errors.append(distance(estimate_centered, truth_centered) ** 2)
                comparison_count += len(pairs)
        if squared_errors:
            candidates.append(
                {
                    "lag_frames": lag,
                    "lag_ms": lag * 1000.0 / 60.0,
                    "centred_rmse_mm": math.sqrt(statistics.fmean(squared_errors))
                    * 1000.0,
                    "comparison_count": comparison_count,
                }
            )
    if not candidates:
        return {"lag_frames": None, "lag_ms": None, "candidates": []}
    best = min(
        candidates,
        key=lambda value: (
            value["centred_rmse_mm"],
            abs(value["lag_frames"]),
            value["lag_frames"],
        ),
    )
    return {**best, "positive_lag_means_estimate_lags_gt": True, "candidates": candidates}


def load_truth(source_dir: Path) -> dict[tuple[int, str], tuple[float, float, float]]:
    rows = read_csv(source_dir / "ground_truth_joints.csv")
    if len(rows) != 3600:
        raise TemporalTrackingError(f"Expected 3600 GT rows in {source_dir}, got {len(rows)}")
    result = {}
    for row in rows:
        joint = row["canonical_joint"]
        if joint in PRIMARY_JOINTS:
            result[(int(row["sequence_index"]), joint)] = gt_point(row)
    if len(result) != FRAME_COUNT * len(PRIMARY_JOINTS):
        raise TemporalTrackingError(f"Incomplete primary-joint GT in {source_dir}")
    return result


def load_condition(run_dir: Path, condition: str) -> tuple[dict, dict, dict, dict]:
    condition_dir = run_dir / condition
    cache_dir = condition_dir / "landmark_cache"
    replay_dir = condition_dir / "replay"
    cache_state = read_json(cache_dir / "landmark_cache_state.json")
    replay_state = read_json(replay_dir / "offline_tracker_state.json")
    performance = read_json(replay_dir / "performance_summary.json")
    if cache_state.get("status") != "complete" or replay_state.get("status") != "complete":
        raise TemporalTrackingError(f"Incomplete condition {condition_dir}")
    expected = {
        "T0_frame_independent": (True, False),
        "T1_tracker_only": (False, False),
        "T2_tracker_plus_smoothing": (False, True),
    }[condition]
    observed = (
        bool(cache_state.get("static_image_mode")),
        bool(cache_state.get("smooth_landmarks")),
    )
    if observed != expected:
        raise TemporalTrackingError(
            f"{condition} temporal flags are {observed}, expected {expected}"
        )
    if cache_state.get("model_complexity") != 0 or cache_state.get("roi_scale") != 0.65:
        raise TemporalTrackingError(f"{condition} changed BlazePose configuration")
    if replay_state.get("depth_sampling_method") != "median_7x7":
        raise TemporalTrackingError(f"{condition} is not stateless median_7x7")
    variant = performance.get("processing_variant", {})
    if variant.get("temporal_state_enabled") is not False:
        raise TemporalTrackingError(f"{condition} depth replay enabled temporal state")

    points = {}
    for row in read_csv(replay_dir / "tracking_joints.csv"):
        if row["canonical_joint"] not in PRIMARY_JOINTS or row["valid"] != "1":
            continue
        points[(int(row["frame_index"]), row["canonical_joint"])] = point(row)
    cache_frames = read_csv(cache_dir / "landmark_frames.csv")
    replay_frames = read_csv(replay_dir / "tracking_timing.csv")
    if len(cache_frames) != FRAME_COUNT or len(replay_frames) != FRAME_COUNT:
        raise TemporalTrackingError(f"{condition} does not contain 240 timing rows")
    timing = {
        "pose_inference_all_ms": summarise(
            [float(row["pose_inference_ms"]) for row in cache_frames]
        ),
        "pose_inference_after_frame0_ms": summarise(
            [float(row["pose_inference_ms"]) for row in cache_frames[1:]]
        ),
        "stateless_depth_replay_ms": summarise(
            [float(row["total_compute_ms"]) for row in replay_frames]
        ),
        "combined_component_compute_after_frame0_ms": summarise(
            [
                float(cache_frames[index]["total_compute_ms"])
                + float(replay_frames[index]["total_compute_ms"])
                for index in range(1, FRAME_COUNT)
            ]
        ),
        "scope": "offline component compute; not camera-to-output latency or actual FPS",
    }
    provenance = {
        "cache_state_sha256": sha256_file(cache_dir / "landmark_cache_state.json"),
        "landmarks_sha256": cache_state["landmarks_sha256"],
        "replay_state_sha256": sha256_file(replay_dir / "offline_tracker_state.json"),
        "tracking_joints_sha256": sha256_file(replay_dir / "tracking_joints.csv"),
    }
    return points, timing, provenance, cache_state


def analyse_run(run_id: str) -> dict:
    source_dir = SOURCE_ROOT / SOURCE_RUNS[run_id]
    manifest = read_csv(source_dir / "rgbd_manifest.csv")
    if len(manifest) != FRAME_COUNT:
        raise TemporalTrackingError(f"{run_id} source manifest is not 240 frames")
    sequences = [int(row["sequence_index"]) for row in manifest]
    if sequences != list(range(FRAME_COUNT)):
        raise TemporalTrackingError(f"{run_id} sequence_index is not exactly 0..239")
    truth = load_truth(source_dir)
    run_dir = OUTPUT_ROOT / "runs" / run_id
    loaded = {condition: load_condition(run_dir, condition) for condition in CONDITIONS}
    points_by_condition = {condition: loaded[condition][0] for condition in CONDITIONS}

    common_frames = [
        frame
        for frame in range(FRAME_COUNT)
        if all(
            (frame, joint) in points_by_condition[condition]
            for condition in CONDITIONS
            for joint in PRIMARY_JOINTS
        )
    ]
    common_edges = [
        frame
        for frame in DYNAMIC_EDGE_ENDS
        if all(
            (index, joint) in points_by_condition[condition]
            for condition in CONDITIONS
            for joint in PRIMARY_JOINTS
            for index in (frame - 1, frame)
        )
    ]

    condition_metrics = {}
    for condition in CONDITIONS:
        points = points_by_condition[condition]
        method_frames = [
            frame
            for frame in range(FRAME_COUNT)
            if all((frame, joint) in points for joint in PRIMARY_JOINTS)
        ]
        method_errors = right_arm_frame_errors_mm(points, truth, method_frames)
        common_errors = right_arm_frame_errors_mm(points, truth, common_frames)
        step_residuals = temporal_step_residuals_mm(points, truth, common_edges)
        condition_metrics[condition] = {
            "right_arm_pair_coverage": len(method_frames) / FRAME_COUNT,
            "right_arm_pair_valid_frames": len(method_frames),
            "position_error_method_specific_mm": summarise(method_errors),
            "position_error_common_valid_mm": summarise(common_errors),
            "dynamic_temporal_step_residual_common_edges_mm": summarise(
                step_residuals
            ),
            "hold_low_static_jitter": static_jitter_mm(points, HOLD_LOW),
            "hold_high_static_jitter": static_jitter_mm(points, HOLD_HIGH),
            "best_alignment_lag": best_alignment_lag(points, truth),
            "timing": loaded[condition][1],
            "provenance": loaded[condition][2],
        }

    baseline = condition_metrics["T0_frame_independent"]
    deployed = condition_metrics["T2_tracker_plus_smoothing"]
    baseline_step = baseline["dynamic_temporal_step_residual_common_edges_mm"]["p95"]
    deployed_step = deployed["dynamic_temporal_step_residual_common_edges_mm"]["p95"]
    baseline_jitter = baseline["hold_high_static_jitter"][
        "right_arm_two_joint_mean_mm"
    ]
    deployed_jitter = deployed["hold_high_static_jitter"][
        "right_arm_two_joint_mean_mm"
    ]
    baseline_position = baseline["position_error_common_valid_mm"]
    deployed_position = deployed["position_error_common_valid_mm"]
    contrast = {
        "dynamic_p95_step_residual_ratio_T2_over_T0": (
            deployed_step / baseline_step if baseline_step not in (None, 0.0) else None
        ),
        "steady_hold_jitter_ratio_T2_over_T0": (
            deployed_jitter / baseline_jitter
            if baseline_jitter not in (None, 0.0)
            else None
        ),
        "mean_position_error_delta_T2_minus_T0_mm": (
            deployed_position["mean"] - baseline_position["mean"]
            if deployed_position["mean"] is not None
            and baseline_position["mean"] is not None
            else None
        ),
        "p95_position_error_delta_T2_minus_T0_mm": (
            deployed_position["p95"] - baseline_position["p95"]
            if deployed_position["p95"] is not None
            and baseline_position["p95"] is not None
            else None
        ),
        "coverage_delta_T2_minus_T0": (
            deployed["right_arm_pair_coverage"]
            - baseline["right_arm_pair_coverage"]
        ),
        "T2_alignment_lag_frames": deployed["best_alignment_lag"]["lag_frames"],
        "T2_alignment_lag_ms": deployed["best_alignment_lag"]["lag_ms"],
    }
    return {
        "run_id": run_id,
        "source_run_id": SOURCE_RUNS[run_id],
        "source_manifest_sha256": sha256_file(source_dir / "rgbd_manifest.csv"),
        "ground_truth_sha256": sha256_file(source_dir / "ground_truth_joints.csv"),
        "common_valid_frame_count": len(common_frames),
        "common_valid_frame_fraction": len(common_frames) / FRAME_COUNT,
        "common_dynamic_edge_count": len(common_edges),
        "common_dynamic_edge_fraction": len(common_edges) / len(DYNAMIC_EDGE_ENDS),
        "condition_metrics": condition_metrics,
        "primary_contrast_T2_vs_T0": contrast,
        "scope_note": (
            "One paired supplementary replay run. Frames are repeated observations; "
            "the independently initiated run is the experimental unit."
        ),
    }


def gate(identifier: str, observed, threshold, passed: bool) -> dict:
    return {
        "id": identifier,
        "observed": observed,
        "threshold": threshold,
        "passed": bool(passed),
    }


def aggregate(runs: list[dict], protocol: dict) -> dict:
    contrasts = [run["primary_contrast_T2_vs_T0"] for run in runs]
    thresholds = protocol["replication_contract"]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in contrasts if row.get(key) is not None]

    dynamic_ratios = values("dynamic_p95_step_residual_ratio_T2_over_T0")
    jitter_ratios = values("steady_hold_jitter_ratio_T2_over_T0")
    mean_deltas = values("mean_position_error_delta_T2_minus_T0_mm")
    p95_deltas = values("p95_position_error_delta_T2_minus_T0_mm")
    coverage_deltas = values("coverage_delta_T2_minus_T0")
    lags = values("T2_alignment_lag_frames")
    t0_position = [
        float(
            run["condition_metrics"]["T0_frame_independent"]
            ["position_error_common_valid_mm"]["mean"]
        )
        for run in runs
    ]
    t0_repeatability = run_summary(t0_position)

    checks = [
        gate("R1_run_count", len(runs), thresholds["R1_run_count"], len(runs) == 3),
        gate(
            "R2_dynamic_p95_ratio_each_run",
            dynamic_ratios,
            thresholds["R2_dynamic_p95_ratio_max"],
            len(dynamic_ratios) == 3
            and all(value <= thresholds["R2_dynamic_p95_ratio_max"] for value in dynamic_ratios),
        ),
        gate(
            "R3_steady_hold_jitter_ratio_each_run",
            jitter_ratios,
            thresholds["R3_steady_hold_jitter_ratio_max"],
            len(jitter_ratios) == 3
            and all(
                value <= thresholds["R3_steady_hold_jitter_ratio_max"]
                for value in jitter_ratios
            ),
        ),
        gate(
            "R4_mean_position_delta_each_run_mm",
            mean_deltas,
            thresholds["R4_mean_position_delta_max_mm"],
            len(mean_deltas) == 3
            and all(value <= thresholds["R4_mean_position_delta_max_mm"] for value in mean_deltas),
        ),
        gate(
            "R5_p95_position_delta_each_run_mm",
            p95_deltas,
            thresholds["R5_p95_position_delta_max_mm"],
            len(p95_deltas) == 3
            and all(value <= thresholds["R5_p95_position_delta_max_mm"] for value in p95_deltas),
        ),
        gate(
            "R6_coverage_loss_each_run",
            coverage_deltas,
            thresholds["R6_coverage_delta_min"],
            len(coverage_deltas) == 3
            and all(value >= thresholds["R6_coverage_delta_min"] for value in coverage_deltas),
        ),
        gate(
            "R7_alignment_lag_each_run_frames",
            lags,
            thresholds["R7_absolute_lag_max_frames"],
            len(lags) == 3
            and all(abs(value) <= thresholds["R7_absolute_lag_max_frames"] for value in lags),
        ),
        gate(
            "R8_T0_position_error_sd_over_mean",
            t0_repeatability["sd_over_mean"],
            thresholds["R8_T0_sd_over_mean_max"],
            t0_repeatability["sd_over_mean"] is not None
            and t0_repeatability["sd_over_mean"]
            <= thresholds["R8_T0_sd_over_mean_max"],
        ),
    ]
    return {
        "schema_version": 1,
        "purpose": "clean_temporal_tracking_three_condition_summary_v1",
        "classification": "registered_paired_supplementary_replay_existing_data",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "run_count": len(runs),
        "runs": runs,
        "run_level_aggregate": {
            "dynamic_p95_step_residual_ratio_T2_over_T0": run_summary(dynamic_ratios),
            "steady_hold_jitter_ratio_T2_over_T0": run_summary(jitter_ratios),
            "mean_position_error_delta_T2_minus_T0_mm": run_summary(mean_deltas),
            "p95_position_error_delta_T2_minus_T0_mm": run_summary(p95_deltas),
            "coverage_delta_T2_minus_T0": run_summary(coverage_deltas),
            "T2_alignment_lag_frames": run_summary(lags),
            "T0_position_error_mm": t0_repeatability,
        },
        "replication_checks": checks,
        "formal_contract_met": all(check["passed"] for check in checks),
        "reporting_rules": [
            "The primary endpoint is right-arm elbow/wrist temporal step residual, not MPJPE.",
            "Jitter is reported only in the registered stationary holds.",
            "Run is the experimental unit; 720 frames are not 720 independent replicates.",
            "Coverage, method-specific error and common-valid paired error are reported together.",
            "Offline timing is component compute, not camera-to-output latency or actual FPS.",
            "A smoother trajectory that fails position-error or lag gates is not a clean temporal benefit.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not PROTOCOL_PATH.is_file():
        raise TemporalTrackingError(f"Missing frozen protocol: {PROTOCOL_PATH}")
    protocol = read_json(PROTOCOL_PATH)
    runs = [analyse_run(run_id) for run_id in RUNS]
    for run in runs:
        write_json_new(
            OUTPUT_ROOT / "runs" / run["run_id"] / "per_run_summary.json",
            run,
            overwrite=args.overwrite,
        )
    summary = aggregate(runs, protocol)
    write_json_new(SUMMARY_PATH, summary, overwrite=args.overwrite)
    print(f"wrote {SUMMARY_PATH}")
    for check in summary["replication_checks"]:
        print(
            "  {}: {} (observed={!r})".format(
                check["id"], "PASS" if check["passed"] else "FAIL", check["observed"]
            )
        )
    print("formal_contract_met:", summary["formal_contract_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
