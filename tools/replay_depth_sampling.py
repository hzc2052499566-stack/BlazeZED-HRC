"""Replay one depth-sampling method on cached, identical 2D landmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from depth_sampling_core import (
    MIN_VISIBILITY,
    SUPPORTED_METHODS,
    sample_candidates,
)


TRACKING_FIELDS = [
    "run_id",
    "tracker",
    "frame_index",
    "wall_time_ns",
    "zed_timestamp_ns",
    "body_id",
    "tracking_state",
    "canonical_joint",
    "core_evaluation",
    "x_m",
    "y_m",
    "z_m",
    "confidence",
    "valid",
    "pixel_x",
    "pixel_y",
    "pixel_x_float",
    "pixel_y_float",
    "in_image",
    "eligible",
    "depth_m",
    "depth_window_radius_px",
    "depth_window_size_px",
    "depth_valid_pixel_count",
    "depth_sampling_method",
    "depth_fallback_distance_px",
    "depth_cluster_count",
    "depth_selected_cluster_size",
    "depth_directional_constraint_used",
    "depth_kinematic_reference_m",
    "invalid_reason",
    "source_detail",
]

FRAME_FIELDS = [
    "run_id",
    "frame_index",
    "wall_time_ns",
    "zed_timestamp_ns",
    "grab_status",
    "body_count",
    "selected_body_id",
    "zed_joint_count",
    "blazepose_joint_count",
    "processing_time_ms",
    "body_bbox_width_px",
    "body_bbox_height_px",
    "rgb_load_ms",
    "depth_load_ms",
    "preprocess_ms",
    "pose_inference_ms",
    "landmark_decode_ms",
    "depth_sampling_ms",
    "backprojection_ms",
    "result_assembly_ms",
    "total_compute_ms",
    "unaccounted_ms",
    "warmup_excluded",
]

TIMING_FIELDS = [
    "run_id",
    "frame_index",
    "wall_time_ns",
    "body_detected",
    "valid_joint_count",
    "depth_load_ms",
    "depth_sampling_ms",
    "backprojection_ms",
    "result_assembly_ms",
    "total_compute_ms",
    "warmup_excluded",
]

TIMING_PHASES = [
    "depth_load_ms",
    "depth_sampling_ms",
    "backprojection_ms",
    "result_assembly_ms",
    "total_compute_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--landmark-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--depth-sampling-method",
        choices=SUPPORTED_METHODS,
        required=True,
    )
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def statistics_for(values: list[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
        "max": max(finite),
    }


def reconstruct(
    depth_m: float,
    pixel_x: int,
    pixel_y: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    return (
        float(depth_m),
        -(float(pixel_x) - cx) * float(depth_m) / fx,
        -(float(pixel_y) - cy) * float(depth_m) / fy,
    )


def bool_field(value: str | int | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).strip() in {"1", "true", "True"}


def main() -> int:
    args = parse_args()
    source_dir = args.experiment_dir.resolve()
    cache_dir = args.landmark_cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    tracker_name = str(args.tracker_name).strip()
    if not tracker_name:
        raise ValueError("--tracker-name must not be empty.")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must not be negative.")

    manifest_path = source_dir / "rgbd_manifest.csv"
    depth_dir = source_dir / "rgbd_frames"
    landmarks_path = cache_dir / "landmarks_2d.csv"
    landmark_frames_path = cache_dir / "landmark_frames.csv"
    cache_state_path = cache_dir / "landmark_cache_state.json"
    required = [
        manifest_path,
        depth_dir,
        landmarks_path,
        landmark_frames_path,
        cache_state_path,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing required input(s):")
        for path in missing:
            print(" -", path)
        return 2

    cache_state = json.loads(cache_state_path.read_text(encoding="utf-8"))
    if cache_state.get("status") != "complete":
        print("Landmark cache is not complete:", cache_dir)
        return 2
    if cache_state.get("source_manifest_sha256") != file_sha256(manifest_path):
        print("Source manifest hash differs from the landmark cache.")
        return 6
    if cache_state.get("landmarks_sha256") != file_sha256(landmarks_path):
        print("Cached landmark CSV hash verification failed.")
        return 6

    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = output_dir / "tracking_joints.csv"
    frames_path = output_dir / "tracking_frames.csv"
    timing_path = output_dir / "tracking_timing.csv"
    performance_path = output_dir / "performance_summary.json"
    state_path = output_dir / "offline_tracker_state.json"
    config = {
        "source_experiment_dir": str(source_dir),
        "landmark_cache_dir": str(cache_dir),
        "landmarks_sha256": cache_state["landmarks_sha256"],
        "output_dir": str(output_dir),
        "tracker": tracker_name,
        "depth_sampling_method": args.depth_sampling_method,
        "model_complexity": cache_state.get("model_complexity"),
        "roi_scale": cache_state.get("roi_scale"),
        "min_visibility": cache_state.get("min_visibility"),
        "warmup_frames": int(args.warmup_frames),
    }
    if state_path.exists() and not args.overwrite:
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("status") == "complete":
            conflicts = {
                key: {"existing": existing.get(key), "requested": value}
                for key, value in config.items()
                if existing.get(key) != value
            }
            if conflicts:
                print("Completed replay has configuration conflicts:")
                print(json.dumps(conflicts, indent=2))
                return 6
            print("SKIP completed depth replay:", output_dir)
            return 0
        print(
            "Incomplete replay output exists. Inspect it or pass --overwrite:",
            output_dir,
        )
        return 4

    state_path.write_text(
        json.dumps({"status": "running", **config}, indent=2),
        encoding="utf-8",
    )
    try:
        manifest_rows = read_csv(manifest_path)
        landmark_rows = read_csv(landmarks_path)
        landmark_frame_rows = read_csv(landmark_frames_path)
        landmark_by_frame: dict[int, list[dict]] = defaultdict(list)
        for row in landmark_rows:
            landmark_by_frame[int(row["frame_index"])].append(row)
        frame_cache = {
            int(row["frame_index"]): row for row in landmark_frame_rows
        }
        if len(frame_cache) != len(manifest_rows):
            raise RuntimeError(
                "Landmark frame count {} differs from manifest count {}.".format(
                    len(frame_cache),
                    len(manifest_rows),
                )
            )

        tracking_rows: list[dict] = []
        frame_rows: list[dict] = []
        timing_rows: list[dict] = []
        sampling_method_counts: dict[str, int] = defaultdict(int)
        invalid_reason_counts: dict[str, int] = defaultdict(int)

        for sequence_index, manifest in enumerate(manifest_rows):
            frame_start_ns = time.perf_counter_ns()
            frame_index = int(manifest["sample_index"])
            run_id = manifest["run_id"]
            cache_frame = frame_cache[frame_index]
            body_detected = bool_field(cache_frame["body_detected"])

            phase_start_ns = time.perf_counter_ns()
            raw_depth = np.load(depth_dir / manifest["depth_file"])
            depth_m = np.asarray(raw_depth, dtype=np.float64) * float(
                manifest.get("depth_scale_to_m", "1.0")
            )
            depth_load_ms = elapsed_ms(phase_start_ns)
            if depth_m.shape[:2] != (
                int(manifest["height"]),
                int(manifest["width"]),
            ):
                raise RuntimeError(
                    "Depth shape mismatch in frame {}: {}".format(
                        frame_index,
                        depth_m.shape,
                    )
                )

            candidates = []
            for cached in landmark_by_frame.get(frame_index, []):
                candidates.append(
                    {
                        **cached,
                        "pixel_x": int(cached["pixel_x"]),
                        "pixel_y": int(cached["pixel_y"]),
                        "pixel_x_float": float(cached["pixel_x_float"]),
                        "pixel_y_float": float(cached["pixel_y_float"]),
                        "visibility": float(cached["visibility"]),
                        "in_image": bool_field(cached["in_image"]),
                        "eligible": bool_field(cached["eligible"]),
                    }
                )

            phase_start_ns = time.perf_counter_ns()
            depth_results = sample_candidates(
                depth_m,
                candidates,
                args.depth_sampling_method,
            )
            depth_sampling_ms = elapsed_ms(phase_start_ns)

            fx = float(manifest["fx"])
            fy = float(manifest["fy"])
            cx = float(manifest["cx"])
            cy = float(manifest["cy"])
            phase_start_ns = time.perf_counter_ns()
            points = []
            valid_flags = []
            invalid_reasons = []
            for candidate, depth_result in zip(candidates, depth_results):
                sampling_valid = bool(depth_result["valid"])
                eligible = bool(candidate["eligible"])
                is_valid = sampling_valid and eligible
                if not candidate["in_image"]:
                    invalid_reason = "landmark_out_of_image"
                elif candidate["visibility"] < MIN_VISIBILITY:
                    invalid_reason = "low_visibility"
                elif not sampling_valid:
                    invalid_reason = str(depth_result["invalid_reason"])
                else:
                    invalid_reason = ""
                points.append(
                    reconstruct(
                        float(depth_result["depth_m"]),
                        int(candidate["pixel_x"]),
                        int(candidate["pixel_y"]),
                        fx,
                        fy,
                        cx,
                        cy,
                    )
                    if is_valid
                    else (None, None, None)
                )
                valid_flags.append(is_valid)
                invalid_reasons.append(invalid_reason)
            backprojection_ms = elapsed_ms(phase_start_ns)

            phase_start_ns = time.perf_counter_ns()
            for candidate, depth_result, point, is_valid, invalid_reason in zip(
                candidates,
                depth_results,
                points,
                valid_flags,
                invalid_reasons,
            ):
                method_used = str(depth_result["sampling_method"])
                sampling_method_counts[method_used] += 1
                if invalid_reason:
                    invalid_reason_counts[invalid_reason] += 1
                tracking_rows.append(
                    {
                        "run_id": run_id,
                        "tracker": tracker_name,
                        "frame_index": frame_index,
                        "wall_time_ns": manifest["wall_time_ns"],
                        "zed_timestamp_ns": "",
                        "body_id": 0,
                        "tracking_state": "DETECTED",
                        "canonical_joint": candidate["canonical_joint"],
                        "core_evaluation": candidate["core_evaluation"],
                        "x_m": point[0] if is_valid else "",
                        "y_m": point[1] if is_valid else "",
                        "z_m": point[2] if is_valid else "",
                        "confidence": candidate["visibility"] * 100.0,
                        "valid": int(is_valid),
                        "pixel_x": candidate["pixel_x"],
                        "pixel_y": candidate["pixel_y"],
                        "pixel_x_float": candidate["pixel_x_float"],
                        "pixel_y_float": candidate["pixel_y_float"],
                        "in_image": int(candidate["in_image"]),
                        "eligible": int(candidate["eligible"]),
                        "depth_m": (
                            depth_result["depth_m"]
                            if depth_result["depth_m"] is not None
                            else ""
                        ),
                        "depth_window_radius_px": (
                            depth_result["window_radius_px"]
                            if depth_result["window_radius_px"] is not None
                            else ""
                        ),
                        "depth_window_size_px": (
                            depth_result["window_size_px"]
                            if depth_result["window_size_px"] is not None
                            else ""
                        ),
                        "depth_valid_pixel_count": depth_result[
                            "valid_pixel_count"
                        ],
                        "depth_sampling_method": method_used,
                        "depth_fallback_distance_px": (
                            depth_result["fallback_distance_px"]
                            if depth_result["fallback_distance_px"] is not None
                            else ""
                        ),
                        "depth_cluster_count": depth_result["cluster_count"],
                        "depth_selected_cluster_size": depth_result[
                            "selected_cluster_size"
                        ],
                        "depth_directional_constraint_used": int(
                            depth_result["directional_constraint_used"]
                        ),
                        "depth_kinematic_reference_m": (
                            depth_result["kinematic_reference_m"]
                            if depth_result["kinematic_reference_m"] is not None
                            else ""
                        ),
                        "invalid_reason": invalid_reason,
                        "source_detail": (
                            "Cached BlazePose pixel; paired depth replay; "
                            "method={}".format(method_used)
                        ),
                    }
                )
            result_assembly_ms = elapsed_ms(phase_start_ns)
            total_compute_ms = elapsed_ms(frame_start_ns)
            accounted_ms = (
                depth_load_ms
                + depth_sampling_ms
                + backprojection_ms
                + result_assembly_ms
            )
            warmup_excluded = int(sequence_index < args.warmup_frames)
            valid_joint_count = sum(int(value) for value in valid_flags)
            frame_row = {
                "run_id": run_id,
                "frame_index": frame_index,
                "wall_time_ns": manifest["wall_time_ns"],
                "zed_timestamp_ns": "",
                "grab_status": "SUCCESS",
                "body_count": int(body_detected),
                "selected_body_id": 0 if body_detected else "",
                "zed_joint_count": 0,
                "blazepose_joint_count": valid_joint_count,
                "processing_time_ms": total_compute_ms,
                "body_bbox_width_px": cache_frame["body_bbox_width_px"],
                "body_bbox_height_px": cache_frame["body_bbox_height_px"],
                "rgb_load_ms": 0.0,
                "depth_load_ms": depth_load_ms,
                "preprocess_ms": 0.0,
                "pose_inference_ms": 0.0,
                "landmark_decode_ms": 0.0,
                "depth_sampling_ms": depth_sampling_ms,
                "backprojection_ms": backprojection_ms,
                "result_assembly_ms": result_assembly_ms,
                "total_compute_ms": total_compute_ms,
                "unaccounted_ms": max(0.0, total_compute_ms - accounted_ms),
                "warmup_excluded": warmup_excluded,
            }
            frame_rows.append(frame_row)
            timing_rows.append(
                {
                    "run_id": run_id,
                    "frame_index": frame_index,
                    "wall_time_ns": manifest["wall_time_ns"],
                    "body_detected": int(body_detected),
                    "valid_joint_count": valid_joint_count,
                    "depth_load_ms": depth_load_ms,
                    "depth_sampling_ms": depth_sampling_ms,
                    "backprojection_ms": backprojection_ms,
                    "result_assembly_ms": result_assembly_ms,
                    "total_compute_ms": total_compute_ms,
                    "warmup_excluded": warmup_excluded,
                }
            )

        write_start_ns = time.perf_counter_ns()
        write_csv(tracking_path, tracking_rows, TRACKING_FIELDS)
        write_csv(frames_path, frame_rows, FRAME_FIELDS)
        write_csv(timing_path, timing_rows, TIMING_FIELDS)
        output_write_ms = elapsed_ms(write_start_ns)

        steady_rows = [
            row for row in frame_rows if int(row["warmup_excluded"]) == 0
        ]
        all_statistics = {
            phase: statistics_for(
                [float(row[phase]) for row in frame_rows]
            )
            for phase in TIMING_PHASES
        }
        steady_statistics = {
            phase: statistics_for(
                [float(row[phase]) for row in steady_rows]
            )
            for phase in TIMING_PHASES
        }
        steady_total_ms = sum(
            float(row["total_compute_ms"]) for row in steady_rows
        )
        performance = {
            "run_id": manifest_rows[0]["run_id"] if manifest_rows else "",
            "tracker": tracker_name,
            "processing_variant": {
                **config,
                "landmark_coordinate_space": "cached_full_frame_pixels",
                "depth_sampling_space": "original_full_frame_metric_depth",
                "camera_intrinsics_space": "original_full_frame",
                "filtering_enabled": False,
                "temporal_state_enabled": False,
            },
            "input_frame_count": len(manifest_rows),
            "processed_frame_count": len(frame_rows),
            "detected_frame_count": sum(
                int(row["body_count"]) for row in frame_rows
            ),
            "detection_rate": (
                sum(int(row["body_count"]) for row in frame_rows)
                / len(frame_rows)
                if frame_rows
                else None
            ),
            "warmup_frames_excluded": min(
                args.warmup_frames,
                len(frame_rows),
            ),
            "output_csv_write_ms": output_write_ms,
            "all_frames": all_statistics,
            "steady_state": steady_statistics,
            "steady_state_processing_throughput_fps": (
                len(steady_rows) * 1000.0 / steady_total_ms
                if steady_total_ms > 0.0
                else None
            ),
            "sampling_method_counts": dict(sampling_method_counts),
            "invalid_reason_counts": dict(invalid_reason_counts),
            "timing_scope": (
                "Paired offline replay using cached 2D landmarks. "
                "total_compute_ms includes depth NPY loading, depth sampling, "
                "back-projection and result assembly only."
            ),
            "latency_limit": (
                "These are replay and sampler timings, not live "
                "camera-to-output latency."
            ),
        }
        performance_path.write_text(
            json.dumps(performance, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        completed = {
            "status": "complete",
            **config,
            "input_frame_count": len(manifest_rows),
            "processed_frame_count": len(frame_rows),
            "detected_frame_count": performance["detected_frame_count"],
            "tracking_row_count": len(tracking_rows),
            "valid_joint_row_count": sum(
                int(row["valid"]) for row in tracking_rows
            ),
            "tracking_csv": str(tracking_path),
            "timing_csv": str(timing_path),
            "performance_summary_json": str(performance_path),
        }
        state_path.write_text(
            json.dumps(completed, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(completed, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        state_path.write_text(
            json.dumps(
                {"status": "failed", **config, "error": repr(error)},
                indent=2,
            ),
            encoding="utf-8",
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
