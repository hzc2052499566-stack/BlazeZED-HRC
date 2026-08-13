"""Freeze and verify the input boundary for kinematic-constraint experiments.

The tool does not copy, rename, or modify any source experiment. It records
the selected run inventory, configuration metadata, byte sizes, CSV row
counts, and SHA-256 hashes in one manifest. ``--check`` verifies the current
workspace against a previously written manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "input_freeze_manifest.json"
)

OFFLINE_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "depth_sampling_ablation"
    / "roi_center_s065_mc1_median_7x7"
)
LIVE_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "depth_sampling_live_ab_formal_3p50_fps60_v1"
    / "wrist_aware_v5"
    / "roi_center_s065_mc0"
    / "d_3p50m"
)
CALIBRATION_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "calibration"
    / "offline_visual_full_roi065_median7x7"
)

OFFLINE_REQUIRED_FILES = (
    "tracking_joints.csv",
    "tracking_frames.csv",
    "tracking_timing.csv",
    "offline_tracker_state.json",
    "performance_summary.json",
    "summary_metrics.json",
    "joint_comparison_metrics.csv",
    "limb_length_comparison_metrics.csv",
    "comparison_samples.csv",
)
LIVE_REQUIRED_FILES = (
    "tracking_joints.csv",
    "tracking_frames.csv",
    "live_tracking_timing.csv",
    "tracker_state.json",
    "live_performance_summary.json",
)
SOURCE_GT_FILES = (
    "ground_truth_joints.csv",
    "rgbd_manifest.csv",
)
PROVENANCE_FILES = (
    "configs/joint_mapping.csv",
    "tools/cache_blazepose_landmarks.py",
    "tools/depth_sampling_core.py",
    "tools/replay_depth_sampling.py",
    "tools/compare_static_tracking.py",
    "tools/zed_blazepose_recorder.py",
    "output/experiments/depth_sampling_ablation/"
    "depth_sampling_report_d3p25-3p50-4p00_r01-02-03.json",
    "output/experiments/depth_sampling_live_ab_formal_3p50_fps60_v1/"
    "live_depth_sampling_report.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify files against an existing manifest instead of writing it.",
    )
    return parser.parse_args()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    record = {
        "path": relative_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".csv":
        record["data_row_count"] = csv_data_row_count(path)
    return record


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (WORKSPACE / path).resolve()


def offline_run_record(
    run_dir: Path,
    role: str,
    required_files: tuple[str, ...] = OFFLINE_REQUIRED_FILES,
) -> dict:
    state_path = run_dir / "offline_tracker_state.json"
    state = read_json(state_path)
    source_dir = resolve_workspace_path(state["source_experiment_dir"])
    files = [file_record(run_dir / name) for name in required_files]
    source_files = [
        file_record(source_dir / name) for name in SOURCE_GT_FILES
    ]
    return {
        "role": role,
        "run_dir": relative_path(run_dir),
        "source_experiment_dir": relative_path(source_dir),
        "status": state.get("status"),
        "tracker": state.get("tracker"),
        "depth_sampling_method": state.get("depth_sampling_method"),
        "model_complexity": state.get("model_complexity"),
        "roi_scale": state.get("roi_scale"),
        "warmup_frames": state.get("warmup_frames"),
        "input_frame_count": state.get("input_frame_count"),
        "processed_frame_count": state.get("processed_frame_count"),
        "detected_frame_count": state.get("detected_frame_count"),
        "tracking_row_count": state.get("tracking_row_count"),
        "valid_joint_row_count": state.get("valid_joint_row_count"),
        "files": files,
        "source_ground_truth_files": source_files,
    }


def live_run_record(run_dir: Path) -> dict:
    state = read_json(run_dir / "tracker_state.json")
    summary = read_json(run_dir / "live_performance_summary.json")
    configuration = summary["configuration"]
    return {
        "role": "real_zed_static_stability_replay_only",
        "run_dir": relative_path(run_dir),
        "status": state.get("status"),
        "run_id": state.get("run_id"),
        "pipeline_algorithm_version": state.get(
            "pipeline_algorithm_version"
        ),
        "configuration": {
            "stream_width": configuration.get("stream_width"),
            "stream_height": configuration.get("stream_height"),
            "zed_depth_mode": configuration.get("zed_depth_mode"),
            "zed_depth_stabilization": configuration.get(
                "zed_depth_stabilization"
            ),
            "depth_sampler_mode": configuration.get("depth_sampler_mode"),
            "model_complexity": configuration.get("model_complexity"),
            "roi_scale": configuration.get("roi_scale"),
            "acquisition_mode": configuration.get("acquisition_mode"),
            "warmup_frames": configuration.get("warmup_frames"),
            "filtering_and_kinematic_constraints_enabled": configuration.get(
                "filtering_and_kinematic_constraints_enabled"
            ),
        },
        "recorded_frame_count": summary.get("recorded_frame_count"),
        "steady_frame_count": summary.get("steady_frame_count"),
        "detection_rate": summary.get("detection_rate"),
        "steady_valid_core_rate": summary.get("steady_valid_core_rate"),
        "actual_output_fps": summary.get("actual_output_fps"),
        "files": [
            file_record(run_dir / name) for name in LIVE_REQUIRED_FILES
        ],
    }


def validate_offline_run(record: dict, issues: list[str]) -> None:
    label = record["run_dir"]
    expected = {
        "status": "complete",
        "depth_sampling_method": "median_7x7",
        "model_complexity": 1,
        "roi_scale": 0.65,
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            issues.append(
                "{}: {}={!r}, expected {!r}".format(
                    label,
                    field,
                    record.get(field),
                    expected_value,
                )
            )
    if record.get("processed_frame_count") != record.get("input_frame_count"):
        issues.append(label + ": processed/input frame counts differ")
    if record.get("detected_frame_count") != record.get("input_frame_count"):
        issues.append(label + ": detection was not complete")


def validate_live_run(record: dict, issues: list[str]) -> None:
    label = record["run_dir"]
    configuration = record["configuration"]
    expected = {
        "stream_width": 960,
        "stream_height": 600,
        "zed_depth_mode": "NEURAL_LIGHT",
        "zed_depth_stabilization": 30,
        "depth_sampler_mode": "wrist_aware_v5",
        "model_complexity": 0,
        "roi_scale": 0.65,
        "acquisition_mode": "latest",
        "filtering_and_kinematic_constraints_enabled": False,
    }
    if record.get("status") != "complete":
        issues.append(label + ": run status is not complete")
    for field, expected_value in expected.items():
        if configuration.get(field) != expected_value:
            issues.append(
                "{}: configuration {}={!r}, expected {!r}".format(
                    label,
                    field,
                    configuration.get(field),
                    expected_value,
                )
            )


def build_manifest() -> dict:
    offline_run_dirs = sorted(OFFLINE_ROOT.glob("d_*m/rep_*"))
    calibration_run_dirs = sorted(CALIBRATION_ROOT.glob("d_*m/rep_*"))
    live_run_dirs = sorted(LIVE_ROOT.glob("rep_*"))
    if len(offline_run_dirs) != 9:
        raise RuntimeError(
            "Expected 9 frozen offline test runs, found {}".format(
                len(offline_run_dirs)
            )
        )
    if len(calibration_run_dirs) != 9:
        raise RuntimeError(
            "Expected 9 offline calibration runs, found {}".format(
                len(calibration_run_dirs)
            )
        )
    if len(live_run_dirs) != 4:
        raise RuntimeError(
            "Expected 4 frozen live v5 runs, found {}".format(
                len(live_run_dirs)
            )
        )

    offline_runs = [
        offline_run_record(run_dir, "held_out_static_gt_test")
        for run_dir in offline_run_dirs
    ]
    calibration_runs = [
        offline_run_record(
            run_dir,
            "independent_offline_visual_calibration",
        )
        for run_dir in calibration_run_dirs
    ]
    live_runs = [live_run_record(run_dir) for run_dir in live_run_dirs]

    issues: list[str] = []
    for record in offline_runs:
        validate_offline_run(record, issues)
    for record in calibration_runs:
        validate_offline_run(record, issues)
    for record in live_runs:
        validate_live_run(record, issues)
    if issues:
        raise RuntimeError(
            "Input freeze validation failed:\n- " + "\n- ".join(issues)
        )

    return {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checksum_algorithm": "SHA-256",
        "workspace": str(WORKSPACE.resolve()),
        "freeze_policy": {
            "source_experiments_are_logically_immutable": True,
            "source_files_copied": False,
            "existing_depth_sampling_outputs_must_not_be_overwritten": True,
            "kinematic_outputs_must_use_a_separate_experiment_root": True,
            "ground_truth_must_not_be_used_as_a_runtime_constraint": True,
        },
        "baseline_contract": {
            "offline": {
                "model": "BlazePose Full",
                "model_complexity": 1,
                "roi_scale": 0.65,
                "depth_sampler": "median_7x7",
                "coordinate_system": "RIGHT_HANDED_Z_UP_X_FWD",
            },
            "live": {
                "model": "BlazePose Lite",
                "model_complexity": 0,
                "roi_scale": 0.65,
                "depth_sampler": "wrist_aware_v5",
                "resolution": "960x600",
                "zed_depth_mode": "NEURAL_LIGHT",
                "zed_depth_stabilization": 30,
                "acquisition_mode": "latest",
                "coordinate_system": "RIGHT_HANDED_Z_UP_X_FWD",
            },
        },
        "dataset_usage_rules": {
            "offline_calibration": (
                "Independent d_2p00m/d_2p50m/d_3p00m "
                "Full+ROI0.65+median_7x7 replay. "
                "May estimate the visual-calibrated bone-length profile."
            ),
            "offline_test": (
                "Nine d_3p25/d_3p50/d_4p00 paired GT runs. Keep held out "
                "until kinematic parameters are frozen."
            ),
            "live_static": (
                "Four real-ZED v5 static runs. May evaluate stability and "
                "validity only; no synchronized per-frame GT, so they cannot "
                "support MPJPE or dynamic-accuracy claims."
            ),
            "live_calibration": (
                "Missing. A separate Lite+ROI0.65+wrist_aware_v5 capture is "
                "required before live kinematic Formal."
            ),
        },
        "inventory": {
            "offline_calibration": {
                "run_count": len(calibration_runs),
                "frame_count": sum(
                    int(run["input_frame_count"]) for run in calibration_runs
                ),
                "runs": calibration_runs,
            },
            "offline_held_out_test": {
                "run_count": len(offline_runs),
                "frame_count": sum(
                    int(run["input_frame_count"]) for run in offline_runs
                ),
                "runs": offline_runs,
            },
            "live_static_stability": {
                "run_count": len(live_runs),
                "frame_count": sum(
                    int(run["recorded_frame_count"]) for run in live_runs
                ),
                "steady_frame_count": sum(
                    int(run["steady_frame_count"]) for run in live_runs
                ),
                "runs": live_runs,
            },
        },
        "provenance_files": [
            file_record(WORKSPACE / path) for path in PROVENANCE_FILES
        ],
        "validation": {
            "status": "passed",
            "issues": [],
        },
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def check_manifest(path: Path) -> list[str]:
    manifest = read_json(path)
    mismatches = []
    records = []
    for group in manifest["inventory"].values():
        for run in group["runs"]:
            records.extend(run.get("files", []))
            records.extend(run.get("source_ground_truth_files", []))
    records.extend(manifest.get("provenance_files", []))
    seen = set()
    for record in records:
        relative = record["path"]
        if relative in seen:
            continue
        seen.add(relative)
        path_on_disk = WORKSPACE / relative
        if not path_on_disk.is_file():
            mismatches.append(relative + ": missing")
            continue
        actual_bytes = path_on_disk.stat().st_size
        if actual_bytes != int(record["bytes"]):
            mismatches.append(
                "{}: bytes {} != {}".format(
                    relative,
                    actual_bytes,
                    record["bytes"],
                )
            )
            continue
        actual_hash = sha256_file(path_on_disk)
        if actual_hash != record["sha256"]:
            mismatches.append(relative + ": SHA-256 mismatch")
    return mismatches


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            raise FileNotFoundError(output)
        mismatches = check_manifest(output)
        if mismatches:
            print("Input freeze check FAILED")
            for mismatch in mismatches:
                print("-", mismatch)
            return 1
        print("Input freeze check PASSED:", output)
        return 0

    manifest = build_manifest()
    write_manifest(output, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(output),
                "offline_calibration_runs": manifest["inventory"][
                    "offline_calibration"
                ]["run_count"],
                "offline_calibration_frames": manifest["inventory"][
                    "offline_calibration"
                ]["frame_count"],
                "offline_test_runs": manifest["inventory"][
                    "offline_held_out_test"
                ]["run_count"],
                "offline_test_frames": manifest["inventory"][
                    "offline_held_out_test"
                ]["frame_count"],
                "live_runs": manifest["inventory"][
                    "live_static_stability"
                ]["run_count"],
                "live_frames": manifest["inventory"][
                    "live_static_stability"
                ]["frame_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
