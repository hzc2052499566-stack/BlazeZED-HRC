"""Estimate a person-specific visual bone-length profile.

Calibration uses only frozen BlazePose + RGB-D tracking coordinates. Isaac
ground truth files may be present beside the source experiments, but this tool
never reads them and never uses them to estimate runtime reference lengths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import freeze_kinematic_inputs


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "input_freeze_manifest.json"
)
DEFAULT_MODEL = WORKSPACE / "configs" / "kinematic_model.json"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "configs"
    / "subject_profiles"
    / "female_police_offline_visual.json"
)
DEFAULT_REPORT_DIR = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "calibration"
    / "profile_report"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--inventory-key",
        default="offline_calibration",
        help="Manifest inventory group containing calibration runs.",
    )
    parser.add_argument(
        "--pipeline-contract-key",
        default="offline",
        help="Manifest baseline_contract key recorded in the profile.",
    )
    parser.add_argument(
        "--profile-name",
        default="female_police_offline_visual",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status if any bone fails its quality gates.",
    )
    return parser.parse_args()


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


def weighted_median(values, weights) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if values_array.ndim != 1 or weights_array.shape != values_array.shape:
        raise ValueError("values and weights must be one-dimensional peers")
    if len(values_array) == 0:
        raise ValueError("weighted_median requires at least one value")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("values must be finite")
    if not np.all(np.isfinite(weights_array)) or np.any(weights_array < 0):
        raise ValueError("weights must be finite and non-negative")
    total_weight = float(np.sum(weights_array))
    if total_weight <= 0.0:
        raise ValueError("at least one weight must be positive")
    order = np.argsort(values_array, kind="stable")
    ordered_values = values_array[order]
    ordered_weights = weights_array[order]
    threshold = total_weight * 0.5
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def robust_location(
    values,
    base_weights,
    huber_delta: float,
    scale_floor_m: float,
    maximum_iterations: int,
    tolerance_m: float,
) -> dict:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(base_weights, dtype=np.float64)
    centre = weighted_median(values_array, weights_array)
    absolute_deviation = np.abs(values_array - centre)
    raw_mad = weighted_median(absolute_deviation, weights_array)
    scale_m = max(1.4826 * raw_mad, float(scale_floor_m))
    final_weights = weights_array.copy()
    iteration_count = 0
    for iteration_count in range(1, int(maximum_iterations) + 1):
        residual = np.abs(values_array - centre)
        threshold = float(huber_delta) * scale_m
        huber_weights = np.ones_like(residual)
        outside = residual > threshold
        huber_weights[outside] = threshold / residual[outside]
        final_weights = weights_array * huber_weights
        updated = float(
            np.sum(final_weights * values_array) / np.sum(final_weights)
        )
        if abs(updated - centre) <= float(tolerance_m):
            centre = updated
            break
        centre = updated
    return {
        "location_m": centre,
        "raw_mad_m": raw_mad,
        "robust_scale_m": scale_m,
        "iterations": iteration_count,
        "final_weights": final_weights,
    }


def coefficient_of_variation(values) -> float | None:
    values_array = np.asarray(values, dtype=np.float64)
    if len(values_array) < 2:
        return None
    mean_value = float(np.mean(values_array))
    if mean_value == 0.0:
        return None
    return float(np.std(values_array, ddof=1) / mean_value)


def depth_sampling_method_allowed(method: str, calibration: dict) -> bool:
    allowed = calibration.get("allowed_depth_sampling_methods")
    if allowed is not None:
        return method in set(allowed)
    return method == calibration["required_depth_sampling_method"]


def stratified_bootstrap_ci(
    values_by_run: dict[str, list[float]],
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(int(seed))
    run_arrays = [
        np.asarray(values, dtype=np.float64)
        for _, values in sorted(values_by_run.items())
        if values
    ]
    if not run_arrays:
        raise ValueError("bootstrap requires at least one populated run")
    estimates = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        parts = [
            run_values[
                generator.integers(
                    0,
                    len(run_values),
                    size=len(run_values),
                )
            ]
            for run_values in run_arrays
        ]
        estimates[index] = float(np.median(np.concatenate(parts)))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def load_calibration_samples(
    manifest: dict,
    model: dict,
    inventory_key: str = "offline_calibration",
) -> tuple[dict[str, list[dict]], list[dict]]:
    calibration = model["calibration"]
    minimum_visibility = float(calibration["minimum_visibility"])
    bones = model["bones"]
    samples = {bone["name"]: [] for bone in bones}
    run_summaries = []

    for run in manifest["inventory"][inventory_key]["runs"]:
        run_dir = WORKSPACE / run["run_dir"]
        warmup_frames = (
            int(run["warmup_frames"])
            if calibration["exclude_warmup_frames"]
            else 0
        )
        frames: dict[int, dict[str, dict]] = defaultdict(dict)
        tracking_path = run_dir / "tracking_joints.csv"
        with tracking_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                frame_index = int(row["frame_index"])
                if frame_index < warmup_frames:
                    continue
                if str(row["valid"]).strip() != "1":
                    continue
                confidence = float(row["confidence"]) / 100.0
                if confidence < minimum_visibility:
                    continue
                if not depth_sampling_method_allowed(
                    row.get("depth_sampling_method", ""),
                    calibration,
                ):
                    continue
                point = np.asarray(
                    [
                        float(row["x_m"]),
                        float(row["y_m"]),
                        float(row["z_m"]),
                    ],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(point)):
                    continue
                frames[frame_index][row["canonical_joint"]] = {
                    "point": point,
                    "confidence": confidence,
                }

        run_id = run["run_dir"]
        per_bone_counts = {}
        for bone in bones:
            bone_samples = samples[bone["name"]]
            count_before = len(bone_samples)
            for frame_index, joints in frames.items():
                joint_a = joints.get(bone["joint_a"])
                joint_b = joints.get(bone["joint_b"])
                if joint_a is None or joint_b is None:
                    continue
                length_m = float(
                    np.linalg.norm(joint_a["point"] - joint_b["point"])
                )
                if not math.isfinite(length_m) or length_m <= 0.0:
                    continue
                bone_samples.append(
                    {
                        "run_id": run_id,
                        "frame_index": frame_index,
                        "length_m": length_m,
                        "weight": min(
                            joint_a["confidence"],
                            joint_b["confidence"],
                        ),
                    }
                )
            per_bone_counts[bone["name"]] = len(bone_samples) - count_before
        run_summaries.append(
            {
                "run_dir": run_id,
                "steady_frame_count": len(frames),
                "bone_sample_counts": per_bone_counts,
            }
        )
    return samples, run_summaries


def estimate_bone(
    bone: dict,
    samples: list[dict],
    model: dict,
    bone_index: int,
) -> dict:
    calibration = model["calibration"]
    gates = model["quality_gates"]
    values = np.asarray(
        [sample["length_m"] for sample in samples],
        dtype=np.float64,
    )
    weights = np.asarray(
        [sample["weight"] for sample in samples],
        dtype=np.float64,
    )
    robust = robust_location(
        values,
        weights,
        calibration["huber_delta"],
        calibration["scale_floor_m"],
        calibration["maximum_iterations"],
        calibration["convergence_tolerance_m"],
    )
    reference_m = float(robust["location_m"])
    inlier_limit_m = (
        float(calibration["inlier_sigma_multiplier"])
        * float(robust["robust_scale_m"])
    )
    inlier_mask = np.abs(values - reference_m) <= inlier_limit_m
    accepted_values = values[inlier_mask]
    accepted_samples = [
        sample
        for sample, is_inlier in zip(samples, inlier_mask)
        if bool(is_inlier)
    ]
    values_by_run: dict[str, list[float]] = defaultdict(list)
    for sample in accepted_samples:
        values_by_run[sample["run_id"]].append(sample["length_m"])
    run_medians = {
        run_id: float(np.median(run_values))
        for run_id, run_values in sorted(values_by_run.items())
    }
    ci_lower_m, ci_upper_m = stratified_bootstrap_ci(
        values_by_run,
        calibration["bootstrap_resamples"],
        int(calibration["bootstrap_seed"]) + int(bone_index),
    )
    accepted_cv = coefficient_of_variation(accepted_values)
    run_median_cv = coefficient_of_variation(list(run_medians.values()))
    ci_width_m = ci_upper_m - ci_lower_m
    allowed_ci_width_m = max(
        float(gates["maximum_ci95_width_m"]),
        float(gates["maximum_ci95_width_fraction"]) * reference_m,
    )
    checks = {
        "sample_count": (
            len(accepted_values)
            >= int(gates["minimum_accepted_sample_count"])
        ),
        "run_count": (
            len(run_medians) >= int(gates["minimum_run_count"])
        ),
        "accepted_frame_cv": (
            accepted_cv is not None
            and accepted_cv <= float(gates["maximum_accepted_frame_cv"])
        ),
        "run_median_cv": (
            run_median_cv is not None
            and run_median_cv <= float(gates["maximum_run_median_cv"])
        ),
        "ci95_width": ci_width_m <= allowed_ci_width_m,
    }
    passed = all(checks.values())
    return {
        "name": bone["name"],
        "side": bone["side"],
        "group": bone["group"],
        "joint_a": bone["joint_a"],
        "joint_b": bone["joint_b"],
        "reference_length_m": reference_m,
        "constraint_enabled": passed,
        "quality_status": "passed" if passed else "failed",
        "sample_count": len(values),
        "accepted_sample_count": len(accepted_values),
        "outlier_sample_count": len(values) - len(accepted_values),
        "run_count": len(run_medians),
        "raw_mean_m": float(np.mean(values)),
        "raw_std_m": float(np.std(values, ddof=1)),
        "raw_cv": coefficient_of_variation(values),
        "raw_median_m": float(np.median(values)),
        "raw_mad_m": float(robust["raw_mad_m"]),
        "robust_scale_m": float(robust["robust_scale_m"]),
        "inlier_limit_m": inlier_limit_m,
        "accepted_mean_m": float(np.mean(accepted_values)),
        "accepted_std_m": float(np.std(accepted_values, ddof=1)),
        "accepted_cv": accepted_cv,
        "run_median_cv": run_median_cv,
        "ci95_lower_m": ci_lower_m,
        "ci95_upper_m": ci_upper_m,
        "ci95_width_m": ci_width_m,
        "allowed_ci95_width_m": allowed_ci_width_m,
        "run_medians_m": run_medians,
        "quality_checks": checks,
        "estimator_iterations": robust["iterations"],
    }


def symmetry_diagnostics(
    bones: list[dict],
    warning_threshold: float,
) -> list[dict]:
    by_group_side = {
        (bone["group"], bone["side"]): bone for bone in bones
    }
    diagnostics = []
    for group in ("upper_arm", "forearm", "thigh", "shank"):
        left = by_group_side[(group, "left")]
        right = by_group_side[(group, "right")]
        mean_length = (
            float(left["reference_length_m"])
            + float(right["reference_length_m"])
        ) / 2.0
        relative_difference = (
            abs(
                float(left["reference_length_m"])
                - float(right["reference_length_m"])
            )
            / mean_length
        )
        diagnostics.append(
            {
                "group": group,
                "left_bone": left["name"],
                "right_bone": right["name"],
                "relative_difference": relative_difference,
                "warning_threshold": float(warning_threshold),
                "warning": relative_difference > float(warning_threshold),
                "hard_equality_applied": False,
            }
        )
    return diagnostics


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    report_dir = args.report_dir.resolve()

    mismatches = freeze_kinematic_inputs.check_manifest(manifest_path)
    if mismatches:
        raise RuntimeError(
            "Frozen input verification failed:\n- "
            + "\n- ".join(mismatches)
        )
    manifest = read_json(manifest_path)
    model = read_json(model_path)
    samples_by_bone, run_summaries = load_calibration_samples(
        manifest,
        model,
        args.inventory_key,
    )
    bone_profiles = [
        estimate_bone(
            bone,
            samples_by_bone[bone["name"]],
            model,
            index,
        )
        for index, bone in enumerate(model["bones"])
    ]
    failed_bones = [
        bone["name"]
        for bone in bone_profiles
        if bone["quality_status"] != "passed"
    ]
    profile = {
        "schema_version": 1,
        "profile_name": args.profile_name,
        "subject": {
            "type": "Isaac Sim character",
            "prim_path": "/World/female_adult_police_03_new",
        },
        "profile_type": "visual_calibrated",
        "runtime_ground_truth_used": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": model["coordinate_system"],
        "unit": model["unit"],
        "kinematic_model_name": model["model_name"],
        "kinematic_model_path": model_path.relative_to(WORKSPACE).as_posix(),
        "kinematic_model_sha256": sha256_file(model_path),
        "input_manifest_path": manifest_path.relative_to(WORKSPACE).as_posix(),
        "input_manifest_sha256": sha256_file(manifest_path),
        "pipeline_contract": manifest["baseline_contract"][
            args.pipeline_contract_key
        ],
        "calibration_method": model["calibration"],
        "quality_gates": model["quality_gates"],
        "overall_quality_status": (
            "passed" if not failed_bones else "partial_pass"
        ),
        "passed_bone_count": len(bone_profiles) - len(failed_bones),
        "failed_bone_count": len(failed_bones),
        "failed_bones": failed_bones,
        "bones": bone_profiles,
        "symmetry_diagnostics": symmetry_diagnostics(
            bone_profiles,
            model["symmetry"]["warning_relative_difference"],
        ),
        "calibration_runs": run_summaries,
        "use_restrictions": (
            [
                "Valid only for Full + ROI 0.65 + median_7x7.",
                "Do not use as a Lite + wrist_aware_v5 live profile.",
                "Do not replace visual lengths with Isaac GT lengths.",
                "Keep failed bones disabled until independent recalibration passes.",
                "Do not update the profile on held-out test sequences.",
            ]
            if args.pipeline_contract_key == "offline"
            else [
                "Valid only for Lite + ROI 0.65 + wrist_aware_v5.",
                "Do not use as a Full + median_7x7 offline profile.",
                "Do not replace visual lengths with Isaac GT lengths.",
                "Keep failed bones disabled until independent recalibration passes.",
                "Arm references currently depend mainly on accepted frame-12 samples.",
                "Do not update the profile on evaluation sequences.",
            ]
        ),
    }
    write_json_atomic(output_path, profile)

    report_rows = []
    for bone in bone_profiles:
        report_rows.append(
            {
                "bone": bone["name"],
                "status": bone["quality_status"],
                "constraint_enabled": int(bone["constraint_enabled"]),
                "reference_length_mm": 1000.0
                * bone["reference_length_m"],
                "sample_count": bone["sample_count"],
                "accepted_sample_count": bone["accepted_sample_count"],
                "run_count": bone["run_count"],
                "raw_cv_percent": 100.0 * bone["raw_cv"],
                "accepted_cv_percent": 100.0 * bone["accepted_cv"],
                "run_median_cv_percent": 100.0
                * bone["run_median_cv"],
                "ci95_lower_mm": 1000.0 * bone["ci95_lower_m"],
                "ci95_upper_mm": 1000.0 * bone["ci95_upper_m"],
                "ci95_width_mm": 1000.0 * bone["ci95_width_m"],
                "failed_checks": ",".join(
                    name
                    for name, passed in bone["quality_checks"].items()
                    if not passed
                ),
            }
        )
    write_csv(
        report_dir / "kinematic_profile_bones.csv",
        report_rows,
        list(report_rows[0]),
    )
    run_median_rows = []
    for bone in bone_profiles:
        for run_id, run_median_m in bone["run_medians_m"].items():
            run_median_rows.append(
                {
                    "bone": bone["name"],
                    "run_dir": run_id,
                    "run_median_length_mm": 1000.0 * run_median_m,
                    "profile_reference_length_mm": (
                        1000.0 * bone["reference_length_m"]
                    ),
                    "difference_from_profile_mm": (
                        1000.0
                        * (run_median_m - bone["reference_length_m"])
                    ),
                    "bone_quality_status": bone["quality_status"],
                }
            )
    write_csv(
        report_dir / "kinematic_profile_run_medians.csv",
        run_median_rows,
        list(run_median_rows[0]),
    )
    write_json_atomic(
        report_dir / "kinematic_profile_report.json",
        {
            "profile_path": output_path.relative_to(WORKSPACE).as_posix(),
            "overall_quality_status": profile["overall_quality_status"],
            "passed_bone_count": profile["passed_bone_count"],
            "failed_bone_count": profile["failed_bone_count"],
            "failed_bones": failed_bones,
            "bones": bone_profiles,
            "symmetry_diagnostics": profile["symmetry_diagnostics"],
        },
    )
    print(
        json.dumps(
            {
                "profile": str(output_path),
                "report_dir": str(report_dir),
                "overall_quality_status": profile["overall_quality_status"],
                "passed_bone_count": profile["passed_bone_count"],
                "failed_bones": failed_bones,
            },
            indent=2,
        )
    )
    if args.strict and failed_bones:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
