"""Replay K0/K1/K2 kinematic constraints on one frozen tracking run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import kinematic_constraints


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    WORKSPACE
    / "configs"
    / "subject_profiles"
    / "female_police_offline_visual.json"
)
DEFAULT_CONFIG = WORKSPACE / "configs" / "kinematic_constraints.json"

ADDED_TRACKING_FIELDS = [
    "raw_x_m",
    "raw_y_m",
    "raw_z_m",
    "raw_valid",
    "kinematic_variant",
    "kinematic_adjusted",
    "kinematic_rejected",
    "kinematic_correction_mm",
]

DIAGNOSTIC_FIELDS = [
    "run_id",
    "frame_index",
    "kinematic_variant",
    "bone",
    "joint_a",
    "joint_b",
    "constraint_enabled",
    "raw_observed_length_m",
    "final_observed_length_m",
    "reference_length_m",
    "raw_length_error_m",
    "final_length_error_m",
    "tolerance_m",
    "raw_gate_violation",
    "final_gate_violation",
]

FRAME_FIELDS = [
    "run_id",
    "frame_index",
    "kinematic_variant",
    "input_valid_joint_count",
    "output_valid_joint_count",
    "adjusted_joint_count",
    "rejected_joint_count",
    "raw_gate_violation_count",
    "final_gate_violation_count",
    "maximum_joint_correction_mm",
    "processing_time_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=kinematic_constraints.VARIANTS, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def tracker_name(input_name: str, variant: str) -> str:
    return "{}_{}".format(input_name, variant)


def writes_projected_coordinates(variant: str) -> bool:
    return variant in ("k2_soft_projection", "k2_guarded_projection")


def frame_input(rows: list[dict[str, str]]) -> tuple[dict, dict, dict]:
    points = {}
    confidence = {}
    valid = {}
    for row in rows:
        joint = row["canonical_joint"]
        is_valid = row.get("valid") == "1"
        x = finite_float(row.get("x_m", ""))
        y = finite_float(row.get("y_m", ""))
        z = finite_float(row.get("z_m", ""))
        valid[joint] = bool(
            is_valid and x is not None and y is not None and z is not None
        )
        if valid[joint]:
            points[joint] = np.asarray([x, y, z], dtype=np.float64)
        confidence[joint] = finite_float(row.get("confidence", "")) or 0.0
    return points, confidence, valid


def format_optional(value) -> str:
    return "" if value is None else repr(float(value))


def process_run(
    input_dir: Path,
    output_dir: Path,
    variant: str,
    profile_path: Path,
    config_path: Path,
    overwrite: bool = False,
) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    profile_path = profile_path.resolve()
    config_path = config_path.resolve()
    input_tracking_path = input_dir / "tracking_joints.csv"
    input_frames_path = input_dir / "tracking_frames.csv"
    required = [
        input_tracking_path,
        input_frames_path,
        profile_path,
        config_path,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(map(str, missing)))

    profile = kinematic_constraints.load_json(profile_path)
    config = kinematic_constraints.load_json(config_path)
    kinematic_constraints.validate_profile(profile)
    kinematic_constraints.validate_config(config)

    requested = {
        "variant": variant,
        "input_tracking_sha256": sha256_file(input_tracking_path),
        "profile_sha256": sha256_file(profile_path),
        "config_sha256": sha256_file(config_path),
        "constraint_module_sha256": sha256_file(
            Path(kinematic_constraints.__file__).resolve()
        ),
        "replay_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    state_path = output_dir / "kinematic_replay_state.json"
    if state_path.is_file() and not overwrite:
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            conflicts = {
                key: {"existing": existing.get(key), "requested": value}
                for key, value in requested.items()
                if existing.get(key) != value
            }
            if conflicts:
                raise RuntimeError(
                    "Completed output has configuration conflicts: "
                    + json.dumps(conflicts, indent=2)
                )
            print("SKIP completed kinematic replay:", output_dir)
            return existing
        raise RuntimeError("Incomplete output exists: " + str(output_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        state_path,
        {
            "status": "running",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            **requested,
        },
    )

    rows = read_csv(input_tracking_path)
    if not rows:
        raise RuntimeError("Input tracking CSV contains no data rows.")
    original_fields = list(rows[0])
    output_fields = original_fields + [
        field for field in ADDED_TRACKING_FIELDS if field not in original_fields
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_index"])].append(row)

    output_rows = []
    diagnostic_rows = []
    frame_rows = []
    processing_times = []
    total_adjusted = 0
    total_rejected = 0
    raw_violation_count = 0
    final_violation_count = 0
    input_tracker = rows[0]["tracker"]
    output_tracker = tracker_name(input_tracker, variant)

    for frame_index in sorted(grouped):
        source_rows = grouped[frame_index]
        points, confidence, valid = frame_input(source_rows)
        started_ns = time.perf_counter_ns()
        result = kinematic_constraints.apply_frame(
            points,
            confidence,
            valid,
            profile,
            config,
            variant,
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        processing_times.append(elapsed_ms)
        adjusted = result["adjusted_joints"]
        rejected = result["rejected_joints"]
        total_adjusted += len(adjusted)
        total_rejected += len(rejected)

        raw_diagnostics = kinematic_constraints.evaluate_lengths(
            points,
            valid,
            profile,
            config["variants"]["k1_length_gate"],
        )
        raw_by_bone = {row["bone"]: row for row in raw_diagnostics}
        final_by_bone = {row["bone"]: row for row in result["diagnostics"]}
        frame_raw_violations = sum(
            bool(row["gate_violation"]) for row in raw_diagnostics
        )
        frame_final_violations = sum(
            bool(row["gate_violation"]) for row in result["diagnostics"]
        )
        raw_violation_count += frame_raw_violations
        final_violation_count += frame_final_violations

        maximum_correction_mm = 0.0
        for row in source_rows:
            output_row = dict(row)
            joint = row["canonical_joint"]
            output_row["tracker"] = output_tracker
            output_row["raw_x_m"] = row.get("x_m", "")
            output_row["raw_y_m"] = row.get("y_m", "")
            output_row["raw_z_m"] = row.get("z_m", "")
            output_row["raw_valid"] = row.get("valid", "")
            output_row["kinematic_variant"] = variant
            output_row["kinematic_adjusted"] = "1" if joint in adjusted else "0"
            output_row["kinematic_rejected"] = "1" if joint in rejected else "0"

            correction_mm = 0.0
            if joint in points and joint in result["points"]:
                correction_mm = 1000.0 * float(
                    np.linalg.norm(result["points"][joint] - points[joint])
                )
            maximum_correction_mm = max(maximum_correction_mm, correction_mm)
            output_row["kinematic_correction_mm"] = repr(correction_mm)

            if joint in result["points"] and writes_projected_coordinates(variant):
                output_row["x_m"] = repr(float(result["points"][joint][0]))
                output_row["y_m"] = repr(float(result["points"][joint][1]))
                output_row["z_m"] = repr(float(result["points"][joint][2]))
            if joint in result["valid"]:
                output_row["valid"] = "1" if result["valid"][joint] else "0"
            output_rows.append(output_row)

        run_id = source_rows[0]["run_id"]
        for bone in profile["bones"]:
            raw = raw_by_bone[bone["name"]]
            final = final_by_bone[bone["name"]]
            diagnostic_rows.append(
                {
                    "run_id": run_id,
                    "frame_index": frame_index,
                    "kinematic_variant": variant,
                    "bone": bone["name"],
                    "joint_a": bone["joint_a"],
                    "joint_b": bone["joint_b"],
                    "constraint_enabled": int(bool(bone["constraint_enabled"])),
                    "raw_observed_length_m": format_optional(
                        raw["observed_length_m"]
                    ),
                    "final_observed_length_m": format_optional(
                        final["observed_length_m"]
                    ),
                    "reference_length_m": repr(
                        float(bone["reference_length_m"])
                    ),
                    "raw_length_error_m": format_optional(raw["length_error_m"]),
                    "final_length_error_m": format_optional(
                        final["length_error_m"]
                    ),
                    "tolerance_m": repr(float(raw["tolerance_m"])),
                    "raw_gate_violation": int(bool(raw["gate_violation"])),
                    "final_gate_violation": int(bool(final["gate_violation"])),
                }
            )
        frame_rows.append(
            {
                "run_id": run_id,
                "frame_index": frame_index,
                "kinematic_variant": variant,
                "input_valid_joint_count": sum(valid.values()),
                "output_valid_joint_count": sum(result["valid"].values()),
                "adjusted_joint_count": len(adjusted),
                "rejected_joint_count": len(rejected),
                "raw_gate_violation_count": frame_raw_violations,
                "final_gate_violation_count": frame_final_violations,
                "maximum_joint_correction_mm": maximum_correction_mm,
                "processing_time_ms": elapsed_ms,
            }
        )

    write_csv(output_dir / "tracking_joints.csv", output_rows, output_fields)
    shutil.copy2(input_frames_path, output_dir / "tracking_frames.csv")
    write_csv(
        output_dir / "kinematic_bone_diagnostics.csv",
        diagnostic_rows,
        DIAGNOSTIC_FIELDS,
    )
    write_csv(
        output_dir / "kinematic_frame_timing.csv",
        frame_rows,
        FRAME_FIELDS,
    )
    sorted_times = sorted(processing_times)
    p95_index = max(0, math.ceil(0.95 * len(sorted_times)) - 1)
    summary = {
        "status": "complete",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_tracker": input_tracker,
        "tracker": output_tracker,
        **requested,
        "frame_count": len(grouped),
        "tracking_row_count": len(output_rows),
        "input_valid_joint_count": sum(
            row.get("raw_valid") == "1" for row in output_rows
        ),
        "output_valid_joint_count": sum(
            row.get("valid") == "1" for row in output_rows
        ),
        "adjusted_joint_count": total_adjusted,
        "rejected_joint_count": total_rejected,
        "raw_gate_violation_count": raw_violation_count,
        "final_gate_violation_count": final_violation_count,
        "processing_time_ms": {
            "mean": sum(processing_times) / len(processing_times),
            "p95": sorted_times[p95_index],
            "max": max(processing_times),
        },
        "outputs": {
            "tracking_joints": "tracking_joints.csv",
            "tracking_frames": "tracking_frames.csv",
            "bone_diagnostics": "kinematic_bone_diagnostics.csv",
            "frame_timing": "kinematic_frame_timing.csv",
        },
    }
    write_json_atomic(state_path, summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    process_run(
        args.input_dir,
        args.output_dir,
        args.variant,
        args.profile,
        args.config,
        args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
