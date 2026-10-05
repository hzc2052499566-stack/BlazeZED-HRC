"""Diagnose FPS and wrist-validity failures in the closed live Formal.

The wrist threshold table is a fixed-reference sensitivity calculation. It
does not re-run the stateful selector and must not be interpreted as an exact
counterfactual replay.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "live_v2_heldout_formal"
)
RAW_ROOT = ROOT / "raw" / "formal_diagonal_extended_ab" / "d_3p50m"
OUTPUT = ROOT / "capture_reliability_diagnosis.json"
WARMUP_FRAMES = 150
WRIST_VALID_TARGET = 0.90
LARGE_JUMP_THRESHOLD_MM = 50.0
GATE_SENSITIVITY_MM = (45, 50, 55, 60, 65, 70, 75, 80, 90, 100)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def wrist_sensitivity(rows: list[dict[str, str]], joint: str) -> dict:
    selected = [
        row
        for row in rows
        if row["canonical_joint"] == joint
        and int(row["frame_index"]) >= WARMUP_FRAMES
    ]
    valid_count = sum(row["valid"] == "1" for row in selected)
    rejected = [
        row
        for row in selected
        if row["invalid_reason"] == "fallback_depth_temporal_discontinuity"
        and row["depth_temporal_delta_m"]
    ]
    rejected_deltas_mm = sorted(
        1000.0 * float(row["depth_temporal_delta_m"]) for row in rejected
    )
    required_valid_count = math.ceil(WRIST_VALID_TARGET * len(selected))
    rescue_needed = max(0, required_valid_count - valid_count)
    minimum_fixed_reference_gate_mm = None
    if rescue_needed and rescue_needed <= len(rejected_deltas_mm):
        minimum_fixed_reference_gate_mm = rejected_deltas_mm[rescue_needed - 1]

    sensitivity = []
    for gate_mm in GATE_SENSITIVITY_MM:
        rescued = [value for value in rejected_deltas_mm if value <= gate_mm]
        sensitivity.append(
            {
                "gate_mm": gate_mm,
                "fixed_reference_rescued_row_count": len(rescued),
                "estimated_valid_rate": (
                    (valid_count + len(rescued)) / len(selected)
                    if selected
                    else None
                ),
                "rescued_rows_above_large_jump_threshold": sum(
                    value > LARGE_JUMP_THRESHOLD_MM for value in rescued
                ),
            }
        )
    return {
        "steady_row_count": len(selected),
        "steady_valid_count": valid_count,
        "steady_valid_rate": valid_count / len(selected) if selected else None,
        "temporal_rejection_count": len(rejected),
        "valid_rows_needed_for_90_percent": rescue_needed,
        "minimum_fixed_reference_gate_for_90_percent_mm": (
            minimum_fixed_reference_gate_mm
        ),
        "rejected_temporal_delta_mm": {
            "minimum": (
                min(rejected_deltas_mm) if rejected_deltas_mm else None
            ),
            "median": (
                statistics.median(rejected_deltas_mm)
                if rejected_deltas_mm
                else None
            ),
            "maximum": (
                max(rejected_deltas_mm) if rejected_deltas_mm else None
            ),
        },
        "fixed_reference_sensitivity": sensitivity,
    }


def phase(summary: dict, name: str) -> dict:
    values = summary["steady_state"][name]
    return {
        "mean_ms": values["mean"],
        "p95_ms": values["p95"],
        "maximum_ms": values["max"],
    }


def main() -> int:
    runs = []
    for repeat in (1, 2, 3):
        tag = f"rep_{repeat:02d}"
        run_dir = RAW_ROOT / tag
        qc = json.loads(
            (run_dir / "live_calibration_run_qc.json").read_text(
                encoding="utf-8"
            )
        )
        summary = json.loads(
            (run_dir / "live_performance_summary.json").read_text(
                encoding="utf-8"
            )
        )
        preflight = json.loads(
            (run_dir / "zed_stream_preflight.json").read_text(
                encoding="utf-8"
            )
        )
        rows = read_csv(run_dir / "tracking_joints.csv")
        source_acquisition = summary["acquisition"][
            "source_acquisition_rate_fps"
        ]
        output_fps = summary["actual_output_fps"]
        runs.append(
            {
                "repeat": repeat,
                "capture_status": qc["status"],
                "failed_checks": [
                    name
                    for name, passed in qc["checks"].items()
                    if not passed
                ],
                "fps": {
                    "sender_preflight_fps": preflight["attempts"][0]["fps"],
                    "source_timestamp_fps": summary["acquisition"][
                        "source_timestamp_rate_fps"
                    ],
                    "source_acquisition_fps": source_acquisition,
                    "actual_output_fps": output_fps,
                    "absolute_output_source_acquisition_gap_fps": abs(
                        output_fps - source_acquisition
                    ),
                },
                "phases": {
                    name: phase(summary, name)
                    for name in (
                        "frame_wait_ms",
                        "grab_wait_ms",
                        "rgb_retrieve_ms",
                        "rgb_copy_convert_ms",
                        "depth_retrieve_ms",
                        "depth_copy_ms",
                        "pose_inference_ms",
                        "depth_sampling_ms",
                        "consumer_compute_latency_ms",
                        "host_pipeline_latency_ms",
                    )
                },
                "left_wrist": wrist_sensitivity(rows, "left_wrist"),
                "right_wrist": wrist_sensitivity(rows, "right_wrist"),
            }
        )

    acquisition_rates = [
        run["fps"]["source_acquisition_fps"] for run in runs
    ]
    output_rates = [run["fps"]["actual_output_fps"] for run in runs]
    output_source_gaps = [
        run["fps"]["absolute_output_source_acquisition_gap_fps"]
        for run in runs
    ]
    report = {
        "schema_version": 1,
        "status": "complete",
        "scope": "Closed Formal upstream capture-reliability diagnosis.",
        "runs": runs,
        "aggregate": {
            "source_acquisition_fps": {
                "mean": statistics.fmean(acquisition_rates),
                "sample_sd": statistics.stdev(acquisition_rates),
                "minimum": min(acquisition_rates),
                "maximum": max(acquisition_rates),
            },
            "actual_output_fps": {
                "mean": statistics.fmean(output_rates),
                "sample_sd": statistics.stdev(output_rates),
                "minimum": min(output_rates),
                "maximum": max(output_rates),
            },
            "maximum_absolute_output_source_gap_fps": max(
                output_source_gaps
            ),
        },
        "findings": {
            "sender_contract_was_60fps_in_every_run": all(
                run["fps"]["sender_preflight_fps"] == 60.0 for run in runs
            ),
            "source_timestamps_were_approximately_60fps_in_every_run": all(
                run["fps"]["source_timestamp_fps"] >= 59.0 for run in runs
            ),
            "output_rate_tracked_receiver_acquisition_rate": max(
                output_source_gaps
            )
            < 0.01,
            "small_gate_increase_to_50mm_would_rescue_repeat1": (
                next(
                    item
                    for item in runs[0]["right_wrist"][
                        "fixed_reference_sensitivity"
                    ]
                    if item["gate_mm"] == 50
                )["estimated_valid_rate"]
                >= WRIST_VALID_TARGET
            ),
            "small_gate_increase_to_50mm_would_rescue_repeat2": (
                next(
                    item
                    for item in runs[1]["right_wrist"][
                        "fixed_reference_sensitivity"
                    ]
                    if item["gate_mm"] == 50
                )["estimated_valid_rate"]
                >= WRIST_VALID_TARGET
            ),
            "repeat2_fixed_reference_gate_needed_for_90_percent_mm": runs[1][
                "right_wrist"
            ]["minimum_fixed_reference_gate_for_90_percent_mm"],
        },
        "interpretation": [
            "The 60 FPS sender metadata and timestamps did not guarantee a "
            "60 FPS host receiver/acquisition rate.",
            "Actual output followed host source acquisition within 0.01 FPS, "
            "so the failed 27 FPS runs were source/receiver limited rather "
            "than K2g limited.",
            "A 50 mm fixed-reference wrist gate would be sufficient for "
            "repeat 1 but not repeat 2.",
            "Repeat 2 would require approximately 72 mm under the recorded "
            "reference, above the 50 mm large-jump threshold. Do not solve "
            "the availability failure by simply widening the temporal gate.",
            "The wrist table is diagnostic only because changing an accepted "
            "sample changes subsequent state and prediction.",
        ],
    }
    write_json(OUTPUT, report)
    print(json.dumps(report, indent=2))
    print("Reliability diagnosis:", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
