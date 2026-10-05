"""Compare the new multi-pose frontend with paired legacy Pose baselines."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "multiperson_limb_profile_pilot_v1"
    / "bridge"
)
PAIRS = (
    (
        "tasks_direct",
        "lite",
        BRIDGE_ROOT / "legacy_pose_lite",
        BRIDGE_ROOT / "female_static_10s_pose_landmarker_lite",
    ),
    (
        "tasks_direct",
        "full",
        BRIDGE_ROOT / "legacy_pose_full",
        BRIDGE_ROOT / "female_static_10s_pose_landmarker_full",
    ),
    (
        "hybrid_legacy_roi_smoothed",
        "lite",
        BRIDGE_ROOT / "legacy_pose_lite",
        BRIDGE_ROOT / "hybrid_legacy_roi_lite_smoothed_v1",
    ),
    (
        "hybrid_legacy_roi_smoothed",
        "full",
        BRIDGE_ROOT / "legacy_pose_full",
        BRIDGE_ROOT / "hybrid_legacy_roi_full_smoothed_v1",
    ),
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def tracker_summary(directory: Path) -> tuple[str, dict]:
    summary = read_json(directory / "summary_metrics.json")
    trackers = summary.get("trackers", {})
    if len(trackers) != 1:
        raise RuntimeError("Expected exactly one tracker in {}".format(directory))
    tracker_name = next(iter(trackers))
    return tracker_name, trackers[tracker_name]


def main() -> int:
    rows = []
    for method_family, variant, baseline_dir, candidate_dir in PAIRS:
        baseline_name, baseline = tracker_summary(baseline_dir)
        candidate_name, candidate = tracker_summary(candidate_dir)
        deltas = {
            "mpjpe_delta_mm": candidate["mpjpe_mm"] - baseline["mpjpe_mm"],
            "limb_mae_delta_mm": candidate["mean_limb_mae_mm"]
            - baseline["mean_limb_mae_mm"],
            "jitter_delta_mm": candidate["mean_joint_jitter_mm"]
            - baseline["mean_joint_jitter_mm"],
            "core_valid_rate_delta": candidate["valid_core_rate"]
            - baseline["valid_core_rate"],
        }
        gates = {
            "mpjpe_non_degradation_10mm": deltas["mpjpe_delta_mm"] <= 10.0,
            "limb_mae_non_degradation_10mm": deltas["limb_mae_delta_mm"] <= 10.0,
            "jitter_non_degradation_5mm": deltas["jitter_delta_mm"] <= 5.0,
            "core_coverage_non_degradation_0p1pp": deltas[
                "core_valid_rate_delta"
            ]
            >= -0.001,
        }
        rows.append(
            {
                "method_family": method_family,
                "variant": variant,
                "baseline_tracker": baseline_name,
                "candidate_tracker": candidate_name,
                "baseline_mpjpe_mm": baseline["mpjpe_mm"],
                "candidate_mpjpe_mm": candidate["mpjpe_mm"],
                "mpjpe_delta_mm": deltas["mpjpe_delta_mm"],
                "baseline_limb_mae_mm": baseline["mean_limb_mae_mm"],
                "candidate_limb_mae_mm": candidate["mean_limb_mae_mm"],
                "limb_mae_delta_mm": deltas["limb_mae_delta_mm"],
                "baseline_jitter_mm": baseline["mean_joint_jitter_mm"],
                "candidate_jitter_mm": candidate["mean_joint_jitter_mm"],
                "jitter_delta_mm": deltas["jitter_delta_mm"],
                "baseline_core_valid_rate": baseline["valid_core_rate"],
                "candidate_core_valid_rate": candidate["valid_core_rate"],
                "core_valid_rate_delta": deltas["core_valid_rate_delta"],
                "passed_gate_count": sum(gates.values()),
                "gate_count": len(gates),
                "all_gates_passed": all(gates.values()),
                "quality_gates": gates,
            }
        )

    csv_fields = [key for key in rows[0] if key != "quality_gates"]
    with (BRIDGE_ROOT / "bridge_comparison_v2.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(
            [{key: row[key] for key in csv_fields} for row in rows]
        )

    hybrid_rows = [
        row
        for row in rows
        if row["method_family"] == "hybrid_legacy_roi_smoothed"
    ]
    hybrid_passed = len(hybrid_rows) == 2 and all(
        row["all_gates_passed"] for row in hybrid_rows
    )
    report = {
        "schema_version": 1,
        "experiment_stage": "excluded_engineering_preflight",
        "comparison": "MediaPipe Tasks multi-pose frontend versus paired legacy Pose",
        "input_identity": (
            "Same frozen 59-frame RGB-D sequence, same full-frame pixels, same "
            "median_7x7 depth, intrinsics and GT within each model variant."
        ),
        "engineering_decision_margins": {
            "maximum_mpjpe_increase_mm": 10.0,
            "maximum_limb_mae_increase_mm": 10.0,
            "maximum_jitter_increase_mm": 5.0,
            "maximum_core_coverage_drop": 0.001,
        },
        "variants": rows,
        "all_candidates_passed": all(row["all_gates_passed"] for row in rows),
        "recommended_hybrid_variants_passed": hybrid_passed,
        "recommended_method": (
            "hybrid_legacy_roi_smoothed_v1" if hybrid_passed else None
        ),
        "decision": (
            "eligible for two-person offline engineering capture with the hybrid method; direct Tasks landmarks remain ineligible"
            if hybrid_passed
            else "bridge non-degradation failed; do not start two-person capture"
        ),
        "limitations": [
            "This is a single static Isaac sequence and an engineering bridge, not multi-person evidence.",
            "These are engineering decision margins from the written pilot plan, not a formal preregistration.",
            "Offline compute timing is not live camera-to-output latency.",
            "A failed bridge does not invalidate the legacy BlazePose + ZED baseline.",
        ],
    }
    (BRIDGE_ROOT / "bridge_comparison_v2.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
