"""Aggregate the fresh-draw view C stage and evaluate the registered gates.

The primary gate is `all_repeats_v2_no_frame_in_latch_state`. Its outcome was
genuinely unknown when the protocol was frozen, because these are new render
noise draws and the latch is triggered by one pixel of jitter at frame 1.

The v1 latch count on the same draws is **reported without a gate**. Under v1
latching is a coin flip, so any value from zero to three is consistent with
the diagnosis and gating it would be meaningless.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    ROOT / "output" / "experiments" / "kinematic_constraints_ablation"
    / "dynamic_partial_occlusion_view_c_fresh_v2_formal"
)
PROTOCOL_PATH = EXPERIMENT_ROOT / "view_c_fresh_v2_protocol.json"
SUMMARY_JSON = EXPERIMENT_ROOT / "view_c_fresh_v2_summary.json"
REPORT_NAME = "partial_occlusion_view_c_report.json"
REPEAT_COUNT = 3
METHODS = ("raw_measured", "k2_guarded", "k3_inferred", "k4_inferred")
PHASES = ("active", "inactive")
ENDPOINT_JOINTS = ("right_elbow", "right_wrist")
TORSO_REFERENCE_JOINTS = ("right_shoulder", "pelvis")
LATCH_REASON = "elbow_directional_depth_unavailable"
COLLAPSE_MARGIN_M = 0.50

GATE_NAMES = (
    "all_repeats_captured_240_frames",
    "all_repeats_secondary_view_checks_pass",
    "all_repeats_detection_rate_unity",
    "all_repeats_gt_not_read_during_estimation",
    "all_repeats_v2_patch_provenance_present",
    "all_repeats_v2_no_frame_in_latch_state",
    "all_repeats_v2_measured_coverage_unity",
    "all_repeats_v2_measured_error_not_inflated_by_occlusion",
    "all_repeats_v2_k4_identical_to_k3",
    "all_repeats_v2_no_depth_collapse_under_occlusion",
)


class AggregationError(RuntimeError):
    """Raised when the fresh-draw aggregation contract is incomplete."""


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarise(values: list) -> dict:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "sample_sd": None, "values": []}
    return {
        "n": len(clean),
        "mean": float(statistics.fmean(clean)),
        "sample_sd": float(statistics.stdev(clean)) if len(clean) > 1 else 0.0,
        "values": clean,
    }


def latch_rows(replay_dir: Path) -> int:
    rows = read_csv(replay_dir / "dynamic_kinematic_estimates.csv")
    return sum(
        1 for r in rows
        if r["valid"].strip() != "1"
        and r.get("invalid_reason") == LATCH_REASON
    )


def elbow_valid_frames(replay_dir: Path) -> int:
    rows = read_csv(replay_dir / "dynamic_kinematic_estimates.csv")
    return sum(
        1 for r in rows
        if r["method"] == "raw_measured"
        and r["canonical_joint"] == "right_elbow"
        and r["valid"].strip() == "1"
    )


def depth_collapse(directory: Path, active: set) -> int:
    rows = read_csv(
        directory / "replay_method_v2" / "dynamic_kinematic_estimates.csv"
    )
    depth = {}
    for r in rows:
        if r["method"] != "raw_measured":
            continue
        key = (int(r["sequence_index"]), r["canonical_joint"])
        depth[key] = (
            float(r["x_m"]) if r["valid"].strip() == "1" else None
        )
    collapsed = 0
    for frame in sorted(active):
        reference = [
            depth.get((frame, j)) for j in TORSO_REFERENCE_JOINTS
        ]
        reference = [v for v in reference if v is not None]
        torso = statistics.fmean(reference) if reference else None
        for joint in ENDPOINT_JOINTS:
            value = depth.get((frame, joint))
            if value is None or torso is None:
                continue
            if torso - value > COLLAPSE_MARGIN_M:
                collapsed += 1
    return collapsed


def collect(index: int) -> dict:
    tag = "rep_{:02d}".format(index)
    directory = EXPERIMENT_ROOT / "unit_corrected" / tag
    report_path = directory / REPORT_NAME
    if not report_path.is_file():
        raise AggregationError("Missing report: {}".format(report_path))
    report = read_json(report_path)
    capture_state = read_json(
        EXPERIMENT_ROOT / "raw" / tag / "capture_state.json"
    )
    replay_state = read_json(
        directory / "replay_method_v2"
        / "dynamic_kinematic_replay_state.json"
    )
    patch_path = directory / "replay_method_v2" / "prelock_relock_patch.json"
    active = set()
    for block in report["occlusion_intervals"]:
        active.update(range(
            int(block["start_frame"]),
            int(block["end_frame_inclusive"]) + 1,
        ))
    return {
        "repeat_index": index,
        "report": report,
        "report_sha256": sha256_file(report_path),
        "capture_complete": (
            capture_state.get("status") == "complete"
            and int(capture_state.get("completed_sequence_count", -1)) == 240
            and int(capture_state.get("ground_truth_row_count", -1)) == 3600
        ),
        "gt_read_during_estimation": bool(
            replay_state.get("gt_read_during_estimation", True)
        ),
        "patch_present": patch_path.is_file(),
        "v1_latch_rows": latch_rows(directory / "replay_method_v1"),
        "v2_latch_rows": latch_rows(directory / "replay_method_v2"),
        "v1_elbow_valid_frames": elbow_valid_frames(
            directory / "replay_method_v1"
        ),
        "v2_elbow_valid_frames": elbow_valid_frames(
            directory / "replay_method_v2"
        ),
        "v2_depth_collapsed_samples": depth_collapse(directory, active),
    }


def evaluate_gates(repeats: list[dict]) -> dict:
    reports = [i["report"] for i in repeats]

    def metric(report, method, phase, key):
        return report["method_metrics"][method][phase][key]

    gates = {
        "all_repeats_captured_240_frames": all(
            i["capture_complete"] for i in repeats
        ),
        "all_repeats_secondary_view_checks_pass": all(
            r["secondary_view_unoccluded_check"]["checks"]["all_passed"]
            and r["secondary_view_resolvable_check"]["checks"]["all_passed"]
            for r in reports
        ),
        "all_repeats_detection_rate_unity": all(
            r["body_detection_rate"][p] == 1.0
            for r in reports for p in PHASES
        ),
        "all_repeats_gt_not_read_during_estimation": all(
            not i["gt_read_during_estimation"] for i in repeats
        ),
        "all_repeats_v2_patch_provenance_present": all(
            i["patch_present"] for i in repeats
        ),
        "all_repeats_v2_no_frame_in_latch_state": all(
            i["v2_latch_rows"] == 0 for i in repeats
        ),
        "all_repeats_v2_measured_coverage_unity": all(
            metric(r, "raw_measured", "active", "joint_sample_coverage") == 1.0
            for r in reports
        ),
        "all_repeats_v2_measured_error_not_inflated_by_occlusion": all(
            metric(
                r, "raw_measured", "active",
                "two_joint_position_error_mean_mm",
            )
            <= metric(
                r, "raw_measured", "inactive",
                "two_joint_position_error_mean_mm",
            )
            for r in reports
        ),
        "all_repeats_v2_k4_identical_to_k3": all(
            r["method_metrics"]["k4_inferred"]
            == r["method_metrics"]["k3_inferred"]
            for r in reports
        ),
        "all_repeats_v2_no_depth_collapse_under_occlusion": all(
            i["v2_depth_collapsed_samples"] == 0 for i in repeats
        ),
    }
    if set(gates) != set(GATE_NAMES):
        raise AggregationError("Gate name set drifted from the registry.")
    gates["all_passed"] = all(gates.values())
    return gates


def main() -> int:
    if not PROTOCOL_PATH.is_file():
        raise AggregationError("Freeze the protocol first.")
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("status") != "frozen":
        raise AggregationError("The fresh-draw protocol is not frozen.")
    if SUMMARY_JSON.exists():
        raise AggregationError("Refusing to overwrite: {}".format(
            SUMMARY_JSON
        ))

    repeats = [collect(i) for i in range(1, REPEAT_COUNT + 1)]
    reports = [i["report"] for i in repeats]

    methods = {
        method: {
            phase: {
                key: summarise([
                    r["method_metrics"][method][phase][key] for r in reports
                ])
                for key in (
                    "joint_sample_coverage",
                    "two_joint_position_error_mean_mm",
                    "two_joint_position_error_p95_mm",
                )
            }
            for phase in PHASES
        }
        for method in METHODS
    }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "dynamic_partial_occlusion_view_c_fresh_v2_formal",
        "claim_eligibility": "formal_three_run_fresh_draws",
        "protocol_sha256": protocol["protocol_sha256"],
        "method_version": protocol["method_version"],
        "repeat_count": len(repeats),
        "statistical_unit": "independent_run",
        "dispersion": "sample_sd_over_runs_ddof_1",
        "latch_comparison_on_fresh_draws": {
            "note": (
                "v1 values are reported without a gate. Latching under v1 is "
                "a coin flip decided by frame-1 jitter, so any value from "
                "zero to three is consistent with the diagnosis."
            ),
            "v1_latch_rows": summarise(
                [i["v1_latch_rows"] for i in repeats]
            ),
            "v2_latch_rows": summarise(
                [i["v2_latch_rows"] for i in repeats]
            ),
            "v1_elbow_valid_frames": summarise(
                [i["v1_elbow_valid_frames"] for i in repeats]
            ),
            "v2_elbow_valid_frames": summarise(
                [i["v2_elbow_valid_frames"] for i in repeats]
            ),
        },
        "method_metrics": methods,
        "preregistered_gates": evaluate_gates(repeats),
        "repeat_reports": {
            "rep_{:02d}".format(i["repeat_index"]): i["report_sha256"]
            for i in repeats
        },
        "interpretation_rules": [
            "Run-level mean +/- sample SD over three runs is the reporting "
            "unit; frames are not independent repeats.",
            "v1 and v2 replay the identical landmark cache per repeat, so "
            "the method comparison is exactly paired on each fresh draw.",
            "The endpoint is right elbow plus right wrist position error, "
            "not MPJPE.",
            "View C ground truth is in view C's own camera frame; absolute "
            "errors are not comparable to view A.",
            "A rendered occluder is not a physical occlusion of a real "
            "human.",
        ],
    }

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(SUMMARY_JSON), flags)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")

    gates = summary["preregistered_gates"]
    print("Summary: {}".format(SUMMARY_JSON))
    for name in GATE_NAMES:
        print("  {:56s} {}".format(name, gates[name]))
    print("  {:56s} {}".format("ALL PASSED", gates["all_passed"]))
    latch = summary["latch_comparison_on_fresh_draws"]
    print()
    print("  latch rows  v1: {}   v2: {}".format(
        latch["v1_latch_rows"]["values"], latch["v2_latch_rows"]["values"]
    ))
    print("  elbow valid v1: {}   v2: {}   (of 240)".format(
        latch["v1_elbow_valid_frames"]["values"],
        latch["v2_elbow_valid_frames"]["values"],
    ))
    raw = methods["raw_measured"]
    for phase in PHASES:
        b = raw[phase]
        print("  raw {:8s} {:.2f} +/- {:.2f} mm, coverage {:.3f}".format(
            phase,
            b["two_joint_position_error_mean_mm"]["mean"],
            b["two_joint_position_error_mean_mm"]["sample_sd"],
            b["joint_sample_coverage"]["mean"],
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
