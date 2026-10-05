#!/usr/bin/env python3
"""Judge the BODY_38 GT comparison runs against the frozen protocol.

Thresholds are read from ``protocol_lock_v1.json`` rather than written here.
That file is the authority; duplicating its numbers in code would let the two
drift, and a gate that quietly disagrees with the protocol it claims to enforce
is worse than no gate.

Per run it evaluates the eight validity gates, and across runs it evaluates the
replication gates and aggregates the endpoints at run level - mean plus sample
SD over independent captures, never over frames, because frames inside one run
are repeated observations rather than repetitions (agents.md 7.3).

The primary endpoint has no registered direction. Whether BODY_38 beats the
BlazePose baseline is the question under test, so this reports which arm came
out lowest and whether the runs agreed, and does not call either outcome a
success. It also enforces the registered decision rule: a raw advantage for A3
is not an advantage unless the bias-removed column agrees, because Isaac GT is
a skeleton pivot and BODY_38 is a fitted skeleton joint while BlazePose
landmarks sit on the visible surface.

Runs that fail a validity gate are reported and excluded from the aggregate,
never silently repaired (agents.md rule 9.7).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    WORKSPACE / "output" / "experiments" / "body38_reference_gt_comparison_v1"
)
PROTOCOL_PATH = OUTPUT_ROOT / "protocol_lock_v1.json"
REPORT_PATH = OUTPUT_ROOT / "formal_summary.json"

ARMS = ("A1", "A2_median_7x7", "A2_wrist_aware_v5", "A3")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_run(run_dir: Path, gates: dict) -> dict:
    """Eight validity gates for one capture, plus the endpoints it produced."""
    capture_state = read_json(run_dir / "capture_state.json")
    comparison = read_json(run_dir / "body38_gt_comparison.json")
    anchor = read_json(run_dir / "arm_a1_gt_comparison.json")
    attribution = read_json(run_dir / "ncc_attribution_v3.json")
    gt_dir = run_dir / "pass1_gt"
    manifest = read_csv(gt_dir / "rgbd_manifest.csv")

    time_codes = [int(row["usd_time_code"]) for row in manifest]
    summary = capture_state["summary"]
    stereo = capture_state.get("stereo_probe", {})

    frames = read_csv(run_dir / "capture_frames.csv")
    covered = {
        int(row["time_code"]) for row in frames if int(row["time_code"]) >= 0
    }
    coverage = len(covered) / (len(set(time_codes)) + 1)

    lead = attribution.get("beacon_minus_attributed", {})
    lead_inside = None
    if lead.get("count"):
        rows = read_csv(run_dir / "frame_time_code_attribution_v3.csv")
        offsets = [
            int(row["beacon_minus_attributed"])
            for row in rows
            if row["accepted"] == "True" and row["beacon_minus_attributed"]
        ]
        if offsets:
            lead_inside = sum(1 for v in offsets if 4 <= v <= 14) / len(offsets)

    detection = min(
        summary["body38_detected_frames"] / summary["scored_frames"],
        summary["blazepose_detected_frames"] / summary["scored_frames"],
    ) if summary.get("scored_frames") else 0.0

    shift = stereo.get("best_horizontal_shift_px", {}).get("median")
    predicted = stereo.get("predicted_disparity_px_at_3p5m")
    anchor_mm = anchor["right_arm_two_joint"]["mean_mm"]
    reference_mm = gates["V8_reference_mm"]
    anchor_deviation = abs(anchor_mm - reference_mm) / reference_mm

    pelvis = comparison["unit_check"]["pelvis_forward_median_m"]
    rendered_median = anchor["rendered_depth_check"][
        "median_finite_depth_after_scale"
    ]

    checks = [
        {
            "id": "V1_pass1_time_codes",
            "observed": f"{len(manifest)} rows, {min(time_codes)}..{max(time_codes)}, "
            f"{len(set(time_codes))} unique",
            "passed": len(manifest) == 240
            and len(set(time_codes)) == 240
            and min(time_codes) == 0
            and max(time_codes) == 239,
        },
        {
            "id": "V2_pelvis_forward_m",
            "observed": pelvis,
            "threshold": gates["V2_pelvis_forward_m"],
            "passed": gates["V2_pelvis_forward_m"][0]
            <= pelvis
            <= gates["V2_pelvis_forward_m"][1],
        },
        {
            "id": "V3_rendered_depth_median_m",
            "observed": rendered_median,
            "threshold": gates["V3_rendered_depth_median_m"],
            "passed": gates["V3_rendered_depth_median_m"][0]
            <= rendered_median
            <= gates["V3_rendered_depth_median_m"][1],
        },
        {
            "id": "V4_detection_rate",
            "observed": round(detection, 6),
            "threshold": gates["V4_detection_rate_min"],
            "passed": detection >= gates["V4_detection_rate_min"],
        },
        {
            "id": "V5_stereo_shift_px",
            "observed": shift,
            "predicted": predicted,
            "threshold": gates["V5_stereo_shift_px_tolerance"],
            "passed": shift is not None
            and predicted is not None
            and abs(shift - predicted) <= gates["V5_stereo_shift_px_tolerance"],
        },
        {
            "id": "V6_attribution_beacon_lag",
            "observed": lead_inside,
            "threshold": gates["V6_attribution_beacon_lag_inside_4_14_min"],
            "passed": lead_inside is not None
            and lead_inside >= gates["V6_attribution_beacon_lag_inside_4_14_min"],
        },
        {
            "id": "V7_time_code_coverage",
            "observed": round(coverage, 6),
            "threshold": gates["V7_time_code_coverage_min"],
            "passed": coverage >= gates["V7_time_code_coverage_min"],
        },
        {
            "id": "V8_a1_anchor_deviation",
            "observed": round(anchor_deviation, 6),
            "anchor_mm": anchor_mm,
            "reference_mm": reference_mm,
            "threshold": gates["V8_a1_anchor_deviation_max"],
            "passed": anchor_deviation <= gates["V8_a1_anchor_deviation_max"],
        },
    ]

    endpoints = {
        "A1": {
            "raw_mm": anchor_mm,
            "bias_removed_mm": None,
            "coverage": None,
            "note": "A1 carries no attribution floor and no bias split",
        }
    }
    for arm, result in comparison["arms"].items():
        endpoints[arm] = {
            "raw_mm": result["right_arm_two_joint_raw"].get("mean_mm"),
            "bias_removed_mm": result["right_arm_two_joint_bias_removed"].get(
                "mean_mm"
            ),
            "coverage": result["coverage"]["valid_fraction"],
        }

    return {
        "run": run_dir.name,
        "scored_frames": summary.get("scored_frames"),
        "attributed_frames": comparison["attribution"]["attributed_frames"],
        "validity_checks": checks,
        "validity_passed": all(check["passed"] for check in checks),
        "endpoints": endpoints,
        "sdk_minus_rendered_median_mm": read_json(
            run_dir / "sdk_stereo_depth_quality.json"
        )["pooled_absolute_sdk_minus_rendered"]["median_mm"]
        if (run_dir / "sdk_stereo_depth_quality.json").exists()
        else None,
    }


def aggregate(values: list[float]) -> dict:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"runs": 0}
    entry = {
        "runs": len(clean),
        "mean": round(statistics.mean(clean), 3),
        "values": [round(value, 3) for value in clean],
    }
    if len(clean) > 1:
        sd = statistics.stdev(clean)
        entry["sample_sd"] = round(sd, 3)
        entry["sd_over_mean"] = round(sd / entry["mean"], 4) if entry["mean"] else None
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if REPORT_PATH.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite {relative(REPORT_PATH)}; pass --overwrite"
        )
    if not PROTOCOL_PATH.exists():
        raise SystemExit(f"{relative(PROTOCOL_PATH)} is missing; freeze first")

    protocol = read_json(PROTOCOL_PATH)
    gates = protocol["gates"]["validity_all_must_pass"]
    replication = protocol["gates"]["replication"]

    runs = []
    for name in args.runs:
        run_dir = OUTPUT_ROOT / "capture" / name
        if not run_dir.exists():
            raise SystemExit(f"missing run directory {relative(run_dir)}")
        runs.append(evaluate_run(run_dir, gates))

    valid = [run for run in runs if run["validity_passed"]]
    excluded = [run["run"] for run in runs if not run["validity_passed"]]

    endpoint_summary = {}
    for arm in ARMS:
        endpoint_summary[arm] = {
            "raw": aggregate([run["endpoints"].get(arm, {}).get("raw_mm") for run in valid]),
            "bias_removed": aggregate(
                [run["endpoints"].get(arm, {}).get("bias_removed_mm") for run in valid]
            ),
            "coverage": aggregate(
                [run["endpoints"].get(arm, {}).get("coverage") for run in valid]
            ),
        }

    # Which arm came lowest, per run, on each column. The direction is not
    # registered, so this records agreement rather than success.
    def lowest(run: dict, column: str) -> str | None:
        candidates = {
            arm: run["endpoints"].get(arm, {}).get(column)
            for arm in ARMS
            if arm != "A1"
        }
        candidates = {k: v for k, v in candidates.items() if v is not None}
        return min(candidates, key=candidates.get) if candidates else None

    raw_winners = [lowest(run, "raw_mm") for run in valid]
    debiased_winners = [lowest(run, "bias_removed_mm") for run in valid]

    a1_values = [run["endpoints"]["A1"]["raw_mm"] for run in valid]
    a1_aggregate = aggregate(a1_values)

    replication_checks = [
        {
            "id": "R1_a1_sd_over_mean",
            "observed": a1_aggregate.get("sd_over_mean"),
            "threshold": replication["R1_a1_sd_over_mean_max"],
            "passed": (
                a1_aggregate.get("sd_over_mean") is not None
                and a1_aggregate["sd_over_mean"]
                <= replication["R1_a1_sd_over_mean_max"]
            ),
            "evaluable": len(valid) > 1,
        },
        {
            "id": "R2_primary_direction_consistent",
            "observed": {
                "raw_lowest_per_run": raw_winners,
                "bias_removed_lowest_per_run": debiased_winners,
            },
            "threshold": replication["R2_primary_direction_consistent_runs"],
            "passed": len(set(raw_winners)) == 1
            and len(set(debiased_winners)) == 1
            and len(valid) >= replication["R2_primary_direction_consistent_runs"],
            "evaluable": len(valid)
            >= replication["R2_primary_direction_consistent_runs"],
        },
        {
            "id": "R3_runs_required",
            "observed": len(valid),
            "threshold": replication["R3_runs_required"],
            "passed": len(valid) >= replication["R3_runs_required"],
            "evaluable": True,
        },
    ]

    decision = None
    if raw_winners and debiased_winners:
        raw_arm = raw_winners[0] if len(set(raw_winners)) == 1 else None
        fixed_arm = (
            debiased_winners[0] if len(set(debiased_winners)) == 1 else None
        )
        decision = {
            "raw_lowest": raw_arm,
            "bias_removed_lowest": fixed_arm,
            "columns_agree": raw_arm is not None and raw_arm == fixed_arm,
            "rule": protocol["gates"]["primary_decision_rule"],
            "direction_was_not_registered": True,
        }

    payload = {
        "schema_version": 1,
        "stage": "body38_gt_comparison_formal_summary",
        "generated_utc": utc_now(),
        "environment": {"platform": platform.platform()},
        "protocol_sha256": protocol["content_sha256"],
        "input_manifest_sha256": protocol["input_manifest_sha256"],
        "runs_supplied": args.runs,
        "runs_valid": [run["run"] for run in valid],
        "runs_excluded_for_validity": excluded,
        "per_run": runs,
        "endpoints_run_level": endpoint_summary,
        "a1_anchor": a1_aggregate,
        "sdk_minus_rendered_median_mm": aggregate(
            [run["sdk_minus_rendered_median_mm"] for run in valid]
        ),
        "replication_checks": replication_checks,
        "primary_decision": decision,
        "formal_contract_met": all(
            check["passed"] for check in replication_checks
        )
        and not excluded,
        "reporting_rules": protocol["reporting_rules"],
        "a3_configuration_caveat": protocol["arms"]["A3"][
            "configuration_caveat"
        ],
        "attribution_floor_mm": protocol["attribution"]["floor_mm"],
    }
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {relative(REPORT_PATH)}")
    for run in runs:
        mark = "PASS" if run["validity_passed"] else "FAIL"
        print(f"\n{run['run']}: validity {mark}")
        for check in run["validity_checks"]:
            if not check["passed"]:
                print(f"    FAILED {check['id']}: {check.get('observed')}")
    print(f"\n{'arm':<20} {'raw mean':>12} {'debiased':>12} {'coverage':>10}")
    for arm in ARMS:
        entry = endpoint_summary[arm]
        print(
            "{:<20} {:>12} {:>12} {:>10}".format(
                arm,
                entry["raw"].get("mean", "-"),
                entry["bias_removed"].get("mean", "-"),
                entry["coverage"].get("mean", "-"),
            )
        )
    print("\nreplication:")
    for check in replication_checks:
        state = (
            ("PASS" if check["passed"] else "FAIL")
            if check["evaluable"]
            else "not yet evaluable"
        )
        print(f"  {check['id']:<34} {state}  observed {check['observed']}")
    print(f"\nformal contract met: {payload['formal_contract_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
