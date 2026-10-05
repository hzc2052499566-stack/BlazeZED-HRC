"""Aggregate the view C v2 stage and evaluate the design gates.

The gates test properties of the fix, not accuracy figures, because the stage
is deterministic on frozen inputs and the figures were already observed
during method validation. See the frozen protocol for the reasoning.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "output" / "experiments" / "kinematic_constraints_ablation"
V1_ROOT = ABLATION / "dynamic_partial_occlusion_view_c_formal_v1"
V2_ROOT = ABLATION / "dynamic_partial_occlusion_view_c_formal_v2"
PROTOCOL_PATH = V2_ROOT / "view_c_formal_v2_protocol.json"
SUMMARY_JSON = V2_ROOT / "view_c_formal_v2_summary.json"
REPORT_NAME = "partial_occlusion_view_c_report.json"
REPEAT_COUNT = 3
METHODS = ("raw_measured", "k2_guarded", "k3_inferred", "k4_inferred")
PHASES = ("active", "inactive")
ENDPOINT_JOINTS = ("right_elbow", "right_wrist")
LATCH_REASON = "elbow_directional_depth_unavailable"

GATE_NAMES = (
    "all_repeats_inert_where_v1_locked",
    "all_repeats_latched_samples_recovered",
    "all_repeats_no_new_threshold_registered",
    "all_repeats_measured_coverage_unity",
    "all_repeats_gt_not_read_during_estimation",
    "all_repeats_secondary_view_checks_pass",
)


class AggregationError(RuntimeError):
    """Raised when the v2 aggregation contract is incomplete."""


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


def estimates(directory: Path) -> dict:
    rows = read_csv(
        directory / "dynamic_k4_replay" / "dynamic_kinematic_estimates.csv"
    )
    return {
        (r["method"], int(r["sequence_index"]), r["canonical_joint"]): r
        for r in rows
    }


def compare_repeat(index: int) -> dict:
    tag = "rep_{:02d}".format(index)
    v1 = V1_ROOT / "unit_corrected" / tag
    v2 = V2_ROOT / "unit_corrected" / tag
    report_path = v2 / REPORT_NAME
    if not report_path.is_file():
        raise AggregationError("Missing v2 report: {}".format(report_path))
    report = read_json(report_path)
    a, b = estimates(v1), estimates(v2)
    if set(a) != set(b):
        raise AggregationError("Estimate row sets differ for {}.".format(tag))

    perturbed = []
    latched = 0
    recovered = 0
    for key, row_a in a.items():
        row_b = b[key]
        a_valid = row_a["valid"].strip() == "1"
        b_valid = row_b["valid"].strip() == "1"
        if a_valid:
            same = b_valid and all(
                row_a[f] == row_b[f] for f in ("x_m", "y_m", "z_m")
            )
            if not same:
                perturbed.append(key)
        elif row_a.get("invalid_reason") == LATCH_REASON:
            latched += 1
            if b_valid:
                recovered += 1

    replay_state = read_json(
        v2 / "dynamic_k4_replay" / "dynamic_kinematic_replay_state.json"
    )
    patch = read_json(v2 / "dynamic_k4_replay" / "prelock_relock_patch.json")
    return {
        "repeat_index": index,
        "report": report,
        "report_sha256": sha256_file(report_path),
        "v1_valid_rows_perturbed": len(perturbed),
        "v1_latched_rows": latched,
        "v1_latched_rows_recovered": recovered,
        "gt_read_during_estimation": bool(
            replay_state.get("gt_read_during_estimation", True)
        ),
        "patch_provenance": patch,
    }


def evaluate_gates(repeats: list[dict]) -> dict:
    reports = [item["report"] for item in repeats]
    gates = {
        "all_repeats_inert_where_v1_locked": all(
            item["v1_valid_rows_perturbed"] == 0 for item in repeats
        ),
        "all_repeats_latched_samples_recovered": all(
            item["v1_latched_rows_recovered"] == item["v1_latched_rows"]
            for item in repeats
        ),
        "all_repeats_no_new_threshold_registered": all(
            "threshold" not in json.dumps(
                item["patch_provenance"]
            ).lower().replace("no threshold was added or changed", "")
            for item in repeats
        ),
        "all_repeats_measured_coverage_unity": all(
            r["method_metrics"]["raw_measured"]["active"][
                "joint_sample_coverage"
            ] == 1.0
            for r in reports
        ),
        "all_repeats_gt_not_read_during_estimation": all(
            not item["gt_read_during_estimation"] for item in repeats
        ),
        "all_repeats_secondary_view_checks_pass": all(
            r["secondary_view_unoccluded_check"]["checks"]["all_passed"]
            and r["secondary_view_resolvable_check"]["checks"]["all_passed"]
            for r in reports
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
        raise AggregationError("The v2 formal protocol is not frozen.")
    if SUMMARY_JSON.exists():
        raise AggregationError(
            "Refusing to overwrite: {}".format(SUMMARY_JSON)
        )

    repeats = [compare_repeat(i) for i in range(1, REPEAT_COUNT + 1)]
    reports = [item["report"] for item in repeats]

    methods = {}
    for method in METHODS:
        methods[method] = {
            phase: {
                key: summarise(
                    [r["method_metrics"][method][phase][key] for r in reports]
                )
                for key in (
                    "joint_sample_coverage",
                    "two_joint_position_error_mean_mm",
                    "two_joint_position_error_p95_mm",
                )
            }
            for phase in PHASES
        }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "dynamic_partial_occlusion_view_c_formal_v2",
        "claim_eligibility": "formal_paired_method_reanalysis",
        "protocol_sha256": protocol["protocol_sha256"],
        "method_version": protocol["method_version"],
        "repeat_count": len(repeats),
        "statistical_unit": "independent_run",
        "dispersion": "sample_sd_over_runs_ddof_1",
        "paired_comparison_against_v1": {
            "v1_valid_rows_perturbed": summarise(
                [i["v1_valid_rows_perturbed"] for i in repeats]
            ),
            "v1_latched_rows": summarise(
                [i["v1_latched_rows"] for i in repeats]
            ),
            "v1_latched_rows_recovered": summarise(
                [i["v1_latched_rows_recovered"] for i in repeats]
            ),
        },
        "method_metrics": methods,
        "preregistered_gates": evaluate_gates(repeats),
        "repeat_reports": {
            "rep_{:02d}".format(i["repeat_index"]): i["report_sha256"]
            for i in repeats
        },
        "interpretation_rules": [
            "This stage re-analyses frozen captures under a new estimation "
            "method. It establishes what the fix changes; it is not "
            "independent evidence about the occlusion result.",
            "Run-level SD reflects the same render noise as the v1 stage, "
            "because the captures are the same three.",
            "The endpoint is right elbow plus right wrist position error, "
            "not MPJPE.",
            "Absolute errors are not comparable to view A.",
            "The frozen view C formal v1 results, including its two failed "
            "gates, remain the registered outcome of that stage.",
        ],
    }

    V2_ROOT.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(SUMMARY_JSON), flags)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")

    gates = summary["preregistered_gates"]
    print("Summary: {}".format(SUMMARY_JSON))
    for name in GATE_NAMES:
        print("  {:48s} {}".format(name, gates[name]))
    print("  {:48s} {}".format("ALL PASSED", gates["all_passed"]))
    paired = summary["paired_comparison_against_v1"]
    print()
    print("  v1 valid rows perturbed by v2 : {}".format(
        paired["v1_valid_rows_perturbed"]["values"]
    ))
    print("  v1 latched rows / recovered   : {} / {}".format(
        paired["v1_latched_rows"]["values"],
        paired["v1_latched_rows_recovered"]["values"],
    ))
    raw = methods["raw_measured"]
    for phase in PHASES:
        block = raw[phase]
        print("  raw {:8s} {:.2f} +/- {:.2f} mm, coverage {:.3f}".format(
            phase,
            block["two_joint_position_error_mean_mm"]["mean"],
            block["two_joint_position_error_mean_mm"]["sample_sd"],
            block["joint_sample_coverage"]["mean"],
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
