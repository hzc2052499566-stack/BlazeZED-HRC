"""Post-observation measured-first View A/C diagnostic (excluded exploratory).

This method was defined only after the registered v1 Pilot showed that using
K4 inferred output unconditionally was harmful in View C.  It is therefore a
method-refinement diagnostic, never confirmatory evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import analyse_view_ac_best_view_fusion_pilot_v1 as v1
import replay_view_ac_common_frame_engineering_v1 as replay_check


CLASSIFICATION = "excluded_post_observation_exploratory_measured_first_v2"
CLAIM_ELIGIBILITY = "excluded_engineering_pilot_only"
METHODS = (
    "view_a_measured_first",
    "view_c_measured_first",
    "mean_fusion_measured_first_available",
    "reliability_best_view_measured_first",
    "oracle_best_view_measured_first",
)
FIELDS = v1.CANDIDATE_FIELDS + ["view_a_candidate_source", "view_c_candidate_source"]


def load_replay(path: Path, run_id: str) -> dict[str, dict[tuple[int, str], dict[str, str]]]:
    validated = replay_check.validate_replay_directory(path.resolve(), "exploratory")
    if validated["state"].get("manifest_qc", {}).get("run_id") != run_id:
        raise v1.AnalysisError("Replay run_id differs from corrected source.")
    selected = {name: {} for name in ("raw_measured", "k2_guarded", "k4_inferred")}
    for row in v1.read_csv(validated["estimates_path"]):
        method = row.get("method", "")
        if method not in selected:
            continue
        key = (v1.integer(row.get("sequence_index"), "sequence_index"), row.get("canonical_joint", ""))
        if key in selected[method]:
            raise v1.AnalysisError("Duplicate replay key: {}".format(key))
        selected[method][key] = row
    if len(selected["raw_measured"]) != 3600 or len(selected["k2_guarded"]) != 3600 or len(selected["k4_inferred"]) != 480:
        raise v1.AnalysisError("Unexpected replay method row counts.")
    return selected


def measured_first_candidate(
    replay: dict[str, dict[tuple[int, str], dict[str, str]]], sequence: int, joint: str
) -> tuple[np.ndarray | None, str, str, float]:
    status = v1.raw_reliability(replay["raw_measured"], sequence, joint)
    measured = v1.row_point(replay["k2_guarded"][(sequence, joint)])
    inferred = v1.row_point(replay["k4_inferred"][(sequence, joint)])
    visibility = v1.visibility(replay["k4_inferred"][(sequence, joint)])
    if status == "reliable" and measured is not None:
        return measured, "k2_reliable_measured", status, visibility
    if inferred is not None:
        return inferred, "k4_fallback_inferred", status, visibility
    return None, "unavailable", status, visibility


def arm_score(candidates: dict[str, tuple[np.ndarray | None, str, str, float]]) -> tuple[int, int, float]:
    measured_count = sum(value[1] == "k2_reliable_measured" for value in candidates.values())
    complete = int(all(value[0] is not None for value in candidates.values()))
    minimum_visibility = min(value[3] for value in candidates.values())
    return measured_count, complete, minimum_visibility


def validate_protocol(protocol: dict[str, Any], protocol_path: Path) -> None:
    if (
        protocol.get("status") != "frozen_post_observation"
        or protocol.get("classification") != CLASSIFICATION
        or protocol.get("claim_eligibility") != CLAIM_ELIGIBILITY
        or not protocol.get("observed_v1_before_method_definition")
        or bool(protocol.get("eligible_for_formal"))
    ):
        raise v1.AnalysisError("Not the registered post-observation exploratory protocol.")
    expected = {
        "methods": list(METHODS),
        "candidate_rule": "reliable_k2_measured_else_k4_inferred",
        "selector_score_lexicographic": [
            "reliable_k2_measured_joint_count", "complete_candidate_arm", "minimum_blazepose_visibility"
        ],
        "selector_tie_break": "view_a",
        "occlusion_depth_margin_m": v1.OCCLUSION_DEPTH_MARGIN_M,
        "repeat_count": 1,
    }
    if protocol.get("registered_design") != expected:
        raise v1.AnalysisError("Exploratory design mismatch.")
    paths = protocol.get("registered_paths", {})
    hashes = protocol.get("registered_sha256", {})
    if paths.get("analysis_protocol") != str(protocol_path.resolve()):
        raise v1.AnalysisError("Exploratory protocol path mismatch.")
    for label, digest in hashes.items():
        path = Path(str(paths.get(label, "")))
        if not path.is_file() or v1.sha256_file(path) != digest:
            raise v1.AnalysisError("Registered exploratory input mismatch: {}".format(label))


def build_rows(replay_a, replay_c, gt_a, calibration) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in range(v1.EXPECTED_FRAMES):
        candidates = {"a": {}, "c": {}}
        for joint in v1.ARM_JOINTS:
            candidates["a"][joint] = measured_first_candidate(replay_a, sequence, joint)
            point, source, status, visibility = measured_first_candidate(replay_c, sequence, joint)
            if point is not None:
                point = v1.transform_c(point, calibration)
            candidates["c"][joint] = (point, source, status, visibility)
        scores = {view: arm_score(candidates[view]) for view in ("a", "c")}
        complete = {view: bool(scores[view][1]) for view in ("a", "c")}
        selected = v1.select_reliability_view(scores["a"], scores["c"], complete["a"], complete["c"])
        if complete["a"] and complete["c"]:
            arm_errors = {}
            for view in ("a", "c"):
                arm_errors[view] = sum(
                    float(np.linalg.norm(candidates[view][joint][0] - np.asarray([
                        v1.finite(gt_a[(sequence, joint)][field], field)
                        for field in ("gt_x_m", "gt_y_m", "gt_z_m")
                    ]))) for joint in v1.ARM_JOINTS
                ) / len(v1.ARM_JOINTS)
            oracle = "c" if arm_errors["c"] < arm_errors["a"] else "a"
        elif complete["a"]:
            oracle = "a"
        elif complete["c"]:
            oracle = "c"
        else:
            oracle = ""
        for joint in v1.ARM_JOINTS:
            pa, source_a, status_a, vis_a = candidates["a"][joint]
            pc, source_c, status_c, vis_c = candidates["c"][joint]
            if pa is not None and pc is not None:
                mean = (pa + pc) / 2.0
                mean_selected, wa, wc = "a+c", 0.5, 0.5
            elif pa is not None:
                mean, mean_selected, wa, wc = pa, "a", 1.0, 0.0
            elif pc is not None:
                mean, mean_selected, wa, wc = pc, "c", 0.0, 1.0
            else:
                mean, mean_selected, wa, wc = None, "", 0.0, 0.0
            values = {
                "view_a_measured_first": (pa, "a" if pa is not None else "", 1.0 if pa is not None else 0.0, 0.0, "view_a_measured_first"),
                "view_c_measured_first": (pc, "c" if pc is not None else "", 0.0, 1.0 if pc is not None else 0.0, "view_c_measured_first"),
                "mean_fusion_measured_first_available": (mean, mean_selected, wa, wc, "available_mean_measured_first"),
                "reliability_best_view_measured_first": (
                    candidates[selected][joint][0] if selected else None, selected,
                    1.0 if selected == "a" else 0.0, 1.0 if selected == "c" else 0.0,
                    "gt_free_arm_level_measured_first_selector",
                ),
                "oracle_best_view_measured_first": (
                    candidates[oracle][joint][0] if oracle else None, oracle,
                    1.0 if oracle == "a" else 0.0, 1.0 if oracle == "c" else 0.0,
                    "evaluation_gt_oracle_measured_first",
                ),
            }
            for method in METHODS:
                point, chosen, wa, wc, provenance = values[method]
                row = v1.candidate_payload(
                    method=method, sequence=sequence, joint=joint, point=point,
                    selected_view=chosen, weight_a=wa, weight_c=wc,
                    point_a=pa, point_c=pc, status_a=status_a, status_c=status_c,
                    visibility_a=vis_a, visibility_c=vis_c,
                    score_a=scores["a"], score_c=scores["c"], provenance=provenance,
                )
                row["classification"] = CLASSIFICATION
                row["view_a_candidate_source"] = source_a
                row["view_c_candidate_source"] = source_c
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise v1.AnalysisError("Refusing to overwrite exploratory output: {}".format(output))
    protocol = v1.read_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    paths = protocol["registered_paths"]
    run_a, camera_a, gt_a, _ = v1.load_ground_truth(Path(paths["view_a_dir"]), "a")
    run_c, camera_c, gt_c, _ = v1.load_ground_truth(Path(paths["view_c_dir"]), "c")
    calibration = v1.build_calibration(gt_a, gt_c, run_a, run_c, camera_a, camera_c)
    replay_a = load_replay(Path(paths["view_a_replay"]), run_a)
    replay_c = load_replay(Path(paths["view_c_replay"]), run_c)
    rows = build_rows(replay_a, replay_c, gt_a, calibration)
    output.mkdir(parents=True, exist_ok=False)
    candidates_path = output / "view_ac_measured_first_exploratory_candidates.csv"
    write_csv(candidates_path, rows)
    summary = {
        "schema_version": 1, "status": "complete", "classification": CLASSIFICATION,
        "claim_eligibility": CLAIM_ELIGIBILITY, "eligible_for_formal": False,
        "analysis_registration_status": "post_observation_method_refinement",
        "observed_v1_before_method_definition": True, "repeat_count": 1,
        "methods": list(METHODS),
        "primary_endpoint_name": "right_arm_two_joint_mean_position_error_mm",
        "metrics": {method: {
            "active": v1.phase_metrics(rows, gt_a, method, True),
            "inactive": v1.phase_metrics(rows, gt_a, method, False),
        } for method in METHODS},
        "calibration": {"source": v1.CALIBRATION_SOURCE, "maximum_fit_residual_m": calibration["maximum_cross_view_fit_residual_m"]},
        "protocol_sha256": v1.sha256_file(protocol_path),
        "candidates_sha256": v1.sha256_file(candidates_path),
        "interpretation_restrictions": [
            "The registered v1 outcome was inspected before this method was defined.",
            "This is an excluded one-capture exploratory diagnostic, not confirmatory evidence.",
            "GT is used for engineering common-frame calibration and evaluation only.",
            "A later formal protocol must preregister the measured-first rule before independent repeats.",
        ],
    }
    v1.write_json_exclusive(output / "view_ac_measured_first_exploratory_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
