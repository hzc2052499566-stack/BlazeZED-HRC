"""Develop CTS5 calibrated trimmed scalar limb-length fusion.

Only E0 rep_01..03 may fit the frozen camera-by-bone offsets.  The already
unblinded rep_04..06 data are read solely for an explicitly post-outcome E1
diagnostic and can never count as confirmatory evidence for CTS5.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimise_multiview_limb_length_v1 as metrics  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = (
    ROOT / "output/experiments/kinematic_constraints_ablation"
    / "multiview_weighted_limb_length_v1"
)
OUTPUT = (
    ROOT / "output/experiments/kinematic_constraints_ablation"
    / "multiview_weighted_limb_length_cts5_v2/development"
)
E0_SAMPLES = V1_ROOT / "development/camera_length_samples.csv"
E1_ERRORS = V1_ROOT / "formal_analysis/formal_length_errors.csv"
E1_PREDICTIONS = V1_ROOT / "formal_predictions"

VIEWS = ("m050", "m045", "m040", "p005", "p010")
BONES = ("right_upper_arm", "right_forearm")
E0_REPEATS = ("rep_01", "rep_02", "rep_03")
E1_REPEATS = ("rep_04", "rep_05", "rep_06")
CTS5_METHOD = "cts5_calibrated_trimmed_scalar"
CAL_UNIFORM_METHOD = "bias_corrected_uniform_mean"
RAW_UNIFORM_METHOD = "raw_uniform_mean"
RAW_TRIM_METHOD = "uncorrected_trimmed_mean"
V1_METHOD = "v1_optimized_shared_weights"
P010_METHOD = "development_selected_best_single_p010"


class CTS5DevelopmentError(RuntimeError):
    """Raised if development data or the CTS5 contract are invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise CTS5DevelopmentError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json_exclusive(path: Path, payload: Any) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = sorted((float(value), float(weight))
                   for value, weight in zip(values, weights))
    if not pairs or any(not math.isfinite(value) or not math.isfinite(weight)
                        or weight < 0.0 for value, weight in pairs):
        raise CTS5DevelopmentError("Weighted median received invalid values.")
    total = sum(weight for _, weight in pairs)
    if total <= 0.0:
        raise CTS5DevelopmentError("Weighted median has no positive mass.")
    threshold = 0.5 * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def fit_balanced_offsets(rows: list[dict[str, Any]],
                         repeats: Iterable[str]) -> dict[str, dict[str, float]]:
    selected_repeats = tuple(sorted(str(value) for value in repeats))
    if not selected_repeats:
        raise CTS5DevelopmentError("At least one training repeat is required.")
    selected = [row for row in rows if row["repeat"] in selected_repeats]
    expected_cells = {(repeat, phase, bone)
                      for repeat in selected_repeats
                      for phase in ("active", "inactive") for bone in BONES}
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        cells[(row["repeat"], row["phase"], row["bone"])].append(row)
    if set(cells) != expected_cells:
        raise CTS5DevelopmentError(
            f"Offset cells are incomplete: missing={sorted(expected_cells - set(cells))}"
        )
    offsets: dict[str, dict[str, float]] = {bone: {} for bone in BONES}
    for bone in BONES:
        for view in VIEWS:
            values: list[float] = []
            weights: list[float] = []
            for repeat in selected_repeats:
                for phase in ("active", "inactive"):
                    group = cells[(repeat, phase, bone)]
                    per_row_mass = 1.0 / (len(selected_repeats) * 2.0 * len(group))
                    for row in group:
                        values.append(float(row[f"{view}_length_mm"])
                                      - float(row["gt_length_mm"]))
                        weights.append(per_row_mass)
            offsets[bone][view] = weighted_median(values, weights)
    return offsets


def trimmed_scalar(values: list[float]) -> float | None:
    """Frozen deployment rule: trim both extremes for N>=4, median for N=3."""

    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if len(ordered) >= 4:
        retained = ordered[1:-1]
        return statistics.fmean(retained)
    if len(ordered) == 3:
        return statistics.median(ordered)
    return None


def cts5_estimate(lengths: dict[str, float | None], bone: str,
                  offsets: dict[str, dict[str, float]]) -> tuple[float | None, int]:
    corrected = [float(lengths[view]) - float(offsets[bone][view])
                 for view in VIEWS if lengths.get(view) is not None]
    return trimmed_scalar(corrected), len(corrected)


def corrected_uniform(lengths: dict[str, float | None], bone: str,
                      offsets: dict[str, dict[str, float]]) -> float | None:
    corrected = [float(lengths[view]) - float(offsets[bone][view])
                 for view in VIEWS if lengths.get(view) is not None]
    return statistics.fmean(corrected) if corrected else None


def load_e0_rows() -> list[dict[str, Any]]:
    rows = read_csv(E0_SAMPLES)
    required = {
        "repeat", "sequence_index", "phase", "bone", "gt_length_mm",
        *[f"{view}_length_mm" for view in VIEWS],
    }
    if not rows or not required.issubset(rows[0]):
        raise CTS5DevelopmentError("E0 camera-length table lacks required fields.")
    observed_repeats = {row["repeat"] for row in rows}
    if observed_repeats != set(E0_REPEATS):
        raise CTS5DevelopmentError(f"Unexpected E0 repeats: {observed_repeats}")
    return rows


def method_predictions(row: dict[str, Any],
                       offsets: dict[str, dict[str, float]]) -> dict[str, float]:
    bone = str(row["bone"])
    lengths = {view: float(row[f"{view}_length_mm"]) for view in VIEWS}
    cts5, count = cts5_estimate(lengths, bone, offsets)
    if cts5 is None or count != 5:
        raise CTS5DevelopmentError("Development sample is not complete-five.")
    raw_values = [lengths[view] for view in VIEWS]
    return {
        CTS5_METHOD: cts5,
        CAL_UNIFORM_METHOD: float(corrected_uniform(lengths, bone, offsets)),
        RAW_UNIFORM_METHOD: statistics.fmean(raw_values),
        RAW_TRIM_METHOD: float(trimmed_scalar(raw_values)),
        P010_METHOD: lengths["p010"],
    }


def prediction_rows(source_rows: list[dict[str, Any]], offsets_by_repeat: dict[
        str, dict[str, dict[str, float]]], classification: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in source_rows:
        repeat = str(row["repeat"])
        methods = method_predictions(row, offsets_by_repeat[repeat])
        truth = float(row["gt_length_mm"])
        for method, estimate in methods.items():
            error = estimate - truth
            output.append({
                "classification": classification,
                "repeat": repeat,
                "sequence_index": int(row["sequence_index"]),
                "phase": str(row["phase"]),
                "bone": str(row["bone"]),
                "method": method,
                "estimated_length_mm": estimate,
                "gt_length_mm": truth,
                "signed_error_mm": error,
                "absolute_error_mm": abs(error),
            })
    return output


def add_e1_v1_estimates(rows: list[dict[str, Any]], repeat: str,
                        estimate_by_key: dict[tuple[str, str], float]) -> None:
    for row in list(rows):
        if row["repeat"] != repeat or row["method"] != CTS5_METHOD:
            continue
        key = (str(row["sequence_index"]), str(row["bone"]))
        if key not in estimate_by_key:
            continue
        estimate = estimate_by_key[key]
        truth = float(row["gt_length_mm"])
        rows.append({**row, "method": V1_METHOD, "estimated_length_mm": estimate,
                     "signed_error_mm": estimate - truth,
                     "absolute_error_mm": abs(estimate - truth)})


def load_e1_rows() -> tuple[list[dict[str, Any]], dict[str, dict[tuple[str, str], float]]]:
    error_rows = read_csv(E1_ERRORS)
    gt = {(row["repeat"], row["sequence_index"], row["bone"]):
          float(row["gt_length_mm"]) for row in error_rows}
    source: list[dict[str, Any]] = []
    v1: dict[str, dict[tuple[str, str], float]] = {}
    for repeat in E1_REPEATS:
        rows = read_csv(E1_PREDICTIONS / repeat / "formal_predictions_gt_blind.csv")
        grouped: dict[tuple[str, str], dict[str, str]] = {}
        estimates: dict[tuple[str, str], float] = {}
        for row in rows:
            key = (row["sequence_index"], row["bone"])
            if row["method"] == "optimized_static_shared_weights" and row[
                    "estimated_length_mm"]:
                estimates[key] = float(row["estimated_length_mm"])
            if row["method"] != "optimized_static_shared_weights":
                continue
            if not all(row[f"{view}_length_mm"] for view in VIEWS):
                continue
            grouped[key] = row
        for key, row in grouped.items():
            gt_key = (repeat, key[0], key[1])
            if gt_key not in gt:
                raise CTS5DevelopmentError(f"E1 GT key is missing: {gt_key}")
            source.append({
                "repeat": repeat,
                "sequence_index": int(key[0]),
                "phase": row["phase"],
                "bone": key[1],
                "gt_length_mm": gt[gt_key],
                **{f"{view}_length_mm": float(row[f"{view}_length_mm"])
                   for view in VIEWS},
            })
        v1[repeat] = estimates
    return source, v1


def r5_diagnostics(summary: dict[str, Any], predictions: list[dict[str, Any]],
                   repeats: Iterable[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for repeat in repeats:
        for bone in BONES:
            cts = [float(row["signed_error_mm"]) for row in predictions
                   if row["repeat"] == repeat and row["bone"] == bone
                   and row["method"] == CTS5_METHOD]
            fair = [float(row["signed_error_mm"]) for row in predictions
                    if row["repeat"] == repeat and row["bone"] == bone
                    and row["method"] == CAL_UNIFORM_METHOD]
            cts_mad = float(metrics.median_absolute_deviation(cts))
            fair_mad = float(metrics.median_absolute_deviation(fair))
            output[f"{repeat}/{bone}"] = {
                "cts5_bias_mm": statistics.fmean(cts),
                "calibrated_uniform_bias_mm": statistics.fmean(fair),
                "cts5_error_mad_mm": cts_mad,
                "calibrated_uniform_error_mad_mm": fair_mad,
                "mad_ratio": cts_mad / fair_mad,
                "development_guard_band_pass": cts_mad / fair_mad <= 0.90,
                "formal_v1_bound_pass": cts_mad / fair_mad <= 1.05,
            }
    return output


def main() -> int:
    if OUTPUT.exists() or OUTPUT.with_name(OUTPUT.name + ".partial").exists():
        raise CTS5DevelopmentError(f"Refusing to overwrite development output: {OUTPUT}")
    e0 = load_e0_rows()
    full_offsets = fit_balanced_offsets(e0, E0_REPEATS)
    offsets_by_repeat = {
        held: fit_balanced_offsets(e0, [repeat for repeat in E0_REPEATS if repeat != held])
        for held in E0_REPEATS
    }
    e0_predictions = prediction_rows(
        e0, offsets_by_repeat, "excluded_e0_leave_one_capture_out_development"
    )
    e0_summary = metrics.summarise_predictions(e0_predictions, list(E0_REPEATS))
    e0_r5 = r5_diagnostics(e0_summary, e0_predictions, E0_REPEATS)

    e1_source, e1_v1 = load_e1_rows()
    e1_predictions = prediction_rows(
        e1_source,
        {repeat: full_offsets for repeat in E1_REPEATS},
        "post_outcome_e1_diagnostic_not_confirmatory",
    )
    for repeat in E1_REPEATS:
        add_e1_v1_estimates(e1_predictions, repeat, e1_v1[repeat])
    e1_summary = metrics.summarise_predictions(e1_predictions, list(E1_REPEATS))
    e1_r5 = r5_diagnostics(e1_summary, e1_predictions, E1_REPEATS)

    offset_payload = {
        "schema_version": 1,
        "status": "frozen_after_excluded_e0_development",
        "method": "cts5_calibrated_trimmed_scalar_v2",
        "fit_repeats": list(E0_REPEATS),
        "fit_data_classification": "excluded_retrospective_development",
        "offset_estimator": "capture_phase_balanced_weighted_median_signed_error_mm",
        "views": list(VIEWS),
        "bones": list(BONES),
        "offset_definition": "corrected_length_mm = raw_length_mm - offset_mm",
        "offsets_mm": full_offsets,
        "complete_five_rule": "sort corrected scalar lengths, discard min/max, mean middle three",
        "deployment_missing_rule": {
            "valid_5_or_4": "discard one minimum and one maximum; mean retained values",
            "valid_3": "median",
            "valid_fewer_than_3": "abstain",
        },
        "forbidden_inputs": ["K2", "K3", "K4", "formal_or_fresh_GT_at_prediction"],
    }
    offset_payload["offset_payload_sha256"] = payload_sha256(offset_payload)

    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    fields = [
        "classification", "repeat", "sequence_index", "phase", "bone", "method",
        "estimated_length_mm", "gt_length_mm", "signed_error_mm", "absolute_error_mm",
    ]
    write_csv_exclusive(partial / "e0_loco_predictions.csv", e0_predictions, fields)
    write_csv_exclusive(partial / "e1_post_outcome_diagnostic_predictions.csv",
                        e1_predictions, fields)
    write_json_exclusive(partial / "cts5_offsets.json", offset_payload)
    report = {
        "schema_version": 1,
        "status": "complete",
        "classification": "method_development_and_post_outcome_diagnostic_only",
        "e0_input": str(E0_SAMPLES.resolve()),
        "e0_input_sha256": sha256_file(E0_SAMPLES),
        "e1_formal_summary_sha256": sha256_file(V1_ROOT / "formal_analysis/formal_summary.json"),
        "offset_payload_sha256": offset_payload["offset_payload_sha256"],
        "e0_loco_metrics": e0_summary,
        "e0_loco_r5_diagnostics": e0_r5,
        "e1_post_outcome_metrics": e1_summary,
        "e1_post_outcome_r5_diagnostics": e1_r5,
        "restriction": (
            "rep_04..06 were already unblinded before CTS5 was proposed. They are "
            "E1 diagnostics only and can never establish CTS5 confirmation."
        ),
        "next_required_stage": "freeze implementation and capture fresh rep_07..11",
    }
    write_json_exclusive(partial / "development_report.json", report)
    write_json_exclusive(partial / "artifact_hashes.json", {
        "schema_version": 1,
        "status": "complete",
        "sha256": {
            name: sha256_file(partial / name) for name in (
                "e0_loco_predictions.csv",
                "e1_post_outcome_diagnostic_predictions.csv",
                "cts5_offsets.json",
                "development_report.json",
            )
        },
    })
    os.replace(partial, OUTPUT)
    print("CTS5 development complete")
    print(f"  offsets payload: {offset_payload['offset_payload_sha256']}")
    print(f"  output: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
