"""Develop static multi-camera weights for right-arm limb-length estimation.

This is the E0 (excluded retrospective development) implementation for
``docs/multiview_weighted_limb_length_optimization_protocol_draft_v1.md``.
It deliberately separates three operations:

1. select a same-render-step-capable camera subset using measured coverage
   only (no Ground Truth);
2. learn non-negative, sum-to-one camera weights with a deterministic convex
   quadratic programme and leave-one-capture-out selection of regularisation;
3. compare frozen fold weights with paired baselines against synchronized
   Isaac skeleton-pivot segment lengths.

Only ``raw_measured`` rows are eligible.  K2/K3/K4 rows are never accepted as
primary length inputs because their calibrated/inferred bone geometry would
make the endpoint circular.  Existing captures are development evidence only;
this tool does not turn them into a fresh confirmatory experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/multiview_weighted_limb_length_development_v1.json"
DEFAULT_OUTPUT = (
    ROOT
    / "output/experiments/kinematic_constraints_ablation"
    / "multiview_weighted_limb_length_v1/development"
)


class DevelopmentError(RuntimeError):
    """Raised when the development contract or an input invariant fails."""


@dataclass(frozen=True)
class JoinToken:
    cycle_index: int
    animation_frame_code: int
    usd_time_code: Decimal


@dataclass
class EstimateStream:
    points: dict[tuple[int, str], np.ndarray | None]
    tokens: dict[int, JoinToken]
    raw_row_count: int
    valid_point_count: int


@dataclass
class GroundTruthStream:
    points: dict[tuple[int, str], np.ndarray]
    tokens: dict[int, JoinToken]
    row_count: int
    unit_factors: set[float]
    semantics: set[str]


@dataclass(frozen=True)
class LengthSample:
    repeat: str
    sequence_index: int
    usd_time_code: str
    phase: str
    bone: str
    gt_length_mm: float
    camera_lengths_mm: tuple[float, ...]


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


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DevelopmentError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DevelopmentError(f"Expected a JSON object: {path}")
    return payload


def write_json_exclusive(path: Path, payload: Any) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv_exclusive(path: Path, rows: Iterable[dict[str, Any]],
                        fieldnames: list[str]) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def finite_point(row: dict[str, str], fields: tuple[str, str, str]) -> np.ndarray:
    try:
        point = np.asarray([float(row[field]) for field in fields], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise DevelopmentError(f"Invalid 3D point fields in row: {row}") from error
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise DevelopmentError(f"Non-finite 3D point in row: {row}")
    return point


def join_token(row: dict[str, str]) -> JoinToken:
    try:
        return JoinToken(
            cycle_index=int(row["cycle_index"]),
            animation_frame_code=int(row["animation_frame_code"]),
            usd_time_code=Decimal(row["usd_time_code"]),
        )
    except (KeyError, ValueError) as error:
        raise DevelopmentError(f"Invalid exact-join fields in row: {row}") from error


def register_token(tokens: dict[int, JoinToken], sequence: int,
                   token: JoinToken, path: Path) -> None:
    prior = tokens.get(sequence)
    if prior is not None and prior != token:
        raise DevelopmentError(
            f"Rows disagree on exact-join token at sequence {sequence} in {path}: "
            f"{prior} != {token}"
        )
    tokens[sequence] = token


def load_estimate_stream(path: Path, contract: dict[str, Any]) -> EstimateStream:
    if not path.is_file():
        raise DevelopmentError(f"Missing estimate table: {path}")
    target_method = str(contract["method"])
    points: dict[tuple[int, str], np.ndarray | None] = {}
    tokens: dict[int, JoinToken] = {}
    raw_rows = 0
    valid_points = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != target_method:
                continue
            raw_rows += 1
            try:
                sequence = int(row["sequence_index"])
                joint = row["canonical_joint"]
            except (KeyError, ValueError) as error:
                raise DevelopmentError(f"Malformed measured row in {path}: {row}") from error
            key = (sequence, joint)
            if key in points:
                raise DevelopmentError(f"Duplicate measured key {key} in {path}")
            register_token(tokens, sequence, join_token(row), path)
            valid = (
                row.get("valid") == str(contract["valid"])
                and row.get("counts_as_measured_valid")
                == str(contract["counts_as_measured_valid"])
            )
            if valid:
                points[key] = finite_point(row, ("x_m", "y_m", "z_m"))
                valid_points += 1
            else:
                points[key] = None
    if raw_rows == 0:
        raise DevelopmentError(f"No {target_method!r} rows in {path}")
    return EstimateStream(points, tokens, raw_rows, valid_points)


def load_ground_truth_stream(path: Path) -> GroundTruthStream:
    if not path.is_file():
        raise DevelopmentError(f"Missing Ground Truth table: {path}")
    points: dict[tuple[int, str], np.ndarray] = {}
    tokens: dict[int, JoinToken] = {}
    unit_factors: set[float] = set()
    semantics: set[str] = set()
    rows = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            try:
                sequence = int(row["sequence_index"])
                joint = row["canonical_joint"]
            except (KeyError, ValueError) as error:
                raise DevelopmentError(f"Malformed GT row in {path}: {row}") from error
            key = (sequence, joint)
            if key in points:
                raise DevelopmentError(f"Duplicate GT key {key} in {path}")
            register_token(tokens, sequence, join_token(row), path)
            points[key] = finite_point(row, ("gt_x_m", "gt_y_m", "gt_z_m"))
            if row.get("operational_unit_correction_factor", "") != "":
                unit_factors.add(float(row["operational_unit_correction_factor"]))
            if row.get("gt_semantics"):
                semantics.add(row["gt_semantics"])
    if not points:
        raise DevelopmentError(f"Ground Truth table is empty: {path}")
    return GroundTruthStream(points, tokens, rows, unit_factors, semantics)


def point_is_valid(stream: EstimateStream, sequence: int, joint: str) -> bool:
    return stream.points.get((sequence, joint)) is not None


def euclidean_length_mm(first: np.ndarray, second: np.ndarray) -> float:
    value = float(np.linalg.norm(np.asarray(first, dtype=float)
                                 - np.asarray(second, dtype=float)) * 1000.0)
    if not math.isfinite(value):
        raise DevelopmentError("A non-finite segment length was produced.")
    return value


def active_frame_set(ranges: list[list[int]]) -> set[int]:
    active: set[int] = set()
    for pair in ranges:
        if len(pair) != 2 or int(pair[1]) < int(pair[0]):
            raise DevelopmentError(f"Invalid active frame range: {pair}")
        active.update(range(int(pair[0]), int(pair[1]) + 1))
    return active


def evaluation_sequences(config: dict[str, Any]) -> list[int]:
    block = config["frame_contract"]
    start, end = (int(value) for value in block["evaluation_sequence_range_inclusive"])
    excluded = {int(value) for value in block["settling_excluded_sequence_indices"]}
    sequences = [value for value in range(start, end + 1) if value not in excluded]
    if not sequences:
        raise DevelopmentError("No evaluation sequence remains after exclusions.")
    return sequences


def estimate_path(root: Path, repeat: str, view: str) -> Path:
    return root / repeat / f"view_{view}" / "replay_method_v2/dynamic_kinematic_estimates.csv"


def gt_path(root: Path, repeat: str, view: str) -> Path:
    return root / repeat / f"view_{view}" / "ground_truth_joints.csv"


def camera_coverage(streams: dict[tuple[str, str], EstimateStream],
                    repeats: list[str], views: list[str], sequences: list[int],
                    required_joints: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    summary: dict[str, dict[str, Any]] = {}
    selected: list[str] = []
    for view in views:
        per_repeat: dict[str, Any] = {}
        for repeat in repeats:
            stream = streams[(repeat, view)]
            complete = sum(
                all(point_is_valid(stream, sequence, joint) for joint in required_joints)
                for sequence in sequences
            )
            per_repeat[repeat] = {
                "complete_frames": complete,
                "expected_frames": len(sequences),
                "coverage": complete / len(sequences),
            }
        summary[view] = {"per_repeat": per_repeat}
    return summary, selected


def eligible_views(coverage: dict[str, dict[str, Any]], minimum: float,
                   tolerance: float = 1e-12) -> list[str]:
    return [
        view
        for view, block in coverage.items()
        if all(float(value["coverage"]) + tolerance >= minimum
               for value in block["per_repeat"].values())
    ]


def assert_join_tokens(estimate: EstimateStream, truth: GroundTruthStream,
                       sequences: Iterable[int], label: str) -> None:
    for sequence in sequences:
        if sequence not in estimate.tokens:
            raise DevelopmentError(f"Estimate lacks sequence {sequence}: {label}")
        if sequence not in truth.tokens:
            raise DevelopmentError(f"GT lacks sequence {sequence}: {label}")
        if estimate.tokens[sequence] != truth.tokens[sequence]:
            raise DevelopmentError(
                f"Exact join failed for {label}, sequence {sequence}: "
                f"{estimate.tokens[sequence]} != {truth.tokens[sequence]}"
            )


def validate_gt_consistency(
    truths: dict[tuple[str, str], GroundTruthStream],
    repeats: list[str],
    views: list[str],
    reference_view: str,
    sequences: list[int],
    bones: dict[str, list[str]],
) -> dict[str, Any]:
    maximum_delta_m = 0.0
    lengths_by_bone: dict[str, list[float]] = defaultdict(list)
    factors: set[float] = set()
    semantics: set[str] = set()
    for repeat in repeats:
        reference = truths[(repeat, reference_view)]
        factors.update(reference.unit_factors)
        semantics.update(reference.semantics)
        for sequence in sequences:
            token = reference.tokens[sequence]
            for view in views:
                other = truths[(repeat, view)]
                if other.tokens.get(sequence) != token:
                    raise DevelopmentError(
                        f"GT exact token mismatch: {repeat} {view} sequence {sequence}"
                    )
            for bone, endpoints in bones.items():
                first, second = endpoints
                reference_length_m = float(np.linalg.norm(
                    reference.points[(sequence, first)]
                    - reference.points[(sequence, second)]
                ))
                lengths_by_bone[bone].append(reference_length_m * 1000.0)
                for view in views:
                    other = truths[(repeat, view)]
                    other_length_m = float(np.linalg.norm(
                        other.points[(sequence, first)]
                        - other.points[(sequence, second)]
                    ))
                    maximum_delta_m = max(
                        maximum_delta_m, abs(other_length_m - reference_length_m)
                    )
                    factors.update(other.unit_factors)
                    semantics.update(other.semantics)
    bone_summary = {
        bone: {
            "mean_mm": statistics.fmean(values),
            "min_mm": min(values),
            "max_mm": max(values),
            "range_mm": max(values) - min(values),
        }
        for bone, values in lengths_by_bone.items()
    }
    return {
        "maximum_cross_camera_length_delta_m": maximum_delta_m,
        "threshold_m": 1e-6,
        "passed": maximum_delta_m <= 1e-6,
        "operational_unit_correction_factors": sorted(factors),
        "gt_semantics": sorted(semantics),
        "per_bone": bone_summary,
    }


def build_length_samples(
    streams: dict[tuple[str, str], EstimateStream],
    truths: dict[tuple[str, str], GroundTruthStream],
    repeats: list[str],
    selected_views: list[str],
    reference_view: str,
    sequences: list[int],
    bones: dict[str, list[str]],
    active: set[int],
) -> list[LengthSample]:
    samples: list[LengthSample] = []
    for repeat in repeats:
        reference = truths[(repeat, reference_view)]
        for view in selected_views:
            assert_join_tokens(streams[(repeat, view)], truths[(repeat, view)],
                               sequences, f"{repeat}/{view}")
        for sequence in sequences:
            for bone, endpoints in bones.items():
                first_joint, second_joint = endpoints
                reference_first = reference.points.get((sequence, first_joint))
                reference_second = reference.points.get((sequence, second_joint))
                if reference_first is None or reference_second is None:
                    raise DevelopmentError(
                        f"Reference GT lacks {bone} at {repeat} sequence {sequence}"
                    )
                gt_length = euclidean_length_mm(reference_first, reference_second)
                lengths: list[float] = []
                for view in selected_views:
                    stream = streams[(repeat, view)]
                    first = stream.points.get((sequence, first_joint))
                    second = stream.points.get((sequence, second_joint))
                    if first is None or second is None:
                        raise DevelopmentError(
                            f"Selected view {view} is incomplete for {bone} at "
                            f"{repeat} sequence {sequence}"
                        )
                    lengths.append(euclidean_length_mm(first, second))
                samples.append(LengthSample(
                    repeat=repeat,
                    sequence_index=sequence,
                    usd_time_code=str(reference.tokens[sequence].usd_time_code),
                    phase="active" if sequence in active else "inactive",
                    bone=bone,
                    gt_length_mm=gt_length,
                    camera_lengths_mm=tuple(lengths),
                ))
    return samples


def balanced_sample_weights(samples: list[LengthSample], repeats: set[str]) -> np.ndarray:
    selected = [sample for sample in samples if sample.repeat in repeats]
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, sample in enumerate(selected):
        groups[(sample.repeat, sample.phase, sample.bone)].append(index)
    expected = {
        (repeat, phase, bone)
        for repeat in repeats
        for phase in ("active", "inactive")
        for bone in sorted({sample.bone for sample in selected})
    }
    if set(groups) != expected:
        raise DevelopmentError(
            f"Balanced groups differ from the complete contract: "
            f"missing={sorted(expected - set(groups))}"
        )
    weights = np.zeros(len(selected), dtype=float)
    group_mass = 1.0 / len(groups)
    for indices in groups.values():
        for index in indices:
            weights[index] = group_mass / len(indices)
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise DevelopmentError("Balanced sample weights do not sum to one.")
    return weights


def quadratic_terms(samples: list[LengthSample], train_repeats: set[str],
                    regularisation: float) -> tuple[np.ndarray, np.ndarray]:
    selected = [sample for sample in samples if sample.repeat in train_repeats]
    if not selected:
        raise DevelopmentError("The optimizer received no training samples.")
    weights = balanced_sample_weights(samples, train_repeats)
    matrix = np.asarray([sample.camera_lengths_mm for sample in selected], dtype=float)
    target = np.asarray([sample.gt_length_mm for sample in selected], dtype=float)
    camera_count = matrix.shape[1]
    uniform = np.full(camera_count, 1.0 / camera_count, dtype=float)
    q_matrix = matrix.T @ (weights[:, None] * matrix)
    linear = matrix.T @ (weights * target)
    q_matrix += float(regularisation) * np.eye(camera_count)
    linear += float(regularisation) * uniform
    return q_matrix, linear


def solve_box_simplex_qp(q_matrix: np.ndarray, linear: np.ndarray,
                         lower: float, upper: float,
                         tolerance: float = 1e-9) -> dict[str, Any]:
    """Solve ``min w'Qw - 2b'w`` over a box-constrained simplex.

    With five registered cameras, enumerating the lower/free/upper state of
    each variable requires only ``3**5 == 243`` KKT systems.  This avoids a
    dependency on a platform-specific external QP solver and is deterministic.
    """

    q_matrix = np.asarray(q_matrix, dtype=float)
    linear = np.asarray(linear, dtype=float)
    if q_matrix.ndim != 2 or q_matrix.shape[0] != q_matrix.shape[1]:
        raise DevelopmentError("Q must be square.")
    count = q_matrix.shape[0]
    if linear.shape != (count,):
        raise DevelopmentError("The QP linear term has the wrong shape.")
    if not np.all(np.isfinite(q_matrix)) or not np.all(np.isfinite(linear)):
        raise DevelopmentError("QP inputs must be finite.")
    if lower < 0.0 or upper > 1.0 or lower >= upper:
        raise DevelopmentError("Invalid box limits for simplex weights.")
    best_weights: np.ndarray | None = None
    best_objective = math.inf
    best_pattern: tuple[int, ...] | None = None
    feasible_patterns = 0
    for pattern in itertools.product((-1, 0, 1), repeat=count):
        free = [index for index, state in enumerate(pattern) if state == 0]
        if not free:
            continue
        fixed = [index for index, state in enumerate(pattern) if state != 0]
        weights = np.zeros(count, dtype=float)
        for index in fixed:
            weights[index] = lower if pattern[index] == -1 else upper
        remaining = 1.0 - float(weights[fixed].sum())
        if (remaining < len(free) * lower - tolerance
                or remaining > len(free) * upper + tolerance):
            continue
        q_ff = q_matrix[np.ix_(free, free)]
        rhs_free = linear[free].copy()
        if fixed:
            rhs_free -= q_matrix[np.ix_(free, fixed)] @ weights[fixed]
        kkt = np.block([
            [q_ff, np.ones((len(free), 1), dtype=float)],
            [np.ones((1, len(free)), dtype=float), np.zeros((1, 1), dtype=float)],
        ])
        rhs = np.concatenate([rhs_free, np.asarray([remaining], dtype=float)])
        try:
            solution = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            solution, residuals, rank, _ = np.linalg.lstsq(kkt, rhs, rcond=None)
            if rank < kkt.shape[0] or (residuals.size and float(residuals.max()) > 1e-12):
                continue
        candidate = solution[:-1]
        if np.any(candidate < lower - tolerance) or np.any(candidate > upper + tolerance):
            continue
        weights[free] = np.clip(candidate, lower, upper)
        if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0,
                            abs_tol=max(tolerance, 1e-10)):
            continue
        feasible_patterns += 1
        objective = float(weights @ q_matrix @ weights - 2.0 * linear @ weights)
        if objective < best_objective - 1e-10:
            best_objective = objective
            best_weights = weights.copy()
            best_pattern = pattern
    if best_weights is None or best_pattern is None:
        raise DevelopmentError("The box-simplex QP had no feasible active set.")
    constraint_residual = abs(float(best_weights.sum()) - 1.0)
    return {
        "weights": best_weights,
        "objective_without_constant": best_objective,
        "active_set_pattern": list(best_pattern),
        "feasible_active_set_count": feasible_patterns,
        "constraint_residual": constraint_residual,
        "minimum_weight": float(best_weights.min()),
        "maximum_weight": float(best_weights.max()),
    }


def fit_weights(samples: list[LengthSample], repeats: Iterable[str],
                regularisation: float, optimizer: dict[str, Any]) -> dict[str, Any]:
    q_matrix, linear = quadratic_terms(samples, set(repeats), regularisation)
    solution = solve_box_simplex_qp(
        q_matrix,
        linear,
        float(optimizer["weight_lower_bound"]),
        float(optimizer["weight_upper_bound"]),
        float(optimizer["constraint_tolerance"]),
    )
    solution["regularisation_lambda_mm2"] = float(regularisation)
    return solution


def prediction_for_weights(sample: LengthSample, weights: np.ndarray) -> float:
    return float(np.asarray(sample.camera_lengths_mm, dtype=float) @ weights)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median_absolute_deviation(values: list[float]) -> float | None:
    if not values:
        return None
    centre = statistics.median(values)
    return statistics.median(abs(value - centre) for value in values)


def pbmae_for_predictor(samples: list[LengthSample], repeats: set[str],
                        predictor: Callable[[LengthSample], float]) -> dict[str, float]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample in samples:
        if sample.repeat not in repeats:
            continue
        groups[(sample.repeat, sample.phase, sample.bone)].append(
            abs(predictor(sample) - sample.gt_length_mm)
        )
    result: dict[str, float] = {}
    for repeat in sorted(repeats):
        values = [
            statistics.fmean(errors)
            for (group_repeat, _phase, _bone), errors in groups.items()
            if group_repeat == repeat
        ]
        if len(values) != 4:
            raise DevelopmentError(
                f"Expected four phase/bone cells for {repeat}, found {len(values)}"
            )
        result[repeat] = statistics.fmean(values)
    return result


def mean_and_sample_sd(values: list[float]) -> dict[str, float]:
    if not values:
        raise DevelopmentError("Cannot summarise an empty list.")
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "standard_error": (
            statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        ),
    }


def select_regularisation(samples: list[LengthSample], repeats: list[str],
                          grid: list[float], optimizer: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for regularisation in grid:
        fold_scores: dict[str, float] = {}
        fold_weights: dict[str, list[float]] = {}
        for held_out in repeats:
            train = [repeat for repeat in repeats if repeat != held_out]
            solution = fit_weights(samples, train, regularisation, optimizer)
            weights = np.asarray(solution["weights"], dtype=float)
            score = pbmae_for_predictor(
                samples, {held_out}, lambda sample, w=weights: prediction_for_weights(sample, w)
            )[held_out]
            fold_scores[held_out] = score
            fold_weights[held_out] = weights.tolist()
        summary = mean_and_sample_sd(list(fold_scores.values()))
        candidates.append({
            "lambda_mm2": float(regularisation),
            "fold_pbmae_mm": fold_scores,
            "fold_weights": fold_weights,
            **summary,
        })
    best = min(candidates, key=lambda row: (float(row["mean"]), float(row["lambda_mm2"])))
    threshold = float(best["mean"]) + float(best["standard_error"])
    eligible = [row for row in candidates if float(row["mean"]) <= threshold + 1e-12]
    selected = max(eligible, key=lambda row: float(row["lambda_mm2"]))
    return {
        "candidates": candidates,
        "minimum_mean_candidate_lambda_mm2": best["lambda_mm2"],
        "minimum_mean_pbmae_mm": best["mean"],
        "one_standard_error_threshold_mm": threshold,
        "selected_lambda_mm2": selected["lambda_mm2"],
        "selection_rule": optimizer["lambda_selection"],
    }


def camera_training_pbmae(samples: list[LengthSample], train_repeats: set[str],
                           camera_index: int) -> float:
    values = pbmae_for_predictor(
        samples,
        train_repeats,
        lambda sample: float(sample.camera_lengths_mm[camera_index]),
    )
    return statistics.fmean(values.values())


def inverse_error_weights(samples: list[LengthSample], train_repeats: set[str],
                          camera_count: int) -> np.ndarray:
    errors = np.asarray([
        camera_training_pbmae(samples, train_repeats, index)
        for index in range(camera_count)
    ], dtype=float)
    inverse = 1.0 / np.maximum(errors, 1e-9)
    return inverse / inverse.sum()


def crossfit(
    samples: list[LengthSample],
    repeats: list[str],
    views: list[str],
    selected_lambda: float,
    optimizer: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    fold_details: dict[str, Any] = {}
    camera_count = len(views)
    uniform = np.full(camera_count, 1.0 / camera_count, dtype=float)
    for held_out in repeats:
        train = {repeat for repeat in repeats if repeat != held_out}
        optimized_solution = fit_weights(samples, train, selected_lambda, optimizer)
        optimized = np.asarray(optimized_solution["weights"], dtype=float)
        inverse = inverse_error_weights(samples, train, camera_count)
        single_scores = [camera_training_pbmae(samples, train, index)
                         for index in range(camera_count)]
        best_index = min(range(camera_count), key=lambda index: (single_scores[index], views[index]))
        method_weights = {
            "optimized_static_shared_weights": optimized,
            "uniform_selected_view_mean": uniform,
            "training_inverse_pbmae_weighting": inverse,
        }
        for method, weights in method_weights.items():
            weight_hash = payload_sha256({
                "held_out_repeat": held_out,
                "method": method,
                "views": views,
                "weights": weights.tolist(),
            })
            for view, value in zip(views, weights):
                weight_rows.append({
                    "held_out_repeat": held_out,
                    "method": method,
                    "view_id": view,
                    "weight": float(value),
                    "weight_hash": weight_hash,
                })
        fold_details[held_out] = {
            "training_repeats": sorted(train),
            "optimized_weights": dict(zip(views, optimized.tolist())),
            "optimized_solution": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in optimized_solution.items()
                if key != "weights"
            },
            "inverse_weights": dict(zip(views, inverse.tolist())),
            "training_single_camera_pbmae_mm": dict(zip(views, single_scores)),
            "training_selected_best_single": views[best_index],
        }
        for sample in samples:
            if sample.repeat != held_out:
                continue
            camera_values = np.asarray(sample.camera_lengths_mm, dtype=float)
            methods = {
                "optimized_static_shared_weights": float(camera_values @ optimized),
                "uniform_selected_view_mean": float(camera_values @ uniform),
                "selected_view_median": float(statistics.median(camera_values.tolist())),
                "training_inverse_pbmae_weighting": float(camera_values @ inverse),
                "training_selected_best_single": float(camera_values[best_index]),
                "gt_oracle_per_frame_best_view": float(
                    camera_values[np.argmin(np.abs(camera_values - sample.gt_length_mm))]
                ),
            }
            for method, estimate in methods.items():
                error = estimate - sample.gt_length_mm
                if method in method_weights:
                    weights = method_weights[method]
                elif method == "training_selected_best_single":
                    weights = np.eye(camera_count, dtype=float)[best_index]
                else:
                    weights = None
                predictions.append({
                    "repeat": sample.repeat,
                    "sequence_index": sample.sequence_index,
                    "usd_time_code": sample.usd_time_code,
                    "phase": sample.phase,
                    "bone": sample.bone,
                    "method": method,
                    "estimated_length_mm": estimate,
                    "gt_length_mm": sample.gt_length_mm,
                    "signed_error_mm": error,
                    "absolute_error_mm": abs(error),
                    "selected_views": "|".join(views),
                    "training_repeats": "|".join(sorted(train)),
                    "best_single_view": views[best_index]
                    if method == "training_selected_best_single" else "",
                    "weights_json": json.dumps(weights.tolist(), separators=(",", ":"))
                    if weights is not None else "",
                })
    return predictions, weight_rows, fold_details


def summarise_predictions(predictions: list[dict[str, Any]],
                          repeats: list[str]) -> dict[str, Any]:
    methods = sorted({row["method"] for row in predictions})
    output: dict[str, Any] = {}
    for method in methods:
        per_run: dict[str, Any] = {}
        for repeat in repeats:
            rows = [row for row in predictions
                    if row["method"] == method and row["repeat"] == repeat]
            if not rows:
                raise DevelopmentError(f"No prediction rows for {method}/{repeat}")
            cells: dict[str, Any] = {}
            cell_mae: list[float] = []
            for phase in ("active", "inactive"):
                for bone in sorted({str(row["bone"]) for row in rows}):
                    cell_rows = [row for row in rows
                                 if row["phase"] == phase and row["bone"] == bone]
                    errors = [float(row["signed_error_mm"]) for row in cell_rows]
                    absolute = [abs(value) for value in errors]
                    key = f"{bone}/{phase}"
                    cell_mae.append(statistics.fmean(absolute))
                    cells[key] = {
                        "samples": len(cell_rows),
                        "mae_mm": statistics.fmean(absolute),
                        "bias_mm": statistics.fmean(errors),
                        "rmse_mm": math.sqrt(statistics.fmean(value * value for value in errors)),
                        "p95_absolute_error_mm": percentile(absolute, 0.95),
                        "error_mad_mm": median_absolute_deviation(errors),
                    }
            all_errors = [float(row["signed_error_mm"]) for row in rows]
            all_absolute = [abs(value) for value in all_errors]
            bone_bias: dict[str, float] = {}
            for bone in sorted({str(row["bone"]) for row in rows}):
                bone_errors = [float(row["signed_error_mm"])
                               for row in rows if row["bone"] == bone]
                bone_bias[bone] = statistics.fmean(bone_errors)
            per_run[repeat] = {
                "right_arm_two_bone_phase_balanced_length_MAE_mm":
                    statistics.fmean(cell_mae),
                "pooled_bias_mm": statistics.fmean(all_errors),
                "pooled_rmse_mm": math.sqrt(
                    statistics.fmean(value * value for value in all_errors)
                ),
                "pooled_p95_absolute_error_mm": percentile(all_absolute, 0.95),
                "pooled_error_mad_mm": median_absolute_deviation(all_errors),
                "paired_sample_count": len(rows),
                "complete_five_view_coverage": 1.0,
                "per_bone_bias_mm": bone_bias,
                "per_bone_phase": cells,
            }
        primary = [
            float(per_run[repeat]["right_arm_two_bone_phase_balanced_length_MAE_mm"])
            for repeat in repeats
        ]
        output[method] = {
            "per_run": per_run,
            "run_level_primary": mean_and_sample_sd(primary),
        }
    return output


def compare_methods(summary: dict[str, Any], repeats: list[str]) -> dict[str, Any]:
    optimized = summary["optimized_static_shared_weights"]["per_run"]
    uniform = summary["uniform_selected_view_mean"]["per_run"]
    best = summary["training_selected_best_single"]["per_run"]
    rows = {}
    for repeat in repeats:
        key = "right_arm_two_bone_phase_balanced_length_MAE_mm"
        opt = float(optimized[repeat][key])
        uni = float(uniform[repeat][key])
        single = float(best[repeat][key])
        rows[repeat] = {
            "optimized_mm": opt,
            "uniform_mm": uni,
            "best_single_mm": single,
            "optimized_to_uniform_ratio": opt / uni,
            "optimized_minus_uniform_mm": opt - uni,
            "optimized_to_best_single_ratio": opt / single,
            "optimized_minus_best_single_mm": opt - single,
        }
    return {
        "per_run": rows,
        "capture_balanced_mean_optimized_minus_uniform_mm": statistics.fmean(
            value["optimized_minus_uniform_mm"] for value in rows.values()
        ),
        "capture_balanced_mean_optimized_minus_best_single_mm": statistics.fmean(
            value["optimized_minus_best_single_mm"] for value in rows.values()
        ),
    }


def draft_threshold_diagnostics(comparison: dict[str, Any],
                                thresholds: dict[str, Any]) -> dict[str, Any]:
    per_run = comparison["per_run"]
    uniform_ratio = float(thresholds["optimized_to_uniform_ratio_max_each_run"])
    uniform_delta = float(thresholds["optimized_minus_uniform_mean_max_mm"])
    single_ratio = float(thresholds["optimized_to_best_single_ratio_max_each_run"])
    single_delta = float(thresholds["optimized_minus_best_single_mean_max_mm"])
    return {
        "classification": "development_only_not_a_formal_verdict",
        "uniform": {
            "per_run_ratio_pass": {
                repeat: value["optimized_to_uniform_ratio"] <= uniform_ratio
                for repeat, value in per_run.items()
            },
            "mean_absolute_delta_pass":
                comparison["capture_balanced_mean_optimized_minus_uniform_mm"]
                <= uniform_delta,
            "passed": (
                all(value["optimized_to_uniform_ratio"] <= uniform_ratio
                    for value in per_run.values())
                and comparison["capture_balanced_mean_optimized_minus_uniform_mm"]
                <= uniform_delta
            ),
        },
        "best_single": {
            "per_run_ratio_pass": {
                repeat: value["optimized_to_best_single_ratio"] <= single_ratio
                for repeat, value in per_run.items()
            },
            "mean_absolute_delta_pass":
                comparison["capture_balanced_mean_optimized_minus_best_single_mm"]
                <= single_delta,
            "passed": (
                all(value["optimized_to_best_single_ratio"] <= single_ratio
                    for value in per_run.values())
                and comparison["capture_balanced_mean_optimized_minus_best_single_mm"]
                <= single_delta
            ),
        },
        "note": thresholds["note"],
    }


def manifest_entry(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "role": role,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_input_manifest(config_path: Path, offline_root: Path, unit_root: Path,
                         config: dict[str, Any], script_path: Path,
                         selected_views: list[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = [
        manifest_entry(config_path, "development_config"),
        manifest_entry(script_path, "development_tool"),
        manifest_entry(ROOT / "configs/joint_mapping.csv", "joint_mapping"),
    ]
    for repeat in config["repeats"]:
        for view in config["candidate_views"]:
            files.append(manifest_entry(
                estimate_path(offline_root, repeat, view),
                "candidate_raw_measured_estimates",
            ))
        gt_views = sorted(set(selected_views) | {config["reference_gt_view"]})
        for view in gt_views:
            files.append(manifest_entry(
                gt_path(unit_root, repeat, view),
                "selected_or_reference_ground_truth",
            ))
    return {
        "schema_version": 1,
        "status": "frozen_for_excluded_development_run",
        "classification": config["classification"],
        "file_count": len(files),
        "files": files,
        "content_manifest_sha256": payload_sha256(files),
    }


def camera_sample_rows(samples: list[LengthSample], views: list[str]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        row: dict[str, Any] = {
            "repeat": sample.repeat,
            "sequence_index": sample.sequence_index,
            "usd_time_code": sample.usd_time_code,
            "phase": sample.phase,
            "bone": sample.bone,
            "gt_length_mm": sample.gt_length_mm,
        }
        for view, value in zip(views, sample.camera_lengths_mm):
            row[f"{view}_length_mm"] = value
        rows.append(row)
    return rows


def validate_config(config: dict[str, Any]) -> None:
    if config.get("classification") != "excluded_retrospective_method_development":
        raise DevelopmentError("This tool only runs the excluded E0 development protocol.")
    repeats = config.get("repeats")
    if not isinstance(repeats, list) or len(repeats) < 3 or len(set(repeats)) != len(repeats):
        raise DevelopmentError("At least three unique capture repeats are required.")
    views = config.get("candidate_views")
    if not isinstance(views, list) or len(views) < 2 or len(set(views)) != len(views):
        raise DevelopmentError("At least two unique candidate views are required.")
    bones = config.get("primary_bones")
    if not isinstance(bones, dict) or set(bones) != {"right_upper_arm", "right_forearm"}:
        raise DevelopmentError("The primary endpoint must contain exactly the two right-arm bones.")
    for endpoints in bones.values():
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise DevelopmentError(f"Invalid bone endpoints: {endpoints}")
    grid = [float(value) for value in config["optimizer"]["lambda_grid_mm2"]]
    if not grid or any(value <= 0.0 or not math.isfinite(value) for value in grid):
        raise DevelopmentError("All regularisation candidates must be finite and positive.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--offline-root", type=Path, default=None)
    parser.add_argument("--unit-corrected-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_config(config)
    offline_root = (args.offline_root or Path(config["data_roots"]["offline_root"])).resolve()
    unit_root = (
        args.unit_corrected_root
        or Path(config["data_roots"]["unit_corrected_root"])
    ).resolve()
    output_dir = args.output_dir.resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists():
        raise DevelopmentError(f"Refusing to overwrite completed output: {output_dir}")
    if partial_dir.exists():
        raise DevelopmentError(f"Refusing to overwrite partial output: {partial_dir}")
    if not offline_root.is_dir() or not unit_root.is_dir():
        raise DevelopmentError(
            f"Input roots are unavailable: offline={offline_root}, unit={unit_root}"
        )

    started = time.time()
    repeats = [str(value) for value in config["repeats"]]
    candidates = [str(value) for value in config["candidate_views"]]
    sequences = evaluation_sequences(config)
    active = active_frame_set(config["frame_contract"]["active_frame_ranges_inclusive"])
    bones = {str(key): [str(value) for value in endpoints]
             for key, endpoints in config["primary_bones"].items()}
    contract = config["measured_input_contract"]

    streams: dict[tuple[str, str], EstimateStream] = {}
    for repeat in repeats:
        for view in candidates:
            streams[(repeat, view)] = load_estimate_stream(
                estimate_path(offline_root, repeat, view), contract
            )
    coverage, _ = camera_coverage(
        streams,
        repeats,
        candidates,
        sequences,
        [str(value) for value in config["camera_eligibility"]["required_joints"]],
    )
    selected = eligible_views(
        coverage,
        float(config["camera_eligibility"]["per_repeat_complete_coverage_min"]),
    )
    expected = [str(value) for value in config["camera_eligibility"]["expected_selected_views"]]
    if selected != expected:
        raise DevelopmentError(
            f"GT-free camera eligibility changed: selected={selected}, expected={expected}"
        )

    truth_views = sorted(set(selected) | {str(config["reference_gt_view"])})
    truths: dict[tuple[str, str], GroundTruthStream] = {}
    for repeat in repeats:
        for view in truth_views:
            truths[(repeat, view)] = load_ground_truth_stream(gt_path(unit_root, repeat, view))
    gt_consistency = validate_gt_consistency(
        truths,
        repeats,
        selected,
        str(config["reference_gt_view"]),
        sequences,
        bones,
    )
    if not gt_consistency["passed"]:
        raise DevelopmentError(
            "Cross-camera GT segment lengths exceed the 1e-6 m validity gate."
        )

    samples = build_length_samples(
        streams,
        truths,
        repeats,
        selected,
        str(config["reference_gt_view"]),
        sequences,
        bones,
        active,
    )
    expected_samples = len(repeats) * len(sequences) * len(bones)
    if len(samples) != expected_samples:
        raise DevelopmentError(
            f"Length sample count {len(samples)} != expected {expected_samples}"
        )

    optimizer = config["optimizer"]
    regularisation = select_regularisation(
        samples,
        repeats,
        [float(value) for value in optimizer["lambda_grid_mm2"]],
        optimizer,
    )
    selected_lambda = float(regularisation["selected_lambda_mm2"])
    predictions, weight_rows, fold_details = crossfit(
        samples, repeats, selected, selected_lambda, optimizer
    )
    prediction_summary = summarise_predictions(predictions, repeats)
    comparison = compare_methods(prediction_summary, repeats)
    diagnostics = draft_threshold_diagnostics(
        comparison, config["development_only_draft_thresholds"]
    )
    final_solution = fit_weights(samples, repeats, selected_lambda, optimizer)
    final_weights = np.asarray(final_solution.pop("weights"), dtype=float)
    if (
        final_weights.min() < float(optimizer["weight_lower_bound"]) - 1e-9
        or final_weights.max() > float(optimizer["weight_upper_bound"]) + 1e-9
        or abs(float(final_weights.sum()) - 1.0) > 1e-9
    ):
        raise DevelopmentError("The final development weights violate their contract.")

    input_manifest = build_input_manifest(
        config_path, offline_root, unit_root, config, Path(__file__).resolve(), selected
    )
    weights_payload = {
        "schema_version": 1,
        "status": "frozen_after_excluded_development",
        "classification": config["classification"],
        "selected_views": selected,
        "selected_lambda_mm2": selected_lambda,
        "weights": dict(zip(selected, final_weights.tolist())),
        "weight_vector": final_weights.tolist(),
        "weight_sum": float(final_weights.sum()),
        "maximum_weight": float(final_weights.max()),
        "effective_camera_number": float(1.0 / np.sum(final_weights ** 2)),
        "optimizer_solution": final_solution,
        "config_sha256": sha256_file(config_path),
        "input_manifest_content_sha256": input_manifest["content_manifest_sha256"],
        "restriction": (
            "Development weights only. They are eligible to be frozen into a future "
            "fresh-capture protocol, but the existing captures are not confirmatory."
        ),
    }
    weights_payload["weights_payload_sha256"] = payload_sha256(weights_payload)

    validity = {
        "candidate_estimate_file_count": len(repeats) * len(candidates),
        "selected_views_match_frozen_expectation": selected == expected,
        "selected_view_count": len(selected),
        "evaluation_frames_per_repeat": len(sequences),
        "length_sample_count": len(samples),
        "expected_length_sample_count": expected_samples,
        "gt_consistency": gt_consistency,
        "selected_camera_complete_coverage": {
            view: coverage[view] for view in selected
        },
        "all_checks_passed": True,
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "multiview_weighted_limb_length_e0_development",
        "classification": config["classification"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_manifest_content_sha256": input_manifest["content_manifest_sha256"],
        "statistical_unit": "independent Isaac capture; frames/cameras/bones are paired observations",
        "selected_views": selected,
        "candidate_coverage": coverage,
        "validity": validity,
        "regularisation_selection": regularisation,
        "crossfit_fold_details": fold_details,
        "final_development_weights": weights_payload,
        "crossfit_metrics": prediction_summary,
        "crossfit_comparison": comparison,
        "draft_threshold_diagnostics": diagnostics,
        "restrictions": [
            "Existing 3-run bank is excluded retrospective method-development evidence.",
            "All runs use one female character, one animation, one scene and one rendered occluder.",
            "The output does not establish fresh repeatability, held-out-pose transfer, real-human validity or multi-person performance.",
            "Isaac skeleton pivots and BlazePose visible-surface landmarks have different semantics.",
            "K2/K3/K4 rows were excluded from the primary length endpoint.",
        ],
        "runtime_seconds": time.time() - started,
    }

    partial_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()
    write_json_exclusive(partial_dir / "input_manifest.json", input_manifest)
    sample_fields = [
        "repeat", "sequence_index", "usd_time_code", "phase", "bone", "gt_length_mm",
        *[f"{view}_length_mm" for view in selected],
    ]
    write_csv_exclusive(
        partial_dir / "camera_length_samples.csv",
        camera_sample_rows(samples, selected),
        sample_fields,
    )
    prediction_fields = [
        "repeat", "sequence_index", "usd_time_code", "phase", "bone", "method",
        "estimated_length_mm", "gt_length_mm", "signed_error_mm", "absolute_error_mm",
        "selected_views", "training_repeats", "best_single_view", "weights_json",
    ]
    write_csv_exclusive(
        partial_dir / "crossfit_predictions.csv", predictions, prediction_fields
    )
    write_csv_exclusive(
        partial_dir / "crossfit_weights.csv",
        weight_rows,
        ["held_out_repeat", "method", "view_id", "weight", "weight_hash"],
    )
    lambda_rows = []
    for candidate in regularisation["candidates"]:
        row: dict[str, Any] = {
            "lambda_mm2": candidate["lambda_mm2"],
            "mean_pbmae_mm": candidate["mean"],
            "sample_sd_mm": candidate["sample_sd"],
            "standard_error_mm": candidate["standard_error"],
            "selected": int(candidate["lambda_mm2"] == selected_lambda),
        }
        for repeat in repeats:
            row[f"{repeat}_pbmae_mm"] = candidate["fold_pbmae_mm"][repeat]
        lambda_rows.append(row)
    write_csv_exclusive(
        partial_dir / "lambda_cross_validation.csv",
        lambda_rows,
        ["lambda_mm2", *[f"{repeat}_pbmae_mm" for repeat in repeats],
         "mean_pbmae_mm", "sample_sd_mm", "standard_error_mm", "selected"],
    )
    write_json_exclusive(partial_dir / "development_weights.json", weights_payload)
    write_json_exclusive(partial_dir / "development_report.json", report)
    write_json_exclusive(partial_dir / "development_state.json", {
        "schema_version": 1,
        "status": "complete",
        "classification": config["classification"],
        "started_wall_time_s": started,
        "completed_wall_time_s": time.time(),
        "output_directory": str(output_dir),
        "selected_views": selected,
        "selected_lambda_mm2": selected_lambda,
        "weights_payload_sha256": weights_payload["weights_payload_sha256"],
    })
    artifact_files = sorted(path for path in partial_dir.iterdir()
                            if path.name != "artifact_hashes.json")
    artifact_hashes = {
        "schema_version": 1,
        "status": "complete",
        "artifacts": [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_files
        ],
    }
    write_json_exclusive(partial_dir / "artifact_hashes.json", artifact_hashes)
    os.replace(partial_dir, output_dir)

    print("=" * 78)
    print("Multiview weighted limb length -- E0 excluded development complete")
    print("=" * 78)
    print("Selected views: {}".format(", ".join(selected)))
    print("Selected lambda: {:.6g} mm^2".format(selected_lambda))
    print("Final weights:")
    for view, value in zip(selected, final_weights):
        print("  {:>4s}: {:.9f}".format(view, value))
    print("Cross-fitted primary endpoint (mm):")
    for repeat, values in comparison["per_run"].items():
        print(
            "  {} optimized {:.3f}, uniform {:.3f}, best-single {:.3f}".format(
                repeat, values["optimized_mm"], values["uniform_mm"],
                values["best_single_mm"]
            )
        )
    print("Development-only draft diagnostics: uniform={}, best-single={}".format(
        diagnostics["uniform"]["passed"], diagnostics["best_single"]["passed"]
    ))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
