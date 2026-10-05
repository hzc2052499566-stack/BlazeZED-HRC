"""Repeat-robust 18->9->5 selector built on the frozen v7 binary scoring.

The selector accepts exactly three complete v7-compatible four-character
matrices.  A candidate ``(F, R)`` must independently satisfy every v7 hard gate
in all three repeats.  Data-derived ordering components use the least favourable
repeat; the two geometry components and the final lexical tie break are the v7
definitions unchanged.

Three leave-one-render-repeat-out (LORO) selections are stability gates.  The
bank and M0 must be exact.  F may change by at most one view, and a replacement
must come from the full-data reserve and be a legal v7 swap in each of the three
complete repeats.  Leave-one-character-out (LOCO) selections are reported only
as sensitivity analyses and never change the decision.

Only binary availability and occluder-attribution matrices are read.  The
module is pure Python and imports the existing pure-Python v7 implementation for
matrix validation, subset scoring and reserve semantics.
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping, Sequence

import select_common_bank_18_9_5_v7 as v7
from common_bank_rule_lock_v7 import (
    CHARACTERS,
    COVERAGE_FLOOR,
    COVERAGE_GATES,
    HISTORICAL_M0_VIEW,
    MIN_SWAPPABLE_SELECTED_VIEWS,
    RESERVE_COUNT,
    RULE_TAG as V7_RULE_TAG,
    SELECTED_COUNT,
    SURVEY_COUNT,
    min_pairwise_angle,
    pairwise_angle_table,
    provably_mandatory,
    reserve_gate_binds,
)


REPEAT_COUNT = 3
RULE_TAG = "fs_cts5_common_bank_18_9_5_selector_v8_repeat_robust"

# The eight numeric components retain the v7 positions.  Only the six
# data-derived components acquire a repeat-wise-worst reduction.  Geometry and
# the ninth, lexical key are deliberately unchanged.
ROBUST_ORDERING_KEYS = (
    "max_repeat_worst_min_measured_views",
    "min_repeat_worst_cells_at_that_minimum",
    "max_repeat_worst_min_view_character_bone_clean_rate",
    "max_min_pairwise_angle_in_f",
    "max_repeat_worst_min_over_f_of_best_swap_min_measured",
    "max_repeat_worst_valid_ordered_swap_pairs",
    "max_repeat_worst_passing_five_subsets_within_b",
    "max_min_pairwise_angle_in_b",
    "view_id_lexicographic",
)


class SelectionError(RuntimeError):
    """Raised when the three-repeat contract cannot be evaluated exactly."""


def pair_space_count(
    survey_count: int | None = None,
    selected_count: int | None = None,
    reserve_count: int | None = None,
) -> int:
    """Return ``C(N,F) * C(N-F,R)`` using current constants by default."""
    n = SURVEY_COUNT if survey_count is None else int(survey_count)
    f = SELECTED_COUNT if selected_count is None else int(selected_count)
    r = RESERVE_COUNT if reserve_count is None else int(reserve_count)
    if min(n, f, r) < 0 or f + r > n:
        raise SelectionError("Invalid survey, selected or reserve count.")
    return math.comb(n, f) * math.comb(n - f, r)


def _rank_key(item: Mapping) -> tuple:
    """Best first: numeric components descend, then F and R ascend."""
    return (
        tuple(-int(value) for value in item["score"]),
        tuple(item["F"]),
        tuple(item["R"]),
    )


def _repeat_label(matrix: Mapping, index: int) -> str:
    """Use supplied non-scoring provenance when available, otherwise position."""
    for key in ("repeat_id", "render_repeat_id", "capture_repeat_id"):
        value = matrix.get(key)
        if value not in (None, ""):
            return str(value)
    return "repeat_{:02d}".format(index + 1)


def _prepare_repeats(matrices: Sequence[Mapping], angle_table: Mapping) -> list:
    """Validate and score exactly three complete v7 matrices independently."""
    if isinstance(matrices, Mapping) or len(matrices) != REPEAT_COUNT:
        raise SelectionError("v8 requires exactly three complete repeat matrices.")
    prepared = []
    expected_views = None
    for index, matrix in enumerate(matrices):
        if not isinstance(matrix, Mapping):
            raise SelectionError("Repeat {} is not a matrix mapping.".format(index + 1))
        try:
            data = v7.load_matrix(matrix)
        except v7.SelectionError as error:
            raise SelectionError(
                "Repeat {} is not a complete v7-compatible matrix: {}".format(
                    index + 1, error
                )
            ) from error
        views = tuple(data["views"])
        if expected_views is None:
            expected_views = views
        elif views != expected_views:
            raise SelectionError("Repeat view order or membership differs.")
        scoring = v7.score_subsets(data, angle_table)
        prepared.append(
            {
                "index": index,
                "label": _repeat_label(matrix, index),
                "data": data,
                "scoring": scoring,
            }
        )
    return prepared


def _mandatory_views(scoring: Mapping) -> list:
    passing = [
        list(combination)
        for combination, entry in scoring["subsets"].items()
        if entry["passes"]
    ]
    return provably_mandatory(passing)


def _worst_clean_profile(scoring_repeats: Sequence[Mapping], views: Sequence[str]) -> dict:
    """Reduce both M0 clean ordering keys by their least favourable repeat."""
    profile = {}
    for view in views:
        per_repeat = [scoring["clean"][view] for scoring in scoring_repeats]
        profile[view] = {
            "min_character_bone_frames": min(
                int(item["min_character_bone_frames"]) for item in per_repeat
            ),
            "total_g0_measured_frames": min(
                int(item["total_g0_measured_frames"]) for item in per_repeat
            ),
            "g0_cell_frames": min(int(item["g0_cell_frames"]) for item in per_repeat),
            "repeat_values": [
                {
                    "min_character_bone_frames": int(
                        item["min_character_bone_frames"]
                    ),
                    "total_g0_measured_frames": int(
                        item["total_g0_measured_frames"]
                    ),
                    "g0_cell_frames": int(item["g0_cell_frames"]),
                }
                for item in per_repeat
            ],
        }
    return profile


def _m0_from_worst_clean(
    selected: Sequence[str],
    angle_table: Mapping,
    scoring_repeats: Sequence[Mapping],
    views: Sequence[str],
) -> dict:
    robust_clean = _worst_clean_profile(scoring_repeats, views)
    result = v7.m0_anchor(selected, angle_table, robust_clean)
    result["clean_reduction"] = "repeat_wise_worst"
    result["repeat_values"] = robust_clean[result["view"]]["repeat_values"]
    return result


def _first_f_failure(entries: Sequence[Mapping]) -> str | None:
    """Mutually exclusive reason for a subset that is not valid in every repeat."""
    if all(entry["passes"] for entry in entries):
        return None
    if any(not entry.get("gate_coverage", False) for entry in entries):
        return "f_coverage_not_all_repeats"
    if any(not entry.get("gate_recoverable_events", False) for entry in entries):
        return "f_recoverable_events_not_all_repeats"
    if any(not entry.get("gate_distinct_event_views", False) for entry in entries):
        return "f_distinct_event_views_not_all_repeats"
    return "f_not_all_repeats"


def _repeat_pair_metrics(swaps: Mapping, reserve: Sequence[str]) -> dict:
    """Evaluate one repeat's v7 reserve gate and its two data ordering values."""
    reserve_mask = 0
    for candidate in reserve:
        try:
            reserve_mask |= 1 << swaps["bit_of"][candidate]
        except KeyError as error:
            raise SelectionError("A reserve view is not outside F.") from error
    useful = 0
    for target in swaps["targets"].values():
        useful |= target["mask"]
    best_swap = []
    for target in swaps["targets"].values():
        if not target["mask"] & reserve_mask:
            return {"passes": False, "failure": "r_swap_coverage_not_all_repeats"}
        for minimum, bit in target["ranked"]:
            if bit & reserve_mask:
                best_swap.append(int(minimum))
                break
    if reserve_mask & ~useful:
        return {"passes": False, "failure": "r_member_without_swap_not_all_repeats"}
    if not best_swap:
        return {"passes": False, "failure": "r_gate_vacuous_not_all_repeats"}
    ordered_pairs = sum(
        (target["mask"] & reserve_mask).bit_count()
        for target in swaps["targets"].values()
    )
    return {
        "passes": True,
        "failure": None,
        "min_best_swap_measured_views": min(best_swap),
        "ordered_legal_swap_pairs": ordered_pairs,
    }


def _bank_metrics(
    scoring: Mapping,
    bank: tuple,
    angle_table: Mapping,
    cache: dict,
) -> tuple[int, int]:
    cached = cache.get(bank)
    if cached is None:
        try:
            passing = sum(
                1
                for subset in itertools.combinations(bank, SELECTED_COUNT)
                if scoring["subsets"][tuple(sorted(subset))]["passes"]
            )
        except KeyError as error:
            raise SelectionError("Subset scoring is incomplete for a bank.") from error
        cached = (passing, min_pairwise_angle(bank, angle_table))
        cache[bank] = cached
    return cached


def _core_select(
    scoring_repeats: Sequence[Mapping],
    angle_table: Mapping,
    views: Sequence[str],
    leaderboard_limit: int = 5,
) -> dict:
    """Exhaustively select one pair using any non-empty set of repeat scorings."""
    if not scoring_repeats:
        raise SelectionError("At least one repeat scoring is required.")
    if leaderboard_limit < 1:
        raise SelectionError("leaderboard_limit must be positive.")
    views = tuple(sorted(views))
    expected_subsets = math.comb(len(views), SELECTED_COUNT)
    for scoring in scoring_repeats:
        if len(scoring["subsets"]) != expected_subsets:
            raise SelectionError("Repeat subset scoring is incomplete.")
    expected_pairs = pair_space_count(len(views), SELECTED_COUNT, RESERVE_COUNT)
    mandatories = [_mandatory_views(scoring) for scoring in scoring_repeats]
    audited = 0
    qualifying = 0
    leaderboard = []
    bank_caches = [{} for _ in scoring_repeats]
    failures = {
        "f_coverage_not_all_repeats": 0,
        "f_recoverable_events_not_all_repeats": 0,
        "f_distinct_event_views_not_all_repeats": 0,
        "f_not_all_repeats": 0,
        "r_swap_coverage_not_all_repeats": 0,
        "r_member_without_swap_not_all_repeats": 0,
        "r_gate_vacuous_not_all_repeats": 0,
    }
    for combination in itertools.combinations(views, SELECTED_COUNT):
        combination = tuple(sorted(combination))
        entries = []
        try:
            entries = [scoring["subsets"][combination] for scoring in scoring_repeats]
        except KeyError as error:
            raise SelectionError("Repeat subset scoring is incomplete.") from error
        outside = tuple(view for view in views if view not in set(combination))
        reserve_space = math.comb(len(outside), RESERVE_COUNT)
        f_failure = _first_f_failure(entries)
        if f_failure is not None:
            audited += reserve_space
            failures[f_failure] += reserve_space
            continue
        if any(
            not reserve_gate_binds(combination, mandatory)
            for mandatory in mandatories
        ):
            audited += reserve_space
            failures["r_gate_vacuous_not_all_repeats"] += reserve_space
            continue
        swaps_by_repeat = [
            v7._swap_targets(  # noqa: SLF001 - exact reuse of frozen v7 semantics
                scoring["subsets"], combination, views, mandatory
            )
            for scoring, mandatory in zip(scoring_repeats, mandatories)
        ]
        for reserve in itertools.combinations(outside, RESERVE_COUNT):
            audited += 1
            repeat_pair = [
                _repeat_pair_metrics(swaps, reserve) for swaps in swaps_by_repeat
            ]
            failed = next((item for item in repeat_pair if not item["passes"]), None)
            if failed is not None:
                failures[failed["failure"]] += 1
                continue
            bank = tuple(sorted(combination + tuple(reserve)))
            bank_per_repeat = [
                _bank_metrics(scoring, bank, angle_table, cache)
                for scoring, cache in zip(scoring_repeats, bank_caches)
            ]
            repeat_data_scores = [
                {
                    "min_measured_views": int(entry["min_measured_views"]),
                    "negative_cells_at_minimum": -int(entry["cells_at_minimum"]),
                    "min_view_character_bone_frames": int(
                        entry["min_view_character_bone_frames"]
                    ),
                    "min_best_swap_measured_views": int(
                        pair["min_best_swap_measured_views"]
                    ),
                    "ordered_legal_swap_pairs": int(
                        pair["ordered_legal_swap_pairs"]
                    ),
                    "passing_selected_subsets_within_bank": int(bank_metric[0]),
                }
                for entry, pair, bank_metric in zip(
                    entries, repeat_pair, bank_per_repeat
                )
            ]
            score = [
                min(item["min_measured_views"] for item in repeat_data_scores),
                min(item["negative_cells_at_minimum"] for item in repeat_data_scores),
                min(
                    item["min_view_character_bone_frames"]
                    for item in repeat_data_scores
                ),
                min_pairwise_angle(combination, angle_table),
                min(
                    item["min_best_swap_measured_views"]
                    for item in repeat_data_scores
                ),
                min(
                    item["ordered_legal_swap_pairs"] for item in repeat_data_scores
                ),
                min(
                    item["passing_selected_subsets_within_bank"]
                    for item in repeat_data_scores
                ),
                bank_per_repeat[0][1],
            ]
            if any(bank_metric[1] != score[7] for bank_metric in bank_per_repeat[1:]):
                raise SelectionError("Geometry changed between repeat evaluations.")
            qualifying += 1
            leaderboard.append(
                {
                    "F": list(combination),
                    "R": list(sorted(reserve)),
                    "B": list(bank),
                    "score": score,
                    "repeat_data_scores": repeat_data_scores,
                    "repeat_coverage": [entry["coverage"] for entry in entries],
                    "repeat_worst_cells": [entry["worst_cell"] for entry in entries],
                    "repeat_cells_at_minimum": [
                        int(entry["cells_at_minimum"]) for entry in entries
                    ],
                }
            )
            if len(leaderboard) > leaderboard_limit:
                leaderboard.sort(key=_rank_key)
                del leaderboard[leaderboard_limit:]
    if audited != expected_pairs:
        raise SelectionError(
            "Audited {} pairs, expected {}.".format(audited, expected_pairs)
        )
    leaderboard.sort(key=_rank_key)
    return {
        "pairs_audited": audited,
        "pairs_expected": expected_pairs,
        "qualifying_pairs": qualifying,
        "hard_gate_failures": failures,
        "provably_mandatory_views_by_repeat": mandatories,
        "chosen": leaderboard[0] if leaderboard else None,
        "runner_up": leaderboard[1] if len(leaderboard) > 1 else None,
        "leaderboard": leaderboard,
    }


def _legal_full_swap_in_repeat(
    scoring: Mapping,
    views: Sequence[str],
    mandatory: Sequence[str],
    full_selected: Sequence[str],
    dropped: str,
    replacement: str,
) -> bool:
    swaps = v7._swap_targets(  # noqa: SLF001 - exact reuse of frozen v7 semantics
        scoring["subsets"], full_selected, views, mandatory
    )
    target = swaps["targets"].get(dropped)
    bit_index = swaps["bit_of"].get(replacement)
    return bool(
        target is not None
        and bit_index is not None
        and target["mask"] & (1 << bit_index)
    )


def _loro_stability_record(
    omitted_index: int,
    full_candidate: Mapping,
    full_m0: Mapping,
    loro_candidate: Mapping | None,
    loro_m0: Mapping | None,
    full_scoring_repeats: Sequence[Mapping],
    views: Sequence[str],
) -> dict:
    """Compare one LORO result, including 3/3 legality of a one-view swap."""
    full_f = set(full_candidate["F"])
    full_r = set(full_candidate["R"])
    if loro_candidate is None:
        return {
            "omitted_repeat_index": omitted_index,
            "candidate": None,
            "B_exact": False,
            "M0_exact": False,
            "F_changes_at_most_one": False,
            "replacement_from_full_R": False,
            "replacement_legal_v7_swap_by_complete_repeat": [False] * REPEAT_COUNT,
            "replacement_legal_v7_swap_3_of_3": False,
            "pass": False,
        }
    loro_f = set(loro_candidate["F"])
    removed = sorted(full_f - loro_f)
    added = sorted(loro_f - full_f)
    one_or_zero = len(removed) == len(added) and len(removed) <= 1
    from_reserve = one_or_zero and set(added).issubset(full_r)
    legal_by_repeat = []
    if not added:
        legal_by_repeat = [True] * len(full_scoring_repeats)
    elif one_or_zero and from_reserve:
        for scoring in full_scoring_repeats:
            mandatory = _mandatory_views(scoring)
            legal_by_repeat.append(
                _legal_full_swap_in_repeat(
                    scoring,
                    views,
                    mandatory,
                    full_candidate["F"],
                    removed[0],
                    added[0],
                )
            )
    else:
        legal_by_repeat = [False] * len(full_scoring_repeats)
    b_exact = tuple(loro_candidate["B"]) == tuple(full_candidate["B"])
    m0_exact = bool(loro_m0 and loro_m0["view"] == full_m0["view"])
    legal_all = len(legal_by_repeat) == REPEAT_COUNT and all(legal_by_repeat)
    passed = bool(b_exact and m0_exact and one_or_zero and from_reserve and legal_all)
    return {
        "omitted_repeat_index": omitted_index,
        "candidate": loro_candidate,
        "m0_anchor": loro_m0,
        "B_exact": b_exact,
        "M0_exact": m0_exact,
        "F_removed": removed,
        "F_added": added,
        "F_changes_at_most_one": one_or_zero,
        "replacement_from_full_R": from_reserve,
        "replacement_legal_v7_swap_by_complete_repeat": legal_by_repeat,
        "replacement_legal_v7_swap_3_of_3": legal_all,
        "pass": passed,
    }


def _drop_character(data: Mapping, omitted: str) -> dict:
    characters = [character for character in data["characters"] if character != omitted]
    if omitted not in data["characters"] or not characters:
        raise SelectionError("LOCO character is absent or would empty the matrix.")
    return {
        "views": list(data["views"]),
        "characters": characters,
        "frame_count": data["frame_count"],
        "E": {
            key: value for key, value in data["E"].items() if key[1] != omitted
        },
        "T": {
            key: value for key, value in data["T"].items() if key[1] != omitted
        },
    }


def _candidate_comparison(candidate: Mapping | None, reference: Mapping) -> dict:
    if candidate is None:
        return {
            "candidate_exists": False,
            "B_exact": False,
            "F_exact": False,
            "R_exact": False,
            "F_symmetric_difference": None,
        }
    return {
        "candidate_exists": True,
        "B_exact": tuple(candidate["B"]) == tuple(reference["B"]),
        "F_exact": tuple(candidate["F"]) == tuple(reference["F"]),
        "R_exact": tuple(candidate["R"]) == tuple(reference["R"]),
        "F_symmetric_difference": sorted(
            set(candidate["F"]) ^ set(reference["F"])
        ),
    }


def _final_pass(full_candidate: Mapping | None, loro_records: Sequence[Mapping]) -> bool:
    """Only full-repeat feasibility and all three LORO gates decide selection."""
    return bool(
        full_candidate
        and len(loro_records) == REPEAT_COUNT
        and all(record["pass"] for record in loro_records)
    )


def select(matrices: Sequence[Mapping], leaderboard_limit: int = 5) -> dict:
    """Run full, three LORO and report-only four LOCO exhaustive selections."""
    angle_table = pairwise_angle_table()
    prepared = _prepare_repeats(matrices, angle_table)
    views = prepared[0]["data"]["views"]
    full_scoring = [item["scoring"] for item in prepared]
    full = _core_select(full_scoring, angle_table, views, leaderboard_limit)
    full_candidate = full["chosen"]
    full_m0 = (
        _m0_from_worst_clean(
            full_candidate["F"], angle_table, full_scoring, views
        )
        if full_candidate
        else None
    )

    loro_records = []
    if full_candidate:
        for omitted in range(REPEAT_COUNT):
            retained = [
                scoring
                for index, scoring in enumerate(full_scoring)
                if index != omitted
            ]
            result = _core_select(retained, angle_table, views, leaderboard_limit)
            candidate = result["chosen"]
            m0 = (
                _m0_from_worst_clean(candidate["F"], angle_table, retained, views)
                if candidate
                else None
            )
            record = _loro_stability_record(
                omitted,
                full_candidate,
                full_m0,
                candidate,
                m0,
                full_scoring,
                views,
            )
            record["omitted_repeat_label"] = prepared[omitted]["label"]
            record["search"] = result
            loro_records.append(record)

    # LOCO is deliberately computed after the decision inputs are complete and
    # is never passed to _final_pass.
    loco_records = []
    if full_candidate:
        for character in CHARACTERS:
            loco_scoring = [
                v7.score_subsets(
                    _drop_character(item["data"], character), angle_table
                )
                for item in prepared
            ]
            result = _core_select(loco_scoring, angle_table, views, leaderboard_limit)
            candidate = result["chosen"]
            m0 = (
                _m0_from_worst_clean(candidate["F"], angle_table, loco_scoring, views)
                if candidate
                else None
            )
            loco_records.append(
                {
                    "omitted_character": character,
                    "report_only": True,
                    "search": result,
                    "candidate": candidate,
                    "m0_anchor": m0,
                    "comparison_to_full": _candidate_comparison(
                        candidate, full_candidate
                    ),
                }
            )

    passed = _final_pass(full_candidate, loro_records)
    if not full_candidate:
        failure = "no_pair_passed_all_hard_gates_in_all_three_repeats"
    elif not passed:
        failure = "loro_stability_gate_failed"
    else:
        failure = None
    return {
        "record": "common_bank_18_9_5_selection_v8",
        "rule": RULE_TAG,
        "scoring_base_rule": V7_RULE_TAG,
        "uses_binary_matrix_only": True,
        "formal_capture_authorized": False,
        "repeat_count": REPEAT_COUNT,
        "repeat_labels": [item["label"] for item in prepared],
        "hard_gate_repeat_rule": "all_three_repeats_independently",
        "data_score_reduction": "repeat_wise_worst",
        "geometry_and_lexical_keys": "v7_unchanged",
        "availability_gate": "coverage_at_deployment_floor",
        "coverage_floor": COVERAGE_FLOOR,
        "coverage_gates": dict(COVERAGE_GATES),
        "min_swappable_selected_views": MIN_SWAPPABLE_SELECTED_VIEWS,
        "ordering_keys": list(ROBUST_ORDERING_KEYS),
        "pairs_audited": full["pairs_audited"],
        "pairs_expected": full["pairs_expected"],
        "qualifying_pairs": full["qualifying_pairs"],
        "hard_gate_failures": full["hard_gate_failures"],
        "provably_mandatory_views_by_repeat": full[
            "provably_mandatory_views_by_repeat"
        ],
        "full_search": full,
        "chosen": full_candidate,
        "runner_up": full["runner_up"],
        "m0_anchor": full_m0,
        "loro": loro_records,
        "loro_all_pass": bool(loro_records) and all(
            record["pass"] for record in loro_records
        ),
        "loco": loco_records,
        "loco_is_report_only": True,
        "selected_view_ids": full_candidate["F"] if passed else None,
        "bank_view_ids": full_candidate["B"] if passed else None,
        "m0_view_id": full_m0["view"] if passed else None,
        "pass": passed,
        "failure": failure,
    }
