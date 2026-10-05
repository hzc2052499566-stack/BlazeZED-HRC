"""Exhaustive 18->9->5 selection under rule lock v3 (pure Python).

Same search as ``select_common_bank_18_9_5_v1`` -- all
``C(18,5) x C(13,4) = 6,126,120`` ``(F,R)`` pairs, integer scores only, the same
nine-key ordering and the same M0 rule.  One thing differs, and it is the whole
point of v3: the availability gate is **measured coverage at the deployment
floor** against 6.6.6's pre-registered thresholds, instead of a per-cell veto.

The per-cell minimum has not gone away; it moved.  Ordering key 1 still
maximises the worst cell, so between two subsets that both clear the coverage
gate the search still prefers the one that leaves fewer cells thin.

Coverage is compared as an exact integer ratio, not a float: a cell either
reaches the floor or it does not, so the comparison is ``covered * 10**6 >=
gate_micro * total`` and no rounding can decide a gate.

No GT coordinates, GT error, limb-length error, M0-M3 output, offsets or weights
are read anywhere in this file.
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping, Sequence

from common_bank_rule_lock_v3 import (
    CHARACTERS,
    COVERAGE_FLOOR,
    COVERAGE_GATES,
    EVENT_LEG_SIDES,
    EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER,
    EVENT_MIN_OCCLUDED_FRAMES,
    FRAME_COUNT,
    HISTORICAL_M0_VIEW,
    MARCH_WINDOWS,
    ORDERING_KEYS,
    RESERVE_COUNT,
    RULE_TAG,
    SELECTED_COUNT,
    SELECTION_SCENARIOS,
    SIDE_BONES,
    SURVEY_COUNT,
    SURVEY_VIEWS,
    TARGET_BONES,
    min_pairwise_angle,
    pairwise_angle_table,
    recoverable_event,
)

# Coverage gates are compared in millionths so the test is exact integer
# arithmetic; 0.98 and 0.90 become 980000 and 900000.
COVERAGE_SCALE = 10 ** 6
COVERAGE_GATES_MICRO = {
    scenario: int(round(float(gate) * COVERAGE_SCALE))
    for scenario, gate in COVERAGE_GATES.items()
}


class SelectionError(RuntimeError):
    """Raised when a matrix or a search cannot satisfy the frozen contract."""


def _bits(value, frame_count: int = FRAME_COUNT) -> list:
    if isinstance(value, str):
        if any(character not in "01" for character in value):
            raise SelectionError("Bit string holds a character other than 0 or 1.")
        flags = [1 if character == "1" else 0 for character in value]
    else:
        flags = [int(flag) for flag in value]
        if any(flag not in (0, 1) for flag in flags):
            raise SelectionError("Binary array holds a value other than 0 or 1.")
    if len(flags) != frame_count:
        raise SelectionError(
            "Expected {} frames, received {}.".format(frame_count, len(flags))
        )
    return flags


def load_matrix(matrix: Mapping) -> dict:
    """Validate the selection matrix against v3's layout and unpack it."""
    views = list(matrix.get("views") or [])
    if sorted(views) != sorted(SURVEY_VIEWS):
        raise SelectionError("The matrix does not cover the frozen v3 layout.")
    characters = list(matrix.get("characters") or [])
    if sorted(characters) != sorted(CHARACTERS):
        raise SelectionError("The matrix does not cover the four characters.")
    if int(matrix.get("frame_count", FRAME_COUNT)) != FRAME_COUNT:
        raise SelectionError("The matrix must hold all {} frames.".format(FRAME_COUNT))
    measured = {}
    occluded = {}
    for view in views:
        for character in characters:
            for scenario in SELECTION_SCENARIOS:
                for bone in TARGET_BONES[scenario]:
                    try:
                        raw = matrix["E"][view][character][scenario][bone]
                    except (KeyError, TypeError) as error:
                        raise SelectionError(
                            "E is missing {}/{}/{}/{}.".format(
                                view, character, scenario, bone
                            )
                        ) from error
                    measured[(view, character, scenario, bone)] = _bits(raw)
            for bone in TARGET_BONES["G1"]:
                try:
                    raw = matrix["T"][view][character][bone]
                except (KeyError, TypeError) as error:
                    raise SelectionError(
                        "T is missing {}/{}/{}.".format(view, character, bone)
                    ) from error
                occluded[(view, character, bone)] = _bits(raw)
    return {
        "views": sorted(views),
        "characters": sorted(characters),
        "frame_count": FRAME_COUNT,
        "E": measured,
        "T": occluded,
    }


def blind_pairs(data: Mapping) -> list:
    """(view, bone) pairs a view never measures in the clean scenario."""
    return [
        {"view": view, "bone": bone}
        for view in data["views"]
        for bone in TARGET_BONES["G0"]
        if not any(
            any(data["E"][(view, character, "G0", bone)])
            for character in data["characters"]
        )
    ]


def clean_frame_counts(data: Mapping) -> dict:
    """Per view: the worst (character, bone) clean count, and the total."""
    profile = {}
    for view in data["views"]:
        per_pair = []
        total = 0
        for character in data["characters"]:
            for bone in TARGET_BONES["G0"]:
                measured = sum(data["E"][(view, character, "G0", bone)])
                per_pair.append(measured)
                total += measured
        profile[view] = {
            "min_character_bone_frames": min(per_pair),
            "total_g0_measured_frames": total,
            "g0_cell_frames": len(per_pair) * FRAME_COUNT,
        }
    return profile


def event_views(data: Mapping, min_occluded: int = EVENT_MIN_OCCLUDED_FRAMES) -> dict:
    """Per character and leg side, the views that carry a recoverable event."""
    carriers = {}
    for character in data["characters"]:
        per_side = {}
        for side in EVENT_LEG_SIDES:
            found = {}
            for view in data["views"]:
                for bone in SIDE_BONES[side]:
                    event = recoverable_event(
                        data["E"][(view, character, "G1", bone)],
                        data["T"][(view, character, bone)],
                        MARCH_WINDOWS[side],
                        min_occluded=min_occluded,
                    )
                    if event["found"]:
                        found[view] = dict(event, bone=bone)
                        break
            per_side[side] = found
        carriers[character] = per_side
    return carriers


def _cell_masks(data: Mapping) -> dict:
    """Per scenario, distinct view-masks over target cells with multiplicity."""
    patterns = {scenario: {} for scenario in SELECTION_SCENARIOS}
    index_of = {view: index for index, view in enumerate(data["views"])}
    for scenario in SELECTION_SCENARIOS:
        for bone in TARGET_BONES[scenario]:
            for character in data["characters"]:
                arrays = {
                    view: data["E"][(view, character, scenario, bone)]
                    for view in data["views"]
                }
                for frame in range(FRAME_COUNT):
                    mask = 0
                    for view, flags in arrays.items():
                        if flags[frame]:
                            mask |= 1 << index_of[view]
                    entry = patterns[scenario].get(mask)
                    if entry is None:
                        patterns[scenario][mask] = {
                            "count": 1,
                            "cell": {
                                "character": character,
                                "scenario": scenario,
                                "frame": frame,
                                "bone": bone,
                            },
                        }
                    else:
                        entry["count"] += 1
    return patterns


def score_subsets(
    data: Mapping,
    angle_table: Mapping,
    min_occluded: int = EVENT_MIN_OCCLUDED_FRAMES,
) -> dict:
    """Score every five-view subset once; the pair sweep then only looks up."""
    views = data["views"]
    index_of = {view: index for index, view in enumerate(views)}
    patterns = _cell_masks(data)
    clean = clean_frame_counts(data)
    carriers = event_views(data, min_occluded)
    totals = {
        scenario: sum(entry["count"] for entry in patterns[scenario].values())
        for scenario in SELECTION_SCENARIOS
    }
    scored = {}
    for combination in itertools.combinations(views, SELECTED_COUNT):
        mask = 0
        for view in combination:
            mask |= 1 << index_of[view]
        minimum = None
        at_minimum = 0
        worst = None
        covered = {scenario: 0 for scenario in SELECTION_SCENARIOS}
        for scenario in SELECTION_SCENARIOS:
            for pattern, entry in patterns[scenario].items():
                count = bin(pattern & mask).count("1")
                if count >= COVERAGE_FLOOR:
                    covered[scenario] += entry["count"]
                if minimum is None or count < minimum:
                    minimum = count
                    at_minimum = entry["count"]
                    worst = entry["cell"]
                elif count == minimum:
                    at_minimum += entry["count"]
        gate_coverage = all(
            covered[scenario] * COVERAGE_SCALE
            >= COVERAGE_GATES_MICRO[scenario] * totals[scenario]
            for scenario in COVERAGE_GATES_MICRO
        )
        chosen = set(combination)
        events = {}
        gate_events = True
        gate_distinct = True
        for character in data["characters"]:
            per_character = {}
            contributing = set()
            for side in EVENT_LEG_SIDES:
                carried = {
                    view: detail
                    for view, detail in carriers[character][side].items()
                    if view in chosen
                }
                per_character[side] = carried
                if not carried:
                    gate_events = False
                contributing.update(carried)
            per_character["distinct_views"] = sorted(contributing)
            if len(contributing) < EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER:
                gate_distinct = False
            events[character] = per_character
        scored[combination] = {
            "subset": list(combination),
            "min_measured_views": minimum,
            "cells_at_minimum": at_minimum,
            "worst_cell": worst,
            "covered_cells": dict(covered),
            "target_cells": dict(totals),
            "coverage": {
                scenario: covered[scenario] / totals[scenario]
                for scenario in SELECTION_SCENARIOS
            },
            "min_view_character_bone_frames": min(
                clean[view]["min_character_bone_frames"] for view in combination
            ),
            "min_pairwise_angle_nanodeg": min_pairwise_angle(combination, angle_table),
            "gate_coverage": gate_coverage,
            "gate_recoverable_events": gate_events,
            "gate_distinct_event_views": gate_distinct,
            "passes": bool(gate_coverage and gate_events and gate_distinct),
            "events": events,
        }
    return {"subsets": scored, "clean": clean, "event_views": carriers}


def _rank_key(item: Mapping) -> tuple:
    """Best first: every score is 'higher is better', so negate and sort up."""
    return (tuple(-value for value in item["score"]), item["F"], item["R"])


def _swap_targets(scored: Mapping, selected: Sequence[str], views: Sequence[str]) -> dict:
    """For each selected view, the reserves whose swap keeps every F gate."""
    chosen = set(selected)
    outside = [view for view in views if view not in chosen]
    bit_of = {view: index for index, view in enumerate(outside)}
    targets = {}
    for dropped in selected:
        remainder = [view for view in selected if view != dropped]
        mask = 0
        ranked = []
        for candidate in outside:
            entry = scored[tuple(sorted(remainder + [candidate]))]
            if entry["passes"]:
                mask |= 1 << bit_of[candidate]
                ranked.append((entry["min_measured_views"], 1 << bit_of[candidate]))
        ranked.sort(key=lambda item: -item[0])
        targets[dropped] = {"mask": mask, "ranked": ranked}
    return {"outside": outside, "bit_of": bit_of, "targets": targets}


def m0_anchor(selected: Sequence[str], angle_table: Mapping, clean: Mapping) -> dict:
    """The single M0 view inside F, by 6.6.20's four keys."""
    if not selected:
        raise SelectionError("M0 needs a selected subset.")
    ranked = sorted(
        selected,
        key=lambda view: (
            -clean[view]["min_character_bone_frames"],
            -clean[view]["total_g0_measured_frames"],
            angle_table[view][HISTORICAL_M0_VIEW],
            view,
        ),
    )
    return {
        "view": ranked[0],
        "ranked": ranked,
        "min_character_bone_frames": clean[ranked[0]]["min_character_bone_frames"],
        "total_g0_measured_frames": clean[ranked[0]]["total_g0_measured_frames"],
        "angle_to_historical_nanodeg": angle_table[ranked[0]][HISTORICAL_M0_VIEW],
        "historical_reference": HISTORICAL_M0_VIEW,
        "fixed_across_characters": True,
    }


def select(matrix: Mapping, leaderboard_limit: int = 5) -> dict:
    """Audit every (F, R) pair and return the ordered result."""
    data = load_matrix(matrix)
    angle_table = pairwise_angle_table()
    scoring = score_subsets(data, angle_table)
    scored = scoring["subsets"]
    views = data["views"]
    expected_pairs = math.comb(SURVEY_COUNT, SELECTED_COUNT) * math.comb(
        SURVEY_COUNT - SELECTED_COUNT, RESERVE_COUNT
    )
    audited = 0
    qualifying_count = 0
    leaderboard = []
    bank_cache = {}
    failures = {
        "f_coverage": 0,
        "f_recoverable_events": 0,
        "f_distinct_event_views": 0,
        "r_swap_coverage": 0,
        "r_member_without_swap": 0,
    }
    for combination, entry in scored.items():
        outside = [view for view in views if view not in set(combination)]
        reserve_space = math.comb(len(outside), RESERVE_COUNT)
        if not entry["passes"]:
            audited += reserve_space
            if not entry["gate_coverage"]:
                failures["f_coverage"] += reserve_space
            elif not entry["gate_recoverable_events"]:
                failures["f_recoverable_events"] += reserve_space
            else:
                failures["f_distinct_event_views"] += reserve_space
            continue
        swaps = _swap_targets(scored, list(combination), views)
        targets = swaps["targets"]
        bit_of = swaps["bit_of"]
        useful = 0
        for target in targets.values():
            useful |= target["mask"]
        for reserve in itertools.combinations(outside, RESERVE_COUNT):
            audited += 1
            reserve_mask = 0
            for candidate in reserve:
                reserve_mask |= 1 << bit_of[candidate]
            best_swap = []
            covered = True
            for target in targets.values():
                if not target["mask"] & reserve_mask:
                    covered = False
                    break
                for minimum, bit in target["ranked"]:
                    if bit & reserve_mask:
                        best_swap.append(minimum)
                        break
            if not covered:
                failures["r_swap_coverage"] += 1
                continue
            if reserve_mask & ~useful:
                failures["r_member_without_swap"] += 1
                continue
            ordered_pairs = sum(
                bin(target["mask"] & reserve_mask).count("1")
                for target in targets.values()
            )
            bank = tuple(sorted(set(combination) | set(reserve)))
            cached = bank_cache.get(bank)
            if cached is None:
                cached = (
                    sum(
                        1
                        for sub in itertools.combinations(bank, SELECTED_COUNT)
                        if scored[sub]["passes"]
                    ),
                    min_pairwise_angle(bank, angle_table),
                )
                bank_cache[bank] = cached
            passing_within_bank, bank_angle = cached
            qualifying_count += 1
            leaderboard.append(
                {
                    "F": list(combination),
                    "R": sorted(reserve),
                    "B": list(bank),
                    "score": [
                        entry["min_measured_views"],
                        -entry["cells_at_minimum"],
                        entry["min_view_character_bone_frames"],
                        entry["min_pairwise_angle_nanodeg"],
                        min(best_swap),
                        ordered_pairs,
                        passing_within_bank,
                        bank_angle,
                    ],
                    "coverage": entry["coverage"],
                    "worst_cell": entry["worst_cell"],
                    "cells_at_minimum": entry["cells_at_minimum"],
                }
            )
            if len(leaderboard) > leaderboard_limit:
                leaderboard.sort(key=_rank_key)
                del leaderboard[leaderboard_limit:]
    if audited != expected_pairs:
        raise SelectionError(
            "Audited {} pairs, the contract fixes {}.".format(audited, expected_pairs)
        )
    leaderboard.sort(key=_rank_key)
    chosen = leaderboard[0] if leaderboard else None
    if chosen:
        chosen = dict(chosen, events=scored[tuple(chosen["F"])]["events"])
    return {
        "record": "common_bank_18_9_5_selection_v3",
        "rule": RULE_TAG,
        "gt_error_used": False,
        "formal_capture_authorized": False,
        "availability_gate": "coverage_at_deployment_floor",
        "coverage_floor": COVERAGE_FLOOR,
        "coverage_gates": dict(COVERAGE_GATES),
        "pairs_audited": audited,
        "pairs_expected": expected_pairs,
        "qualifying_pairs": qualifying_count,
        "hard_gate_failures": failures,
        "ordering_keys": list(ORDERING_KEYS),
        "chosen": chosen,
        "runner_up": leaderboard[1] if len(leaderboard) > 1 else None,
        "m0_anchor": (
            m0_anchor(chosen["F"], angle_table, scoring["clean"]) if chosen else None
        ),
        "blind_pairs": blind_pairs(data),
        "pass": bool(chosen),
        "failure": None if chosen else "no_qualifying_pair",
    }
