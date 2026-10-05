"""Exhaustive 18->9->5 common camera selection (pure Python).

Implements the gates and the ordering frozen in AGENTS.md 6.6.20 and mirrored in
``common_bank_rule_lock_v1``.  It consumes a selection matrix -- the binary ``E``
and ``T`` arrays produced by the raw BlazePose replay -- and produces the freeze
record.  It never sees GT coordinates, GT error, limb-length error, M0-M3 output,
offsets or weights, and it has no notion of which nine views were surveyed first.

Two properties are worth stating because they are what make the freeze auditable:

* **every score is an integer.**  Rates are kept as frame counts over a constant
  241-frame denominator and angles as frozen nanodegrees, so no tie is ever
  broken by platform floating point.
* **the search is exhaustive by construction.**  All
  ``C(18,5) x C(13,4) = 6,126,120`` ``(F,R)`` pairs are audited; the count is
  recomputed and reported so a truncated search cannot pass unnoticed.
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping, Sequence

from common_bank_rule_lock_v1 import (
    CHARACTERS,
    EVENT_LEG_SIDES,
    EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER,
    FRAME_COUNT,
    HISTORICAL_M0_VIEW,
    MARCH_WINDOWS,
    ORDERING_KEYS,
    REQUIRED_MEASURED_VIEWS,
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


class SelectionError(RuntimeError):
    """Raised when a matrix or a search cannot satisfy the frozen contract."""


def _bits(value, frame_count: int = FRAME_COUNT) -> list:
    """Accept a 0/1 string or a 0/1 sequence; reject anything else."""
    if isinstance(value, str):
        flags = [1 if character == "1" else 0 for character in value]
        if any(character not in "01" for character in value):
            raise SelectionError("Bit string holds a character other than 0 or 1.")
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
    """Validate the selection matrix and unpack it into frame arrays."""
    views = list(matrix.get("views") or [])
    if sorted(views) != sorted(SURVEY_VIEWS):
        raise SelectionError("The matrix does not cover the frozen survey layout.")
    characters = list(matrix.get("characters") or [])
    if sorted(characters) != sorted(CHARACTERS):
        raise SelectionError("The matrix does not cover the four characters.")
    frame_count = int(matrix.get("frame_count", FRAME_COUNT))
    if frame_count != FRAME_COUNT:
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
                    measured[(view, character, scenario, bone)] = _bits(raw, frame_count)
            for bone in TARGET_BONES["G1"]:
                try:
                    raw = matrix["T"][view][character][bone]
                except (KeyError, TypeError) as error:
                    raise SelectionError(
                        "T is missing {}/{}/{}.".format(view, character, bone)
                    ) from error
                occluded[(view, character, bone)] = _bits(raw, frame_count)
    return {
        "views": sorted(views),
        "characters": sorted(characters),
        "frame_count": frame_count,
        "E": measured,
        "T": occluded,
    }


def blind_pairs(data: Mapping) -> list:
    """(view, bone) pairs a view never measures in the clean scenario.

    Not a veto on its own -- FS-CTS5 fuses per bone, so a specialist view can
    still earn a slot -- but 6.6.20 requires every one of them in the record.
    """
    found = []
    for view in data["views"]:
        for bone in TARGET_BONES["G0"]:
            if not any(
                any(data["E"][(view, character, "G0", bone)])
                for character in data["characters"]
            ):
                found.append({"view": view, "bone": bone})
    return found


def clean_frame_counts(data: Mapping) -> dict:
    """Per view: the worst (character, bone) clean count, and the total.

    Counts, not rates: the denominator is a constant 241 frames, so integers
    order these identically to the means 6.6.20 names, without a float in sight.
    """
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
            "g0_cell_frames": len(per_pair) * data["frame_count"],
        }
    return profile


def event_views(data: Mapping) -> dict:
    """Per character and leg side, the views that carry a recoverable event."""
    carriers = {}
    for character in data["characters"]:
        per_side = {}
        for side in EVENT_LEG_SIDES:
            window = MARCH_WINDOWS[side]
            views = {}
            for view in data["views"]:
                for bone in SIDE_BONES[side]:
                    event = recoverable_event(
                        data["E"][(view, character, "G1", bone)],
                        data["T"][(view, character, bone)],
                        window,
                    )
                    if event["found"]:
                        views[view] = dict(event, bone=bone)
                        break
            per_side[side] = views
        carriers[character] = per_side
    return carriers


def _cell_masks(data: Mapping) -> dict:
    """Distinct view-masks over target cells, with multiplicity.

    Thousands of cells share a few hundred patterns; collapsing them is what lets
    every one of the 8,568 five-view subsets be scored exactly.
    """
    patterns = {}
    index_of = {view: index for index, view in enumerate(data["views"])}
    for scenario in SELECTION_SCENARIOS:
        for bone in TARGET_BONES[scenario]:
            for character in data["characters"]:
                arrays = {
                    view: data["E"][(view, character, scenario, bone)]
                    for view in data["views"]
                }
                for frame in range(data["frame_count"]):
                    mask = 0
                    for view, flags in arrays.items():
                        if flags[frame]:
                            mask |= 1 << index_of[view]
                    entry = patterns.get(mask)
                    if entry is None:
                        patterns[mask] = {
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


def score_subsets(data: Mapping, angle_table: Mapping) -> dict:
    """Score every five-view subset once; the pair sweep then only looks up."""
    views = data["views"]
    index_of = {view: index for index, view in enumerate(views)}
    patterns = _cell_masks(data)
    clean = clean_frame_counts(data)
    carriers = event_views(data)
    scored = {}
    for combination in itertools.combinations(views, SELECTED_COUNT):
        mask = 0
        for view in combination:
            mask |= 1 << index_of[view]
        minimum = None
        at_minimum = 0
        worst = None
        for pattern, entry in patterns.items():
            count = bin(pattern & mask).count("1")
            if minimum is None or count < minimum:
                minimum = count
                at_minimum = entry["count"]
                worst = entry["cell"]
            elif count == minimum:
                at_minimum += entry["count"]
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
        passes = (
            minimum is not None
            and minimum >= REQUIRED_MEASURED_VIEWS
            and gate_events
            and gate_distinct
        )
        scored[combination] = {
            "subset": list(combination),
            "min_measured_views": minimum,
            "cells_at_minimum": at_minimum,
            "worst_cell": worst,
            "min_view_character_bone_frames": min(
                clean[view]["min_character_bone_frames"] for view in combination
            ),
            "min_pairwise_angle_nanodeg": min_pairwise_angle(combination, angle_table),
            "gate_measured_views": bool(
                minimum is not None and minimum >= REQUIRED_MEASURED_VIEWS
            ),
            "gate_recoverable_events": gate_events,
            "gate_distinct_event_views": gate_distinct,
            "passes": passes,
            "events": events,
        }
    return {"subsets": scored, "clean": clean, "event_views": carriers}


def _rank_key(item: Mapping) -> tuple:
    """Best first: every score is 'higher is better', so negate and sort up.

    The final tie-break is 6.6.20's key 9 -- ascending lexicographic on the view
    ids -- which makes the winner unique even when two pairs score identically.
    """
    return (tuple(-value for value in item["score"]), item["F"], item["R"])


def _swap_targets(scored: Mapping, selected: Sequence[str], views: Sequence[str]) -> dict:
    """For each selected view, the reserves whose swap keeps every F gate.

    Returned as bitmasks over the thirteen views outside ``F``.  The pair sweep
    visits six million reserves, and set-membership arithmetic on a handful of
    integers is what keeps that a few seconds rather than a few minutes; the
    ranked list lets the best swap be read off in a step or two instead of
    rescanning every candidate.
    """
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


def select(matrix: Mapping) -> dict:
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
    # Only a short leaderboard is kept.  Hundreds of thousands of pairs can
    # qualify, and holding an event report for each one would cost gigabytes to
    # answer a question that needs the winner, the runner-up and the counts.
    leaderboard = []
    leaderboard_limit = 5
    bank_pass_cache = {}
    failures = {
        "f_measured_views": 0,
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
            if not entry["gate_measured_views"]:
                failures["f_measured_views"] += reserve_space
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
            # Every reserve must earn its place by replacing at least one
            # selected view; four geometric bystanders are not a nine-view bank.
            if reserve_mask & ~useful:
                failures["r_member_without_swap"] += 1
                continue
            ordered_pairs = sum(
                bin(target["mask"] & reserve_mask).count("1")
                for target in targets.values()
            )
            bank = tuple(sorted(set(combination) | set(reserve)))
            cached = bank_pass_cache.get(bank)
            if cached is None:
                cached = (
                    sum(
                        1
                        for sub in itertools.combinations(bank, SELECTED_COUNT)
                        if scored[sub]["passes"]
                    ),
                    min_pairwise_angle(bank, angle_table),
                )
                bank_pass_cache[bank] = cached
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
        "record": "common_bank_18_9_5_selection_v1",
        "rule": RULE_TAG,
        "gt_error_used": False,
        "formal_capture_authorized": False,
        "pairs_audited": audited,
        "pairs_expected": expected_pairs,
        "qualifying_pairs": qualifying_count,
        "hard_gate_failures": failures,
        "ordering_keys": list(ORDERING_KEYS),
        "chosen": chosen,
        "runner_up": leaderboard[1] if len(leaderboard) > 1 else None,
        "m0_anchor": (
            m0_anchor(data, chosen["F"], angle_table, scoring["clean"])
            if chosen
            else None
        ),
        "blind_pairs": blind_pairs(data),
        "pass": bool(chosen),
        "failure": None if chosen else "no_qualifying_pair",
    }


def m0_anchor(
    data: Mapping,
    selected: Sequence[str],
    angle_table: Mapping,
    clean: Mapping,
) -> dict:
    """The single M0 view inside F, by 6.6.20's four keys.

    ``az000_el00`` gets no privilege beyond being the angular reference of the
    third key; if it is not in F it plays no part at all.
    """
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
