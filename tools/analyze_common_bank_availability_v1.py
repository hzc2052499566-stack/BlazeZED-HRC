"""Diagnose a recorded common-bank availability matrix (pure Python).

AGENTS.md 6.6.18 closed the first nine-view render with two limitations: the
selected five sits at the three-view floor with no margin, and the ``g1_capable``
label said only one of the five could produce the workbench leg occlusion.  Both
statements were read off the *selection*; neither was read off the matrix.  This
module reads the matrix.

It answers, GT-free and offline:

* how much of the bank each view can actually measure, and which bones it is
  **blind** to for the whole capture -- a lateral view that never sees the far
  arm spends a selection slot without contributing to that bone;
* how strong the prop occlusion really is, per view and per cell, by differencing
  the props-hidden scenario against the props-visible ones;
* which cells are hard bank-wide, and which views are **forced** -- present in
  every subset that meets the floor -- because a forced bank has no margin to
  give and a revision, not a re-selection, is what buys it.

Nothing here reads GT error or picks a per-character optimum; it consumes the
same record ``common_camera_bank_v1`` consumes, and is safe to re-run on a denser
matrix without re-rendering.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from common_camera_bank_v1 import (
    MIN_MEASURED_VIEWS,
    SCENARIO_TARGET_BONES,
    SELECTED_COUNT,
    assert_gt_free,
)
from fs_cts5_camera_bank_v1 import BONES

CLEAN_SCENARIO = "G0"
OCCLUDED_SCENARIOS = ("G1", "G2")


class BankAnalysisError(RuntimeError):
    """Raised when a record cannot support the diagnosis being asked of it."""


def _frames(availability: Mapping, character: str, scenario: str) -> list:
    frames = availability[character].get(scenario) or {}
    return sorted(frames, key=lambda frame: int(frame))


def bone_measured(
    availability: Mapping,
    character: str,
    scenario: str,
    frame: str,
    view: str,
    bone: str,
) -> bool:
    """A bone is measured by a view when both of its endpoints are available."""
    try:
        joints = availability[character][scenario][frame][view]
    except KeyError as error:
        raise BankAnalysisError(
            "No availability for {}/{}/{}/{}.".format(character, scenario, frame, view)
        ) from error
    first, second = BONES[bone]
    return bool(joints[first]) and bool(joints[second])


def view_profiles(
    availability: Mapping,
    views: Sequence[str],
    scenario_targets: Mapping = SCENARIO_TARGET_BONES,
) -> dict:
    """Per view: what it measures, and what it is blind to in the clean scenario.

    ``blind_bones_clean`` is the sharp one.  A bone that a view never measures in
    ``G0`` -- across every character and frame -- is not a hard frame or an
    unlucky pose; it is the body standing in its own way, and no prop change
    fixes it.
    """
    profiles = {}
    for view in views:
        per_scenario = {}
        for scenario, targets in scenario_targets.items():
            measured = 0
            total = 0
            for character in availability:
                for frame in _frames(availability, character, scenario):
                    for bone in targets:
                        total += 1
                        measured += bone_measured(
                            availability, character, scenario, frame, view, bone
                        )
            per_scenario[scenario] = {"measured": measured, "total": total}
        blind = []
        partial = {}
        for bone in BONES:
            measured = 0
            total = 0
            for character in availability:
                for frame in _frames(availability, character, CLEAN_SCENARIO):
                    total += 1
                    measured += bone_measured(
                        availability, character, CLEAN_SCENARIO, frame, view, bone
                    )
            if total and measured == 0:
                blind.append(bone)
            elif total and measured < total:
                partial[bone] = {"measured": measured, "total": total}
        profiles[view] = {
            "per_scenario": per_scenario,
            "blind_bones_clean": blind,
            "partial_bones_clean": partial,
            "measured_total": sum(item["measured"] for item in per_scenario.values()),
            "cell_total": sum(item["total"] for item in per_scenario.values()),
        }
    return profiles


def prop_occlusion_strength(
    availability: Mapping,
    views: Sequence[str],
    scenario: str,
    scenario_targets: Mapping = SCENARIO_TARGET_BONES,
) -> dict:
    """Cells each view loses to props, differenced against the clean render.

    Only cells the view could measure when the props were hidden are counted, so
    self-occlusion is never billed to the workbench.
    """
    targets = scenario_targets[scenario]
    per_view = {}
    for view in views:
        lost = 0
        total = 0
        per_frame = collections.Counter()
        for character in availability:
            for frame in _frames(availability, character, scenario):
                for bone in targets:
                    total += 1
                    clean = bone_measured(
                        availability, character, CLEAN_SCENARIO, frame, view, bone
                    )
                    occluded = bone_measured(
                        availability, character, scenario, frame, view, bone
                    )
                    if clean and not occluded:
                        lost += 1
                        per_frame[frame] += 1
        per_view[view] = {
            "occluded_cells": lost,
            "target_cells": total,
            "occluded_fraction": (lost / total) if total else 0.0,
            "occluded_by_frame": dict(
                sorted(per_frame.items(), key=lambda item: int(item[0]))
            ),
        }
    return per_view


def cell_census(
    availability: Mapping,
    views: Sequence[str],
    scenario: str,
    scenario_targets: Mapping = SCENARIO_TARGET_BONES,
    hardest_limit: int = 20,
) -> dict:
    """Bank-wide per-cell counts: how many views still measure, how many props took.

    The floor in 6.6.5 is per cell, so the distribution over cells -- not the mean
    over the scenario -- is what decides feasibility.
    """
    targets = scenario_targets[scenario]
    measurable_histogram = collections.Counter()
    occluded_histogram = collections.Counter()
    cells = []
    for character in availability:
        for frame in _frames(availability, character, scenario):
            for bone in targets:
                measuring = [
                    view
                    for view in views
                    if bone_measured(
                        availability, character, scenario, frame, view, bone
                    )
                ]
                occluded = sum(
                    1
                    for view in views
                    if bone_measured(
                        availability, character, CLEAN_SCENARIO, frame, view, bone
                    )
                    and not bone_measured(
                        availability, character, scenario, frame, view, bone
                    )
                )
                measurable_histogram[len(measuring)] += 1
                occluded_histogram[occluded] += 1
                cells.append(
                    {
                        "character": character,
                        "frame": frame,
                        "bone": bone,
                        "measurable_views": len(measuring),
                        "prop_occluded_views": occluded,
                        "measuring": measuring,
                    }
                )
    cells.sort(
        key=lambda cell: (cell["measurable_views"], -cell["prop_occluded_views"])
    )
    return {
        "cells": len(cells),
        "measurable_views_histogram": dict(sorted(measurable_histogram.items())),
        "prop_occluded_views_histogram": dict(sorted(occluded_histogram.items())),
        "min_measurable_views": min(measurable_histogram) if measurable_histogram else 0,
        "hardest_cells": cells[:hardest_limit],
    }


def _cell_masks(
    availability: Mapping,
    views: Sequence[str],
    scenario_targets: Mapping,
) -> dict:
    """Per scenario, one bitmask per *distinct* cell, with a representative cell.

    A cell is fully described, for subset arithmetic, by which views measure it.
    Thousands of cells share only a few hundred such patterns, so collapsing them
    turns the subset sweep from tens of millions of lookups into a popcount over
    the distinct patterns.  The representative is kept so a worst cell can still
    be named.
    """
    per_scenario = {}
    for scenario, targets in scenario_targets.items():
        patterns = {}
        for character in availability:
            for frame in _frames(availability, character, scenario):
                for bone in targets:
                    mask = 0
                    for index, view in enumerate(views):
                        if bone_measured(
                            availability, character, scenario, frame, view, bone
                        ):
                            mask |= 1 << index
                    if mask not in patterns:
                        patterns[mask] = {
                            "character": character,
                            "frame": frame,
                            "bone": bone,
                        }
        per_scenario[scenario] = patterns
    return per_scenario


def subset_margins(
    availability: Mapping,
    views: Sequence[str],
    scenario_targets: Mapping = SCENARIO_TARGET_BONES,
    selected_count: int = SELECTED_COUNT,
    floor: int = MIN_MEASURED_VIEWS,
    near_miss_limit: int = 10,
    stored_subset_limit: int = 50,
) -> dict:
    """Every subset's worst cell, and which views no feasible subset can drop.

    A view present in every feasible subset is *forced*.  Forced views measure how
    much freedom the bank has left: if the feasible subsets are all the same four
    views plus one, the bank has no margin to trade, and 6.6.5's remedy for that
    is to revise the bank before the freeze rather than to re-rank it.

    Counts are exact over all subsets; only the listed subsets are capped, so a
    survey bank of eighteen stays analysable without a record of every subset.
    """
    ordered = sorted(views)
    index_of = {view: index for index, view in enumerate(ordered)}
    patterns = _cell_masks(availability, ordered, scenario_targets)
    feasible = []
    near_miss = []
    subset_count = 0
    feasible_count = 0
    near_miss_count = 0
    forced = None
    best_margin = None
    for combination in itertools.combinations(ordered, selected_count):
        subset_count += 1
        subset_mask = 0
        for view in combination:
            subset_mask |= 1 << index_of[view]
        worst = None
        per_scenario = {}
        for scenario in scenario_targets:
            scenario_worst = None
            for mask, cell in patterns[scenario].items():
                count = bin(mask & subset_mask).count("1")
                if scenario_worst is None or count < scenario_worst["measured_views"]:
                    scenario_worst = dict(cell, measured_views=count)
            per_scenario[scenario] = scenario_worst
            if scenario_worst is not None and (
                worst is None
                or scenario_worst["measured_views"] < worst["measured_views"]
            ):
                worst = dict(scenario_worst, scenario=scenario)
        minimum = worst["measured_views"] if worst else 0
        entry = {
            "subset": list(combination),
            "min_measured_views": minimum,
            "margin": minimum - floor,
            "worst_cell": worst,
            "per_scenario_worst": per_scenario,
        }
        if minimum >= floor:
            feasible_count += 1
            forced = (
                set(combination) if forced is None else forced & set(combination)
            )
            if best_margin is None or entry["margin"] > best_margin:
                best_margin = entry["margin"]
            feasible.append(entry)
        elif minimum == floor - 1:
            near_miss_count += 1
            if len(near_miss) < near_miss_limit:
                near_miss.append(entry)
    feasible.sort(key=lambda entry: (-entry["margin"], entry["subset"]))
    return {
        "floor": floor,
        "subset_count": subset_count,
        "feasible_count": feasible_count,
        "feasible_subsets": feasible[:stored_subset_limit],
        "forced_views": sorted(forced) if forced else [],
        "free_slots": (selected_count - len(forced)) if forced is not None else None,
        "best_margin": best_margin,
        "near_miss_count": near_miss_count,
        "near_miss": near_miss,
    }


def diagnose(
    record: Mapping, scenario_targets: Mapping = SCENARIO_TARGET_BONES
) -> dict:
    """The whole read of one availability record."""
    assert_gt_free(record.get("selection", {}).get("evidence_used", []) or [])
    availability = record.get("availability") or {}
    if not availability:
        raise BankAnalysisError("The record holds no availability matrix.")
    views = [entry["name"] for entry in record["bank"]]
    scenarios = sorted(
        {
            scenario
            for character in availability
            for scenario in availability[character]
        }
    )
    first_character = sorted(availability)[0]
    return {
        "record": "common_bank_availability_diagnosis_v1",
        "classification": record.get("classification"),
        "formal_capture_authorized": bool(record.get("formal_capture_authorized", False)),
        "gt_error_used": False,
        "source_attempt": record.get("attempt"),
        "characters": sorted(availability),
        "scenarios": scenarios,
        "views": views,
        "sampled_frames": {
            scenario: _frames(availability, first_character, scenario)
            for scenario in scenarios
        },
        "view_profiles": view_profiles(availability, views, scenario_targets),
        "prop_occlusion": {
            scenario: prop_occlusion_strength(
                availability, views, scenario, scenario_targets
            )
            for scenario in OCCLUDED_SCENARIOS
            if scenario in scenarios
        },
        "cell_census": {
            scenario: cell_census(availability, views, scenario, scenario_targets)
            for scenario in OCCLUDED_SCENARIOS
            if scenario in scenarios
        },
        "subset_margins": subset_margins(availability, views, scenario_targets),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path, help="common_bank_availability_v1.json")
    parser.add_argument("output", type=Path, help="diagnosis JSON to write (write-once)")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise BankAnalysisError(
            "{} exists; records are write-once, use a new attempt.".format(args.output)
        )
    record = json.loads(args.record.read_text(encoding="utf-8"))
    diagnosis = diagnose(record)
    diagnosis["source_record"] = str(args.record)
    diagnosis["source_record_sha256"] = _sha256_file(args.record)
    diagnosis["script_sha256"] = _sha256_file(Path(__file__).resolve())
    diagnosis["created_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
