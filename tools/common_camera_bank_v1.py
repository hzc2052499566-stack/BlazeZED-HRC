"""Common nine-view candidate bank and five-view selection (pure Python).

AGENTS.md 6.6.5 fixes the rules this module enforces:

* all four characters share **one** five-view subset;
* selection may use GT-free evidence only -- endpoint availability, ROI and
  projection, camera geometry, RGB/depth manipulation, target and control
  coverage.  Reading GT error, or picking a different best subset per character,
  is forbidden;
* if no common subset keeps at least three measured views for every target cell
  in the recoverable scenarios, the candidate bank must be revised *before* the
  freeze rather than the requirement relaxed.

The occlusion pilot (6.6.16) constrains the bank: the workbench only occludes
legs from elevated views in the az 0-120 sector, so a bank without such views
cannot produce the G1 scenario at all.

No Isaac Sim or ``pxr`` dependency; the Isaac side renders availability and
calls in here.
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping, Sequence

from fs_cts5_camera_bank_v1 import BONES


BANK_TAG = "fs_cts5_common_camera_bank_v1"
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False

CANDIDATE_COUNT = 9
SELECTED_COUNT = 5
# 6.6.5: a target cell needs at least three views with measured endpoints.
MIN_MEASURED_VIEWS = 3
# 6.6.5's "every target cell" reading, settled 2026-08-19: a scenario's target
# cells are the bones that scenario is about.  G1 is a leg-occlusion scenario, G2
# a target-arm one, and G0 the clean control against which both are read, so G0
# carries all eight.  The strictest alternative -- every bone in every scenario --
# was measured infeasible on this bank: its best five-view subset leaves a cell
# with a single measured view, because a forearm at reach onset is self-occluded
# from all but one or two viewpoints.
LEG_BONES = ("left_thigh", "right_thigh", "left_shank", "right_shank")
ARM_BONES = ("left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm")
TARGET_ARM_BONES = ("right_upper_arm", "right_forearm")
SCENARIO_TARGET_BONES = {
    "G0": LEG_BONES + ARM_BONES,
    "G1": LEG_BONES,
    "G2": TARGET_ARM_BONES,
}
# Evidence a subset may be scored on.  Anything not on this list -- above all
# GT error -- must not reach this module.
ALLOWED_EVIDENCE = (
    "endpoint_availability",
    "roi_and_projection",
    "camera_geometry",
    "rgb_depth_manipulation",
    "target_and_control_coverage",
)
FORBIDDEN_EVIDENCE = ("gt_error", "ground_truth", "per_character_optimum")
# The pilot measured leg occlusion only from elevated views in this sector.
G1_SECTOR_AZIMUTH_RANGE_DEG = (0.0, 120.0)
G1_MIN_ELEVATION_DEG = 10.0
MIN_G1_CAPABLE_VIEWS = 2


class CameraBankError(RuntimeError):
    """Raised when a bank or an availability record cannot support a freeze."""


def _wrap_azimuth(degrees: float) -> float:
    return float(degrees) % 360.0


def build_views(
    aim: Sequence[float],
    radius_m: float,
    layout: Sequence[Mapping],
) -> list:
    """Named ring views, each carrying the reason it exists.

    Any number of views: 6.6.19 revises the candidate bank, and revising it means
    *surveying* more candidates than the nine that get frozen.  The nine-view
    contract lives in ``build_candidate_bank`` below, which is what the freeze
    calls; a survey renderer calls this and freezes nine of the result.
    """
    if radius_m <= 0.0:
        raise CameraBankError("Bank radius must be positive.")
    views = []
    seen = set()
    for entry in layout:
        for field in ("name", "azimuth_deg", "elevation_deg", "role"):
            if field not in entry:
                raise CameraBankError("Candidate is missing {!r}.".format(field))
        if entry["name"] in seen:
            raise CameraBankError("Duplicate candidate name {!r}.".format(entry["name"]))
        seen.add(entry["name"])
        azimuth = math.radians(float(entry["azimuth_deg"]))
        elevation = math.radians(float(entry["elevation_deg"]))
        horizontal = float(radius_m) * math.cos(elevation)
        views.append(
            {
                "name": str(entry["name"]),
                "azimuth_deg": _wrap_azimuth(entry["azimuth_deg"]),
                "elevation_deg": float(entry["elevation_deg"]),
                "role": str(entry["role"]),
                "eye": [
                    float(aim[0]) + horizontal * math.cos(azimuth),
                    float(aim[1]) + horizontal * math.sin(azimuth),
                    float(aim[2]) + float(radius_m) * math.sin(elevation),
                ],
                "aim": [float(value) for value in aim],
            }
        )
    return views


def build_candidate_bank(
    aim: Sequence[float],
    radius_m: float,
    layout: Sequence[Mapping],
) -> list:
    """Exactly nine named candidates, each carrying the reason it is in the bank.

    The reasons are recorded so the freeze can be audited: a bank whose elevated
    az 0-120 views were dropped could not produce G1, and that has to be visible
    rather than inferred.
    """
    if len(layout) != CANDIDATE_COUNT:
        raise CameraBankError(
            "The bank must hold exactly {} candidates, received {}.".format(
                CANDIDATE_COUNT, len(layout)
            )
        )
    return build_views(aim, radius_m, layout)


def g1_capable(view: Mapping) -> bool:
    """Can this view see the legs occluded by the bench at all?"""
    low, high = G1_SECTOR_AZIMUTH_RANGE_DEG
    azimuth = _wrap_azimuth(view["azimuth_deg"])
    return (
        low <= azimuth <= high
        and float(view["elevation_deg"]) >= G1_MIN_ELEVATION_DEG
    )


def check_bank_can_produce_g1(views: Sequence[Mapping]) -> dict:
    """The bank must contain enough views for the bench occlusion to exist."""
    capable = [view["name"] for view in views if g1_capable(view)]
    return {
        "g1_capable_views": capable,
        "required": MIN_G1_CAPABLE_VIEWS,
        "pass": len(capable) >= MIN_G1_CAPABLE_VIEWS,
        "basis": (
            "occlusion pilot 6.6.16: the bench occludes legs only from elevated "
            "views in the az {}-{} sector".format(*G1_SECTOR_AZIMUTH_RANGE_DEG)
        ),
    }


def assert_gt_free(evidence_keys: Sequence[str]) -> None:
    """Refuse any evidence 6.6.5 does not allow into the selection."""
    for key in evidence_keys:
        lowered = str(key).lower()
        for banned in FORBIDDEN_EVIDENCE:
            if banned in lowered:
                raise CameraBankError(
                    "{!r} is not GT-free evidence; 6.6.5 forbids it in view "
                    "selection.".format(key)
                )
        if lowered not in ALLOWED_EVIDENCE:
            raise CameraBankError(
                "{!r} is not one of the allowed GT-free evidence kinds "
                "{}.".format(key, list(ALLOWED_EVIDENCE))
            )


def cell_measured_counts(
    availability: Mapping,
    subset: Sequence[str],
    scenario_targets: Mapping = None,
) -> dict:
    """Measured-view count per (character, scenario, frame, bone) target cell.

    ``availability`` is ``{character: {scenario: {frame: {view: {joint: bool}}}}}``
    where the boolean says the endpoint was measurable from that view.
    """
    subset = list(subset)
    if not subset:
        raise CameraBankError("An empty subset has nothing to measure.")
    worst = None
    worst_cell = None
    per_scenario_worst = {}
    cells = 0
    for character, scenarios in sorted(availability.items()):
        for scenario, frames in sorted(scenarios.items()):
            for frame, views in sorted(frames.items()):
                missing = [name for name in subset if name not in views]
                if missing:
                    raise CameraBankError(
                        "{}/{} frame {} has no availability for {}.".format(
                            character, scenario, frame, missing
                        )
                    )
                targets = (
                    scenario_targets.get(scenario)
                    if scenario_targets is not None
                    else None
                )
                for bone, (start, end) in BONES.items():
                    if targets is not None and bone not in targets:
                        continue
                    measured = sum(
                        1
                        for name in subset
                        if views[name].get(start) and views[name].get(end)
                    )
                    cells += 1
                    cell = (character, scenario, frame, bone)
                    if worst is None or measured < worst:
                        worst = measured
                        worst_cell = cell
                    key = scenario
                    if key not in per_scenario_worst or measured < per_scenario_worst[key][0]:
                        per_scenario_worst[key] = (measured, cell)
    return {
        "cell_count": cells,
        "min_measured_views": worst,
        "worst_cell": list(worst_cell) if worst_cell else None,
        "per_scenario_worst": {
            scenario: {"min_measured_views": value, "cell": list(cell)}
            for scenario, (value, cell) in per_scenario_worst.items()
        },
        "meets_minimum": bool(worst is not None and worst >= MIN_MEASURED_VIEWS),
    }


def angular_spread(views: Sequence[Mapping]) -> float:
    """Smallest angular gap between chosen azimuths, as a diversity tiebreak."""
    azimuths = sorted(_wrap_azimuth(view["azimuth_deg"]) for view in views)
    if len(azimuths) < 2:
        return 360.0
    gaps = [
        azimuths[index + 1] - azimuths[index] for index in range(len(azimuths) - 1)
    ]
    gaps.append(360.0 - azimuths[-1] + azimuths[0])
    return min(gaps)


def select_common_five(
    views: Sequence[Mapping],
    availability: Mapping,
    evidence_keys: Sequence[str] = ALLOWED_EVIDENCE,
    scenario_targets: Mapping = SCENARIO_TARGET_BONES,
) -> dict:
    """Choose one five-view subset for every character, on GT-free evidence.

    Ranking, in order: the hard three-view minimum, then the worst-cell margin,
    then G1 capability, then angular spread, then the view names.  Nothing in
    that list depends on accuracy against ground truth.
    """
    assert_gt_free(evidence_keys)
    if len(views) != CANDIDATE_COUNT:
        raise CameraBankError(
            "Selection expects {} candidates, received {}.".format(
                CANDIDATE_COUNT, len(views)
            )
        )
    by_name = {view["name"]: view for view in views}
    ranked = []
    for combination in itertools.combinations(sorted(by_name), SELECTED_COUNT):
        chosen = [by_name[name] for name in combination]
        counts = cell_measured_counts(
            availability, combination, scenario_targets
        )
        ranked.append(
            {
                "subset": list(combination),
                "min_measured_views": counts["min_measured_views"],
                "meets_minimum": counts["meets_minimum"],
                "g1_capable_count": sum(1 for view in chosen if g1_capable(view)),
                "angular_spread_deg": angular_spread(chosen),
                "per_scenario_worst": counts["per_scenario_worst"],
                "worst_cell": counts["worst_cell"],
            }
        )
    ranked.sort(
        key=lambda entry: (
            -entry["min_measured_views"],
            -entry["g1_capable_count"],
            -entry["angular_spread_deg"],
            entry["subset"],
        )
    )
    usable = [entry for entry in ranked if entry["meets_minimum"]]
    best = usable[0] if usable else None
    return {
        "bank_tag": BANK_TAG,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "evidence_used": list(evidence_keys),
        "scenario_targets": (
            {scenario: list(bones) for scenario, bones in scenario_targets.items()}
            if scenario_targets
            else None
        ),
        "gt_error_used": False,
        "candidate_count": len(views),
        "subset_count": len(ranked),
        "selected": best,
        "ranking_head": ranked[:5],
        "usable_subset_count": len(usable),
        "pass": best is not None,
        "failure": (
            None
            if best
            else (
                "no five-view subset keeps {} measured views for every target "
                "cell; 6.6.5 requires revising the candidate bank rather than "
                "relaxing the minimum".format(MIN_MEASURED_VIEWS)
            )
        ),
        "note": (
            "One subset for all four characters. Ranked on availability, "
            "geometry and coverage only; no GT error was read."
        ),
    }
