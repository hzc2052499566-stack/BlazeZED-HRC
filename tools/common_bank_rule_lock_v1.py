"""Normative constants and gates for the 18->9->5 common camera freeze.

This module is the executable form of AGENTS.md 6.6.20.  6.6.20 is the
specification; nothing here may add a threshold, reorder a tie-break or soften a
gate, and the rule lock JSON built from this module has to be frozen write-once
*before* any new survey result exists.

What lives here, and only here:

* the eighteen surveyed candidates, so the renderer, the selector and the freeze
  record cannot drift apart;
* the fixed three-batch contract and its cross-batch anchor;
* the raw-replay configuration and the human-surface plausibility window;
* ``T_joint`` / ``T``: which losses count as table-caused and which do not;
* the recoverable-event shape, the hard gates on ``F`` and ``R``, the nine-key
  ordering and the M0 rule;
* the 18x18 pairwise angle table, quantised to integer nanodegrees, because
  6.6.20 forbids the selector recomputing angles in platform floating point.

No Isaac Sim or ``pxr`` dependency: the Isaac side renders, this side decides.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Mapping, Sequence

from fs_cts5_camera_bank_v1 import BONES

RULE_TAG = "fs_cts5_common_bank_18_9_5_rule_lock_v1"
RULE_SOURCE = "AGENTS.md 6.6.20"
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False

CHARACTERS = ("F01", "F02", "M01", "M02")
FRAME_COUNT = 241
FRAMES = tuple(range(FRAME_COUNT))

# G2 is measured but non-binding: the robot currently takes 0/16 target-arm
# cells, so scoring on it would be scoring on a scenario identical to G0.
SELECTION_SCENARIOS = ("G0", "G1")
DIAGNOSTIC_SCENARIOS = ("G2",)
LEG_BONES = ("left_thigh", "right_thigh", "left_shank", "right_shank")
ARM_BONES = ("left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm")
TARGET_BONES = {"G0": LEG_BONES + ARM_BONES, "G1": LEG_BONES}

SURVEY_COUNT = 18
BANK_COUNT = 9
SELECTED_COUNT = 5
RESERVE_COUNT = BANK_COUNT - SELECTED_COUNT
# The deployment floor is three measured views; the freeze keeps one in hand.
DEPLOYMENT_FLOOR = 3
REQUIRED_MEASURED_VIEWS = 4

BANK_RADIUS_M = 3.5
RESOLUTION = (640, 480)
FOCAL_LENGTH = 18.0
HORIZONTAL_APERTURE = 20.955

# The nine already surveyed in common_bank_v1/attempt_02, carried over with their
# geometry untouched so the two matrices stay comparable, plus nine probes
# registered before the new survey.  Original membership confers no priority.
SURVEY_LAYOUT = (
    {"name": "az000_el15", "azimuth_deg": 0.0, "elevation_deg": 15.0, "role": "g1_sector_elevated"},
    {"name": "az060_el15", "azimuth_deg": 60.0, "elevation_deg": 15.0, "role": "g1_sector_elevated"},
    {"name": "az120_el15", "azimuth_deg": 120.0, "elevation_deg": 15.0, "role": "g1_sector_elevated"},
    {"name": "az000_el00", "azimuth_deg": 0.0, "elevation_deg": 0.0, "role": "g1_sector_level"},
    {"name": "az060_el00", "azimuth_deg": 60.0, "elevation_deg": 0.0, "role": "g1_sector_level"},
    {"name": "az180_el10", "azimuth_deg": 180.0, "elevation_deg": 10.0, "role": "opposite"},
    {"name": "az240_el10", "azimuth_deg": 240.0, "elevation_deg": 10.0, "role": "opposite"},
    {"name": "az300_el10", "azimuth_deg": 300.0, "elevation_deg": 10.0, "role": "flank"},
    {"name": "az300_el20", "azimuth_deg": 300.0, "elevation_deg": 20.0, "role": "flank_elevated"},
    {"name": "az030_el00", "azimuth_deg": 30.0, "elevation_deg": 0.0, "role": "g1_sector_level_infill"},
    {"name": "az030_el15", "azimuth_deg": 30.0, "elevation_deg": 15.0, "role": "g1_sector_elevated_infill"},
    {"name": "az090_el00", "azimuth_deg": 90.0, "elevation_deg": 0.0, "role": "lateral_falloff_probe"},
    {"name": "az150_el10", "azimuth_deg": 150.0, "elevation_deg": 10.0, "role": "lateral_falloff_probe"},
    {"name": "az210_el10", "azimuth_deg": 210.0, "elevation_deg": 10.0, "role": "clean_arc_infill"},
    {"name": "az270_el10", "azimuth_deg": 270.0, "elevation_deg": 10.0, "role": "lateral_falloff_probe"},
    {"name": "az330_el10", "azimuth_deg": 330.0, "elevation_deg": 10.0, "role": "lateral_falloff_probe"},
    {"name": "az180_el00", "azimuth_deg": 180.0, "elevation_deg": 0.0, "role": "clean_arc_low"},
    {"name": "az240_el20", "azimuth_deg": 240.0, "elevation_deg": 20.0, "role": "clean_arc_elevated"},
)
SURVEY_VIEWS = tuple(entry["name"] for entry in SURVEY_LAYOUT)

# About nine render products per step is the reliable ceiling on this machine, so
# eighteen views are rendered in three batches rather than mounted at once:
# dropped frames must never be able to masquerade as unavailability.  The anchor
# is in every batch purely to prove the batches agree; it earns no selection
# priority and is not the M0 anchor.
SURVEY_ANCHOR = "az000_el00"
BATCHES = (
    (
        "batch_1",
        (
            "az000_el00",
            "az000_el15",
            "az060_el15",
            "az120_el15",
            "az060_el00",
            "az180_el10",
            "az240_el10",
        ),
    ),
    (
        "batch_2",
        (
            "az000_el00",
            "az300_el10",
            "az300_el20",
            "az030_el00",
            "az030_el15",
            "az090_el00",
            "az150_el10",
        ),
    ),
    (
        "batch_3",
        (
            "az000_el00",
            "az210_el10",
            "az270_el10",
            "az330_el10",
            "az180_el00",
            "az240_el20",
        ),
    ),
)
ANCHOR_KEPT_FROM_BATCH = "batch_1"
TRAVERSAL_ORDER = ("character", "scenario", "batch", "frame")

# Raw replay.  Any of these that cannot be reused verbatim stops the survey
# before a single RGB-D frame is produced; silent substitution is forbidden.
RAW_REPLAY = {
    "model": "blazepose_lite",
    "roi_fraction": 0.65,
    "running_mode": "video",
    "smooth_landmarks": True,
    "warmup_frames": 0,
    "depth_sampler": "median_7x7",
    "endpoint_source": "raw_measured",
    "forbidden": (
        "K2",
        "K4",
        "temporal_completion",
        "offsets",
        "fusion_weights",
        "gt_coordinates",
        "gt_error",
    ),
    "instance_scope": "character_scenario_view",
    "deterministic_replay_required": True,
}

# Frozen V3 permissive human-surface window.
HUMAN_SURFACE = {
    "requires_both_endpoints_raw_valid": True,
    "camera_forward_min_m": 2.0,
    "camera_forward_max_m": 5.0,
    "bone_length_min_m": 0.10,
    "bone_length_max_m": 0.80,
}

# T_joint attribution.  Same pixel rounding and same 7x7 window as the depth
# sampler, median over positive finite depths only, three medians compared:
# G0 human, G1 human+workbench, workbench-only.
TABLE_ATTRIBUTION = {
    "window": "median_7x7",
    "positive_finite_only": True,
    "min_depth_advance_m": 0.02,
    "max_surface_match_m": 0.05,
    "no_workbench_median_means_no_attribution": True,
}

# Recoverable event: measurable, then table-occluded, then measurable again, all
# inside one march window, contiguous and without gaps.
MARCH_WINDOWS = {"left": (75, 150), "right": (155, 230)}
EVENT_MIN_CLEAR_FRAMES = 5
EVENT_MIN_OCCLUDED_FRAMES = 30
EVENT_LEG_SIDES = ("left", "right")
EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER = 2
SIDE_BONES = {
    "left": ("left_thigh", "left_shank"),
    "right": ("right_thigh", "right_shank"),
}

ORDERING_KEYS = (
    "max_min_measured_views",
    "min_cells_at_that_minimum",
    "max_min_view_character_bone_clean_rate",
    "max_min_pairwise_angle_in_f",
    "max_min_over_f_of_best_swap_min_measured",
    "max_valid_ordered_swap_pairs",
    "max_passing_five_subsets_within_b",
    "max_min_pairwise_angle_in_b",
    "view_id_lexicographic",
)
M0_ORDERING_KEYS = (
    "max_min_character_bone_clean_rate",
    "max_total_g0_measured_fraction",
    "min_angle_to_historical_az000_el00",
    "view_id_lexicographic",
)
HISTORICAL_M0_VIEW = "az000_el00"

ANGLE_QUANTUM = Decimal("1e-9")
NANODEGREES_PER_DEGREE = 1000000000

STATEMENT_BOUNDARY = (
    "cohort-designed common camera bank evaluated across four characters; the "
    "layout itself is not held-out generalisation to unseen characters"
)


class RuleLockError(RuntimeError):
    """Raised when an input cannot satisfy the frozen contract."""


def _unit_camera_to_aim(azimuth_deg: float, elevation_deg: float) -> tuple:
    """Unit vector from the camera toward the aim point.

    The eye sits at ``aim + radius * (cos el cos az, cos el sin az, sin el)``, so
    the camera-to-aim direction is the negation of that offset direction and does
    not depend on the aim or the radius at all.  That is why this table can be
    frozen before the survey names an aim point.
    """
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    return (
        -math.cos(elevation) * math.cos(azimuth),
        -math.cos(elevation) * math.sin(azimuth),
        -math.sin(elevation),
    )


def _angle_degrees(first: Sequence[float], second: Sequence[float]) -> float:
    """Angle between two unit vectors, via the half-angle tangent form.

    ``acos(dot)`` is the obvious formula and the wrong one here: near zero angle
    it loses half the mantissa, and it put 854 nanodegrees on this table's
    diagonal where the answer is exactly zero.  ``2*atan2(|a-b|, |a+b|)`` is
    well-conditioned across the whole range, so a nanodegree quantum means what
    it says at both ends.
    """
    difference = math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))
    total = math.sqrt(sum((a + b) ** 2 for a, b in zip(first, second)))
    return math.degrees(2.0 * math.atan2(difference, total))


def _quantise_nanodegrees(degrees: float) -> int:
    return int(
        (Decimal(repr(float(degrees))) / ANGLE_QUANTUM).quantize(
            Decimal(1), rounding=ROUND_HALF_EVEN
        )
    )


def pairwise_angle_table(layout: Sequence[Mapping] = SURVEY_LAYOUT) -> dict:
    """Every pair's 3-D angle, in integer nanodegrees.

    Frozen into the rule lock and compared as integers thereafter: 6.6.20 forbids
    the selector recomputing these at run time, where a different platform's
    floating point could reorder two otherwise tied candidates.
    """
    vectors = {
        entry["name"]: _unit_camera_to_aim(
            entry["azimuth_deg"], entry["elevation_deg"]
        )
        for entry in layout
    }
    table = {}
    for first in sorted(vectors):
        row = {}
        for second in sorted(vectors):
            row[second] = _quantise_nanodegrees(
                _angle_degrees(vectors[first], vectors[second])
            )
        table[first] = row
    return table


def min_pairwise_angle(views: Sequence[str], table: Mapping) -> int:
    """Smallest angle between any two distinct views, in nanodegrees."""
    ordered = sorted(views)
    if len(ordered) < 2:
        raise RuleLockError("A pairwise angle needs at least two views.")
    return min(
        table[first][second]
        for index, first in enumerate(ordered)
        for second in ordered[index + 1 :]
    )


def _runs(flags: Sequence[int], value: int, lo: int, hi: int) -> list:
    """Maximal runs of ``value`` inside the closed frame window ``[lo, hi]``."""
    runs = []
    start = None
    for frame in range(lo, hi + 1):
        if int(flags[frame]) == value:
            if start is None:
                start = frame
        elif start is not None:
            runs.append((start, frame - 1))
            start = None
    if start is not None:
        runs.append((start, hi))
    return runs


def recoverable_event(
    measurable: Sequence[int],
    table_occluded: Sequence[int],
    window: Sequence[int],
    min_clear: int = EVENT_MIN_CLEAR_FRAMES,
    min_occluded: int = EVENT_MIN_OCCLUDED_FRAMES,
) -> dict:
    """Is there a clear -> table-occluded -> clear event inside this window?

    The occluded stretch has to be a *maximal* run: ``T`` implies the bone is not
    measurable, so a shorter slice of a longer run could never be flanked by
    measurable frames.  Both flanks must lie inside the window too -- an event
    that only recovers after the march has ended is not a recovery this scenario
    can claim.
    """
    lo, hi = int(window[0]), int(window[1])
    if lo < 0 or hi >= len(measurable) or lo >= hi:
        raise RuleLockError("March window {} is outside the motion.".format(window))
    for start, end in _runs(table_occluded, 1, lo, hi):
        if end - start + 1 < min_occluded:
            continue
        before = (start - min_clear, start - 1)
        after = (end + 1, end + min_clear)
        if before[0] < lo or after[1] > hi:
            continue
        if all(int(measurable[frame]) == 1 for frame in range(before[0], before[1] + 1)) and all(
            int(measurable[frame]) == 1 for frame in range(after[0], after[1] + 1)
        ):
            return {
                "found": True,
                "clear_before": list(before),
                "occluded": [start, end],
                "clear_after": list(after),
            }
    return {"found": False}


def layout_sha256(layout: Sequence[Mapping] = SURVEY_LAYOUT) -> str:
    """Hash of the canonical layout, so a renamed or nudged view is visible."""
    payload = json.dumps(
        [
            {
                "name": entry["name"],
                "azimuth_deg": float(entry["azimuth_deg"]),
                "elevation_deg": float(entry["elevation_deg"]),
                "role": entry["role"],
            }
            for entry in layout
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_batches(batches: Sequence = BATCHES, layout: Sequence[Mapping] = SURVEY_LAYOUT) -> dict:
    """The batches must cover the survey exactly once, plus the anchor."""
    names = [entry["name"] for entry in layout]
    covered = []
    for label, views in batches:
        if SURVEY_ANCHOR not in views:
            raise RuleLockError("{} does not carry the anchor.".format(label))
        if len(set(views)) != len(views):
            raise RuleLockError("{} repeats a view.".format(label))
        covered.extend(view for view in views if view != SURVEY_ANCHOR)
    if sorted(covered) != sorted(view for view in names if view != SURVEY_ANCHOR):
        raise RuleLockError(
            "The batches do not partition the survey exactly once."
        )
    return {
        "batches": {label: list(views) for label, views in batches},
        "anchor": SURVEY_ANCHOR,
        "anchor_kept_from": ANCHOR_KEPT_FROM_BATCH,
        "render_products_per_step": {label: len(views) for label, views in batches},
    }


def rule_lock_document(
    selector_hashes: Mapping,
    scene_hashes: Mapping,
    layout: Sequence[Mapping] = SURVEY_LAYOUT,
) -> dict:
    """The machine-readable rule lock, ready to be frozen write-once."""
    if len(layout) != SURVEY_COUNT:
        raise RuleLockError(
            "The survey holds exactly {} candidates.".format(SURVEY_COUNT)
        )
    if sorted(scene_hashes) != sorted(CHARACTERS):
        raise RuleLockError("Scene hashes must cover exactly {}.".format(CHARACTERS))
    return {
        "record": RULE_TAG,
        "rule_source": RULE_SOURCE,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "gt_error_used": False,
        "candidates": [dict(entry) for entry in layout],
        "candidate_count": len(layout),
        "layout_sha256": layout_sha256(layout),
        "geometry": {
            "radius_m": BANK_RADIUS_M,
            "resolution": list(RESOLUTION),
            "focal_length": FOCAL_LENGTH,
            "horizontal_aperture": HORIZONTAL_APERTURE,
            "shared_across_characters": True,
        },
        "selection_unit": {
            "characters": list(CHARACTERS),
            "frame_count": FRAME_COUNT,
            "selection_scenarios": list(SELECTION_SCENARIOS),
            "diagnostic_scenarios": list(DIAGNOSTIC_SCENARIOS),
            "target_bones": {key: list(value) for key, value in TARGET_BONES.items()},
        },
        "survey_contract": dict(
            check_batches(BATCHES, layout), traversal_order=list(TRAVERSAL_ORDER)
        ),
        "raw_replay": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in RAW_REPLAY.items()
        },
        "human_surface": dict(HUMAN_SURFACE),
        "table_attribution": dict(TABLE_ATTRIBUTION),
        "hard_gates": {
            "deployment_floor": DEPLOYMENT_FLOOR,
            "required_measured_views": REQUIRED_MEASURED_VIEWS,
            "march_windows": {key: list(value) for key, value in MARCH_WINDOWS.items()},
            "event_min_clear_frames": EVENT_MIN_CLEAR_FRAMES,
            "event_min_occluded_frames": EVENT_MIN_OCCLUDED_FRAMES,
            "event_leg_sides": list(EVENT_LEG_SIDES),
            "event_min_distinct_views_per_character": EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER,
            "side_bones": {key: list(value) for key, value in SIDE_BONES.items()},
            "reserve_must_cover_every_selected_view": True,
            "reserve_member_must_have_a_valid_swap": True,
        },
        "ordering": {
            "bank_count": BANK_COUNT,
            "selected_count": SELECTED_COUNT,
            "reserve_count": RESERVE_COUNT,
            "keys": list(ORDERING_KEYS),
            "m0_keys": list(M0_ORDERING_KEYS),
            "historical_m0_view": HISTORICAL_M0_VIEW,
            "pair_space": math.comb(SURVEY_COUNT, SELECTED_COUNT)
            * math.comb(SURVEY_COUNT - SELECTED_COUNT, RESERVE_COUNT),
            "exhaustive_required": True,
        },
        "pairwise_angle_nanodegrees": pairwise_angle_table(layout),
        "angle_quantisation": {
            "unit": "nanodegree",
            "rounding": "ROUND_HALF_EVEN",
            "recomputation_at_selection_time_forbidden": True,
        },
        "selector_hashes": dict(selector_hashes),
        "scene_sha256": dict(scene_hashes),
        "statement_boundary": STATEMENT_BOUNDARY,
        "failure_policy": (
            "Any incomplete input, absent qualifying pair, failed raw preflight or "
            "non-reproducible recomputation is saved as a FAIL with "
            "formal_capture_authorized=false.  After results exist, only a new "
            "candidate layout in a new attempt may reopen the search; the gates, "
            "thresholds, event shape and ordering stay as frozen here."
        ),
    }
