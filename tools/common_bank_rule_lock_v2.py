"""Rule lock v2: revised layout, relaxed event minimum, per-scenario floor.

v1 (``common_bank_rule_lock_v1``) was frozen before any new result existed and is
preserved unchanged.  The 241-frame geometric probe then measured v1 to be
**unexecutable** (AGENTS.md 6.6.22): zero five-view subsets satisfied its gates,
for three independent reasons.  6.6.20 permits a new rule version for exactly
that finding, before any 18-view RGB-D exists, with the user's explicit
confirmation -- which is where this file comes from.

**Three changes, and the honest provenance of each.**

1. *Layout.*  Seven candidates that the probe measured blind on 30%+ of all
   target cells are replaced.  Three of them (``az000_el15``, ``az030_el15``,
   ``az060_el15``) were the "strong occluders" -- they take the legs for all 241
   frames and never give them back, which is permanent blindness rather than a
   recoverable event.  The seven replacements sample the elevation cliff between
   el 00 and el 15 in the occluding band, which v1 never sampled: at az 000 the
   bench takes 323 leg cells at el 00 and 3,556 at el 15, and nothing in between
   was measured.  This change is a candidate-layout revision, which 6.6.5
   prescribes and which touches no threshold.

2. *Event minimum, 30 -> 20 frames.*  **Post-hoc.**  The probe's numbers were
   known when this was chosen: F02's longest transient bench occlusion is 28
   frames on the right leg and 24 on the left, so 30 was unreachable for it.  Two
   things make 20 defensible rather than fitted: the carrier set is *identical*
   for every value in [8, 24], so this is a plateau and not a knife edge; and 20
   frames is a third of a second at 60 fps, still ten times the frozen
   ``<=2 frame`` recovery gate.  It is recorded as post-hoc regardless.

3. *Availability floor, per scenario.*  v1 demanded four measured views in both
   G0 and G1.  G1 is the scenario built to remove views, so requiring a margin
   above the deployment floor there is self-defeating: a view that occludes a leg
   is, by construction, a view that cannot measure it.  G0, the clean control,
   keeps the margin at four; G1 uses the deployment floor of three, which is
   exactly what M3 needs in order not to abstain.  Chosen after seeing v1 fail,
   so also recorded as post-hoc.

Everything not listed above -- the batch discipline, raw replay, human-surface
window, ``T_joint``, the event shape, the R gates, the nine-key ordering, the M0
rule -- is inherited from v1 unchanged.

A v2 *selector* implementing the per-scenario floor does not exist yet and must
be frozen before any selection is run; this lock fixes the thresholds so that
writing it cannot change them.
"""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

import common_bank_rule_lock_v1 as v1
from common_bank_rule_lock_v1 import (  # noqa: F401  (re-exported unchanged)
    ANGLE_QUANTUM,
    CHARACTERS,
    CLASSIFICATION,
    DIAGNOSTIC_SCENARIOS,
    EVENT_LEG_SIDES,
    EVENT_MIN_CLEAR_FRAMES,
    EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER,
    FOCAL_LENGTH,
    FORMAL_CAPTURE_AUTHORIZED,
    FRAME_COUNT,
    FRAMES,
    HISTORICAL_M0_VIEW,
    HORIZONTAL_APERTURE,
    HUMAN_SURFACE,
    M0_ORDERING_KEYS,
    MARCH_WINDOWS,
    NANODEGREES_PER_DEGREE,
    ORDERING_KEYS,
    RAW_REPLAY,
    RESOLUTION,
    SELECTION_SCENARIOS,
    SIDE_BONES,
    STATEMENT_BOUNDARY,
    TABLE_ATTRIBUTION,
    TARGET_BONES,
    RuleLockError,
    layout_sha256,
    min_pairwise_angle,
    pairwise_angle_table,
    recoverable_event,
)

RULE_TAG = "fs_cts5_common_bank_18_9_5_rule_lock_v2"
RULE_SOURCE = "AGENTS.md 6.6.20 as amended by 6.6.23"
SUPERSEDES = v1.RULE_TAG

BANK_RADIUS_M = v1.BANK_RADIUS_M
SURVEY_COUNT = v1.SURVEY_COUNT
BANK_COUNT = v1.BANK_COUNT
SELECTED_COUNT = v1.SELECTED_COUNT
RESERVE_COUNT = v1.RESERVE_COUNT
DEPLOYMENT_FLOOR = v1.DEPLOYMENT_FLOOR

# Change 2.  See the module docstring for why this is recorded as post-hoc.
EVENT_MIN_OCCLUDED_FRAMES = 20

# Change 3.  The clean control keeps a view in hand; the scenario designed to
# take views away is held to the deployment floor.
MEASURED_VIEWS_FLOOR = {"G0": 4, "G1": DEPLOYMENT_FLOOR}

# Change 1.  Eleven carried over from v1, seven replaced.
#
# Dropped, with the probe's measured blindness over 11,568 target cells:
#   az120_el15 (7,560)  az300_el20 (5,851)  az300_el10 (5,021)
#   az000_el15 (4,156)  az060_el15 (3,940)  az030_el15 (3,856)
#   az330_el10 (3,466)
# The three el15 views in the occluding sector were v1's strongest occluders and
# its worst measurers at once: they take every leg cell in all 241 frames and
# never return them, so they can only ever be permanent blindness.
DROPPED_FROM_V1 = (
    "az000_el15",
    "az030_el15",
    "az060_el15",
    "az120_el15",
    "az300_el10",
    "az300_el20",
    "az330_el10",
)
SURVEY_LAYOUT = (
    # Carried over: the occluding band at ground level, where every event v1
    # found came from.
    {"name": "az000_el00", "azimuth_deg": 0.0, "elevation_deg": 0.0, "role": "occluding_band_level"},
    {"name": "az030_el00", "azimuth_deg": 30.0, "elevation_deg": 0.0, "role": "occluding_band_level"},
    {"name": "az060_el00", "azimuth_deg": 60.0, "elevation_deg": 0.0, "role": "occluding_band_level"},
    {"name": "az090_el00", "azimuth_deg": 90.0, "elevation_deg": 0.0, "role": "occluding_band_edge"},
    # New: the elevation cliff v1 never sampled.  Between el 00 and el 15 the
    # bench goes from taking 323 leg cells to 3,556 at az 000; a moderate
    # elevation is where an occlusion long enough to be an event might still
    # leave the leg measurable on both sides of it.
    {"name": "az000_el05", "azimuth_deg": 0.0, "elevation_deg": 5.0, "role": "occluding_band_cliff"},
    {"name": "az000_el10", "azimuth_deg": 0.0, "elevation_deg": 10.0, "role": "occluding_band_cliff"},
    {"name": "az030_el05", "azimuth_deg": 30.0, "elevation_deg": 5.0, "role": "occluding_band_cliff"},
    {"name": "az030_el10", "azimuth_deg": 30.0, "elevation_deg": 10.0, "role": "occluding_band_cliff"},
    {"name": "az060_el05", "azimuth_deg": 60.0, "elevation_deg": 5.0, "role": "occluding_band_cliff"},
    {"name": "az060_el10", "azimuth_deg": 60.0, "elevation_deg": 10.0, "role": "occluding_band_cliff"},
    {"name": "az090_el05", "azimuth_deg": 90.0, "elevation_deg": 5.0, "role": "occluding_band_cliff"},
    # Carried over: the measuring arc.  az210_el10 missed nothing at all across
    # 11,568 cells; az240_el10 missed two, az240_el20 eight.
    {"name": "az150_el10", "azimuth_deg": 150.0, "elevation_deg": 10.0, "role": "measuring_arc"},
    {"name": "az180_el00", "azimuth_deg": 180.0, "elevation_deg": 0.0, "role": "measuring_arc"},
    {"name": "az180_el10", "azimuth_deg": 180.0, "elevation_deg": 10.0, "role": "measuring_arc"},
    {"name": "az210_el10", "azimuth_deg": 210.0, "elevation_deg": 10.0, "role": "measuring_arc"},
    {"name": "az240_el10", "azimuth_deg": 240.0, "elevation_deg": 10.0, "role": "measuring_arc"},
    {"name": "az240_el20", "azimuth_deg": 240.0, "elevation_deg": 20.0, "role": "measuring_arc"},
    {"name": "az270_el10", "azimuth_deg": 270.0, "elevation_deg": 10.0, "role": "measuring_arc"},
)
SURVEY_VIEWS = tuple(entry["name"] for entry in SURVEY_LAYOUT)

SURVEY_ANCHOR = v1.SURVEY_ANCHOR
ANCHOR_KEPT_FROM_BATCH = v1.ANCHOR_KEPT_FROM_BATCH
TRAVERSAL_ORDER = v1.TRAVERSAL_ORDER
BATCHES = (
    (
        "batch_1",
        (
            "az000_el00",
            "az000_el05",
            "az000_el10",
            "az030_el00",
            "az030_el05",
            "az030_el10",
            "az060_el00",
        ),
    ),
    (
        "batch_2",
        (
            "az000_el00",
            "az060_el05",
            "az060_el10",
            "az090_el00",
            "az090_el05",
            "az150_el10",
            "az180_el00",
        ),
    ),
    (
        "batch_3",
        (
            "az000_el00",
            "az180_el10",
            "az210_el10",
            "az240_el10",
            "az240_el20",
            "az270_el10",
        ),
    ),
)

# What was decided after seeing a result, and what the result was.  A rule
# version written in response to a measurement cannot claim to be pre-registered;
# it can only be honest about which parts are not.
POST_HOC = {
    "written_after": "common_bank_v1 event feasibility probe, attempt_01 (6.6.22)",
    "authorised_by": "user, 2026-08-19, after being shown that v1 has no solution",
    "pre_registered_in_v2": [
        "everything inherited from v1 unchanged",
        "the revised layout: chosen on measured blindness, not on any event outcome",
    ],
    "post_hoc_in_v2": {
        "event_min_occluded_frames": {
            "was": v1.EVENT_MIN_OCCLUDED_FRAMES,
            "now": EVENT_MIN_OCCLUDED_FRAMES,
            "known_when_chosen": (
                "F02's longest transient bench occlusion measured 28 frames "
                "(right leg) and 24 frames (left leg), so 30 was unreachable."
            ),
            "mitigation": (
                "The carrier set is identical for every threshold in [8, 24], so "
                "the choice sits on a plateau rather than on F02's number; 20 "
                "frames is 1/3 s at 60 fps and still ten times the frozen "
                "two-frame recovery gate."
            ),
        },
        "measured_views_floor": {
            "was": {"G0": v1.REQUIRED_MEASURED_VIEWS, "G1": v1.REQUIRED_MEASURED_VIEWS},
            "now": dict(MEASURED_VIEWS_FLOOR),
            "known_when_chosen": (
                "Only 8 of 8,568 subsets met a floor of 4 everywhere, and none "
                "of those contained two event-carrying views."
            ),
            "mitigation": (
                "G1 is the scenario built to remove views, and a view that "
                "occludes a leg cannot measure it; the clean control keeps the "
                "margin, and G1's floor of 3 is what M3 needs not to abstain."
            ),
        },
    },
    "not_changed": [
        "scene geometry, which 6.6.5 forbids as a response to no-solution",
        "the event shape, the R gates, the ordering keys and the M0 rule",
        "the deployment floor of three measured endpoints",
    ],
}


def check_batches(batches: Sequence = BATCHES, layout: Sequence[Mapping] = SURVEY_LAYOUT) -> dict:
    """Same partition discipline as v1, over v2's layout."""
    names = [entry["name"] for entry in layout]
    covered = []
    for label, views in batches:
        if SURVEY_ANCHOR not in views:
            raise RuleLockError("{} does not carry the anchor.".format(label))
        if len(set(views)) != len(views):
            raise RuleLockError("{} repeats a view.".format(label))
        covered.extend(view for view in views if view != SURVEY_ANCHOR)
    if sorted(covered) != sorted(view for view in names if view != SURVEY_ANCHOR):
        raise RuleLockError("The batches do not partition the survey exactly once.")
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
    """The v2 lock: v1's document with the three changes and their provenance."""
    if len(layout) != SURVEY_COUNT:
        raise RuleLockError(
            "The survey holds exactly {} candidates.".format(SURVEY_COUNT)
        )
    if sorted(scene_hashes) != sorted(CHARACTERS):
        raise RuleLockError("Scene hashes must cover exactly {}.".format(CHARACTERS))
    document = {
        "record": RULE_TAG,
        "rule_source": RULE_SOURCE,
        "supersedes": SUPERSEDES,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "gt_error_used": False,
        "candidates": [dict(entry) for entry in layout],
        "candidate_count": len(layout),
        "layout_sha256": layout_sha256(layout),
        "dropped_from_v1": list(DROPPED_FROM_V1),
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
            "measured_views_floor": dict(MEASURED_VIEWS_FLOOR),
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
        "provenance": dict(POST_HOC),
        "selector_hashes": dict(selector_hashes),
        "scene_sha256": dict(scene_hashes),
        "statement_boundary": STATEMENT_BOUNDARY,
        "selector_status": (
            "The v1 selector implements a single measured-views floor and cannot "
            "enforce v2's per-scenario floor.  A v2 selector must be written and "
            "frozen before any selection is run; these thresholds are frozen here "
            "so that writing it cannot change them."
        ),
        "failure_policy": (
            "Any incomplete input, absent qualifying pair, failed raw preflight or "
            "non-reproducible recomputation is saved as a FAIL with "
            "formal_capture_authorized=false.  v1 is preserved as the record of "
            "what was pre-registered before any result existed."
        ),
    }
    return document


def summary() -> str:
    """One-line diff against v1, for logs and steps documents."""
    return json.dumps(
        {
            "supersedes": SUPERSEDES,
            "layout_replaced": len(DROPPED_FROM_V1),
            "event_min_occluded_frames": [
                v1.EVENT_MIN_OCCLUDED_FRAMES,
                EVENT_MIN_OCCLUDED_FRAMES,
            ],
            "measured_views_floor": dict(MEASURED_VIEWS_FLOOR),
        },
        sort_keys=True,
    )
