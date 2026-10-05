"""Rule lock v7: four permanent blinders swapped for four measured carriers.

v1 through v6 are preserved unchanged.  v7 is a **candidate-layout revision** --
the remedy 6.6.5 prescribes when no feasible subset exists -- and it adds no
relaxation.  The post-hoc count stays at four; every gate, threshold, ordering key
and the mandatory-view exemption come from v6 untouched.

**Why.**  The first fully clean geometric screen (6.6.26 section twenty-two)
passed every measurement gate -- anchor identity 0 of 1205 on all four characters
and both scenarios -- and failed the event gate on one cell: F02's left leg
carried no recoverable event from any of v6's eighteen candidates, at any
threshold down to five frames.  The reason was visible in the frame patterns:

    F02 az030_el00   MMMM ... MMMM   76/76 measurable -- the bench never reaches it
    F02 az030_el05   TTTT ... TTTT   76/76 occluded  -- permanently behind it

Zero to seventy-six across five degrees, with nothing sampled in between.  A
targeted sweep of that gap (6.6.26 section twenty-three) found the transition is
gradual and that five of the eight sampled elevations give F02 the shape the gate
wants, with the el 00 and el 05 controls reproducing the 0 and 76 exactly:

    az030_el01   20 clear / 36 occluded / 20 clear
    az030_el02   12 clear / 52 occluded / 12 clear
    az060_el01   27 clear / 22 occluded / 27 clear
    az060_el02   18 clear / 40 occluded / 18 clear
    az060_el03    8 clear / 60 occluded /  8 clear

**What changed.**  Four candidates are dropped, all of them measured on the clean
screen to be *permanent* blinders in G1 -- occluded for every frame of a march
window, for most or all characters, which is blindness rather than a recoverable
event, exactly the property that got the el 15 views dropped back in v2:

    az030_el05  16/16 constant, 3,856 cells unmeasured
    az030_el10  16/16 constant, 3,856
    az060_el05  14/16 constant, 3,888
    az060_el10  12/16 constant, 3,936

``az030_el00`` is *not* dropped despite being constant for 10 of 16: it is F01's
only left-leg carrier, and the rule that would drop it would take that away.

Four of the five measured carriers replace them, chosen to span both azimuths and
both elevations and to cover a range of occlusion lengths (22 to 52 frames) so
that characters of different heights each have one that fits.  ``az060_el03`` is
left out: its flanks are the thinnest at 8 frames, and 8 is close enough to the
five-frame minimum that a slightly different character could lose the event.

**What this does not settle.**  The sweep measured F02 only.  How these four
behave for F01, M01 and M02, and what they cost in clean-scenario measurability,
is what the next four-character screen is for.  Nothing is frozen by adopting
them beyond the layout itself.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import common_bank_rule_lock_v1 as v1
import common_bank_rule_lock_v2 as v2
import common_bank_rule_lock_v3 as v3
import common_bank_rule_lock_v4 as v4
import common_bank_rule_lock_v5 as v5
import common_bank_rule_lock_v6 as v6
from common_bank_rule_lock_v6 import (  # noqa: F401  (re-exported unchanged)
    ANCHOR_KEPT_FROM_BATCH,
    BANK_COUNT,
    BANK_RADIUS_M,
    CHARACTERS,
    CLASSIFICATION,
    COVERAGE_FLOOR,
    COVERAGE_GATE_SOURCE,
    COVERAGE_GATES,
    DEPLOYMENT_FLOOR,
    DIAGNOSTIC_SCENARIOS,
    EVENT_LEG_SIDES,
    EVENT_MIN_CLEAR_FRAMES,
    EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER,
    EVENT_MIN_OCCLUDED_FRAMES,
    FOCAL_LENGTH,
    FORMAL_CAPTURE_AUTHORIZED,
    FRAME_COUNT,
    HISTORICAL_M0_VIEW,
    HORIZONTAL_APERTURE,
    HUMAN_SURFACE,
    M0_ORDERING_KEYS,
    MARCH_WINDOWS,
    MAX_RENDER_PRODUCTS_PER_STEP,
    MIN_SWAPPABLE_SELECTED_VIEWS,
    NANODEGREES_PER_DEGREE,
    ORDERING_KEYS,
    RAW_REPLAY,
    RESERVE_COUNT,
    RESERVE_EXEMPTS_PROVABLY_MANDATORY,
    RESOLUTION,
    SELECTED_COUNT,
    SELECTION_SCENARIOS,
    SIDE_BONES,
    STATEMENT_BOUNDARY,
    SURVEY_ANCHOR,
    SURVEY_COUNT,
    TABLE_ATTRIBUTION,
    TARGET_BONES,
    TRAVERSAL_ORDER,
    RuleLockError,
    coverage,
    coverage_gate_met,
    min_pairwise_angle,
    provably_mandatory,
    recoverable_event,
    reserve_gate_binds,
)

RULE_TAG = "fs_cts5_common_bank_18_9_5_rule_lock_v7"
RULE_SOURCE = "AGENTS.md 6.6.20 as amended by 6.6.23, 6.6.26 and 6.6.27"
SUPERSEDES = v6.RULE_TAG
SUPERSEDES_CHAIN = (
    v1.RULE_TAG,
    v2.RULE_TAG,
    v3.RULE_TAG,
    v4.RULE_TAG,
    v5.RULE_TAG,
    v6.RULE_TAG,
)

DROPPED_FROM_V6 = ("az030_el05", "az030_el10", "az060_el05", "az060_el10")
ADDED_IN_V7 = ("az030_el01", "az030_el02", "az060_el01", "az060_el02")

SURVEY_LAYOUT = tuple(
    entry
    for entry in v6.SURVEY_LAYOUT
    if entry["name"] not in DROPPED_FROM_V6
) + (
    {"name": "az030_el01", "azimuth_deg": 30.0, "elevation_deg": 1.0, "role": "occluding_band_crossing"},
    {"name": "az030_el02", "azimuth_deg": 30.0, "elevation_deg": 2.0, "role": "occluding_band_crossing"},
    {"name": "az060_el01", "azimuth_deg": 60.0, "elevation_deg": 1.0, "role": "occluding_band_crossing"},
    {"name": "az060_el02", "azimuth_deg": 60.0, "elevation_deg": 2.0, "role": "occluding_band_crossing"},
)
SURVEY_VIEWS = tuple(entry["name"] for entry in SURVEY_LAYOUT)

BATCHES = (
    ("batch_1", ("az000_el00", "az030_el00", "az060_el00", "az090_el00")),
    ("batch_2", ("az000_el00", "az000_el05", "az000_el10", "az030_el01")),
    ("batch_3", ("az000_el00", "az030_el02", "az060_el01", "az060_el02")),
    ("batch_4", ("az000_el00", "az090_el05", "az150_el10", "az180_el00")),
    ("batch_5", ("az000_el00", "az180_el10", "az210_el10", "az240_el10")),
    ("batch_6", ("az000_el00", "az240_el20", "az270_el10")),
)

PROVENANCE = {
    "kind": "candidate layout revision",
    "prescribed_by": "AGENTS.md 6.6.5",
    "adds_no_relaxation": True,
    "post_hoc_count_unchanged": 4,
    "written_after": "geometric_screen/attempt_01 and f02_left_elevation/attempt_01",
    "dropped": {
        "az030_el05": "16/16 constant in G1, 3,856 cells unmeasured",
        "az030_el10": "16/16 constant in G1, 3,856 cells unmeasured",
        "az060_el05": "14/16 constant in G1, 3,888 cells unmeasured",
        "az060_el10": "12/16 constant in G1, 3,936 cells unmeasured",
    },
    "kept_despite_being_mostly_constant": {
        "az030_el00": "10/16 constant, but F01's only left-leg carrier",
    },
    "added": {
        "az030_el01": "F02 left: 20 clear / 36 occluded / 20 clear",
        "az030_el02": "F02 left: 12 clear / 52 occluded / 12 clear",
        "az060_el01": "F02 left: 27 clear / 22 occluded / 27 clear",
        "az060_el02": "F02 left: 18 clear / 40 occluded / 18 clear",
    },
    "measured_but_not_added": {
        "az060_el03": (
            "F02 left: 8 clear / 60 occluded / 8 clear.  Flanks of 8 sit close to "
            "the five-frame minimum, so a different character could lose the event."
        )
    },
    "not_yet_known": (
        "The sweep measured F02 only.  How the four added views behave for F01, "
        "M01 and M02, and what they cost in clean-scenario measurability, is what "
        "the next four-character screen answers."
    ),
}


def layout_sha256(layout: Sequence[Mapping] = SURVEY_LAYOUT) -> str:
    return v1.layout_sha256(layout)


def pairwise_angle_table(layout: Sequence[Mapping] = SURVEY_LAYOUT) -> dict:
    return v1.pairwise_angle_table(layout)


def check_batches(batches: Sequence = BATCHES, layout: Sequence[Mapping] = SURVEY_LAYOUT) -> dict:
    return v5.check_batches(batches, layout)


def rule_lock_document(
    selector_hashes: Mapping,
    scene_hashes: Mapping,
    layout: Sequence[Mapping] = SURVEY_LAYOUT,
) -> dict:
    """v6's document on v7's layout.

    v6 is asked for a document on *its own* layout, because its builder validates
    its own batches against whatever layout it is handed and v7 changes both.
    Every layout-dependent field is then replaced, which keeps the two versions
    from having to know about each other's batching.
    """
    if len(layout) != SURVEY_COUNT:
        raise RuleLockError(
            "The survey holds exactly {} candidates.".format(SURVEY_COUNT)
        )
    document = v6.rule_lock_document(selector_hashes, scene_hashes, v6.SURVEY_LAYOUT)
    document["candidates"] = [dict(entry) for entry in layout]
    document["candidate_count"] = len(layout)
    document["record"] = RULE_TAG
    document["rule_source"] = RULE_SOURCE
    document["supersedes"] = SUPERSEDES
    document["supersedes_chain"] = list(SUPERSEDES_CHAIN)
    document["layout_sha256"] = layout_sha256(layout)
    document["pairwise_angle_nanodegrees"] = pairwise_angle_table(layout)
    document["survey_contract"] = dict(
        check_batches(BATCHES, layout), traversal_order=list(TRAVERSAL_ORDER)
    )
    document["dropped_from_v6"] = list(DROPPED_FROM_V6)
    document["added_in_v7"] = list(ADDED_IN_V7)
    document["layout_revision"] = dict(PROVENANCE)
    return document


def summary() -> str:
    return json.dumps(
        {
            "supersedes": SUPERSEDES,
            "layout_swapped": len(DROPPED_FROM_V6),
            "availability_gate": "coverage",
            "coverage_gates": dict(COVERAGE_GATES),
            "event_min_occluded_frames": EVENT_MIN_OCCLUDED_FRAMES,
            "reserve_exempts_provably_mandatory": RESERVE_EXEMPTS_PROVABLY_MANDATORY,
            "batches": len(BATCHES),
        },
        sort_keys=True,
    )
