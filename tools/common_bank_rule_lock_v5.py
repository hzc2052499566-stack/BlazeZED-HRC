"""Rule lock v5: withdraws one post-hoc relaxation, and splits the batches.

v1 through v4 are preserved unchanged.  v5 exists because the first *clean*
geometric matrix (AGENTS.md 6.6.26 section fourteen) showed that one of the four
post-hoc relaxations was never necessary.

**Change 1 -- the availability gate goes back to v2's per-cell floor.**  v3
replaced "every target cell at or above 4 (G0) / 3 (G1)" with a coverage gate
because the per-cell reading admitted *zero* subsets.  That zero was an artifact:
the matrix had been captured with an orchestrator step that advanced the timeline
a frame per call, so G1 was rendered one animation frame after G0 and after the
joint positions.  On the corrected matrix the per-cell floor admits **14**
subsets.  v5 therefore restores it and drops the coverage gate.  This is a
*strengthening*: it removes a relaxation rather than adding one, and it takes the
post-hoc count in this chain from four down to three.

The other three relaxations were re-checked on the same clean matrix and all three
are still load-bearing, so they stay:

* the event minimum of 20 frames -- at 30, F01 and F02 have no left-leg carrier
  at all, and at 25 F02 still has none;
* the G0/G1 split of 4 and 3 -- a flat 4 admits zero subsets;
* the reserve gate's exemption for provably mandatory views -- ``az030_el00``
  remains mandatory, and none of the 14 qualifying subsets has every view
  swappable.

**Change 2 -- six smaller batches instead of three.**  The anchor identity gate
has failed on F01 alone in every round (450, then 12, then 144, then 62 of 482)
while F02, M01 and M02 matched on every comparison.  The clock gate now proves
rendering is a pure function of the time code *within* a batch, so the residue is
a difference *between* batches, and what differs between batches is how many
render products are mounted.  The machine reports 347 MiB of free VRAM and 2.0
GiB of free process memory during a run, which makes mounting fewer products at
once the one physical explanation not yet tested.  Batches go from 7/7/6 to
4/4/4/4/4/3, halving the peak product count; the anchor is still in every batch.

This costs a longer survey -- six sweeps of 241 frames per character instead of
three -- and buys a test of the only remaining hypothesis.  If the anchor still
disagrees at four products per step, batching is not the variable and the gate
itself has to be reconsidered.

Layout, event contract, ordering, M0 and the exemption rule are inherited
unchanged.
"""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

import common_bank_rule_lock_v1 as v1
import common_bank_rule_lock_v2 as v2
import common_bank_rule_lock_v3 as v3
import common_bank_rule_lock_v4 as v4
from common_bank_rule_lock_v4 import (  # noqa: F401  (re-exported unchanged)
    ANCHOR_KEPT_FROM_BATCH,
    BANK_COUNT,
    BANK_RADIUS_M,
    CHARACTERS,
    CLASSIFICATION,
    DEPLOYMENT_FLOOR,
    DIAGNOSTIC_SCENARIOS,
    DROPPED_FROM_V1,
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
    SURVEY_LAYOUT,
    SURVEY_VIEWS,
    TABLE_ATTRIBUTION,
    TARGET_BONES,
    TRAVERSAL_ORDER,
    RuleLockError,
    layout_sha256,
    min_pairwise_angle,
    pairwise_angle_table,
    provably_mandatory,
    recoverable_event,
    reserve_gate_binds,
)

RULE_TAG = "fs_cts5_common_bank_18_9_5_rule_lock_v5"
RULE_SOURCE = "AGENTS.md 6.6.20 as amended by 6.6.23, 6.6.26 and 6.6.27"
SUPERSEDES = v4.RULE_TAG
SUPERSEDES_CHAIN = (v1.RULE_TAG, v2.RULE_TAG, v3.RULE_TAG, v4.RULE_TAG)

# Change 1: v2's per-cell floor, restored.  The coverage gate is withdrawn.
MEASURED_VIEWS_FLOOR = dict(v2.MEASURED_VIEWS_FLOOR)
WITHDRAWN_FROM_V3 = ("coverage gate on measured availability",)

# Change 2: six batches at four products per step instead of three at seven.
MAX_RENDER_PRODUCTS_PER_STEP = 4
BATCHES = (
    ("batch_1", ("az000_el00", "az030_el00", "az060_el00", "az090_el00")),
    ("batch_2", ("az000_el00", "az000_el05", "az000_el10", "az030_el05")),
    ("batch_3", ("az000_el00", "az030_el10", "az060_el05", "az060_el10")),
    ("batch_4", ("az000_el00", "az090_el05", "az150_el10", "az180_el00")),
    ("batch_5", ("az000_el00", "az180_el10", "az210_el10", "az240_el10")),
    ("batch_6", ("az000_el00", "az240_el20", "az270_el10")),
)

POST_HOC = {
    "written_after": "the first clean geometric matrix, event_feasibility_v3/attempt_01",
    "authorised_by": "user, 2026-08-19",
    "still_post_hoc": {
        "event_min_occluded_frames": "30 -> 20 (v2); re-checked clean, still required",
        "measured_views_floor": "4/4 -> 4/3 (v2); re-checked clean, still required",
        "reserve_exempts_provably_mandatory": "v4; re-checked clean, still required",
    },
    "withdrawn_in_v5": {
        "availability_gate": {
            "was": "coverage at the deployment floor against 6.6.6's thresholds (v3)",
            "now": "back to v2's per-cell floor of 4 (G0) and 3 (G1)",
            "why": (
                "v3's premise was that the per-cell floor admitted zero subsets.  "
                "That zero came from a defective capture routine, not from the "
                "geometry: the orchestrator step advanced the timeline one frame "
                "per call, so G1 was rendered a frame after G0.  On the corrected "
                "matrix the per-cell floor admits 14 subsets, so the relaxation is "
                "withdrawn and the post-hoc count drops from four to three."
            ),
        }
    },
    "not_changed": [
        "the deployment floor of three measured endpoints",
        "the layout, event shape and event minimum",
        "the nine-key ordering, the M0 rule and the mandatory-view exemption",
        "scene geometry",
    ],
    "batch_split_rationale": (
        "Not a threshold and not post-hoc reasoning about a result: the anchor "
        "identity gate failed on F01 in every round while the other three "
        "characters matched on every comparison, the clock gate rules out "
        "within-batch nondeterminism, and the machine runs at 347 MiB free VRAM.  "
        "Mounting four products per step instead of seven is the one physical "
        "hypothesis left untested."
    ),
}


def check_batches(batches: Sequence = BATCHES, layout: Sequence[Mapping] = SURVEY_LAYOUT) -> dict:
    """Partition discipline, plus v5's ceiling on products per step."""
    names = [entry["name"] for entry in layout]
    covered = []
    for label, views in batches:
        if SURVEY_ANCHOR not in views:
            raise RuleLockError("{} does not carry the anchor.".format(label))
        if len(set(views)) != len(views):
            raise RuleLockError("{} repeats a view.".format(label))
        if len(views) > MAX_RENDER_PRODUCTS_PER_STEP:
            raise RuleLockError(
                "{} mounts {} products; v5 caps it at {}.".format(
                    label, len(views), MAX_RENDER_PRODUCTS_PER_STEP
                )
            )
        covered.extend(view for view in views if view != SURVEY_ANCHOR)
    if sorted(covered) != sorted(view for view in names if view != SURVEY_ANCHOR):
        raise RuleLockError("The batches do not partition the survey exactly once.")
    return {
        "batches": {label: list(views) for label, views in batches},
        "anchor": SURVEY_ANCHOR,
        "anchor_kept_from": ANCHOR_KEPT_FROM_BATCH,
        "render_products_per_step": {label: len(views) for label, views in batches},
        "max_render_products_per_step": MAX_RENDER_PRODUCTS_PER_STEP,
    }


def rule_lock_document(
    selector_hashes: Mapping,
    scene_hashes: Mapping,
    layout: Sequence[Mapping] = SURVEY_LAYOUT,
) -> dict:
    """v4's document with the coverage gate withdrawn and the batches split."""
    document = v4.rule_lock_document(selector_hashes, scene_hashes, layout)
    document["record"] = RULE_TAG
    document["rule_source"] = RULE_SOURCE
    document["supersedes"] = SUPERSEDES
    document["supersedes_chain"] = list(SUPERSEDES_CHAIN)
    document["survey_contract"] = dict(
        check_batches(BATCHES, layout), traversal_order=list(TRAVERSAL_ORDER)
    )
    gates = dict(document["hard_gates"])
    for key in ("coverage_floor", "coverage_gates", "coverage_gate_source",
                "per_cell_minimum_is_a_preference_not_a_veto"):
        gates.pop(key, None)
    gates["measured_views_floor"] = dict(MEASURED_VIEWS_FLOOR)
    gates["availability_gate"] = "per_cell_floor"
    document["hard_gates"] = gates
    document["provenance"] = dict(POST_HOC)
    document["withdrawn_from_v3"] = list(WITHDRAWN_FROM_V3)
    return document


def summary() -> str:
    return json.dumps(
        {
            "supersedes": SUPERSEDES,
            "availability_gate": "per_cell_floor",
            "measured_views_floor": dict(MEASURED_VIEWS_FLOOR),
            "event_min_occluded_frames": EVENT_MIN_OCCLUDED_FRAMES,
            "reserve_exempts_provably_mandatory": RESERVE_EXEMPTS_PROVABLY_MANDATORY,
            "batches": len(BATCHES),
            "max_products_per_step": MAX_RENDER_PRODUCTS_PER_STEP,
        },
        sort_keys=True,
    )
