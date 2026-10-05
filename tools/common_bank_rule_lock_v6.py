"""Rule lock v6: v4's gates with v5's smaller batches.  v5 was a failed attempt.

v1 through v5 are preserved unchanged, v5 included, because a version that was
frozen and then measured unusable is part of the record of how this contract was
arrived at.

**What v5 tried and why it failed.**  On the first clean geometric matrix, v2's
per-cell availability floor admitted 14 five-view subsets, where under the
contaminated matrix it had admitted none.  That looked like grounds to withdraw
v3's coverage gate and take the post-hoc count from four back to three, and v5
was frozen to do exactly that.  The check behind that decision was incomplete: it
established that *subsets* qualify and never asked whether any ``(F,R)`` **pair**
does.  Measured afterwards, v5 yields **zero** pairs -- all 10,010 built from its
14 subsets fail the reserve gate, 9,150 on swap coverage and 860 on a reserve
that can replace nothing.  Fourteen qualifying subsets are simply too few for a
one-view swap to land inside the qualifying set.

So the coverage gate is load-bearing after all, and for a better reason than the
one v3 gave for it: not because subsets are otherwise unavailable, but because the
reserve requirement needs a qualifying set wide enough to swap within.  v6
restores it and records that.  The post-hoc count stays at four.

**What v6 keeps from v5.**  The six-batch split at four render products per step,
which is independent of the gates and is the untested physical hypothesis for the
anchor identity failure: F01 has missed on every round while the other three
characters matched on every comparison, the clock gate rules out within-batch
nondeterminism, and the machine runs at 347 MiB free VRAM.

Everything else -- layout, event contract, ordering, M0, the mandatory-view
exemption -- is inherited unchanged.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import common_bank_rule_lock_v1 as v1
import common_bank_rule_lock_v2 as v2
import common_bank_rule_lock_v3 as v3
import common_bank_rule_lock_v4 as v4
import common_bank_rule_lock_v5 as v5
from common_bank_rule_lock_v4 import (  # noqa: F401  (re-exported unchanged)
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
    coverage,
    coverage_gate_met,
    layout_sha256,
    min_pairwise_angle,
    pairwise_angle_table,
    provably_mandatory,
    recoverable_event,
    reserve_gate_binds,
)
from common_bank_rule_lock_v5 import (  # noqa: F401  (batch split kept)
    BATCHES,
    MAX_RENDER_PRODUCTS_PER_STEP,
    check_batches,
)

RULE_TAG = "fs_cts5_common_bank_18_9_5_rule_lock_v6"
RULE_SOURCE = "AGENTS.md 6.6.20 as amended by 6.6.23, 6.6.26 and 6.6.27"
SUPERSEDES = v5.RULE_TAG
SUPERSEDES_CHAIN = (v1.RULE_TAG, v2.RULE_TAG, v3.RULE_TAG, v4.RULE_TAG, v5.RULE_TAG)

FAILED_ATTEMPT = {
    "version": v5.RULE_TAG,
    "tried": "withdraw v3's coverage gate and restore v2's per-cell floor",
    "measured": (
        "14 qualifying five-view subsets but zero qualifying (F,R) pairs: of the "
        "10,010 pairs built from them, 9,150 failed swap coverage and 860 had a "
        "reserve that could replace nothing."
    ),
    "why_the_check_was_incomplete": (
        "Feasibility was established at the subset level and never tested at the "
        "pair level.  Fourteen qualifying subsets are too few for a one-view swap "
        "to land inside the qualifying set, so the reserve gate cannot be met."
    ),
    "consequence": (
        "The coverage gate is load-bearing, for the reserve requirement rather "
        "than for subset availability.  The post-hoc count stays at four."
    ),
}


def rule_lock_document(
    selector_hashes: Mapping,
    scene_hashes: Mapping,
    layout: Sequence[Mapping] = SURVEY_LAYOUT,
) -> dict:
    """v4's document with v5's batch split and the failed attempt recorded."""
    document = v4.rule_lock_document(selector_hashes, scene_hashes, layout)
    document["record"] = RULE_TAG
    document["rule_source"] = RULE_SOURCE
    document["supersedes"] = SUPERSEDES
    document["supersedes_chain"] = list(SUPERSEDES_CHAIN)
    document["survey_contract"] = dict(
        check_batches(BATCHES, layout), traversal_order=list(TRAVERSAL_ORDER)
    )
    document["failed_attempt"] = dict(FAILED_ATTEMPT)
    document["provenance"] = dict(
        v4.POST_HOC,
        batch_split_rationale=v5.POST_HOC["batch_split_rationale"],
        withdrawal_attempted_and_measured_unusable=dict(FAILED_ATTEMPT),
    )
    return document


def summary() -> str:
    return json.dumps(
        {
            "supersedes": SUPERSEDES,
            "availability_gate": "coverage",
            "coverage_gates": dict(COVERAGE_GATES),
            "event_min_occluded_frames": EVENT_MIN_OCCLUDED_FRAMES,
            "reserve_exempts_provably_mandatory": RESERVE_EXEMPTS_PROVABLY_MANDATORY,
            "batches": len(BATCHES),
            "max_products_per_step": MAX_RENDER_PRODUCTS_PER_STEP,
        },
        sort_keys=True,
    )
