"""Rule lock v4: the reserve gate stops demanding the impossible.

v1, v2 and v3 are preserved unchanged.  Under v3 the availability gate finally
admitted 161 five-view subsets (AGENTS.md 6.6.25), and then **every** one of the
115,115 ``(F,R)`` pairs built from them failed the reserve gate -- all of them,
for one reason.

``az030_el00`` is the only view in the whole eighteen that produces a properly
shaped left-leg event for F01 and F02.  6.6.20's reserve gate asks that every
selected view have some reserve that can replace it; replacing that one leaves
the scenario with no left-leg occlusion at all, so it cannot be replaced by
anything, and the gate can never be met while it is selected -- which is always,
since every qualifying subset contains it.

**The change.**  A view that is *provably mandatory* is exempt from the swap
requirement.  Provably mandatory has an exact meaning here, and it is not a
judgement call: a view is mandatory when it appears in **every** subset that
passes the F hard gates.  That definition and unswappability are the same
statement -- if some reserve could replace it, the repaired subset would itself
be a qualifying subset that omits it, and it would not be in every one.  So the
exemption removes a requirement that is provably unsatisfiable, and nothing else.

Every other selected view must still have a valid swap, every reserve must still
replace at least one selected view, and the reserve gate must still **bind on at
least one selected view** -- a bank whose every member were exempt would be five
cameras plus four decorations, which is the thing 6.6.20's reserve gate exists to
prevent.  The exempt views are named in the freeze record, so a nine-view bank of
"eight substitutable plus one structurally fixed" reads as exactly that.

**Provenance.**  Fourth post-hoc change in this chain, recorded as one.  What
narrows it: the exemption is defined by a property that is computed from the
candidate set rather than chosen, it cannot be widened by picking a threshold,
and the measured count under v3's geometric matrix is one view of eighteen.

Layout, batches, coverage gates, event contract, ordering and M0 are inherited
from v3 unchanged.
"""

from __future__ import annotations

import json
import math
from typing import Iterable, Mapping, Sequence

import common_bank_rule_lock_v1 as v1
import common_bank_rule_lock_v2 as v2
import common_bank_rule_lock_v3 as v3
from common_bank_rule_lock_v3 import (  # noqa: F401  (re-exported unchanged)
    ANCHOR_KEPT_FROM_BATCH,
    BANK_COUNT,
    BANK_RADIUS_M,
    BATCHES,
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
    NANODEGREES_PER_DEGREE,
    ORDERING_KEYS,
    RAW_REPLAY,
    RESERVE_COUNT,
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
    check_batches,
    coverage,
    coverage_gate_met,
    layout_sha256,
    min_pairwise_angle,
    pairwise_angle_table,
    recoverable_event,
)

RULE_TAG = "fs_cts5_common_bank_18_9_5_rule_lock_v4"
RULE_SOURCE = "AGENTS.md 6.6.20 as amended by 6.6.23, 6.6.25 and 6.6.26"
SUPERSEDES = v3.RULE_TAG
SUPERSEDES_CHAIN = (v1.RULE_TAG, v2.RULE_TAG, v3.RULE_TAG)

RESERVE_EXEMPTS_PROVABLY_MANDATORY = True
# The reserve gate has to bite somewhere.  If every selected view were exempt the
# four reserves would be answering no question at all.
MIN_SWAPPABLE_SELECTED_VIEWS = 1

POST_HOC = {
    "written_after": "v3 geometric trial selection over the 6.6.24 matrix (6.6.25)",
    "authorised_by": "user, 2026-08-19, after being shown v3 yields zero (F,R) pairs",
    "inherits_post_hoc_from": {
        "v2": sorted(v2.POST_HOC["post_hoc_in_v2"]),
        "v3": sorted(v3.POST_HOC["post_hoc_in_v3"]),
    },
    "post_hoc_in_v4": {
        "reserve_gate": {
            "was": "every selected view must have a valid swap in the reserve",
            "now": (
                "every selected view that is not provably mandatory must have a "
                "valid swap; a provably mandatory view is exempt and named in the "
                "record"
            ),
            "known_when_chosen": (
                "Under v3, 161 subsets passed the F gates and all 115,115 pairs "
                "built from them failed the reserve gate, because az030_el00 is "
                "the only view producing a shaped left-leg event for F01 and F02."
            ),
            "mitigation": (
                "Provably mandatory is computed, not chosen: a view in every "
                "qualifying subset cannot be swapped out of any of them, since a "
                "repaired subset would be a qualifying subset omitting it.  The "
                "exemption therefore drops a requirement that is provably "
                "unsatisfiable and nothing more.  Every other selected view must "
                "still be swappable, every reserve must still replace at least "
                "one, and at least one selected view must remain swappable."
            ),
        }
    },
    "not_changed": [
        "the deployment floor, the coverage gates and their 6.6.6 source",
        "the layout, batches, event shape and event minimum",
        "the nine-key ordering and the M0 rule",
        "scene geometry",
    ],
}


def provably_mandatory(qualifying_subsets: Iterable[Sequence[str]]) -> list:
    """Views that appear in every subset passing the F hard gates.

    Equivalently, and this is why the exemption is safe: views that no reserve
    can replace.  If some ``r`` made ``F - v + r`` pass, that subset would be a
    qualifying subset without ``v``, and ``v`` would not be in every one.
    """
    subsets = [set(subset) for subset in qualifying_subsets]
    if not subsets:
        return []
    return sorted(set.intersection(*subsets))


def reserve_gate_binds(selected: Sequence[str], mandatory: Sequence[str]) -> bool:
    """At least one selected view must still be held to the swap requirement."""
    exempt = set(mandatory)
    return sum(1 for view in selected if view not in exempt) >= (
        MIN_SWAPPABLE_SELECTED_VIEWS
    )


def rule_lock_document(
    selector_hashes: Mapping,
    scene_hashes: Mapping,
    layout: Sequence[Mapping] = SURVEY_LAYOUT,
) -> dict:
    """The v4 lock: v3's document with the reserve gate amended."""
    document = v3.rule_lock_document(selector_hashes, scene_hashes, layout)
    document["record"] = RULE_TAG
    document["rule_source"] = RULE_SOURCE
    document["supersedes"] = SUPERSEDES
    document["supersedes_chain"] = list(SUPERSEDES_CHAIN)
    document["hard_gates"] = dict(
        document["hard_gates"],
        reserve_exempts_provably_mandatory=RESERVE_EXEMPTS_PROVABLY_MANDATORY,
        min_swappable_selected_views=MIN_SWAPPABLE_SELECTED_VIEWS,
        provably_mandatory_definition=(
            "a view present in every five-view subset that passes the F hard "
            "gates; equivalently, a view no reserve can replace"
        ),
        reserve_must_cover_every_selected_view=False,
        reserve_must_cover_every_non_mandatory_selected_view=True,
        reserve_member_must_have_a_valid_swap=True,
    )
    document["provenance"] = dict(POST_HOC)
    document["failure_policy"] = (
        "Any incomplete input, absent qualifying pair, failed raw preflight or "
        "non-reproducible recomputation is saved as a FAIL with "
        "formal_capture_authorized=false.  v1, v2 and v3 are preserved as the "
        "record of what each earlier version pre-registered."
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
        },
        sort_keys=True,
    )
