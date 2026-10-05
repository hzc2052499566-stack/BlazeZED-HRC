"""Rule lock v3: the availability gate speaks the experiment's own currency.

v1 and v2 are preserved unchanged.  The 241-frame probe under v2 (AGENTS.md
6.6.24) found the event gate satisfied for all four characters on both legs --
the layout revision worked -- while **no** five-view subset met v2's availability
gate.  The gap was two cells out of 3,856.

The one change here comes from noticing that v2's availability gate and the
experiment's own success criterion were written in different currencies:

* 6.6.20 asks for *every target cell* to clear a per-cell floor;
* 6.6.6, pre-registered long before any of this, asks for measured **coverage**
  of ``>=0.98`` in G0 and ``>=0.90`` in G1, where a cell counts as covered when
  M3 can use it -- that is, at three measured views, the deployment floor.

The per-cell reading is strictly stronger, and it was written without checking it
against the coverage gate it is supposed to serve.  v3 states the availability
gate in 6.6.6's currency: **coverage at the deployment floor, against 6.6.6's own
thresholds.**  The best subset under it covers G0 at ``1.0000`` and G1 at
``0.9995``, against gates of ``0.98`` and ``0.90``.

Two things this deliberately does *not* do.  It does not lower the deployment
floor: a cell still only counts as covered at three measured views, because that
is what M3 needs in order not to abstain.  And it does not touch the ordering:
key 1 is still "maximise the worst cell", so among subsets that clear the
coverage gate the search still prefers the one whose worst cell is best -- the
per-cell minimum stops being a veto and stays a preference.

**Provenance.**  This is the third post-hoc change in this chain and is recorded
as one.  What distinguishes it from the other two is that the replacement
criterion was not invented after the fact: G0 ``>=0.98`` and G1 ``>=0.90`` are
lifted verbatim from 6.6.6, which predates the whole camera-selection effort.
The decision to *use* it here was still taken after seeing v2 fail.

Layout, batches, event shape, event minimum, raw replay, ``T_joint``, the R
gates, the nine-key ordering and the M0 rule are all inherited from v2 unchanged.
"""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

import common_bank_rule_lock_v1 as v1
import common_bank_rule_lock_v2 as v2
from common_bank_rule_lock_v2 import (  # noqa: F401  (re-exported unchanged)
    ANCHOR_KEPT_FROM_BATCH,
    BANK_COUNT,
    BANK_RADIUS_M,
    BATCHES,
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
    min_pairwise_angle,
    recoverable_event,
)

RULE_TAG = "fs_cts5_common_bank_18_9_5_rule_lock_v3"
RULE_SOURCE = "AGENTS.md 6.6.20 as amended by 6.6.23 and 6.6.25"
SUPERSEDES = v2.RULE_TAG
SUPERSEDES_CHAIN = (v1.RULE_TAG, v2.RULE_TAG)

# A cell counts as covered when M3 can use it: three measured views (6.6.6).
# Unchanged from the deployment floor -- v3 does not lower it.
COVERAGE_FLOOR = DEPLOYMENT_FLOOR
# Lifted verbatim from 6.6.6, which predates the camera-selection work entirely.
COVERAGE_GATES = {"G0": 0.98, "G1": 0.90}
COVERAGE_GATE_SOURCE = "AGENTS.md 6.6.6 pre-registered coverage gates"

POST_HOC = {
    "written_after": "common_bank_v2 event feasibility probe, attempt_01 (6.6.24)",
    "authorised_by": "user, 2026-08-19, after being shown v2 has no solution",
    "inherits_post_hoc_from_v2": sorted(v2.POST_HOC["post_hoc_in_v2"]),
    "post_hoc_in_v3": {
        "availability_gate": {
            "was": "every target cell at or above a per-scenario floor of 4 (G0) / 3 (G1)",
            "now": (
                "measured coverage at the deployment floor of {} views, against "
                "6.6.6's pre-registered thresholds".format(COVERAGE_FLOOR)
            ),
            "known_when_chosen": (
                "Under v2 no five-view subset met the per-cell gate; the best "
                "event-satisfying subset missed by two cells out of 3,856, at "
                "G1 coverage 0.9995 and G0 coverage 1.0000."
            ),
            "mitigation": (
                "The replacement thresholds are not new: G0 >=0.98 and G1 >=0.90 "
                "are lifted verbatim from 6.6.6, pre-registered before the "
                "camera-selection work began.  The per-cell minimum is not "
                "abandoned either -- ordering key 1 still maximises it, so it "
                "changes from a veto into a preference.  The deployment floor of "
                "three measured views is unchanged."
            ),
        }
    },
    "not_changed": [
        "the deployment floor of three measured endpoints",
        "the layout, batches, event shape and event minimum inherited from v2",
        "the nine-key ordering, the R gates and the M0 rule",
        "scene geometry",
    ],
}


def coverage(counts: Sequence[int], floor: int = COVERAGE_FLOOR) -> float:
    """Fraction of cells a subset can actually measure at the deployment floor."""
    counts = list(counts)
    if not counts:
        raise RuleLockError("Coverage needs at least one cell.")
    return sum(1 for count in counts if int(count) >= int(floor)) / len(counts)


def coverage_gate_met(per_scenario: Mapping, gates: Mapping = COVERAGE_GATES) -> bool:
    """Every selection scenario must clear its own pre-registered threshold."""
    for scenario, gate in gates.items():
        if scenario not in per_scenario:
            raise RuleLockError("Coverage for {} is missing.".format(scenario))
        if float(per_scenario[scenario]) < float(gate):
            return False
    return True


def layout_sha256(layout: Sequence[Mapping] = SURVEY_LAYOUT) -> str:
    """v2's layout, hashed with v2's layout as the default rather than v1's.

    The v2 probe's first run died because ``layout_sha256`` was re-exported from
    v1 and still defaulted to v1's eighteen cameras.  Binding the default here
    means a bare call inside a v3 script hashes what a v3 script means.
    """
    return v1.layout_sha256(layout)


def pairwise_angle_table(layout: Sequence[Mapping] = SURVEY_LAYOUT) -> dict:
    """Same rebinding as ``layout_sha256``, for the same reason."""
    return v1.pairwise_angle_table(layout)


def rule_lock_document(
    selector_hashes: Mapping,
    scene_hashes: Mapping,
    layout: Sequence[Mapping] = SURVEY_LAYOUT,
) -> dict:
    """The v3 lock: v2's document with the availability gate restated."""
    if len(layout) != SURVEY_COUNT:
        raise RuleLockError(
            "The survey holds exactly {} candidates.".format(SURVEY_COUNT)
        )
    if sorted(scene_hashes) != sorted(CHARACTERS):
        raise RuleLockError("Scene hashes must cover exactly {}.".format(CHARACTERS))
    return {
        "record": RULE_TAG,
        "rule_source": RULE_SOURCE,
        "supersedes": SUPERSEDES,
        "supersedes_chain": list(SUPERSEDES_CHAIN),
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
            "coverage_floor": COVERAGE_FLOOR,
            "coverage_gates": dict(COVERAGE_GATES),
            "coverage_gate_source": COVERAGE_GATE_SOURCE,
            "per_cell_minimum_is_a_preference_not_a_veto": True,
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
        "failure_policy": (
            "Any incomplete input, absent qualifying pair, failed raw preflight or "
            "non-reproducible recomputation is saved as a FAIL with "
            "formal_capture_authorized=false.  v1 and v2 are preserved as the "
            "record of what each earlier version pre-registered."
        ),
    }


def summary() -> str:
    return json.dumps(
        {
            "supersedes": SUPERSEDES,
            "availability_gate": "coverage",
            "coverage_floor": COVERAGE_FLOOR,
            "coverage_gates": dict(COVERAGE_GATES),
            "event_min_occluded_frames": EVENT_MIN_OCCLUDED_FRAMES,
        },
        sort_keys=True,
    )
