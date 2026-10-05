"""Scientific rule generation v8 for randomized-block common-bank selection.

v8 preserves v7's 18-view layout, hard gates, geometry, pair space, and
ordering semantics.  It changes the acquisition/selection unit because the
registered scenario-major v2 preflight showed that a view fixed to one render
batch is confounded with batch-dependent RGB realisation.  Three fresh cohort
repeats are therefore evaluated jointly; no cross-render binary-disagreement
tolerance is introduced.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import common_bank_rule_lock_v7 as v7
from common_bank_rule_lock_v7 import (  # noqa: F401 - v8 re-exports v7 constants
    ADDED_IN_V7,
    ANCHOR_KEPT_FROM_BATCH,
    BANK_COUNT,
    BANK_RADIUS_M,
    CHARACTERS,
    COVERAGE_FLOOR,
    COVERAGE_GATE_SOURCE,
    COVERAGE_GATES,
    DEPLOYMENT_FLOOR,
    DIAGNOSTIC_SCENARIOS,
    DROPPED_FROM_V6,
    EVENT_LEG_SIDES,
    EVENT_MIN_CLEAR_FRAMES,
    EVENT_MIN_DISTINCT_VIEWS_PER_CHARACTER,
    EVENT_MIN_OCCLUDED_FRAMES,
    FOCAL_LENGTH,
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
    SURVEY_ANCHOR,
    SURVEY_COUNT,
    SURVEY_LAYOUT,
    SURVEY_VIEWS,
    TABLE_ATTRIBUTION,
    TARGET_BONES,
    RuleLockError,
    layout_sha256,
    min_pairwise_angle,
    pairwise_angle_table,
    provably_mandatory,
    recoverable_event,
    reserve_gate_binds,
)

RULE_TAG = "fs_cts5_common_bank_18_9_5_rule_lock_v8"
RULE_SOURCE = "AGENTS.md section 42, user selected 3-repeat on 2026-08-22"
SUPERSEDES = v7.RULE_TAG
SUPERSEDES_CHAIN = tuple(v7.SUPERSEDES_CHAIN) + (v7.RULE_TAG,)

RENDERER_REPEATS = 3
ROLE_SESSION_COUNT = len(CHARACTERS) * RENDERER_REPEATS
CAPTURE_SCENARIOS = ("G0", "G1", "W")
BATCH_COUNT = 6
PRODUCTS_PER_BATCH = 4
PRIMARY_OCCURRENCES_PER_BATCH = 3
PRIMARY_OCCURRENCES_PER_REPEAT = SURVEY_COUNT
BRIDGE_A = SURVEY_ANCHOR
SAME_INPUT_REPLAY_PROCESSES = 2
PAIR_SPACE = 6_126_120

CLASSIFICATION = "excluded_camera_bank_selection_asset_not_formal_evaluation"
STATEMENT_BOUNDARY = (
    "cohort-designed bank stable across three fresh randomized render repeats "
    "for four fixed simulated characters, registered scenarios, machine and "
    "runtime; not unseen-character or general renderer generalisation"
)


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_v7_constants_unchanged() -> None:
    if SURVEY_COUNT != 18 or BANK_COUNT != 9:
        raise RuleLockError("v8 must preserve the frozen 18-to-9 layout.")
    if SELECTED_COUNT != 5 or RESERVE_COUNT != 4:
        raise RuleLockError("v8 must preserve F=5 and R=4.")
    if dict(COVERAGE_GATES) != {"G0": 0.98, "G1": 0.9}:
        raise RuleLockError("v7 coverage gates drifted before v8 freeze.")
    if COVERAGE_FLOOR != 3 or DEPLOYMENT_FLOOR != 3:
        raise RuleLockError("v7 deployment floor drifted before v8 freeze.")
    if EVENT_MIN_CLEAR_FRAMES != 5 or EVENT_MIN_OCCLUDED_FRAMES != 20:
        raise RuleLockError("v7 recoverable-event thresholds drifted.")


def rule_lock_document(
    selector_hashes: Mapping,
    scene_hashes: Mapping,
    schedule: Mapping,
) -> dict:
    """Build the v8 lock around an already materialised, validated schedule."""

    from common_bank_randomized_block_schedule_v1 import validate_schedule

    _require_v7_constants_unchanged()
    schedule_report = validate_schedule(schedule)
    if not schedule_report.get("pass"):
        raise RuleLockError("The randomized-block schedule did not validate.")
    schedule_sha256 = str(schedule.get("schedule_sha256") or "")
    if (
        len(schedule_sha256) != 64
        or schedule_report.get("calculated_schedule_sha256") != schedule_sha256
    ):
        raise RuleLockError("The randomized-block schedule hash is not self-consistent.")

    document = v7.rule_lock_document(selector_hashes, scene_hashes, SURVEY_LAYOUT)
    document.update(
        {
            "record": RULE_TAG,
            "rule_source": RULE_SOURCE,
            "supersedes": SUPERSEDES,
            "supersedes_chain": list(SUPERSEDES_CHAIN),
            "classification": CLASSIFICATION,
            "statement_boundary": STATEMENT_BOUNDARY,
            "renderer_repeat_scale_decision": {
                "selected": "3-repeat",
                "renderer_repeats_per_character": RENDERER_REPEATS,
                "fresh_role_sessions": ROLE_SESSION_COUNT,
                "selected_utc": "2026-08-22",
                "source": "explicit_user_choice",
                "two_repeat_exploratory_option_selected": False,
            },
            "scientific_rule_change": True,
            "scientific_threshold_change": False,
            "formal_capture_authorized": False,
            "confirmatory_capture_authorized": False,
            "execution_shakedown_capture_authorized": True,
            "scientific_final_five_selected": False,
            "selected_view_ids": None,
            "gt_error_used": False,
            "randomized_block_schedule": schedule,
            # The schedule module hashes the canonical payload with the
            # self-referential ``schedule_sha256`` field omitted.  Reuse that
            # one identity here rather than publishing a second digest for the
            # same design artifact.
            "randomized_block_schedule_sha256": schedule_sha256,
            "randomized_block_schedule_validation": schedule_report,
        }
    )
    document["survey_contract"] = {
        "renderer_repeats": RENDERER_REPEATS,
        "role_sessions": ROLE_SESSION_COUNT,
        "characters": list(CHARACTERS),
        "scenarios": list(CAPTURE_SCENARIOS),
        "post_selection_diagnostic_scenarios": list(DIAGNOSTIC_SCENARIOS),
        "frames": FRAME_COUNT,
        "batches_per_session": BATCH_COUNT,
        "render_products_per_batch": PRODUCTS_PER_BATCH,
        "primary_occurrences_per_batch": PRIMARY_OCCURRENCES_PER_BATCH,
        "primary_occurrences_per_repeat": PRIMARY_OCCURRENCES_PER_REPEAT,
        "creation_slots": list(schedule.get("creation_slots") or []),
        "bridge_A": BRIDGE_A,
        "bridge_B": schedule.get("bridge_B"),
        "bridge_B_geometry_role": schedule.get("bridge_B_geometry_role"),
        "bridge_B_geometry_eligible": list(
            schedule.get("bridge_B_geometry_eligible") or []
        ),
        "bridge_occurrences_per_repeat": BATCH_COUNT,
        "bridge_creation_slot_globally_balanced": True,
        "bridge_creation_slot_per_batch_counts_across_sessions": {
            str(slot): 3 for slot in schedule.get("creation_slots") or []
        },
        "each_primary_view_creation_slot_counts_across_sessions": {
            str(slot): 3 for slot in schedule.get("creation_slots") or []
        },
        "within_character_primary_creation_slots_distinct": True,
        "each_primary_view_batch_counts_across_sessions": 2,
        "scenario_major": True,
        "nonpersistent_products": True,
        "full_drain_per_step": True,
        "step_delta_time": 0.0,
        "primary_occurrence_is_preassigned": True,
        "duplicate_bridge_can_enter_selection": False,
    }
    document["same_input_QC"] = {
        "ordinary_python_processes": SAME_INPUT_REPLAY_PROCESSES,
        "fresh_pose_instance_per_process": True,
        "binary_fields": ["detection", "endpoint_codes", "E", "T", "T_joint"],
        "binary_exact_required": True,
        "continuous_coordinates": "report_only",
        "counts_as_renderer_repeat": False,
    }
    document["cross_renderer_policy"] = {
        "RGB_exact_required": False,
        "depth_exact_required": False,
        "binary_exact_required": False,
        "CameraParams_exact_required": True,
        "all_differences_reported": True,
        "numeric_disagreement_threshold": None,
        "outcome_based_block_deletion_forbidden": True,
    }
    document["robust_selection"] = {
        "replicate_matrices": RENDERER_REPEATS,
        "same_F_R_must_pass_v7_hard_gates_in_repeats": RENDERER_REPEATS,
        "data_ordering_key_reducer": "minimum_worst_repeat",
        "cells_at_minimum_reducer": "maximum_worst_repeat",
        "geometry_keys": "unchanged_from_v7",
        "lexicographic_tie_break": "unchanged_from_v7",
        "pair_space": PAIR_SPACE,
        "exhaustive_required": True,
        "leave_one_renderer_repeat_out_fits": RENDERER_REPEATS,
        "LORO_B_identity": "exact",
        "LORO_M0_identity": "exact",
        "LORO_F_max_changes": 1,
        "LORO_F_replacement_source": "full_data_R_and_registered_valid_swap",
        "leave_one_character_out": "pre_registered_report_only_sensitivity",
    }
    document["data_separation"] = {
        "v1_to_v7_and_failed_probe_data_are_selector_input": False,
        "old_data_may_set_thresholds_or_schedule": False,
        "execution_shakedown_is_selector_input": False,
        "v8_selection_data_supports_unbiased_performance_claim": False,
        "fresh_formal_bundles_after_bank_freeze": 36,
        "excluded_calibration_bundles_after_bank_freeze": 12,
        "formal_reselection_forbidden": True,
    }
    document["failure_policy"] = (
        "preserve write-once; no outcome-based rerun, replacement, added n, "
        "threshold change, schedule/seed change, camera selection or formal "
        "authorization after any validity/scientific gate failure"
    )
    document["next_gate"] = (
        "run_one_excluded_F01_execution_shakedown_without_opening_provisional_"
        "E_T_or_selection_outputs"
    )
    return document


def summary() -> str:
    return json.dumps(
        {
            "record": RULE_TAG,
            "renderer_repeats": RENDERER_REPEATS,
            "role_sessions": ROLE_SESSION_COUNT,
            "threshold_change": False,
            "pair_space": PAIR_SPACE,
            "next_gate": "excluded_F01_execution_shakedown_only",
        },
        sort_keys=True,
    )
