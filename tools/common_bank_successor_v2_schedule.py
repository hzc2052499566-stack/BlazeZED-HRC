"""Pure deterministic randomized-block schedule for successor-v2 18-view capture.

The schedule is parameterised only by the already frozen successor 18-view
identity and a new result-blind seed.  It performs no file I/O.  The algebraic
construction gives exact balance for twelve fresh role sessions:

* three repeats by four characters;
* six batches by four render products (one QA bridge plus three primaries);
* every primary occurs exactly once in every session;
* every primary occupies every batch and execution position twice and every
  creation slot three times across the twelve sessions; and
* the QA bridge occupies every slot three times within every batch.

This is an independent successor schedule.  Its seed and payload identity are
required to differ from the superseded schedule, which is never imported.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "fs_cts5_common_bank_successor_v2_randomized_block_schedule_v1"
SEED_TEXT = (
    "fs-cts5-common-bank-successor-v2|randomized-block-schedule-v1|"
    "fresh-after-transition-probe-pass|result-blind|2026-08-25"
)
SEED_SHA256 = hashlib.sha256(SEED_TEXT.encode("utf-8")).hexdigest()

# Public historical identities are recorded only as negative controls.  This
# module neither imports nor opens their implementation or output tree.
LEGACY_SEED_SHA256 = "7e659d4a8c9bcdb052d25354fc95cd731e1cda4a00f1fe9207519ba9d089f8de"
LEGACY_SCHEDULE_SHA256 = (
    "3947750b234113fd1ff16e90aaa7bae548945f0ca3082c33d07e0b2d25984cca"
)

CHARACTERS = ("F01", "F02", "M01", "M02")
REPEATS = (1, 2, 3)
SCENARIOS = ("G0", "G1", "W")
BATCH_LABELS = tuple(f"batch_{index:02d}" for index in range(1, 7))
CREATION_SLOTS = (0, 1, 2, 3)
QA_BRIDGE = "az000_el00"
CANDIDATE_COUNT = 18
PRODUCTS_PER_BATCH = 4
PRIMARY_PRODUCTS_PER_BATCH = 3

# For the two role pairs sharing the same six primary rotations, these two
# permutations make primary-batch, execution-position and their difference
# each cover every residue exactly twice across the twelve sessions.
_EXECUTION_POSITION_ROTATIONS = (
    (0, 2, 1, 3, 5, 4),
    (2, 4, 0, 5, 1, 3),
)


class SuccessorScheduleError(RuntimeError):
    """Raised when a proposed successor candidate set cannot be scheduled."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorScheduleError(message)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_candidates(candidate_views: Sequence[str]) -> tuple[str, ...]:
    _require(
        not isinstance(candidate_views, (str, bytes))
        and isinstance(candidate_views, Sequence),
        "Candidate views must be a sequence.",
    )
    views = tuple(str(view) for view in candidate_views)
    _require(len(views) == CANDIDATE_COUNT, "The successor schedule needs 18 views.")
    _require(
        all(view and view.strip() == view for view in views),
        "Candidate view IDs must be non-empty canonical text.",
    )
    _require(len(set(views)) == len(views), "Candidate view IDs must be unique.")
    _require(QA_BRIDGE not in views, "The QA bridge cannot be a science primary.")
    return tuple(sorted(views))


def _digest(namespace: str, value: str) -> str:
    material = f"{SEED_TEXT}\n{namespace}\n{value}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _seeded_order(values: Sequence[str], namespace: str) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: (_digest(namespace, value), value)))


def _role_order() -> tuple[str, ...]:
    return _seeded_order(CHARACTERS, "role-order")


def _scenario_orders() -> tuple[tuple[str, ...], ...]:
    orders = tuple(itertools.permutations(SCENARIOS))
    return tuple(
        sorted(
            orders,
            key=lambda order: (
                _digest("scenario-permutation-order", ">".join(order)),
                order,
            ),
        )
    )


def _schedule_payload_sha256(schedule: Mapping[str, Any]) -> str:
    payload = dict(schedule)
    payload.pop("schedule_sha256", None)
    return canonical_sha256(payload)


def _session_specs(role_order: Sequence[str]) -> list[dict[str, int | str]]:
    specs: list[dict[str, int | str]] = []
    for repeat_offset, repeat in enumerate(REPEATS):
        order = tuple(
            role_order[(position + repeat_offset) % len(role_order)]
            for position in range(len(role_order))
        )
        for role_position, character in enumerate(order, start=1):
            role_rank = role_order.index(character)
            primary_rotation = (3 * role_rank + repeat_offset) % 6
            duplicate_group = role_rank // 2
            execution_position_rotation = _EXECUTION_POSITION_ROTATIONS[
                duplicate_group
            ][primary_rotation]
            batch_execution_rotation = (
                primary_rotation - execution_position_rotation
            ) % 6
            specs.append(
                {
                    "character": character,
                    "repeat": repeat,
                    "role_rank": role_rank,
                    "role_session_position": role_position,
                    "primary_batch_rotation": primary_rotation,
                    "primary_execution_position_rotation": (
                        execution_position_rotation
                    ),
                    "batch_execution_rotation": batch_execution_rotation,
                }
            )
    return specs


def _construct_schedule(candidate_views: Sequence[str]) -> dict[str, Any]:
    candidates = _normalise_candidates(candidate_views)
    candidate_order = _seeded_order(candidates, "candidate-primary-order")
    role_order = _role_order()
    scenario_orders = _scenario_orders()
    sessions: list[dict[str, Any]] = []

    for session_offset, spec in enumerate(_session_specs(role_order)):
        global_index = session_offset + 1
        character = str(spec["character"])
        repeat = int(spec["repeat"])
        primary_rotation = int(spec["primary_batch_rotation"])
        execution_rotation = int(spec["batch_execution_rotation"])
        session_id = f"{character}_rep_{repeat:02d}"
        slot_rotation = session_offset % len(CREATION_SLOTS)

        primary_by_batch: dict[int, list[dict[str, Any]]] = {
            index: [] for index in range(len(BATCH_LABELS))
        }
        for candidate_index, view_id in enumerate(candidate_order):
            base_batch = candidate_index % len(BATCH_LABELS)
            candidate_row = candidate_index // len(BATCH_LABELS)
            logical_batch = (
                base_batch + primary_rotation
            ) % len(BATCH_LABELS)
            creation_slot = (
                slot_rotation + candidate_row
            ) % len(CREATION_SLOTS)
            primary_by_batch[logical_batch].append(
                {
                    "occurrence_id": f"{session_id}:primary:{view_id}",
                    "occurrence_kind": "primary",
                    "science_input": True,
                    "view_id": view_id,
                    "creation_slot": creation_slot,
                }
            )

        batches: list[dict[str, Any]] = []
        for logical_batch, batch_label in enumerate(BATCH_LABELS):
            products = list(primary_by_batch[logical_batch])
            _require(
                len(products) == PRIMARY_PRODUCTS_PER_BATCH,
                "Algebraic partition lost a primary product.",
            )
            bridge_slot = (
                slot_rotation + PRIMARY_PRODUCTS_PER_BATCH
            ) % len(CREATION_SLOTS)
            products.append(
                {
                    "occurrence_id": f"{session_id}:qa_bridge:{batch_label}",
                    "occurrence_kind": "qa_bridge",
                    "science_input": False,
                    "view_id": QA_BRIDGE,
                    "creation_slot": bridge_slot,
                }
            )
            products.sort(key=lambda product: int(product["creation_slot"]))
            _require(
                {product["creation_slot"] for product in products}
                == set(CREATION_SLOTS),
                "One batch does not occupy all four creation slots.",
            )
            execution_position = (
                (logical_batch - execution_rotation) % len(BATCH_LABELS)
            ) + 1
            batches.append(
                {
                    "batch_id": batch_label,
                    "logical_batch_index": logical_batch + 1,
                    "execution_position": execution_position,
                    "products": products,
                }
            )
        batches.sort(key=lambda batch: int(batch["execution_position"]))
        sessions.append(
            {
                "session_id": session_id,
                "global_execution_index": global_index,
                "character": character,
                "repeat": repeat,
                "repeat_index": repeat,
                "role_rank": int(spec["role_rank"]) + 1,
                "role_session_position": int(spec["role_session_position"]),
                "primary_batch_rotation": primary_rotation,
                "primary_execution_position_rotation": int(
                    spec["primary_execution_position_rotation"]
                ),
                "batch_execution_rotation": execution_rotation,
                "scenario_order": list(scenario_orders[primary_rotation]),
                "batch_execution_order": [batch["batch_id"] for batch in batches],
                "batches": batches,
            }
        )

    schedule: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "deterministic_result_blind_successor_schedule",
        "seed_text": SEED_TEXT,
        "seed_sha256": SEED_SHA256,
        "legacy_seed_identity_reused": False,
        "legacy_schedule_identity_reused": False,
        "construction_reads_experimental_results": False,
        "construction_uses_file_io": False,
        "characters": list(CHARACTERS),
        "repeats_per_character": len(REPEATS),
        "role_sessions": len(CHARACTERS) * len(REPEATS),
        "batches_per_session": len(BATCH_LABELS),
        "products_per_batch": PRODUCTS_PER_BATCH,
        "primary_products_per_batch": PRIMARY_PRODUCTS_PER_BATCH,
        "candidate_views": list(candidates),
        "candidate_set_sha256": canonical_sha256(list(candidates)),
        "candidate_primary_order": list(candidate_order),
        "qa_bridge": QA_BRIDGE,
        "qa_bridge_is_science_input": False,
        "creation_slots": list(CREATION_SLOTS),
        "scenario_labels": list(SCENARIOS),
        "scenario_permutations_seeded_order": [
            list(order) for order in scenario_orders
        ],
        "role_seeded_order": list(role_order),
        "sessions": sessions,
    }
    digest = _schedule_payload_sha256(schedule)
    _require(SEED_SHA256 != LEGACY_SEED_SHA256, "Successor seed reused the old seed.")
    _require(
        digest != LEGACY_SCHEDULE_SHA256,
        "Successor schedule reused the old schedule identity.",
    )
    schedule["schedule_sha256"] = digest
    return schedule


def build_schedule(candidate_views: Sequence[str]) -> dict[str, Any]:
    """Return a new deterministic schedule for one fixed 18-view identity."""

    return _construct_schedule(candidate_views)


def validate_schedule(
    schedule: Mapping[str, Any],
    candidate_views: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Independently check identity plus every registered balance invariant."""

    checks: dict[str, bool] = {}
    errors: list[str] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks[name] = bool(passed)
        if not passed:
            errors.append(f"{name}: {detail}")

    if not isinstance(schedule, Mapping):
        return {
            "pass": False,
            "valid": False,
            "errors": ["schedule_type: schedule must be a mapping"],
            "checks": {"schedule_type": False},
            "counts": {},
        }

    try:
        candidates = _normalise_candidates(
            candidate_views
            if candidate_views is not None
            else schedule.get("candidate_views", ())
        )
        expected = _construct_schedule(candidates)
    except Exception as error:  # noqa: BLE001 - validator must return a report
        return {
            "pass": False,
            "valid": False,
            "errors": [f"construction: {type(error).__name__}: {error}"],
            "checks": {"construction": False},
            "counts": {},
        }

    record("schema_exact", schedule.get("schema") == SCHEMA, "schema drifted")
    record(
        "seed_exact_and_new",
        schedule.get("seed_text") == SEED_TEXT
        and schedule.get("seed_sha256") == SEED_SHA256
        and SEED_SHA256 != LEGACY_SEED_SHA256,
        "seed differs from the successor constant or aliases the old identity",
    )
    try:
        calculated_hash = _schedule_payload_sha256(schedule)
    except (TypeError, ValueError):
        calculated_hash = None
    record(
        "payload_hash_exact_and_new",
        calculated_hash is not None
        and schedule.get("schedule_sha256") == calculated_hash
        and calculated_hash != LEGACY_SCHEDULE_SHA256,
        "schedule hash is inconsistent or aliases the old identity",
    )
    record(
        "candidate_set_exact",
        schedule.get("candidate_views") == list(candidates),
        "candidate set differs from the frozen successor layout",
    )
    record(
        "deterministic_construction_exact",
        dict(schedule) == expected,
        "schedule differs from the new-seed algebraic construction",
    )

    primary_batch_counts: dict[str, Counter] = defaultdict(Counter)
    primary_slot_counts: dict[str, Counter] = defaultdict(Counter)
    primary_position_counts: dict[str, Counter] = defaultdict(Counter)
    bridge_batch_slot_counts: dict[str, Counter] = defaultdict(Counter)
    scenario_orders: Counter = Counter()
    occurrence_ids: list[str] = []
    session_pairs: set[tuple[str, int]] = set()
    structure_ok = True
    sessions = schedule.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 12:
        structure_ok = False
        sessions = []
    try:
        for session in sessions:
            session_pairs.add((str(session["character"]), int(session["repeat"])))
            scenario_orders[tuple(session["scenario_order"])] += 1
            seen_primary: list[str] = []
            batches = session["batches"]
            if len(batches) != 6:
                structure_ok = False
            for batch in batches:
                products = batch["products"]
                if (
                    len(products) != 4
                    or {int(product["creation_slot"]) for product in products}
                    != set(CREATION_SLOTS)
                ):
                    structure_ok = False
                primaries = [
                    product
                    for product in products
                    if product["occurrence_kind"] == "primary"
                ]
                bridges = [
                    product
                    for product in products
                    if product["occurrence_kind"] == "qa_bridge"
                ]
                if len(primaries) != 3 or len(bridges) != 1:
                    structure_ok = False
                for product in primaries:
                    view = str(product["view_id"])
                    seen_primary.append(view)
                    primary_batch_counts[view][batch["batch_id"]] += 1
                    primary_slot_counts[view][int(product["creation_slot"])] += 1
                    primary_position_counts[view][int(batch["execution_position"])] += 1
                    if product.get("science_input") is not True:
                        structure_ok = False
                for product in bridges:
                    bridge_batch_slot_counts[str(batch["batch_id"])][
                        int(product["creation_slot"])
                    ] += 1
                    if (
                        product.get("view_id") != QA_BRIDGE
                        or product.get("science_input") is not False
                    ):
                        structure_ok = False
                occurrence_ids.extend(str(product["occurrence_id"]) for product in products)
            if len(seen_primary) != 18 or set(seen_primary) != set(candidates):
                structure_ok = False
    except (KeyError, TypeError, ValueError):
        structure_ok = False

    record("session_batch_product_structure", structure_ok, "nested schedule structure drifted")
    record(
        "role_product_exact",
        session_pairs
        == {(character, repeat) for character in CHARACTERS for repeat in REPEATS},
        "the 3-repeat by 4-character product is incomplete",
    )
    record(
        "primary_batch_balance",
        all(
            primary_batch_counts[view]
            == Counter({batch: 2 for batch in BATCH_LABELS})
            for view in candidates
        ),
        "a primary does not use each batch twice",
    )
    record(
        "primary_slot_balance",
        all(
            primary_slot_counts[view]
            == Counter({slot: 3 for slot in CREATION_SLOTS})
            for view in candidates
        ),
        "a primary does not use each creation slot three times",
    )
    record(
        "primary_execution_position_balance",
        all(
            primary_position_counts[view]
            == Counter({position: 2 for position in range(1, 7)})
            for view in candidates
        ),
        "a primary does not use each execution position twice",
    )
    record(
        "qa_bridge_batch_slot_balance",
        all(
            bridge_batch_slot_counts[batch]
            == Counter({slot: 3 for slot in CREATION_SLOTS})
            for batch in BATCH_LABELS
        ),
        "QA bridge slot balance drifted within a batch",
    )
    expected_scenario_orders = set(itertools.permutations(SCENARIOS))
    record(
        "scenario_order_balance",
        set(scenario_orders) == expected_scenario_orders
        and all(scenario_orders[order] == 2 for order in expected_scenario_orders),
        "the six scenario orders do not each occur twice",
    )
    record(
        "occurrence_ids_unique",
        len(occurrence_ids) == 288 and len(set(occurrence_ids)) == 288,
        "occurrence IDs are missing or duplicated",
    )
    valid = bool(checks) and all(checks.values())
    return {
        "pass": valid,
        "valid": valid,
        "errors": errors,
        "checks": checks,
        "counts": {
            "sessions": len(sessions),
            "batches": len(sessions) * 6,
            "render_product_occurrences": len(occurrence_ids),
            "primary_occurrences": len(sessions) * 18,
            "qa_bridge_occurrences": len(sessions) * 6,
        },
        "calculated_schedule_sha256": calculated_hash,
    }


if __name__ == "__main__":
    raise SystemExit(
        "This module is pure and parameterised; call build_schedule(frozen_18_view_ids)."
    )
