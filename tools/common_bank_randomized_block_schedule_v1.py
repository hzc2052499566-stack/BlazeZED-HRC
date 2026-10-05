"""Deterministic, result-blind randomized-block schedule for common-bank v8.

The schedule is a design artifact, not an analysis result.  Its only inputs are
the constants in this module and :data:`SEED_TEXT`; construction performs no
file I/O and cannot inspect an experiment directory.  The fixed seed supplies
an auditable pseudo-random ordering while the algebraic block construction
provides exact balance rather than balance that merely holds by chance.

Public interface
----------------
``build_schedule()`` returns the complete 12-role-session schedule.
``validate_schedule(schedule)`` independently checks the frozen identity and
all scientific balance constraints and returns a report dictionary.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "fs_cts5_common_bank_randomized_block_schedule_v1"
SEED_TEXT = (
    "fs-cts5-common-bank-v8|randomized-block-schedule-v1|"
    "fixed-before-capture|result-blind|2026-08-22"
)
FROZEN_SCHEDULE_SHA256 = (
    "3947750b234113fd1ff16e90aaa7bae548945f0ca3082c33d07e0b2d25984cca"
)

CHARACTERS = ("F01", "F02", "M01", "M02")
REPEATS = (1, 2, 3)
SCENARIOS = ("G0", "G1", "W")
BATCH_LABELS = tuple(f"batch_{index}" for index in range(1, 7))
CREATION_SLOTS = (0, 1, 2, 3)

# Exact v7 candidate layout.  It is repeated here deliberately: schedule
# construction must not import a mutable protocol or inspect any capture/result.
CANDIDATE_VIEWS = (
    "az000_el00",
    "az030_el00",
    "az060_el00",
    "az090_el00",
    "az000_el05",
    "az000_el10",
    "az090_el05",
    "az150_el10",
    "az180_el00",
    "az180_el10",
    "az210_el10",
    "az240_el10",
    "az240_el20",
    "az270_el10",
    "az030_el01",
    "az030_el02",
    "az060_el01",
    "az060_el02",
)

BRIDGE_A = "az000_el00"
BRIDGE_B_GEOMETRY_ROLE = "level_view_at_least_90deg_from_bridge_A"
BRIDGE_B_MIN_AZIMUTH_SEPARATION_DEG = 90

# For the two role pairs that share the same six primary-batch offsets, these
# permutations assign a second, orthogonal offset.  Let p be the primary-batch
# rotation and d the primary execution-position rotation.  The batch execution
# rotation is e = p - d (mod 6).  Across the twelve sessions p, d, and e each
# contain every residue exactly twice.  Within every character's three repeats,
# all three values are different.
_EXECUTION_POSITION_ROTATIONS = (
    (0, 2, 1, 3, 5, 4),
    (2, 4, 0, 5, 1, 3),
)


def _digest(namespace: str, value: str) -> str:
    material = f"{SEED_TEXT}\n{namespace}\n{value}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _seeded_order(values: Sequence[str], namespace: str) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: (_digest(namespace, value), value)))


def _candidate_order() -> tuple[str, ...]:
    return _seeded_order(CANDIDATE_VIEWS, "candidate-primary-order")


def _role_order() -> tuple[str, ...]:
    return _seeded_order(CHARACTERS, "role-order")


def _scenario_orders() -> tuple[tuple[str, ...], ...]:
    permutations = tuple(itertools.permutations(SCENARIOS))
    return tuple(
        sorted(
            permutations,
            key=lambda order: (
                _digest("scenario-permutation-order", ">".join(order)),
                order,
            ),
        )
    )


def _view_geometry(view_id: str) -> tuple[int, int]:
    """Parse the frozen ``azDDD_elDD`` geometry identity."""

    try:
        azimuth_text, elevation_text = view_id.split("_el", 1)
        if not azimuth_text.startswith("az"):
            raise ValueError
        return int(azimuth_text[2:]), int(elevation_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid frozen view geometry ID: {view_id!r}") from exc


def _azimuth_separation_deg(first: str, second: str) -> int:
    first_azimuth, _ = _view_geometry(first)
    second_azimuth, _ = _view_geometry(second)
    difference = abs(first_azimuth - second_azimuth) % 360
    return min(difference, 360 - difference)


def _bridge_b_eligible(candidate_order: Sequence[str]) -> tuple[str, ...]:
    anchor_residue = candidate_order.index(BRIDGE_A) % len(BATCH_LABELS)
    eligible = []
    for index, view in enumerate(candidate_order):
        _, elevation = _view_geometry(view)
        if (
            view != BRIDGE_A
            and elevation == 0
            and _azimuth_separation_deg(BRIDGE_A, view)
            >= BRIDGE_B_MIN_AZIMUTH_SEPARATION_DEG
            and index % len(BATCH_LABELS) != anchor_residue
        ):
            eligible.append(view)
    if not eligible:
        raise RuntimeError("no candidate satisfies the frozen bridge B geometry role")
    return tuple(eligible)


def _bridge_b(candidate_order: Sequence[str]) -> str:
    """Choose B from geometry identities and the fixed seed only.

    B must have a different base batch residue from A.  Because every session
    rotates every candidate by the same amount, this proves that B's primary is
    never in A's primary batch; the B filler is therefore a genuine duplicate.
    """

    eligible = _bridge_b_eligible(candidate_order)
    return min(
        eligible,
        key=lambda view: (
            _digest("result-blind-bridge-B-geometry-tie", view),
            view,
        ),
    )


def _slot_design(anchor_base_batch: int) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Select a result-blind four-slot Latin design from a finite exact set.

    ``base_offsets`` has one value per candidate base-batch residue.  Adding the
    role rank modulo four makes each column a Latin permutation.  The filters
    enforce the divisibility-optimal role balance and exact logical-batch
    balance before the fixed seed chooses among feasible designs.
    """

    expected_three_each = Counter({slot: 3 for slot in CREATION_SLOTS})
    feasible: list[tuple[str, tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for base_offsets in itertools.product(CREATION_SLOTS, repeat=6):
        if sorted(Counter(base_offsets).values()) != [1, 1, 2, 2]:
            continue
        non_anchor_offsets = [
            value
            for base_batch, value in enumerate(base_offsets)
            if base_batch != anchor_base_batch
        ]
        if sorted(Counter(non_anchor_offsets).values()) != [1, 1, 1, 2]:
            continue
        latin = tuple(
            tuple((base_offsets[base_batch] + role_rank) % 4 for base_batch in range(6))
            for role_rank in range(4)
        )
        logical_batch_balanced = True
        for logical_batch in range(6):
            filler_slots: Counter[int] = Counter()
            for role_rank in range(4):
                for repeat_index in range(3):
                    primary_rotation = (3 * role_rank + repeat_index) % 6
                    base_batch = (logical_batch - primary_rotation) % 6
                    filler_slot = (
                        repeat_index + latin[role_rank][base_batch]
                    ) % 4
                    filler_slots[filler_slot] += 1
            if filler_slots != expected_three_each:
                logical_batch_balanced = False
                break
        if not logical_batch_balanced:
            continue
        encoded = ",".join(str(value) for value in base_offsets)
        feasible.append(
            (
                _digest("creation-slot-offset-design", encoded),
                tuple(base_offsets),
                latin,
            )
        )
    if not feasible:
        raise RuntimeError("no feasible fixed-seed creation-slot design")
    _, base_offsets, latin = min(feasible)
    return base_offsets, latin


def _schedule_payload_sha256(schedule: Mapping[str, Any]) -> str:
    payload = dict(schedule)
    payload.pop("schedule_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _session_specs(role_order: Sequence[str]) -> list[dict[str, int | str]]:
    """Return sessions in their counterbalanced global execution order."""

    specs: list[dict[str, int | str]] = []
    for repeat_index, repeat in enumerate(REPEATS):
        # Three cyclic orders use three distinct positions for every role while
        # each of the four positions contains exactly three sessions overall.
        order = tuple(
            role_order[(position + repeat_index) % len(role_order)]
            for position in range(len(role_order))
        )
        for role_position, character in enumerate(order, start=1):
            role_rank = role_order.index(character)
            primary_rotation = (3 * role_rank + repeat_index) % 6
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
                    "primary_execution_position_rotation": execution_position_rotation,
                    "batch_execution_rotation": batch_execution_rotation,
                }
            )
    return specs


def _construct_schedule() -> dict[str, Any]:
    candidate_order = _candidate_order()
    role_order = _role_order()
    scenario_orders = _scenario_orders()
    bridge_b = _bridge_b(candidate_order)
    anchor_base_batch = candidate_order.index(BRIDGE_A) % 6
    slot_base_offsets, slot_latin = _slot_design(anchor_base_batch)
    sessions: list[dict[str, Any]] = []

    for global_index, spec in enumerate(_session_specs(role_order), start=1):
        character = str(spec["character"])
        repeat = int(spec["repeat"])
        repeat_index = repeat - 1
        role_rank = int(spec["role_rank"])
        primary_rotation = int(spec["primary_batch_rotation"])
        batch_execution_rotation = int(spec["batch_execution_rotation"])
        session_id = f"{character}_rep_{repeat:02d}"

        primary_by_batch: dict[int, list[dict[str, Any]]] = {
            index: [] for index in range(6)
        }
        for candidate_index, view_id in enumerate(candidate_order):
            base_batch = candidate_index % 6
            candidate_row = candidate_index // 6
            logical_batch = (base_batch + primary_rotation) % 6
            filler_slot = (repeat_index + slot_latin[role_rank][base_batch]) % 4
            creation_slot = (filler_slot + 1 + candidate_row) % 4
            primary_by_batch[logical_batch].append(
                {
                    "occurrence_id": f"{session_id}:primary:{view_id}",
                    "occurrence_kind": "primary",
                    "science_input": True,
                    "view_id": view_id,
                    "creation_slot": creation_slot,
                }
            )

        anchor_primary_batch = next(
            batch_index
            for batch_index, products in primary_by_batch.items()
            if any(product["view_id"] == BRIDGE_A for product in products)
        )
        bridge_b_primary_batch = next(
            batch_index
            for batch_index, products in primary_by_batch.items()
            if any(product["view_id"] == bridge_b for product in products)
        )
        if anchor_primary_batch == bridge_b_primary_batch:  # construction invariant
            raise RuntimeError("bridge B primary unexpectedly shares bridge A batch")

        batches: list[dict[str, Any]] = []
        for logical_batch, batch_label in enumerate(BATCH_LABELS):
            execution_position = (
                (logical_batch - batch_execution_rotation) % 6
            ) + 1
            products = list(primary_by_batch[logical_batch])
            products.sort(key=lambda product: int(product["creation_slot"]))
            if logical_batch == anchor_primary_batch:
                filler_kind = "bridge_B"
                filler_view = bridge_b
            else:
                filler_kind = "bridge_A"
                filler_view = BRIDGE_A
            base_batch = (logical_batch - primary_rotation) % 6
            filler_slot = (repeat_index + slot_latin[role_rank][base_batch]) % 4
            products.append(
                {
                    "occurrence_id": f"{session_id}:{filler_kind}:{batch_label}",
                    "occurrence_kind": filler_kind,
                    "science_input": False,
                    "view_id": filler_view,
                    "creation_slot": filler_slot,
                }
            )
            products.sort(key=lambda product: int(product["creation_slot"]))
            batches.append(
                {
                    "batch_id": batch_label,
                    "logical_batch_index": logical_batch + 1,
                    "execution_position": execution_position,
                    "products": products,
                }
            )
        batches.sort(key=lambda batch: int(batch["execution_position"]))

        scenario_order = scenario_orders[primary_rotation]
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
                "batch_execution_rotation": batch_execution_rotation,
                "scenario_order": list(scenario_order),
                "batch_execution_order": [batch["batch_id"] for batch in batches],
                "batches": batches,
            }
        )

    schedule: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "deterministic_result_blind_schedule",
        "seed_text": SEED_TEXT,
        "seed_sha256": hashlib.sha256(SEED_TEXT.encode("utf-8")).hexdigest(),
        "construction_reads_experimental_results": False,
        "construction_uses_file_io": False,
        "characters": list(CHARACTERS),
        "repeats_per_character": len(REPEATS),
        "role_sessions": len(CHARACTERS) * len(REPEATS),
        "batches_per_session": len(BATCH_LABELS),
        "products_per_batch": 4,
        "primary_products_per_batch": 3,
        "candidate_views": list(CANDIDATE_VIEWS),
        "candidate_primary_order": list(candidate_order),
        "bridge_A": BRIDGE_A,
        "bridge_B": bridge_b,
        "bridge_B_geometry_role": BRIDGE_B_GEOMETRY_ROLE,
        "bridge_B_min_azimuth_separation_deg": (
            BRIDGE_B_MIN_AZIMUTH_SEPARATION_DEG
        ),
        "bridge_B_geometry_eligible": list(_bridge_b_eligible(candidate_order)),
        "bridge_B_selection": (
            "minimum fixed-seed SHA256 among frozen geometry-role candidates "
            "whose primary base-batch residue differs from A"
        ),
        "creation_slots": list(CREATION_SLOTS),
        "creation_slot_base_offsets": list(slot_base_offsets),
        "creation_slot_role_by_base_latin": [list(row) for row in slot_latin],
        "creation_slot_selection": (
            "minimum fixed-seed SHA256 among the finite exact-balance 4^6 "
            "base-offset designs"
        ),
        "scenario_labels": list(SCENARIOS),
        "scenario_permutations_seeded_order": [
            list(order) for order in scenario_orders
        ],
        "role_seeded_order": list(role_order),
        "sessions": sessions,
    }
    schedule_digest = _schedule_payload_sha256(schedule)
    if schedule_digest != FROZEN_SCHEDULE_SHA256:
        raise RuntimeError(
            "constructed schedule differs from FROZEN_SCHEDULE_SHA256: "
            f"{schedule_digest}"
        )
    schedule["schedule_sha256"] = schedule_digest
    return schedule


def build_schedule() -> dict:
    """Build and return the frozen deterministic v8 schedule.

    A new dictionary is constructed on every call.  No module-global mutable
    schedule is returned, and no filesystem or experimental result is read.
    """

    return _construct_schedule()


def validate_schedule(schedule: Mapping[str, Any]) -> dict:
    """Validate schedule identity and every registered balance constraint.

    The function is deliberately non-raising for invalid candidate schedules;
    callers receive ``{"valid": False, ...}`` with actionable error strings.
    """

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

    expected = _construct_schedule()
    record("schema_exact", schedule.get("schema") == SCHEMA, "unexpected schema")
    record(
        "fixed_seed_exact",
        schedule.get("seed_text") == SEED_TEXT
        and schedule.get("seed_sha256")
        == hashlib.sha256(SEED_TEXT.encode("utf-8")).hexdigest(),
        "seed text or digest differs from the preregistered constant",
    )
    record(
        "result_blind_declaration",
        schedule.get("construction_reads_experimental_results") is False
        and schedule.get("construction_uses_file_io") is False,
        "construction must declare no result reads and no file I/O",
    )
    record(
        "candidate_set_exact",
        schedule.get("candidate_views") == list(CANDIDATE_VIEWS),
        "candidate list differs from the fixed eighteen-view v7 layout",
    )
    record(
        "candidate_primary_order_exact",
        schedule.get("candidate_primary_order") == list(_candidate_order()),
        "primary order is not the fixed-seed order",
    )
    expected_bridge_b = _bridge_b(_candidate_order())
    record("bridge_A_exact", schedule.get("bridge_A") == BRIDGE_A, "bridge A changed")
    record(
        "bridge_B_result_blind_exact",
        schedule.get("bridge_B") == expected_bridge_b,
        "bridge B is not the fixed-seed result-blind choice",
    )
    bridge_b_geometry_ok = (
        schedule.get("bridge_B_geometry_role") == BRIDGE_B_GEOMETRY_ROLE
        and schedule.get("bridge_B_min_azimuth_separation_deg")
        == BRIDGE_B_MIN_AZIMUTH_SEPARATION_DEG
        and schedule.get("bridge_B_geometry_eligible")
        == list(_bridge_b_eligible(_candidate_order()))
        and _azimuth_separation_deg(BRIDGE_A, expected_bridge_b)
        >= BRIDGE_B_MIN_AZIMUTH_SEPARATION_DEG
        and _view_geometry(expected_bridge_b)[1] == 0
    )
    record(
        "bridge_B_geometry_role_exact",
        bridge_b_geometry_ok,
        "bridge B geometry eligibility or fixed-seed tie-break drifted",
    )
    anchor_base_batch = _candidate_order().index(BRIDGE_A) % 6
    expected_slot_offsets, expected_slot_latin = _slot_design(anchor_base_batch)
    record(
        "creation_slot_design_exact",
        schedule.get("creation_slots") == list(CREATION_SLOTS)
        and schedule.get("creation_slot_base_offsets")
        == list(expected_slot_offsets)
        and schedule.get("creation_slot_role_by_base_latin")
        == [list(row) for row in expected_slot_latin],
        "creation-slot Latin design differs from the fixed-seed exact solution",
    )

    supplied_hash = schedule.get("schedule_sha256")
    try:
        calculated_hash = _schedule_payload_sha256(schedule)
    except (TypeError, ValueError) as exc:
        calculated_hash = None
        errors.append(f"schedule_hash_computable: {exc}")
        checks["schedule_hash_computable"] = False
    else:
        checks["schedule_hash_computable"] = True
    record(
        "schedule_hash_exact",
        isinstance(supplied_hash, str) and supplied_hash == calculated_hash,
        "stored schedule SHA256 does not match canonical payload",
    )

    sessions = schedule.get("sessions")
    record(
        "twelve_role_sessions",
        isinstance(sessions, list) and len(sessions) == 12,
        "expected four characters times three repeats",
    )
    if not isinstance(sessions, list):
        sessions = []

    character_repeat_counts: Counter[tuple[str, int]] = Counter()
    global_execution_indices: list[int] = []
    session_ids: list[str] = []
    role_position_counts: Counter[int] = Counter()
    role_positions: defaultdict[str, set[int]] = defaultdict(set)
    repeat_role_positions: defaultdict[int, dict[int, str]] = defaultdict(dict)
    scenario_order_counts: Counter[tuple[str, ...]] = Counter()
    batch_position_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    view_batch_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    view_execution_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    view_character_batches: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    view_character_slots: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    view_creation_slot_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    character_primary_slot_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    filler_slot_counts: Counter[int] = Counter()
    bridge_a_slot_counts: Counter[int] = Counter()
    bridge_b_slot_counts: Counter[int] = Counter()
    character_filler_slot_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    character_bridge_a_slot_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    character_bridge_b_slot_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    repeat_filler_slot_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
    batch_filler_slot_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    all_occurrence_ids: list[str] = []
    session_structure_ok = True
    bridge_structure_ok = True

    valid_permutations = set(itertools.permutations(SCENARIOS))
    for session in sessions:
        if not isinstance(session, Mapping):
            session_structure_ok = False
            continue
        try:
            character = str(session["character"])
            repeat = int(session["repeat"])
            repeat_index = int(session["repeat_index"])
            session_id = str(session["session_id"])
            global_index = int(session["global_execution_index"])
            role_position = int(session["role_session_position"])
            raw_scenario_order = session["scenario_order"]
            batches = session["batches"]
        except (KeyError, TypeError, ValueError, OverflowError):
            session_structure_ok = False
            continue
        if not isinstance(raw_scenario_order, (list, tuple)) or not all(
            isinstance(item, str) for item in raw_scenario_order
        ):
            scenario_order: tuple[str, ...] = ()
            session_structure_ok = False
        else:
            scenario_order = tuple(raw_scenario_order)

        character_repeat_counts[(character, repeat)] += 1
        if repeat_index != repeat:
            session_structure_ok = False
        global_execution_indices.append(global_index)
        session_ids.append(session_id)
        role_position_counts[role_position] += 1
        role_positions[character].add(role_position)
        if role_position in repeat_role_positions[repeat]:
            session_structure_ok = False
        repeat_role_positions[repeat][role_position] = character
        scenario_order_counts[scenario_order] += 1
        if scenario_order not in valid_permutations:
            session_structure_ok = False
        if not isinstance(batches, list) or len(batches) != 6:
            session_structure_ok = False
            continue

        batch_ids: list[str] = []
        execution_positions: list[int] = []
        session_primary: list[dict[str, Any]] = []
        session_fillers: list[tuple[str, dict[str, Any]]] = []
        session_occurrences: list[str] = []
        primary_batch_for_view: dict[str, str] = {}

        for batch in batches:
            if not isinstance(batch, Mapping):
                session_structure_ok = False
                continue
            try:
                batch_id = str(batch["batch_id"])
                execution_position = int(batch["execution_position"])
                products = batch["products"]
            except (KeyError, TypeError, ValueError, OverflowError):
                session_structure_ok = False
                continue
            batch_ids.append(batch_id)
            execution_positions.append(execution_position)
            batch_position_counts[batch_id][execution_position] += 1
            if not isinstance(products, list) or len(products) != 4:
                session_structure_ok = False
                continue
            slots: list[int] = []
            primaries: list[dict[str, Any]] = []
            fillers: list[dict[str, Any]] = []
            for product in products:
                if not isinstance(product, Mapping):
                    session_structure_ok = False
                    continue
                try:
                    slot = int(product["creation_slot"])
                    kind = str(product["occurrence_kind"])
                    view_id = str(product["view_id"])
                    occurrence_id = str(product["occurrence_id"])
                    science_input = product["science_input"]
                except (KeyError, TypeError, ValueError, OverflowError):
                    session_structure_ok = False
                    continue
                slots.append(slot)
                session_occurrences.append(occurrence_id)
                all_occurrence_ids.append(occurrence_id)
                if kind == "primary":
                    if science_input is not True or slot not in CREATION_SLOTS:
                        session_structure_ok = False
                    item = dict(product)
                    primaries.append(item)
                    session_primary.append(item)
                    primary_batch_for_view[view_id] = batch_id
                    view_batch_counts[view_id][batch_id] += 1
                    view_execution_counts[view_id][execution_position] += 1
                    view_character_batches[(view_id, character)].add(batch_id)
                    view_character_slots[(view_id, character)].add(slot)
                    view_creation_slot_counts[view_id][slot] += 1
                    character_primary_slot_counts[character][slot] += 1
                else:
                    if science_input is not False or slot not in CREATION_SLOTS:
                        bridge_structure_ok = False
                    item = dict(product)
                    fillers.append(item)
                    session_fillers.append((batch_id, item))
                    filler_slot_counts[slot] += 1
                    character_filler_slot_counts[character][slot] += 1
                    repeat_filler_slot_counts[repeat][slot] += 1
                    batch_filler_slot_counts[batch_id][slot] += 1
                    if kind == "bridge_A":
                        bridge_a_slot_counts[slot] += 1
                        character_bridge_a_slot_counts[character][slot] += 1
                    elif kind == "bridge_B":
                        bridge_b_slot_counts[slot] += 1
                        character_bridge_b_slot_counts[character][slot] += 1
            if sorted(slots) != [0, 1, 2, 3] or len(primaries) != 3 or len(fillers) != 1:
                session_structure_ok = False

        if set(batch_ids) != set(BATCH_LABELS) or set(execution_positions) != set(
            range(1, 7)
        ):
            session_structure_ok = False
        ordered_ids = [
            batch_id
            for _, batch_id in sorted(zip(execution_positions, batch_ids))
        ]
        if session.get("batch_execution_order") != ordered_ids:
            session_structure_ok = False
        primary_views = [str(product.get("view_id")) for product in session_primary]
        if len(primary_views) != 18 or set(primary_views) != set(CANDIDATE_VIEWS):
            session_structure_ok = False
        if len(session_occurrences) != len(set(session_occurrences)):
            session_structure_ok = False

        anchor_primary_batch = primary_batch_for_view.get(BRIDGE_A)
        bridge_b_primary_batch = primary_batch_for_view.get(expected_bridge_b)
        if (
            anchor_primary_batch is None
            or bridge_b_primary_batch is None
            or anchor_primary_batch == bridge_b_primary_batch
        ):
            bridge_structure_ok = False
        kinds = Counter(str(product.get("occurrence_kind")) for _, product in session_fillers)
        if kinds != Counter({"bridge_A": 5, "bridge_B": 1}):
            bridge_structure_ok = False
        for batch_id, filler in session_fillers:
            expected_kind = "bridge_B" if batch_id == anchor_primary_batch else "bridge_A"
            expected_view = expected_bridge_b if expected_kind == "bridge_B" else BRIDGE_A
            if (
                filler.get("occurrence_kind") != expected_kind
                or filler.get("view_id") != expected_view
            ):
                bridge_structure_ok = False

    expected_character_repeats = Counter(
        (character, repeat) for character in CHARACTERS for repeat in REPEATS
    )
    record(
        "character_repeat_factorial",
        character_repeat_counts == expected_character_repeats,
        "each character/repeat pair must occur exactly once",
    )
    record(
        "session_identity_unique_and_ordered",
        len(session_ids) == len(set(session_ids)) == 12
        and global_execution_indices == list(range(1, 13)),
        "session IDs must be unique and global execution indices must be 1..12",
    )
    record(
        "per_session_product_structure",
        session_structure_ok,
        "a session, batch, primary, slot, or occurrence invariant failed",
    )
    record(
        "bridge_structure",
        bridge_structure_ok,
        "A must fill five batches and result-blind B must fill A's primary batch",
    )

    role_balance_ok = (
        role_position_counts == Counter({1: 3, 2: 3, 3: 3, 4: 3})
        and all(
            role_positions[character] and len(role_positions[character]) == 3
            for character in CHARACTERS
        )
        and all(
            set(repeat_role_positions[repeat]) == {1, 2, 3, 4}
            and set(repeat_role_positions[repeat].values()) == set(CHARACTERS)
            for repeat in REPEATS
        )
    )
    record(
        "role_session_order_balanced",
        role_balance_ok,
        (
            "each role needs three distinct within-repeat positions; "
            "each position needs three sessions"
        ),
    )

    all_scenario_orders = set(itertools.permutations(SCENARIOS))
    scenario_balance_ok = (
        set(scenario_order_counts) == all_scenario_orders
        and all(scenario_order_counts[order] == 2 for order in all_scenario_orders)
    )
    record(
        "scenario_orders_each_twice",
        scenario_balance_ok,
        "all six G0/G1/W permutations must occur exactly twice",
    )

    batch_position_ok = all(
        batch_position_counts[label] == Counter({position: 2 for position in range(1, 7)})
        for label in BATCH_LABELS
    )
    record(
        "batch_execution_positions_each_twice",
        batch_position_ok,
        "each logical batch must occupy every execution position exactly twice",
    )

    view_batch_ok = all(
        view_batch_counts[view] == Counter({label: 2 for label in BATCH_LABELS})
        for view in CANDIDATE_VIEWS
    )
    record(
        "view_primary_batches_each_twice",
        view_batch_ok,
        "every primary view must occur twice in every logical batch",
    )
    within_character_batch_ok = all(
        len(view_character_batches[(view, character)]) == 3
        for view in CANDIDATE_VIEWS
        for character in CHARACTERS
    )
    record(
        "within_character_primary_batches_distinct",
        within_character_batch_ok,
        "a view's three character repeats must use three different batches",
    )
    within_character_slot_ok = all(
        len(view_character_slots[(view, character)]) == 3
        and view_character_slots[(view, character)].issubset(set(CREATION_SLOTS))
        for view in CANDIDATE_VIEWS
        for character in CHARACTERS
    )
    record(
        "within_character_creation_slots_distinct",
        within_character_slot_ok,
        "a view's three character repeats must use three distinct slots",
    )
    creation_slot_global_ok = all(
        view_creation_slot_counts[view]
        == Counter({slot: 3 for slot in CREATION_SLOTS})
        for view in CANDIDATE_VIEWS
    )
    record(
        "view_creation_slots_globally_balanced",
        creation_slot_global_ok,
        "each primary view must use all four creation slots exactly three times",
    )

    role_slot_balance_ok = all(
        sorted(character_primary_slot_counts[character].values())
        == [13, 13, 14, 14]
        and sorted(character_filler_slot_counts[character].values())
        == [4, 4, 5, 5]
        and sorted(character_bridge_a_slot_counts[character].values())
        == [3, 4, 4, 4]
        and len(character_bridge_b_slot_counts[character]) == 3
        and set(character_bridge_b_slot_counts[character].values()) == {1}
        for character in CHARACTERS
    )
    record(
        "role_by_creation_slot_balanced",
        role_slot_balance_ok,
        "primary/filler/A/B slot counts exceed the divisibility-optimal role balance",
    )

    bridge_slot_balance_ok = (
        filler_slot_counts == Counter({slot: 18 for slot in CREATION_SLOTS})
        and bridge_a_slot_counts == Counter({slot: 15 for slot in CREATION_SLOTS})
        and bridge_b_slot_counts == Counter({slot: 3 for slot in CREATION_SLOTS})
        and all(
            repeat_filler_slot_counts[repeat]
            == Counter({slot: 6 for slot in CREATION_SLOTS})
            for repeat in REPEATS
        )
        and all(
            batch_filler_slot_counts[batch]
            == Counter({slot: 3 for slot in CREATION_SLOTS})
            for batch in BATCH_LABELS
        )
    )
    record(
        "bridge_creation_slots_balanced",
        bridge_slot_balance_ok,
        "filler, bridge A, or bridge B remains confounded with creation slot",
    )

    # Stronger than the requested batch-label balance: it proves that the view
    # itself is not confounded with execution position after both rotations.
    view_execution_ok = all(
        view_execution_counts[view]
        == Counter({position: 2 for position in range(1, 7)})
        for view in CANDIDATE_VIEWS
    )
    record(
        "view_primary_execution_positions_each_twice",
        view_execution_ok,
        "every primary view must occupy every execution position exactly twice",
    )
    record(
        "occurrence_ids_globally_unique",
        len(all_occurrence_ids) == len(set(all_occurrence_ids)) == 12 * 6 * 4,
        "all 288 product occurrences require unique IDs",
    )

    record(
        "frozen_schedule_exact",
        schedule == expected,
        "schedule differs from the fixed-seed canonical construction",
    )

    counts = {
        "role_sessions": len(sessions),
        "product_occurrences": len(all_occurrence_ids),
        "primary_occurrences": sum(
            sum(counter.values()) for counter in view_batch_counts.values()
        ),
        "bridge_occurrences": len(all_occurrence_ids)
        - sum(sum(counter.values()) for counter in view_batch_counts.values()),
        "scenario_order_counts": {
            ">".join(order): scenario_order_counts[order]
            for order in sorted(all_scenario_orders)
        },
    }
    is_valid = not errors and all(checks.values())
    return {
        "pass": is_valid,
        "valid": is_valid,
        "errors": errors,
        "checks": checks,
        "counts": counts,
        "schedule_sha256": supplied_hash,
        "calculated_schedule_sha256": calculated_hash,
    }


__all__ = [
    "build_schedule",
    "validate_schedule",
]
