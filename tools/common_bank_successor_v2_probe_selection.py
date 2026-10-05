"""Pure-Python selector for the successor-v2 raw-event probe.

This module is deliberately separate from the frozen v8 selector.  It consumes
exactly two *raw-event summaries* from an excluded development probe and chooses
12 event-band candidates from the frozen 24-view grid.  Its closed input shape is
``repeats[repeat_id][view_id][character][side] = side_summary``.  It never reads
images, depth, coordinates, ground-truth values, errors, or an old selection
result, and it has no file-system or command-line side effects.

The replay sets ``found`` only for a contiguous
``clear >= 5 -> table-occluded >= 20 -> clear >= 5`` event inside the registered
march window.  The selector validates the counts of every positive record.  A
view is a robust carrier for one ``character x side`` stratum only when ``found``
is true independently in both input repeats.

The exhaustive ``C(24, 12)`` ordering is, best first:

1. greatest worst-stratum third-best clear-flank slack;
2. greatest worst-stratum third-best occluded-run slack;
3. greatest minimum robust-carrier count;
4. greatest total robust-carrier incidence;
5. greatest minimum pairwise camera angle; and
6. lexicographically smallest sorted view-ID tuple.

The probe only designs a fresh candidate layout.  Its output is not a camera
bank, a formal result, or an authorization to reuse the failed v8/rpr chain.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Mapping, Sequence


RECORD = "common_bank_successor_v2_raw_event_probe_selection_v1"
REPEAT_COUNT = 2
SELECTED_COUNT = 12
MIN_ROBUST_CARRIERS_PER_STRATUM = 3
MIN_CLEAR_FRAMES = 5
MIN_OCCLUDED_FRAMES = 20
RADIUS_M = 3.5
CHARACTERS = ("F01", "F02", "M01", "M02")
SIDES = ("left", "right")
STRATA = tuple((character, side) for character in CHARACTERS for side in SIDES)
MARCH_WINDOWS = {"left": (75, 150), "right": (155, 230)}
SIDE_BONES = {
    "left": ("left_thigh", "left_shank"),
    "right": ("right_thigh", "right_shank"),
}

# The grid is the prospective, excluded raw-RGB-D event probe.  Every camera
# uses the same aim, radius, resolution and camera model; only azimuth/elevation
# differ.  IDs encode signed decimal elevations without punctuation so they are
# safe as directory names on every supported host.
PROBE_LAYOUT = (
    {"view_id": "az345_elm02p0", "azimuth_deg": 345.0, "elevation_deg": -2.0},
    {"view_id": "az345_el01p5", "azimuth_deg": 345.0, "elevation_deg": 1.5},
    {"view_id": "az345_el05p5", "azimuth_deg": 345.0, "elevation_deg": 5.5},
    {"view_id": "az345_el07p5", "azimuth_deg": 345.0, "elevation_deg": 7.5},
    {"view_id": "az000_elm02p0", "azimuth_deg": 0.0, "elevation_deg": -2.0},
    {"view_id": "az000_el01p5", "azimuth_deg": 0.0, "elevation_deg": 1.5},
    {"view_id": "az000_el05p5", "azimuth_deg": 0.0, "elevation_deg": 5.5},
    {"view_id": "az000_el07p5", "azimuth_deg": 0.0, "elevation_deg": 7.5},
    {"view_id": "az015_elm02p0", "azimuth_deg": 15.0, "elevation_deg": -2.0},
    {"view_id": "az015_el01p5", "azimuth_deg": 15.0, "elevation_deg": 1.5},
    {"view_id": "az015_el05p5", "azimuth_deg": 15.0, "elevation_deg": 5.5},
    {"view_id": "az015_el07p5", "azimuth_deg": 15.0, "elevation_deg": 7.5},
    {"view_id": "az045_elm02p0", "azimuth_deg": 45.0, "elevation_deg": -2.0},
    {"view_id": "az045_el00p5", "azimuth_deg": 45.0, "elevation_deg": 0.5},
    {"view_id": "az045_el01p5", "azimuth_deg": 45.0, "elevation_deg": 1.5},
    {"view_id": "az045_el02p5", "azimuth_deg": 45.0, "elevation_deg": 2.5},
    {"view_id": "az060_elm02p0", "azimuth_deg": 60.0, "elevation_deg": -2.0},
    {"view_id": "az060_el00p5", "azimuth_deg": 60.0, "elevation_deg": 0.5},
    {"view_id": "az060_el01p5", "azimuth_deg": 60.0, "elevation_deg": 1.5},
    {"view_id": "az060_el02p5", "azimuth_deg": 60.0, "elevation_deg": 2.5},
    {"view_id": "az075_elm02p0", "azimuth_deg": 75.0, "elevation_deg": -2.0},
    {"view_id": "az075_el00p5", "azimuth_deg": 75.0, "elevation_deg": 0.5},
    {"view_id": "az075_el01p5", "azimuth_deg": 75.0, "elevation_deg": 1.5},
    {"view_id": "az075_el02p5", "azimuth_deg": 75.0, "elevation_deg": 2.5},
)
PROBE_VIEW_IDS = tuple(entry["view_id"] for entry in PROBE_LAYOUT)

_ANGLE_QUANTUM = Decimal("1e-9")
_NANODEGREES_PER_DEGREE = 1_000_000_000
_MAX_ANGLE_NANODEG = 180 * _NANODEGREES_PER_DEGREE
_SIDE_SUMMARY_KEYS = frozenset(
    (
        "found",
        "bone",
        "clear_before_frames",
        "occluded_frames",
        "clear_after_frames",
    )
)
_FORBIDDEN_KEYS = frozenset(
    (
        "gt",
        "gt_error",
        "ground_truth",
        "mpjpe",
        "position_error",
        "limb_error",
        "accuracy",
    )
)


class ProbeSelectionError(RuntimeError):
    """Raised when a raw-event summary cannot be evaluated exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeSelectionError(message)


def _closed_keys(value: Mapping, expected: frozenset[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping.")
    _require(all(isinstance(key, str) for key in value), f"{label} field names must be text.")
    keys = frozenset(value)
    _require(not (keys & _FORBIDDEN_KEYS), f"{label} contains a forbidden result field.")
    _require(keys == expected, f"{label} fields differ from the closed schema.")


def _non_negative_count(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer frame count.",
    )
    return int(value)


def side_summary_slacks(summary: Mapping, side: str) -> tuple[int, int] | None:
    """Validate one replay side summary and return its two ranking slacks.

    ``found`` is a replay-derived raw-event fact.  Every positive record must
    satisfy the supplied 5/20/5 counts; a negative record may retain near-miss
    counts because count-only input cannot restate the replay's contiguity fact.
    The three counts must fit inside the side's 76-timecode march window.  A
    valid non-event returns ``None``.
    """

    _require(side in SIDES, "Unknown event side.")
    _closed_keys(summary, _SIDE_SUMMARY_KEYS, "side summary")
    found = summary["found"]
    _require(isinstance(found, bool), "found must be Boolean.")
    _require(summary["bone"] in SIDE_BONES[side], "Event bone is on the wrong side.")
    before_length = _non_negative_count(
        summary["clear_before_frames"], "clear_before_frames"
    )
    occluded_length = _non_negative_count(
        summary["occluded_frames"], "occluded_frames"
    )
    after_length = _non_negative_count(
        summary["clear_after_frames"], "clear_after_frames"
    )
    lo, hi = MARCH_WINDOWS[side]
    _require(
        before_length + occluded_length + after_length <= hi - lo + 1,
        "Side-summary counts exceed the march window.",
    )
    if not found:
        return None
    _require(
        before_length >= MIN_CLEAR_FRAMES
        and occluded_length >= MIN_OCCLUDED_FRAMES
        and after_length >= MIN_CLEAR_FRAMES,
        "A found event does not satisfy the supplied 5/20/5 counts.",
    )
    return (
        min(before_length, after_length) - MIN_CLEAR_FRAMES,
        occluded_length - MIN_OCCLUDED_FRAMES,
    )


def choose_side_summary(bone_summaries: Sequence[Mapping], side: str) -> dict:
    """Choose thigh/shank by max(found,min-clear,occ,total-clear,bone-ID).

    This is the canonical pure helper for the replay summarizer.  Including the
    bone ID in the maximized tuple makes a complete tie deterministic.
    """

    _require(side in SIDES, "Unknown event side.")
    _require(
        not isinstance(bone_summaries, Mapping)
        and isinstance(bone_summaries, Sequence)
        and len(bone_summaries) == len(SIDE_BONES[side]),
        "Exactly one thigh and one shank summary are required.",
    )
    validated = []
    for summary in bone_summaries:
        _require(isinstance(summary, Mapping), "Each bone summary must be a mapping.")
        side_summary_slacks(summary, side)
        validated.append(summary)
    _require(
        {summary["bone"] for summary in validated} == set(SIDE_BONES[side]),
        "The side candidates must be exactly its thigh and shank.",
    )

    def ordering(summary: Mapping) -> tuple:
        before = int(summary["clear_before_frames"])
        after = int(summary["clear_after_frames"])
        return (
            int(summary["found"]),
            min(before, after),
            int(summary["occluded_frames"]),
            before + after,
            str(summary["bone"]),
        )

    return dict(max(validated, key=ordering))


def robust_carrier_profiles(repeats: Mapping[str, Mapping]) -> dict:
    """Build repeat-worst carrier margins for all 24 views and eight strata."""

    _require(
        isinstance(repeats, Mapping) and len(repeats) == REPEAT_COUNT,
        "Exactly two repeat summaries are required.",
    )
    _require(
        all(isinstance(repeat_id, str) and repeat_id for repeat_id in repeats),
        "Repeat IDs must be non-empty text.",
    )
    expected_views = set(PROBE_VIEW_IDS)
    per_repeat = []
    repeat_ids = sorted(repeats)
    for repeat_id in repeat_ids:
        by_view = repeats[repeat_id]
        _require(isinstance(by_view, Mapping), "events_by_view must be a mapping.")
        _require(set(by_view) == expected_views, "Repeat does not cover the exact 24-view grid.")
        repeat_profile = {}
        for view_id in PROBE_VIEW_IDS:
            by_character = by_view[view_id]
            _require(isinstance(by_character, Mapping), f"{view_id} must map characters.")
            _require(set(by_character) == set(CHARACTERS), f"{view_id} character set drifted.")
            for character in CHARACTERS:
                by_side = by_character[character]
                _require(isinstance(by_side, Mapping), f"{view_id}/{character} must map sides.")
                _require(set(by_side) == set(SIDES), f"{view_id}/{character} side set drifted.")
                for side in SIDES:
                    summary = by_side[side]
                    _require(
                        isinstance(summary, Mapping),
                        f"{view_id}/{character}/{side} must be a side summary.",
                    )
                    repeat_profile[(view_id, character, side)] = side_summary_slacks(
                        summary, side
                    )
        per_repeat.append(repeat_profile)

    profiles = {stratum: {} for stratum in STRATA}
    for view_id in PROBE_VIEW_IDS:
        for character, side in STRATA:
            values = [
                profile[(view_id, character, side)] for profile in per_repeat
            ]
            if all(value is not None for value in values):
                profiles[(character, side)][view_id] = (
                    min(value[0] for value in values),
                    min(value[1] for value in values),
                )
    return {
        "repeat_ids": repeat_ids,
        "profiles": profiles,
        "robust_carriers_by_stratum": {
            f"{character}|{side}": sorted(profiles[(character, side)])
            for character, side in STRATA
        },
    }


def _unit_camera_to_aim(azimuth_deg: float, elevation_deg: float) -> tuple[float, ...]:
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    return (
        -math.cos(elevation) * math.cos(azimuth),
        -math.cos(elevation) * math.sin(azimuth),
        -math.sin(elevation),
    )


def _angle_nanodegrees(first: Sequence[float], second: Sequence[float]) -> int:
    difference = math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))
    total = math.sqrt(sum((a + b) ** 2 for a, b in zip(first, second)))
    degrees = math.degrees(2.0 * math.atan2(difference, total))
    return int(
        (Decimal(repr(float(degrees))) / _ANGLE_QUANTUM).quantize(
            Decimal(1), rounding=ROUND_HALF_EVEN
        )
    )


def pairwise_angle_table(layout: Sequence[Mapping] = PROBE_LAYOUT) -> dict:
    """Return a platform-stable integer-nanodegree table for one layout."""

    ids = [str(entry["view_id"]) for entry in layout]
    _require(len(ids) == len(set(ids)), "Layout view IDs are not unique.")
    vectors = {
        str(entry["view_id"]): _unit_camera_to_aim(
            float(entry["azimuth_deg"]), float(entry["elevation_deg"])
        )
        for entry in layout
    }
    return {
        first: {
            second: (
                0
                if first == second
                else _angle_nanodegrees(vectors[first], vectors[second])
            )
            for second in ids
        }
        for first in ids
    }


def _minimum_angle(view_ids: Sequence[str], angle_table: Mapping) -> int:
    if len(view_ids) < 2:
        return _MAX_ANGLE_NANODEG
    return min(
        int(angle_table[first][second])
        for index, first in enumerate(view_ids)
        for second in view_ids[index + 1 :]
    )


def _third_best(values: Sequence[int]) -> int:
    _require(len(values) >= 3, "A third-best value needs at least three carriers.")
    return sorted((int(value) for value in values), reverse=True)[2]


def _fixed_popcount_masks(item_count: int, selected_count: int):
    """Yield every fixed-popcount mask exactly once using Gosper's hack."""

    mask = (1 << selected_count) - 1
    limit = 1 << item_count
    while mask < limit:
        yield mask
        low_bit = mask & -mask
        ripple = mask + low_bit
        mask = ripple | (((ripple ^ mask) >> 2) // low_bit)


def _lexicographically_less_mask(candidate: int, incumbent: int) -> bool:
    """Compare equal-popcount subsets in ascending view-ID order."""

    difference = candidate ^ incumbent
    if not difference:
        return False
    first_difference = difference & -difference
    return bool(candidate & first_difference)


def _threshold_masks(values_by_index: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    """Return descending ``(slack, views-at-or-above-slack mask)`` levels."""

    exact = {}
    for index, value in values_by_index.items():
        exact.setdefault(int(value), 0)
        exact[int(value)] |= 1 << int(index)
    cumulative = 0
    levels = []
    for value in sorted(exact, reverse=True):
        cumulative |= exact[value]
        levels.append((value, cumulative))
    return tuple(levels)


def _third_best_from_levels(selected_mask: int, levels: Sequence[tuple[int, int]]) -> int:
    for value, threshold_mask in levels:
        if (selected_mask & threshold_mask).bit_count() >= 3:
            return int(value)
    raise ProbeSelectionError("Carrier-mask indexing lost a third-best value.")


def _angle_mask_index(view_ids: Sequence[str], angle_table: Mapping) -> dict:
    """Precompute two 12-bit halves so subset angles need at most 12 lookups."""

    item_count = len(view_ids)
    split = item_count // 2
    right_count = item_count - split
    matrix = []
    for first in view_ids:
        _require(first in angle_table, "Angle table is missing a view row.")
        row = []
        for second in view_ids:
            _require(second in angle_table[first], "Angle table is missing a view pair.")
            value = angle_table[first][second]
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                "Angles must be non-negative integer nanodegrees.",
            )
            row.append(int(value))
        matrix.append(tuple(row))

    def subset_minima(offset: int, size: int) -> list[int]:
        count = 1 << size
        to_subset = []
        for local_first in range(size):
            row = [_MAX_ANGLE_NANODEG] * count
            for mask in range(1, count):
                bit = mask & -mask
                local_second = bit.bit_length() - 1
                row[mask] = min(
                    row[mask ^ bit],
                    matrix[offset + local_first][offset + local_second],
                )
            to_subset.append(row)
        within = [_MAX_ANGLE_NANODEG] * count
        for mask in range(1, count):
            bit = mask & -mask
            local_first = bit.bit_length() - 1
            rest = mask ^ bit
            within[mask] = min(within[rest], to_subset[local_first][rest])
        return within

    right_masks = 1 << right_count
    cross = []
    for left_index in range(split):
        row = [_MAX_ANGLE_NANODEG] * right_masks
        for mask in range(1, right_masks):
            bit = mask & -mask
            right_index = bit.bit_length() - 1
            row[mask] = min(
                row[mask ^ bit], matrix[left_index][split + right_index]
            )
        cross.append(row)
    return {
        "split": split,
        "left_mask": (1 << split) - 1,
        "within_left": subset_minima(0, split),
        "within_right": subset_minima(split, right_count),
        "cross": cross,
    }


def _minimum_angle_mask(selected_mask: int, index: Mapping) -> int:
    left = selected_mask & int(index["left_mask"])
    right = selected_mask >> int(index["split"])
    minimum = min(index["within_left"][left], index["within_right"][right])
    remaining = left
    while remaining:
        bit = remaining & -remaining
        left_index = bit.bit_length() - 1
        minimum = min(minimum, index["cross"][left_index][right])
        remaining ^= bit
    return int(minimum)


def _search_profiles(
    profiles: Mapping,
    view_ids: Sequence[str],
    angle_table: Mapping,
    selected_count: int,
    min_carriers: int,
) -> dict:
    """Exhaustively search profiles; parameters exist for compact unit fixtures."""

    ordered_views = tuple(sorted(str(view_id) for view_id in view_ids))
    _require(len(ordered_views) == len(set(ordered_views)), "Search view IDs are not unique.")
    _require(0 < selected_count <= len(ordered_views), "selected_count is invalid.")
    _require(3 <= min_carriers <= selected_count, "min_carriers is invalid.")
    strata = tuple(sorted(profiles))
    _require(bool(strata), "At least one stratum is required.")
    for stratum in strata:
        _require(isinstance(profiles[stratum], Mapping), "Each stratum profile must be a mapping.")
        _require(
            set(profiles[stratum]).issubset(set(ordered_views)),
            "A carrier profile names a view outside the layout.",
        )
        for value in profiles[stratum].values():
            _require(
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value),
                "Carrier margins must be two non-negative integers.",
            )

    view_index = {view_id: index for index, view_id in enumerate(ordered_views)}
    indexed_profiles = []
    for stratum in strata:
        carrier_mask = 0
        clear_by_index = {}
        occluded_by_index = {}
        for view_id, value in profiles[stratum].items():
            index = view_index[view_id]
            carrier_mask |= 1 << index
            clear_by_index[index] = int(value[0])
            occluded_by_index[index] = int(value[1])
        indexed_profiles.append(
            (
                stratum,
                carrier_mask,
                _threshold_masks(clear_by_index),
                _threshold_masks(occluded_by_index),
            )
        )
    angle_index = _angle_mask_index(ordered_views, angle_table)

    expected = math.comb(len(ordered_views), selected_count)
    audited = 0
    qualifying = 0
    best_mask = None
    best_score = None
    for selected_mask in _fixed_popcount_masks(len(ordered_views), selected_count):
        audited += 1
        carrier_counts = []
        valid = True
        for _, carrier_mask, _, _ in indexed_profiles:
            count = (selected_mask & carrier_mask).bit_count()
            if count < min_carriers:
                valid = False
                break
            carrier_counts.append(count)
        if not valid:
            continue
        qualifying += 1

        worst_clear = min(
            _third_best_from_levels(selected_mask, clear_levels)
            for _, _, clear_levels, _ in indexed_profiles
        )
        if best_score is not None and worst_clear < best_score[0]:
            continue
        worst_occluded = min(
            _third_best_from_levels(selected_mask, occluded_levels)
            for _, _, _, occluded_levels in indexed_profiles
        )
        score_prefix = (
            worst_clear,
            worst_occluded,
            min(carrier_counts),
            sum(carrier_counts),
        )
        if best_score is not None and score_prefix < best_score[:4]:
            continue
        score = (
            *score_prefix,
            _minimum_angle_mask(selected_mask, angle_index),
        )
        if (
            best_score is None
            or score > best_score
            or (
                score == best_score
                and _lexicographically_less_mask(selected_mask, best_mask)
            )
        ):
            best_score = score
            best_mask = selected_mask
    _require(audited == expected, "The exhaustive combination audit was truncated.")

    best_ids = (
        tuple(
            view_id
            for index, view_id in enumerate(ordered_views)
            if best_mask & (1 << index)
        )
        if best_mask is not None
        else None
    )
    best_detail = None
    if best_ids is not None:
        third_clear = {}
        third_occluded = {}
        carrier_counts = {}
        for stratum in strata:
            values = [profiles[stratum][view_id] for view_id in best_ids if view_id in profiles[stratum]]
            third_clear[stratum] = _third_best([value[0] for value in values])
            third_occluded[stratum] = _third_best([value[1] for value in values])
            carrier_counts[stratum] = len(values)
        best_detail = {
            "third_best_clear_flank_slack_by_stratum": {
                str(stratum): value for stratum, value in third_clear.items()
            },
            "third_best_occluded_run_slack_by_stratum": {
                str(stratum): value for stratum, value in third_occluded.items()
            },
            "robust_carrier_count_by_stratum": {
                str(stratum): value for stratum, value in carrier_counts.items()
            },
        }
    return {
        "subsets_audited": audited,
        "subsets_expected": expected,
        "qualifying_subsets": qualifying,
        "selected_view_ids": list(best_ids) if best_ids is not None else None,
        "score": list(best_score) if best_score is not None else None,
        "detail": best_detail,
    }


def select(repeats: Mapping[str, Mapping]) -> dict:
    """Select 12 event-band candidates in memory; never read or write a file."""

    robust = robust_carrier_profiles(repeats)
    search = _search_profiles(
        robust["profiles"],
        PROBE_VIEW_IDS,
        pairwise_angle_table(PROBE_LAYOUT),
        SELECTED_COUNT,
        MIN_ROBUST_CARRIERS_PER_STRATUM,
    )
    passed = search["selected_view_ids"] is not None
    return {
        "record": RECORD,
        "pass": passed,
        "failure": (
            None
            if passed
            else "no_12_view_subset_has_three_robust_carriers_for_every_stratum"
        ),
        "candidate_count": len(PROBE_VIEW_IDS),
        "selected_count": SELECTED_COUNT,
        "repeat_count": REPEAT_COUNT,
        "minimum_robust_carriers_per_stratum": MIN_ROBUST_CARRIERS_PER_STRATUM,
        "event_thresholds": {
            "clear_frames": MIN_CLEAR_FRAMES,
            "occluded_frames": MIN_OCCLUDED_FRAMES,
        },
        "radius_m": RADIUS_M,
        "repeat_ids": robust["repeat_ids"],
        "robust_carriers_by_stratum": robust["robust_carriers_by_stratum"],
        "subsets_audited": search["subsets_audited"],
        "subsets_expected": search["subsets_expected"],
        "qualifying_subsets": search["qualifying_subsets"],
        "selected_view_ids": search["selected_view_ids"],
        "score": search["score"],
        "score_order": [
            "max_worst_stratum_third_best_clear_flank_slack",
            "max_worst_stratum_third_best_occluded_run_slack",
            "max_min_robust_carrier_count",
            "max_total_robust_carrier_incidence",
            "max_min_pairwise_angle_nanodeg",
            "view_id_lexicographic_ascending",
        ],
        "detail": search["detail"],
        "writes_output": False,
        "imports_old_selector": False,
    }
