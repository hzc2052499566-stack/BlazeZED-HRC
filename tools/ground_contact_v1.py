"""Pure-Python ground-contact gates for the cross-character industrial scene.

AGENTS.md 6.6.2 requires that "the character stands on the floor" is decided
geometrically rather than from a screenshot: the support sole must sit on a
frozen floor plane, nothing may sink through it, the support foot may not slide,
the root height may not jump, and at least one foot must support the body in
every frame.

The existing motion contract in ``fs_cts5_full_body_motion_v2`` validates only
*relative* geometry (support drift, swing clearance, knee angles, bone
rigidity).  It has no floor at all, which is exactly the gap this module fills.

No Isaac Sim or ``pxr`` dependency: the Isaac probe measures per-frame sole
points and calls in here, and the unit tests exercise every gate in a normal
interpreter.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


CONTACT_TAG = "fs_cts5_industrial_ground_contact_v1"
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False
FRAME_COUNT = 241

# Draft engineering thresholds, in operational metres, recorded before any
# character is probed.  They gate scene authoring only; they are not the formal
# comparison thresholds of AGENTS.md 6.6.6.
MAX_PENETRATION_M = 0.005
MAX_SUPPORT_GAP_M = 0.010
MAX_SUPPORT_SLIDE_PER_FRAME_M = 0.002
MAX_SUPPORT_SLIDE_TOTAL_M = 0.010
MAX_ROOT_HEIGHT_STEP_M = 0.020
MAX_ROOT_HEIGHT_RANGE_M = 0.120
MAX_SUPPORT_SWITCHES = 8
MIN_SUPPORTED_FRACTION = 1.0
# Each foot must actually leave the floor at some point.  Without this a motion
# whose swing foot only ever dips *below* the stance plane -- exactly what the
# F01 attempt_01 probe found -- satisfies every other gate about support.
# Kept just under the existing motion contract's lower bound (2% of a ~0.83 m
# leg is ~0.0166 m) so the two contracts cannot contradict each other: this gate
# asks only that the foot demonstrably leaves the ground.
MIN_PEAK_SWING_CLEARANCE_M = 0.015

SIDES = ("left", "right")
REQUIRED_FRAME_FIELDS = ("frame", "root_height_m", "left_sole", "right_sole")
REQUIRED_SOLE_FIELDS = ("lowest_point_height_m", "contact_centroid_xy")

# Known approximations, recorded so the report never overstates what was tested.
APPROXIMATIONS = (
    "sole points are carried rigidly by the ankle joint, so toe flexion is not modelled",
    "penetration is evaluated for the feet only, not for the whole body surface",
    "the floor is a single frozen horizontal plane, not per-tile scene geometry",
)


class GroundContactError(RuntimeError):
    """Raised when a probe series is unusable rather than merely failing gates."""


def _finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise GroundContactError("Non-finite value in the probe series.")
    return number


def _check_frame(entry: Mapping) -> None:
    for field in REQUIRED_FRAME_FIELDS:
        if field not in entry:
            raise GroundContactError("Probe frame is missing {!r}.".format(field))
    for side in SIDES:
        sole = entry["{}_sole".format(side)]
        for field in REQUIRED_SOLE_FIELDS:
            if field not in sole:
                raise GroundContactError(
                    "Probe frame {} is missing {}_sole.{}".format(
                        entry.get("frame"), side, field
                    )
                )
        if len(sole["contact_centroid_xy"]) != 2:
            raise GroundContactError("contact_centroid_xy must hold two values.")


def support_side(entry: Mapping, floor_height_m: float) -> str:
    """The sole closest to the floor plane is the supporting one.

    Distance to the plane, not lowest height: a foot that drives *through* the
    floor is the failure being measured, not the foot carrying the body, and
    picking the lowest sole would hand the whole slide/stance analysis to the
    penetrating swing foot.  Ties resolve to the left.
    """
    left = abs(_finite(entry["left_sole"]["lowest_point_height_m"]) - floor_height_m)
    right = abs(_finite(entry["right_sole"]["lowest_point_height_m"]) - floor_height_m)
    return "left" if left <= right else "right"


def evaluate_ground_contact(
    frames: Sequence[Mapping],
    floor_height_m: float,
    expected_frame_count: int = FRAME_COUNT,
) -> dict:
    """Evaluate one character's motion against the frozen floor plane."""
    if len(frames) != expected_frame_count:
        raise GroundContactError(
            "Expected {} probe frames, received {}.".format(
                expected_frame_count, len(frames)
            )
        )
    floor = _finite(floor_height_m)
    for index, entry in enumerate(frames):
        _check_frame(entry)
        if int(entry["frame"]) != index:
            raise GroundContactError(
                "Probe frames must be ordered 0..{}; index {} carries frame {}.".format(
                    expected_frame_count - 1, index, entry["frame"]
                )
            )

    failures = []
    worst_penetration_m = 0.0
    worst_penetration_frame = None
    worst_support_gap_m = 0.0
    worst_support_gap_frame = None
    unsupported_frames = []
    supports = []
    root_heights = []

    for entry in frames:
        frame = int(entry["frame"])
        root_heights.append(_finite(entry["root_height_m"]))
        heights = {
            side: _finite(entry["{}_sole".format(side)]["lowest_point_height_m"])
            for side in SIDES
        }
        for side in SIDES:
            penetration = floor - heights[side]
            if penetration > worst_penetration_m:
                worst_penetration_m = penetration
                worst_penetration_frame = frame
        support = support_side(entry, floor)
        supports.append(support)
        gap = abs(heights[support] - floor)
        if gap > worst_support_gap_m:
            worst_support_gap_m = gap
            worst_support_gap_frame = frame
        if gap > MAX_SUPPORT_GAP_M:
            unsupported_frames.append(frame)

    if worst_penetration_m > MAX_PENETRATION_M:
        failures.append(
            "sole sinks {:.4f} m below the floor at frame {} (limit {:.4f})".format(
                worst_penetration_m, worst_penetration_frame, MAX_PENETRATION_M
            )
        )
    supported_fraction = 1.0 - (len(unsupported_frames) / float(len(frames)))
    if supported_fraction < MIN_SUPPORTED_FRACTION:
        failures.append(
            "{} frame(s) have no foot on the floor, first at {} (worst gap "
            "{:.4f} m at frame {}, limit {:.4f})".format(
                len(unsupported_frames),
                unsupported_frames[0],
                worst_support_gap_m,
                worst_support_gap_frame,
                MAX_SUPPORT_GAP_M,
            )
        )

    peak_clearance = {
        side: max(
            _finite(entry["{}_sole".format(side)]["lowest_point_height_m"]) - floor
            for entry in frames
        )
        for side in SIDES
    }
    for side in SIDES:
        if peak_clearance[side] < MIN_PEAK_SWING_CLEARANCE_M:
            failures.append(
                "{} foot never rises more than {:.4f} m above the floor (needs "
                "{:.4f}); it is not swinging, it is being pushed into the "
                "ground".format(side, peak_clearance[side], MIN_PEAK_SWING_CLEARANCE_M)
            )

    switches = [
        index for index in range(1, len(supports)) if supports[index] != supports[index - 1]
    ]
    if len(switches) > MAX_SUPPORT_SWITCHES:
        failures.append(
            "support foot changes {} times (limit {})".format(
                len(switches), MAX_SUPPORT_SWITCHES
            )
        )

    worst_slide_step_m = 0.0
    worst_slide_step_frame = None
    worst_slide_total_m = 0.0
    run_start = 0
    for index in range(1, len(frames) + 1):
        ended = index == len(frames) or supports[index] != supports[run_start]
        if not ended:
            continue
        side = supports[run_start]
        key = "{}_sole".format(side)
        anchor = frames[run_start][key]["contact_centroid_xy"]
        previous = anchor
        for cursor in range(run_start + 1, index):
            current = frames[cursor][key]["contact_centroid_xy"]
            step = math.hypot(
                _finite(current[0]) - _finite(previous[0]),
                _finite(current[1]) - _finite(previous[1]),
            )
            if step > worst_slide_step_m:
                worst_slide_step_m = step
                worst_slide_step_frame = int(frames[cursor]["frame"])
            total = math.hypot(
                _finite(current[0]) - _finite(anchor[0]),
                _finite(current[1]) - _finite(anchor[1]),
            )
            worst_slide_total_m = max(worst_slide_total_m, total)
            previous = current
        run_start = index

    if worst_slide_step_m > MAX_SUPPORT_SLIDE_PER_FRAME_M:
        failures.append(
            "support foot slides {:.4f} m in one frame at {} (limit {:.4f})".format(
                worst_slide_step_m, worst_slide_step_frame, MAX_SUPPORT_SLIDE_PER_FRAME_M
            )
        )
    if worst_slide_total_m > MAX_SUPPORT_SLIDE_TOTAL_M:
        failures.append(
            "support foot drifts {:.4f} m within one stance (limit {:.4f})".format(
                worst_slide_total_m, MAX_SUPPORT_SLIDE_TOTAL_M
            )
        )

    root_steps = [
        abs(root_heights[index] - root_heights[index - 1])
        for index in range(1, len(root_heights))
    ]
    worst_root_step_m = max(root_steps) if root_steps else 0.0
    root_range_m = max(root_heights) - min(root_heights)
    if worst_root_step_m > MAX_ROOT_HEIGHT_STEP_M:
        failures.append(
            "root height jumps {:.4f} m between frames (limit {:.4f})".format(
                worst_root_step_m, MAX_ROOT_HEIGHT_STEP_M
            )
        )
    if root_range_m > MAX_ROOT_HEIGHT_RANGE_M:
        failures.append(
            "root height spans {:.4f} m (limit {:.4f})".format(
                root_range_m, MAX_ROOT_HEIGHT_RANGE_M
            )
        )

    return {
        "contact_tag": CONTACT_TAG,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "frame_count": len(frames),
        "floor_height_operational_m": floor,
        "worst_penetration_m": worst_penetration_m,
        "worst_penetration_frame": worst_penetration_frame,
        "worst_support_gap_m": worst_support_gap_m,
        "worst_support_gap_frame": worst_support_gap_frame,
        "unsupported_frame_count": len(unsupported_frames),
        "unsupported_frames": unsupported_frames[:20],
        "peak_swing_clearance_m": peak_clearance,
        "support_switch_count": len(switches),
        "support_switch_frames": switches,
        "worst_support_slide_step_m": worst_slide_step_m,
        "worst_support_slide_step_frame": worst_slide_step_frame,
        "worst_support_slide_total_m": worst_slide_total_m,
        "worst_root_height_step_m": worst_root_step_m,
        "root_height_range_m": root_range_m,
        "approximations": list(APPROXIMATIONS),
        "failures": failures,
        "pass": not failures,
    }


def infer_floor_height(frames: Sequence[Mapping]) -> float:
    """Lowest sole height across the series, for reporting a *candidate* floor.

    Never use this as the frozen floor of a scene that already has one: it is a
    measurement of where the character happens to be, not of where the floor is.
    """
    if not frames:
        raise GroundContactError("No probe frames to infer a floor from.")
    return min(
        _finite(entry["{}_sole".format(side)]["lowest_point_height_m"])
        for entry in frames
        for side in SIDES
    )
