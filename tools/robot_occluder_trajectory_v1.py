"""Frozen UR10e occluder trajectory maths (pure Python).

AGENTS.md 6.6.3 fixes the robot's role precisely: a deterministic known-geometry
occluder that enters in front of the target arm along a frozen trajectory, holds,
and retracts.  It is not a collision-avoidance, separation-distance or robot
control experiment, and nothing here should ever be reported as one.

The schedule is derived from the motion's own reach control rather than picked
by hand: the arm is only worth occluding while it is extended.  No Isaac Sim or
``pxr`` dependency.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from cross_character_rig_preflight_v1 import BONES
from fs_cts5_full_body_motion_v2 import FRAME_COUNT, controls_for_frame, smoothstep


ROBOT_TAG = "fs_cts5_ur10e_occluder_v1"
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False

# The target arm of the FS-CTS5 dynamic work is the right one; the occluder is
# aimed at its forearm segment.
TARGET_SEGMENT = ("right_elbow", "right_wrist")

# Draft engineering values, recorded before the robot is placed.
REACH_HOLD_THRESHOLD = 0.9
ENTER_FRAMES = 30
RETRACT_FRAMES = 30
# The reach control stays near peak from frame 53 to 215, so occluding the whole
# reach would hide the arm for 68% of the motion and leave one clear frame at the
# end.  AGENTS.md 6.6.6 gates recovery at <=2 frames *after* occlusion, which is
# unmeasurable without a clear tail, so the hold is capped and centred instead.
MAX_HOLD_FRAMES = 60
MIN_CLEAR_TAIL_FRAMES = 20
MIN_ROBOT_CHARACTER_CLEARANCE_M = 0.10
# Hold and idle are judged by line of sight, the same criterion the placement
# search uses and the occlusion pilot measures.  Horizontal distance was a proxy
# that failed both ways: it passed a placement the pilot measured at 0% arm
# occlusion, and then rejected a parked pose that blocks nothing at all because
# a 0.06 m high arm happened to lie under the sight line.
MIN_HOLD_BLOCKED_VIEWS = 3
MAX_IDLE_BLOCKED_VIEWS = 0
# Horizontal, not 3D.  Occlusion is a line-of-sight property and the camera bank
# does not exist until AGENTS.md 6.6.7 step 4, so this gate only asserts that the
# occluder is in the right place at the right time; whether it actually hides the
# arm from a given view is decided there, against real cameras.
MAX_HOLD_DISTANCE_TO_ARM_M = 0.35
MIN_RETRACTED_DISTANCE_M = 0.60

PHASES = ("idle", "enter", "hold", "retract")


class TrajectoryError(RuntimeError):
    """Raised when the schedule or the measured geometry is unusable."""


def reach_hold_window(threshold: float = REACH_HOLD_THRESHOLD) -> dict:
    """Frames where the character's reach is near its peak.

    Derived from the frozen motion contract, so the occluder cannot drift out of
    step with the arm it is supposed to hide.
    """
    values = [float(controls_for_frame(frame)["reach"]) for frame in range(FRAME_COUNT)]
    peak = max(values)
    if peak <= 0.0:
        raise TrajectoryError("The motion has no reach phase to occlude.")
    frames = [
        frame for frame, value in enumerate(values) if value >= threshold * peak
    ]
    return {
        "hold_start": min(frames),
        "hold_end": max(frames),
        "peak_reach": peak,
        "threshold": float(threshold),
    }


def build_schedule(
    enter_frames: int = ENTER_FRAMES, retract_frames: int = RETRACT_FRAMES
) -> dict:
    """Enter / hold / retract windows clipped to the 241-frame motion."""
    window = reach_hold_window()
    centre = (int(window["hold_start"]) + int(window["hold_end"])) // 2
    half_hold = int(MAX_HOLD_FRAMES) // 2
    hold_start = max(int(window["hold_start"]), centre - half_hold)
    hold_end = min(int(window["hold_end"]), centre + half_hold)
    enter_start = hold_start - int(enter_frames)
    retract_end = hold_end + int(retract_frames)
    if enter_start < 0:
        raise TrajectoryError(
            "The occluder cannot enter before frame 0; shorten ENTER_FRAMES."
        )
    if retract_end + MIN_CLEAR_TAIL_FRAMES > FRAME_COUNT - 1:
        raise TrajectoryError(
            "Retraction ends at frame {} leaving fewer than {} clear frames; "
            "recovery would be unmeasurable.".format(retract_end, MIN_CLEAR_TAIL_FRAMES)
        )
    if enter_start >= hold_start or retract_end <= hold_end:
        raise TrajectoryError("The schedule has no room to enter and retract.")
    return {
        "enter_start": enter_start,
        "hold_start": hold_start,
        "hold_end": hold_end,
        "retract_end": retract_end,
        "clear_tail_frames": FRAME_COUNT - 1 - retract_end,
        "reach_window": window,
    }


def phase_for_frame(frame: int, schedule: Mapping[str, int]) -> str:
    frame = int(frame)
    if frame < schedule["enter_start"] or frame > schedule["retract_end"]:
        return "idle"
    if frame < schedule["hold_start"]:
        return "enter"
    if frame <= schedule["hold_end"]:
        return "hold"
    return "retract"


def engagement_for_frame(frame: int, schedule: Mapping[str, int]) -> float:
    """0 while parked, 1 while holding, smoothly ramped in between."""
    phase = phase_for_frame(frame, schedule)
    if phase == "idle":
        return 0.0
    if phase == "hold":
        return 1.0
    if phase == "enter":
        span = schedule["hold_start"] - schedule["enter_start"]
        return smoothstep((int(frame) - schedule["enter_start"]) / float(span))
    span = schedule["retract_end"] - schedule["hold_end"]
    return smoothstep(1.0 - (int(frame) - schedule["hold_end"]) / float(span))


def joint_angles_for_frame(
    frame: int,
    schedule: Mapping[str, int],
    parked_angles: Mapping[str, float],
    engaged_angles: Mapping[str, float],
) -> dict:
    """Interpolate every joint between its parked and engaged angle."""
    if set(parked_angles) != set(engaged_angles):
        raise TrajectoryError("Parked and engaged poses must cover the same joints.")
    blend = engagement_for_frame(frame, schedule)
    return {
        joint: float(parked_angles[joint])
        + blend * (float(engaged_angles[joint]) - float(parked_angles[joint]))
        for joint in parked_angles
    }


def point_to_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    span = [float(end[axis]) - float(start[axis]) for axis in range(3)]
    length_squared = sum(value * value for value in span)
    if length_squared <= 1.0e-12:
        return math.sqrt(
            sum((float(point[axis]) - float(start[axis])) ** 2 for axis in range(3))
        )
    projection = sum(
        (float(point[axis]) - float(start[axis])) * span[axis] for axis in range(3)
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = [float(start[axis]) + projection * span[axis] for axis in range(3)]
    return math.sqrt(
        sum((float(point[axis]) - closest[axis]) ** 2 for axis in range(3))
    )


# A UR10e link is roughly this thick, so a point within this distance of the
# sight line stands in front of what the camera is looking at.
LINK_RADIUS_M = 0.075


def blocks_line_of_sight(
    points: Sequence[Sequence[float]],
    eye: Sequence[float],
    target: Sequence[float],
    radius_m: float = LINK_RADIUS_M,
) -> dict:
    """Does any robot point sit on the segment from the camera to the target?

    This is the criterion the occlusion pilot actually measures.  The earlier
    distance-to-arm proxy passed while the pilot measured 0% arm occlusion,
    because being beside the arm is not the same as being in front of it.
    """
    span = [float(target[axis]) - float(eye[axis]) for axis in range(3)]
    length_squared = sum(value * value for value in span)
    if length_squared <= 1.0e-12:
        raise TrajectoryError("Camera and target coincide.")
    closest = None
    for point in points:
        offset = [float(point[axis]) - float(eye[axis]) for axis in range(3)]
        projection = sum(
            offset[axis] * span[axis] for axis in range(3)
        ) / length_squared
        if projection <= 0.0 or projection >= 1.0:
            continue  # behind the camera, or past the target
        perpendicular = math.sqrt(
            sum(
                (offset[axis] - projection * span[axis]) ** 2 for axis in range(3)
            )
        )
        if closest is None or perpendicular < closest:
            closest = perpendicular
    return {
        "blocks": bool(closest is not None and closest <= float(radius_m)),
        "closest_perpendicular_m": closest,
    }


def blocked_endpoint_views(
    points: Sequence[Sequence[float]],
    viewpoints: Sequence[Mapping],
    positions: Mapping[str, Sequence[float]],
    endpoints: Sequence[str] = TARGET_SEGMENT,
    radius_m: float = LINK_RADIUS_M,
) -> dict:
    """Count (view, endpoint) pairs the robot stands in front of.

    Endpoints, not the bone's midpoint.  FS-CTS5 measures endpoints, so hiding
    the middle of a forearm removes nothing: the pose diagnosis showed the arm
    exactly where the search wanted it, blocking the midpoint from six views,
    while the render measured zero endpoint occlusion.
    """
    blocked = 0
    per_view = []
    for viewpoint in viewpoints:
        hits = [
            endpoint
            for endpoint in endpoints
            if endpoint in positions
            and blocks_line_of_sight(
                points, viewpoint["eye"], positions[endpoint], radius_m
            )["blocks"]
        ]
        blocked += len(hits)
        per_view.append(len(hits))
    return {
        "blocked_pairs": blocked,
        "views_with_any_blocked": sum(1 for count in per_view if count),
        "per_view_blocked": per_view,
    }


def _flatten(point: Sequence[float], up_axis_index: int) -> list:
    return [
        0.0 if axis == up_axis_index else float(point[axis]) for axis in range(3)
    ]


def evaluate_occlusion_series(
    robot_points_by_frame: Sequence[Sequence[Sequence[float]]],
    character_positions_by_frame: Sequence[Mapping[str, Sequence[float]]],
    schedule: Mapping[str, int],
    up_axis_index: int = 2,
    viewpoints: Sequence[Mapping] = None,
) -> dict:
    """Check the occluder gets close to the arm, and never touches the person."""
    if len(robot_points_by_frame) != FRAME_COUNT:
        raise TrajectoryError(
            "Expected {} frames of robot geometry.".format(FRAME_COUNT)
        )
    if len(character_positions_by_frame) != FRAME_COUNT:
        raise TrajectoryError(
            "Expected {} frames of character geometry.".format(FRAME_COUNT)
        )

    worst_clearance = None
    worst_clearance_frame = None
    hold_distances = []
    idle_distances = []
    hold_blocked = []
    idle_blocked = []
    vertical_gaps = []
    for frame in range(FRAME_COUNT):
        points = robot_points_by_frame[frame]
        positions = character_positions_by_frame[frame]
        if not points:
            raise TrajectoryError("Frame {} has no robot geometry.".format(frame))
        present_segments = [
            (start, end)
            for start, end in BONES.values()
            if start in positions and end in positions
        ]
        for joint in TARGET_SEGMENT:
            if joint not in positions:
                raise TrajectoryError(
                    "Frame {} is missing {}.".format(frame, joint)
                )
        # Distance to the limb *segments*, not just to joint pivots: a link can
        # pass through the middle of a forearm while sitting far from both its
        # endpoints.
        clearance = min(
            min(
                min(
                    point_to_segment_distance(
                        point, positions[start], positions[end]
                    )
                    for start, end in present_segments
                )
                if present_segments
                else float("inf"),
                min(
                    math.sqrt(
                        sum(
                            (float(point[axis]) - float(position[axis])) ** 2
                            for axis in range(3)
                        )
                    )
                    for position in positions.values()
                ),
            )
            for point in points
        )
        if worst_clearance is None or clearance < worst_clearance:
            worst_clearance = clearance
            worst_clearance_frame = frame

        flat_start = _flatten(positions[TARGET_SEGMENT[0]], up_axis_index)
        flat_end = _flatten(positions[TARGET_SEGMENT[1]], up_axis_index)
        arm_distance = min(
            point_to_segment_distance(
                _flatten(point, up_axis_index), flat_start, flat_end
            )
            for point in points
        )
        vertical_gaps.append(
            min(
                abs(
                    float(point[up_axis_index])
                    - (
                        float(positions[TARGET_SEGMENT[0]][up_axis_index])
                        + float(positions[TARGET_SEGMENT[1]][up_axis_index])
                    )
                    / 2.0
                )
                for point in points
            )
        )
        phase = phase_for_frame(frame, schedule)
        if phase == "hold":
            hold_distances.append(arm_distance)
        elif phase == "idle":
            idle_distances.append(arm_distance)

        if viewpoints:
            blocked = blocked_endpoint_views(points, viewpoints, positions)[
                "views_with_any_blocked"
            ]
            if phase == "hold":
                hold_blocked.append(blocked)
            elif phase == "idle":
                idle_blocked.append(blocked)

    failures = []
    if worst_clearance < MIN_ROBOT_CHARACTER_CLEARANCE_M:
        failures.append(
            "robot comes within {:.4f} m of the character at frame {} (needs "
            "{:.4f})".format(
                worst_clearance, worst_clearance_frame, MIN_ROBOT_CHARACTER_CLEARANCE_M
            )
        )
    if not hold_distances:
        failures.append("the schedule has no hold phase")
    if viewpoints:
        if hold_blocked and min(hold_blocked) < MIN_HOLD_BLOCKED_VIEWS:
            failures.append(
                "the occluder blocks only {} of {} candidate views at its weakest "
                "hold frame (needs {})".format(
                    min(hold_blocked), len(viewpoints), MIN_HOLD_BLOCKED_VIEWS
                )
            )
        if idle_blocked and max(idle_blocked) > MAX_IDLE_BLOCKED_VIEWS:
            failures.append(
                "the parked robot still blocks {} candidate views (allowed {}); "
                "it would occlude outside its window".format(
                    max(idle_blocked), MAX_IDLE_BLOCKED_VIEWS
                )
            )
    else:
        # Degraded path: no viewpoints were supplied, so the older distance
        # proxy still applies in full.  It is weaker, not laxer.
        if hold_distances and min(hold_distances) > MAX_HOLD_DISTANCE_TO_ARM_M:
            failures.append(
                "the occluder never gets closer than {:.4f} m to the target arm "
                "while holding (needs {:.4f})".format(
                    min(hold_distances), MAX_HOLD_DISTANCE_TO_ARM_M
                )
            )
        if idle_distances and min(idle_distances) < MIN_RETRACTED_DISTANCE_M:
            failures.append(
                "the parked robot stays within {:.4f} m of the target arm (needs "
                "{:.4f}); it would occlude outside its window".format(
                    min(idle_distances), MIN_RETRACTED_DISTANCE_M
                )
            )

    return {
        "robot_tag": ROBOT_TAG,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "schedule": dict(schedule),
        "target_segment": list(TARGET_SEGMENT),
        "worst_character_clearance_m": worst_clearance,
        "worst_character_clearance_frame": worst_clearance_frame,
        "closest_hold_distance_to_arm_m": min(hold_distances) if hold_distances else None,
        "closest_idle_distance_to_arm_m": min(idle_distances) if idle_distances else None,
        "criterion": "line_of_sight" if viewpoints else "horizontal_distance",
        "candidate_view_count": len(viewpoints) if viewpoints else 0,
        "min_hold_blocked_views": min(hold_blocked) if hold_blocked else None,
        "max_idle_blocked_views": max(idle_blocked) if idle_blocked else None,
        "distance_measured": "horizontal_only",
        "min_vertical_gap_to_arm_m": min(vertical_gaps) if vertical_gaps else None,
        "failures": failures,
        "pass": not failures,
        "note": (
            "Horizontal placement only: the occluder is shown to be beside the "
            "target arm during its window, not shown to hide it. Line-of-sight "
            "occlusion is verified against the camera bank in step 4. This is "
            "not collision avoidance, not a separation-distance study, and not a "
            "robot control result."
        ),
    }
