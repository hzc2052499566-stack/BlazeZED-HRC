"""Deterministic layout maths for the industrial workcell (pure Python).

No Isaac Sim or ``pxr`` dependency: the Isaac authoring script measures the
scene, asks this module where things go, and the unit tests exercise the rules
in a normal interpreter.

Two decisions are encoded here and both are constructions, not measurements, so
they are named as such wherever they surface:

* the character is dropped onto the floor, rather than the floor being raised to
  meet the character.  The floor is the scene's own Z=0 plane, chosen
  independently of any character; frame-0 contact then holds by construction,
  and the remaining 240 frames are a genuine test of it.
* the workbench sits in front of the character's facing direction at a working
  distance derived from its own measured depth, not from a hand-tuned number.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


LAYOUT_TAG = "fs_cts5_industrial_workcell_layout_v1"

# Chosen from measured candidates, not from a catalogue name.  PackingTable
# (1.083 m) and both Mounts tables (1.755 m) sit above a ~0.85 m hip and would
# hide the torso; thor_table is at working height.
CHOSEN_WORKBENCH = {
    "asset_suffix": "Isaac/Props/Mounts/thor_table.usd",
    "measured_extent_m": (0.9103, 0.7679, 0.7950),
    "rejected": {
        "Isaac/Props/PackingTable/packing_table.usd": "1.083 m top, above the hip",
        "Isaac/Props/Mounts/table.usd": "1.755 m top, far above the hip",
        "Isaac/Props/Mounts/SeattleLabTable/table.usd": "1.755 m top, far above the hip",
    },
}
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False

# Draft engineering values, recorded before anything is placed.
FLOOR_HEIGHT_OPERATIONAL_M = 0.0
WORKBENCH_FRONT_CLEARANCE_M = 0.12
MIN_CHARACTER_WORKBENCH_CLEARANCE_M = 0.05
MAX_WORKBENCH_TOP_ABOVE_HIP_M = 0.10
MIN_WORKBENCH_TOP_ABOVE_KNEE_M = 0.05


class LayoutError(RuntimeError):
    """Raised when the measured scene cannot support a sane layout."""


def _sub(left, right):
    return tuple(float(left[axis]) - float(right[axis]) for axis in range(3))


def _norm(vector):
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _unit(vector):
    length = _norm(vector)
    if length <= 1.0e-9:
        raise LayoutError("Cannot normalise a zero-length direction.")
    return tuple(float(value) / length for value in vector)


def _cross(left, right):
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def character_drop(lowest_sole_height_m: float, floor_height_m: float = FLOOR_HEIGHT_OPERATIONAL_M) -> float:
    """Vertical offset that lands the soles on the floor.

    Positive result means the character currently floats and must move down.
    """
    return float(lowest_sole_height_m) - float(floor_height_m)


def facing_direction(positions: Mapping[str, Sequence[float]], up_axis_index: int):
    """Horizontal facing direction derived from the rig, not from a guess.

    Up crossed with the hip-to-hip axis gives the forward normal; the sign is
    resolved so that the nose leads the pelvis.
    """
    for joint in ("left_hip", "right_hip", "pelvis", "nose"):
        if joint not in positions:
            raise LayoutError("Facing needs {} in the rest pose.".format(joint))
    up = tuple(1.0 if axis == up_axis_index else 0.0 for axis in range(3))
    lateral = _unit(_sub(positions["left_hip"], positions["right_hip"]))
    forward = _unit(_cross(lateral, up))
    nose_lead = sum(
        forward[axis] * (float(positions["nose"][axis]) - float(positions["pelvis"][axis]))
        for axis in range(3)
        if axis != up_axis_index
    )
    if nose_lead < 0.0:
        forward = tuple(-value for value in forward)
    return forward


def forward_excursion_below(
    positions_by_frame: Sequence[Mapping[str, Sequence[float]]],
    origin: Sequence[float],
    forward: Sequence[float],
    up_axis_index: int,
    height_limit_m: float,
) -> dict:
    """Furthest any joint reaches forward while low enough to hit a prop.

    Joints above ``height_limit_m`` pass over the prop rather than into it, so
    they are excluded.  This is what a fixed front clearance got wrong: a
    standing pose says nothing about where the knee swings during the march.
    """
    if not positions_by_frame:
        raise LayoutError("No frames to measure forward excursion from.")
    worst = None
    worst_frame = None
    worst_joint = None
    for frame, positions in enumerate(positions_by_frame):
        for joint, position in positions.items():
            if float(position[up_axis_index]) > float(height_limit_m):
                continue
            reach = sum(
                float(forward[axis]) * (float(position[axis]) - float(origin[axis]))
                for axis in range(3)
                if axis != up_axis_index
            )
            if worst is None or reach > worst:
                worst = reach
                worst_frame = frame
                worst_joint = joint
    if worst is None:
        raise LayoutError(
            "No joint stays below {:.4f}; nothing could collide with the prop.".format(
                height_limit_m
            )
        )
    return {
        "max_forward_excursion_m": worst,
        "frame": worst_frame,
        "joint": worst_joint,
        "height_limit_m": float(height_limit_m),
    }


def required_front_clearance(
    max_forward_excursion_m: float,
    margin_m: float = MIN_CHARACTER_WORKBENCH_CLEARANCE_M,
) -> float:
    """Near-edge distance that keeps the prop clear of the whole motion."""
    return max(
        WORKBENCH_FRONT_CLEARANCE_M, float(max_forward_excursion_m) + float(margin_m)
    )


MIN_FOOT_ASYMMETRY_M = 0.02


def facing_from_foot_geometry(
    ankle_positions: Sequence[Sequence[float]],
    sole_centroids: Sequence[Sequence[float]],
    up_axis_index: int,
) -> dict:
    """Facing direction from the feet: toes reach much further than heels.

    The nose-based test got this backwards.  ``FacialBone`` is a head-centre
    pivot whose horizontal offset from the pelvis is small and rig-dependent, so
    its sign said the character faced +Y when the feet say -Y, and the workbench
    was authored behind them.  Foot asymmetry is physical, not a rig convention.
    """
    if len(ankle_positions) != len(sole_centroids) or not ankle_positions:
        raise LayoutError("Need one sole centroid per ankle.")
    offsets = []
    for ankle, centroid in zip(ankle_positions, sole_centroids):
        offsets.append(
            tuple(
                0.0 if axis == up_axis_index else float(centroid[axis]) - float(ankle[axis])
                for axis in range(3)
            )
        )
    mean = tuple(
        sum(offset[axis] for offset in offsets) / len(offsets) for axis in range(3)
    )
    magnitude = _norm(mean)
    if magnitude < MIN_FOOT_ASYMMETRY_M:
        raise LayoutError(
            "Feet are too symmetric to read a facing direction ({:.4f} m < "
            "{:.4f}).".format(magnitude, MIN_FOOT_ASYMMETRY_M)
        )
    return {
        "forward": _unit(mean),
        "toe_offset_m": magnitude,
        "per_foot_offsets": offsets,
    }


def facing_disagreement_deg(first: Sequence[float], second: Sequence[float]) -> float:
    """Angle between two facing estimates, for recording the cross-check."""
    left = _unit(first)
    right = _unit(second)
    dot = max(-1.0, min(1.0, sum(left[axis] * right[axis] for axis in range(3))))
    return math.degrees(math.acos(dot))


def workbench_placement(
    positions: Mapping[str, Sequence[float]],
    workbench_extent: Sequence[float],
    up_axis_index: int,
    floor_height_m: float = FLOOR_HEIGHT_OPERATIONAL_M,
    front_clearance_m: float = WORKBENCH_FRONT_CLEARANCE_M,
    facing_override: Sequence[float] = None,
) -> dict:
    """Place the workbench in front of the character, standing on the floor.

    ``positions`` are the character's grounded rest-pose joints; the extent is
    the workbench's own measured bounding box.
    """
    horizontal_axes = [axis for axis in range(3) if axis != up_axis_index]
    height = float(workbench_extent[up_axis_index])
    forward = (
        _unit(facing_override)
        if facing_override is not None
        else facing_direction(positions, up_axis_index)
    )

    # The character faces a diagonal, not a world axis, so the bench is turned to
    # face it and every distance below is measured in the bench's own frame.
    # Snapping the bench to world axes left the left ankle 0.0143 m from it while
    # the layout believed it had 0.05.
    long_axis, short_axis = sorted(
        horizontal_axes, key=lambda axis: float(workbench_extent[axis]), reverse=True
    )
    depth = float(workbench_extent[short_axis])
    width = float(workbench_extent[long_axis])
    if depth <= 0.0 or height <= 0.0:
        raise LayoutError("Workbench extent is degenerate.")
    yaw_deg = yaw_about_up(forward, short_axis, up_axis_index)
    pelvis = positions["pelvis"]
    # The near edge sits one clearance ahead of the pelvis, so the centre is a
    # further half-depth out.  Nothing here is hand-tuned to a scene.
    centre_distance = front_clearance_m + depth / 2.0
    centre = [float(pelvis[axis]) for axis in range(3)]
    for axis in horizontal_axes:
        centre[axis] = float(pelvis[axis]) + forward[axis] * centre_distance
    centre[up_axis_index] = float(floor_height_m)

    top_height = float(floor_height_m) + height
    hip_height = float(positions["pelvis"][up_axis_index])
    knee_height = min(
        float(positions["left_knee"][up_axis_index]),
        float(positions["right_knee"][up_axis_index]),
    )
    failures = []
    if top_height > hip_height + MAX_WORKBENCH_TOP_ABOVE_HIP_M:
        failures.append(
            "workbench top {:.4f} rises more than {:.2f} m above the hip {:.4f}; "
            "it would hide the torso, not just the legs".format(
                top_height, MAX_WORKBENCH_TOP_ABOVE_HIP_M, hip_height
            )
        )
    if top_height < knee_height + MIN_WORKBENCH_TOP_ABOVE_KNEE_M:
        failures.append(
            "workbench top {:.4f} is below knee {:.4f} + {:.2f} m; it would not "
            "occlude the legs".format(top_height, knee_height, MIN_WORKBENCH_TOP_ABOVE_KNEE_M)
        )
    return {
        "translation": centre,
        "forward": list(forward),
        "centre_distance_from_pelvis_m": centre_distance,
        "near_edge_distance_from_pelvis_m": front_clearance_m,
        "top_height_operational_m": top_height,
        "hip_height_operational_m": hip_height,
        "knee_height_operational_m": knee_height,
        "depth_used_m": depth,
        "width_used_m": width,
        "yaw_deg_about_up": yaw_deg,
        "short_axis_index": short_axis,
        "long_axis_index": long_axis,
        "local_extent_m": {"depth": depth, "width": width, "height": height},
        "failures": failures,
        "pass": not failures,
    }


def align_translation(
    intended_centre: Sequence[float],
    local_min: Sequence[float],
    local_max: Sequence[float],
    yaw_deg: float,
    up_axis_index: int,
    floor_height_m: float = FLOOR_HEIGHT_OPERATIONAL_M,
) -> dict:
    """Translation that lands an asset's *bounds* where the layout wants them.

    Isaac props are not modelled around their own origin -- thor_table's bounds
    sit 0.2237 m off it -- so translating the origin to the intended centre puts
    the visible object somewhere else.  The vertical component lands the asset's
    lowest point on the floor rather than its origin.
    """
    horizontal_axes = [axis for axis in range(3) if axis != up_axis_index]
    first, second = horizontal_axes
    centre_local = [
        (float(local_min[axis]) + float(local_max[axis])) / 2.0 for axis in range(3)
    ]
    angle = math.radians(float(yaw_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = [0.0, 0.0, 0.0]
    rotated[first] = centre_local[first] * cosine - centre_local[second] * sine
    rotated[second] = centre_local[first] * sine + centre_local[second] * cosine

    translation = [0.0, 0.0, 0.0]
    for axis in horizontal_axes:
        translation[axis] = float(intended_centre[axis]) - rotated[axis]
    translation[up_axis_index] = float(floor_height_m) - float(local_min[up_axis_index])
    return {
        "translation": translation,
        "local_bounds_centre": centre_local,
        "rotated_bounds_centre": rotated,
        "origin_to_bounds_offset_m": math.sqrt(
            sum(rotated[axis] ** 2 for axis in horizontal_axes)
        ),
    }


def yaw_about_up(forward: Sequence[float], local_axis_index: int, up_axis_index: int) -> float:
    """Rotation about the up axis that turns the prop's local axis toward `forward`.

    Reported in degrees so it can be authored directly as a rotate op.
    """
    first, second = [axis for axis in range(3) if axis != up_axis_index]
    base = [0.0, 0.0, 0.0]
    base[local_axis_index] = 1.0
    current = math.atan2(base[second], base[first])
    target = math.atan2(float(forward[second]), float(forward[first]))
    degrees = math.degrees(target - current)
    return (degrees + 180.0) % 360.0 - 180.0


def _horizontal_basis(forward: Sequence[float], up_axis_index: int):
    up = tuple(1.0 if axis == up_axis_index else 0.0 for axis in range(3))
    ahead = _unit(
        tuple(
            0.0 if axis == up_axis_index else float(forward[axis]) for axis in range(3)
        )
    )
    return ahead, _unit(_cross(ahead, up))


def clearance_to_oriented_box(
    positions: Mapping[str, Sequence[float]],
    centre: Sequence[float],
    forward: Sequence[float],
    up_axis_index: int,
    local_extent: Mapping[str, float],
) -> dict:
    """Smallest gap between any joint and a box turned to face `forward`.

    Measuring in the box's own frame is what an axis-aligned test could not do
    once the character turned out to face a diagonal: the world-aligned box
    reported 0.0143 m of clearance the layout had not accounted for.  Negative
    means the joint is inside the prop.
    """
    ahead, across = _horizontal_basis(forward, up_axis_index)
    half = {
        "depth": float(local_extent["depth"]) / 2.0,
        "width": float(local_extent["width"]) / 2.0,
    }
    worst = None
    worst_joint = None
    for joint, position in positions.items():
        delta = _sub(position, centre)
        along = sum(delta[axis] * ahead[axis] for axis in range(3))
        sideways = sum(delta[axis] * across[axis] for axis in range(3))
        vertical = float(position[up_axis_index]) - float(centre[up_axis_index])
        gap = max(
            abs(along) - half["depth"],
            abs(sideways) - half["width"],
            vertical - float(local_extent["height"]),
        )
        if worst is None or gap < worst:
            worst = gap
            worst_joint = joint
    return {"min_clearance_m": worst, "closest_joint": worst_joint}
