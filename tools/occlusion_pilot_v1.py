"""Depth-based occlusion measurement for the industrial workcell (pure Python).

Step 3 of the AGENTS.md 6.6.7 order asks whether the workbench and the UR10e
actually occlude what they are meant to occlude.  That cannot be answered with
bounding boxes: ``thor_table`` is an open frame, so a box would count the empty space
under its top as solid and assume away the very question.  Occlusion is
therefore read from rendered depth, the same mechanism the project already uses
for GT-free availability: project a joint into the image, sample the rendered
depth there, and call it occluded when something nearer than the joint was
drawn in that pixel.

No Isaac Sim or ``pxr`` dependency; the Isaac script renders and calls in here.
Projection reuses the validated 20-bundle camera maths rather than a new copy.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from fs_cts5_camera_bank_v1 import BONES, JOINTS, camera_basis, project_from_eye


PILOT_TAG = "fs_cts5_workcell_occlusion_pilot_v1"
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False

# A joint counts as occluded when the rendered surface at its pixel is nearer
# than the joint by more than this, which keeps skin depth and depth noise from
# reading as occlusion.  A joint pivot sits inside the body, so the margin also
# has to exceed roughly half a limb thickness.
OCCLUSION_DEPTH_MARGIN_M = 0.06
# Props are detected differentially: the same view and frame is rendered with
# the props hidden, and a prop counts as occluding only when it puts something
# nearer than that baseline.  A fixed margin against the joint pivot cannot
# work, because the character's own skin sits 6-8 cm in front of its pivots and
# more at oblique angles, which is what made idle frames read as 43% occluded.
PROP_DEPTH_MARGIN_M = 0.02
# Endpoint groups the two scenarios are supposed to affect.
LEG_ENDPOINTS = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")
TARGET_ARM_ENDPOINTS = ("right_shoulder", "right_elbow", "right_wrist")
# AGENTS.md 6.6.5: a target cell needs at least three views with measured
# endpoints, so occlusion is only "recoverable" while that many views still see
# both endpoints of the bone.
MIN_MEASURED_VIEWS = 3
# Rendered depth arrives in the stage's declared units while joint positions are
# computed in operational metres, and AGENTS.md 4.4 records that this scene runs
# at metersPerUnit = 0.01 while being used as one unit per operational metre.
# Taking depth at face value made every endpoint in every view read as occluded,
# including idle frames, because 0.035 is always less than 3.5 - 0.06.  The
# caller must state the scale it applied, and this range then checks the result
# actually lands where a physical scene would put it.
DEPTH_RATIO_RANGE = (0.5, 2.0)
# An endpoint is available when a view can measure it: in frame, not hidden by a
# prop, and not hidden behind the body itself.  The last one is read from the
# props-hidden render: the surface at a joint's pixel normally belongs to that
# joint's own limb and sits 6-8 cm in front of the pivot, so a much larger gap
# means another body part is in the way.
SELF_OCCLUSION_MARGIN_M = 0.25


class OcclusionPilotError(RuntimeError):
    """Raised when a pilot sample is unusable rather than merely occluded."""


def ring_aim_point(positions: Mapping[str, Sequence[float]]) -> list:
    """The single definition of what the pilot ring looks at.

    Both the occluder placement search and the pilot render must aim the ring at
    the same point, or they score different cameras: the search once optimised
    against a ring aimed at the forearm midpoint while the pilot rendered one
    aimed at the mid-body, and the two disagreed about every sight line.
    """
    for joint in ("left_hip", "right_hip", "right_shoulder"):
        if joint not in positions:
            raise OcclusionPilotError("Ring aim needs {}.".format(joint))
    aim = [
        (float(positions["left_hip"][axis]) + float(positions["right_hip"][axis])) / 2.0
        for axis in range(3)
    ]
    aim[2] = (aim[2] + float(positions["right_shoulder"][2])) / 2.0
    return aim


# One ring definition, used by the placement search and the render alike.
RING_RADIUS_M = 3.5
RING_AZIMUTHS_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
RING_ELEVATIONS_DEG = (-5.0, 15.0)


def standard_ring(positions: Mapping[str, Sequence[float]]) -> list:
    """The pilot ring for a given character pose."""
    return ring_camera_poses(
        ring_aim_point(positions),
        RING_RADIUS_M,
        RING_AZIMUTHS_DEG,
        RING_ELEVATIONS_DEG,
    )


def ring_camera_poses(
    aim: Sequence[float],
    radius_m: float,
    azimuths_deg: Sequence[float],
    elevations_deg: Sequence[float],
) -> list:
    """Candidate pilot viewpoints on a ring around the character.

    These are pilot viewpoints only.  The formal common five-view bank is
    selected in step 4 under its own GT-free rules; nothing here selects it.
    """
    if radius_m <= 0.0:
        raise OcclusionPilotError("Ring radius must be positive.")
    poses = []
    for elevation in elevations_deg:
        for azimuth in azimuths_deg:
            azimuth_rad = math.radians(float(azimuth))
            elevation_rad = math.radians(float(elevation))
            horizontal = float(radius_m) * math.cos(elevation_rad)
            poses.append(
                {
                    "name": "az{:+04.0f}_el{:+03.0f}".format(azimuth, elevation),
                    "azimuth_deg": float(azimuth),
                    "elevation_deg": float(elevation),
                    "eye": [
                        float(aim[0]) + horizontal * math.cos(azimuth_rad),
                        float(aim[1]) + horizontal * math.sin(azimuth_rad),
                        float(aim[2]) + float(radius_m) * math.sin(elevation_rad),
                    ],
                    "aim": [float(value) for value in aim],
                }
            )
    return poses


def sample_depth(depth_image: Sequence[Sequence[float]], u: float, v: float):
    """Nearest-pixel depth sample, or None when the point is off-image."""
    height = len(depth_image)
    if height == 0:
        raise OcclusionPilotError("Depth image is empty.")
    width = len(depth_image[0])
    column = int(round(float(u)))
    row = int(round(float(v)))
    if column < 0 or row < 0 or column >= width or row >= height:
        return None
    value = float(depth_image[row][column])
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def classify_endpoint(
    world_point: Sequence[float],
    eye: Sequence[float],
    aim: Sequence[float],
    intrinsics: Mapping[str, float],
    depth_image: Sequence[Sequence[float]],
    margin_m: float = OCCLUSION_DEPTH_MARGIN_M,
) -> dict:
    """Decide whether one endpoint is visible, occluded, or out of frame."""
    projected = project_from_eye(eye, aim, world_point, intrinsics)
    if projected is None:
        return {"state": "behind_camera"}
    u, v, joint_depth = projected
    rendered = sample_depth(depth_image, u, v)
    if rendered is None:
        return {"state": "out_of_frame", "u": u, "v": v, "joint_depth_m": joint_depth}
    occluded = rendered < joint_depth - float(margin_m)
    return {
        "state": "occluded" if occluded else "visible",
        "u": u,
        "v": v,
        "joint_depth_m": joint_depth,
        "rendered_depth_m": rendered,
        "depth_gap_m": joint_depth - rendered,
    }


def depth_consistency(states: Sequence[Mapping]) -> dict:
    """Is rendered depth on the same scale as the joint depths it is compared to?

    Takes classified endpoint states and reports the median rendered/joint
    ratio.  A physical scene puts that near 1 whether or not things are
    occluded; a unit mismatch puts it near 0.01 or 100.
    """
    ratios = sorted(
        float(state["rendered_depth_m"]) / float(state["joint_depth_m"])
        for state in states
        if state.get("rendered_depth_m") and state.get("joint_depth_m")
    )
    if not ratios:
        raise OcclusionPilotError("No depth samples to check the scale with.")
    middle = len(ratios) // 2
    median = (
        ratios[middle]
        if len(ratios) % 2
        else (ratios[middle - 1] + ratios[middle]) / 2.0
    )
    plausible = DEPTH_RATIO_RANGE[0] <= median <= DEPTH_RATIO_RANGE[1]
    return {
        "median_rendered_over_joint": median,
        "sample_count": len(ratios),
        "plausible_range": list(DEPTH_RATIO_RANGE),
        "plausible": plausible,
        "diagnosis": (
            ""
            if plausible
            else (
                "rendered depth is {:.4g}x the joint depth; this is a unit scale "
                "mismatch, not occlusion".format(median)
            )
        ),
    }


def classify_endpoint_differential(
    world_point: Sequence[float],
    eye: Sequence[float],
    aim: Sequence[float],
    intrinsics: Mapping[str, float],
    depth_with_props: Sequence[Sequence[float]],
    depth_without_props: Sequence[Sequence[float]],
    margin_m: float = PROP_DEPTH_MARGIN_M,
) -> dict:
    """Is this endpoint hidden *by a prop*, judged against a props-hidden render."""
    projected = project_from_eye(eye, aim, world_point, intrinsics)
    if projected is None:
        return {"state": "behind_camera"}
    u, v, joint_depth = projected
    with_props = sample_depth(depth_with_props, u, v)
    without_props = sample_depth(depth_without_props, u, v)
    if with_props is None or without_props is None:
        return {"state": "out_of_frame", "u": u, "v": v, "joint_depth_m": joint_depth}
    occluded = with_props < without_props - float(margin_m)
    return {
        "state": "occluded" if occluded else "visible",
        "u": u,
        "v": v,
        "joint_depth_m": joint_depth,
        "rendered_depth_m": with_props,
        "baseline_depth_m": without_props,
        "prop_depth_gap_m": without_props - with_props,
        "self_depth_gap_m": joint_depth - without_props,
    }


def endpoint_available(
    state: Mapping, self_margin_m: float = SELF_OCCLUSION_MARGIN_M
) -> bool:
    """GT-free availability: in frame, no prop in the way, not self-occluded."""
    if state.get("state") != "visible":
        return False
    gap = state.get("self_depth_gap_m")
    if gap is None:
        return True
    return float(gap) <= float(self_margin_m)


def evaluate_frame(
    positions: Mapping[str, Sequence[float]],
    views: Sequence[Mapping],
    margin_m: float = PROP_DEPTH_MARGIN_M,
) -> dict:
    """Per-endpoint visibility across every pilot view for a single frame.

    Each view carries ``eye``, ``aim``, ``intrinsics`` and ``depth`` (a 2D
    array).  Returns per-endpoint per-view states plus the bone-level count of
    views that can still measure both endpoints.
    """
    if not views:
        raise OcclusionPilotError("No views to evaluate.")
    endpoints = {}
    for joint in JOINTS:
        if joint not in positions:
            raise OcclusionPilotError("Frame is missing {}.".format(joint))
        endpoints[joint] = [
            classify_endpoint_differential(
                positions[joint],
                view["eye"],
                view["aim"],
                view["intrinsics"],
                view["depth"],
                view["depth_clear"],
                margin_m,
            )
            if "depth_clear" in view
            else classify_endpoint(
                positions[joint],
                view["eye"],
                view["aim"],
                view["intrinsics"],
                view["depth"],
                margin_m,
            )
            for view in views
        ]

    bones = {}
    for bone, (start, end) in BONES.items():
        measured = sum(
            1
            for index in range(len(views))
            if endpoints[start][index]["state"] == "visible"
            and endpoints[end][index]["state"] == "visible"
        )
        bones[bone] = {
            "measured_view_count": measured,
            "meets_minimum": measured >= MIN_MEASURED_VIEWS,
        }
    return {"endpoints": endpoints, "bones": bones}


def summarise(frames: Sequence[Mapping], view_names: Sequence[str]) -> dict:
    """Aggregate per-frame results into the numbers step 3 has to answer."""
    if not frames:
        raise OcclusionPilotError("No frames to summarise.")
    per_view_occluded = {name: {} for name in view_names}
    bone_shortfall = {}
    for frame in frames:
        for joint, states in frame["endpoints"].items():
            for index, state in enumerate(states):
                bucket = per_view_occluded[view_names[index]].setdefault(
                    joint, {"occluded": 0, "visible": 0, "other": 0}
                )
                if state["state"] == "occluded":
                    bucket["occluded"] += 1
                elif state["state"] == "visible":
                    bucket["visible"] += 1
                else:
                    bucket["other"] += 1
        for bone, result in frame["bones"].items():
            entry = bone_shortfall.setdefault(bone, {"frames": 0, "below_minimum": 0})
            entry["frames"] += 1
            if not result["meets_minimum"]:
                entry["below_minimum"] += 1

    def group_rate(group):
        totals = {"occluded": 0, "counted": 0}
        for view in per_view_occluded.values():
            for joint in group:
                bucket = view.get(joint)
                if not bucket:
                    continue
                totals["occluded"] += bucket["occluded"]
                totals["counted"] += bucket["occluded"] + bucket["visible"]
        return totals["occluded"] / totals["counted"] if totals["counted"] else 0.0

    return {
        "pilot_tag": PILOT_TAG,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "frame_count": len(frames),
        "view_count": len(view_names),
        "per_view_endpoint_counts": per_view_occluded,
        "leg_occlusion_rate": group_rate(LEG_ENDPOINTS),
        "target_arm_occlusion_rate": group_rate(TARGET_ARM_ENDPOINTS),
        "bone_view_shortfall": bone_shortfall,
        "min_measured_views": MIN_MEASURED_VIEWS,
        "occlusion_depth_margin_m": OCCLUSION_DEPTH_MARGIN_M,
        "note": (
            "Pilot viewpoints only. Occlusion is measured from rendered depth, "
            "not from bounding boxes, because the chosen bench is an open frame. "
            "The formal common five-view bank is selected in step 4."
        ),
    }
