"""Retarget the frozen FS-CTS5 motion onto another character (pure Python).

The four cohort characters share one rig family: step 1 verified that all of
them expose the same 101 `RL_BoneRoot/...` joint tokens and resolve the same 15
canonical joints (AGENTS.md 6.6.8).  The motion is therefore carried as *extra
local rotations* relative to each rig's own rest pose, never as absolute joint
orientations: a taller or differently proportioned skeleton keeps its own rest
pose and receives the same joint articulation.

What this module does NOT do: it does not decide whether the retargeted result
is usable.  Bone-length invariance, the v2 relative geometry contract and the
absolute ground-contact gates all still have to pass per character, and each of
those lives in its own module.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from fs_cts5_ankle_compensation_v3 import (
    AnkleCompensationError,
    FRAME_COUNT,
    mat3_multiply,
    require_rotation,
    rotation_angle_deg,
)


RETARGET_TAG = "fs_cts5_motion_retarget_v1"
CLASSIFICATION = "excluded_engineering_asset_not_formal_data"
FORMAL_CAPTURE_AUTHORIZED = False

# Joints the frozen v2 animation drives, by canonical name.
DRIVEN_JOINTS = (
    "left_shoulder", "left_elbow",
    "right_shoulder", "right_elbow",
    "left_hip", "left_knee",
    "right_hip", "right_knee",
)
# Joints v3 adds through ankle compensation.
COMPENSATED_JOINTS = ("left_ankle", "right_ankle")


class RetargetError(RuntimeError):
    """Raised when source and target rigs cannot carry the same motion."""


def check_topology(source_tokens: Mapping[str, str], target_tokens: Mapping[str, str]) -> list:
    """Compare the driven joints' token paths between two rigs.

    Identical token paths mean the two skeletons articulate the same chain, so
    an extra rotation means the same thing on both.  Differences are returned
    rather than raised: a cousin rig may still be usable, but the caller has to
    decide that deliberately instead of inheriting it silently.
    """
    differences = []
    for joint in DRIVEN_JOINTS + COMPENSATED_JOINTS:
        if joint not in source_tokens:
            differences.append("source rig does not resolve {}".format(joint))
            continue
        if joint not in target_tokens:
            differences.append("target rig does not resolve {}".format(joint))
            continue
        if source_tokens[joint] != target_tokens[joint]:
            differences.append(
                "{}: source {!r} vs target {!r}".format(
                    joint, source_tokens[joint], target_tokens[joint]
                )
            )
    return differences


def retarget_local_rotation(extra, target_rest_rotation) -> list:
    """Apply a source extra rotation to a target joint's own rest orientation."""
    return mat3_multiply(
        require_rotation(extra, "extra rotation"),
        require_rotation(target_rest_rotation, "target rest rotation"),
    )


def retarget_series(extras_by_joint, target_rest_rotations) -> dict:
    """Retarget every driven joint across all 241 frames.

    ``extras_by_joint`` is ``{canonical_joint: {frame: 3x3}}`` recovered from the
    source animation; ``target_rest_rotations`` is ``{canonical_joint: 3x3}``
    read from the target skeleton's own restTransforms.
    """
    expected = set(range(FRAME_COUNT))
    missing = [joint for joint in DRIVEN_JOINTS if joint not in extras_by_joint]
    if missing:
        raise RetargetError("No source rotations for {}.".format(sorted(missing)))
    missing_rest = [
        joint for joint in DRIVEN_JOINTS if joint not in target_rest_rotations
    ]
    if missing_rest:
        raise RetargetError("No target rest pose for {}.".format(sorted(missing_rest)))

    locals_by_joint = {}
    max_articulation_deg = 0.0
    max_articulation_joint = None
    for joint in DRIVEN_JOINTS:
        frames = extras_by_joint[joint]
        if set(frames) != expected:
            raise RetargetError(
                "{} must supply frames 0..{}.".format(joint, FRAME_COUNT - 1)
            )
        locals_by_joint[joint] = {}
        for frame in range(FRAME_COUNT):
            extra = frames[frame]
            locals_by_joint[joint][frame] = retarget_local_rotation(
                extra, target_rest_rotations[joint]
            )
            angle = rotation_angle_deg(require_rotation(extra, "extra rotation"))
            if angle > max_articulation_deg:
                max_articulation_deg = angle
                max_articulation_joint = joint

    return {
        "retarget_tag": RETARGET_TAG,
        "classification": CLASSIFICATION,
        "formal_capture_authorized": FORMAL_CAPTURE_AUTHORIZED,
        "frame_count": FRAME_COUNT,
        "driven_joints": list(DRIVEN_JOINTS),
        "local_rotations": locals_by_joint,
        "max_articulation_deg": max_articulation_deg,
        "max_articulation_joint": max_articulation_joint,
        "note": (
            "Joint articulation is carried, not joint orientation: each rig keeps "
            "its own rest pose. Bone-length invariance, the v2 relative geometry "
            "contract and ground contact are still gated per character."
        ),
    }


def bone_length_invariance(lengths_by_frame: Sequence[Mapping[str, float]]) -> dict:
    """Bone lengths must not change while only rotations are applied."""
    if not lengths_by_frame:
        raise RetargetError("No per-frame bone lengths to check.")
    baseline = lengths_by_frame[0]
    worst_ratio = 0.0
    worst_bone = None
    worst_frame = None
    for frame, lengths in enumerate(lengths_by_frame):
        if set(lengths) != set(baseline):
            raise RetargetError("Frame {} measures a different bone set.".format(frame))
        for bone, value in lengths.items():
            reference = float(baseline[bone])
            if reference <= 0.0:
                raise RetargetError("Baseline length for {} is not positive.".format(bone))
            ratio = abs(float(value) - reference) / reference
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_bone = bone
                worst_frame = frame
    return {
        "worst_relative_change": worst_ratio,
        "worst_bone": worst_bone,
        "worst_frame": worst_frame,
        "frame_count": len(lengths_by_frame),
    }
