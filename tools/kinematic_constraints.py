"""Pure 3D kinematic-constraint operations for the eight major limb bones."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np


VARIANTS = (
    "k0_passthrough",
    "k1_length_gate",
    "k2_soft_projection",
    "k2_guarded_projection",
)


def load_json(path: Path) -> dict:
    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_profile(profile: dict) -> None:
    if profile.get("overall_quality_status") != "passed":
        raise ValueError("Kinematic profile must have overall status 'passed'.")
    bones = profile.get("bones", [])
    if len(bones) != 8:
        raise ValueError("Kinematic profile must contain exactly eight bones.")
    names = set()
    for bone in bones:
        name = str(bone.get("name", ""))
        if not name or name in names:
            raise ValueError("Bone names must be non-empty and unique.")
        names.add(name)
        reference = float(bone["reference_length_m"])
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("Invalid reference length for " + name)


def validate_config(config: dict) -> None:
    required = ("k0_passthrough", "k1_length_gate", "k2_soft_projection")
    missing = [variant for variant in required if variant not in config["variants"]]
    if missing:
        raise ValueError("Constraint config is missing: " + ", ".join(missing))
    for variant in ("k2_soft_projection", "k2_guarded_projection"):
        if variant not in config["variants"]:
            continue
        soft = config["variants"][variant]
        if int(soft["iterations"]) < 1:
            raise ValueError(variant + " iterations must be positive.")
        if not 0.0 < float(soft["stiffness"]) <= 1.0:
            raise ValueError(variant + " stiffness must be in (0, 1].")
        if float(soft["maximum_joint_correction_m"]) <= 0.0:
            raise ValueError(variant + " correction cap must be positive.")


def finite_point(value) -> np.ndarray | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point


def bone_length(points: dict[str, np.ndarray], joint_a: str, joint_b: str) -> float | None:
    point_a = finite_point(points.get(joint_a))
    point_b = finite_point(points.get(joint_b))
    if point_a is None or point_b is None:
        return None
    return float(np.linalg.norm(point_b - point_a))


def profile_scale_m(bone: dict) -> float:
    value = bone.get("robust_scale_m", 0.0)
    return float(value) if value is not None and math.isfinite(float(value)) else 0.0


def gate_tolerance_m(bone: dict, settings: dict) -> float:
    reference = float(bone["reference_length_m"])
    return max(
        float(settings["minimum_absolute_tolerance_m"]),
        float(settings["minimum_relative_tolerance"]) * reference,
        float(settings["profile_scale_multiplier"]) * profile_scale_m(bone),
    )


def soft_deadband_m(bone: dict, settings: dict) -> float:
    reference = float(bone["reference_length_m"])
    return max(
        float(settings["minimum_deadband_m"]),
        float(settings["relative_deadband"]) * reference,
        float(settings["profile_scale_multiplier"]) * profile_scale_m(bone),
    )


def joint_role(joint: str) -> str:
    if joint.endswith(("shoulder", "hip")):
        return "root"
    if joint.endswith(("elbow", "knee")):
        return "intermediate"
    if joint.endswith(("wrist", "ankle")):
        return "distal"
    return "other"


def mobility(joint: str, confidence: float, settings: dict) -> float:
    confidence_01 = min(1.0, max(0.0, float(confidence) / 100.0))
    confidence_01 = max(float(settings["confidence_floor"]), confidence_01)
    multiplier = float(
        settings["joint_mobility_multipliers"].get(joint_role(joint), 1.0)
    )
    return multiplier / (
        confidence_01 ** float(settings["confidence_power"])
    )


def enabled_bones(profile: dict) -> list[dict]:
    return [bone for bone in profile["bones"] if bool(bone["constraint_enabled"])]


def evaluate_lengths(
    points: dict[str, np.ndarray],
    valid: dict[str, bool],
    profile: dict,
    gate_settings: dict,
) -> list[dict]:
    diagnostics = []
    for bone in profile["bones"]:
        joint_a = bone["joint_a"]
        joint_b = bone["joint_b"]
        observed = (
            bone_length(points, joint_a, joint_b)
            if valid.get(joint_a, False) and valid.get(joint_b, False)
            else None
        )
        reference = float(bone["reference_length_m"])
        tolerance = gate_tolerance_m(bone, gate_settings)
        error = observed - reference if observed is not None else None
        diagnostics.append(
            {
                "bone": bone["name"],
                "joint_a": joint_a,
                "joint_b": joint_b,
                "constraint_enabled": bool(bone["constraint_enabled"]),
                "observed_length_m": observed,
                "reference_length_m": reference,
                "length_error_m": error,
                "tolerance_m": tolerance,
                "gate_violation": (
                    bool(bone["constraint_enabled"])
                    and error is not None
                    and abs(error) > tolerance
                ),
            }
        )
    return diagnostics


def clip_from_raw(
    candidate: np.ndarray,
    raw: np.ndarray,
    maximum_displacement_m: float,
) -> np.ndarray:
    displacement = candidate - raw
    length = float(np.linalg.norm(displacement))
    if length <= maximum_displacement_m or length <= 0.0:
        return candidate
    return raw + displacement * (maximum_displacement_m / length)


def apply_frame(
    points: dict[str, np.ndarray],
    confidence: dict[str, float],
    valid: dict[str, bool],
    profile: dict,
    config: dict,
    variant: str,
) -> dict:
    """Apply one constraint variant without mutating any input dictionary."""
    if variant not in VARIANTS:
        raise ValueError("Unsupported kinematic variant: " + variant)
    validate_profile(profile)
    validate_config(config)

    raw_points = {
        joint: point.copy()
        for joint, value in points.items()
        if (point := finite_point(value)) is not None
    }
    output_points = {joint: point.copy() for joint, point in raw_points.items()}
    output_valid = {joint: bool(value) for joint, value in valid.items()}
    gate_settings = config["variants"]["k1_length_gate"]
    raw_diagnostics = evaluate_lengths(
        raw_points,
        output_valid,
        profile,
        gate_settings,
    )

    if variant == "k0_passthrough":
        return {
            "points": output_points,
            "valid": output_valid,
            "diagnostics": raw_diagnostics,
            "adjusted_joints": set(),
            "rejected_joints": set(),
        }

    if variant == "k1_length_gate":
        rejected = set()
        for diagnostic in raw_diagnostics:
            if diagnostic["gate_violation"]:
                # The profile is a root-to-distal tree, so joint_b is the
                # endpoint whose rejection least disturbs the torso estimate.
                rejected.add(diagnostic["joint_b"])
        for joint in rejected:
            output_valid[joint] = False
        return {
            "points": output_points,
            "valid": output_valid,
            "diagnostics": raw_diagnostics,
            "adjusted_joints": set(),
            "rejected_joints": rejected,
        }

    if variant not in config["variants"]:
        raise ValueError("Constraint config is missing: " + variant)
    settings = config["variants"][variant]
    enabled = enabled_bones(profile)
    max_correction = float(settings["maximum_joint_correction_m"])
    iterations = int(settings["iterations"])
    stiffness = float(settings["stiffness"])
    raw_gate_by_bone = {
        row["bone"]: bool(row["gate_violation"])
        for row in raw_diagnostics
    }
    skip_raw_gate_violations = bool(
        settings.get("skip_raw_gate_violations", False)
    )
    skipped_bones = {
        name for name, violation in raw_gate_by_bone.items() if violation
    }
    if (
        skip_raw_gate_violations
        and settings.get("skip_connected_chain_on_raw_gate_violation", False)
    ):
        skipped_joints = {
            joint
            for bone in enabled
            if bone["name"] in skipped_bones
            for joint in (bone["joint_a"], bone["joint_b"])
        }
        changed = True
        while changed:
            changed = False
            for bone in enabled:
                if bone["name"] in skipped_bones:
                    continue
                if (
                    bone["joint_a"] in skipped_joints
                    or bone["joint_b"] in skipped_joints
                ):
                    skipped_bones.add(bone["name"])
                    skipped_joints.update((bone["joint_a"], bone["joint_b"]))
                    changed = True

    for iteration in range(iterations):
        ordered_bones = enabled if iteration % 2 == 0 else list(reversed(enabled))
        for bone in ordered_bones:
            if (
                skip_raw_gate_violations
                and bone["name"] in skipped_bones
            ):
                continue
            joint_a = bone["joint_a"]
            joint_b = bone["joint_b"]
            if not output_valid.get(joint_a, False) or not output_valid.get(
                joint_b, False
            ):
                continue
            if joint_a not in output_points or joint_b not in output_points:
                continue
            vector = output_points[joint_b] - output_points[joint_a]
            observed = float(np.linalg.norm(vector))
            if observed <= 1e-9:
                continue
            reference = float(bone["reference_length_m"])
            error = observed - reference
            deadband = soft_deadband_m(bone, settings)
            if abs(error) <= deadband:
                continue
            effective_error = math.copysign(abs(error) - deadband, error)
            correction = stiffness * effective_error * (vector / observed)
            mobility_a = mobility(
                joint_a,
                confidence.get(joint_a, 0.0),
                settings,
            )
            mobility_b = mobility(
                joint_b,
                confidence.get(joint_b, 0.0),
                settings,
            )
            total_mobility = mobility_a + mobility_b
            candidate_a = (
                output_points[joint_a]
                + correction * (mobility_a / total_mobility)
            )
            candidate_b = (
                output_points[joint_b]
                - correction * (mobility_b / total_mobility)
            )
            output_points[joint_a] = clip_from_raw(
                candidate_a,
                raw_points[joint_a],
                max_correction,
            )
            output_points[joint_b] = clip_from_raw(
                candidate_b,
                raw_points[joint_b],
                max_correction,
            )

    adjusted = {
        joint
        for joint in output_points
        if float(np.linalg.norm(output_points[joint] - raw_points[joint])) > 1e-12
    }
    final_diagnostics = evaluate_lengths(
        output_points,
        output_valid,
        profile,
        gate_settings,
    )
    raw_by_bone = {row["bone"]: row for row in raw_diagnostics}
    for diagnostic in final_diagnostics:
        raw = raw_by_bone[diagnostic["bone"]]
        diagnostic["raw_observed_length_m"] = raw["observed_length_m"]
        diagnostic["raw_length_error_m"] = raw["length_error_m"]
    return {
        "points": output_points,
        "valid": output_valid,
        "diagnostics": final_diagnostics,
        "adjusted_joints": adjusted,
        "rejected_joints": set(),
    }
