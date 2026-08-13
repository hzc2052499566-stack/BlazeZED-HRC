"""Pre-lock relock patch for directional depth sampling. Method version v2.

Defect, confirmed in docs/elbow_directional_depth_latch_diagnosis.md
-------------------------------------------------------------------
Before a joint's directional depth tracker has locked on, the pipeline passes
the **kinematic reference** (shoulder depth plus calibrated bone length) as
`previous_depth_m` to `clustered_valid_depth_m`. The frame-to-frame temporal
jump gate then rejects the measured cluster whenever it disagrees with that
prior by more than `max_temporal_jump_m`, which is 60 mm.

That gate is designed for tracking continuity between consecutive
measurements. Applying it against a static prior is a category error: the
reference carries the bone-length profile's calibration bias, and at the
geometry measured it shifts about 20 mm per pixel of landmark jitter, so a
one-pixel change flips accept to reject.

When it rejects, the pre-lock branch clears the recovery buffer and never
seeds history, so the joint can never lock. In the view C formal that killed
the right elbow for 237 and 230 consecutive frames in two of three runs,
while the *rejected* candidate was the correct value and the reference was
the wrong one.

The fix
-------
When there is no established measurement history, do not apply the
frame-to-frame continuity gate.

`previous_depth_m == kinematic_reference_m` is an exact discriminator for
that state: before lock the caller passes the reference as the previous
depth, and after lock it passes a real measurement, which differs. This was
verified on the instrumented traces of both a failing and a passing run.

No new threshold is introduced. Validation still rests on the mechanisms
designed for establishing a lock: the three-frame consistency confirmation,
and the bone-length `kinematic_rejected` gate, both untouched.

Scope
-----
The patch applies to any joint in the pre-lock state, not only the elbow.
That is the principled scope, because the argument is about locking rather
than about a particular joint.

This module never edits the v1 pipeline on disk. It wraps one function at
runtime, so `zed_blazepose_recorder_kinematic_v1.py` stays byte-identical and
every frozen protocol that hashes it remains valid.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path


METHOD_VERSION = "prelock_relock_v2_20260801"
PATCHED_FUNCTION = "clustered_valid_depth_m"
_REFERENCE_MATCH_TOLERANCE_M = 1.0e-12


def _is_prelock_reference(previous_depth_m, kinematic_reference_m) -> bool:
    """True when the caller has no measurement history for this joint."""
    if previous_depth_m is None or kinematic_reference_m is None:
        return False
    return (
        abs(float(previous_depth_m) - float(kinematic_reference_m))
        <= _REFERENCE_MATCH_TOLERANCE_M
    )


def relock_result(result: dict, prelock: bool) -> dict:
    """Accept a temporally rejected candidate while still pre-lock.

    Pure function so the behaviour is testable without the pipeline.
    """
    if not prelock:
        return result
    if result.get("depth_m") is not None:
        return result
    if not result.get("temporal_rejected"):
        return result
    candidate = result.get("rejected_candidate_depth_m")
    if candidate is None:
        return result
    patched = dict(result)
    patched["depth_m"] = float(candidate)
    patched["temporal_rejected"] = False
    patched["prelock_relock_applied"] = True
    return patched


def apply(pipeline_module) -> dict:
    """Wrap the pipeline's cluster search. Returns a provenance record."""
    original = getattr(pipeline_module, PATCHED_FUNCTION)
    if getattr(original, "_prelock_relock_v2", False):
        raise RuntimeError("The pre-lock relock patch is already applied.")
    signature = inspect.signature(original)

    def patched(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, dict):
            return result
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return result
        bound.apply_defaults()
        prelock = _is_prelock_reference(
            bound.arguments.get("previous_depth_m"),
            bound.arguments.get("kinematic_reference_m"),
        )
        return relock_result(result, prelock)

    patched._prelock_relock_v2 = True
    patched._original = original
    setattr(pipeline_module, PATCHED_FUNCTION, patched)
    source = Path(pipeline_module.__file__)
    return {
        "method_version": METHOD_VERSION,
        "patched_function": PATCHED_FUNCTION,
        "patched_module": source.name,
        "patched_module_sha256": _sha256(source),
        "patch_module": Path(__file__).name,
        "patch_module_sha256": _sha256(Path(__file__).resolve()),
        "behaviour": (
            "While previous_depth_m equals kinematic_reference_m, that is "
            "while the joint has no measurement history, a temporally "
            "rejected cluster is accepted at its rejected candidate depth "
            "so the confirmation buffer can fill and the tracker can lock. "
            "No threshold was added or changed."
        ),
    }


def revert(pipeline_module) -> None:
    current = getattr(pipeline_module, PATCHED_FUNCTION)
    if not getattr(current, "_prelock_relock_v2", False):
        raise RuntimeError("The pre-lock relock patch is not applied.")
    setattr(pipeline_module, PATCHED_FUNCTION, current._original)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
