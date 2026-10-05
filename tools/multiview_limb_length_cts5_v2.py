"""Pure GT-free CTS5 scalar-fusion implementation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


VIEWS = ("m050", "m045", "m040", "p005", "p010")
BONES = ("right_upper_arm", "right_forearm")
CTS5_METHOD = "cts5_calibrated_trimmed_scalar"
CAL_UNIFORM_METHOD = "bias_corrected_uniform_mean"
RAW_UNIFORM_METHOD = "raw_uniform_mean"
RAW_TRIM_METHOD = "uncorrected_trimmed_mean"
V1_METHOD = "v1_optimized_shared_weights"
P010_METHOD = "development_selected_best_single_p010"


class CTS5Error(RuntimeError):
    """Raised when a CTS5 parameter or input violates the frozen contract."""


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_offsets(path: Path) -> tuple[dict[str, dict[str, float]], str]:
    if not path.is_file():
        raise CTS5Error(f"Missing offset payload: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_after_excluded_e0_development":
        raise CTS5Error("CTS5 offsets are not frozen after excluded E0 development.")
    expected = str(payload.get("offset_payload_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("offset_payload_sha256", None)
    if payload_sha256(unhashed) != expected:
        raise CTS5Error("CTS5 offset payload hash is invalid.")
    if tuple(payload.get("views", [])) != VIEWS or tuple(payload.get("bones", [])) != BONES:
        raise CTS5Error("CTS5 offset view/bone contract changed.")
    observed = payload.get("offsets_mm")
    if not isinstance(observed, dict):
        raise CTS5Error("CTS5 offsets_mm is not an object.")
    offsets: dict[str, dict[str, float]] = {}
    for bone in BONES:
        if set(observed.get(bone, {})) != set(VIEWS):
            raise CTS5Error(f"CTS5 offsets are incomplete for {bone}.")
        offsets[bone] = {}
        for view in VIEWS:
            value = float(observed[bone][view])
            if not math.isfinite(value):
                raise CTS5Error(f"CTS5 offset is not finite: {bone}/{view}")
            offsets[bone][view] = value
    return offsets, expected


def trimmed_scalar(values: list[float]) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if len(ordered) >= 4:
        return statistics.fmean(ordered[1:-1])
    if len(ordered) == 3:
        return statistics.median(ordered)
    return None


def cts5_estimate(lengths: dict[str, float | None], bone: str,
                  offsets: dict[str, dict[str, float]]) -> tuple[float | None, int]:
    if bone not in BONES:
        raise CTS5Error(f"Unregistered bone: {bone}")
    corrected = [float(lengths[view]) - offsets[bone][view]
                 for view in VIEWS if lengths.get(view) is not None]
    return trimmed_scalar(corrected), len(corrected)


def corrected_uniform(lengths: dict[str, float | None], bone: str,
                      offsets: dict[str, dict[str, float]]) -> float | None:
    corrected = [float(lengths[view]) - offsets[bone][view]
                 for view in VIEWS if lengths.get(view) is not None]
    return statistics.fmean(corrected) if corrected else None


def raw_uniform(lengths: dict[str, float | None]) -> float | None:
    valid = [float(lengths[view]) for view in VIEWS if lengths.get(view) is not None]
    return statistics.fmean(valid) if valid else None


def uncorrected_trim(lengths: dict[str, float | None]) -> float | None:
    return trimmed_scalar([float(lengths[view]) for view in VIEWS
                           if lengths.get(view) is not None])


def weighted_available(lengths: dict[str, float | None], weights: dict[str, float],
                       minimum_count: int = 2,
                       minimum_mass: float = 0.2) -> float | None:
    valid = [view for view in VIEWS if lengths.get(view) is not None]
    mass = sum(float(weights[view]) for view in valid)
    if len(valid) < minimum_count or mass < minimum_mass:
        return None
    return sum(float(weights[view]) * float(lengths[view]) for view in valid) / mass


def predict_all(lengths: dict[str, float | None], bone: str,
                offsets: dict[str, dict[str, float]],
                v1_weights: dict[str, float]) -> dict[str, float | None]:
    cts5, _ = cts5_estimate(lengths, bone, offsets)
    return {
        CTS5_METHOD: cts5,
        CAL_UNIFORM_METHOD: corrected_uniform(lengths, bone, offsets),
        RAW_UNIFORM_METHOD: raw_uniform(lengths),
        RAW_TRIM_METHOD: uncorrected_trim(lengths),
        V1_METHOD: weighted_available(lengths, v1_weights),
        P010_METHOD: lengths.get("p010"),
    }
