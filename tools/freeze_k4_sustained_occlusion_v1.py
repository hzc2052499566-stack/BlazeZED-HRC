"""Freeze the sustained K4 root-loss sensitivity protocol.

This study reuses the three completed synchronized Dynamic-GT runs read-only.
It is a deterministic post-sampling robustness experiment, not a rendered
occlusion capture.  The protocol is frozen before estimates are generated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import freeze_k4_root_fault_injection_v1 as root_fault


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "k4_sustained_occlusion_sensitivity_v1"
)
PROTOCOL_PATH = OUTPUT_ROOT / "sustained_occlusion_protocol.json"
RUNNER_PATH = WORKSPACE / "tools" / "run_k4_sustained_occlusion_v1.py"
START_FRAMES = (45, 105, 165)
DURATIONS = (1, 3, 6, 12, 30)
LANDMARK_DROPOUT_DURATION = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def intervals(duration: int) -> list[dict[str, int]]:
    return [
        {
            "start_frame": start,
            "end_frame_inclusive": start + duration - 1,
            "duration_frames": duration,
        }
        for start in START_FRAMES
    ]


def fault_frames(duration: int) -> list[int]:
    return [
        frame
        for block in intervals(duration)
        for frame in range(
            block["start_frame"],
            block["end_frame_inclusive"] + 1,
        )
    ]


def scenarios() -> list[dict]:
    rows = [
        {
            "name": "clean_control",
            "fault_type": "none",
            "fault_frames": [],
            "fault_intervals": [],
            "duration_frames": 0,
            "shoulder_depth_offset_m": 0.0,
        }
    ]
    for duration in DURATIONS:
        rows.append(
            {
                "name": f"missing_root_depth_{duration:02d}f",
                "fault_type": "missing_depth_impulse",
                "fault_frames": fault_frames(duration),
                "fault_intervals": intervals(duration),
                "duration_frames": duration,
                "shoulder_depth_offset_m": None,
            }
        )
    rows.append(
        {
            "name": (
                f"missing_root_landmark_{LANDMARK_DROPOUT_DURATION:02d}f"
            ),
            "fault_type": "missing_landmark_impulse",
            "fault_frames": fault_frames(LANDMARK_DROPOUT_DURATION),
            "fault_intervals": intervals(LANDMARK_DROPOUT_DURATION),
            "duration_frames": LANDMARK_DROPOUT_DURATION,
            "shoulder_depth_offset_m": None,
        }
    )
    return rows


def build_protocol() -> dict:
    required = (
        root_fault.PROFILE_PATH,
        root_fault.CONFIG_PATH,
        root_fault.MAPPING_PATH,
        RUNNER_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen input(s): {missing}")
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "protocol_name": "k4_sustained_occlusion_sensitivity_v1",
        "purpose": (
            "Measure how K3 and frozen K4 respond to registered sustained "
            "right-shoulder depth loss, and distinguish depth loss from full "
            "2D landmark loss."
        ),
        "experimental_unit": (
            "one independently captured 240-frame Dynamic-GT confirmatory run"
        ),
        "source_root": relative(root_fault.SOURCE_ROOT),
        "run_ids": list(root_fault.RUN_IDS),
        "injection_stage": (
            "after frozen depth sampling and before K3/K4 inference"
        ),
        "injected_joint": "right_shoulder",
        "source_fps": 60.0,
        "registered_start_frames": list(START_FRAMES),
        "duration_sweep_frames": list(DURATIONS),
        "k4_shoulder_rate_limit_m": 0.060,
        "scenarios": scenarios(),
        "methods": ["k3_inferred", "k4_inferred"],
        "reporting_joints": ["right_elbow", "right_wrist"],
        "primary_endpoints": [
            "fault-window joint-sample coverage",
            "fault-window displacement from matched clean output",
            "recovery frames after each registered interval",
            "maximum consecutive invalid frames",
        ],
        "secondary_endpoints": [
            "synchronized GT error during the injected interval",
            "K4 shoulder-root clip/prediction count",
            "paired K4-minus-K3 active-window coverage",
        ],
        "pre_registered_checks": {
            "clean_reproduction_max_xyz_difference_m": 1e-12,
            "depth_loss_k4_minimum_active_coverage": 0.99,
            "depth_loss_k4_maximum_clean_displacement_mm": 100.0,
            "depth_loss_k4_maximum_recovery_frames": 3,
            "landmark_loss_expected_k3_active_coverage": 0.0,
            "landmark_loss_expected_k4_active_coverage": 0.0,
        },
        "analysis_rules": [
            "Estimate every scenario before loading synchronized Isaac GT.",
            "Use start frames 45, 105 and 165 exactly.",
            "Use duration sweep 1, 3, 6, 12 and 30 frames exactly.",
            "Do not tune the frozen 60 mm K4 root rate limit.",
            "Recovery requires both reported joints to be valid and within "
            "1 mm of clean output for three consecutive frames.",
            "Report all three runs and run-level mean plus sample SD.",
        ],
        "interpretation_limits": [
            "This is a deterministic post-sampling missing-data proxy, not "
            "a rendered object-occlusion or independent sensor capture.",
            "Missing depth with a retained 2D landmark is easier than full "
            "visual landmark loss.",
            "K4 inferred validity is not measured-depth validity.",
            "Accuracy during injected frames is secondary because inputs are "
            "synthetically altered after sampling.",
        ],
        "frozen_inputs": {
            "profile": {
                "path": relative(root_fault.PROFILE_PATH),
                "sha256": sha256_file(root_fault.PROFILE_PATH),
            },
            "config": {
                "path": relative(root_fault.CONFIG_PATH),
                "sha256": sha256_file(root_fault.CONFIG_PATH),
            },
            "joint_mapping": {
                "path": relative(root_fault.MAPPING_PATH),
                "sha256": sha256_file(root_fault.MAPPING_PATH),
            },
            "runs": {
                run_id: root_fault.source_artifacts(run_id)
                for run_id in root_fault.RUN_IDS
            },
        },
        "implementation": {
            "freeze_script": {
                "path": relative(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "runner": {
                "path": relative(RUNNER_PATH),
                "sha256": sha256_file(RUNNER_PATH),
            },
        },
    }
    payload["protocol_sha256"] = canonical_payload_sha256(payload)
    return payload


def main() -> int:
    payload = build_protocol()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "Existing sustained-occlusion protocol differs; refusing "
                f"to replace it: {PROTOCOL_PATH}"
            )
        print(f"Protocol already frozen: {PROTOCOL_PATH}")
    else:
        PROTOCOL_PATH.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Frozen protocol: {PROTOCOL_PATH}")
    print(f"Protocol SHA-256: {payload['protocol_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
