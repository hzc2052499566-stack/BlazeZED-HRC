"""Freeze the post-sampling K4 shoulder-root fault-injection protocol.

The protocol is intentionally separate from the completed Dynamic-GT
confirmatory experiment.  It references the frozen confirmatory artifacts
read-only and refuses to replace an existing, different protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "dynamic_k4_gt_confirmatory_v1"
    / "unit_corrected"
)
OUTPUT_ROOT = (
    WORKSPACE
    / "output"
    / "experiments"
    / "kinematic_constraints_ablation"
    / "k4_root_fault_injection_v1"
)
PROTOCOL_PATH = OUTPUT_ROOT / "fault_injection_protocol.json"
PROFILE_PATH = (
    WORKSPACE
    / "configs"
    / "subject_profiles"
    / "female_police_live_visual_lite_roi065_wristv5_v2.json"
)
CONFIG_PATH = WORKSPACE / "configs" / "kinematic_constraints_live_v2.json"
MAPPING_PATH = WORKSPACE / "configs" / "joint_mapping.csv"
RUN_IDS = (
    "confirmatory_rep_01",
    "confirmatory_rep_02",
    "confirmatory_rep_03",
)
FAULT_FRAMES = (60, 120, 180)
SCENARIOS = (
    {
        "name": "clean_control",
        "fault_type": "none",
        "fault_frames": [],
        "shoulder_depth_offset_m": 0.0,
    },
    {
        "name": "positive_impulse_150mm",
        "fault_type": "additive_depth_impulse",
        "fault_frames": list(FAULT_FRAMES),
        "shoulder_depth_offset_m": 0.150,
    },
    {
        "name": "negative_impulse_150mm",
        "fault_type": "additive_depth_impulse",
        "fault_frames": list(FAULT_FRAMES),
        "shoulder_depth_offset_m": -0.150,
    },
    {
        "name": "missing_root_impulse",
        "fault_type": "missing_depth_impulse",
        "fault_frames": list(FAULT_FRAMES),
        "shoulder_depth_offset_m": None,
    },
)


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


def source_artifacts(run_id: str) -> dict:
    run_dir = SOURCE_ROOT / run_id
    paths = {
        "manifest": run_dir / "rgbd_manifest.csv",
        "landmark_cache_state": (
            run_dir / "landmark_cache" / "landmark_cache_state.json"
        ),
        "source_estimates": (
            run_dir
            / "dynamic_k4_replay"
            / "dynamic_kinematic_estimates.csv"
        ),
        "source_replay_state": (
            run_dir
            / "dynamic_k4_replay"
            / "dynamic_kinematic_replay_state.json"
        ),
        "ground_truth": run_dir / "ground_truth_joints.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{run_id} is missing source artifact(s): {missing}"
        )
    return {
        key: {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
    }


def build_protocol() -> dict:
    required = (PROFILE_PATH, CONFIG_PATH, MAPPING_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen input(s): {missing}")
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "protocol_name": "k4_root_fault_injection_v1",
        "purpose": (
            "Mechanically validate the K4 shoulder-root guard on the three "
            "completed synchronized Dynamic-GT confirmatory runs."
        ),
        "experimental_unit": (
            "one independently captured 240-frame confirmatory run"
        ),
        "source_root": relative(SOURCE_ROOT),
        "run_ids": list(RUN_IDS),
        "injection_stage": (
            "after frozen depth sampling and before K3/K4 inference"
        ),
        "injected_joint": "right_shoulder",
        "source_fps": 60.0,
        "k4_shoulder_rate_limit_m": 0.060,
        "scenarios": list(SCENARIOS),
        "methods": ["k3_inferred", "k4_inferred"],
        "reporting_joints": ["right_elbow", "right_wrist"],
        "primary_endpoints": [
            "fault-frame valid joint-sample coverage",
            "fault-frame 3D displacement from the method's clean output",
            "K4 transmission attenuation relative to K3",
            "post-fault recovery frames to <=1 mm clean-output displacement",
        ],
        "secondary_endpoints": [
            "fault-frame synchronized 3D GT error",
            "fault-frame paired K4-minus-K3 GT error",
            "K4 shoulder-root clip count and reason",
        ],
        "analysis_rules": [
            "Estimate all scenarios without reading Isaac GT.",
            "Load GT only after every scenario estimate has been written.",
            "Use frame indices 60, 120 and 180 exactly; do not move events.",
            "Do not tune the frozen 60 mm K4 rate limit.",
            "Use the clean scenario to verify exact reproduction of the "
            "completed K3/K4 confirmatory replay.",
            "Report all three runs and run-level mean plus sample SD.",
            "Treat GT accuracy as secondary because the experiment injects "
            "synthetic post-sampling faults.",
        ],
        "interpretation_limits": [
            "This is deterministic robustness evidence, not an independent "
            "real-sensor capture.",
            "A one-frame injected fault does not represent every occlusion or "
            "multi-frame surface-switch failure.",
            "K4 inferred validity is not measured-depth validity.",
        ],
        "frozen_inputs": {
            "profile": {
                "path": relative(PROFILE_PATH),
                "sha256": sha256_file(PROFILE_PATH),
            },
            "config": {
                "path": relative(CONFIG_PATH),
                "sha256": sha256_file(CONFIG_PATH),
            },
            "joint_mapping": {
                "path": relative(MAPPING_PATH),
                "sha256": sha256_file(MAPPING_PATH),
            },
            "runs": {
                run_id: source_artifacts(run_id) for run_id in RUN_IDS
            },
        },
        "freeze_source": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
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
                "Existing fault-injection protocol differs; refusing to "
                f"replace it: {PROTOCOL_PATH}"
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
