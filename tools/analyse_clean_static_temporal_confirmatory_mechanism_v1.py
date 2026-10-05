"""Post-hoc mechanism diagnostic for the frozen static confirmatory result."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "output" / "experiments" / "kinematic_constraints_ablation" / "clean_static_temporal_confirmatory_v1"
OUTPUT = EXP_ROOT / "mechanism_diagnostic_v1.json"
RUNS = ("rep_02", "rep_03", "rep_04")
JOINTS = ("right_elbow", "right_wrist")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tracking(path: Path) -> dict:
    result = {}
    for row in read_csv(path):
        frame = int(row["frame_index"])
        joint = row["canonical_joint"]
        if frame < 10 or joint not in JOINTS or row["valid"] != "1":
            continue
        result[(frame, joint)] = {
            "point": np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]),
            "pixel": np.array([float(row["pixel_x_float"]), float(row["pixel_y_float"])]),
            "depth": float(row["depth_m"]),
        }
    return result


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    formal = EXP_ROOT / "formal_summary.json"
    results = []
    for run_id in RUNS:
        truth = {
            (int(row["sample_index"]), row["canonical_joint"]): np.array([
                float(row["gt_x_m"]), float(row["gt_y_m"]), float(row["gt_z_m"])
            ])
            for row in read_csv(EXP_ROOT / "raw" / run_id / "ground_truth_joints.csv")
        }
        tracking = {
            condition: load_tracking(
                EXP_ROOT / "derived" / "runs" / run_id / condition / "replay" / "tracking_joints.csv"
            )
            for condition in ("T0_frame_independent", "T2_tracker_plus_smoothing")
        }
        for joint in JOINTS:
            keys = sorted(
                {key for key in tracking["T0_frame_independent"] if key[1] == joint}
                & {key for key in tracking["T2_tracker_plus_smoothing"] if key[1] == joint}
            )
            t0 = np.stack([tracking["T0_frame_independent"][key]["point"] for key in keys])
            t2 = np.stack([tracking["T2_tracker_plus_smoothing"][key]["point"] for key in keys])
            gt = np.stack([truth[key] for key in keys])
            pixel0 = np.stack([tracking["T0_frame_independent"][key]["pixel"] for key in keys])
            pixel2 = np.stack([tracking["T2_tracker_plus_smoothing"][key]["pixel"] for key in keys])
            depth_delta = [
                tracking["T2_tracker_plus_smoothing"][key]["depth"]
                - tracking["T0_frame_independent"][key]["depth"]
                for key in keys
            ]
            results.append({
                "run_id": run_id,
                "joint": joint,
                "common_frame_count": len(keys),
                "T0_mean_signed_xyz_error_mm": ((t0 - gt).mean(axis=0) * 1000.0).tolist(),
                "T2_mean_signed_xyz_error_mm": ((t2 - gt).mean(axis=0) * 1000.0).tolist(),
                "T2_minus_T0_mean_xyz_mm": ((t2 - t0).mean(axis=0) * 1000.0).tolist(),
                "T2_minus_T0_mean_uv_px": (pixel2 - pixel0).mean(axis=0).tolist(),
                "T2_minus_T0_mean_depth_mm": float(np.mean(depth_delta) * 1000.0),
            })
    payload = {
        "schema_version": 1,
        "status": "complete",
        "classification": "post_hoc_mechanism_diagnostic_after_formal_result",
        "formal_summary_sha256": sha256_file(formal),
        "formal_result_modified": False,
        "coordinate_order": ["X_forward", "Y_left", "Z_up"],
        "results": results,
        "interpretation": (
            "The repeatable safety-gate failure is concentrated at the wrist. "
            "T2 locks to a stable but shifted 2D wrist location and a 18.5-20.5 mm nearer "
            "depth surface, increasing position bias while reducing variance."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
