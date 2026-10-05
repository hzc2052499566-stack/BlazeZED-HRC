"""Occlusion-aware reliability flag for right-arm endpoint joints.

The stabilisation study closed 2D filtering as a route to accuracy under
occlusion and left one usable signal: an occluded endpoint joint samples the
occluder surface, so its measured depth collapses far in front of the torso.
This module turns that into a deployable per-sample reliability class, and
evaluates what flagging actually buys on frozen data.

The classifier is causal, per-frame, uses no ground truth and no future
frames, and needs only quantities the live pipeline already computes.

Registered configuration
------------------------
collapse_margin_m = 0.50

Derived from scene geometry, not scanned: over the whole controlled-arm
sequence the forearm ground-truth forward depth spans 27.2 mm, while the
registered occluder front plane sits 1.6 m in front of the subject.  A 0.50 m
margin is therefore more than an order of magnitude above natural arm
excursion and far below the occluder offset.  A sensitivity sweep is reported
as a robustness check; the registered value remains the primary result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "output" / "experiments" / "kinematic_constraints_ablation"
FORMAL_ROOT = ABLATION / "dynamic_partial_occlusion_formal_v1"
CLEAN_ROOT = ABLATION / "dynamic_k4_gt_confirmatory_v1"
OUTPUT_ROOT = ABLATION / "occlusion_reliability_flag_v1"
SCENE_MANIFEST = (
    ROOT
    / "output"
    / "isaac_scenes"
    / "controlled_arm_motion_v6"
    / "controlled_arm_motion_v6_06_partial_occlusion_v2_manifest.json"
)
ENDPOINT_JOINTS = ("right_elbow", "right_wrist")
TORSO_REFERENCE_JOINTS = ("right_shoulder", "pelvis")
COLLAPSE_MARGIN_M = 0.50
SWEEP_MARGINS_M = (0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.20)
REPEAT_COUNT = 3
FRAME_COUNT = 240

RELIABLE = "reliable"
OCCLUDED_SUSPECTED = "occluded_suspected"
NO_MEASURED_DEPTH = "no_measured_depth"

POLICIES = {
    "strict": (OCCLUDED_SUSPECTED,),
    "sensitive": (OCCLUDED_SUSPECTED, NO_MEASURED_DEPTH),
}


class FlagError(RuntimeError):
    """Raised when the reliability-flag evaluation contract is incomplete."""


def classify_endpoint(
    joint_depth_m: float | None,
    torso_reference_depth_m: float | None,
    collapse_margin_m: float = COLLAPSE_MARGIN_M,
) -> str:
    """Classify one endpoint joint sample.

    Pipeline-ready: no ground truth, no future frames, no history.
    """
    if joint_depth_m is None or not math.isfinite(joint_depth_m):
        return NO_MEASURED_DEPTH
    if torso_reference_depth_m is None or not math.isfinite(
        torso_reference_depth_m
    ):
        return RELIABLE
    if torso_reference_depth_m - joint_depth_m > collapse_margin_m:
        return OCCLUDED_SUSPECTED
    return RELIABLE


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(values: list[float]) -> dict:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "sample_sd": None, "values": []}
    return {
        "n": len(clean),
        "mean": float(statistics.fmean(clean)),
        "sample_sd": float(statistics.stdev(clean)) if len(clean) > 1 else 0.0,
        "values": clean,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(
        ordered[low] * (high - position) + ordered[high] * (position - low)
    )


def active_frames() -> set[int]:
    scene = json.loads(SCENE_MANIFEST.read_text(encoding="utf-8"))
    return {
        frame
        for block in scene["occlusion_intervals"]
        for frame in range(
            int(block["start_frame"]), int(block["end_frame_inclusive"]) + 1
        )
    }


def load_run(directory: Path) -> dict:
    """Collect per-sample depth, classification inputs and K3 error."""
    rows = read_csv(
        directory / "dynamic_k4_replay" / "dynamic_kinematic_estimates.csv"
    )
    gt = {
        (int(r["sequence_index"]), r["canonical_joint"]): (
            float(r["gt_x_m"]),
            float(r["gt_y_m"]),
            float(r["gt_z_m"]),
        )
        for r in read_csv(directory / "ground_truth_joints.csv")
    }
    raw_depth: dict[tuple[int, str], float | None] = {}
    k3: dict[tuple[int, str], float | None] = {}
    for row in rows:
        key = (int(row["sequence_index"]), row["canonical_joint"])
        valid = str(row.get("valid", "")).strip() == "1"
        if row["method"] == "raw_measured":
            raw_depth[key] = float(row["x_m"]) if valid else None
        elif row["method"] == "k3_inferred" and key[1] in ENDPOINT_JOINTS:
            if not valid:
                k3[key] = None
                continue
            point = (
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"]),
            )
            k3[key] = (
                1000.0 * math.dist(point, gt[key])
                if all(math.isfinite(v) for v in point)
                else None
            )
    samples = []
    for frame in range(FRAME_COUNT):
        reference = [
            raw_depth.get((frame, joint))
            for joint in TORSO_REFERENCE_JOINTS
        ]
        reference = [v for v in reference if v is not None]
        torso = statistics.fmean(reference) if reference else None
        for joint in ENDPOINT_JOINTS:
            samples.append(
                {
                    "frame": frame,
                    "joint": joint,
                    "joint_depth_m": raw_depth.get((frame, joint)),
                    "torso_reference_depth_m": torso,
                    "k3_error_mm": k3.get((frame, joint)),
                }
            )
    return {"samples": samples}


def evaluate_policy(
    samples: list[dict],
    frames: set[int],
    policy: str,
    margin: float,
) -> dict:
    flagged_classes = POLICIES[policy]
    selected = [s for s in samples if s["frame"] in frames]
    if not selected:
        raise FlagError("No samples in the requested frame set.")
    flags = [
        classify_endpoint(
            s["joint_depth_m"], s["torso_reference_depth_m"], margin
        )
        in flagged_classes
        for s in selected
    ]
    produced = [
        (s, f)
        for s, f in zip(selected, flags)
        if s["k3_error_mm"] is not None
    ]
    retained = [s["k3_error_mm"] for s, f in produced if not f]
    suppressed = [s["k3_error_mm"] for s, f in produced if f]
    return {
        "sample_count": len(selected),
        "flag_rate": sum(flags) / len(flags),
        "k3_output_count": len(produced),
        "retained_count": len(retained),
        "retained_fraction_of_outputs": (
            len(retained) / len(produced) if produced else None
        ),
        "retained_mean_mm": statistics.fmean(retained) if retained else None,
        "retained_p95_mm": percentile(retained, 95.0),
        "suppressed_mean_mm": (
            statistics.fmean(suppressed) if suppressed else None
        ),
        "unflagged_baseline_mean_mm": (
            statistics.fmean([s["k3_error_mm"] for s, _ in produced])
            if produced
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=REPEAT_COUNT)
    args = parser.parse_args()

    summary_path = OUTPUT_ROOT / "reliability_flag_summary.json"
    if summary_path.exists():
        raise FlagError("Refusing to overwrite: {}".format(summary_path))

    active = active_frames()
    inactive = set(range(FRAME_COUNT)) - active
    datasets = {
        "occluded_formal": [
            FORMAL_ROOT / "unit_corrected" / "rep_{:02d}".format(i)
            for i in range(1, args.repeats + 1)
        ],
        "clean_confirmatory": [
            CLEAN_ROOT
            / "unit_corrected"
            / "confirmatory_rep_{:02d}".format(i)
            for i in range(1, args.repeats + 1)
        ],
    }
    runs = {
        name: [load_run(path) for path in paths]
        for name, paths in datasets.items()
    }

    phases = {"active": active, "inactive": inactive}
    results: dict = {}
    for dataset, loaded in runs.items():
        results[dataset] = {}
        for policy in POLICIES:
            results[dataset][policy] = {}
            for phase, frames in phases.items():
                per_run = [
                    evaluate_policy(
                        run["samples"], frames, policy, COLLAPSE_MARGIN_M
                    )
                    for run in loaded
                ]
                results[dataset][policy][phase] = {
                    key: summarise([item[key] for item in per_run])
                    for key in (
                        "flag_rate",
                        "retained_fraction_of_outputs",
                        "retained_mean_mm",
                        "retained_p95_mm",
                        "suppressed_mean_mm",
                        "unflagged_baseline_mean_mm",
                    )
                }

    sweep = {}
    for margin in SWEEP_MARGINS_M:
        key = "{:.2f}".format(margin)
        sweep[key] = {}
        for dataset, loaded in runs.items():
            sweep[key][dataset] = {
                policy: {
                    phase: summarise(
                        [
                            evaluate_policy(
                                run["samples"], frames, policy, margin
                            )["flag_rate"]
                            for run in loaded
                        ]
                    )["mean"]
                    for phase, frames in phases.items()
                }
                for policy in POLICIES
            }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "occlusion_reliability_flag_v1",
        "claim_eligibility": "offline_evaluation_over_frozen_captures",
        "classifier": {
            "classes": [RELIABLE, OCCLUDED_SUSPECTED, NO_MEASURED_DEPTH],
            "collapse_margin_m": COLLAPSE_MARGIN_M,
            "torso_reference_joints": list(TORSO_REFERENCE_JOINTS),
            "uses_ground_truth": False,
            "uses_future_frames": False,
            "uses_history": False,
            "margin_derivation": (
                "Forearm GT forward depth spans 27.2 mm over the sequence "
                "while the occluder front plane is 1.6 m nearer, so 0.50 m "
                "sits an order of magnitude above arm excursion and well "
                "below the occluder offset. Derived from geometry, not "
                "scanned."
            ),
        },
        "policies": {
            "strict": "flag only occluded_suspected",
            "sensitive": "flag occluded_suspected or no_measured_depth",
        },
        "datasets": {
            "occluded_formal": "3 formal partial-occlusion runs",
            "clean_confirmatory": (
                "3 Dynamic-GT confirmatory runs, no occluder, negative "
                "control"
            ),
        },
        "statistical_unit": "independent_run",
        "dispersion": "sample_sd_over_runs_ddof_1",
        "results": results,
        "margin_sensitivity_flag_rate": sweep,
        "interpretation_rules": [
            "Flag rate on the clean confirmatory dataset is the false-alarm "
            "rate; there is no occluder in those captures.",
            "retained_mean_mm is a selective-prediction figure: the error "
            "the system would report after suppressing flagged samples. It "
            "is not comparable to a full-coverage mean.",
            "The sensitive policy inherits the baseline depth-sampling "
            "failure rate as false alarms; that ambiguity is structural.",
            "This is an offline evaluation. The classifier is not wired "
            "into the live pipeline.",
        ],
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(summary_path), flags)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")

    rows = []
    for dataset in results:
        for policy in POLICIES:
            for phase in phases:
                block = results[dataset][policy][phase]
                rows.append(
                    {
                        "dataset": dataset,
                        "policy": policy,
                        "phase": phase,
                        **{
                            key: block[key]["mean"]
                            for key in (
                                "flag_rate",
                                "retained_fraction_of_outputs",
                                "retained_mean_mm",
                                "retained_p95_mm",
                                "suppressed_mean_mm",
                                "unflagged_baseline_mean_mm",
                            )
                        },
                    }
                )
    with (OUTPUT_ROOT / "reliability_flag_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Summary: {}".format(summary_path))
    header = "{:18s} {:10s} {:9s} {:>10s} {:>12s} {:>12s} {:>12s}".format(
        "dataset", "policy", "phase", "flag", "retained", "retain_mm",
        "baseline_mm",
    )
    print(header)
    for row in rows:
        print(
            "{:18s} {:10s} {:9s} {:10.3f} {:12.3f} {:12.2f} {:12.2f}".format(
                row["dataset"],
                row["policy"],
                row["phase"],
                row["flag_rate"],
                row["retained_fraction_of_outputs"],
                row["retained_mean_mm"] if row["retained_mean_mm"] else 0.0,
                row["unflagged_baseline_mean_mm"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
