#!/usr/bin/env python3
"""Compare BODY_38 and BlazePose against Isaac Skeleton GT, per time code.

This is the first stage of Track A that produces error rather than agreement.
S3 (agents.md 4.12) could only report how far the two pipelines sit from each
other, because the real SVO has no human ground truth. Here both pipelines saw
the same frames and Isaac exported the pose that produced them, so the question
"which is closer to the truth" is finally answerable - within the limits below.

Arms, all scored on the same frames:

* ``A2_median``  BlazePose Lite + ROI 0.65 + median_7x7, measured
* ``A2_v5``      BlazePose Lite + ROI 0.65 + wrist_aware_v5, measured
* ``A3``         ZED BODY_38 + HUMAN_BODY_MEDIUM + fitting + tracking

BODY_38 is the **reference**, never the baseline and never ground truth
(agents.md 1, 13).

Three limits are built into every number this writes, and none of them are
optional reading:

1. **Attribution floor.** Frames are matched to a time code by NCC, which
   resolves to about two or three time codes. The right wrist moves 3.29 mm per
   time code (median; p95 5.73), so the ground truth paired with a frame is
   itself uncertain by roughly 7 to 10 mm. Against the 32.6 to 43.2 mm effects
   frozen in 5.8 that is 20 to 30%. Differences smaller than the floor mean
   nothing here.
2. **Definitional bias.** Isaac GT is a skeleton pivot, inside the body.
   BODY_38 is a fitted skeleton joint, also inside the body. BlazePose
   landmarks sit on the visible surface (agents.md 4.2). BODY_38 is therefore
   structurally advantaged on this ground truth, and that is not an algorithmic
   advantage. Every endpoint is reported twice: raw, and with a per-joint
   constant offset removed, estimated on a calibration split disjoint from the
   test split. Raw is deployment error; bias-removed is what compares random
   error and tails.
3. **Unit correction.** The controlled-arm USD declares metersPerUnit 0.01
   while the scene is built at one unit per operational metre, so GT is scaled
   by 100 per agents.md 4.4. The check is that pelvis lands at 3.5 m, the
   frozen subject distance.

Coverage and error are always reported together. A sampler that abstains on
half the wrists produces a flattering mean over what it kept, which is exactly
the selection effect 4.12 had to correct for.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import platform
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]

# agents.md 4.4: declared USD metres to operational metres.
UNIT_SCALE = 100.0
# P0c: the SVO stores whole microseconds while the live stream reports
# nanoseconds, so frames match modulo one quantisation step.
TIMESTAMP_TOLERANCE_NS = 1000
# Calibrated against the independent beacon-lead check: at this margin the
# attributed lag stayed inside [4, 14] for 100% of frames, while 0.002 let
# 10.36% fall outside.
DEFAULT_MIN_MARGIN = 0.03

PRIMARY_JOINTS = ("right_elbow", "right_wrist")
ATTRIBUTION_FLOOR_MM = (7.0, 10.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_ground_truth(gt_dir: Path) -> tuple[dict[int, dict[str, tuple]], dict]:
    """GT joints per time code, in the frozen ZED camera frame, unit corrected."""
    rows = read_csv(gt_dir / "ground_truth_joints.csv")
    by_time_code: dict[int, dict[str, tuple]] = defaultdict(dict)
    for row in rows:
        time_code = int(float(row["usd_time_code"]))
        by_time_code[time_code][row["canonical_joint"]] = (
            float(row["gt_x_m"]) * UNIT_SCALE,
            float(row["gt_y_m"]) * UNIT_SCALE,
            float(row["gt_z_m"]) * UNIT_SCALE,
        )
    pelvis_forward = [
        joints["pelvis"][0]
        for joints in by_time_code.values()
        if "pelvis" in joints
    ]
    check = {
        "time_codes": len(by_time_code),
        "pelvis_forward_median_m": round(statistics.median(pelvis_forward), 4)
        if pelvis_forward
        else None,
        "unit_scale_applied": UNIT_SCALE,
        "expected_subject_distance_m": 3.50,
    }
    if pelvis_forward:
        deviation = abs(statistics.median(pelvis_forward) - 3.50)
        check["pelvis_check_passed"] = deviation <= 0.15
        if not check["pelvis_check_passed"]:
            raise SystemExit(
                "unit check failed: pelvis sits at "
                f"{statistics.median(pelvis_forward):.3f} m after the x{UNIT_SCALE} "
                "correction, expected 3.50. Do not trust any error computed "
                "from this ground truth."
            )
    return by_time_code, check


def load_attribution(
    capture_dir: Path, min_margin: float
) -> tuple[dict[int, int], dict]:
    """Attributed time code per capture timestamp, filtered and re-checked.

    The stored attribution was written at the tool's default margin. Rather
    than regenerate it, the calibrated margin is applied here and monotonicity
    is re-checked on whatever survives, so the recorded artifact is left alone.
    """
    rows = read_csv(capture_dir / "frame_time_code_attribution_v3.csv")
    kept: list[dict] = []
    for row in rows:
        if not row["time_code"] or not row["margin"]:
            continue
        reasons = {r for r in row["reject_reasons"].split(";") if r}
        blocking = reasons - {"margin_below_threshold"}
        if blocking:
            continue
        if float(row["margin"]) < min_margin:
            continue
        kept.append(row)

    # Playback is one-directional; a decrease is a loop wrap or a bad match.
    attributed: dict[int, int] = {}
    previous = None
    dropped_non_monotone = 0
    for row in kept:
        time_code = int(row["time_code"])
        if previous is not None and time_code < previous:
            if time_code >= previous / 2.0:
                dropped_non_monotone += 1
                continue
        previous = time_code
        attributed[int(row["image_timestamp_ns"])] = time_code

    return attributed, {
        "attribution_rows": len(rows),
        "passed_margin": len(kept),
        "dropped_non_monotone": dropped_non_monotone,
        "attributed_frames": len(attributed),
        "min_margin": min_margin,
        "margin_calibration": (
            "chosen against the independent beacon-lead check: 0.00% of "
            "attributed frames fell outside a lag of [4, 14] at this margin, "
            "against 10.36% at 0.002"
        ),
    }


def nearest_time_code(
    keys: list[int], attributed: dict[int, int], timestamp_ns: int
) -> int | None:
    position = bisect.bisect_left(keys, timestamp_ns)
    for index in (position - 1, position):
        if 0 <= index < len(keys):
            key = keys[index]
            if abs(key - timestamp_ns) < TIMESTAMP_TOLERANCE_NS:
                return attributed[key]
    return None


def collect_samples(
    capture_dir: Path,
    attributed: dict[int, int],
    ground_truth: dict[int, dict[str, tuple]],
) -> tuple[dict[str, list[dict]], dict]:
    """One record per (arm, joint, frame) that has an attributed time code."""
    keys = sorted(attributed)
    samples: dict[str, list[dict]] = defaultdict(list)
    slots: dict[str, int] = defaultdict(int)

    for row in read_csv(capture_dir / "body38_joints.csv"):
        time_code = nearest_time_code(
            keys, attributed, int(row["image_timestamp_ns"])
        )
        if time_code is None or time_code not in ground_truth:
            continue
        truth = ground_truth[time_code].get(row["canonical_joint"])
        slots["A3"] += 1
        if truth is None or row["x_m"] == "":
            continue
        samples["A3"].append(
            {
                "time_code": time_code,
                "joint": row["canonical_joint"],
                "estimate": (
                    float(row["x_m"]),
                    float(row["y_m"]),
                    float(row["z_m"]),
                ),
                "truth": truth,
            }
        )

    for row in read_csv(capture_dir / "blazepose_joints.csv"):
        arm = row["arm"]
        time_code = nearest_time_code(
            keys, attributed, int(row["image_timestamp_ns"])
        )
        if time_code is None or time_code not in ground_truth:
            continue
        truth = ground_truth[time_code].get(row["canonical_joint"])
        slots[arm] += 1
        if truth is None or row["valid"] != "1":
            continue
        samples[arm].append(
            {
                "time_code": time_code,
                "joint": row["canonical_joint"],
                "estimate": (
                    float(row["x_m"]),
                    float(row["y_m"]),
                    float(row["z_m"]),
                ),
                "truth": truth,
            }
        )

    return samples, dict(slots)


def estimate_bias(
    records: list[dict], calibration_time_codes: set[int]
) -> dict[str, tuple[float, float, float]]:
    """Per-joint constant 3D offset, median over the calibration split.

    BlazePose landmarks sit on the body surface and Isaac GT is a skeleton
    pivot, so a constant offset is expected and is not algorithm error. Removing
    it is what lets random error and tails be compared across arms that differ
    structurally.
    """
    by_joint: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for record in records:
        if record["time_code"] not in calibration_time_codes:
            continue
        by_joint[record["joint"]].append(
            tuple(
                estimate - truth
                for estimate, truth in zip(record["estimate"], record["truth"])
            )
        )
    return {
        joint: (
            statistics.median(value[0] for value in values),
            statistics.median(value[1] for value in values),
            statistics.median(value[2] for value in values),
        )
        for joint, values in by_joint.items()
        if values
    }


def summarise(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    index_p95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(ordered),
        "mean_mm": round(statistics.mean(ordered), 3),
        "median_mm": round(statistics.median(ordered), 3),
        "p95_mm": round(ordered[index_p95], 3),
        "max_mm": round(ordered[-1], 3),
    }


def evaluate(
    records: list[dict],
    test_time_codes: set[int],
    bias: dict[str, tuple[float, float, float]],
) -> dict:
    raw: dict[str, list[float]] = defaultdict(list)
    corrected: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["time_code"] not in test_time_codes:
            continue
        joint = record["joint"]
        offset = bias.get(joint, (0.0, 0.0, 0.0))
        raw_error = (
            math.dist(record["estimate"], record["truth"]) * 1000.0
        )
        shifted = tuple(
            estimate - correction
            for estimate, correction in zip(record["estimate"], offset)
        )
        raw[joint].append(raw_error)
        corrected[joint].append(math.dist(shifted, record["truth"]) * 1000.0)

    def pooled(source: dict[str, list[float]], joints) -> dict:
        values: list[float] = []
        for joint in joints:
            values.extend(source.get(joint, []))
        return summarise(values)

    all_joints = sorted(set(raw) | set(corrected))
    return {
        "per_joint_raw": {joint: summarise(raw[joint]) for joint in all_joints},
        "per_joint_bias_removed": {
            joint: summarise(corrected[joint]) for joint in all_joints
        },
        "right_arm_two_joint_raw": pooled(raw, PRIMARY_JOINTS),
        "right_arm_two_joint_bias_removed": pooled(corrected, PRIMARY_JOINTS),
        "bias_mm": {
            joint: [round(value * 1000.0, 3) for value in offset]
            for joint, offset in sorted(bias.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--core-joints-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    capture_dir = args.capture_dir.resolve()
    report_path = capture_dir / "body38_gt_comparison.json"
    if report_path.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite {relative(report_path)}; pass --overwrite"
        )

    ground_truth, unit_check = load_ground_truth(args.gt_dir.resolve())
    print(
        "ground truth: {} time codes, pelvis at {} m after x{} correction".format(
            unit_check["time_codes"],
            unit_check["pelvis_forward_median_m"],
            UNIT_SCALE,
        )
    )

    attributed, attribution_stats = load_attribution(capture_dir, args.min_margin)
    print(
        "attribution: {} frames at margin >= {} ({} dropped non-monotone)".format(
            attribution_stats["attributed_frames"],
            args.min_margin,
            attribution_stats["dropped_non_monotone"],
        )
    )
    if not attributed:
        raise SystemExit("no frames survived attribution; nothing to compare")

    samples, slots = collect_samples(capture_dir, attributed, ground_truth)

    # Calibration and test splits are disjoint by construction and both span
    # the whole motion, so the bias is not estimated on the frames it is then
    # removed from.
    time_codes = sorted({record["time_code"] for arm in samples.values() for record in arm})
    calibration = {code for code in time_codes if code % 2 == 0}
    test = {code for code in time_codes if code % 2 == 1}

    arms: dict[str, Any] = {}
    for arm, records in sorted(samples.items()):
        bias = estimate_bias(records, calibration)
        result = evaluate(records, test, bias)
        valid_slots = slots.get(arm, 0)
        result["coverage"] = {
            "joint_slots": valid_slots,
            "valid_samples": len(records),
            "valid_fraction": round(len(records) / valid_slots, 6)
            if valid_slots
            else 0.0,
            "note": (
                "coverage and error must be quoted together: an arm that "
                "abstains on hard samples produces a flattering mean over "
                "what it kept"
            ),
        }
        arms[arm] = result

    payload = {
        "schema_version": 1,
        "stage": "body38_gt_comparison",
        "classification": "engineering_result_protocol_not_frozen",
        "generated_utc": utc_now(),
        "plan_document": "docs/body38_vs_blazepose_gt_comparison_plan_v1.md",
        "environment": {"platform": platform.platform()},
        "inputs": {
            "capture_dir": relative(capture_dir),
            "ground_truth_dir": relative(args.gt_dir.resolve()),
        },
        "unit_check": unit_check,
        "attribution": attribution_stats,
        "split": {
            "calibration_time_codes": len(calibration),
            "test_time_codes": len(test),
            "rule": "even time codes calibrate the bias, odd ones are scored",
        },
        "arms": arms,
        "interpretation_limits": [
            "BODY_38 is a reference pipeline, never a baseline and never "
            "ground truth",
            "NCC attribution resolves to about two or three time codes and the "
            "right wrist moves 3.29 mm per time code, so every error here "
            f"carries a floor of roughly {ATTRIBUTION_FLOOR_MM[0]}-"
            f"{ATTRIBUTION_FLOOR_MM[1]} mm; differences smaller than that mean "
            "nothing",
            "Isaac GT is a skeleton pivot and BODY_38 is a fitted skeleton "
            "joint, both inside the body, while BlazePose landmarks are on the "
            "surface. Raw error therefore flatters BODY_38 for reasons that "
            "are definitional, not algorithmic; the bias-removed columns are "
            "what compare random error",
            "single scene, single subject, single camera, synthetic imagery; "
            "nothing here generalises to real humans, clothing or multi-person",
            "the protocol is not frozen, so no number here is a registered "
            "result",
        ],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    print(f"\nwrote {relative(report_path)}")
    print(
        "\n{:<12} {:>9} {:>12} {:>14} {:>14}".format(
            "arm", "coverage", "n(test)", "raw mean mm", "debiased mm"
        )
    )
    for arm, result in arms.items():
        primary_raw = result["right_arm_two_joint_raw"]
        primary_fixed = result["right_arm_two_joint_bias_removed"]
        print(
            "{:<12} {:>9.4f} {:>12} {:>14} {:>14}".format(
                arm,
                result["coverage"]["valid_fraction"],
                primary_raw.get("count", 0),
                primary_raw.get("mean_mm", "-"),
                primary_fixed.get("mean_mm", "-"),
            )
        )
    print(
        "\nprimary endpoint: right_arm_two_joint_mean_position_error_mm "
        f"(floor {ATTRIBUTION_FLOOR_MM[0]}-{ATTRIBUTION_FLOOR_MM[1]} mm)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
