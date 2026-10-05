#!/usr/bin/env python3
"""How good is the ZED SDK's stereo depth on the Isaac stream?

This is the dominant unknown left by the first GT comparison. Arm A1, which
samples Isaac's rendered metric depth, reached a right-arm two-joint error of
40.12 mm - within 7% of the value frozen in agents.md 5.8. Arm A2, identical
except that its depth comes from the SDK's stereo matching of the streamed
synthetic images, reached 211 to 289 mm. The depth source is the only
substantive difference between them, yet its quality has never been measured.

The comparison here is deliberately three-way, because the total error against
ground truth mixes two very different things:

    rendered_depth - gt_forward   definitional: rendered depth is the visible
                                  surface, GT is the skeleton pivot inside the
                                  body. A1 measured this at -132 mm on the
                                  pelvis and -26 mm on a wrist.
    sdk_depth - rendered_depth    the SDK stereo error proper, which is what
                                  this tool is for.
    sdk_depth - gt_forward        the total, already reported.

Both depths are sampled at the *same pixel* with the *same* median_7x7 window,
so the middle quantity isolates stereo matching from landmark placement and
from the surface-versus-pivot offset.

One caveat is measured rather than assumed. The captured frames are matched to
a Pass 1 time code by NCC, which resolves to about two or three time codes.
The pixel is fixed while the limb moves, so on a thin limb the rendered depth
at that pixel can change between neighbouring time codes even though the pixel
did not. Every joint therefore also reports how much the rendered depth varies
over the attributed time code plus or minus two: where that spread is small the
attribution does not matter, and where it is large the number for that joint is
not to be trusted.

Nothing here is an accuracy claim about either pipeline. It measures one
sensor-modelling gap on synthetic imagery, on one scene.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import depth_sampling_core  # noqa: E402
import analyse_body38_gt_comparison_v1 as comparison  # noqa: E402


WORKSPACE = Path(__file__).resolve().parents[1]
UNIT_SCALE = 100.0
SAMPLER = "median_7x7"
ATTRIBUTION_SENSITIVITY = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def summarise(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    median = statistics.median(ordered)
    index_p05 = max(0, int(round(0.05 * (len(ordered) - 1))))
    index_p95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(ordered),
        "median_mm": round(median, 2),
        "mad_mm": round(
            statistics.median([abs(v - median) for v in ordered]), 2
        ),
        "p05_mm": round(ordered[index_p05], 2),
        "p95_mm": round(ordered[index_p95], 2),
        "min_mm": round(ordered[0], 2),
        "max_mm": round(ordered[-1], 2),
    }


def sample_rendered(depth: np.ndarray, pixel_x: int, pixel_y: int) -> float | None:
    candidate = {
        "canonical_joint": "probe",
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "in_image": True,
        "visibility": 1.0,
    }
    result = depth_sampling_core.sample_candidates(depth, [candidate], SAMPLER)[0]
    if not result.get("valid"):
        return None
    return float(result["depth_m"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--arm", default="A2_median_7x7")
    parser.add_argument(
        "--min-margin", type=float, default=comparison.DEFAULT_MIN_MARGIN
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    capture_dir = args.capture_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    report_path = capture_dir / "sdk_stereo_depth_quality.json"
    if report_path.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite {relative(report_path)}; pass --overwrite"
        )

    truth, unit_check = comparison.load_ground_truth(gt_dir)
    attributed, attribution_stats = comparison.load_attribution(
        capture_dir, args.min_margin
    )
    keys = sorted(attributed)
    print(
        "ground truth {} time codes, pelvis {} m; {} attributed frames".format(
            unit_check["time_codes"],
            unit_check["pelvis_forward_median_m"],
            attribution_stats["attributed_frames"],
        )
    )

    depth_files = {
        int(row["usd_time_code"]): gt_dir / "rgbd_frames" / row["depth_file"]
        for row in comparison.read_csv(gt_dir / "rgbd_manifest.csv")
    }

    # Group by time code so each rendered depth frame is loaded once. Holding
    # all 240 would cost about half a gigabyte.
    by_time_code: dict[int, list[dict]] = defaultdict(list)
    invalid_sdk: dict[str, int] = defaultdict(int)
    slots: dict[str, int] = defaultdict(int)
    for row in comparison.read_csv(capture_dir / "blazepose_joints.csv"):
        if row["arm"] != args.arm:
            continue
        time_code = comparison.nearest_time_code(
            keys, attributed, int(row["image_timestamp_ns"])
        )
        if time_code is None or time_code not in depth_files:
            continue
        slots[row["canonical_joint"]] += 1
        if row["valid"] != "1":
            invalid_sdk[row["canonical_joint"]] += 1
            continue
        by_time_code[time_code].append(row)

    sdk_minus_rendered: dict[str, list[float]] = defaultdict(list)
    rendered_minus_gt: dict[str, list[float]] = defaultdict(list)
    sdk_minus_gt: dict[str, list[float]] = defaultdict(list)
    rendered_spread: dict[str, list[float]] = defaultdict(list)
    rendered_invalid: dict[str, int] = defaultdict(int)

    cache: dict[int, np.ndarray] = {}

    def rendered_depth(time_code: int) -> np.ndarray | None:
        if time_code not in depth_files:
            return None
        if time_code not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[time_code] = (
                np.load(depth_files[time_code]).astype(np.float64) * UNIT_SCALE
            )
        return cache[time_code]

    for index, (time_code, rows) in enumerate(sorted(by_time_code.items())):
        frame = rendered_depth(time_code)
        if frame is None:
            continue
        for row in rows:
            joint = row["canonical_joint"]
            pixel_x = int(round(float(row["pixel_x"])))
            pixel_y = int(round(float(row["pixel_y"])))
            reference = sample_rendered(frame, pixel_x, pixel_y)
            if reference is None:
                rendered_invalid[joint] += 1
                continue
            sdk = float(row["depth_m"])
            sdk_minus_rendered[joint].append((sdk - reference) * 1000.0)

            gt_point = truth.get(time_code, {}).get(joint)
            if gt_point is not None:
                rendered_minus_gt[joint].append(
                    (reference - gt_point[0]) * 1000.0
                )
                sdk_minus_gt[joint].append((sdk - gt_point[0]) * 1000.0)

            # How much does the rendered depth at this fixed pixel move if the
            # attributed time code is off by a couple? Where this is large the
            # joint's number is dominated by attribution, not by the sensor.
            neighbours = []
            for offset in range(
                -ATTRIBUTION_SENSITIVITY, ATTRIBUTION_SENSITIVITY + 1
            ):
                neighbour = rendered_depth(time_code + offset)
                if neighbour is None:
                    continue
                value = sample_rendered(neighbour, pixel_x, pixel_y)
                if value is not None:
                    neighbours.append(value)
            if len(neighbours) >= 2:
                rendered_spread[joint].append(
                    (max(neighbours) - min(neighbours)) * 1000.0
                )
        if (index + 1) % 40 == 0:
            print(f"  {index + 1} time codes", flush=True)

    joints = sorted(sdk_minus_rendered)
    per_joint = {}
    for joint in joints:
        per_joint[joint] = {
            "sdk_minus_rendered": summarise(sdk_minus_rendered[joint]),
            "rendered_minus_gt": summarise(rendered_minus_gt[joint]),
            "sdk_minus_gt": summarise(sdk_minus_gt[joint]),
            "rendered_spread_over_attribution_window": summarise(
                rendered_spread[joint]
            ),
            "sdk_invalid_samples": invalid_sdk.get(joint, 0),
            "sdk_valid_fraction": round(
                1.0 - invalid_sdk.get(joint, 0) / slots[joint], 6
            )
            if slots.get(joint)
            else 0.0,
            "rendered_invalid_samples": rendered_invalid.get(joint, 0),
        }

    pooled = [value for joint in joints for value in sdk_minus_rendered[joint]]
    pooled_abs = [abs(value) for value in pooled]

    payload = {
        "schema_version": 1,
        "stage": "sdk_stereo_depth_quality",
        "classification": "engineering_result_protocol_not_frozen",
        "generated_utc": utc_now(),
        "environment": {"platform": platform.platform()},
        "inputs": {
            "capture_dir": relative(capture_dir),
            "ground_truth_dir": relative(gt_dir),
            "arm": args.arm,
            "sampler": SAMPLER,
        },
        "unit_check": unit_check,
        "attribution": attribution_stats,
        "attribution_sensitivity_window": ATTRIBUTION_SENSITIVITY,
        "pooled_sdk_minus_rendered": summarise(pooled),
        "pooled_absolute_sdk_minus_rendered": summarise(pooled_abs),
        "per_joint": per_joint,
        "decomposition": (
            "rendered_minus_gt is definitional - rendered depth is the visible "
            "surface and GT is the skeleton pivot. sdk_minus_rendered is the "
            "stereo error proper, both sampled at the same pixel with the same "
            "median_7x7 window. sdk_minus_gt is their sum and is what the "
            "first comparison reported."
        ),
        "limits": [
            "one scene, one subject, one camera, synthetic imagery",
            "frames are matched to a time code by NCC to within about two or "
            "three time codes; joints whose "
            "rendered_spread_over_attribution_window is large are dominated by "
            "that, not by the sensor",
            "this measures a sensor-modelling gap, not the accuracy of either "
            "pipeline",
            "the protocol is not frozen, so nothing here is a registered result",
        ],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nwrote {relative(report_path)}")
    print(
        "\n{:<16} {:>10} {:>10} {:>10} {:>12} {:>9}".format(
            "joint", "SDK-rend", "rend-GT", "SDK-GT", "attr spread", "SDK ok"
        )
    )
    for joint in joints:
        entry = per_joint[joint]
        print(
            "{:<16} {:>10} {:>10} {:>10} {:>12} {:>9.4f}".format(
                joint,
                entry["sdk_minus_rendered"].get("median_mm", "-"),
                entry["rendered_minus_gt"].get("median_mm", "-"),
                entry["sdk_minus_gt"].get("median_mm", "-"),
                entry["rendered_spread_over_attribution_window"].get(
                    "median_mm", "-"
                ),
                entry["sdk_valid_fraction"],
            )
        )
    print(f"\npooled |SDK - rendered|: {payload['pooled_absolute_sdk_minus_rendered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
