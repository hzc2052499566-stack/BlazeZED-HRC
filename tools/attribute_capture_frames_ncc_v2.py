#!/usr/bin/env python3
"""Assign a time code to every captured frame, matching only what discriminates.

v1 correlated whole frames and failed completely: every one of 1,477 frames was
rejected on margin. The numbers say why - correlations ran 0.999855 to 0.999989
across *every* candidate, with margins of 1e-7 to 1e-4. This scene is a static
room in which one arm moves. Whole-frame NCC is therefore a measurement of the
background, which all 240 time codes share, and the arm contributes too little
to separate them. The argmax was effectively random.

v2 keeps v1's decision rules unchanged and fixes the measurement: it correlates
only over pixels that actually differ between time codes. The reference stack
gives that for free - per-pixel standard deviation across the 240 Pass 1 frames
is high on the moving arm and near zero on the walls - so the mask is derived
from the data rather than drawn by hand.

Everything else is as registered in v1, and this is still not a
nearest-neighbour join. A frame is attributed only when all of these hold, and
is dropped otherwise:

* ``margin`` - the best correlation beats the best candidate outside an
  exclusion band around the winner by at least ``--min-margin``. Adjacent time
  codes look alike, so without the band a neighbour would always suppress the
  margin.
* ``absolute`` - the best correlation clears ``--min-correlation``.
* ``monotone`` - attributed time codes never decrease within a playback pass;
  loop wraps are detected and allowed.
* ``beacon`` - the result sits within ``--beacon-window`` of the beacon the
  consumer recorded for that frame.

The beacon window default is wider than v1's. The beacon says where Isaac was
when it announced; the frame the consumer grabs next is older by the pipeline
latency, and v1's rejected output already showed the offset running around
twenty time codes. The window has to admit that lag, and the measured offset is
reported so it can be tightened on evidence later.

v1 is kept as the record of the failed measurement (rule 9.1).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attribute_capture_frames_ncc_v1 import (  # noqa: E402
    attribute,
    correlate,
    load_reference_images,
    to_grayscale,
)


WORKSPACE = Path(__file__).resolve().parents[1]

DEFAULT_MIN_CORRELATION = 0.50
DEFAULT_MIN_MARGIN = 0.02
DEFAULT_EXCLUSION_RADIUS = 3
DEFAULT_BEACON_WINDOW = 60
DEFAULT_DOWNSAMPLE = 2
# Fraction of pixels kept, taken from the top of the across-time-code standard
# deviation. The moving arm is a small part of the frame, so most pixels carry
# no information about which time code this is.
DEFAULT_MASK_QUANTILE = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def downsample_grayscale(image: np.ndarray, downsample: int) -> np.ndarray:
    gray = to_grayscale(image)
    if downsample > 1:
        gray = gray[::downsample, ::downsample]
    return gray


def build_discriminative_mask(
    references_raw: dict[int, np.ndarray], downsample: int, quantile: float
) -> tuple[np.ndarray, dict]:
    """Pixels whose value varies across time codes.

    Standard deviation over the reference stack is the discriminability of each
    pixel: zero on a wall that never changes, large on the arm. Keeping the top
    ``quantile`` of them is what turns a 1e-5 margin into a usable one.
    """
    stack = np.stack(
        [
            downsample_grayscale(references_raw[time_code], downsample)
            for time_code in sorted(references_raw)
        ]
    )
    deviation = stack.std(axis=0)
    if not np.any(deviation > 0):
        raise SystemExit(
            "the reference frames are identical to each other; there is "
            "nothing to match on"
        )
    threshold = float(np.quantile(deviation, 1.0 - quantile))
    mask = deviation >= max(threshold, 1e-6)
    if mask.sum() < 32:
        raise SystemExit(
            f"the discriminative mask kept only {int(mask.sum())} pixels; "
            "lower --mask-quantile"
        )
    stats = {
        "mask_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
        "mask_fraction": round(float(mask.mean()), 6),
        "deviation_threshold": round(threshold, 6),
        "deviation_max": round(float(deviation.max()), 6),
        "deviation_median": round(float(np.median(deviation)), 6),
    }
    return mask, stats


def prepare_masked(
    image: np.ndarray, downsample: int, mask: np.ndarray
) -> np.ndarray:
    """Grayscale, downsample, keep the discriminative pixels, then normalise."""
    gray = downsample_grayscale(image, downsample)
    if gray.shape != mask.shape:
        raise SystemExit(
            f"frame shape {gray.shape} does not match the reference mask "
            f"{mask.shape}; Pass 1 and Pass 2 must share a resolution"
        )
    values = gray[mask]
    centred = values - values.mean()
    norm = float(np.linalg.norm(centred))
    if norm == 0.0:
        return np.zeros_like(centred)
    return centred / norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--svo", type=Path, default=None)
    parser.add_argument(
        "--min-correlation", type=float, default=DEFAULT_MIN_CORRELATION
    )
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument(
        "--exclusion-radius", type=int, default=DEFAULT_EXCLUSION_RADIUS
    )
    parser.add_argument(
        "--beacon-window", type=int, default=DEFAULT_BEACON_WINDOW
    )
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument(
        "--mask-quantile", type=float, default=DEFAULT_MASK_QUANTILE
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_dir = args.capture_dir.resolve()
    report_path = capture_dir / "ncc_attribution_v2.json"
    csv_path = capture_dir / "frame_time_code_attribution_v2.csv"
    if report_path.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite {relative(report_path)}; pass --overwrite"
        )

    references_raw, reference_source = load_reference_images(
        args.reference_dir.resolve()
    )
    if not references_raw:
        raise SystemExit(f"no reference images under {args.reference_dir}")
    print(f"time-code source: {reference_source}")
    print(f"loaded {len(references_raw)} reference time codes")

    mask, mask_stats = build_discriminative_mask(
        references_raw, args.downsample, args.mask_quantile
    )
    print(
        "discriminative mask: {} of {} pixels ({:.2%}), deviation threshold "
        "{}".format(
            mask_stats["mask_pixels"],
            mask_stats["total_pixels"],
            mask_stats["mask_fraction"],
            mask_stats["deviation_threshold"],
        )
    )
    references = {
        code: prepare_masked(image, args.downsample, mask)
        for code, image in references_raw.items()
    }

    svo_path = args.svo or next(iter(capture_dir.glob("*.svo2")), None)
    if svo_path is None:
        raise SystemExit(f"no SVO found in {relative(capture_dir)}")

    beacons_by_timestamp: dict[int, int] = {}
    frames_csv = capture_dir / "capture_frames.csv"
    if frames_csv.exists():
        with frames_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                beacons_by_timestamp[int(row["image_timestamp_ns"])] = int(
                    row["time_code"]
                )
    beacon_keys = sorted(beacons_by_timestamp)

    def beacon_for(timestamp_ns: int) -> int | None:
        if not beacon_keys:
            return None
        position = bisect.bisect_left(beacon_keys, timestamp_ns)
        for index in (position - 1, position):
            if 0 <= index < len(beacon_keys):
                key = beacon_keys[index]
                if abs(key - timestamp_ns) < 1000:
                    return beacons_by_timestamp[key]
        return None

    import pyzed.sl as sl  # noqa: PLC0415

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    camera = sl.Camera()
    if camera.open(init) != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"failed to open {svo_path}")
    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()

    frames: list[dict] = []
    try:
        index = 0
        while camera.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            camera.retrieve_image(image_mat, sl.VIEW.LEFT)
            rgba = image_mat.get_data()
            timestamp_ns = int(
                camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            )
            beacon = beacon_for(timestamp_ns)
            frames.append(
                {
                    "frame_index": index,
                    "image_timestamp_ns": timestamp_ns,
                    "beacon_time_code": beacon,
                    "scored": beacon is not None,
                    "prepared": prepare_masked(
                        rgba[:, :, :3][:, :, ::-1], args.downsample, mask
                    ),
                }
            )
            index += 1
    finally:
        camera.close()
    matched_to_scored = sum(1 for frame in frames if frame["scored"])
    print(f"decoded {len(frames)} capture frames from {relative(svo_path)}")
    print(
        "  {} of them join a scored live frame by timestamp; the rest are "
        "recorder lead-in and flush".format(matched_to_scored)
    )

    results = attribute(
        frames,
        references,
        args.min_correlation,
        args.min_margin,
        args.exclusion_radius,
        args.beacon_window,
    )
    for record, frame in zip(results, frames):
        record["image_timestamp_ns"] = frame["image_timestamp_ns"]
        record["joins_a_scored_live_frame"] = int(frame["scored"])
        if (
            record.get("time_code") is not None
            and frame["beacon_time_code"] is not None
        ):
            record["beacon_minus_attributed"] = (
                frame["beacon_time_code"] - record["time_code"]
            )
        record.pop("prepared", None)

    accepted = [record for record in results if record["accepted"]]
    correlations = [record["correlation"] for record in accepted]
    margins = [
        record["margin"] for record in accepted if record["margin"] is not None
    ]
    offsets = [
        record["beacon_minus_attributed"]
        for record in accepted
        if record.get("beacon_minus_attributed") is not None
    ]
    reasons: dict[str, int] = {}
    for record in results:
        for reason in record["reject_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    fieldnames = [
        "frame_index", "image_timestamp_ns", "joins_a_scored_live_frame",
        "beacon_time_code", "time_code", "beacon_minus_attributed",
        "correlation", "runner_up_correlation", "margin", "candidates_scored",
        "accepted", "reject_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for record in results:
            row = dict(record)
            row["reject_reasons"] = ";".join(record["reject_reasons"])
            writer.writerow(row)

    payload = {
        "schema_version": 1,
        "stage": "ncc_time_code_attribution_v2_masked",
        "generated_utc": utc_now(),
        "plan_document": "docs/body38_vs_blazepose_gt_comparison_plan_v1.md",
        "environment": {"platform": platform.platform()},
        "supersedes": (
            "attribute_capture_frames_ncc_v1, whose whole-frame correlations "
            "ran 0.999855-0.999989 across every candidate and rejected all "
            "1,477 frames on margin; the background dominated the measurement"
        ),
        "inputs": {
            "capture_dir": relative(capture_dir),
            "svo": relative(svo_path),
            "reference_dir": relative(args.reference_dir.resolve()),
            "reference_time_codes": len(references),
            "reference_time_code_source": reference_source,
        },
        "discriminative_mask": mask_stats,
        "thresholds": {
            "min_correlation": args.min_correlation,
            "min_margin": args.min_margin,
            "exclusion_radius": args.exclusion_radius,
            "beacon_window": args.beacon_window,
            "downsample": args.downsample,
            "mask_quantile": args.mask_quantile,
        },
        "frames": len(results),
        "frames_joining_a_scored_live_frame": matched_to_scored,
        "accepted": len(accepted),
        "accepted_among_scored": sum(
            1
            for record in results
            if record["accepted"] and record["joins_a_scored_live_frame"]
        ),
        "accepted_fraction": round(len(accepted) / len(results), 6)
        if results
        else 0.0,
        "distinct_time_codes_attributed": len(
            {record["time_code"] for record in accepted}
        ),
        "correlation_min": min(correlations) if correlations else None,
        "correlation_median": float(np.median(correlations))
        if correlations
        else None,
        "margin_min": min(margins) if margins else None,
        "margin_median": float(np.median(margins)) if margins else None,
        "beacon_minus_attributed": {
            "count": len(offsets),
            "median": float(np.median(offsets)) if offsets else None,
            "p05": float(np.quantile(offsets, 0.05)) if offsets else None,
            "p95": float(np.quantile(offsets, 0.95)) if offsets else None,
            "note": (
                "the beacon leads the frame by the pipeline latency; this is "
                "the measured lag, and it is what the beacon window has to "
                "admit"
            ),
        },
        "reject_reason_counts": reasons,
        "join_rule": (
            "registered primary criterion: unique NCC argmax over the "
            "discriminative mask, with a margin over the best candidate "
            "outside the exclusion band, monotone within a playback pass, "
            "inside the beacon window. Frames failing any condition are "
            "dropped, never nearest-neighbour matched."
        ),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {relative(report_path)}")
    print(
        "  accepted {}/{} ({}), of which scored {}, distinct time codes {}".format(
            payload["accepted"],
            payload["frames"],
            payload["accepted_fraction"],
            payload["accepted_among_scored"],
            payload["distinct_time_codes_attributed"],
        )
    )
    print(
        "  correlation min/median {} / {} | margin min/median {} / {}".format(
            payload["correlation_min"],
            payload["correlation_median"],
            payload["margin_min"],
            payload["margin_median"],
        )
    )
    print(f"  beacon lead: {json.dumps(payload['beacon_minus_attributed'])}")
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  rejected {count:>5}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
