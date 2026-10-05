#!/usr/bin/env python3
"""Assign a time code to every captured frame, correlating residuals.

Third measurement of the same registered decision rule. The rule has not
changed; what keeps changing is the feature it is measured on, because each
version showed the previous feature carried almost no signal:

* **v1, whole frame.** Correlations 0.999855-0.999989 across *every* candidate,
  margins 1e-7 to 1e-4, all 1,477 frames rejected. The scene is a static room
  with one moving arm, so whole-frame NCC measures the background that all 240
  time codes share.
* **v2, masked to the top 5% of across-time-code deviation.** The argmax became
  stable and plausible - consecutive frames mapped to consecutive time codes -
  but correlations were still 0.997-0.9994 with a median margin of 0.0025.
  Masking removes static *pixels*; inside the mask the frames still share a
  large common component, because the body occupies that region in every time
  code and only the arm shifts.

v3 removes that common component too. Both references and queries have the
reference stack's per-pixel mean subtracted before normalisation, so what is
correlated is each frame's *deviation from the average pose* rather than the
pose itself. That is the part which actually identifies a time code.

The decision rule is unchanged and is still not a nearest-neighbour join. A
frame is attributed only when all of these hold, and dropped otherwise:

* ``margin`` over the best candidate outside an exclusion band around the
  winner - adjacent time codes look alike, so without the band a neighbour
  would always suppress the margin;
* ``absolute`` correlation above a floor;
* ``monotone`` within a playback pass, with loop wraps detected and allowed;
* inside the ``beacon`` window.

Thresholds are looser than v1's because residual correlation lives on a
different scale from whole-frame correlation - a residual correlation of 0.6 is
a strong match, where a whole-frame 0.999 was noise. They are still thresholds
that reject: the run reports the achieved distributions so they can be set on
evidence before anything is frozen.

Attribution deliberately uses imagery, not landmarks. Matching on BlazePose or
BODY_38 output would attribute frames using the very estimator whose error is
then measured against those frames, which is circular.

v1 and v2 are kept as the record of the two failed features (rule 9.1).
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attribute_capture_frames_ncc_v1 import (  # noqa: E402
    attribute,
    load_reference_images,
)
from attribute_capture_frames_ncc_v2 import (  # noqa: E402
    downsample_grayscale,
)


WORKSPACE = Path(__file__).resolve().parents[1]

DEFAULT_MIN_CORRELATION = 0.30
DEFAULT_MIN_MARGIN = 0.05
DEFAULT_EXCLUSION_RADIUS = 3
DEFAULT_BEACON_WINDOW = 60
DEFAULT_DOWNSAMPLE = 2
DEFAULT_MASK_QUANTILE = 0.10


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_residual_basis(
    references_raw: dict[int, np.ndarray], downsample: int, quantile: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Discriminative mask plus the mean image to subtract.

    Deviation across the reference stack says which pixels carry information;
    the stack mean says what they have in common. Removing the second is what
    separates a residual correlation of 0.6 from a raw correlation of 0.999.
    """
    stack = np.stack(
        [
            downsample_grayscale(references_raw[time_code], downsample)
            for time_code in sorted(references_raw)
        ]
    )
    deviation = stack.std(axis=0)
    mean_image = stack.mean(axis=0)
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
    }
    return mask, mean_image, stats


def prepare_residual(
    image: np.ndarray,
    downsample: int,
    mask: np.ndarray,
    mean_image: np.ndarray,
) -> np.ndarray:
    gray = downsample_grayscale(image, downsample)
    if gray.shape != mask.shape:
        raise SystemExit(
            f"frame shape {gray.shape} does not match the reference mask "
            f"{mask.shape}; Pass 1 and Pass 2 must share a resolution"
        )
    residual = (gray - mean_image)[mask]
    centred = residual - residual.mean()
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
    report_path = capture_dir / "ncc_attribution_v3.json"
    csv_path = capture_dir / "frame_time_code_attribution_v3.csv"
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

    mask, mean_image, mask_stats = build_residual_basis(
        references_raw, args.downsample, args.mask_quantile
    )
    print(
        "residual basis: {} of {} pixels ({:.2%})".format(
            mask_stats["mask_pixels"],
            mask_stats["total_pixels"],
            mask_stats["mask_fraction"],
        )
    )
    references = {
        code: prepare_residual(image, args.downsample, mask, mean_image)
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
                    "prepared": prepare_residual(
                        rgba[:, :, :3][:, :, ::-1],
                        args.downsample,
                        mask,
                        mean_image,
                    ),
                }
            )
            index += 1
    finally:
        camera.close()
    matched_to_scored = sum(1 for frame in frames if frame["scored"])
    print(f"decoded {len(frames)} capture frames from {relative(svo_path)}")
    print(f"  {matched_to_scored} join a scored live frame by timestamp")

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
    all_correlations = [
        record["correlation"] for record in results if record.get("correlation")
    ]
    all_margins = [
        record["margin"] for record in results if record.get("margin")
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

    def summarise(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=float)
        return {
            "count": int(array.size),
            "min": round(float(array.min()), 6),
            "p05": round(float(np.quantile(array, 0.05)), 6),
            "median": round(float(np.median(array)), 6),
            "p95": round(float(np.quantile(array, 0.95)), 6),
            "max": round(float(array.max()), 6),
        }

    payload = {
        "schema_version": 1,
        "stage": "ncc_time_code_attribution_v3_residual",
        "generated_utc": utc_now(),
        "plan_document": "docs/body38_vs_blazepose_gt_comparison_plan_v1.md",
        "environment": {"platform": platform.platform()},
        "supersedes": (
            "v1 whole-frame (margins 1e-7 to 1e-4, all rejected) and v2 masked "
            "(median margin 0.0025, all rejected); both measured a signal the "
            "background dominated"
        ),
        "inputs": {
            "capture_dir": relative(capture_dir),
            "svo": relative(svo_path),
            "reference_dir": relative(args.reference_dir.resolve()),
            "reference_time_codes": len(references),
            "reference_time_code_source": reference_source,
        },
        "residual_basis": mask_stats,
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
        "correlation_all_candidates_best": summarise(all_correlations),
        "margin_all": summarise(all_margins),
        "correlation_accepted": summarise(correlations),
        "margin_accepted": summarise(margins),
        "beacon_minus_attributed": {
            **summarise([float(value) for value in offsets]),
            "note": (
                "the beacon leads the frame by the pipeline latency; this is "
                "the measured lag and it is what the beacon window must admit"
            ),
        },
        "reject_reason_counts": reasons,
        "join_rule": (
            "registered primary criterion: unique NCC argmax over residuals "
            "on the discriminative mask, with a margin over the best candidate "
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
        "  accepted {}/{} ({}), scored {}, distinct time codes {}".format(
            payload["accepted"],
            payload["frames"],
            payload["accepted_fraction"],
            payload["accepted_among_scored"],
            payload["distinct_time_codes_attributed"],
        )
    )
    print(f"  correlation (all best): {json.dumps(payload['correlation_all_candidates_best'])}")
    print(f"  margin (all)          : {json.dumps(payload['margin_all'])}")
    print(f"  beacon lead           : {json.dumps(payload['beacon_minus_attributed'])}")
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  rejected {count:>5}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
