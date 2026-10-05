#!/usr/bin/env python3
"""Assign a time code to every captured frame by NCC against the Pass 1 exports.

The capture cannot hold a pose: the ZED streamer publishes only while the Isaac
timeline plays (plan document 2.2d). So Pass 2 records whatever frames arrive
while the animation runs, and this tool decides afterwards which time code each
frame actually shows, by matching it against the Pass 1 RGB exported at every
time code with the timeline paused.

This is the registered primary join criterion. The producer's beacons are
advisory only - they narrow the search window and cross-check the answer, they
never decide it.

It is deliberately **not** a nearest-neighbour match. A frame is attributed only
when all of these hold, and is dropped otherwise rather than falling back:

* ``margin`` - the best correlation beats the runner-up outside the winner's
  neighbourhood by at least ``--min-margin``. Adjacent time codes look alike, so
  the runner-up is taken from outside a small exclusion band around the winner;
  otherwise the neighbour would always suppress the margin and nothing would
  ever pass.
* ``absolute`` - the best correlation itself clears ``--min-correlation``. Pass 2
  frames went through the streamer's lossy encode while Pass 1 frames are clean
  renders, so this threshold has to be calibrated on real data, never assumed.
* ``monotone`` - attributed time codes never decrease along the capture inside
  one playback pass. Playback is one-directional; a backwards jump means the
  match is wrong, not that time went backwards.
* ``beacon`` - the attributed time code sits within ``--beacon-window`` of the
  beacon the consumer recorded for that frame.

Looping playback breaks strict monotonicity at each wrap, so the monotone check
runs inside detected passes rather than across the whole capture.

Correlation is zero-mean normalised cross-correlation on grayscale, downsampled
by ``--downsample``, which makes 240 candidates per frame cheap while keeping
the arm pose - the thing that actually differs between time codes - resolvable.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]

DEFAULT_MIN_CORRELATION = 0.90
DEFAULT_MIN_MARGIN = 0.01
DEFAULT_EXCLUSION_RADIUS = 3
DEFAULT_BEACON_WINDOW = 30
DEFAULT_DOWNSAMPLE = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"unsupported image shape: {array.shape}")
    channels = array[:, :, :3].astype(np.float32)
    return channels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def prepare(image: np.ndarray, downsample: int) -> np.ndarray:
    """Grayscale, downsample, then zero-mean and unit-norm.

    Normalising once per image turns every later correlation into a single dot
    product, which is what makes an exhaustive 240-candidate search affordable.
    """
    gray = to_grayscale(image)
    if downsample > 1:
        gray = gray[::downsample, ::downsample]
    centred = gray - gray.mean()
    norm = float(np.linalg.norm(centred))
    if norm == 0.0:
        return np.zeros_like(centred)
    return centred / norm


def correlate(query: np.ndarray, reference: np.ndarray) -> float:
    if query.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: query {query.shape} vs reference {reference.shape}"
        )
    return float(np.tensordot(query, reference, axes=query.ndim))


def best_match(
    query: np.ndarray,
    references: dict[int, np.ndarray],
    candidate_time_codes: list[int],
    exclusion_radius: int,
) -> dict[str, Any]:
    """Best candidate plus the runner-up from outside the winner's band."""
    scores = {
        time_code: correlate(query, references[time_code])
        for time_code in candidate_time_codes
        if time_code in references
    }
    if not scores:
        return {"matched": False, "reason": "no_candidate_in_window"}
    winner = max(scores, key=lambda key: scores[key])
    outside = [
        value
        for time_code, value in scores.items()
        if abs(time_code - winner) > exclusion_radius
    ]
    runner_up = max(outside) if outside else None
    return {
        "matched": True,
        "time_code": winner,
        "correlation": scores[winner],
        "runner_up_correlation": runner_up,
        "margin": (scores[winner] - runner_up) if runner_up is not None else None,
        "candidates_scored": len(scores),
    }


def attribute(
    frames: list[dict],
    references: dict[int, np.ndarray],
    min_correlation: float,
    min_margin: float,
    exclusion_radius: int,
    beacon_window: int,
) -> list[dict]:
    """Attribute each frame, then enforce monotonicity inside each pass.

    ``frames`` is a list of {"frame_index", "beacon_time_code", "prepared"}.
    """
    reference_codes = sorted(references)
    results: list[dict] = []
    for frame in frames:
        beacon = frame.get("beacon_time_code")
        if beacon is not None and beacon >= 0 and beacon_window > 0:
            window = [
                code
                for code in reference_codes
                if abs(code - beacon) <= beacon_window
            ]
        else:
            window = reference_codes
        outcome = best_match(
            frame["prepared"], references, window, exclusion_radius
        )
        record = {
            "frame_index": frame["frame_index"],
            "beacon_time_code": beacon,
            **outcome,
        }
        reasons = []
        if not outcome.get("matched"):
            reasons.append(outcome.get("reason", "no_match"))
        else:
            if outcome["correlation"] < min_correlation:
                reasons.append("correlation_below_threshold")
            if outcome["margin"] is None:
                reasons.append("no_runner_up_outside_exclusion_band")
            elif outcome["margin"] < min_margin:
                reasons.append("margin_below_threshold")
            if (
                beacon is not None
                and beacon >= 0
                and abs(outcome["time_code"] - beacon) > beacon_window
            ):
                reasons.append("outside_beacon_window")
        record["reject_reasons"] = reasons
        record["accepted"] = not reasons
        results.append(record)

    _enforce_monotonicity(results)
    return results


def _enforce_monotonicity(results: list[dict]) -> None:
    """Drop frames that go backwards without a plausible loop wrap.

    Playback is one-directional, so a decrease is either a loop wrap or a bad
    match. A decrease that lands near the start of the range after a value near
    the end is a wrap; anything else is rejected.
    """
    previous = None
    for record in results:
        if not record["accepted"]:
            continue
        current = record["time_code"]
        if previous is not None and current < previous:
            wrapped = current < previous / 2.0
            if not wrapped:
                record["accepted"] = False
                record["reject_reasons"].append("non_monotone_time_code")
                continue
            record["loop_wrap"] = True
        previous = current


def load_reference_images(directory: Path) -> tuple[dict[int, np.ndarray], str]:
    """Load Pass 1 exports keyed by time code.

    ``rgbd_manifest.csv`` is preferred when present: it states the time code of
    every exported frame, so the mapping is read rather than inferred. Filename
    parsing is only the fallback, and it is a guess - it assumes the trailing
    integer in the stem is the time code, which happens to hold for the frozen
    exporter's ``rgb_0042.npy`` but is not guaranteed by anything.
    """
    manifest = _find_manifest(directory)
    if manifest is not None:
        return _load_from_manifest(manifest), relative(manifest)
    return _load_by_filename(directory), "filename_trailing_integer"


def _find_manifest(directory: Path) -> Path | None:
    for candidate in (
        directory / "rgbd_manifest.csv",
        directory.parent / "rgbd_manifest.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def _load_from_manifest(manifest: Path) -> dict[int, np.ndarray]:
    import csv  # noqa: PLC0415

    root = manifest.parent
    references: dict[int, np.ndarray] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            time_code = int(row["usd_time_code"])
            if time_code in references:
                raise SystemExit(
                    f"{relative(manifest)} lists time code {time_code} twice; "
                    "the reference set must be one frame per time code"
                )
            path = root / "rgbd_frames" / row["rgb_file"]
            if not path.exists():
                path = root / row["rgb_file"]
            references[time_code] = _read_image(path)
    return references


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import cv2  # noqa: PLC0415

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {relative(path)}")
    return image[:, :, ::-1]


def _load_by_filename(directory: Path) -> dict[int, np.ndarray]:
    import re  # noqa: PLC0415

    references: dict[int, np.ndarray] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = re.search(r"(\d+)$", path.stem)
        if match is None:
            continue
        time_code = int(match.group(1))
        if path.suffix.lower() == ".npy":
            image = np.load(path)
        else:
            import cv2  # noqa: PLC0415

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = image[:, :, ::-1]
        references[time_code] = image
    return references


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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    import csv  # noqa: PLC0415

    args = parse_args()
    capture_dir = args.capture_dir.resolve()
    report_path = capture_dir / "ncc_attribution.json"
    csv_path = capture_dir / "frame_time_code_attribution.csv"
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
    references = {
        code: prepare(image, args.downsample)
        for code, image in references_raw.items()
    }
    print(f"loaded {len(references)} reference time codes")

    svo_path = args.svo or next(iter(capture_dir.glob("*.svo2")), None)
    if svo_path is None:
        raise SystemExit(f"no SVO found in {relative(capture_dir)}")

    # Beacons are joined to SVO frames by timestamp, never by position. The
    # recorder writes frames the scoring loop never saw - recording leads the
    # first scored frame and flushes past the last - so SVO frame i and
    # capture_frames row i are different frames. The join tolerance is one
    # microsecond, the quantisation step verified in P0c: the live stream
    # reports nanoseconds while the SVO stores whole microseconds.
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
        import bisect  # noqa: PLC0415

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
            frames.append(
                {
                    "frame_index": index,
                    "image_timestamp_ns": timestamp_ns,
                    "beacon_time_code": beacon_for(timestamp_ns),
                    "scored": beacon_for(timestamp_ns) is not None,
                    "prepared": prepare(
                        rgba[:, :, :3][:, :, ::-1], args.downsample
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
        record.pop("prepared", None)

    accepted = [record for record in results if record["accepted"]]
    correlations = [record["correlation"] for record in accepted]
    margins = [
        record["margin"] for record in accepted if record["margin"] is not None
    ]
    reasons: dict[str, int] = {}
    for record in results:
        for reason in record["reject_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    fieldnames = [
        "frame_index", "image_timestamp_ns", "joins_a_scored_live_frame",
        "beacon_time_code", "time_code", "correlation",
        "runner_up_correlation", "margin", "candidates_scored", "accepted",
        "reject_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in results:
            row = dict(record)
            row["reject_reasons"] = ";".join(record["reject_reasons"])
            writer.writerow(row)

    payload = {
        "schema_version": 1,
        "stage": "ncc_time_code_attribution",
        "generated_utc": utc_now(),
        "plan_document": "docs/body38_vs_blazepose_gt_comparison_plan_v1.md",
        "environment": {"platform": platform.platform()},
        "inputs": {
            "capture_dir": relative(capture_dir),
            "svo": relative(svo_path),
            "reference_dir": relative(args.reference_dir.resolve()),
            "reference_time_codes": len(references),
            "reference_time_code_source": reference_source,
        },
        "thresholds": {
            "min_correlation": args.min_correlation,
            "min_margin": args.min_margin,
            "exclusion_radius": args.exclusion_radius,
            "beacon_window": args.beacon_window,
            "downsample": args.downsample,
        },
        "frames": len(results),
        "frames_joining_a_scored_live_frame": matched_to_scored,
        "accepted_among_scored": sum(
            1
            for record in results
            if record["accepted"] and record["joins_a_scored_live_frame"]
        ),
        "accepted": len(accepted),
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
        "reject_reason_counts": reasons,
        "join_rule": (
            "registered primary criterion: unique NCC argmax with a margin "
            "over the best candidate outside the exclusion band, monotone "
            "within a playback pass, inside the beacon window. Frames failing "
            "any condition are dropped, never nearest-neighbour matched."
        ),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {relative(report_path)}")
    print(
        "  accepted {}/{} ({}), distinct time codes {}".format(
            payload["accepted"],
            payload["frames"],
            payload["accepted_fraction"],
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
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  rejected {count:>5}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
