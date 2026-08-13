from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "tools") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "tools"))

import timecode_marker_v1 as marker


BBOX = (10, 10, 10 + marker.CELL_COUNT * 12, 40)
SHAPE = (600, 960)


class EncodeTests(unittest.TestCase):
    def test_bit_order_is_little_endian(self) -> None:
        self.assertEqual(marker.encode_bits(1)[:3], [1, 0, 0])
        self.assertEqual(marker.encode_bits(2)[:3], [0, 1, 0])
        self.assertEqual(marker.encode_bits(4)[:3], [0, 0, 1])

    def test_parity_is_even_over_the_data_bits(self) -> None:
        for value in (0, 1, 3, 7, 129, 239, 255):
            bits = marker.encode_bits(value)
            self.assertEqual(sum(bits[: marker.DATA_BITS]) % 2, bits[-1])

    def test_range_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            marker.encode_bits(-1)
        with self.assertRaises(ValueError):
            marker.encode_bits(marker.MAX_TIME_CODE + 1)

    def test_covers_the_whole_animation(self) -> None:
        self.assertGreaterEqual(marker.MAX_TIME_CODE, 239)


class RoundTripTests(unittest.TestCase):
    def test_every_animation_time_code_survives_a_round_trip(self) -> None:
        for value in range(240):
            gray = marker.render_reference_strip(value, BBOX, SHAPE)
            outcome = marker.decode(gray, BBOX)
            self.assertEqual(outcome["time_code"], value, outcome)

    def test_decoding_is_invariant_to_exposure(self) -> None:
        """Thresholding against the in-frame references is the whole point."""
        gray = marker.render_reference_strip(137, BBOX, SHAPE)
        for gain, offset in ((0.5, 0.0), (1.4, 10.0), (0.7, 40.0)):
            outcome = marker.decode(np.clip(gray * gain + offset, 0, 255), BBOX)
            self.assertEqual(outcome["time_code"], 137, (gain, offset, outcome))

    def test_decoding_survives_moderate_noise(self) -> None:
        rng = np.random.default_rng(0)
        gray = marker.render_reference_strip(200, BBOX, SHAPE)
        noisy = gray + rng.normal(0.0, 8.0, size=gray.shape)
        self.assertEqual(marker.decode(noisy, BBOX)["time_code"], 200)


class RefusalTests(unittest.TestCase):
    """A wrong time code silently pairs a frame with the wrong ground truth,
    so every ambiguous case must refuse rather than guess."""

    def test_missing_marker_is_refused_not_guessed(self) -> None:
        gray = np.full(SHAPE, 90.0, dtype=np.float32)
        outcome = marker.decode(gray, BBOX)
        self.assertIsNone(outcome["time_code"])
        self.assertEqual(outcome["reason"], "low_reference_contrast")

    def test_a_bit_near_the_midpoint_is_refused(self) -> None:
        gray = marker.render_reference_strip(64, BBOX, SHAPE)
        left, top, right, bottom = marker.cell_bounds(
            BBOX, marker.FIRST_DATA_CELL + 3
        )
        gray[top:bottom, left:right] = 128.0
        outcome = marker.decode(gray, BBOX)
        self.assertIsNone(outcome["time_code"])
        self.assertEqual(outcome["reason"], "ambiguous_bit")

    def test_a_flipped_bit_is_caught_by_parity(self) -> None:
        gray = marker.render_reference_strip(100, BBOX, SHAPE)
        left, top, right, bottom = marker.cell_bounds(
            BBOX, marker.FIRST_DATA_CELL + 2
        )
        current = gray[top + 1, left + 1]
        gray[top:bottom, left:right] = 20.0 if current > 128 else 235.0
        outcome = marker.decode(gray, BBOX)
        self.assertIsNone(outcome["time_code"])
        self.assertEqual(outcome["reason"], "parity_mismatch")

    def test_a_value_beyond_the_animation_is_refused(self) -> None:
        gray = marker.render_reference_strip(250, BBOX, SHAPE)
        outcome = marker.decode(gray, BBOX, max_time_code=239)
        self.assertIsNone(outcome["time_code"])
        self.assertEqual(outcome["reason"], "out_of_range")

    def test_a_bbox_off_the_frame_is_refused(self) -> None:
        outcome = marker.decode(
            np.zeros(SHAPE, dtype=np.float32), (900, 10, 1200, 40)
        )
        self.assertEqual(outcome["reason"], "bbox_outside_frame")

    def test_inverted_references_are_refused(self) -> None:
        """A strip read upside down must not decode as something."""
        gray = marker.render_reference_strip(77, BBOX, SHAPE)
        gray = 255.0 - gray
        outcome = marker.decode(gray, BBOX)
        self.assertIsNone(outcome["time_code"])


class PlacementTests(unittest.TestCase):
    def test_cells_tile_the_strip_without_gaps(self) -> None:
        edges = [
            marker.cell_bounds(BBOX, index)
            for index in range(marker.CELL_COUNT)
        ]
        self.assertEqual(edges[0][0], BBOX[0])
        self.assertEqual(edges[-1][2], BBOX[2])
        for earlier, later in zip(edges, edges[1:]):
            self.assertEqual(earlier[2], later[0])

    def test_too_narrow_a_strip_raises(self) -> None:
        with self.assertRaises(ValueError):
            marker.cell_bounds((0, 0, 5, 10), 0)

    def test_strip_sits_outside_the_roi_065_crop(self) -> None:
        """BlazePose only ever sees the ROI crop, so the marker must not
        intrude on it; agents.md 3.2 freezes ROI 0.65."""
        roi_left, roi_top = 168, 105
        self.assertTrue(BBOX[2] <= roi_left or BBOX[3] <= roi_top)


if __name__ == "__main__":
    unittest.main()
