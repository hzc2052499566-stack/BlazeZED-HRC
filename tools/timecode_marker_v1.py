"""Frame-embedded time-code marker: layout, encoding and decoding.

NCC attribution works but its resolution is about +/- 2 to 3 time codes, and
the ground truth moves 3.29 mm per time code at the right wrist (median, p95
5.73). That is a 7 to 10 mm floor on a comparison whose frozen effect sizes are
32.6 to 43.2 mm - 20 to 30% of the thing being measured. A marker rendered into
the frame removes the floor entirely: each frame then states which time code it
shows.

Layout, a horizontal strip of equal cells:

    [ WHITE ][ BLACK ][ b0 ][ b1 ] ... [ b7 ][ PARITY ]

* **WHITE and BLACK are always on and always off.** They are not decoration:
  they are the per-frame reference for thresholding, which makes decoding
  independent of exposure, lighting and the streamer's lossy encode. A fixed
  grey threshold would not survive any of those.
* **b0..b7** carry the time code, least significant bit first, so 0..255
  covers the 0..239 range with room to spare.
* **PARITY** is even parity over b0..b7. A misdecoded time code is worse than
  no time code, because it silently pairs a frame with the wrong ground truth;
  parity turns most single-bit errors into a refusal instead.

A frame is decoded only when the white reference is brighter than the black
one by ``min_reference_contrast``, every bit sits clear of the midpoint by
``min_bit_margin`` of that contrast, parity checks, and the value is within the
animation range. Anything else returns no time code rather than a guess.

The strip is placed outside the centre ROI 0.65 crop, so BlazePose never sees
it at all. BODY_38 does see the full frame, which is why the capture has to run
an A/B against a marker-free run before any of it is trusted.
"""

from __future__ import annotations

from typing import Any

import numpy as np


MARKER_VERSION = "timecode_marker_v1"
DATA_BITS = 8
# WHITE, BLACK, eight data bits, parity.
CELL_COUNT = 2 + DATA_BITS + 1
WHITE_CELL = 0
BLACK_CELL = 1
FIRST_DATA_CELL = 2
PARITY_CELL = CELL_COUNT - 1

MAX_TIME_CODE = (1 << DATA_BITS) - 1

DEFAULT_MIN_REFERENCE_CONTRAST = 40.0
DEFAULT_MIN_BIT_MARGIN = 0.25
# Fraction of a cell sampled at its centre, avoiding edges where the renderer
# and the stream's lossy encode both blur across cell boundaries.
DEFAULT_SAMPLE_FRACTION = 0.5


def encode_bits(time_code: int) -> list[int]:
    """Bit pattern for one time code: b0..b7 little-endian, then parity."""
    if not 0 <= time_code <= MAX_TIME_CODE:
        raise ValueError(
            f"time code {time_code} is outside 0..{MAX_TIME_CODE}"
        )
    bits = [(time_code >> index) & 1 for index in range(DATA_BITS)]
    bits.append(sum(bits) % 2)
    return bits


def cell_bounds(
    bbox: tuple[int, int, int, int], cell_index: int
) -> tuple[int, int, int, int]:
    """Pixel bounds of one cell inside the strip's bounding box."""
    left, top, right, bottom = bbox
    width = right - left
    if width < CELL_COUNT:
        raise ValueError(
            f"strip is {width} px wide, too narrow for {CELL_COUNT} cells"
        )
    cell_width = width / CELL_COUNT
    cell_left = left + cell_width * cell_index
    return (
        int(round(cell_left)),
        int(top),
        int(round(cell_left + cell_width)),
        int(bottom),
    )


def sample_cell(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    cell_index: int,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
) -> float:
    left, top, right, bottom = cell_bounds(bbox, cell_index)
    inset_x = int(round((right - left) * (1.0 - sample_fraction) / 2.0))
    inset_y = int(round((bottom - top) * (1.0 - sample_fraction) / 2.0))
    patch = gray[
        top + inset_y : max(bottom - inset_y, top + inset_y + 1),
        left + inset_x : max(right - inset_x, left + inset_x + 1),
    ]
    if patch.size == 0:
        raise ValueError(f"cell {cell_index} sampled an empty patch")
    return float(np.median(patch))


def decode(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    max_time_code: int = MAX_TIME_CODE,
    min_reference_contrast: float = DEFAULT_MIN_REFERENCE_CONTRAST,
    min_bit_margin: float = DEFAULT_MIN_BIT_MARGIN,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
) -> dict[str, Any]:
    """Read one frame's time code, or refuse.

    Returns a dict with ``time_code`` set only when every check passes;
    ``reason`` says which one failed otherwise.
    """
    left, top, right, bottom = bbox
    if (
        left < 0
        or top < 0
        or bottom > gray.shape[0]
        or right > gray.shape[1]
        or right <= left
        or bottom <= top
    ):
        return {"time_code": None, "reason": "bbox_outside_frame"}

    values = [
        sample_cell(gray, bbox, index, sample_fraction)
        for index in range(CELL_COUNT)
    ]
    white = values[WHITE_CELL]
    black = values[BLACK_CELL]
    contrast = white - black
    result: dict[str, Any] = {
        "white": round(white, 3),
        "black": round(black, 3),
        "contrast": round(contrast, 3),
        "cell_values": [round(value, 3) for value in values],
    }
    if contrast < min_reference_contrast:
        result.update({"time_code": None, "reason": "low_reference_contrast"})
        return result

    midpoint = (white + black) / 2.0
    threshold_margin = contrast * min_bit_margin
    bits = []
    worst_margin = float("inf")
    for index in range(FIRST_DATA_CELL, CELL_COUNT):
        value = values[index]
        margin = abs(value - midpoint)
        worst_margin = min(worst_margin, margin)
        if margin < threshold_margin:
            result.update(
                {
                    "time_code": None,
                    "reason": "ambiguous_bit",
                    "ambiguous_cell": index,
                    "worst_bit_margin": round(worst_margin, 3),
                }
            )
            return result
        bits.append(1 if value > midpoint else 0)
    result["worst_bit_margin"] = round(worst_margin, 3)

    data_bits = bits[:DATA_BITS]
    parity = bits[DATA_BITS]
    if sum(data_bits) % 2 != parity:
        result.update({"time_code": None, "reason": "parity_mismatch"})
        return result

    time_code = sum(bit << index for index, bit in enumerate(data_bits))
    if time_code > max_time_code:
        result.update(
            {
                "time_code": None,
                "reason": "out_of_range",
                "decoded_value": time_code,
            }
        )
        return result

    result.update({"time_code": time_code, "reason": ""})
    return result


def render_reference_strip(
    time_code: int,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    white_value: float = 235.0,
    black_value: float = 20.0,
    background: float = 90.0,
) -> np.ndarray:
    """A synthetic frame carrying the marker, for tests and for calibration.

    Having the encoder and decoder meet on a generated image is what lets the
    decode path be verified without Isaac in the loop.
    """
    gray = np.full(shape, background, dtype=np.float32)
    bits = encode_bits(time_code)
    cells = [1, 0] + bits
    for index, on in enumerate(cells):
        left, top, right, bottom = cell_bounds(bbox, index)
        gray[top:bottom, left:right] = white_value if on else black_value
    return gray
