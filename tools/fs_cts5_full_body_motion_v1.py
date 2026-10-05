"""Pure timing contract for the 240-step FS-CTS5 full-body pilot motion."""

from __future__ import annotations


FPS = 60
END_FRAME = 240
FRAME_COUNT = END_FRAME + 1
MOTION_TAG = "fs_cts5_full_body_reach_step_squat_v1"


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _ramp(frame: int, start: int, end: int) -> float:
    if end <= start:
        raise ValueError("Ramp end must be after start.")
    return smoothstep((int(frame) - start) / float(end - start))


def controls_for_frame(frame: int) -> dict[str, float]:
    """Return reach, step-like and squat-like controls in the closed [0, 1] range.

    The cycle contains a neutral baseline, bilateral arm reach, a left-leg
    step-like lift, a bilateral squat-like bend, and a smooth return.  Frame
    240 is identical to frame 0 so fresh-load replay can loop without a jump.
    """
    frame = int(frame)
    if frame < 0 or frame > END_FRAME:
        raise ValueError(f"Frame outside 0..{END_FRAME}: {frame}")

    if frame < 20:
        reach = 0.0
    elif frame <= 60:
        reach = _ramp(frame, 20, 60)
    elif frame < 210:
        reach = 1.0
    else:
        reach = 1.0 - _ramp(frame, 210, END_FRAME)

    if frame < 90:
        step = 0.0
    elif frame <= 120:
        step = _ramp(frame, 90, 120)
    elif frame < 140:
        step = 1.0
    elif frame <= 170:
        step = 1.0 - _ramp(frame, 140, 170)
    else:
        step = 0.0

    if frame < 140:
        squat = 0.0
    elif frame <= 180:
        squat = _ramp(frame, 140, 180)
    elif frame < 210:
        squat = 1.0
    else:
        squat = 1.0 - _ramp(frame, 210, END_FRAME)

    return {"reach": reach, "step": step, "squat": squat}


def phase_for_frame(frame: int) -> str:
    controls = controls_for_frame(frame)
    if frame < 20:
        return "neutral_baseline"
    if controls["step"] > controls["squat"] and controls["step"] > 0.0:
        return "left_step_like"
    if controls["squat"] > 0.0:
        return "bilateral_squat_like"
    if controls["reach"] > 0.0:
        return "bilateral_reach"
    return "neutral_return"


def schedule_summary() -> dict[str, object]:
    samples = [controls_for_frame(frame) for frame in range(FRAME_COUNT)]
    return {
        "fps": FPS,
        "end_frame": END_FRAME,
        "frame_count": FRAME_COUNT,
        "motion_tag": MOTION_TAG,
        "loop_closes": samples[0] == samples[-1],
        "max_reach": max(sample["reach"] for sample in samples),
        "max_step": max(sample["step"] for sample in samples),
        "max_squat": max(sample["squat"] for sample in samples),
    }
