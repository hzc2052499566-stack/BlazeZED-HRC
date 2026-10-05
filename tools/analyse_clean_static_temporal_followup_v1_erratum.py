"""Execution erratum: correct one validation JSON path, then run frozen analysis.

The frozen analysis looked for ``temporal_state_enabled`` under
``method_contract``.  ``replay_depth_sampling.py`` records it under
``processing_variant``.  No metric, window, threshold or output calculation is
changed here.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyse_clean_static_temporal_followup_v1 as frozen


def validate_condition(condition_root: Path, condition: str, protocol: dict) -> None:
    cache = frozen.read_json(condition_root / "landmark_cache" / "landmark_cache_state.json")
    replay = frozen.read_json(condition_root / "replay" / "offline_tracker_state.json")
    expected = protocol["estimation_contract"]["conditions"][condition]
    checks = {
        "cache complete": cache.get("status") == "complete",
        "replay complete": replay.get("status") == "complete",
        "static_image_mode": cache.get("static_image_mode") == expected["static_image_mode"],
        "smooth_landmarks": cache.get("smooth_landmarks") == expected["smooth_landmarks"],
        "model_complexity": cache.get("model_complexity") == 0,
        "roi_scale": cache.get("roi_scale") == 0.65,
        "depth_sampling": replay.get("depth_sampling_method") == "median_7x7",
        "warmup": replay.get("warmup_frames") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{condition_root}: invalid config {failed}")
    performance = frozen.read_json(condition_root / "replay" / "performance_summary.json")
    if performance.get("processing_variant", {}).get("temporal_state_enabled") is not False:
        raise RuntimeError(f"{condition_root}: depth replay is not stateless")


if __name__ == "__main__":
    frozen.validate_condition = validate_condition
    raise SystemExit(frozen.main())
