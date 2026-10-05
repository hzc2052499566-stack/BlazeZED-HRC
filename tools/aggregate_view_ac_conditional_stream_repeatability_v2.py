"""Run the registered v1 aggregator against the clean v2 recovery root."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_SOURCE = ROOT / "tools" / "aggregate_view_ac_conditional_stream_repeatability_v1.py"
BASE = ROOT / "output" / "experiments" / "kinematic_constraints_ablation" / "conditional_view_c_streaming_repeatability_v2"
PROTOCOL = BASE / "repeatability_protocol_v2.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["registered_sha256"].get("repeatability_aggregator_launcher_v2") != sha(Path(__file__).resolve()):
        raise RuntimeError("Repeatability aggregator launcher v2 differs from frozen protocol.")
    spec = importlib.util.spec_from_file_location("view_ac_repeatability_aggregator_v1_embedded", V1_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen repeatability v1 aggregator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE = BASE
    module.PROTOCOL = PROTOCOL
    module.OUTPUT_JSON = BASE / "repeatability_summary_v2.json"
    module.OUTPUT_CSV = BASE / "repeatability_run_summary_v2.csv"
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
