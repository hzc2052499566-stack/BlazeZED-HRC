"""Aggregate the registered one-run conditional streaming engineering pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "experiments" / "kinematic_constraints_ablation" / "conditional_view_c_streaming_pilot_v1"
PROTOCOL = BASE / "streaming_pilot_protocol.json"
OUTPUT = BASE / "streaming_pilot_summary.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("Refusing to overwrite streaming pilot summary.")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_streaming_pilot" or protocol["registered_sha256"].get("aggregator") != sha(Path(__file__).resolve()):
        raise RuntimeError("Aggregator differs from frozen protocol.")
    consumer = json.loads((BASE / "rep_01" / "consumer" / "consumer_state.json").read_text(encoding="utf-8"))
    producer = json.loads((BASE / "rep_01" / "producer" / "producer_state.json").read_text(encoding="utf-8"))
    gates = {
        "producer_and_consumer_complete": producer.get("status") == "complete" and consumer.get("status") == "complete",
        "exactly_240_scored_frames": producer.get("frame_count") == 240 and consumer.get("frame_count") == 240,
        "exactly_10_warmup_frames": consumer.get("warmup_frame_count") == 10,
        "view_a_detection_rate_unity": consumer.get("detection_rate_a") == 1.0,
        "active_trigger_fraction_unity": consumer.get("trigger_fraction_active") == 1.0,
        "active_selected_c_fraction_ge_0p95": consumer.get("selected_c_fraction_active", 0.0) >= 0.95,
        "overall_trigger_fraction_le_0p35": consumer.get("trigger_fraction_overall", 1.0) <= 0.35,
        "effective_consumer_fps_ge_27": consumer.get("effective_consumer_fps", 0.0) >= 27.0,
        "selector_p95_le_2_ms": consumer.get("timing_ms", {}).get("selector_ms", {}).get("p95", 999.0) <= 2.0,
        "consumer_frame_wall_p95_le_120_ms": consumer.get("timing_ms", {}).get("consumer_frame_wall_ms", {}).get("p95", 999.0) <= 120.0,
        "host_render_to_output_p95_le_150_ms": consumer.get("timing_ms", {}).get("host_render_to_output_ready_ms", {}).get("p95", 999.0) <= 150.0,
        "gt_files_opened_zero": consumer.get("gt_files_opened") == 0 and producer.get("uses_skeleton_gt") is False,
    }
    if set(gates) != set(protocol["registered_gates"]):
        raise RuntimeError("Observed gate set differs from frozen protocol.")
    report = {
        "schema_version": 1, "status": "complete",
        "classification": "engineering_view_ac_conditional_streaming_pilot_v1_aggregate",
        "all_registered_gates_passed": all(gates.values()), "gates": gates,
        "consumer": consumer, "producer": producer,
        "protocol_sha256": sha(PROTOCOL),
        "restrictions": [
            "One engineering run; no repeatability claim.",
            "Isaac paused-timeline same-render-step source, not physical ZED synchronization.",
            "Consumer FPS excludes Isaac render acquisition; producer capture and full host latency are reported separately.",
            "No GT is transmitted or opened; this pilot measures continuity and timing, not a new accuracy result.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_registered_gates_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
