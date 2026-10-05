"""Aggregate the registered four-frame-pre-roll conditional streaming Pilot v2."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "experiments" / "kinematic_constraints_ablation" / "conditional_view_c_streaming_pilot_v2"
PROTOCOL = BASE / "streaming_pilot_v2_protocol.json"
OUTPUT = BASE / "streaming_pilot_v2_summary.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("Refusing to overwrite streaming Pilot v2 summary.")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_streaming_pilot" or protocol["registered_sha256"].get("aggregator_v2") != sha(Path(__file__).resolve()):
        raise RuntimeError("Aggregator v2 differs from frozen protocol.")

    consumer_dir = BASE / "rep_01" / "consumer"
    producer_dir = BASE / "rep_01" / "producer"
    consumer = json.loads((consumer_dir / "consumer_state.json").read_text(encoding="utf-8"))
    producer = json.loads((producer_dir / "producer_state.json").read_text(encoding="utf-8"))
    joint_rows = rows(consumer_dir / "consumer_joints.csv")
    frame_rows = rows(consumer_dir / "consumer_frames.csv")
    active_joints = [row for row in joint_rows if row["phase"] == "active"]
    bursts = [row for row in frame_rows if int(row["preroll_frame_count"]) > 0]
    burst_host_max = max((float(row["host_render_to_output_ready_ms"]) for row in bursts), default=float("inf"))

    gates = {
        "producer_and_consumer_complete": producer.get("status") == "complete" and consumer.get("status") == "complete",
        "exactly_240_scored_frames": producer.get("frame_count") == 240 and consumer.get("frame_count") == 240,
        "exactly_10_warmup_frames": consumer.get("warmup_frame_count") == 10,
        "view_a_detection_rate_unity": consumer.get("detection_rate_a") == 1.0,
        "active_trigger_fraction_unity": consumer.get("trigger_fraction_active") == 1.0,
        "active_selected_c_fraction_ge_0p95": consumer.get("selected_c_fraction_active", 0.0) >= 0.95,
        "overall_trigger_fraction_le_0p35": consumer.get("trigger_fraction_overall", 1.0) <= 0.35,
        "active_joint_rows_exactly_120": len(active_joints) == 120,
        "active_all_k2_reliable_measured": len(active_joints) == 120 and all(row["valid"] == "1" and row["selected_source"] == "k2_reliable_measured" for row in active_joints),
        "two_activation_bursts_with_four_frame_preroll": len(bursts) == 2 and [int(row["sequence_index"]) for row in bursts] == [60, 150] and all(int(row["preroll_frame_count"]) == 4 for row in bursts),
        "activation_burst_host_latency_max_le_180_ms": burst_host_max <= 180.0,
        "effective_consumer_fps_ge_27": consumer.get("effective_consumer_fps", 0.0) >= 27.0,
        "selector_p95_le_2_ms": consumer.get("timing_ms", {}).get("selector_ms", {}).get("p95", 999.0) <= 2.0,
        "consumer_frame_wall_p95_le_120_ms": consumer.get("timing_ms", {}).get("consumer_frame_wall_ms", {}).get("p95", 999.0) <= 120.0,
        "host_render_to_output_p95_le_150_ms": consumer.get("timing_ms", {}).get("host_render_to_output_ready_ms", {}).get("p95", 999.0) <= 150.0,
        "gt_files_opened_zero": consumer.get("gt_files_opened") == 0 and producer.get("uses_skeleton_gt") is False,
    }
    if set(gates) != set(protocol["registered_gates"]):
        raise RuntimeError("Observed gate set differs from frozen Pilot v2 protocol.")
    report = {
        "schema_version": 1,
        "status": "complete",
        "classification": "engineering_view_ac_conditional_streaming_pilot_v2_aggregate",
        "all_registered_gates_passed": all(gates.values()),
        "gates": gates,
        "activation_burst_host_latency_max_ms": burst_host_max,
        "active_selected_source_counts": {
            source: sum(row["selected_source"] == source for row in active_joints)
            for source in sorted({row["selected_source"] for row in active_joints})
        },
        "consumer": consumer,
        "producer": producer,
        "protocol_sha256": sha(PROTOCOL),
        "restrictions": [
            "One engineering run; no repeatability claim.",
            "Isaac paused-timeline same-render-step source, not physical ZED synchronization.",
            "Consumer FPS excludes Isaac render acquisition; producer capture and full host latency are reported separately.",
            "No GT is transmitted or opened; frozen offline evidence remains the accuracy source.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_registered_gates_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
