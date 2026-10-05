"""Aggregate the frozen three-run conditional View-C streaming experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "experiments" / "kinematic_constraints_ablation" / "conditional_view_c_streaming_repeatability_v1"
PROTOCOL = BASE / "repeatability_protocol_v1.json"
OUTPUT_JSON = BASE / "repeatability_summary_v1.json"
OUTPUT_CSV = BASE / "repeatability_run_summary_v1.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_stats(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    return {"n": len(values), "mean": mean, "sample_sd": sd, "cv": sd / mean if mean else 0.0, "min": min(values), "max": max(values)}


def evaluate_run(base: Path, repeat_index: int, protocol_sha256: str, registered_gate_names: set[str]) -> dict:
    repeat = base / "rep_{:02d}".format(repeat_index)
    consumer_dir = repeat / "consumer"
    producer_dir = repeat / "producer"
    consumer = json.loads((consumer_dir / "consumer_state.json").read_text(encoding="utf-8"))
    producer = json.loads((producer_dir / "producer_state.json").read_text(encoding="utf-8"))
    audit = json.loads((repeat / "post_stream_gt_audit_v1.json").read_text(encoding="utf-8"))
    joint_path = consumer_dir / "consumer_joints.csv"
    frame_path = consumer_dir / "consumer_frames.csv"
    producer_frame_path = producer_dir / "producer_frames.csv"
    joint_rows = rows(joint_path)
    frame_rows = rows(frame_path)
    active_joints = [row for row in joint_rows if row["phase"] == "active"]
    bursts = [row for row in frame_rows if int(row["preroll_frame_count"]) > 0]
    burst_host_max = max((float(row["host_render_to_output_ready_ms"]) for row in bursts), default=float("inf"))
    active = audit.get("active", {})
    file_hashes_valid = (
        consumer.get("frame_rows_sha256") == sha(frame_path)
        and consumer.get("joint_rows_sha256") == sha(joint_path)
        and producer.get("producer_frames_sha256") == sha(producer_frame_path)
    )
    gates = {
        "producer_and_consumer_complete": producer.get("status") == "complete" and consumer.get("status") == "complete",
        "repeat_identity_matches": producer.get("repeat_index") == repeat_index and str(consumer.get("session_id", "")).startswith("isaac_conditional_stream_rep_{:02d}_".format(repeat_index)),
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
        "gt_files_opened_zero_during_stream": consumer.get("gt_files_opened") == 0 and producer.get("uses_skeleton_gt") is False,
        "protocol_hash_matches": consumer.get("protocol_sha256") == protocol_sha256 and producer.get("protocol_sha256") == protocol_sha256,
        "stream_output_hashes_valid": file_hashes_valid,
        "post_stream_audit_binds_consumer_joints": audit.get("status") == "complete" and audit.get("classification") == "registered_post_stream_gt_comparator_for_repeatability_v1" and audit.get("repeat_index") == repeat_index and audit.get("gt_opened_only_after_stream_completion") is True and audit.get("protocol_sha256") == protocol_sha256 and audit.get("consumer_joints_sha256") == sha(joint_path),
        "active_gt_coverage_unity": active.get("coverage") == 1.0,
        "active_gt_mean_le_30_mm": active.get("mean_frame_two_joint_error_mm", 999.0) <= 30.0,
        "active_gt_p95_le_40_mm": active.get("p95_frame_two_joint_error_mm", 999.0) <= 40.0,
        "active_gt_max_frame_le_75_mm": active.get("max_frame_two_joint_error_mm", 999.0) <= 75.0,
        "active_gt_max_joint_le_100_mm": active.get("max_joint_error_mm", 999.0) <= 100.0,
    }
    if set(gates) != registered_gate_names:
        raise RuntimeError("Observed per-run gate set differs from frozen protocol.")
    metrics = {
        "repeat_index": repeat_index,
        "isaac_process_id": producer.get("isaac_process_id"),
        "overall_trigger_fraction": consumer["trigger_fraction_overall"],
        "effective_consumer_fps": consumer["effective_consumer_fps"],
        "consumer_wall_p95_ms": consumer["timing_ms"]["consumer_frame_wall_ms"]["p95"],
        "host_p95_ms": consumer["timing_ms"]["host_render_to_output_ready_ms"]["p95"],
        "burst_host_max_ms": burst_host_max,
        "active_coverage": active["coverage"],
        "active_mean_error_mm": active["mean_frame_two_joint_error_mm"],
        "active_p95_error_mm": active["p95_frame_two_joint_error_mm"],
        "active_max_frame_error_mm": active["max_frame_two_joint_error_mm"],
        "active_max_joint_error_mm": active["max_joint_error_mm"],
    }
    return {"repeat_index": repeat_index, "all_per_run_gates_passed": all(gates.values()), "gates": gates, "metrics": metrics}


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_CSV.exists():
        raise RuntimeError("Refusing to overwrite repeatability aggregate outputs.")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_sha256 = sha(PROTOCOL)
    if protocol.get("status") != "frozen_before_streaming_pilot" or protocol.get("freeze_role") != "frozen_before_three_run_repeatability_capture":
        raise RuntimeError("Repeatability protocol is not frozen.")
    if protocol["registered_sha256"].get("repeatability_aggregator") != sha(Path(__file__).resolve()):
        raise RuntimeError("Repeatability aggregator differs from frozen protocol.")

    per_run_gate_names = set(protocol["registered_per_run_gates"])
    run_reports = [evaluate_run(BASE, index, protocol_sha256, per_run_gate_names) for index in (1, 2, 3)]
    metrics = [report["metrics"] for report in run_reports]
    active_means = [row["active_mean_error_mm"] for row in metrics]
    active_p95s = [row["active_p95_error_mm"] for row in metrics]
    trigger_fractions = [row["overall_trigger_fraction"] for row in metrics]
    process_ids = [row["isaac_process_id"] for row in metrics]
    replication_gates = {
        "all_three_per_run_gate_sets_pass": all(report["all_per_run_gates_passed"] for report in run_reports),
        "three_distinct_isaac_process_ids": None not in process_ids and len(set(process_ids)) == 3,
        "active_mean_error_cv_le_0p10": metric_stats(active_means)["cv"] <= 0.10,
        "active_p95_error_cv_le_0p10": metric_stats(active_p95s)["cv"] <= 0.10,
        "overall_trigger_fraction_range_le_0p05": max(trigger_fractions) - min(trigger_fractions) <= 0.05,
    }
    if set(replication_gates) != set(protocol["registered_replication_gates"]):
        raise RuntimeError("Observed replication gate set differs from frozen protocol.")

    summary_stats = {
        key: metric_stats([float(row[key]) for row in metrics])
        for key in (
            "overall_trigger_fraction", "effective_consumer_fps", "consumer_wall_p95_ms",
            "host_p95_ms", "burst_host_max_ms", "active_coverage", "active_mean_error_mm",
            "active_p95_error_mm", "active_max_frame_error_mm", "active_max_joint_error_mm",
        )
    }
    all_passed = all(replication_gates.values())
    report = {
        "schema_version": 1,
        "status": "complete",
        "classification": "registered_three_run_same_scene_engineering_repeatability",
        "decision": "PASS_three_run_engineering_repeatability_supported" if all_passed else "FAIL_registered_mixed_or_negative_result",
        "all_registered_gates_passed": all_passed,
        "replication_gates": replication_gates,
        "runs": run_reports,
        "run_level_summary_mean_sample_sd": summary_stats,
        "protocol_sha256": protocol_sha256,
        "restrictions": [
            "Three independent Isaac processes in one fixed single-person simulated scene.",
            "Supports only same-scene engineering repeatability when all registered gates pass.",
            "Does not establish physical dual-ZED synchronization, 60 FPS camera-to-output, cross-subject generalisation, or multi-person performance.",
            "GT was opened only by registered post-stream comparators after each GT-free stream completed.",
        ],
    }
    with OUTPUT_CSV.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    report["run_summary_csv_sha256"] = sha(OUTPUT_CSV)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
