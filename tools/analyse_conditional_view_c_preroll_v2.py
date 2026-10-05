"""Evaluate two-frame buffered pre-roll for the sleeping View-C worker."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import analyse_view_ac_best_view_fusion_pilot_v1 as formal_io
import multiview_best_view_measured_first_v1 as best_view
import multiview_conditional_view_c_v1 as conditional
import replay_multiview_best_view_deployment_v1 as deployment


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "experiments" / "kinematic_constraints_ablation" / "multiview_ac_measured_first_formal_v1"
COND = BASE / "conditional_view_c_v1"
WORKER = COND / "sleeping_worker_preroll_v2"
OUTPUT = WORKER / "preroll_v2_summary.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] if lower == upper else ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("Refusing to overwrite pre-roll analysis.")
    protocol = json.loads((COND / "conditional_view_c_protocol.json").read_text(encoding="utf-8"))
    paths = protocol["registered_paths"]
    original_rows = rows(COND / "replay" / "conditional_frame_decisions.csv")
    schedule_rows = rows(WORKER / "worker_schedule.csv")
    decision_ms = json.loads((COND / "replay" / "conditional_replay_state.json").read_text(encoding="utf-8"))["mean_decision_and_selector_ms"]
    reports = []
    for repeat in (1, 2, 3):
        rep = f"rep_{repeat:02d}"
        triggers = {int(row["sequence_index"]) for row in original_rows if int(row["repeat_index"]) == repeat and row["trigger_view_c"] == "1"}
        invoked = {int(row["sequence_index"]) for row in schedule_rows if int(row["repeat_index"]) == repeat and row["trigger_view_c"] == "1"}
        priming = invoked - triggers
        burst_starts = sorted(frame for frame in triggers if frame == 0 or frame - 1 not in triggers)
        replay_a = deployment.load_replay(paths[f"{rep}_view_a_replay"])
        replay_c = deployment.load_replay(WORKER / rep / "replay")
        transform = deployment.read_json(paths[f"{rep}_extrinsics"])["relative_transforms"]["a_from_c"]
        _, _, gt, _ = formal_io.load_ground_truth(Path(paths[f"{rep}_view_a_unit"]), "a")
        metrics = {phase: {"errors": [], "complete": 0, "selected_c": 0} for phase in ("active", "inactive")}
        for sequence in range(240):
            phase = "active" if 60 <= sequence <= 89 or 150 <= sequence <= 179 else "inactive"
            arm_a = best_view.build_view_arm(deployment.view_input(replay_a, sequence))
            arm_c = best_view.build_view_arm(deployment.view_input(replay_c, sequence), transform) if sequence in triggers else None
            selected = conditional.select_conditionally(arm_a, arm_c)
            metrics[phase]["selected_c"] += int(selected["selected_view"] == "c")
            points = [selected["joints"][joint]["point_a"] for joint in best_view.ARM_JOINTS]
            if any(point is None for point in points):
                continue
            metrics[phase]["complete"] += 1
            errors = []
            for joint, point in zip(best_view.ARM_JOINTS, points):
                truth = gt[(sequence, joint)]
                target = tuple(float(truth[field]) for field in ("gt_x_m", "gt_y_m", "gt_z_m"))
                errors.append(math.dist(point, target) * 1000.0)
            metrics[phase]["errors"].append(statistics.fmean(errors))

        cache_state = json.loads((WORKER / rep / "landmark_cache" / "landmark_cache_state.json").read_text(encoding="utf-8"))
        landmark_c = rows(WORKER / rep / "landmark_cache" / "landmark_frames.csv")
        replay_c_frames = rows(WORKER / rep / "replay" / "dynamic_kinematic_replay_frames.csv")
        landmark_a = rows(BASE / "offline" / rep / "view_a" / "landmark_cache" / "landmark_frames.csv")
        replay_a_frames = rows(BASE / "offline" / rep / "view_a" / "replay_method_v2" / "dynamic_kinematic_replay_frames.csv")
        base_cost = []
        c_cost = []
        for sequence in range(240):
            base_cost.append(float(landmark_a[sequence]["total_compute_ms"]) + float(replay_a_frames[sequence]["depth_sampling_ms"]) + float(replay_a_frames[sequence]["inference_and_kinematics_ms"]) + decision_ms)
            c_cost.append(float(landmark_c[sequence]["total_compute_ms"]) + float(replay_c_frames[sequence]["depth_sampling_ms"]) + float(replay_c_frames[sequence]["inference_and_kinematics_ms"]) if sequence in invoked else 0.0)
        service_cost = [base_cost[i] + c_cost[i] for i in range(240)]
        for start in burst_starts:
            if start == 0:
                continue
            for prior in range(max(0, start - 2), start):
                if prior in priming:
                    service_cost[start] += c_cost[prior]
                    service_cost[prior] -= c_cost[prior]
        reports.append({
            "repeat_index": repeat,
            "output_trigger_count": len(triggers),
            "priming_frame_count": len(priming),
            "worker_invocation_fraction": len(invoked) / 240.0,
            "triggered_detection_rate": cache_state["triggered_detection_rate"],
            "active": {
                "coverage": metrics["active"]["complete"] / 60.0,
                "mean_frame_two_joint_error_mm": statistics.fmean(metrics["active"]["errors"]),
                "p95_frame_two_joint_error_mm": p95(metrics["active"]["errors"]),
                "selected_c_frame_count": metrics["active"]["selected_c"],
            },
            "inactive": {
                "coverage": metrics["inactive"]["complete"] / 180.0,
                "mean_frame_two_joint_error_mm": statistics.fmean(metrics["inactive"]["errors"]),
                "p95_frame_two_joint_error_mm": p95(metrics["inactive"]["errors"]),
                "selected_c_frame_count": metrics["inactive"]["selected_c"],
            },
            "component_conditioned_sequential_mean_ms": statistics.fmean(service_cost),
            "component_conditioned_sequential_p95_ms": p95(service_cost),
            "component_conditioned_sequential_max_ms": max(service_cost),
            "component_conditioned_sequential_fps": 1000.0 / statistics.fmean(service_cost),
            "activation_burst_service_ms": {str(start): service_cost[start] for start in burst_starts},
        })
    report = {
        "schema_version": 1,
        "status": "complete",
        "classification": "engineering_conditional_view_c_buffered_preroll_v2",
        "runs": reports,
        "run_level": {
            "worker_invocation_fraction_mean": statistics.fmean(run["worker_invocation_fraction"] for run in reports),
            "triggered_detection_rate_mean": statistics.fmean(run["triggered_detection_rate"] for run in reports),
            "active_coverage_mean": statistics.fmean(run["active"]["coverage"] for run in reports),
            "active_error_mean_mm": statistics.fmean(run["active"]["mean_frame_two_joint_error_mm"] for run in reports),
            "active_error_sample_sd_mm": statistics.stdev(run["active"]["mean_frame_two_joint_error_mm"] for run in reports),
            "component_conditioned_sequential_fps_mean": statistics.fmean(run["component_conditioned_sequential_fps"] for run in reports),
            "component_conditioned_sequential_fps_sample_sd": statistics.stdev(run["component_conditioned_sequential_fps"] for run in reports),
            "maximum_activation_burst_service_ms": max(max(run["activation_burst_service_ms"].values()) for run in reports),
        },
        "restrictions": [
            "Post-observation engineering intervention on saved single-person simulation frames.",
            "Two buffered frames are processed after the A trigger and before the current C frame, so burst-start latency is larger than steady latency.",
            "Component-conditioned FPS is not measured streaming or camera-to-output FPS.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
