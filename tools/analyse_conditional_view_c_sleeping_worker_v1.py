"""Analyse a persistent View-C worker that runs only on triggered frames."""

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
CONDITIONAL = BASE / "conditional_view_c_v1"
OUTPUT = CONDITIONAL / "sleeping_worker" / "sleeping_worker_summary.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] if lower == upper else ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("Refusing to overwrite sleeping-worker summary.")
    protocol = json.loads((CONDITIONAL / "conditional_view_c_protocol.json").read_text(encoding="utf-8"))
    paths = protocol["registered_paths"]
    decisions = rows(CONDITIONAL / "replay" / "conditional_frame_decisions.csv")
    decision_ms = json.loads((CONDITIONAL / "replay" / "conditional_replay_state.json").read_text(encoding="utf-8"))["mean_decision_and_selector_ms"]
    run_reports = []
    for repeat in (1, 2, 3):
        rep = f"rep_{repeat:02d}"
        triggers = {
            int(row["sequence_index"])
            for row in decisions
            if int(row["repeat_index"]) == repeat and row["trigger_view_c"] == "1"
        }
        replay_a = deployment.load_replay(paths[f"{rep}_view_a_replay"])
        replay_c = deployment.load_replay(CONDITIONAL / "sleeping_worker" / rep / "replay")
        extrinsics = deployment.read_json(paths[f"{rep}_extrinsics"])
        transform = extrinsics["relative_transforms"]["a_from_c"]
        _, _, gt, _ = formal_io.load_ground_truth(Path(paths[f"{rep}_view_a_unit"]), "a")
        phase_errors = {"active": [], "inactive": []}
        complete = {"active": 0, "inactive": 0}
        expected = {"active": 0, "inactive": 0}
        selected_c = {"active": 0, "inactive": 0}
        for sequence in range(240):
            phase = "active" if 60 <= sequence <= 89 or 150 <= sequence <= 179 else "inactive"
            expected[phase] += 1
            arm_a = best_view.build_view_arm(deployment.view_input(replay_a, sequence))
            arm_c = None
            if sequence in triggers:
                arm_c = best_view.build_view_arm(deployment.view_input(replay_c, sequence), transform)
            selected = conditional.select_conditionally(arm_a, arm_c)
            selected_c[phase] += int(selected["selected_view"] == "c")
            if not all(selected["joints"][joint]["point_a"] is not None for joint in best_view.ARM_JOINTS):
                continue
            complete[phase] += 1
            frame_error = []
            for joint in best_view.ARM_JOINTS:
                point = selected["joints"][joint]["point_a"]
                truth = gt[(sequence, joint)]
                target = tuple(float(truth[field]) for field in ("gt_x_m", "gt_y_m", "gt_z_m"))
                frame_error.append(math.dist(point, target) * 1000.0)
            phase_errors[phase].append(statistics.fmean(frame_error))

        cache_state = json.loads((CONDITIONAL / "sleeping_worker" / rep / "landmark_cache" / "landmark_cache_state.json").read_text(encoding="utf-8"))
        landmark_c = rows(CONDITIONAL / "sleeping_worker" / rep / "landmark_cache" / "landmark_frames.csv")
        replay_c_frames = rows(CONDITIONAL / "sleeping_worker" / rep / "replay" / "dynamic_kinematic_replay_frames.csv")
        landmark_a = rows(BASE / "offline" / rep / "view_a" / "landmark_cache" / "landmark_frames.csv")
        replay_a_frames = rows(BASE / "offline" / rep / "view_a" / "replay_method_v2" / "dynamic_kinematic_replay_frames.csv")
        costs = []
        for sequence in range(240):
            cost = float(landmark_a[sequence]["total_compute_ms"])
            cost += float(replay_a_frames[sequence]["depth_sampling_ms"])
            cost += float(replay_a_frames[sequence]["inference_and_kinematics_ms"])
            if sequence in triggers:
                cost += float(landmark_c[sequence]["total_compute_ms"])
                cost += float(replay_c_frames[sequence]["depth_sampling_ms"])
                cost += float(replay_c_frames[sequence]["inference_and_kinematics_ms"])
            cost += float(decision_ms)
            costs.append(cost)
        run_reports.append({
            "repeat_index": repeat,
            "triggered_frame_count": len(triggers),
            "triggered_detection_rate": cache_state["triggered_detection_rate"],
            "view_c_triggered_compute_mean_ms": cache_state["triggered_total_compute_ms"]["mean"],
            "view_c_triggered_compute_p95_ms": cache_state["triggered_total_compute_ms"]["p95"],
            "view_c_triggered_compute_max_ms": cache_state["triggered_total_compute_ms"]["max"],
            "active": {
                "coverage": complete["active"] / expected["active"],
                "mean_frame_two_joint_error_mm": statistics.fmean(phase_errors["active"]),
                "p95_frame_two_joint_error_mm": percentile(phase_errors["active"], 0.95),
                "selected_c_frame_count": selected_c["active"],
            },
            "inactive": {
                "coverage": complete["inactive"] / expected["inactive"],
                "mean_frame_two_joint_error_mm": statistics.fmean(phase_errors["inactive"]),
                "p95_frame_two_joint_error_mm": percentile(phase_errors["inactive"], 0.95),
                "selected_c_frame_count": selected_c["inactive"],
            },
            "component_conditioned_sequential_mean_ms": statistics.fmean(costs),
            "component_conditioned_sequential_p95_ms": percentile(costs, 0.95),
            "component_conditioned_sequential_max_ms": max(costs),
            "component_conditioned_sequential_fps": 1000.0 / statistics.fmean(costs),
        })
    values = lambda key: [run[key] for run in run_reports]
    report = {
        "schema_version": 1,
        "status": "complete",
        "classification": "engineering_sleeping_view_c_worker_v1",
        "runs": run_reports,
        "run_level": {
            "triggered_detection_rate_mean": statistics.fmean(values("triggered_detection_rate")),
            "active_coverage_mean": statistics.fmean(run["active"]["coverage"] for run in run_reports),
            "active_error_mean_mm": statistics.fmean(run["active"]["mean_frame_two_joint_error_mm"] for run in run_reports),
            "active_error_sample_sd_mm": statistics.stdev(run["active"]["mean_frame_two_joint_error_mm"] for run in run_reports),
            "component_conditioned_sequential_fps_mean": statistics.fmean(values("component_conditioned_sequential_fps")),
            "component_conditioned_sequential_fps_sample_sd": statistics.stdev(values("component_conditioned_sequential_fps")),
            "maximum_observed_component_conditioned_latency_ms": max(values("component_conditioned_sequential_max_ms")),
        },
        "restrictions": [
            "Saved-frame offline engineering replay, not source or camera-to-output FPS.",
            "View A timing is reused from the frozen continuous run; View C landmarks are newly measured with a persistent worker invoked only on triggers.",
            "Component-conditioned latency excludes camera acquisition, queues and disk scheduling; it includes triggered frame RGB load, inference, depth replay and selector components.",
            "The worker was initialized once but not explicitly inference-warmed before its first trigger.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
