"""Retrospective timing decomposition from the completed View A/C formal runs."""

from __future__ import annotations

import csv,json,statistics
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"output"/"experiments"/"kinematic_constraints_ablation"/"multiview_ac_measured_first_formal_v1"
OUTPUT=BASE/"performance_decomposition_v1.json"


def rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def stats(values):
    ordered=sorted(values);return {"count":len(values),"mean":statistics.fmean(values),"median":statistics.median(values),"p95":ordered[int(.95*(len(ordered)-1))],"max":max(values)}


def main():
    if OUTPUT.exists():raise RuntimeError("Refusing overwrite.")
    runs=[]
    for repeat in (1,2,3):
        rep=f"rep_{repeat:02d}"; per_view={}
        for view in ("a","c"):
            landmark=rows(BASE/"offline"/rep/f"view_{view}"/"landmark_cache"/"landmark_frames.csv")
            replay=rows(BASE/"offline"/rep/f"view_{view}"/"replay_method_v2"/"dynamic_kinematic_replay_frames.csv")
            per_view[view]=[float(landmark[i]["total_compute_ms"])+float(replay[i]["depth_sampling_ms"])+float(replay[i]["inference_and_kinematics_ms"]) for i in range(240)]
        sequential=[per_view["a"][i]+per_view["c"][i]+.08779902777777777 for i in range(240)]
        parallel_bound=[max(per_view["a"][i],per_view["c"][i])+.08779902777777777 for i in range(240)]
        pair=rows(BASE/"raw"/rep/"view_ac_pair_manifest.csv"); times=[int(r["pair_capture_wall_time_ns"])/1e9 for r in pair]
        intervals=[(times[i]-times[i-1])*1000 for i in range(1,len(times))]
        metadata=json.loads((BASE/"raw"/rep/"view_ac_pair_metadata.json").read_text(encoding="utf-8"))
        runs.append({"repeat_index":repeat,"view_a_consumer_compute_ms":stats(per_view["a"]),"view_c_consumer_compute_ms":stats(per_view["c"]),"paired_sequential_consumer_ms":stats(sequential),"paired_parallel_lower_bound_ms":stats(parallel_bound),"formal_export_inter_pair_ms":stats(intervals),"formal_export_effective_fps":240/float(metadata["capture_duration_s"])})
    payload={"schema_version":1,"status":"complete","classification":"retrospective_engineering_timing_decomposition","runs":runs,"run_level":{"formal_export_effective_fps":{"values":[r["formal_export_effective_fps"] for r in runs],"mean":statistics.fmean(r["formal_export_effective_fps"] for r in runs),"sample_sd":statistics.stdev(r["formal_export_effective_fps"] for r in runs)},"paired_sequential_consumer_mean_ms":{"values":[r["paired_sequential_consumer_ms"]["mean"] for r in runs],"mean":statistics.fmean(r["paired_sequential_consumer_ms"]["mean"] for r in runs),"sample_sd":statistics.stdev(r["paired_sequential_consumer_ms"]["mean"] for r in runs)},"paired_parallel_lower_bound_mean_ms":{"values":[r["paired_parallel_lower_bound_ms"]["mean"] for r in runs],"mean":statistics.fmean(r["paired_parallel_lower_bound_ms"]["mean"] for r in runs),"sample_sd":statistics.stdev(r["paired_parallel_lower_bound_ms"]["mean"] for r in runs)}},"restrictions":["Formal exporter FPS includes render, two app updates, annotation reads, RGB/depth/GT hashing, and disk writes; it is not source acquisition FPS.","Consumer compute is offline saved-frame replay, not camera-to-output latency.","Parallel value is max(A,C), a computed lower bound, not measured concurrent execution."]}
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");print(json.dumps(payload,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
